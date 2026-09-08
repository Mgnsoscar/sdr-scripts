#!/usr/bin/env python3
"""
mock_gps_ca_code_1.023Mcps_tx — a NO-HARDWARE stand-in for gps_ca_code_1.023Mcps.py.

Same calibration surface as the real GPS C/A (1.023 Mcps) transmitter (identical parameter schema,
CAL_SIGNAL_ID, CAL_FREQ_PARAM, CAL_POWER_LAWS and the hidden enbw_mhz derived field), so the
client's Run/tune form drives it EXACTLY like the real PRN — the same power card with its spectral
density / main-lobe / full-signal-power quantities, the --sidelobes slider, and the DEPENDS ON row.
But it never imports UHD / GNU Radio, generates no Gold code and transmits nothing: a fake "radio"
only LOGS the SDR gain (and amplitude) it *would* command.

Use it to check, with no SDR connected, that a requested --power — in whatever quantity you set it
(density dBm/Hz, main-lobe or full power dBm) — maps to the right gain, and that a live --sidelobes
change re-maps a held full/absolute power the way the real signal does (the full-power quantity
tracks the filter's equivalent-noise bandwidth).

The calibration MATH (the sinc² power fractions → enbw_mhz(n)) is the SAME pure-Python code the real
script uses; only the Gold-code generation, the IQ loop, the passband filter DSP and the GNU Radio
flowgraph are dropped (the mock doesn't transmit). --self-test asserts the baked enbw table still
matches enbw_mhz(), exactly as the real script does, so the two can't silently drift.

Getting a calibration (needed for absolute --power)
───────────────────────────────────────────────────
Absolute --power only has meaning with a calibration; uncalibrated the script runs on a relative
--gain. Three ways to supply one:
  • under the agent — a task with SDR_CAL_SIGNAL_ID="GPS C/A (1.023 Mcps)" gets this unit's resolved
    calibration injected (env SDR_CALIBRATION_FILE), exactly like the real PRN;
  • --calibration <artifact.json> — point it at a resolved calibration artifact yourself;
  • --make-sample-calibration <out.json> — build + resolve a representative spectral-density
    calibration so you can run standalone. Needs sdr-agent on PYTHONPATH (the resolver lives there).

CLI
───
    mock_gps_ca_code_1.023Mcps_tx.py --calibration cal.json --power -30 --sidelobes 5 --once
    mock_gps_ca_code_1.023Mcps_tx.py --gain 60 --sidelobes 5 --once     # raw-gain override
    PYTHONPATH=/path/to/sdr-agent mock_gps_ca_code_1.023Mcps_tx.py --make-sample-calibration cal.json
    mock_gps_ca_code_1.023Mcps_tx.py --power -30 --sidelobes 5          # run like a task (no hardware)
    mock_gps_ca_code_1.023Mcps_tx.py --self-test        # exercise the density→gain math, no loop
    mock_gps_ca_code_1.023Mcps_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import logging
import math
import os
import signal
import sys
import threading
import time

# Make paramkit importable both on a unit (scripts flattened one level under BASE_DIR, next to
# paramkit/) and in the dev repo (scripts two levels under the repo root). PYTHONPATH is honoured
# too (e.g. the agent sets it).
_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from paramkit import Script, PowerMap

# ── Calibration surface — kept IN STEP with gps_ca_code_1.023Mcps.py so the client treats this mock
#    as the real PRN. The static schema reader (agent/argspec.py) reads these literals from THIS
#    file, so they must live here verbatim (an import wouldn't be seen). ────────────────────────────
CAL_SIGNAL_ID = "GPS C/A (1.023 Mcps)"   # same signal id as the real script → same injected calibration
CAL_FREQ_PARAM = "freq"                  # the carrier the calibration folds at (live)

# ── RF-chain limits (no baked dBm scale; absolute --power comes only from the calibration) ───────
GAIN_AT_MAX_DB = 89.75              # gain ceiling: never command a gain above this
HW_MAX_GAIN_DB = 89.75             # B200-mini physical TX-gain ceiling
AMPLITUDE = 0.5                    # the amplitude the calibration is measured at

# ── Signal constants (fixed — this IS GPS C/A at 1.023 Mcps) ────────────────────────
CARRIER_HZ = 1575.42e6             # GPS L1 default; --freq retunes to L2 (1227.6) or any bench freq
CODE_RATE_HZ = 1.023e6            # C/A chip rate (~2 MHz null-to-null)
CA_NULL_HZ = 1.023e6              # main-lobe null spacing == the chip rate; sidelobes step by this

# Carrier presets: the same 1.023 Mcps C/A signal on either GPS band. Default is L1 (presets in MHz).
FREQUENCIES = {"GPS L1 (1575.42 MHz)": CARRIER_HZ / 1e6, "GPS L2 (1227.6 MHz)": 1227.6}

MAX_SIDELOBES = 28
DEFAULT_SIDELOBES = 5
CA_NULL_MHZ = 1.023               # sidelobe/main-lobe null spacing (MHz) == the chip rate


# ── Spectral-density calibration math (verbatim from gps_ca_code_1.023Mcps.py) ───────
# A C/A signal is a BPSK(1) sinc² spectrum, so its whole power distribution is fixed by ONE measured
# number: the power spectral DENSITY at the main-lobe PEAK, in dBm/Hz. From it CAL_POWER_LAWS derive
# the MAIN-LOBE integrated power (constant) and the FULL signal power passed by the filter (tracks
# the sidelobe count). The pure-Python sinc² integration below is the SAME code the real script runs.

def _sinc2(x: float) -> float:
    """sinc²(x) with sinc(x) = sin(πx)/(πx); the normalized C/A power-spectral shape."""
    if x == 0.0:
        return 1.0
    s = math.sin(math.pi * x) / (math.pi * x)
    return s * s


def _power_fraction_table(nmax: int, step: float = 1e-3) -> tuple:
    """frac(n) = 2·∫₀^(n+1) sinc²(x) dx for n = 0..nmax — the fraction of the signal's total power
    within ±(n+1) chip-rates (the passband for n kept sidelobes). Pure-Python trapezoid so the
    module imports without numpy."""
    frac = {}
    acc, prev = 0.0, _sinc2(0.0)
    per = int(round(1.0 / step))                    # samples per unit x; boundaries hit integers
    for i in range(1, (nmax + 1) * per + 1):
        cur = _sinc2(i * step)
        acc += 0.5 * (prev + cur) * step
        prev = cur
        if i % per == 0:                            # x == an integer == (n+1)
            frac[i // per - 1] = 2.0 * acc
    return tuple(frac[n] for n in range(nmax + 1))


_POWER_FRACTION = _power_fraction_table(MAX_SIDELOBES)   # frac(0..MAX_SIDELOBES)
CA_MAIN_LOBE_FRACTION = _POWER_FRACTION[0]               # I_ML ≈ 0.902823


def enbw_mhz(sidelobes: int) -> float:
    """The equivalent-noise bandwidth (MHz) mapping the measured PEAK density to the FULL power
    passed by the filter with `sidelobes` sidelobes: full_dBm = peak_dBm/Hz + 10·log10(enbw·1e6).
    Equals Rc·frac(n); passed live to the power map so the delivered power and the limiting cap both
    track the sidelobe count as it is tuned."""
    n = max(0, min(MAX_SIDELOBES, int(sidelobes)))
    return (CODE_RATE_HZ / 1e6) * _POWER_FRACTION[n]


# Static enbw_mhz(n) lookup for the GUI — the client's schema extractor is a static AST reader (it
# can't run the sinc² integration), and the full-power law keys on enbw_mhz (a value with no input
# field), so the schema exposes it as a HIDDEN derived field: a nearest-integer table lookup on
# --sidelobes. The first element names the source field; the rest are enbw_mhz(0..MAX_SIDELOBES).
# Kept a literal so the extractor can read it; --self-test asserts it matches enbw_mhz().
_ENBW_TABLE_ARGS = [
    "sidelobes",
    0.923588, 0.971788, 0.988638, 0.997168, 1.002311, 1.005749, 1.008208, 1.010054,
    1.011490, 1.012640, 1.013581, 1.014365, 1.015029, 1.015598, 1.016091, 1.016523,
    1.016904, 1.017242, 1.017545, 1.017818, 1.018065, 1.018289, 1.018494, 1.018682,
    1.018854, 1.019014, 1.019161, 1.019298, 1.019426,
]


# The power-quantity conversion laws this signal OFFERS the calibration editor (verbatim from
# gps_ca_code_1.023Mcps.py). Both convert the measured spectral density (dBm/Hz at the peak) to an
# absolute power (dBm): 60 = 10·log10(1 MHz / 1 Hz); the full-power term adds 10·log10(enbw_mhz);
# the main-lobe k = 10·log10(Rc · I_ML) = 59.654784. `rep` = enbw_mhz(DEFAULT_SIDELOBES).
CAL_POWER_LAWS = [
    {"id": "full_power", "name": "Full signal power (filter passband)", "unit": "dBm",
     "in": "density", "out": "abs",
     "k": 60.0, "param": "enbw_mhz", "coeff": 10.0, "ref": 1.0, "rep": 0.988638},
    {"id": "main_lobe_power", "name": "Main-lobe integrated power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 59.654784},
]

log = logging.getLogger("mock_gps_ca_tx")

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE / --calibration), else uncalibrated (relative gain only). Cached so
    build_script and main share one — and so --power's schema bounds match the real range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── The fake radio: logs instead of touching hardware ───────────────────────────

class FakeRadio:
    """Stand-in for the GNU Radio flowgraph. Records / logs the gain, carrier and amplitude it
    would command; builds no Gold code, no filtered loop, and transmits nothing."""

    def __init__(self, freq_hz: float):
        self._gain = 0.0
        self._amp = 0.0
        self._freq = float(freq_hz)

    def set_gain(self, g: float) -> None:
        self._gain = float(g)
        log.info("  radio.set_gain(%.2f dB)", self._gain)

    def set_center_frequency(self, hz: float) -> None:
        self._freq = float(hz)
        log.info("  radio.set_center_freq(%.3f MHz)", self._freq / 1e6)

    def set_amplitude(self, a: float) -> None:
        self._amp = float(a)
        log.info("  radio.set_amplitude(%.3f)", self._amp)

    def swap_filter(self, sidelobes: int) -> None:
        log.info("  radio.swap_filter(main + %d sidelobe(s), ±%.2f MHz)",
                 sidelobes, (sidelobes + 1) * CA_NULL_MHZ)

    def actual_gain(self) -> float:
        return self._gain             # a real SDR quantises; the mock reports what it was set to

    def actual_freq(self) -> float:
        return self._freq


# ── Parameter schema (verbatim from gps_ca_code_1.023Mcps.py) ────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Mock GPS C/A (1.023 Mcps) transmitter (NO HARDWARE) — same calibration surface as "
               "gps_ca_code_1.023Mcps.py (power card, spectral-density laws, the --sidelobes "
               "filter), but logs the SDR gain it would command instead of transmitting. Level is "
               "set in dBm via the unit's calibration; uncalibrated it runs on a relative gain.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .number("-Center-frequency", "--freq", unit="MHz", min=1000, max=1800,
                presets=FREQUENCIES, default=CARRIER_HZ / 1e6,
                help="RF carrier in MHz (default L1 = 1575.42; L2 = 1227.6 preset). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="GPS satellite PRN / Gold code index (1..32). Fixed per run.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, step=1,
                 default=DEFAULT_SIDELOBES, required=False, live=True,
                 help="Passband width, as the number of C/A sidelobes KEPT beside the main "
                      "lobe: a ±(n+1)·1.023 MHz band. 0 keeps the main lobe only. The filter is "
                      "always on (unity passband gain). More sidelobes pass more of the signal's "
                      "power (the full-power calibration quantity tracks this). Max 28 keeps the "
                      "band inside the sample rate. Live (rebuilds the filtered loop).")
        .derived("-Passband-bandwidth", name="passband_bw_mhz", unit="MHz",
                 formula={"linear": ["sidelobes", 2.046, 2.046]},
                 help="Occupied bandwidth the filter passes at the current sidelobe count: "
                      "2·(n+1)·1.023 MHz (i.e. ±(n+1)·1.023 MHz).")
        .derived("-Full-power-bandwidth", name="enbw_mhz", unit="MHz", hidden=True,
                 formula={"table": _ENBW_TABLE_ARGS},
                 help="Equivalent-noise bandwidth mapping the measured peak density to the full "
                      "in-band power. Feeds the full-power calibration law; not shown.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Standalone calibration helper (no agent/SDR needed to try it out) ────────────────────────────

def _make_sample_calibration(out_path: str) -> int:
    """Build a representative GPS C/A unit calibration and resolve it to an artifact JSON, so the
    mock can be run standalone. The SDR is measured in spectral density (dBm/Hz at the main-lobe
    peak) over a 0..89.75 dB / 0.25 dB gain grid; the dBm safety ceiling is gauged through the
    full-power law (a density measurement requires a law that returns dBm). Needs the resolver,
    which lives in sdr-agent (put it on PYTHONPATH)."""
    import json
    try:
        from agent.calibration import resolve
    except Exception as exc:                        # noqa: BLE001 — surface any import failure
        print("error: could not import the calibration resolver (agent.calibration). Put the "
              "sdr-agent checkout on PYTHONPATH, e.g. PYTHONPATH=/path/to/sdr-agent. "
              f"({exc})", file=sys.stderr)
        return 2
    full = dict(CAL_POWER_LAWS[0])                   # the full_power (density → dBm) law
    points = [{"gain_db": 0.0, "power_dbm": -145.0}, {"gain_db": 89.75, "power_dbm": -57.5}]
    doc = {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": 0.0, "max_gain_db": 89.75, "gain_step_db": 0.25},
            "operating_plane": "sdr_output",
            "limits": [{"plane": "sdr_output", "max_dbm": 2.0, "reason": "SDR TX P1dB"}],
            "planes": {"sdr_output": {
                "type": "measured", "quantity": "spectral density",
                "limiting": {"kind": "law", "law": full}}},
        },
        "signals": {CAL_SIGNAL_ID: {
            "measurement": {"quantity": "spectral density", "unit": "dBm/Hz"},
            "curves": {"sdr_output": {"interp": "linear", "points": points}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": AMPLITUDE},
    }
    artifact = resolve(doc, None, CAL_SIGNAL_ID).to_public_dict()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"wrote sample {CAL_SIGNAL_ID} calibration → {out_path}")
    print("run e.g.:  mock_gps_ca_code_1.023Mcps_tx.py --calibration %s --power -30 --sidelobes 5 "
          "--once" % out_path)
    return 0


def _pop_option(argv, flag):
    """Remove ``flag <value>`` from argv (in place) and return the value, or None. Lets the mock
    accept its own options (--calibration, --make-sample-calibration) that paramkit doesn't know."""
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            val = argv[i + 1]
            del argv[i:i + 2]
            return val
        del argv[i:i + 1]
    return None


# ── Self-test: exercise the density→gain math + the enbw table, no loop, no hardware ─────────────

def _self_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    fr = _POWER_FRACTION
    mono = all(fr[i] < fr[i + 1] for i in range(len(fr) - 1))
    bounded = 0.9025 < fr[0] < 0.9035 and fr[-1] < 1.0
    full0 = 60.0 + 10 * math.log10(enbw_mhz(0))     # full(0) must equal the main-lobe k
    table_ok = (len(_ENBW_TABLE_ARGS) == MAX_SIDELOBES + 2
                and _ENBW_TABLE_ARGS[0] == "sidelobes"
                and all(abs(_ENBW_TABLE_ARGS[1 + n] - enbw_mhz(n)) < 5e-6
                        for n in range(MAX_SIDELOBES + 1)))
    laws_ok = mono and bounded and abs(full0 - 59.654784) < 0.01 and table_ok
    log.info("calibration: I_ML=%.6f, frac(max)=%.6f, full(0)=%.4f dB == main-lobe 59.6548 dB, "
             "enbw table %s [%s]", fr[0], fr[-1], full0,
             "matches" if table_ok else "DRIFTED", "OK" if laws_ok else "FAIL")

    pmap = power_map()
    log.info("power map source : %s", pmap.source)
    log.info("gain limits      : %.2f … %.2f dB", pmap.min_gain_db, pmap.max_gain_db)
    if pmap.has_absolute:
        log.info("power range      : %.2f … %.2f (base quantity)",
                 pmap.min_power_dbm, pmap.max_power_dbm)
        req = round((pmap.min_power_dbm + pmap.max_power_dbm) / 2.0, 2)
        log.info("mapping --power %g (base) across sidelobe counts:", req)
        for n in (0, 5, 12, 28):
            g = pmap.gain_for_power(req, freq=CARRIER_HZ, params={"enbw_mhz": enbw_mhz(n)})
            back = pmap.power_for_gain(g, freq=CARRIER_HZ, params={"enbw_mhz": enbw_mhz(n)})
            log.info("  sidelobes %2d (enbw %.4f MHz) → gain %6.2f dB → reads back %+8.3f",
                     n, enbw_mhz(n), g, back)
    else:
        log.info("power range      : (uncalibrated — pass --calibration / SDR_CALIBRATION_FILE, "
                 "or use --gain)")
    log.info("SELF-TEST OK" if laws_ok else "SELF-TEST FAILED")
    return 0 if laws_ok else 1


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    argv = sys.argv[1:]
    sample_out = _pop_option(argv, "--make-sample-calibration")
    if sample_out is not None:
        return _make_sample_calibration(sample_out)
    cal_file = _pop_option(argv, "--calibration")
    if cal_file is not None:
        os.environ["SDR_CALIBRATION_FILE"] = cal_file
    once = "--once" in argv
    if once:
        argv = [a for a in argv if a != "--once"]
    if "--self-test" in argv:
        return _self_test()
    sys.argv = [sys.argv[0], *argv]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    script = build_script()
    args = script.parse()
    center_freq_hz = args.freq * 1e6

    pmap = power_map()
    amplitude = pmap.amplitude

    shape = {"sidelobes": int(getattr(args, "sidelobes", DEFAULT_SIDELOBES) or 0)}

    def pwr_params() -> dict:
        """The live keyed-parameter values the power laws read: the filter's equivalent-noise
        bandwidth, so the FULL-power reading and its limiting cap track the sidelobe count."""
        return {"enbw_mhz": enbw_mhz(shape["sidelobes"])}

    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(gain_cal)))
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz, params=pwr_params())
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; pass --calibration/SDR_CALIBRATION_FILE, or set a "
                  "relative --gain (the client does this for you).", file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    radio = FakeRadio(center_freq_hz)
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": center_freq_hz, "power": _target_power}

    log.info("── mock GPS C/A (1.023 Mcps) TX (no hardware) ──────────────")
    log.info("  signal id      : %s", CAL_SIGNAL_ID)
    log.info("  PRN            : %s", getattr(args, "prn", 1))
    log.info("  carrier        : %.3f MHz", center_freq_hz / 1e6)
    log.info("  sidelobes      : %d (±%.2f MHz passband, enbw %.4f MHz)",
             shape["sidelobes"], (shape["sidelobes"] + 1) * CA_NULL_MHZ, enbw_mhz(shape["sidelobes"]))
    if pmap.has_absolute and gain_cal is None:
        log.info("  power (target) : %g  (%s)", args.power, pmap.label)
        log.info("  power (on grid): %.2f",
                 pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params()))
    log.info("  → gain         : %.2f dB (max %g), amplitude %g",
             gain_db, pmap.max_gain_db, amplitude)
    log.info("  calibration    : %s", pmap.describe())
    if pmap.warning:
        log.info("  ⚠ CALIBRATION  : %s", pmap.warning)
    log.info("  RF             : %s", "ON" if state["rf_on"] else "OFF (muted)")
    if gain_cal is not None:
        log.info("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    log.info("────────────────────────────────────────────────────────────")
    _grid = (pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params())
             if pmap.has_absolute else None)
    print("RESULT gain_db=%.6g power_dbm=%s source=%s"
          % (gain_db, ("%.6g" % _grid) if _grid is not None else "na",
             "calibrated" if pmap.has_absolute else "uncalibrated"))
    sys.stdout.flush()

    if state["rf_on"]:
        radio.set_amplitude(amplitude)
        radio.set_gain(state["gain"])
    else:
        radio.set_gain(0.0)
        radio.set_amplitude(0.0)

    if once:                                         # one-shot: no live loop
        return 0

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "power" and pmap.has_absolute:
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(float(value), freq=center_freq_hz,
                                                params=pwr_params())
            log.info("live: --power %s → gain %.2f dB", value, state["gain"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=center_freq_hz,
                                                           params=pwr_params()), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            state["power"] = None                    # raw gain takes over the level
            log.info("live: --gain %.2f dB (raw)", state["gain"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            ctrl.report("gain", round(state["gain"], 2))
        elif name == "rf":
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            log.info("live: --rf %s", "on" if on else "off")
            if on:
                radio.set_amplitude(amplitude)
                radio.set_gain(state["gain"])
            else:
                radio.set_gain(0.0)
                radio.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")
        elif name == "sidelobes":
            shape["sidelobes"] = max(0, min(MAX_SIDELOBES, int(value)))
            radio.swap_filter(shape["sidelobes"])
            # Widening/narrowing the passband changes the equivalent bandwidth, so a held absolute
            # --power must re-map to keep the delivered (full) power constant; the amp's limiting
            # cap moves with it too. A main-lobe/relative target is unaffected (the embedded law).
            if state["power"] is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=center_freq_hz,
                                                    params=pwr_params())
                if state["rf_on"]:
                    radio.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(
                    state["gain"], freq=center_freq_hz, params=pwr_params()), 2))
            ctrl.report("sidelobes", shape["sidelobes"])

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    while not stop.is_set():
        for change in ctrl.drain():
            apply_change(change.name, change.value)
        time.sleep(0.1)

    ctrl.close()
    log.info("mock GPS C/A TX stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

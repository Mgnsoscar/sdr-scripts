#!/usr/bin/env python3
"""
mock_fm_chirp_tx — a NO-HARDWARE stand-in for fm_chirp_tx.py.

Same calibration surface as the real FM-chirp transmitter (identical parameter schema,
CAL_SIGNAL_ID, CAL_FREQ_PARAM and CAL_POWER_LAWS), so the client's Run/tune form drives it
EXACTLY like the real chirp — the same power card with its density / total-power / dBm-per-Hz
quantities. But it never imports UHD / GNU Radio and never transmits: a fake "radio" only LOGS
the SDR gain (and amplitude) it *would* command. Use it to check, with no SDR connected, that a
requested ``--power`` — in whatever quantity you set it — maps to the right gain, and that it
re-maps correctly as you live-tune the carrier or sweep bandwidth.

How --power → gain works (identical to the real chirp)
─────────────────────────────────────────────────────
The client always SENDS ``--power`` in the calibration's base (measured/reported) quantity,
whatever quantity you were controlling it in on the form. This script maps that number to a gain
through the unit's calibration, folded at the current carrier and sweep bandwidth:

    gain = power_map().gain_for_power(--power, freq=carrier, params={"bw": sweep_bw_MHz})

so it is the same fold, on the same PowerMap, that the real transmitter uses. Switching which
quantity you control on the form never changes the commanded output — only which number you type
— so the gain this mock reports is the ground truth for "did the power quantities map correctly".

Getting a calibration (needed for absolute --power)
───────────────────────────────────────────────────
Absolute ``--power`` (dBm / a density) only has meaning with a calibration; uncalibrated the
script runs on a relative ``--gain``. Three ways to supply one:
  • under the agent — a task with SDR_CAL_SIGNAL_ID=fm_chirp gets this unit's resolved
    calibration injected (env SDR_CALIBRATION_FILE), exactly like the real chirp;
  • --calibration <artifact.json> — point it at a resolved calibration artifact yourself;
  • --make-sample-calibration <out.json> — build + resolve a representative chirp calibration
    (a spectral-density measurement at the 10 MHz reference sweep) and write it out, so you can
    run standalone. Needs sdr-agent on PYTHONPATH (the resolver lives there).

CLI
───
    # one-shot: print the gain a request maps to and exit
    mock_fm_chirp_tx.py --calibration cal.json --freq 1575.42 --bw 20 --power -120 --once
    mock_fm_chirp_tx.py --calibration cal.json --gain 60 --once      # raw-gain override

    # generate a sample calibration to test against (needs sdr-agent on PYTHONPATH)
    PYTHONPATH=/path/to/sdr-agent mock_fm_chirp_tx.py --make-sample-calibration cal.json

    # run like a task (long-lived, live-tunable from the client), no hardware
    mock_fm_chirp_tx.py --freq 1575.42 --bw 20 --rate 200 --power -120

    mock_fm_chirp_tx.py --self-test        # exercise the density→gain math, no loop
    mock_fm_chirp_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

# Make paramkit + calkit importable both on a unit (scripts flattened one level under BASE_DIR,
# next to paramkit/) and in the dev repo (scripts two levels under the repo root). Insert both
# candidate roots; whichever holds paramkit wins. (PYTHONPATH is honoured too, e.g. the agent.)
_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from paramkit import Script, PowerMap

# ── Calibration surface — kept IN STEP with fm_chirp_tx.py so the client treats this mock as the
#    real chirp. The static schema reader (agent/argspec.py) reads these literals from THIS file,
#    so they must live here verbatim (an import wouldn't be seen). ────────────────────────────────
CAL_SIGNAL_ID = "fm_chirp"          # same signal id as fm_chirp_tx.py → same injected calibration
CAL_FREQ_PARAM = "freq"             # the carrier the calibration folds at (live)
CAL_MEAS_BW_MHZ = 10.0              # sweep bandwidth the spectral-density calibration is measured at

# Power-quantity conversion laws this signal offers (verbatim from fm_chirp_tx.py). A chirp's
# baseband is CONSTANT-AMPLITUDE, so its TOTAL power depends only on gain; widening the sweep
# spreads the same power over more spectrum (density drops). From one density measured at
# CAL_MEAS_BW_MHZ, both readings below are exact at any live sweep width.
CAL_POWER_LAWS = [
    {"id": "psd_live", "name": "Spectral density", "unit": "dBm/MHz",
     "in": "density", "out": "density", "restates_measurement": True,
     "param": "bw", "coeff": -10.0, "ref": 10.0, "rep": 10.0},   # −10·log10(bw / CAL_MEAS_BW_MHZ)
    {"id": "fbw_power", "name": "Full-bandwidth (total) power", "unit": "dBm",
     "in": "density", "out": "abs",
     "k": 10.0, "rep": 10.0},                                    # +10·log10(CAL_MEAS_BW_MHZ)
    {"id": "psd_hz", "name": "Spectral density", "unit": "dBm/Hz",
     "in": "density", "out": "density", "restates_measurement": True,
     "param": "bw", "coeff": -10.0, "ref": 10.0, "k": -60.0, "rep": 10.0},
]

# ── RF-chain limits (no baked dBm scale; absolute --power comes only from the calibration) ───────
GAIN_AT_MAX_DB = 89.75              # gain ceiling: never command a gain above this
HW_MAX_GAIN_DB = 89.75
AMPLITUDE = 0.5                     # the amplitude the calibration is measured at
MAX_SWEEP_BW_MHZ = 55.0            # sweep is ±bw/2; keep inside ±Nyquist with margin

# Named GNSS carriers (MHz), same preset list as the real chirp so the form matches.
FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": 1575.42, "GPS L2 (1227.60 MHz)": 1227.60,
    "GPS L5 (1176.45 MHz)": 1176.45,
    "Galileo E1 (1575.42 MHz)": 1575.42, "Galileo E5a (1176.45 MHz)": 1176.45,
    "Galileo E5b (1207.14 MHz)": 1207.14,
    "Galileo E5 (1191.795 MHz)": 1191.795, "Galileo E6 (1278.75 MHz)": 1278.75,
    "BeiDou B1I (1561.098 MHz)": 1561.098, "BeiDou B1C (1575.42 MHz)": 1575.42,
    "BeiDou B2a (1176.45 MHz)": 1176.45,
    "BeiDou B2b (1207.14 MHz)": 1207.14, "BeiDou B2 (1191.795 MHz)": 1191.795,
    "BeiDou B3 (1268.52 MHz)": 1268.52,
    "GLONASS L1 (1602.0 MHz)": 1602.0, "GLONASS L2 (1246.0 MHz)": 1246.0,
    "GLONASS L3 (1202.025 MHz)": 1202.025,
    "Iridium (1621.25 MHz)": 1621.25,
}

log = logging.getLogger("mock_fm_chirp_tx")

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
    would command; builds no buffer and transmits nothing."""

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

    def actual_gain(self) -> float:
        return self._gain             # a real SDR quantises; the mock reports what it was set to

    def actual_freq(self) -> float:
        return self._freq


# ── Parameter schema (verbatim from fm_chirp_tx.py, minus --rate's buffer effect) ───────────────

def build_script() -> Script:
    return (
        Script("Mock FM-chirp transmitter (NO HARDWARE) — same calibration surface as "
               "fm_chirp_tx.py (power card, laws, band modes), but logs the SDR gain it would "
               "command instead of transmitting. Level is set in dBm via the unit's calibration; "
               "uncalibrated it runs on a relative gain.")
        .choice("-Band-mode", "--band-mode",
                options={"Centre + width": "center_bw", "Start / stop": "start_stop"},
                default="center_bw", required=False,
                help="How the sweep band is entered: a centre carrier + width, or absolute "
                     "start/stop edges (carrier = their midpoint, width = their span). Fixed "
                     "at launch.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=1575.42, required=False, live=True,
                show_when={"band_mode": "center_bw"},
                help="RF carrier in MHz. Live.")
        .number("-Start-frequency", "--start", unit="MHz", min=70.0, max=6000.0,
                step=0.01, default=1570.42, required=False, live=True,
                show_when={"band_mode": "start_stop"},
                help="Sweep start — the low RF edge, in MHz. Live.")
        .number("-Stop-frequency", "--stop", unit="MHz", min=70.0, max=6000.0,
                step=0.01, default=1580.42, required=False, live=True,
                show_when={"band_mode": "start_stop"},
                help="Sweep stop — the high RF edge, in MHz. Live.")
        .derived("-Carrier", name="band_center", unit="MHz",
                 formula={"center": ["start", "stop"]}, is_freq=True,
                 show_when={"band_mode": "start_stop"},
                 help="Carrier the LO tunes to = midpoint of start/stop. Power is calibrated "
                      "here.")
        .derived("-Sweep-width", name="band_span", unit="MHz",
                 formula={"span": ["start", "stop"]}, min=0.001, max=MAX_SWEEP_BW_MHZ,
                 show_when={"band_mode": "start_stop"}, provides="bw",
                 help="Resulting sweep width = stop − start. This is the sweep bandwidth the "
                      "calibration power laws key on in start/stop mode (it stands in for --bw).")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration (folded at the current carrier and sweep bandwidth) and snaps "
                     "to its achievable grid; ignored if --gain is given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .number("-Sweep-BW", "--bw", unit="MHz", min=0.001, max=MAX_SWEEP_BW_MHZ, default=20.0,
                required=False, live=True, show_when={"band_mode": "center_bw"},
                help="Peak-to-peak sweep width; f sweeps ±bw/2 around the carrier. The power "
                     "laws key on this, so widening the sweep drops the density. Live.")
        .number("-Sweep-rate", "--rate", unit="kHz", min=0.1, max=5000.0,
                default=200.0, required=True, live=True,
                help="How fast the sweep repeats, in kHz. Informational for the mock. Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Band resolution (center+width ↔ start/stop), verbatim from fm_chirp_tx.py ────────────────────

def resolve_band(band_mode: str, freq, bw, start, stop):
    """Resolve the sweep band from whichever mode is selected into a canonical
    (center_freq_hz, sweep_bw_hz). In 'start_stop' the carrier is the midpoint and the width is
    the span; in 'center_bw' they are given directly."""
    if band_mode == "start_stop":
        if start is None or stop is None:
            raise ValueError("start_stop mode needs both --start and --stop (MHz).")
        lo, hi = sorted((float(start), float(stop)))
        span = hi - lo
        if span <= 0:
            raise ValueError("--start and --stop must differ (the sweep has zero width).")
        if span > MAX_SWEEP_BW_MHZ:
            raise ValueError(
                f"start/stop span {span:g} MHz exceeds the maximum sweep width "
                f"{MAX_SWEEP_BW_MHZ:g} MHz — narrow the start/stop range.")
        return (lo + hi) / 2.0 * 1e6, span * 1e6
    if freq is None or bw is None:
        raise ValueError("center_bw mode needs --freq and --bw (MHz).")
    return float(freq) * 1e6, float(bw) * 1e6


# ── Standalone calibration helpers (no agent/SDR needed to try it out) ───────────────────────────

def _make_sample_calibration(out_path: str) -> int:
    """Build a representative fm_chirp unit calibration and resolve it to an artifact JSON, so the
    mock can be run standalone. The SDR is measured in spectral density (dBm/MHz) at the 10 MHz
    reference sweep over a 0..70 dB / 0.5 dB gain grid (density = gain − 150); the dBm safety
    ceiling is gauged through the total-power (fbw) law, as a density measurement requires. Needs
    the resolver, which lives in sdr-agent (put it on PYTHONPATH)."""
    import json
    try:
        from agent.calibration import resolve
    except Exception as exc:                        # noqa: BLE001 — surface any import failure
        print("error: could not import the calibration resolver (agent.calibration). Put the "
              "sdr-agent checkout on PYTHONPATH, e.g. PYTHONPATH=/path/to/sdr-agent. "
              f"({exc})", file=sys.stderr)
        return 2
    fbw = {"id": "fbw_power", "name": "Full-bandwidth (total) power",
           "in": "density", "out": "abs", "k": 10.0, "rep": 10.0}
    points = [{"gain_db": 0.0, "power_dbm": -150.0}, {"gain_db": 70.0, "power_dbm": -80.0}]
    doc = {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": 0.0, "max_gain_db": 70.0, "gain_step_db": 0.5},
            "operating_plane": "sdr_output",
            "planes": {"sdr_output": {
                "type": "measured", "quantity": "spectral density",
                # A density measurement can't be capped by a bare dBm ceiling, so the limiting
                # reading is the total-power law (returns dBm). Set high so it never binds here.
                "limiting": {"kind": "law", "law": fbw, "max_dbm": 0.0}}},
        },
        "signals": {CAL_SIGNAL_ID: {
            "measurement": {"quantity": "spectral density", "unit": "dBm/MHz"},
            "curves": {"sdr_output": {"interp": "linear", "points": points}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": AMPLITUDE},
    }
    artifact = resolve(doc, None, CAL_SIGNAL_ID).to_public_dict()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"wrote sample fm_chirp calibration → {out_path}")
    print("run e.g.:  mock_fm_chirp_tx.py --calibration %s --bw 20 --power -120 --once"
          % out_path)
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


# ── Self-test: exercise the density→gain math, no loop, no hardware ─────────────

def _self_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    pmap = power_map()
    log.info("power map source : %s", pmap.source)
    log.info("operating label  : %s", pmap.label)
    log.info("gain limits      : %.2f … %.2f dB", pmap.min_gain_db, pmap.max_gain_db)
    if not pmap.has_absolute:
        log.info("power range      : (uncalibrated — no absolute scale; pass --calibration or "
                 "SDR_CALIBRATION_FILE, or use --gain)")
        log.info("SELF-TEST OK")
        return 0
    log.info("power range      : %.2f … %.2f (base quantity)", pmap.min_power_dbm, pmap.max_power_dbm)
    freq = 1575.42e6
    req = round((pmap.min_power_dbm + pmap.max_power_dbm) / 2.0, 2)
    log.info("mapping --power %g (base) at %.2f MHz across sweep widths:", req, freq / 1e6)
    for bw in (5.0, 10.0, 20.0, 40.0):
        g = pmap.gain_for_power(req, freq=freq, params={"bw": bw})
        back = pmap.power_for_gain(g, freq=freq, params={"bw": bw})
        log.info("  bw %5.1f MHz → gain %6.2f dB → reads back %+8.3f", bw, g, back)
    log.info("(the base quantity maps bw-independently; the client converts your chosen quantity "
             "to it, so total-power / density track --bw before the send.)")
    log.info("SELF-TEST OK")
    return 0


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    argv = sys.argv[1:]
    sample_out = _pop_option(argv, "--make-sample-calibration")
    if sample_out is not None:
        return _make_sample_calibration(sample_out)
    # --calibration <file> is a standalone convenience for SDR_CALIBRATION_FILE; set it BEFORE
    # power_map() is first built (it caches the loaded map).
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
    try:
        center_freq_hz, sweep_bw_hz = resolve_band(
            getattr(args, "band_mode", "center_bw"),
            getattr(args, "freq", None), getattr(args, "bw", None),
            getattr(args, "start", None), getattr(args, "stop", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pmap = power_map()
    amplitude = pmap.amplitude

    def pwr_params():
        """Live params a power-quantity bridge may key on — the current sweep bandwidth in MHz,
        so a bridged --power (or a law companion) tracks --bw as it is tuned."""
        return {"bw": shape["bw_hz"] / 1e6}

    shape = {"bw_hz": sweep_bw_hz}

    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(gain_cal)))
    elif pmap.has_absolute:                          # calibrated: the authored absolute --power
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz, params=pwr_params())
    else:                                            # uncalibrated: a persisted fallback gain, or refuse
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
             "freq": center_freq_hz, "power": _target_power,
             "start": getattr(args, "start", None), "stop": getattr(args, "stop", None)}

    band_mode = getattr(args, "band_mode", "center_bw")
    log.info("── mock FM chirp TX (no hardware) ──────────────────────────")
    log.info("  signal id      : %s", CAL_SIGNAL_ID)
    log.info("  band mode      : %s", band_mode)
    log.info("  carrier        : %.3f MHz", center_freq_hz / 1e6)
    log.info("  sweep bw       : %g MHz (±%g MHz)", sweep_bw_hz / 1e6, sweep_bw_hz / 2e6)
    log.info("  sweep rate     : %g kHz", getattr(args, "rate", 0.0))
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
    # One machine-readable line for tests / tooling (mirrors mock_sdr_tx.py).
    _grid = (pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params())
             if pmap.has_absolute else None)
    print("RESULT gain_db=%.6g power_dbm=%s source=%s"
          % (gain_db, ("%.6g" % _grid) if _grid is not None else "na",
             "calibrated" if pmap.has_absolute else "uncalibrated"))
    sys.stdout.flush()

    # Apply the initial state to the (fake) radio.
    if state["rf_on"]:
        radio.set_amplitude(amplitude)
        radio.set_gain(state["gain"])
    else:
        radio.set_gain(0.0)
        radio.set_amplitude(0.0)

    if once:                                         # one-shot: no live loop
        return 0

    ctrl = script.live_control(args)

    def _report_power():
        g = radio.actual_gain() if state["rf_on"] else state["gain"]
        ctrl.report("power", round(pmap.power_for_gain(g, freq=state["freq"],
                                                       params=pwr_params()), 2))

    def apply_change(name, value):
        # Mirrors fm_chirp_tx.py: power/gain edits stage into state["gain"] and only reach the
        # (fake) radio while RF is on; a freq/bw change re-maps the held target power so the
        # delivered (reported) power stays as requested; --rf mutes/restores gain + amplitude.
        if name == "freq":
            hz = float(value) * 1e6
            radio.set_center_frequency(hz)
            state["freq"] = hz
            ctrl.report("freq", radio.actual_freq() / 1e6)
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                    params=pwr_params())
                if state["rf_on"]:
                    radio.set_gain(state["gain"])
                _report_power()
        elif name == "power":
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                params=pwr_params())
            log.info("live: --power %s → gain %.2f dB", value, state["gain"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            else:
                log.info("  (staged — RF is off; applies on next --rf on)")
            _report_power()
        elif name == "gain":
            state["power"] = None                    # a raw gain drops any held target power
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            log.info("live: --gain %.2f dB (raw)", state["gain"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            ctrl.report("gain", round(state["gain"], 2))
        elif name in ("start", "stop"):
            state[name] = float(value)
            if state.get("start") is not None and state.get("stop") is not None:
                try:
                    c_hz, bw_hz = resolve_band("start_stop", None, None,
                                               state["start"], state["stop"])
                except ValueError:
                    ctrl.report(name, value)         # keep the last good band
                    return
                radio.set_center_frequency(c_hz)
                state["freq"] = c_hz
                shape["bw_hz"] = bw_hz
                if state.get("power") is not None:
                    state["gain"] = pmap.gain_for_power(state["power"], freq=c_hz,
                                                        params=pwr_params())
                    if state["rf_on"]:
                        radio.set_gain(state["gain"])
                    _report_power()
            ctrl.report(name, value)
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
        elif name == "bw":
            shape["bw_hz"] = float(value) * 1e6
            # The full-bandwidth-power law keys on --bw, so a held target power in a bw-keyed
            # quantity re-maps: keep the delivered reported power as requested.
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                    params=pwr_params())
                log.info("live: --bw %s MHz → gain %.2f dB", value, state["gain"])
                if state["rf_on"]:
                    radio.set_gain(state["gain"])
                _report_power()
            else:
                ctrl.report("bw", value)
        elif name == "rate":
            ctrl.report("rate", value)               # informational for the mock

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    while not stop.is_set():
        for change in ctrl.drain():
            apply_change(change.name, change.value)
        time.sleep(0.1)

    ctrl.close()
    log.info("mock FM chirp TX stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

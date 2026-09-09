#!/usr/bin/env python3
"""
mock_cw_tx — a NO-HARDWARE stand-in for cw_tx.py (the pure CW tone).

Same calibration surface as the real CW transmitter (identical parameter schema, CAL_SIGNAL_ID
and CAL_FREQ_PARAM), so the client's Run/tune form drives it EXACTLY like the real tone — the
calibrated dBm --power field folded at the live carrier. But it never imports UHD / GNU Radio and
never transmits: a fake "radio" only LOGS the SDR gain (and amplitude) it *would* command. Use it
to check, with no SDR connected, that a requested dBm --power maps to the right gain and re-maps as
you live-tune the carrier.

CW carries NO power-quantity laws — a tone is a single dBm quantity — so there is no power card,
just the dBm field (matching cw_tx.py).

Getting a calibration (needed for absolute --power)
───────────────────────────────────────────────────
Absolute --power (dBm) only has meaning with a calibration; uncalibrated the script runs on a
relative --gain. Three ways to supply one:
  • under the agent — a task with SDR_CAL_SIGNAL_ID="cw_tone" gets this unit's resolved calibration
    injected (env SDR_CALIBRATION_FILE), exactly like the real tone;
  • --calibration <artifact.json> — point it at a resolved calibration artifact yourself;
  • --make-sample-calibration <out.json> — build + resolve a representative dBm calibration so you
    can run standalone. Needs sdr-agent on PYTHONPATH (the resolver lives there).

CLI
───
    mock_cw_tx.py --calibration cal.json --freq 1575.42e6 --power -30 --once
    mock_cw_tx.py --gain 60 --once            # raw-gain override
    PYTHONPATH=/path/to/sdr-agent mock_cw_tx.py --make-sample-calibration cal.json
    mock_cw_tx.py --freq 1575.42e6 --power -30 --rf on   # run like a task (no hardware)
    mock_cw_tx.py --self-test                 # exercise the dBm→gain math, no loop
    mock_cw_tx.py --describe-params           # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import logging
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

# ── Calibration surface — kept IN STEP with cw_tx.py so the client treats this mock as the real
#    tone. The static schema reader (agent/argspec.py) reads these literals from THIS file, so they
#    must live here verbatim (an import wouldn't be seen). ─────────────────────────────────────────
CAL_SIGNAL_ID = "cw_tone"           # same signal id as cw_tx.py → same injected calibration
CAL_FREQ_PARAM = "freq"             # the carrier the calibration folds at (live)

# ── RF-chain limits (no baked dBm scale; absolute --power comes only from the calibration) ───────
GAIN_AT_MAX_DB = 89.75              # gain ceiling: never command a gain above this
HW_MAX_GAIN_DB = 89.75             # B200-mini physical TX-gain ceiling
AMPLITUDE = 0.5                    # the amplitude the calibration is measured at

# Named GNSS carriers (Hz), same preset list as the real tone so the form matches.
FREQUENCIES = {
    "GPS L1 / Galileo E1 / BeiDou B1C (1575.42 MHz)": 1575.42e6,
    "GPS L2 (1227.60 MHz)": 1227.60e6,
    "GPS L5 / Galileo E5a (1176.45 MHz)": 1176.45e6,
    "Galileo E5b / BeiDou B2b (1207.14 MHz)": 1207.14e6,
    "Galileo E5 centre (1191.795 MHz)": 1191.795e6,
    "Galileo E6 (1278.75 MHz)": 1278.75e6,
    "BeiDou B1I (1561.098 MHz)": 1561.098e6,
    "BeiDou B3I (1268.52 MHz)": 1268.52e6,
    "GLONASS L1 (1602.0 MHz)": 1602.0e6,
    "GLONASS L2 (1246.0 MHz)": 1246.0e6,
    "Iridium (1621.25 MHz)": 1621.25e6,
}

log = logging.getLogger("mock_cw_tx")

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE / --calibration), else uncalibrated (relative gain only). Cached so
    build_script and main share one — and so --power's schema bounds match the real range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── The fake radio: records state instead of touching hardware ──────────────────

class FakeRadio:
    """Stand-in for the GNU Radio flowgraph. Records the gain, carrier and amplitude it would
    command; builds no buffer and transmits nothing. It stays SILENT — the agent's run/task log
    already records the launch and every parameter change, so the mock must not print them too."""

    def __init__(self, freq_hz: float):
        self._gain = 0.0
        self._amp = 0.0
        self._freq = float(freq_hz)

    def set_gain(self, g: float) -> None:
        self._gain = float(g)

    def set_center_frequency(self, hz: float) -> None:
        self._freq = float(hz)

    def set_amplitude(self, a: float) -> None:
        self._amp = float(a)

    def actual_gain(self) -> float:
        return self._gain             # a real SDR quantises; the mock reports what it was set to

    def actual_freq(self) -> float:
        return self._freq


# ── Parameter schema (verbatim from cw_tx.py) ────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Mock CW-tone transmitter (NO HARDWARE) — same calibration surface as cw_tx.py "
               "(a single steady tone, calibrated dBm --power), but logs the SDR gain it would "
               "command instead of transmitting. Level is set in dBm via the unit's calibration; "
               "uncalibrated it runs on a relative gain.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="Tone frequency. Presets are GNSS carriers; any value allowed. --power "
                     "is calibrated here. Live.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration (folded at the current carrier) and snaps to its achievable "
                     "grid; ignored if --gain is given. Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="off",
                required=False, live=True,
                help="RF output on/off. Starts OFF (muted pre-roll): set the power, then switch "
                     "ON to go on-air. OFF mutes gain AND baseband amplitude; power edits made "
                     "while OFF are staged and applied when you switch ON.")
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
    )


# ── Standalone calibration helper (no agent/SDR needed to try it out) ────────────────────────────

def _make_sample_calibration(out_path: str) -> int:
    """Build a representative CW (dBm) unit calibration and resolve it to an artifact JSON, so the
    mock can be run standalone. The SDR is measured in dBm over a 0..89.75 dB / 0.25 dB gain grid.
    Needs the resolver, which lives in sdr-agent (put it on PYTHONPATH)."""
    import json
    try:
        from agent.calibration import resolve
    except Exception as exc:                        # noqa: BLE001 — surface any import failure
        print("error: could not import the calibration resolver (agent.calibration). Put the "
              "sdr-agent checkout on PYTHONPATH, e.g. PYTHONPATH=/path/to/sdr-agent. "
              f"({exc})", file=sys.stderr)
        return 2
    points = [{"gain_db": 0.0, "power_dbm": -85.0}, {"gain_db": 89.75, "power_dbm": 2.5}]
    doc = {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": 0.0, "max_gain_db": 89.75, "gain_step_db": 0.25},
            "operating_plane": "sdr_output",
            "limits": [{"plane": "sdr_output", "max_dbm": 2.0, "reason": "SDR TX P1dB"}],
            "planes": {"sdr_output": {"type": "measured", "quantity": "power"}},
        },
        "signals": {CAL_SIGNAL_ID: {
            "measurement": {"quantity": "power", "unit": "dBm"},
            "curves": {"sdr_output": {"interp": "linear", "points": points}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": AMPLITUDE},
    }
    artifact = resolve(doc, None, CAL_SIGNAL_ID).to_public_dict()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"wrote sample {CAL_SIGNAL_ID} calibration → {out_path}")
    print("run e.g.:  mock_cw_tx.py --calibration %s --freq 1575.42e6 --power -30 --once"
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


# ── Self-test: exercise the dBm→gain math, no loop, no hardware ─────────────────

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
    log.info("power range      : %.2f … %.2f dBm", pmap.min_power_dbm, pmap.max_power_dbm)
    req = round((pmap.min_power_dbm + pmap.max_power_dbm) / 2.0, 2)
    for freq in (1575.42e6, 1227.6e6):
        g = pmap.gain_for_power(req, freq=freq)
        back = pmap.power_for_gain(g, freq=freq)
        log.info("  --power %g at %.2f MHz → gain %6.2f dB → reads back %+8.3f dBm",
                 req, freq / 1e6, g, back)
    log.info("SELF-TEST OK")
    return 0


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

    # Stay silent on a normal run — only errors reach the log. The agent's run/task log already
    # records the launch and every parameter change; printing them here too would double them.
    logging.basicConfig(level=logging.ERROR,
                        format="%(levelname)s %(message)s", stream=sys.stderr)

    script = build_script()
    args = script.parse()
    freq = float(args.freq)

    pmap = power_map()
    amplitude = pmap.amplitude

    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(gain_cal)))
    elif pmap.has_absolute:                          # calibrated: the authored absolute --power
        gain_db = pmap.gain_for_power(args.power, freq=freq)
    else:                                            # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; pass --calibration/SDR_CALIBRATION_FILE, or set a "
                  "relative --gain (the client does this for you).", file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    radio = FakeRadio(freq)
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "off") == "on", "gain": gain_db,
             "freq": freq, "power": _target_power}

    # Apply the initial state to the (fake) radio (default --rf off → muted pre-roll).
    if state["rf_on"]:
        radio.set_amplitude(amplitude)
        radio.set_gain(state["gain"])
    else:
        radio.set_gain(0.0)
        radio.set_amplitude(0.0)

    if once:                                         # one-shot: print the resolved mapping, no loop
        _grid = pmap.power_for_gain(gain_db, freq=freq) if pmap.has_absolute else None
        print("RESULT gain_db=%.6g power_dbm=%s source=%s"
              % (gain_db, ("%.6g" % _grid) if _grid is not None else "na",
                 "calibrated" if pmap.has_absolute else "uncalibrated"))
        sys.stdout.flush()
        return 0

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "freq":
            hz = float(value)
            radio.set_center_frequency(hz)
            state["freq"] = hz
            ctrl.report("freq", hz)
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=hz)
                if state["rf_on"]:
                    radio.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=hz), 2))
        elif name == "power":
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
        elif name == "gain":
            state["power"] = None                    # a raw gain drops any held target power
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                radio.set_gain(state["gain"])
            ctrl.report("gain", round(state["gain"], 2))
        elif name == "rf":
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            if on:
                radio.set_amplitude(amplitude)
                radio.set_gain(state["gain"])
            else:
                radio.set_gain(0.0)
                radio.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    while not stop.is_set():
        for change in ctrl.drain():
            apply_change(change.name, change.value)
        time.sleep(0.1)

    ctrl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

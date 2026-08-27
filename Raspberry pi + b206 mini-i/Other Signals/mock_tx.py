#!/usr/bin/env python3
"""
mock_tx — a NO-HARDWARE transmit script for exercising the power-calibration
machinery without any SDR connected.

It never imports UHD / GNU Radio and never transmits. A fake "radio" only LOGS the
gain and amplitude it *would* command. Use it to get familiar with everything we
built around calibration:

  • --power (dBm) and --rf on/off, and how a requested power maps to a gain;
  • the per-unit calibration artifact the agent injects (env SDR_CALIBRATION_FILE) —
    with it, --power reads through this unit's MEASURED curve at its real operating
    plane (e.g. EIRP); without it, the baked fallback constants below are used;
  • live retuning while the task runs (paramkit.live): change --power / --rf on the
    fly from FleetView and watch the log react.

Everything goes to the logger (stdout), which the agent captures as the task log.

CLI
───
    mock_tx.py --power -30 --rf on          # baked fallback (no SDR, no calibration)
    mock_tx.py --describe-params            # paramkit JSON schema for the GUI
    mock_tx.py --self-test                  # exercise the calibration math, no loop
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time

# Make paramkit + calkit importable both on a unit (scripts flattened one level
# under BASE_DIR, next to paramkit/) and in the dev repo (scripts two levels under
# the repo root). Insert both candidate roots; whichever holds paramkit/calkit wins.
_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from paramkit import Script, PowerMap

# Stable calibration signal id. Set a task's SDR_CAL_SIGNAL_ID to this and the agent
# injects this unit's resolved calibration for it (see the agent docs/calibration.md).
CAL_SIGNAL_ID = "mock"

log = logging.getLogger("mock_tx")


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, it runs on a relative gain.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # gain ceiling: never command a gain above this
HW_MAX_GAIN_DB = 89.75
AMPLITUDE = 0.5

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else an uncalibrated (relative gain only) map — no baked dBm
    scale. Cached, so --power's schema bounds match the real operating range when present."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── The fake radio: logs instead of touching hardware ──────────────────────────

class FakeRadio:
    """Stand-in for the GNU Radio flowgraph. Records/loggs the gain and amplitude it
    would command; transmits nothing."""

    def __init__(self, freq_hz: float):
        self._gain = 0.0
        self._amp = 0.0
        self._freq = freq_hz

    def set_gain(self, g: float) -> None:
        self._gain = float(g)
        log.info("  radio.set_gain(%.2f dB)", self._gain)

    def set_amplitude(self, a: float) -> None:
        self._amp = float(a)
        log.info("  radio.set_amplitude(%.3f)", self._amp)

    def actual_gain(self) -> float:
        return self._gain             # a real SDR quantises; the mock is exact


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("Mock transmitter (NO HARDWARE) — logs the gain/amplitude it would "
               "command, to exercise the power-calibration path. Transmits nothing.")
        .number("-Frequency", "--freq", unit="Hz", min=1e6, max=6e9,
                default=1575.42e6,
                help="Informational carrier for the log lines. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given. Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="on",
                required=False, live=True,
                help="RF output on/off (mock: on-air vs muted). Live. Power changes "
                     "made while OFF are staged and applied when you switch ON.")
        .number("-Heartbeat", "--interval", unit="s", min=0.5, max=60.0, default=2.0,
                help="How often to log an on-air/muted heartbeat line.")
        # RELATIVE power: raw SDR gain (dB). No default → its presence selects
        # relative mode and overrides --power.
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. "
                     "Live.")
    )
    return s


# ── Self-test: exercise the calibration math, no loop, no hardware ─────────────

def _self_test() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    pmap = power_map()
    log.info("power map source : %s", pmap.source)
    log.info("operating label  : %s", pmap.label)
    log.info("gain limits      : %.2f … %.2f dB", pmap.min_gain_db, pmap.max_gain_db)
    if not pmap.has_absolute:
        log.info("power range      : (uncalibrated — no absolute dBm scale; use --gain)")
        log.info("SELF-TEST OK")
        return 0
    log.info("power range      : %.2f … %.2f dBm", pmap.min_power_dbm, pmap.max_power_dbm)
    for req in (pmap.max_power_dbm, pmap.max_power_dbm - 10, pmap.max_power_dbm - 30):
        g = pmap.gain_for_power(req)
        log.info("  --power %+7.2f dBm  →  gain %6.2f dB  →  reads back %+7.2f dBm",
                 req, g, pmap.power_for_gain(g))
    log.info("SELF-TEST OK")
    return 0


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    script = build_script()
    args = script.parse()

    # Power map: the unit's injected calibration curve if present, else baked.
    pmap = power_map()
    amplitude = pmap.amplitude

    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:                          # calibrated: the authored absolute --power
        gain_db = pmap.gain_for_power(args.power)
    else:                                            # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    radio = FakeRadio(args.freq)
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}

    log.info("── mock TX (no hardware) ───────────────────────────────────")
    log.info("  signal id      : %s", CAL_SIGNAL_ID)
    log.info("  carrier (info) : %.3f MHz", args.freq / 1e6)
    if pmap.has_absolute:
        log.info("  power (target) : %g dBm  (%s)", args.power, pmap.label)
    log.info("  → gain         : %.2f dB (max %g), amplitude %g",
             gain_db, pmap.max_gain_db, amplitude)
    log.info("  calibration    : %s", pmap.source)
    if pmap.warning:                       # calibration measured at another amplitude
        log.info("  ⚠ CALIBRATION  : %s", pmap.warning)
    log.info("  RF             : %s", "ON" if state["rf_on"] else "OFF (muted)")
    if gain_cal is not None:
        log.info("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    log.info("────────────────────────────────────────────────────────────")

    # Apply the initial state to the (fake) radio.
    if state["rf_on"]:
        radio.set_amplitude(amplitude)
        radio.set_gain(state["gain"])
    else:
        radio.set_gain(0.0)
        radio.set_amplitude(0.0)

    ctrl = script.live_control(args)

    def apply_change(name, value):
        # Mirrors the real scripts: power/gain edits stage into state["gain"] and only
        # reach the (fake) radio while RF is on; --rf mutes/restores gain + amplitude.
        if name == "power":
            state["gain"] = pmap.gain_for_power(float(value))
            log.info("live: --power %s dBm → gain %.2f dB", value, state["gain"])
            if state["rf_on"]:
                radio.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(radio.actual_gain()), 2))
            else:
                log.info("  (staged — RF is off; applies on next --rf on)")
                ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
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

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    last_beat = 0.0
    while not stop.is_set():
        for change in ctrl.drain():
            apply_change(change.name, change.value)
        now = time.monotonic()
        if now - last_beat >= args.interval:
            last_beat = now
            if state["rf_on"]:
                log.info("[on-air]  %.3f MHz — %s → gain %.2f dB, amp %g",
                         args.freq / 1e6, pmap.label, state["gain"], amplitude)
            else:
                log.info("[muted]   %.3f MHz — staged gain %.2f dB (switch --rf on)",
                         args.freq / 1e6, state["gain"])
        time.sleep(0.1)

    ctrl.close()
    log.info("mock TX stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

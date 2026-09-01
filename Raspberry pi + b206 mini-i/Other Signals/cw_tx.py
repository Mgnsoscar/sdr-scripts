#!/usr/bin/env python3
"""
Pure CW-tone transmitter for GNU Radio + UHD (Ettus B200-mini family).

Emits a single, steady continuous-wave tone at a chosen frequency (GNSS carrier
presets, or any value) — no modulation, no drift. The tone is the USRP LO itself
(a constant baseband driving the DAC), so it is as clean and stable as the radio's
reference. For a tone that ramps between two frequencies over time, use the
companion cw_drift_tx.py.

Because it is a single constant-envelope tone, this is the natural signal to
CALIBRATE a unit with: measure its delivered power against commanded gain across
frequency, and that reference transfers to the other constant-modulus signals.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never radiate over the air.

Level set in dBm (--power) with a live RF on/off (--rf); uncalibrated it runs on a
relative --gain. The tone frequency (--freq) retunes live too.

CLI
───
    cw_tx.py --freq 1575.42e6 --power -30 --rf on      # calibrated dBm at L1
    cw_tx.py --freq 1227.6e6 --gain 60 --rf on         # relative gain at L2
    cw_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# Quiet UHD/GNU Radio BEFORE the libs load (imported lazily inside main()).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane.
# Absent it, the script runs uncalibrated (relative gain only).
CAL_SIGNAL_ID = "cw_tone"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# has a --power scale that MOVES with frequency, so the map is folded at THIS param's value.
CAL_FREQ_PARAM = "freq"


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task parameter:
# the calibration is measured at THIS amplitude, so a unit calibrated at a different
# amplitude no longer matches. calkit detects that at load and runs UNCALIBRATED with a
# loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum.
HW_MAX_GAIN_DB = 89.75

# A pure tone sits at the LO (DC baseband), so it needs only a low sample rate. Fixed
# (not a parameter) — nothing here occupies bandwidth.
SAMP_RATE_HZ = 1.0e6


# ── Power map: the unit's injected calibration curve if present, else uncalibrated ──

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script
    and main share one — and so --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Constants ─────────────────────────────────────────────────────────────────

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


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(center_freq_hz: float, gain_db: float, amplitude: float):
    """A constant baseband (a 0 Hz NCO) mixed up by the USRP LO — i.e. the LO carrier
    itself, a pure CW at center_freq_hz. Imported lazily so the module loads without a
    radio stack for --describe-params."""
    from gnuradio import gr, analog, blocks, uhd

    class CwTx(gr.top_block):
        def __init__(self):
            super().__init__("CW tone TX")
            self.usrp = uhd.usrp_sink(
                "",
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            # 0 Hz baseband tone → the pure LO carrier at center_freq_hz.
            self.src = analog.sig_source_c(SAMP_RATE_HZ, analog.GR_COS_WAVE, 0.0, 1.0, 0)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def set_center_frequency(self, hz: float) -> None:
            self.usrp.set_center_freq(uhd.tune_request(hz), 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

    return CwTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Pure CW-tone transmitter — a single steady continuous-wave tone (no "
               "modulation, no drift). Level is set in dBm via the unit's calibration, or a "
               "relative gain uncalibrated. For a drifting tone use cw_drift_tx.py. Transmit "
               "only into an authorised, shielded setup.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="Tone frequency. Presets are GNSS carriers; any value allowed. --power "
                     "is calibrated here. Live.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the unit's "
                     "calibration when present (e.g. EIRP), else the baked SDR-port scale. "
                     "Ignored if --gain is given (relative wins). Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="off",
                required=False, live=True,
                help="RF output on/off. Starts OFF (muted pre-roll): set the power, then "
                     "switch ON to go on-air. OFF mutes gain AND baseband amplitude; power "
                     "edits made while OFF are staged and applied when you switch ON.")
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power). This is
        # also the calibration knob — set it while measuring output vs gain on an analyser.
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing the "
                     "dBm calibration. When given, overrides --power. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    script = build_script()
    args = script.parse()

    freq = float(args.freq)
    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:                         # calibrated: the authored absolute --power
        # Fold the calibration at the transmit frequency (frequency-dependent chains).
        gain_db = pmap.gain_for_power(args.power, freq=freq)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    tb = _build_top_block(freq, gain_db, amplitude)

    # Track the live transmit frequency and (in absolute mode) the held target power, so a
    # live retune can re-map --power at the new frequency on a frequency-dependent chain.
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    # RF on/off state + the gain RF-on applies. Defaults to --rf off so it starts muted.
    state = {"rf_on": getattr(args, "rf", "off") == "on", "gain": gain_db,
             "freq": freq, "power": _target_power}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── CW tone TX ──────────────────────────────────────────────")
    print(f"  tone           : {freq/1e6:.6f} MHz (pure CW)")
    print(f"  sample rate    : {SAMP_RATE_HZ/1e6:g} MHz")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=freq):.2f} dBm")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:                       # e.g. calibration amplitude != this
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")   # script's fixed amplitude
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted — switch --rf on to unmute)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "freq":
            # A live retune re-tunes the LO and, on a frequency-dependent chain, re-maps the
            # held target power at the new frequency so delivered power stays as requested.
            hz = float(value)
            tb.set_center_frequency(hz)
            state["freq"] = hz
            ctrl.report("freq", hz)
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=hz)
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain(), freq=hz), 2))
                else:
                    ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=hz), 2))
        elif name == "power":
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"]), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
        elif name == "gain":
            state["power"] = None
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("gain", round(tb.actual_gain(), 2))
            else:
                ctrl.report("gain", round(state["gain"], 2))
        elif name == "rf":
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            if on:
                tb.set_amplitude(amplitude)
                tb.set_gain(state["gain"])
            else:
                tb.set_gain(0.0)
                tb.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                apply_change(change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

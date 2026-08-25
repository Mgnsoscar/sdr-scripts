#!/usr/bin/env python3
"""
CW-tone transmitter for GNU Radio + UHD (Ettus B200-mini family), with an optional
slow frequency drift.

Emits a pure continuous-wave tone at a chosen frequency (GNSS carrier presets, or
any value), optionally drifting from a start frequency to an end frequency over a
duration as long as 20 minutes. The tone is a baseband NCO (analog.sig_source_c)
mixed up by the USRP LO, so the drift is smooth and continuous-phase — GNU Radio's
sig_source changes frequency without a phase discontinuity, and there is no file
to precompute however slow the drift.

How the drift works
───────────────────
The LO sits at the drift centre f_c = (start+end)/2, and the baseband tone offset
moves over time so the emitted frequency f(t) ramps from start to end. The drift
runs on its own timeline from the moment the script starts — it is independent of
RF. --rf on/off is a pure mute/unmute and does NOT start, stop, or restart the
sweep; use --restart to re-run the ramp from the start frequency.

    --drift once      : ramp start→end over --duration, then hold at end (default)
    --drift loop      : ramp start→end, jump back to start, repeat
    --drift pingpong  : ramp start→end→start→… (triangle)

The occupied span is |end−start|, carried in baseband, so --samp_rate must exceed
it. A pure tone (no --freq_end) needs only a low rate.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never radiate over the air.

Level set in dBm (--power) with a live RF on/off (--rf); see the USER CALIBRATION
block. Frequencies / duration / sample rate are fixed per run (restart to change).

CLI
───
    cw_tx.py --freq 1575.42e6 --power -30 --rf on             # pure CW at L1
    cw_tx.py --freq 1575.42e6 --freq_end 1575.43e6 --duration 1200 --power -30 --rf on  # 10 kHz / 20 min
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
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only)
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "cw_tone"


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75



# ── Power map: the unit's injected calibration curve if present, else the baked
#    constants above (identical to the old single-anchor slope-1 behaviour) ────────

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script and
    main share one — and so --power's schema bounds match the real operating range
    (calibrated → e.g. EIRP; else the baked SDR-port range)."""
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
SAMPLE_RATES_MHZ = {"1 MHz (narrow drift)": 1.0, "2 MHz (default)": 2.0,
                    "10 MHz": 10.0, "40 MHz": 40.0}
MAX_DURATION_S = 1200.0        # 20 minutes


def drift_freq(elapsed: float, start: float, end: float, duration: float,
               mode: str) -> float:
    """The emitted frequency at `elapsed` seconds into the drift."""
    if duration <= 0 or end == start:
        return end
    u = elapsed / duration
    if mode == "loop":
        frac = u % 1.0
    elif mode == "pingpong":
        tri = u % 2.0
        frac = tri if tri <= 1.0 else 2.0 - tri
    else:  # once
        frac = min(u, 1.0)
    return start + (end - start) * frac


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(center_freq_hz: float, samp_rate_hz: float, tone_hz: float,
                     gain_db: float, amplitude: float, extra_args: str):
    """A baseband NCO (sig_source_c) mixed up by the USRP LO. Imported lazily so
    the module loads without a radio stack for --describe-params."""
    from gnuradio import gr, analog, blocks, uhd

    class CwTx(gr.top_block):
        def __init__(self):
            super().__init__("CW tone TX")
            self.usrp = uhd.usrp_sink(
                extra_args,
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            # Complex exponential at the baseband offset; phase-continuous when
            # set_frequency() is called mid-run (that's what makes the drift smooth).
            self.src = analog.sig_source_c(samp_rate_hz, analog.GR_COS_WAVE,
                                           tone_hz, 1.0, 0)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def set_tone(self, hz: float) -> None:
            self.src.set_frequency(hz)          # continuous-phase frequency change

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

    return CwTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("CW-tone transmitter — a continuous-wave tone, optionally drifting "
               "from a start to an end frequency over up to 20 minutes (baseband "
               "NCO, continuous phase). Transmit only into an authorised, shielded "
               "setup.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True,
                help="Tone frequency (drift START if --freq_end is set). Presets are "
                     "GNSS carriers; any value allowed.")
        .number("-End-frequency", "--freq_end", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=0.0,
                help="Drift END frequency. 0 (default) or equal to --freq = a pure, "
                     "non-drifting CW.")
        .number("-Duration", "--duration", unit="s", min=1.0, max=MAX_DURATION_S,
                default=600.0,
                help="Seconds to drift start→end (up to 1200 = 20 min).")
        .choice("-Drift", "--drift", options=["once", "loop", "pingpong"],
                default="once",
                help="once = ramp then hold; loop = repeat start→end; pingpong = "
                     "start→end→start…")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=0.2, max=61.44,
                presets=SAMPLE_RATES_MHZ, default=2.0, required=True,
                help="Host/DAC sample rate. Must exceed the drift range |end−start|.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given (relative wins). Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="off",
                required=False, live=True,
                help="RF output on/off. Starts OFF (muted pre-roll): set the power, "
                     "then switch ON to go on-air — which is when a drift begins. OFF "
                     "mutes gain AND baseband amplitude; power edits made while OFF "
                     "are staged and applied when you switch ON.")
        .flag("-Restart", "--restart", live=True,
              help="Live trigger (tune-step): restart the drift from the start "
                   "frequency. Fire it to re-run the ramp from the beginning.")
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power).
        # This is also the calibration knob — set it while measuring output vs gain on
        # a spectrum analyser to fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB above.
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. Live.")
    )
    return s


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    script = build_script()
    args = script.parse()

    start = args.freq
    end = args.freq_end if args.freq_end and args.freq_end > 0 else start
    center = 0.5 * (start + end)
    drift_range = abs(end - start)
    samp_rate = args.sample_rate * 1e6
    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present,
    # so you can command a gain directly or measure output power at it.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif power_map().has_absolute:                  # calibrated: the authored absolute --power
        gain_db = power_map().gain_for_power(args.power)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))
    if drift_range >= samp_rate:
        return _fail(f"drift range {drift_range/1e6:g} MHz does not fit the baseband "
                     f"at {args.sample_rate:g} MHz — raise --sample_rate above it")

    drifting = end != start
    tb = _build_top_block(center, samp_rate, start - center, gain_db,
                          amplitude, extra_args="")

    # RF on/off state + the gain RF-on applies. CW defaults to --rf off so it starts
    # muted; RF is a pure mute/unmute and does NOT touch the sweep. Power/gain edits
    # made while OFF are staged for the next ON.
    state = {"rf_on": getattr(args, "rf", "off") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── CW tone TX ──────────────────────────────────────────────")
    if drifting:
        print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
              f"{args.duration:g} s ({args.drift})")
        print(f"  LO centre      : {center/1e6:.6f} MHz  (baseband ±{drift_range/2e6:g} MHz)")
    else:
        print(f"  tone           : {start/1e6:.6f} MHz (pure CW, no drift)")
    print(f"  sample rate    : {args.sample_rate:g} MHz")
    if power_map().has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({power_map().label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:                       # e.g. calibration amplitude != this
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")   # script's fixed amplitude
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted — switch --rf on to unmute)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    if drifting:
        print("  drift runs on its own timeline from start; --rf is a pure mute, "
              "--restart re-runs the sweep.")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)
    # The drift runs on its own timeline from start — independent of RF. RF on/off is
    # a pure mute; only --restart re-runs the sweep from the start frequency.
    t0 = time.monotonic() if drifting else None

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                if change.name == "power":
                    # dBm → gain via the calibration; staged, applied only when RF is on.
                    state["gain"] = pmap.gain_for_power(float(change.value))
                    if state["rf_on"]:
                        tb.set_gain(state["gain"])
                        ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain()), 2))
                    else:
                        ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
                elif change.name == "gain":
                    # Calibration knob: raw TX gain (dB), bypassing the dBm mapping.
                    state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(change.value)))
                    if state["rf_on"]:
                        tb.set_gain(state["gain"])
                        ctrl.report("gain", round(tb.actual_gain(), 2))
                    else:
                        ctrl.report("gain", round(state["gain"], 2))
                elif change.name == "rf":
                    # Pure mute/unmute — does NOT start or restart the sweep (the
                    # drift keeps its own timeline; use --restart to re-run it).
                    on = str(change.value).strip().lower() in ("on", "1", "true", "yes")
                    state["rf_on"] = on
                    if on:
                        tb.set_amplitude(amplitude)
                        tb.set_gain(state["gain"])
                    else:
                        tb.set_gain(0.0)
                        tb.set_amplitude(0.0)
                    ctrl.report("rf", "on" if on else "off")
                elif change.name == "restart" and change.value:
                    t0 = time.monotonic()              # re-run the drift from start
                    tb.set_tone(start - center)
                    ctrl.report("restart", True)
            if t0 is not None and drifting:
                f = drift_freq(time.monotonic() - t0, start, end, args.duration, args.drift)
                tb.set_tone(f - center)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


def _fail(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

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
from paramkit import Script


# ═══════════════════════════════════════════════════════════════════════════════
# USER CALIBRATION — MEASURE THESE ONCE, THEN EDIT THE VALUES BELOW
# ═══════════════════════════════════════════════════════════════════════════════
# You set the transmit level in dBm. That only works if the script knows how the
# SDR's gain maps to real output power, which you establish once with a spectrum
# analyser: leave AMPLITUDE at the value below, run with --power at its maximum
# (that commands GAIN_AT_MAX_DB), measure the actual output power at the SDR RF
# port, and put that number in OUTPUT_POWER_DBM. From that anchor the script maps
# any requested power to a gain (1 dB gain ≈ 1 dB power, across the B200's linear
# range). CABLE_LOSS_DB / AMPLIFIER_GAIN_DB describe the RF chain AFTER the port,
# so the number you dial in is the power delivered at the far end.

OUTPUT_POWER_DBM = -20.0    # max output (dBm) at GAIN_AT_MAX_DB and AMPLITUDE — MEASURE THIS
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands
CABLE_LOSS_DB = 0.0         # cabling insertion loss after the SDR port (positive dB)
AMPLIFIER_GAIN_DB = 0.0     # external amplifier gain after the SDR port (positive dB)

# Fixed baseband digital amplitude (0..1). NOT a user control: OUTPUT_POWER_DBM is
# calibrated at THIS amplitude, so changing it invalidates the dBm↔gain mapping —
# if you change it, re-measure OUTPUT_POWER_DBM at GAIN_AT_MAX_DB.
AMPLITUDE = 0.8

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75

# Derived delivered-power limits (computed — do not edit).
MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) that puts `delivered_dbm` at the far end of the RF chain, clamped
    to [0, GAIN_AT_MAX_DB] so it can never exceed the calibrated maximum."""
    port_dbm = float(delivered_dbm) + CABLE_LOSS_DB - AMPLIFIER_GAIN_DB
    gain = GAIN_AT_MAX_DB + (port_dbm - OUTPUT_POWER_DBM)
    return max(0.0, min(GAIN_AT_MAX_DB, gain))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) for an actual hardware gain — to report what the radio
    really settled on after quantisation."""
    port_dbm = OUTPUT_POWER_DBM - (GAIN_AT_MAX_DB - float(gain_db))
    return port_dbm - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB


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
                min=round(MIN_DELIVERED_DBM, 2), max=round(MAX_DELIVERED_DBM, 2),
                default=round(MAX_DELIVERED_DBM, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .choice("-RF", "--rf", options=["on", "off"], default="off",
                required=False, live=True,
                help="RF output on/off. Starts OFF (muted pre-roll): set the power, "
                     "then switch ON to go on-air — which is when a drift begins. OFF "
                     "mutes gain AND baseband amplitude; power edits made while OFF "
                     "are staged and applied when you switch ON.")
        .flag("-Restart", "--restart", live=True,
              help="Live trigger (tune-step): restart the drift from the start "
                   "frequency. Fire it to re-run the ramp from the beginning.")
    )
    # ── CALIBRATION KNOB (normally commented OUT) ───────────────────────────────
    # Uncomment to expose a raw TX-gain slider (dB) so you can measure output power
    # vs gain on a spectrum analyser and fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB
    # above. While present it OVERRIDES --power (whichever you touch last wins).
    # s = s.number(
    #     "-Cal-gain", "--gain", unit="dB",
    #     min=0, max=HW_MAX_GAIN_DB, default=HW_MAX_GAIN_DB,
    #     required=False, live=True,
    #     help="CALIBRATION ONLY — set SDR TX gain directly, bypassing the dBm "
    #          "mapping. Comment out again for normal dBm operation.")
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
    # A raw calibration gain (the normally-commented --gain knob) overrides the dBm
    # mapping when present, so you can measure output power at a chosen gain.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else gain_for_power(args.power)
    if drift_range >= samp_rate:
        return _fail(f"drift range {drift_range/1e6:g} MHz does not fit the baseband "
                     f"at {args.sample_rate:g} MHz — raise --sample_rate above it")

    drifting = end != start
    tb = _build_top_block(center, samp_rate, start - center, gain_db,
                          AMPLITUDE, extra_args="")

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
    print(f"  power (target) : {args.power:g} dBm delivered "
          f"(cable −{CABLE_LOSS_DB:g} dB, amp +{AMPLIFIER_GAIN_DB:g} dB)")
    print(f"  → gain         : {gain_db:.2f} dB (max {GAIN_AT_MAX_DB:g}), "
          f"amplitude {AMPLITUDE:g}")
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
                    state["gain"] = gain_for_power(float(change.value))
                    if state["rf_on"]:
                        tb.set_gain(state["gain"])
                        ctrl.report("power", round(power_for_gain(tb.actual_gain()), 2))
                    else:
                        ctrl.report("power", round(power_for_gain(state["gain"]), 2))
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
                        tb.set_amplitude(AMPLITUDE)
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

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
begins at ON-AIR (the first moment amplitude > 0), so a pre-roll works: start with
--amplitude 0, then raise it (a live tune) to un-mute and start the ramp. A
non-zero --amplitude starts the drift immediately.

    --drift once      : ramp start→end over --duration, then hold at end (default)
    --drift loop      : ramp start→end, jump back to start, repeat
    --drift pingpong  : ramp start→end→start→… (triangle)

The occupied span is |end−start|, carried in baseband, so --samp_rate must exceed
it. A pure tone (no --freq_end) needs only a low rate.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never radiate over the air.

Live tuning: gain and amplitude (instant). Frequencies / duration / sample rate
are fixed per run (restart to change them).

CLI
───
    cw_tx.py --freq 1575.42e6 --gain 45                       # pure CW at L1
    cw_tx.py --freq 1575.42e6 --freq_end 1575.43e6 --duration 1200   # 10 kHz over 20 min
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
    return (
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
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=45,
                required=True, live=True, help="USRP TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.0,
                required=True, live=True,
                help="Baseband digital amplitude (0..1). The drift begins when "
                     "amplitude first goes >0 (on-air). Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    script = build_script()
    args = script.parse()

    start = args.freq
    end = args.freq_end if args.freq_end and args.freq_end > 0 else start
    center = 0.5 * (start + end)
    drift_range = abs(end - start)
    samp_rate = args.sample_rate * 1e6
    if drift_range >= samp_rate:
        return _fail(f"drift range {drift_range/1e6:g} MHz does not fit the baseband "
                     f"at {args.sample_rate:g} MHz — raise --sample_rate above it")

    drifting = end != start
    tb = _build_top_block(center, samp_rate, start - center, args.gain,
                          args.amplitude, extra_args="")

    print("── CW tone TX ──────────────────────────────────────────────")
    if drifting:
        print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
              f"{args.duration:g} s ({args.drift})")
        print(f"  LO centre      : {center/1e6:.6f} MHz  (baseband ±{drift_range/2e6:g} MHz)")
    else:
        print(f"  tone           : {start/1e6:.6f} MHz (pure CW, no drift)")
    print(f"  sample rate    : {args.sample_rate:g} MHz")
    print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
          f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
    print("  drift begins at on-air (first amplitude > 0).")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)
    t0 = time.monotonic() if args.amplitude > 0 else None

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                if change.name == "amplitude":
                    tb.set_amplitude(change.value)
                    ctrl.report("amplitude", change.value)
                    if change.value > 0 and t0 is None:
                        t0 = time.monotonic()          # drift starts at on-air
                elif change.name == "gain":
                    tb.set_gain(change.value)
                    ctrl.report("gain", tb.actual_gain())
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

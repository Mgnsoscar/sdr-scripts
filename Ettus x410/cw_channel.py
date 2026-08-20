#!/usr/bin/env python3
"""
cw_channel — continuous-wave (CW) tone channel-task for the X410 engine, with an
optional slow frequency drift.

Emits a pure CW tone at a chosen frequency (GNSS carrier presets, or any value),
optionally drifting from a start frequency to an end frequency over a duration as
long as 20 minutes. It uses the engine's generated 'tone' mode — a continuous-
phase CW synthesised on the fly — so the drift is smooth (no phase glitches) and
costs no buffer, however slow it is.

How the drift works
───────────────────
The engine LO sits at the drift centre f_c = (start+end)/2, and the channel-task
moves the baseband tone offset over time so the emitted frequency f(t) ramps from
start to end. The drift begins at ON-AIR (the first moment amplitude > 0), so the
usual pre-roll works: start the task early with --amplitude 0, and a timeline
tune-step both un-mutes it and starts the ramp. A non-zero --amplitude starts the
drift immediately on load.

    --drift once      : ramp start→end over --duration, then hold at end (default)
    --drift loop      : ramp start→end, jump back to start, repeat
    --drift pingpong  : ramp start→end→start→… (triangle)

The occupied span is |end−start|, carried in baseband, so the sample rate must
exceed it — pick --samp_rate ≳ the drift range. A pure tone (no --freq_end, or
equal to --freq) needs only a low rate.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use.

CLI
───
    cw_channel.py --channel 0 --freq 1575.42e6 --gain 45 --amplitude 0   # pure CW at L1
    cw_channel.py --channel 1 --freq 1575.42e6 --freq_end 1575.43e6 --duration 1200
    cw_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import connect_engine
from engine_client import EngineError


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
SAMPLE_RATES_MHZ = {"1.024 MHz (narrow drift)": 1.024, "2.048 MHz (default)": 2.048,
                    "10.24 MHz": 10.24, "61.44 MHz (wide)": 61.44}
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


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("CW tone channel-task — a continuous-wave tone on one X410 engine "
               "channel, optionally drifting from a start to an end frequency over "
               "up to 20 minutes (generated 'tone' mode, continuous phase).")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True,
                help="Tone frequency (drift START if --freq_end is set). Presets are "
                     "GNSS carriers; any value allowed. Fixed per run.")
        .number("-End-frequency", "--freq_end", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=0.0,
                help="Drift END frequency. 0 (default) or equal to --freq = a pure, "
                     "non-drifting CW. Fixed per run.")
        .number("-Duration", "--duration", unit="s", min=1.0, max=MAX_DURATION_S,
                default=600.0, help="Seconds to drift start→end (up to 1200 = "
                     "20 min). Fixed per run.")
        .choice("-Drift", "--drift", options=["once", "loop", "pingpong"],
                default="once",
                help="once = ramp then hold at end; loop = repeat start→end; "
                     "pingpong = start→end→start… Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=0.5, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=2.048, required=True,
                help="Target channel sample rate (negotiated). Must exceed the drift "
                     "range |end−start|. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=45,
                required=True, live=True, help="Channel TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.0,
                required=True, live=True,
                help="Digital amplitude 0..1. Start at 0 for a pre-roll — the drift "
                     "begins when amplitude first goes >0 (on-air). Live.")
        .flag("-Restart", "--restart", live=True,
              help="Live trigger (tune-step): restart the drift from the start "
                   "frequency. Fire it to re-run the ramp from the beginning.")
        .text("-Engine-socket", "--engine_socket", default="/tmp/x410_engine.sock",
              help="Unix socket of the running x410_engine.")
        .text("-Owner", "--owner", default="",
              help="Channel ownership tag (default: auto from channel + PID).")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    script = build_script()
    args = script.parse()
    ch = args.channel
    owner = args.owner or f"ch{ch}-{os.getpid()}"

    start = args.freq
    end = args.freq_end if args.freq_end and args.freq_end > 0 else start
    center = 0.5 * (start + end)
    drift_range = abs(end - start)

    eng = connect_engine(args.engine_socket)
    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)
        if drift_range >= actual_rate:
            raise SystemExit(
                f"drift range {drift_range/1e6:g} MHz does not fit the baseband at "
                f"{actual_rate/1e6:g} MHz — raise --samp_rate above the range")

        drifting = end != start
        print("── CW tone channel-task ────────────────────────────────────")
        print(f"  engine channel : {ch}   owner {owner}")
        if drifting:
            print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
                  f"{args.duration:g} s ({args.drift})")
            print(f"  LO centre      : {center/1e6:.6f} MHz  (baseband ±{drift_range/2e6:g} MHz)")
        else:
            print(f"  tone           : {start/1e6:.6f} MHz (pure CW, no drift)")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        print("  drift begins at on-air (first amplitude > 0).")
        print("────────────────────────────────────────────────────────────")
        sys.stdout.flush()

        # Load the generated tone: LO at the drift centre, baseband offset = start.
        eng.load(ch, owner, {
            "mode": "tone", "freq_hz": center, "gain_db": args.gain,
            "amplitude": args.amplitude, "tone_hz": start - center, "label": "cw"})

        ctrl = script.live_control(args)
        t0 = time.monotonic() if args.amplitude > 0 else None   # on-air reference
        last_tone = None

        stop = _stop_event()
        while not stop.is_set():
            for change in ctrl.drain():
                try:
                    if change.name == "amplitude":
                        eng.set(ch, owner, amplitude=change.value)
                        ctrl.report("amplitude", change.value)
                        if change.value > 0 and t0 is None:
                            t0 = time.monotonic()          # drift starts at on-air
                    elif change.name == "gain":
                        eng.set(ch, owner, gain_db=change.value)
                        ctrl.report("gain", change.value)
                    elif change.name == "restart" and change.value:
                        t0 = time.monotonic()              # re-run the drift from start
                        eng.set(ch, owner, tone_hz=start - center)
                        last_tone = start - center
                        ctrl.report("restart", True)
                except EngineError as exc:
                    print(f"[warn] live {change.name} rejected: {exc}", flush=True)

            if t0 is not None and drifting:
                f = drift_freq(time.monotonic() - t0, start, end, args.duration, args.drift)
                tone_hz = f - center
                if last_tone is None or abs(tone_hz - last_tone) >= 0.5:
                    try:
                        eng.set(ch, owner, tone_hz=tone_hz)
                        last_tone = tone_hz
                    except EngineError:
                        pass
            time.sleep(0.1)
        ctrl.close()
    finally:
        # Best-effort: tolerate a dropped connection so cleanup never raises a
        # secondary BrokenPipeError over the original traceback.
        try:
            eng.release(ch, owner)
        except Exception:
            pass
        try:
            eng.close()
        except Exception:
            pass
    return 0


def _stop_event():
    import threading
    ev = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: ev.set())
    signal.signal(signal.SIGINT, lambda *_: ev.set())
    return ev


if __name__ == "__main__":
    raise SystemExit(main())

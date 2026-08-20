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
The emitted frequency is the engine LO plus a baseband tone offset. The channel-
task moves that offset over time so f(t) ramps from start to end. The drift begins
at ON-AIR (the first moment amplitude > 0), so the usual pre-roll works: start the
task early with --amplitude 0, and a timeline tune-step both un-mutes it and starts
the ramp. A non-zero --amplitude starts the drift immediately on load.

    --drift once      : ramp start→end over --duration, then hold at end (default)
    --drift loop      : ramp start→end, jump back to start, repeat
    --drift pingpong  : ramp start→end→start→… (triangle)

Narrow vs. wide sweeps
──────────────────────
A software CW baseband tone can only occupy ±(sample_rate/2). Two regimes, chosen
automatically from the span |end−start| and the negotiated rate:

  • Narrow (span fits one baseband window): the LO sits fixed at the drift centre
    and the whole sweep is carried in the software baseband tone — perfectly
    continuous, no retunes.

  • Wide (span bigger than a window, e.g. 1600→1545 MHz = 55 MHz): the sweep is
    carried by the hardware DUC/NCO instead. The analog LO is PINNED (a MANUAL-LO
    tune) and the digital NCO moves the carrier across a window of ±lo_span/2 — a
    purely digital frequency change with no analog synth relock, so no settle
    glitch and a flat amplitude across the window. Crucially this avoids the old
    software-baseband-vs-analog-LO race (where the baseband jumped before the LO
    settled, kicking the tone briefly off-frequency). The analog LO only re-anchors
    when the tone would leave the window; if --lo_span ≥ the span it never moves at
    all, so the sweep is fully glitchless. Those rare re-anchors are blanked
    (--hop_blank) to hide their retune settle. A wide sweep is a near-DC signal, so
    a LOW --samp_rate (1–2 MHz) is ideal — it keeps ARM load minimal.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use.

CLI
───
    cw_channel.py --channel 0 --freq 1575.42e6 --gain 45 --amplitude 0   # pure CW at L1
    cw_channel.py --channel 1 --freq 1575.42e6 --freq_end 1575.43e6 --duration 1200
    cw_channel.py --channel 0 --freq 1600e6 --freq_end 1545e6 --duration 900 \
        --samp_rate 2.048 --lo_span 60       # wide 55 MHz sweep, 15 min, NCO (no hops)
    cw_channel.py --self-test        # drift + LO-hop planner math, no engine/hardware
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
                    "5.0 MHz (wide sweep)": 5.0, "10.24 MHz": 10.24}
MAX_DURATION_S = 1200.0        # 20 minutes

# Fraction of Nyquist the baseband tone may use before the LO hops. Keeps the tone
# inside the flat part of the DUC response and clear of the anti-alias rolloff.
SWEEP_MARGIN = 0.7


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


def plan_lo(f_hz: float, lo_hz: float, half_window_hz: float) -> float:
    """The LO/anchor to emit f_hz while keeping the offset (f − LO) within
    ±half_window. Steps the LO by whole windows (2·half_window) as the sweep crosses
    a window edge. Used for the NCO anchor in a wide sweep: within a window the LO is
    fixed and the offset is realised by the digital NCO, so only a whole-window
    re-anchor ever moves the analog LO."""
    step = 2.0 * half_window_hz
    while f_hz - lo_hz > half_window_hz:
        lo_hz += step
    while f_hz - lo_hz < -half_window_hz:
        lo_hz -= step
    return lo_hz


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
                     "non-drifting CW. A span wider than one baseband window is swept "
                     "by hopping the LO automatically (e.g. 1600→1545 MHz). Fixed per "
                     "run.")
        .number("-Duration", "--duration", unit="s", min=1.0, max=MAX_DURATION_S,
                default=600.0, help="Seconds to drift start→end (up to 1200 = "
                     "20 min). Fixed per run.")
        .choice("-Drift", "--drift", options=["once", "loop", "pingpong"],
                default="once",
                help="once = ramp then hold at end; loop = repeat start→end; "
                     "pingpong = start→end→start… Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=0.5, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=2.048, required=True,
                help="Target channel sample rate (negotiated). A narrow sweep is "
                     "carried in baseband so this must exceed |end−start|; a wide "
                     "sweep is carried by the hardware NCO, so a low rate is fine and "
                     "keeps ARM load down. Fixed per run.")
        .number("-LO-span", "--lo_span", unit="MHz", min=1.0, max=200.0, default=40.0,
                help="Wide sweep only: how far the tone may move on the digital NCO "
                     "before the analog LO re-anchors. The sweep is glitchless within "
                     "one span; a span wider than |end−start| means the analog LO "
                     "never moves (fully glitchless). Keep ≤ the channel's NCO/analog "
                     "range. Fixed per run.")
        .number("-Hop-blank", "--hop_blank", unit="ms", min=0.0, max=100.0, default=5.0,
                help="Wide sweep only: mute for this long around each (rare) analog-LO "
                     "re-anchor to hide its retune settle. 0 disables blanking. Live-"
                     "safe. Fixed per run.")
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


# ── Self-test (no engine, no hardware) ──────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok ' if cond else 'FAIL'}· {msg}")

    S, E, D = 1600e6, 1545e6, 900.0     # the user's 1600→1545 MHz over 15 min
    check(abs(drift_freq(0, S, E, D, "once") - S) < 1e-3, "drift starts at start")
    check(abs(drift_freq(D, S, E, D, "once") - E) < 1e-3, "drift 'once' reaches end")
    check(abs(drift_freq(2 * D, S, E, D, "once") - E) < 1e-3, "drift 'once' holds past end")
    check(abs(drift_freq(D / 2, S, E, D, "once") - 0.5 * (S + E)) < 1e-3, "drift is linear")
    check(abs(drift_freq(2 * D, S, E, D, "loop") - S) < 1e-3, "drift 'loop' wraps to start")
    check(abs(drift_freq(D, S, E, D, "pingpong") - E) < 1e-3, "pingpong turns at end")
    check(abs(drift_freq(1.5 * D, S, E, D, "pingpong") - 0.5 * (S + E)) < 1e-3,
          "pingpong returns toward start")
    check(abs(drift_freq(5, S, S, D, "once") - S) < 1e-3, "no --freq_end ⇒ pure CW")

    # NCO-anchor planner over the full 55 MHz sweep: the tone offset stays inside the
    # ±half_window NCO window, the analog LO only ever re-anchors by whole windows,
    # and each re-anchor is exactly one window (so the emitted freq never steps).
    hw = 20e6 / 2.0                      # a 20 MHz lo_span
    anc = plan_lo(S, 0.5 * (S + E), hw)  # window containing start
    steps = 0
    win_ok = step_ok = True
    for k in range(int(D) + 1):
        f = drift_freq(k, S, E, D, "once")
        new_anc = plan_lo(f, anc, hw)
        if new_anc != anc:
            steps += 1
            if abs(abs(new_anc - anc) - 2.0 * hw) > 1e-3:
                step_ok = False
            anc = new_anc
        if abs(f - anc) > hw + 1e-6:
            win_ok = False
    check(win_ok, "NCO offset stays within ±half_window across the whole 55 MHz sweep")
    check(step_ok, "each analog re-anchor is exactly one window (2·half_window)")
    check(steps >= 1, f"a 55 MHz sweep in 20 MHz windows re-anchors the LO ({steps}×)")

    # a span that fits one window never re-anchors — fully glitchless.
    center = 0.5 * (1600e6 + 1580e6)     # a 20 MHz span in a 40 MHz window
    hw40 = 40e6 / 2.0
    anc2 = plan_lo(1600e6, center, hw40)
    reanchors = sum(
        1 for k in range(901)
        if plan_lo(drift_freq(k, 1600e6, 1580e6, 900, "once"), anc2, hw40) != anc2)
    check(reanchors == 0, "a span ≤ lo_span never re-anchors the analog LO (glitchless)")

    print("ALL CW CHECKS PASSED" if ok else "CW SELF-TEST FAILED")
    return 0 if ok else 1


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    ch = args.channel
    owner = args.owner or f"ch{ch}-{os.getpid()}"

    start = args.freq
    end = args.freq_end if args.freq_end and args.freq_end > 0 else start
    drifting = end != start
    span = abs(end - start)

    eng = connect_engine(args.engine_socket)
    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)

        # Pick the sweep mechanism:
        #   pure      — no drift.
        #   baseband  — span fits one baseband window: LO fixed at the drift centre,
        #               the software tone carries the whole sweep (perfectly clean).
        #   dsp       — wider span: the analog LO is pinned and the sweep is carried
        #               by the hardware DUC/NCO, so there is no software-baseband vs.
        #               analog-LO race. The analog LO only re-anchors when the tone
        #               leaves ±lo_span/2; those rare steps are blanked.
        bb_half = SWEEP_MARGIN * actual_rate / 2.0
        lo_span = args.lo_span * 1e6
        dsp_half = lo_span / 2.0
        blank_s = args.hop_blank / 1e3
        if not drifting:
            mode = "pure"
        elif span <= 2.0 * bb_half:
            mode = "baseband"
        else:
            mode = "dsp"

        center = 0.5 * (start + end)
        if mode == "dsp":
            anchor0 = plan_lo(start, center, dsp_half)   # window containing `start`
            lo0, tone0 = start, 0.0                       # HW tune places the carrier
            n_steps = int(span // lo_span)
        else:
            anchor0 = 0.0
            lo0 = center if mode == "baseband" else start
            tone0 = start - lo0
            n_steps = 0

        print("── CW tone channel-task ────────────────────────────────────")
        print(f"  engine channel : {ch}   owner {owner}")
        if mode == "dsp":
            print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
                  f"{args.duration:g} s ({args.drift})")
            print(f"  sweep mode     : NCO — analog LO pinned, {span/1e6:.3f} MHz swept "
                  f"on the digital NCO (±{dsp_half/1e6:g} MHz windows, ~{n_steps} analog "
                  f"re-anchor{'s' if n_steps != 1 else ''}"
                  f"{f', {args.hop_blank:g} ms blank each' if n_steps and blank_s else ''})")
        elif mode == "baseband":
            print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
                  f"{args.duration:g} s ({args.drift})")
            print(f"  sweep mode     : baseband — LO fixed at {lo0/1e6:.6f} MHz "
                  f"(baseband ±{span/2e6:g} MHz, fully continuous)")
        else:
            print(f"  tone           : {start/1e6:.6f} MHz (pure CW, no drift)")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        if mode == "dsp" and n_steps == 0:
            print("  note           : the whole sweep fits one NCO window — the analog "
                  "LO never moves, so the sweep is fully glitchless.")
        print("  drift begins at on-air (first amplitude > 0).")
        print("────────────────────────────────────────────────────────────")
        sys.stdout.flush()

        # Load the tone. In dsp mode the carrier is placed by a MANUAL-LO tune
        # (rf_freq_hz pins the analog LO; the NCO reaches freq_hz); the software tone
        # stays at DC. In baseband/pure mode the LO is fixed and the software tone
        # carries any offset.
        spec = {"mode": "tone", "freq_hz": lo0, "gain_db": args.gain,
                "amplitude": args.amplitude, "tone_hz": tone0, "label": "cw"}
        if mode == "dsp":
            spec["rf_freq_hz"] = anchor0
        eng.load(ch, owner, spec)

        ctrl = script.live_control(args)
        t0 = time.monotonic() if args.amplitude > 0 else None   # on-air reference
        cur_anchor = anchor0
        cur_amp = args.amplitude
        last_tone = None
        last_f = start

        def emit(f: float) -> None:
            """Point the emitted frequency at f. In dsp mode the hardware NCO moves
            (analog LO pinned) and only re-anchors — blanked — when f leaves the
            window; in baseband mode only the software tone moves."""
            nonlocal cur_anchor, last_tone, last_f
            if mode == "dsp":
                new_anchor = plan_lo(f, cur_anchor, dsp_half)
                if new_anchor != cur_anchor:
                    do_blank = cur_amp > 0 and blank_s > 0
                    if do_blank:
                        eng.set(ch, owner, amplitude=0.0)
                    eng.set(ch, owner, freq_hz=f, rf_freq_hz=new_anchor)
                    cur_anchor = new_anchor
                    if do_blank:
                        time.sleep(blank_s)
                        eng.set(ch, owner, amplitude=cur_amp)
                    last_f = f
                elif abs(f - last_f) >= 1.0:
                    eng.set(ch, owner, freq_hz=f, rf_freq_hz=cur_anchor)
                    last_f = f
            else:  # baseband
                tone_hz = f - lo0
                if last_tone is None or abs(tone_hz - last_tone) >= 0.5:
                    eng.set(ch, owner, tone_hz=tone_hz)
                    last_tone = tone_hz

        def reset_to_start() -> None:
            """Re-run the drift from `start` (the live restart trigger)."""
            nonlocal cur_anchor, last_tone, last_f
            if mode == "dsp":
                do_blank = cur_amp > 0 and blank_s > 0
                if do_blank:
                    eng.set(ch, owner, amplitude=0.0)
                cur_anchor = anchor0
                eng.set(ch, owner, freq_hz=start, rf_freq_hz=anchor0)
                if do_blank:
                    time.sleep(blank_s)
                    eng.set(ch, owner, amplitude=cur_amp)
                last_f = start
            else:
                eng.set(ch, owner, tone_hz=tone0)
                last_tone = tone0

        stop = _stop_event()
        while not stop.is_set():
            for change in ctrl.drain():
                try:
                    if change.name == "amplitude":
                        cur_amp = change.value
                        eng.set(ch, owner, amplitude=change.value)
                        ctrl.report("amplitude", change.value)
                        if change.value > 0 and t0 is None:
                            t0 = time.monotonic()          # drift starts at on-air
                    elif change.name == "gain":
                        eng.set(ch, owner, gain_db=change.value)
                        ctrl.report("gain", change.value)
                    elif change.name == "restart" and change.value:
                        t0 = time.monotonic()              # re-run the drift from start
                        reset_to_start()
                        ctrl.report("restart", True)
                except EngineError as exc:
                    print(f"[warn] live {change.name} rejected: {exc}", flush=True)

            if t0 is not None and drifting:
                f = drift_freq(time.monotonic() - t0, start, end, args.duration, args.drift)
                try:
                    emit(f)
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

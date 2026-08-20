#!/usr/bin/env python3
"""
fm_chirp_channel — FM-chirp (swept-tone) channel-task for the X410 engine.

A sine / triangle / sawtooth / square frequency sweep, precomputed once and
replayed on one engine channel. The instantaneous frequency is
f(t) = waveform(t)·(sweep_bw/2), i.e. it sweeps ±sweep_bw/2 around the carrier,
and the modulating waveform repeats at `sweep_rate`.

This is a *channel-task*: the persistent x410_engine owns UHD; this builds the IQ
and drives one channel (mode "expanded"). See gps_prn_channel.py for the lifecycle
and the on-air pre-roll handshake (load muted, a timeline tune-step raises it).

Seamless looping
────────────────
A frequency sweep only loops without a seam if the accumulated phase closes at the
wrap. Two things guarantee it: the buffer holds a whole number of sweep periods
(integer samples/period), and the swept frequency is forced to EXACT zero mean, so
the phase integrated over the buffer returns to its start (verified to ~1e-14 rad;
see --self-test). The buffer is built at the rate the engine negotiates.

Live tuning
───────────
    freq, gain, amplitude → forwarded to the engine          (instant)
    bw, sweep_rate, waveform → rebuild the buffer + reload    (brief swap, then loops)
The reload streams a fresh buffer on the same channel at the same rate; the engine
swaps it under lock, so RF only blips at the swap. Sample rate is fixed per run.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable +
   attenuators) on frequencies you are LICENSED / AUTHORISED to use.

CLI
───
    fm_chirp_channel.py --channel 0 --freq 1.57542e9 --bw 20 --rate 0.2 --waveform Sawtooth
    fm_chirp_channel.py --self-test        # verify seamless phase closure, no hardware
    fm_chirp_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from engine_client import EngineClient, EngineError


# ── Constants ─────────────────────────────────────────────────────────────────

WAVEFORMS = ["Sine", "Triangle", "Sawtooth", "Square"]

# Tile the single sweep period up to at least this many samples so the buffer wraps
# infrequently at high rate (whole periods only → still seamless).
MIN_BUFFER_SAMPS = 1 << 18   # 262144 samples ≈ 2 MB as fc32

# Named GNSS carriers (Hz) for the dropdown; any raw frequency is also accepted.
FREQUENCIES = {
    "GPS L1": 1575.42e6, "GPS L2": 1227.60e6, "GPS L5": 1176.45e6,
    "Galileo E1": 1575.42e6, "Galileo E5a": 1176.45e6, "Galileo E5b": 1207.14e6,
    "Galileo E5": 1191.795e6, "Galileo E6": 1278.75e6,
    "BeiDou B1I": 1561.098e6, "BeiDou B1C": 1575.42e6, "BeiDou B2a": 1176.45e6,
    "BeiDou B2b": 1207.14e6, "BeiDou B2": 1191.795e6, "BeiDou B3": 1268.52e6,
    "GLONASS L1": 1602.0e6, "GLONASS L2": 1246.0e6, "GLONASS L3": 1202.025e6,
    "Iridium": 1621.25e6,
}

# Target sample rates (negotiated to the nearest engine-clock divisor).
SAMPLE_RATES_MHZ = {
    "10.24 MHz": 10.24, "20.48 MHz": 20.48,
    "40.96 MHz (default)": 40.96, "61.44 MHz (max)": 61.44,
}


# ── Modulating waveform + seamless chirp buffer ───────────────────────────────

def _mod_waveform(kind: str, p):
    """Value in [-1, 1] for phase p in [0, 1). Zero-mean over a period. p is a NumPy
    array; the result modulates the instantaneous frequency."""
    import numpy as np
    if kind == "Sine":
        return np.sin(2 * np.pi * p)
    if kind == "Triangle":
        return 1.0 - 2.0 * np.abs(2.0 * p - 1.0)          # -1 → +1 → -1
    if kind == "Sawtooth":
        return 2.0 * p - 1.0                               # -1 → +1 (reset)
    if kind == "Square":
        return np.where(np.sin(2 * np.pi * p) >= 0, 1.0, -1.0)
    raise ValueError(f"unknown waveform {kind!r}")


def build_chirp_buffer(waveform: str, sweep_bw_hz: float, sweep_rate_hz: float,
                       samp_rate_hz: float):
    """Build a complex64 baseband buffer holding a whole number of sweep periods
    that loops with no seam. Unit magnitude (amplitude is applied by the engine).
    Returns (iq, samples_per_period, reps, actual_sweep_rate_hz)."""
    import numpy as np

    n_per = max(2, int(round(samp_rate_hz / sweep_rate_hz)))
    actual_sweep_rate = samp_rate_hz / n_per       # rate quantised to the grid

    p = np.arange(n_per, dtype=np.float64) / n_per
    freq = _mod_waveform(waveform, p) * (sweep_bw_hz / 2.0)
    freq = freq - freq.mean()                      # exact zero-mean → phase closes
    phase = (2.0 * np.pi / samp_rate_hz) * np.cumsum(freq)
    period = np.exp(1j * phase).astype(np.complex64)

    reps = max(1, -(-MIN_BUFFER_SAMPS // n_per))   # ceil division
    iq = np.tile(period, reps)
    return iq, n_per, reps, actual_sweep_rate


# ── Self-test: seamless phase closure, no NumPy / no hardware ─────────────────

def _self_test() -> int:
    import cmath
    import math

    def wave(kind, p):
        if kind == "Sine":     return math.sin(2 * math.pi * p)
        if kind == "Triangle": return 1.0 - 2.0 * abs(2.0 * p - 1.0)
        if kind == "Sawtooth": return 2.0 * p - 1.0
        if kind == "Square":   return 1.0 if math.sin(2 * math.pi * p) >= 0 else -1.0
        raise ValueError(kind)

    fs, bw, rate = 40.96e6, 20e6, 200e3
    ok = True
    for kind in WAVEFORMS:
        n = max(2, round(fs / rate))
        freq = [wave(kind, i / n) * (bw / 2.0) for i in range(n)]
        m = sum(freq) / n
        freq = [f - m for f in freq]
        acc, phase = 0.0, []
        for f in freq:
            acc += 2 * math.pi / fs * f
            phase.append(acc)
        iq = [cmath.exp(1j * ph) for ph in phase]
        expected = 2 * math.pi / fs * freq[0]
        measured = cmath.phase(iq[0] / iq[-1])
        seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
        good = seam_err < 1e-9
        ok = ok and good
        print(f"{kind:9s}: n={n} seam_err={seam_err:.2e} rad [{'OK' if good else 'FAIL'}]")
    print("SEAMLESS — ALL WAVEFORMS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("FM-chirp (swept-tone) channel-task — plays a sine/triangle/sawtooth/"
               "square frequency sweep on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Sweep-BW", "--bw", unit="MHz", min=0.001, max=60.0, default=20.0,
                required=True, live=True,
                help="Peak-to-peak sweep width; f sweeps ±bw/2. Live (regenerates).")
        .number("-Sweep-rate", "--rate", unit="MHz", min=0.0001, max=5.0, default=0.2,
                required=True, live=True,
                help="How fast the sweep repeats. Live (regenerates).")
        .choice("-Waveform", "--waveform", options=WAVEFORMS, default="Sawtooth",
                live=True, help="Sweep shape. Live (regenerates).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=1.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate; the engine negotiates the nearest "
                     "supported rate. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=50,
                required=True, live=True, help="Channel TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.0,
                required=True, live=True,
                help="Digital amplitude 0..1. Start at 0 for a pre-roll and raise "
                     "on-air via a tune-step; or set >0 to transmit on load. Live.")
        .text("-Engine-socket", "--engine_socket", default="/tmp/x410_engine.sock",
              help="Unix socket of the running x410_engine.")
        .text("-Owner", "--owner", default="",
              help="Channel ownership tag (default: auto from channel + PID).")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def _write_shm(iq) -> str:
    import tempfile
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(prefix="fm_chirp_", suffix=".fc32", dir=shm)
    os.close(fd)
    iq.tofile(path)
    return path


def _connect_engine(socket_path: str, attempts: int = 20) -> EngineClient:
    last = None
    for _ in range(attempts):
        try:
            return EngineClient(socket_path).connect()
        except OSError as exc:
            last = exc
            time.sleep(0.25)
    raise SystemExit(f"could not reach engine at {socket_path}: {last}")


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    ch = args.channel
    owner = args.owner or f"ch{ch}-{os.getpid()}"

    eng = _connect_engine(args.engine_socket)
    # Current sweep shape (the regeneration-requiring params), mutated by live changes.
    shape = {"waveform": args.waveform, "bw_hz": args.bw * 1e6, "rate_hz": args.rate * 1e6}

    def build_and_load(rate_hz: float, amplitude: float, gain: float, freq: float):
        iq, n_per, reps, actual_sweep = build_chirp_buffer(
            shape["waveform"], shape["bw_hz"], shape["rate_hz"], rate_hz)
        path = _write_shm(iq)
        try:
            eng.load(ch, owner, {
                "mode": "expanded", "freq_hz": freq, "gain_db": gain,
                "amplitude": amplitude, "iq_file": path, "label": "fm_chirp"})
        finally:
            try:
                os.unlink(path)      # engine copied it into RAM at load
            except OSError:
                pass
        return n_per, reps, actual_sweep

    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)
        n_per, reps, actual_sweep = build_and_load(
            actual_rate, args.amplitude, args.gain, args.freq)

        print("── FM chirp channel-task ───────────────────────────────────")
        print(f"  engine channel : {ch}   owner {owner}")
        print(f"  carrier        : {args.freq/1e6:.3f} MHz")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        print(f"  waveform       : {args.waveform}")
        print(f"  sweep bw       : {args.bw:g} MHz (±{args.bw/2:g} MHz)")
        print(f"  sweep rate     : requested {args.rate*1e3:g} kHz, "
              f"got {actual_sweep/1e3:.3f} kHz ({n_per} samples/period × {reps} reps)")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        print("────────────────────────────────────────────────────────────")
        sys.stdout.flush()

        ctrl = script.live_control(args)
        # Track the live gain so a shape-reload keeps the current gain/amp/freq.
        state = {"gain": args.gain, "amp": args.amplitude, "freq": args.freq}

        def apply_batch(changes):
            """Apply a drain batch: instant params (freq/gain/amplitude) forward
            straight through; shape params (bw/rate/waveform) are coalesced into a
            SINGLE rebuild+reload so a simultaneous change is one RF blip, not many."""
            shape_dirty = False
            for change in changes:
                name, value = change.name, change.value
                if name == "amplitude":
                    state["amp"] = value
                    eng.set(ch, owner, amplitude=value); ctrl.report("amplitude", value)
                elif name == "gain":
                    state["gain"] = value
                    eng.set(ch, owner, gain_db=value); ctrl.report("gain", value)
                elif name == "freq":
                    state["freq"] = value
                    eng.set(ch, owner, freq_hz=value); ctrl.report("freq", value)
                elif name == "bw":
                    shape["bw_hz"] = value * 1e6; shape_dirty = True
                elif name == "rate":
                    shape["rate_hz"] = value * 1e6; shape_dirty = True
                elif name == "waveform":
                    shape["waveform"] = value; shape_dirty = True
            if shape_dirty:
                _, _, actual = build_and_load(actual_rate, state["amp"],
                                              state["gain"], state["freq"])
                # Report back each shape param the host may be waiting on.
                for change in changes:
                    if change.name == "rate":
                        ctrl.report("rate", actual / 1e6)
                    elif change.name in ("bw", "waveform"):
                        ctrl.report(change.name, change.value)

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        while not stop.is_set():
            changes = ctrl.drain()
            if changes:
                try:
                    apply_batch(changes)
                except EngineError as exc:
                    print(f"[warn] live change rejected: {exc}", flush=True)
            time.sleep(0.1)
        ctrl.close()
    finally:
        try:
            eng.release(ch, owner)
        except EngineError:
            pass
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

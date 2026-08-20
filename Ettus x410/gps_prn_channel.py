#!/usr/bin/env python3
"""
gps_prn_channel — GPS C/A Gold-code PRN channel-task for the X410 engine.

Plays a GPS C/A Gold code on one engine channel, at a selectable chip rate and
carrier — which covers three signals from one script (create a task per config):

  • GPS L1 C/A   : 1.023 Mcps @ 1575.42 MHz  (~2 MHz null-to-null)
  • GPS L1 P(Y)  : 10.23 Mcps @ 1575.42 MHz  (~20 MHz — the C/A code clocked 10×,
  • GPS L2 P(Y)  : 10.23 Mcps @ 1227.60 MHz   an unclassified P(Y) surrogate: the
                                               real precision code is encrypted,
                                               so this matches the RF/spectral
                                               shape with a C/A Gold code, like
                                               the M-code surrogate script)

This is a *channel-task*: it does NOT own the radio. The persistent x410_engine
owns UHD (all four channels); this task builds one signal's IQ, hands it to the
engine over a socket to play on one channel, and forwards live tune changes. Many
of these run at once — one per channel — while only the engine touches UHD.

What it does
────────────
  1. connect to the engine and `acquire` its channel,
  2. `configure` the channel's sample rate — the engine returns the ACTUAL rate
     UHD locked to (a divisor of the fixed master clock),
  3. build a seamless-looping C/A Gold-code buffer at that exact rate, into
     /dev/shm, and `load` it (mode "expanded") — starting MUTED (amplitude 0),
  4. forward live amplitude/gain/freq changes to the engine (paramkit.live), so
     a timeline tune-step raises it on-air, and
  5. `release` the channel on stop (SIGTERM/Ctrl-C), always.

The engine copies the IQ into RAM at load, so the /dev/shm file is deleted right
after — no lingering buffer.

At 10.23 Mcps the ~20 MHz main lobe needs a wide sample rate — use 40.96 MHz or
higher (61.44 MHz is cleanest); the 20 MHz option only just spans the main lobe.

On-air handshake (pre-roll)
───────────────────────────
Start this task ~10 s before on-air with `--amplitude 0`; it streams silence
(the engine feeds zeros, so the channel stays fed and glitch-free). At the on-air
instant a timeline tune-step raises `amplitude` (and/or `gain`) live. Set a
non-zero `--amplitude` instead to transmit immediately on load.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) and L2 (1227.60 MHz) are live GNSS bands.
   Transmit ONLY into a shielded/conducted setup (cable + attenuators) you are
   LICENSED / AUTHORISED to use. Radiating a PRN over the air can jam or spoof
   real receivers.

CLI
───
    gps_prn_channel.py --channel 0 --prn 5 --code_rate 1.023 --freq 1.57542e9  # L1 C/A
    gps_prn_channel.py --channel 1 --prn 5 --code_rate 10.23 --freq 1.57542e9 --samp_rate 61.38  # L1 P(Y)
    gps_prn_channel.py --channel 2 --prn 5 --code_rate 10.23 --freq 1.2276e9 --samp_rate 61.38   # L2 P(Y)
    gps_prn_channel.py --self-test        # Gold code + negotiation fidelity, no engine
    gps_prn_channel.py --describe-params  # paramkit JSON schema for the GUI
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

L1_HZ = 1575.42e6
L2_HZ = 1227.60e6

CODE_LEN = 1023                 # chips in a GPS C/A Gold code period

# Selectable spreading-code rates. 1.023 Mcps is the true C/A rate (~2 MHz
# null-to-null); 10.23 Mcps is the same Gold code clocked 10× (~20 MHz) — the
# unclassified P(Y) surrogate.
CODE_RATES_MCPS = {
    "C/A — 1.023 Mcps (~2 MHz BW)":          1.023,
    "P(Y) surrogate — 10.23 Mcps (~20 MHz)": 10.23,
}

FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": L1_HZ,
    "GPS L2 (1227.60 MHz)": L2_HZ,
}

# Target sample rates the operator can pick; the engine negotiates the nearest
# rate its master clock actually supports and the buffer is built for that.
SAMPLE_RATES_MHZ = {
    "4.092 MHz (min — main lobe + first skirt)": 4.092,
    "20.46 MHz (default — faithful C/A)":         20.46,
    "61.38 MHz (max — widest skirts)":            61.38,
}

# GPS ICD-200 Table 3-Ia: G2 code-phase tap pairs (1-indexed) per satellite PRN.
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}

# ICD reference: first 10 chips of each PRN's C/A code, as an octal integer
# (used only by --self-test to prove the generator matches the standard).
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── GPS C/A Gold-code generation (pure Python, no NumPy) ───────────────────────

def ca_code(prn: int):
    """Return the 1023-chip GPS C/A Gold code for a PRN (1..32) as a list of 0/1.

    Two 10-stage LFSRs (both seeded all-ones):
      G1: x^10 + x^3 + 1
      G2: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
    The chip is G1's output XOR the sum of two PRN-specific G2 taps.
    """
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1 to 32, got {prn}")
    g1 = [1] * 10
    g2 = [1] * 10
    ta, tb = G2_TAPS[prn]
    out = []
    for _ in range(CODE_LEN):
        out.append(g1[9] ^ g2[ta - 1] ^ g2[tb - 1])
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return out


# ── Baseband buffer (one seamless-looping stretch of code periods) ─────────────

def build_iq_buffer(prn: int, chip_rate_hz: float, samp_rate_hz: float):
    """Build a complex64 baseband buffer holding a whole number of C/A code periods
    that is also an exact integer number of samples, so it loops with no seam.
    BPSK: I = ±1 chip, Q = 0. Unit magnitude (amplitude is applied by the engine).

    Returns (iq: np.ndarray[complex64], n_samples: int, n_periods: int).
    """
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(chip_rate_hz))
    spp = Fraction(sr * CODE_LEN, cr)      # samples per code period, exact
    n_periods = spp.denominator            # tile this many to reach integer samples
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code             # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr // sr) % CODE_LEN   # exact zero-order-hold chip mapping
    iq = bipolar[chip_idx].astype(np.complex64)   # Q stays 0 (real BPSK)
    return iq, n_samples, n_periods


# ── Self-test (Gold-code generator vs ICD, no engine / no hardware) ────────────

def _self_test() -> int:
    ok = True
    for prn in range(1, 33):
        code = ca_code(prn)
        first10 = 0
        for b in code[:10]:
            first10 = (first10 << 1) | b
        good = (len(code) == CODE_LEN and first10 == _FIRST10_OCTAL[prn]
                and sum(code) == 512)
        ok = ok and good
        print(f"PRN {prn:2d}: first10={first10:#06o} "
              f"expect={_FIRST10_OCTAL[prn]:#06o} ones={sum(code)} "
              f"[{'OK' if good else 'FAIL'}]")
    # Buffer builder loops seamlessly at a stock-clock rate (skip if no NumPy).
    try:
        import numpy as np
        iq, n, periods = build_iq_buffer(1, 1.023e6, 20.48e6)  # 245.76/12
        good = (iq.dtype == np.complex64 and n == 20480 and periods == 1
                and np.all(np.abs(iq) == 1.0))
        ok = ok and good
        print(f"buffer: n={n} periods={periods} unit-mag={bool(np.all(np.abs(iq)==1.0))} "
              f"[{'OK' if good else 'FAIL'}]")

        # Modulation fidelity: the engine negotiates 20.46→20.48 MHz (non-integer
        # samples/chip); prove that still acquires as well as the integer-rate ideal.
        from gnss_acq import check_negotiation_fidelity, cross_isolation_db
        ok = check_negotiation_fidelity(
            lambda r: build_iq_buffer(5, 1.023e6, r)[0],
            chip_rate_hz=1.023e6, ideal_rate_hz=20.46e6, negotiated_rate_hz=20.48e6,
            label="L1 C/A") and ok
        iso = cross_isolation_db(build_iq_buffer(5, 1.023e6, 20.48e6)[0],
                                 build_iq_buffer(7, 1.023e6, 20.48e6)[0])
        good_iso = iso < -18.0
        ok = ok and good_iso
        print(f"cross-PRN isolation (5 vs 7): {iso:.2f} dB [{'OK' if good_iso else 'FAIL'}]")

        # P(Y) surrogate: same Gold code at 10.23 Mcps (~20 MHz), wide sample rate.
        # Engine negotiates 61.38→61.44 MHz; prove it still acquires cleanly.
        ok = check_negotiation_fidelity(
            lambda r: build_iq_buffer(5, 10.23e6, r)[0],
            chip_rate_hz=10.23e6, ideal_rate_hz=61.38e6, negotiated_rate_hz=61.44e6,
            label="P(Y) surrogate", min_db=20.0) and ok
    except ImportError:
        print("buffer/fidelity: skipped (no NumPy here)")
    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GPS C/A Gold-code PRN channel-task — plays a C/A Gold code on one "
               "X410 engine channel at a selectable chip rate/carrier: L1 C/A "
               "(1.023 Mcps), or the P(Y) surrogate (10.23 Mcps) on L1 or L2.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="GPS satellite PRN / Gold-code index (1..32). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L1_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Code-rate", "--code_rate", unit="Mcps", min=0.1, max=20.0,
                presets=CODE_RATES_MCPS, default=1.023, required=True,
                help="Spreading-code chip rate. Fixed per run (sets the bandwidth).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=1.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=20.46, required=True,
                help="Target channel sample rate; the engine negotiates the nearest "
                     "rate its master clock supports. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=45,
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
    """Write the IQ buffer to a unique /dev/shm file (falls back to a tempdir)."""
    import tempfile
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(prefix="gps_l1ca_", suffix=".fc32", dir=shm)
    os.close(fd)
    iq.tofile(path)
    return path


def _connect_engine(socket_path: str, attempts: int = 20) -> EngineClient:
    """Connect to the engine, retrying briefly (the engine may still be coming up
    when the agent launches a channel-task alongside it)."""
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
    chip_rate_hz = args.code_rate * 1e6

    eng = _connect_engine(args.engine_socket)
    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)

        iq, n_samples, n_periods = build_iq_buffer(args.prn, chip_rate_hz, actual_rate)
        iq_path = _write_shm(iq)

        # Name the signal by chip rate: C/A (1.023) vs the P(Y) surrogate (10.23).
        kind = "c/a" if abs(args.code_rate - 1.023) < 0.05 else "p(y)"

        print("── GPS PRN channel-task ────────────────────────────────────")
        print(f"  engine channel : {ch}   owner {owner}")
        print(f"  PRN            : {args.prn}   signal {kind.upper()}")
        print(f"  carrier        : {args.freq/1e6:.3f} MHz")
        print(f"  code rate      : {args.code_rate:g} Mcps")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        print(f"  buffer         : {n_samples} samples ({n_periods} code period(s))")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        print("────────────────────────────────────────────────────────────")
        sys.stdout.flush()

        try:
            eng.load(ch, owner, {
                "mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
                "amplitude": args.amplitude, "iq_file": iq_path,
                "label": f"gps_{kind} prn{args.prn}"})
        finally:
            # The engine copied the IQ into RAM at load — the file is done with.
            try:
                os.unlink(iq_path)
            except OSError:
                pass

        # ── live control: forward paramkit changes to the engine ──────────────
        ctrl = script.live_control(args)

        def apply_change(name, value):
            if name == "amplitude":
                eng.set(ch, owner, amplitude=value); ctrl.report("amplitude", value)
            elif name == "gain":
                eng.set(ch, owner, gain_db=value); ctrl.report("gain", value)
            elif name == "freq":
                eng.set(ch, owner, freq_hz=value); ctrl.report("freq", value)

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        while not stop.is_set():
            for change in ctrl.drain():
                try:
                    apply_change(change.name, change.value)
                except EngineError as exc:
                    print(f"[warn] live {change.name}={change.value} rejected: {exc}",
                          flush=True)
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

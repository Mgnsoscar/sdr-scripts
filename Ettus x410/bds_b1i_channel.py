#!/usr/bin/env python3
"""
bds_b1i_channel — BeiDou B1I channel-task for the X410 engine (expanded mode).

Bit-exact BeiDou **B1I** signal (1561.098 MHz): BPSK-R(2) — a 2.046 Mcps,
2046-chip ranging code, 1 ms period (~4 MHz wide). One seamless 1 ms loop.

Code (BDS-SIS-ICD-B1I §4.3): a balanced Gold code (truncated by its last chip →
2046) from two 11-bit LFSRs —
  G1: 1+X+X^7+X^8+X^9+X^10+X^11, G2: 1+X+X^2+X^3+X^4+X^5+X^8+X^9+X^11,
both seeded 01010101010; the SV is selected by a per-PRN G2 phase-tap set.
Validated in --self-test (G1 period 2047 + ICD first-24 check values).

No navigation data / 1 kHz NH secondary (those ride on the data). See
gps_prn_channel.py for the channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: B1 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    bds_b1i_channel.py --channel 0 --prn 6 --gain 55 --amplitude 0
    bds_b1i_channel.py --self-test        # codes vs ICD + fidelity, no engine
    bds_b1i_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

B1I_HZ = 1561.098e6
CHIP_RATE_HZ = 2_046_000
CODE_LEN = 2046
G_INIT = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)   # 01010101010, both registers

FREQUENCIES = {"BeiDou B1I (1561.098 MHz)": B1I_HZ}
SAMPLE_RATES_MHZ = {"8.192 MHz (default)": 8.192, "16.384 MHz": 16.384,
                    "20.48 MHz": 20.48}

# Per-SV G2 phase selection (XORed G2 stages, 1-indexed), BDS-SIS-ICD-B1I Table 4-1.
G2_TAPS = (
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 8), (1, 9), (1, 10), (1, 11), (2, 7), (3, 4),
    (3, 5), (3, 6), (3, 8), (3, 9), (3, 10), (3, 11), (4, 5), (4, 6), (4, 8), (4, 9),
    (4, 10), (4, 11), (5, 6), (5, 8), (5, 9), (5, 10), (5, 11), (6, 8), (6, 9), (6, 10),
    (6, 11), (8, 9), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11),
    (1, 2, 7), (1, 3, 4), (1, 3, 6), (1, 3, 8), (1, 3, 10), (1, 3, 11), (1, 4, 5), (1, 4, 9),
    (1, 5, 6), (1, 5, 8), (1, 5, 10), (1, 5, 11), (1, 6, 9), (1, 8, 9), (1, 9, 10), (1, 9, 11),
    (2, 3, 7), (2, 5, 7), (2, 7, 9), (3, 4, 5), (3, 4, 9), (3, 5, 6), (3, 5, 8), (3, 5, 10),
    (3, 5, 11), (3, 6, 9),
)


# ── B1I ranging code (bit-exact, BDS-SIS-ICD-B1I §4.3) ─────────────────────────

def _g1_step(r):
    return [r[0] ^ r[6] ^ r[7] ^ r[8] ^ r[9] ^ r[10]] + r[0:10]        # taps 1,7,8,9,10,11


def _g2_step(r):
    return [r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[7] ^ r[8] ^ r[10]] + r[0:10]  # 1,2,3,4,5,8,9,11


def b1i_code(prn: int):
    """The 2046-chip B1I ranging code (0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    taps = G2_TAPS[prn - 1]
    g1 = list(G_INIT)
    g2 = list(G_INIT)
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        g2_out = 0
        for t in taps:
            g2_out ^= g2[t - 1]
        out[i] = g1[10] ^ g2_out
        g1 = _g1_step(g1)
        g2 = _g2_step(g2)
    return out


def build_b1i_buffer(prn: int, samp_rate_hz: float):
    """Complex64 B1I buffer over one 1 ms code period (seamless). Real BPSK, Q=0."""
    import numpy as np
    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b1i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    seen, r = {}, list(G_INIT)
    per = None
    for i in range(3000):
        t = tuple(r)
        if t in seen:
            per = i - seen[t]
            break
        seen[t] = i
        r = _g1_step(r)
    ok = ok and per == 2047
    print(f"G1 period={per} (expect 2047) [{'OK' if per==2047 else 'FAIL'}]")

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o31333315, 2: 0o44461070, 6: 0o32442011, 38: 0o67733254, 63: 0o74366441}
    for prn, want in checks.items():
        c = b1i_code(prn)
        good = o24(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B1I PRN{prn:2d}: first24={oct(o24(c))} expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b1i_code(p)) for p in range(1, 11)}) == 10
    ok = ok and distinct
    print(f"PRN 1..10 distinct: {distinct} [{'OK' if distinct else 'FAIL'}]")

    try:
        from gnss_acq import check_negotiation_fidelity, cross_isolation_db
        ok = check_negotiation_fidelity(
            lambda r: build_b1i_buffer(6, r)[0], chip_rate_hz=CHIP_RATE_HZ,
            ideal_rate_hz=8.184e6, negotiated_rate_hz=8.192e6, label="B1I",
            min_db=18.0) and ok
        iso = cross_isolation_db(build_b1i_buffer(6, 8.192e6)[0],
                                 build_b1i_buffer(7, 8.192e6)[0])
        good_iso = iso < -18.0
        ok = ok and good_iso
        print(f"cross-PRN isolation (6 vs 7): {iso:.2f} dB [{'OK' if good_iso else 'FAIL'}]")
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BeiDou B1I channel-task — bit-exact BPSK-R(2) ranging code "
               "(BDS-SIS-ICD-B1I) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=6, required=True,
                 help="BeiDou B1I PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B1I_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=4.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=8.192, required=True,
                help="Target channel sample rate (negotiated). B1I is ~4 MHz wide; "
                     "~8 MHz is plenty. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=55,
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


def build(args, rate_hz):
    iq, n_samples = build_b1i_buffer(args.prn, rate_hz)
    path = write_shm(iq, "bds_b1i")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path, "label": f"bds_b1i prn{args.prn}"}
    return spec, [path], [f"PRN            : {args.prn}",
                          f"buffer         : {n_samples} samples (1 ms code period)"]


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="BeiDou B1I channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

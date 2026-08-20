#!/usr/bin/env python3
"""
bds_b3i_channel — BeiDou B3I channel-task for the X410 engine (expanded mode).

Bit-exact BeiDou **B3I** signal (1268.52 MHz): BPSK-R(10) — a 10.23 Mcps,
10230-chip ranging code, 1 ms period (~20.46 MHz wide). One seamless 1 ms loop.

Code (BDS-SIS-ICD-B3I §4.3): modulo-2 sum of two 13-bit LFSRs —
  G1: X^13+X^4+X^3+X+1 (all-ones init, short-cycled to 8190 when it hits
      1111111111100), G2: X^13+X^12+X^10+X^9+X^7+X^6+X^5+X+1 (period 8191,
      per-SV initial phase). Validated in --self-test (periods + ICD check values).

No navigation data or the 1 kHz NH secondary (those ride on the data). See
gps_prn_channel.py for the channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: B3 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    bds_b3i_channel.py --channel 0 --prn 6 --gain 55 --amplitude 0
    bds_b3i_channel.py --self-test        # codes vs ICD + fidelity, no engine
    bds_b3i_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

B3_HZ = 1268.52e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
G1_RESET = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0)   # short-cycle state

FREQUENCIES = {"BeiDou B3I (1268.52 MHz)": B3_HZ}
SAMPLE_RATES_MHZ = {"20.48 MHz (min)": 20.48, "40.96 MHz (default)": 40.96,
                    "61.44 MHz (max)": 61.44}

# Per-SV G2 initial phase (stage1..13), BDS-SIS-ICD-B3I Table 4-1.
G2_INIT = (
    "1010111111111", "1111000101011", "1011110001010", "1111111111011",
    "1100100011111", "1001001100100", "1111111010010", "1110111111101",
    "1010000000010", "0010000011011", "1110101110000", "0010110011110",
    "0110010010101", "0111000100110", "1000110001001", "1110001111100",
    "0010011000101", "0000011101100", "1000101010111", "0001011011110",
    "0010000101101", "0010110001010", "0001011001111", "0011001100010",
    "0011101001000", "0100100101001", "1011011010011", "1010111100010",
    "0001011110101", "0111111111111", "0110110001111", "1010110001001",
    "1001010101011", "1100110100101", "1101001011101", "1111101110100",
    "0010101100111", "1110100010000", "1101110010000", "1101011001110",
    "1000000110100", "0101111011001", "0110110111100", "1101001110001",
    "0011100100010", "0101011000101", "1001111100110", "1111101001000",
    "0000101001001", "1000010101100", "1111001001100", "0100110001111",
    "0000000011000", "1000000000100", "0011010100110", "1011001000110",
    "0111001111000", "0010111001010", "1100111110110", "1001001000101",
    "0111000100000", "0011001000010", "0010001001110",
)


# ── B3I ranging code (bit-exact, BDS-SIS-ICD-B3I §4.3) ─────────────────────────

def _g1_step(r):
    return [r[0] ^ r[2] ^ r[3] ^ r[12]] + r[0:12]              # taps 1,3,4,13


def _g2_step(r):
    return [r[0] ^ r[4] ^ r[5] ^ r[6] ^ r[8] ^ r[9] ^ r[11] ^ r[12]] + r[0:12]  # 1,5,6,7,9,10,12,13


def b3i_code(prn: int):
    """The 10230-chip B3I ranging code (0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    g1 = [1] * 13
    g2 = [int(c) for c in G2_INIT[prn - 1]]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        out[i] = g1[12] ^ g2[12]
        g1 = [1] * 13 if tuple(g1) == G1_RESET else _g1_step(g1)
        g2 = _g2_step(g2)
    return out


def build_b3i_buffer(prn: int, samp_rate_hz: float):
    """Complex64 B3I buffer over one 1 ms code period (seamless). Real BPSK, Q=0."""
    import numpy as np
    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b3i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def period(step, init, reset=None):
        seen, r = {}, list(init)
        for i in range(9000):
            t = tuple(r)
            if t in seen:
                return i - seen[t]
            seen[t] = i
            r = [1] * 13 if (reset and t == reset) else step(r)
        return None

    g1p = period(_g1_step, [1]*13, G1_RESET)
    g2p = period(_g2_step, [1]*13)
    good = g1p == 8190 and g2p == 8191
    ok = ok and good
    print(f"G1 period={g1p} (8190), G2 period={g2p} (8191) [{'OK' if good else 'FAIL'}]")

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o51340, 2: 0o12700750, 6: 0o66330754, 30: 0o7411}
    for prn, want in checks.items():
        c = b3i_code(prn)
        good = o24(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B3I PRN{prn:2d}: first24={oct(o24(c))} expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b3i_code(p)) for p in range(1, 11)}) == 10
    ok = ok and distinct
    print(f"PRN 1..10 distinct: {distinct} [{'OK' if distinct else 'FAIL'}]")

    try:
        from gnss_acq import check_negotiation_fidelity, cross_isolation_db
        ok = check_negotiation_fidelity(
            lambda r: build_b3i_buffer(6, r)[0], chip_rate_hz=CHIP_RATE_HZ,
            ideal_rate_hz=40.92e6, negotiated_rate_hz=40.96e6, label="B3I",
            min_db=18.0) and ok
        iso = cross_isolation_db(build_b3i_buffer(6, 40.96e6)[0],
                                 build_b3i_buffer(7, 40.96e6)[0])
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
        Script("BeiDou B3I channel-task — bit-exact BPSK-R(10) ranging code "
               "(BDS-SIS-ICD-B3I) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=6, required=True,
                 help="BeiDou B3I PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B3_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=15.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate (negotiated). B3I is ~20 MHz wide; "
                     "~40 MHz recommended. Fixed per run.")
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
    iq, n_samples = build_b3i_buffer(args.prn, rate_hz)
    path = write_shm(iq, "bds_b3i")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path, "label": f"bds_b3i prn{args.prn}"}
    return spec, [path], [f"PRN            : {args.prn}",
                          f"buffer         : {n_samples} samples (1 ms code period)"]


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="BeiDou B3I channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

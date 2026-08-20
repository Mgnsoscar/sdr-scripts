#!/usr/bin/env python3
"""
bds_b2b_channel — BeiDou B2b (I) channel-task for the X410 engine (expanded mode).

Bit-exact BeiDou **B2b_I** open-service signal (1207.14 MHz): BPSK-R(10) — a
10.23 Mcps, 10230-chip ranging code, 1 ms period (~20.46 MHz wide). One seamless
1 ms loop.

This is the Open Service B2b_I ranging signal; PPP-B2b (correction data) is a
separate service on the same carrier — not this. No navigation data (bare code);
B2b_I has no pilot component or secondary code.

Code (BDS-SIS-ICD-B2b §5): Gold code from two 13-bit LFSRs (register 1 all-ones,
short-cycled at chip 8190; register 2 a per-SV initial state). Validated in
--self-test against the ICD's first-24 and last-24 chip check values.

See gps_prn_channel.py for the channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: B2b is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    bds_b2b_channel.py --channel 0 --prn 6 --gain 55 --amplitude 0
    bds_b2b_channel.py --self-test        # codes vs ICD + fidelity, no engine
    bds_b2b_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

B2B_HZ = 1207.14e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
G1_TAPS = (1, 9, 10, 13)
G2_TAPS = (3, 4, 6, 9, 12, 13)

FREQUENCIES = {"BeiDou B2b (1207.14 MHz)": B2B_HZ}
SAMPLE_RATES_MHZ = {"20.48 MHz (min)": 20.48, "40.96 MHz (default)": 40.96,
                    "61.44 MHz (max)": 61.44}

# Register-2 initial values (stage1..13), BDS-SIS-ICD-B2b Table 5-1 (PRN 6..58).
B2B_REG2 = {
    6: "1000110101110", 7: "1000111101110", 8: "1000111111011", 9: "1001100101001",
    10: "1001111011010", 11: "1010000110101", 12: "1010001000100", 13: "1010001010101",
    14: "1010001011011", 15: "1010001011100", 16: "1010010100011", 17: "1010011110111",
    18: "1010100000001", 19: "1010100111110", 20: "1010110101011", 21: "1010110110001",
    22: "1011001010011", 23: "1011001100010", 24: "1011010011000", 25: "1011010110110",
    26: "1011011110010", 27: "1011011111111", 28: "1011100010010", 29: "1011100111100",
    30: "1011110100001", 31: "1011111001000", 32: "1011111010100", 33: "1011111101011",
    34: "1011111110011", 35: "1100001010001", 36: "1100010010100", 37: "1100010110111",
    38: "1100100010001", 39: "1100100011001", 40: "1100110101011", 41: "1100110110001",
    42: "1100111010010", 43: "1101001010101", 44: "1101001110100", 45: "1101011001011",
    46: "1101101010111", 47: "1110000110100", 48: "1110010000011", 49: "1110010001011",
    50: "1110010100011", 51: "1110010101000", 52: "1110100111011", 53: "1110110010111",
    54: "1111001001000", 55: "1111010010100", 56: "1111010011001", 57: "1111011011010",
    58: "1111011111000",
}


# ── B2b_I ranging code (bit-exact, BDS-SIS-ICD-B2b §5) ─────────────────────────

def b2b_code(prn: int):
    """The 10230-chip B2b_I ranging code (0/1) for a PRN (6..58)."""
    if prn not in B2B_REG2:
        raise ValueError(f"PRN must be 6..58, got {prn}")
    r1 = [1] * 13
    r2 = [int(c) for c in B2B_REG2[prn]]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        if i == RESET_CHIP:
            r1 = [1] * 13
        out[i] = r1[12] ^ r2[12]
        f1 = 0
        for t in G1_TAPS:
            f1 ^= r1[t - 1]
        r1 = [f1] + r1[:12]
        f2 = 0
        for t in G2_TAPS:
            f2 ^= r2[t - 1]
        r2 = [f2] + r2[:12]
    return out


def build_b2b_buffer(prn: int, samp_rate_hz: float):
    """Complex64 B2b_I buffer over one 1 ms code period (seamless). Real BPSK, Q=0."""
    import numpy as np
    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b2b_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def octs(bits):
        f = l = 0
        for b in bits[:24]:
            f = (f << 1) | b
        for b in bits[-24:]:
            l = (l << 1) | b
        return "%08o" % f, "%08o" % l

    checks = {6: ("42471422", "44530033"), 7: ("42071026", "63454537"),
              58: ("70100474", "31701764")}
    for prn, want in checks.items():
        c = b2b_code(prn)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B2b PRN{prn:2d}: {octs(c)} expect {want} [{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b2b_code(p)) for p in range(6, 16)}) == 10
    ok = ok and distinct
    print(f"PRN 6..15 distinct: {distinct} [{'OK' if distinct else 'FAIL'}]")

    try:
        from gnss_acq import check_negotiation_fidelity, cross_isolation_db
        ok = check_negotiation_fidelity(
            lambda r: build_b2b_buffer(6, r)[0], chip_rate_hz=CHIP_RATE_HZ,
            ideal_rate_hz=40.92e6, negotiated_rate_hz=40.96e6, label="B2b",
            min_db=18.0) and ok
        iso = cross_isolation_db(build_b2b_buffer(6, 40.96e6)[0],
                                 build_b2b_buffer(7, 40.96e6)[0])
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
        Script("BeiDou B2b_I channel-task — bit-exact BPSK-R(10) open-service "
               "ranging code (BDS-SIS-ICD-B2b) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=6, max=58, default=6, required=True,
                 help="BeiDou B2b PRN (6..58). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B2B_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=15.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate (negotiated). B2b is ~20 MHz wide; "
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
    iq, n_samples = build_b2b_buffer(args.prn, rate_hz)
    path = write_shm(iq, "bds_b2b")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path, "label": f"bds_b2b prn{args.prn}"}
    return spec, [path], [f"PRN            : {args.prn}",
                          f"buffer         : {n_samples} samples (1 ms code period)"]


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="BeiDou B2b_I channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

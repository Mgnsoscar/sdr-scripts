#!/usr/bin/env python3
"""
mcode_channel — M-code-like BOC(10,5) channel-task for the X410 engine.

Reproduces the GPS **M-code modulation** — sine-phased **BOC(10,5)**: a ±1
spreading code at 5.115 Mcps multiplied by a 10.23 MHz square subcarrier — giving
M-code's characteristic split spectrum (two lobes at ±10.23 MHz, ~30 MHz total).
Built once and replayed on one engine channel (mode "expanded").

⚠  NOT the real M-code. The actual military spreading sequence is CLASSIFIED and
   encrypted and cannot be generated here. This uses an UNCLASSIFIED surrogate PRN
   (a GPS C/A Gold code) under the BOC(10,5) subcarrier, so the RF/spectral shape
   matches M-code but the signal is a test surrogate — it is not, and cannot be,
   tracked as the real military code. Use it for front-end / spectrum / interference
   testing only.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) and L2 (1227.60 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup (cable + attenuators) you are
   LICENSED / AUTHORISED to use — never radiate over the air.

BOC(10,5): s(t) = code(t) · sc(t), code ±1 at 5.115 Mcps, sc a sine-phased square
subcarrier at 10.23 MHz (= 2× the code rate, so it stays commensurate → seamless
loop). Real baseband (I = ±1, Q = 0). Main lobes at ±10.23 MHz → use ≥40 MHz
sample rate; 60 MHz is noticeably cleaner (the square subcarrier's 3rd-harmonic
lobes at ±30.69 MHz alias near the main lobes at 40).

See gps_prn_channel.py for the channel-task lifecycle and on-air pre-roll.

CLI
───
    mcode_channel.py --channel 0 --prn 5 --freq 1.57542e9 --samp_rate 61.44 --gain 55
    mcode_channel.py --self-test        # Gold code + sizing + fidelity, no engine
    mcode_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

L1_HZ = 1575.42e6
L2_HZ = 1227.60e6

CODE_LEN = 1023                 # surrogate Gold-code length (chips)
CODE_RATE_HZ = 5_115_000        # BOC(10,5) code rate: 5 × 1.023 Mcps
SUBCARRIER_HZ = 10_230_000      # BOC(10,5) subcarrier: 10 × 1.023 MHz (= 2 × code rate)

FREQUENCIES = {"GPS L1 (1575.42 MHz)": L1_HZ, "GPS L2 (1227.60 MHz)": L2_HZ}

SAMPLE_RATES_MHZ = {
    "40.96 MHz (min — main lobes)":  40.96,
    "61.44 MHz (default — cleaner)": 61.44,
}

# GPS ICD-200 Table 3-Ia G2 tap pairs (1-indexed) — the surrogate spreading code.
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── Surrogate spreading code (GPS C/A Gold code, pure Python) ──────────────────

def ca_code(prn: int):
    """1023-chip GPS C/A Gold code for a PRN (1..32) as 0/1 — the unclassified
    surrogate standing in for the classified M-code sequence."""
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1..32, got {prn}")
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


# ── Baseband buffer (one seamless-looping BOC(10,5) period) ────────────────────

def build_boc_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 BOC(10,5) buffer: surrogate Gold code × sine-phased
    10.23 MHz square subcarrier, sized to a whole number of code periods that is
    also an integer number of samples (seamless loop). Unit magnitude.
    Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
    n_periods = spp.denominator
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code                        # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * CODE_RATE_HZ // sr) % CODE_LEN     # exact chip mapping
    sub = np.where((n * (2 * SUBCARRIER_HZ) // sr) % 2 == 0, 1.0, -1.0)   # square subcarrier
    iq = (bipolar[chip_idx] * sub).astype(np.complex64)   # real BPSK × subcarrier
    return iq, n_samples, n_periods


# ── Self-test (surrogate code + BOC invariants + fidelity, no engine) ──────────

def _self_test() -> int:
    from fractions import Fraction
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
              f"expect={_FIRST10_OCTAL[prn]:#06o} [{'OK' if good else 'FAIL'}]")

    boc_ok = SUBCARRIER_HZ == 2 * CODE_RATE_HZ
    print(f"fsub == 2·fc (BOC(10,5) commensurate): {boc_ok} [{'OK' if boc_ok else 'FAIL'}]")
    ok = ok and boc_ok
    for samp_mhz in (40.96, 61.44):
        sr = int(round(samp_mhz * 1e6))
        spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
        halfcyc = Fraction(2 * SUBCARRIER_HZ * spp.numerator, sr)
        seam = halfcyc.denominator == 1 and halfcyc.numerator % 2 == 0
        ok = ok and seam
        print(f"{samp_mhz:g} MHz → {spp.numerator} samples / {spp.denominator} period(s); "
              f"subcarrier closes: {seam} [{'OK' if seam else 'FAIL'}]")

    try:
        from gnss_acq import check_negotiation_fidelity, cross_isolation_db
        # Surrogate code acquires under BOC(10,5) at the negotiated 61.44 vs ideal 61.38.
        # BOC has a subcarrier whose edges also jitter under ZOH at a non-
        # commensurate stock rate, so allow a looser loss cap than plain BPSK; the
        # absolute peak/sidelobe (min_db) is the real acquisition criterion.
        ok = check_negotiation_fidelity(
            lambda r: build_boc_buffer(5, r)[0],
            chip_rate_hz=CODE_RATE_HZ, ideal_rate_hz=61.38e6, negotiated_rate_hz=61.44e6,
            label="M-code BOC(10,5)", min_db=18.0, max_loss_db=2.5) and ok
        iso = cross_isolation_db(build_boc_buffer(5, 61.44e6)[0],
                                 build_boc_buffer(7, 61.44e6)[0])
        good_iso = iso < -15.0
        ok = ok and good_iso
        print(f"cross-PRN isolation (5 vs 7): {iso:.2f} dB [{'OK' if good_iso else 'FAIL'}]")
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("M-code-like BOC(10,5) channel-task — surrogate GPS M-code (split "
               "spectrum ±10.23 MHz) on one X410 engine channel. Unclassified "
               "C/A-code surrogate; matches the RF shape, not the real sequence.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="Surrogate Gold-code index (1..32). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L1_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=30.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=61.44, required=True,
                help="Target channel sample rate (negotiated). ≥40 MHz for the "
                     "±10.23 MHz lobes; 60+ MHz cleaner. Fixed per run.")
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


# ── Entry point ─────────────────────────────────────────────────────────────────

def build(args, rate_hz):
    iq, n_samples, n_periods = build_boc_buffer(args.prn, rate_hz)
    path = write_shm(iq, "mcode")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"mcode prn{args.prn}"}
    info = [f"PRN (surrogate): {args.prn}",
            f"buffer         : {n_samples} samples ({n_periods} code period(s))"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="M-code BOC(10,5) channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

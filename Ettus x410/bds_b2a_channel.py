#!/usr/bin/env python3
"""
bds_b2a_channel — BeiDou B2a channel-task for the X410 engine (expanded mode).

Bit-exact BeiDou **B2a** signal (1176.45 MHz, shares GPS L5's band): QPSK — a
data component (I) and a pilot component (Q), equal power, each BPSK(10) at
10.23 Mcps.

Both components are TIERED (primary ⊕ secondary), BDS-SIS-ICD-B2a §5:
  data  : Gold primary (10230) ⊕ fixed 5-chip secondary "00010"
  pilot : Gold primary (10230) ⊕ truncated-Weil secondary (100 chips, per-PRN)
Primaries come from two 13-bit LFSRs (register 1 all-ones, short-cycled at chip
8190; register 2 per-PRN). Validated in --self-test against the ICD's first-24 /
last-24 chip check values (primaries + pilot secondary + Legendre).

Loop length (--loop)
────────────────────
  full (default) : one 100 ms tiered period (pilot secondary is 100 ms; the 5 ms
                   data secondary divides it). Bit-exact — a ~33 MB buffer at
                   40.96 MHz.
  primary        : one 1 ms primary period (no secondary cycling). Tiny; same
                   BPSK envelope for spectrum checks.

B2a is ~20 MHz wide → ~40 MHz sample rate. See gps_prn_channel.py for the
channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: B2a is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    bds_b2a_channel.py --channel 0 --prn 6 --gain 55 --amplitude 0
    bds_b2a_channel.py --channel 1 --prn 6 --component pilot --loop primary
    bds_b2a_channel.py --self-test        # codes vs ICD + fidelity, no engine
    bds_b2a_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

B2A_HZ = 1176.45e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
WEIL_N = 1021
DATA_SEC = (0, 0, 0, 1, 0)                # fixed 5-chip data secondary "00010"

DATA_G1 = (1, 5, 11, 13)
DATA_G2 = (3, 5, 9, 11, 12, 13)
PILOT_G1 = (3, 6, 7, 13)
PILOT_G2 = (1, 5, 7, 8, 12, 13)

FREQUENCIES = {"BeiDou B2a (1176.45 MHz)": B2A_HZ}
SAMPLE_RATES_MHZ = {"20.48 MHz (min)": 20.48, "40.96 MHz (default)": 40.96,
                    "61.44 MHz (max)": 61.44}

# Register-2 initial values (stage1..13), BDS-SIS-ICD-B2a Tables 5-2 / 5-3.
B2A_DATA_REG2 = (
    "1000000100101", "1000000110100", "1000010101101", "1000101001111", "1000101010101", "1000110101110",
    "1000111101110", "1000111111011", "1001100101001", "1001111011010", "1010000110101", "1010001000100",
    "1010001010101", "1010001011011", "1010001011100", "1010010100011", "1010011110111", "1010100000001",
    "1010100111110", "1010110101011", "1010110110001", "1011001010011", "1011001100010", "1011010011000",
    "1011010110110", "1011011110010", "1011011111111", "1011100010010", "1011100111100", "1011110100001",
    "1011111001000", "1011111010100", "1011111101011", "1011111110011", "1100001010001", "1100010010100",
    "1100010110111", "1100100010001", "1100100011001", "1100110101011", "1100110110001", "1100111010010",
    "1101001010101", "1101001110100", "1101011001011", "1101101010111", "1110000110100", "1110010000011",
    "1110010001011", "1110010100011", "1110010101000", "1110100111011", "1110110010111", "1111001001000",
    "1111010010100", "1111010011001", "1111011011010", "1111011111000", "1111011111111", "1111110110101",
    "0010000000010", "1101111110101", "0001111010010",
)
B2A_PILOT_REG2 = (
    "1000000100101", "1000000110100", "1000010101101", "1000101001111", "1000101010101", "1000110101110",
    "1000111101110", "1000111111011", "1001100101001", "1001111011010", "1010000110101", "1010001000100",
    "1010001010101", "1010001011011", "1010001011100", "1010010100011", "1010011110111", "1010100000001",
    "1010100111110", "1010110101011", "1010110110001", "1011001010011", "1011001100010", "1011010011000",
    "1011010110110", "1011011110010", "1011011111111", "1011100010010", "1011100111100", "1011110100001",
    "1011111001000", "1011111010100", "1011111101011", "1011111110011", "1100001010001", "1100010010100",
    "1100010110111", "1100100010001", "1100100011001", "1100110101011", "1100110110001", "1100111010010",
    "1101001010101", "1101001110100", "1101011001011", "1101101010111", "1110000110100", "1110010000011",
    "1110010001011", "1110010100011", "1110010101000", "1110100111011", "1110110010111", "1111001001000",
    "1111010010100", "1111010011001", "1111011011010", "1111011111000", "1111011111111", "1111110110101",
    "1010010000110", "0010111111000", "0001101010101",
)

# Pilot secondary (truncated Weil): per-PRN (phase w, truncation point p), Table 5-4.
B2A_PILOT_SEC = (
    (123, 138), (55, 570), (40, 351), (139, 77), (31, 885), (175, 247),
    (350, 413), (450, 180), (478, 3), (8, 26), (73, 17), (97, 172),
    (213, 30), (407, 1008), (476, 646), (4, 158), (15, 170), (47, 99),
    (163, 53), (280, 179), (322, 925), (353, 114), (375, 10), (510, 584),
    (332, 60), (7, 3), (13, 684), (16, 263), (18, 545), (25, 22),
    (50, 546), (81, 190), (118, 303), (127, 234), (132, 38), (134, 822),
    (164, 57), (177, 668), (208, 697), (249, 93), (276, 18), (349, 66),
    (439, 318), (477, 133), (498, 98), (88, 70), (155, 132), (330, 26),
    (3, 354), (21, 58), (84, 41), (111, 182), (128, 944), (153, 205),
    (197, 23), (199, 1), (214, 792), (256, 641), (265, 83), (291, 7),
    (324, 111), (326, 96), (340, 92),
)


# ── B2a codes (bit-exact, BDS-SIS-ICD-B2a §5.2.1) ──────────────────────────────

def _primary(prn: int, component: str):
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    if component == "data":
        g1t, g2t, reg2 = DATA_G1, DATA_G2, B2A_DATA_REG2[prn - 1]
    else:
        g1t, g2t, reg2 = PILOT_G1, PILOT_G2, B2A_PILOT_REG2[prn - 1]
    r1 = [1] * 13
    r2 = [int(c) for c in reg2]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        if i == RESET_CHIP:
            r1 = [1] * 13
        out[i] = r1[12] ^ r2[12]
        f1 = 0
        for t in g1t:
            f1 ^= r1[t - 1]
        r1 = [f1] + r1[:12]
        f2 = 0
        for t in g2t:
            f2 ^= r2[t - 1]
        r2 = [f2] + r2[:12]
    return out


_LEG = None


def _legendre():
    global _LEG
    if _LEG is None:
        qr = {(x * x) % WEIL_N for x in range(1, WEIL_N)}
        _LEG = [0] + [1 if k in qr else 0 for k in range(1, WEIL_N)]
    return _LEG


def _pilot_secondary(prn: int):
    w, p = B2A_PILOT_SEC[prn - 1]
    L = _legendre()
    W = [L[k] ^ L[(k + w) % WEIL_N] for k in range(WEIL_N)]
    return [W[(n + p - 1) % WEIL_N] for n in range(100)]


def build_b2a_buffer(prn: int, component: str, loop: str, samp_rate_hz: float):
    """Complex64 B2a buffer. component 'both'|'data'|'pilot'; loop 'full' (100 ms
    tiered) | 'primary' (1 ms). Returns (iq, n_samples)."""
    import numpy as np

    pd = np.asarray(_primary(prn, "data"), dtype=np.int8)
    pp = np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    sd = np.asarray(DATA_SEC, dtype=np.int8)
    sp = np.asarray(_pilot_secondary(prn), dtype=np.int8)

    n_periods = 100 if loop == "full" else 1
    sr = int(round(samp_rate_hz))
    n_samples = int(round(n_periods * CODE_LEN / CHIP_RATE_HZ * sr))

    n = np.arange(n_samples, dtype=np.int64)
    gc = n * CHIP_RATE_HZ // sr
    m = gc // CODE_LEN                    # primary-period index
    c = gc % CODE_LEN                     # chip within primary
    d_bit = pd[c] ^ sd[m % 5]
    p_bit = pp[c] ^ sp[m % 100]
    d = 1.0 - 2.0 * d_bit                 # logic 1→−1, 0→+1
    p = 1.0 - 2.0 * p_bit

    if component == "data":
        iq = d.astype(np.complex64)
    elif component == "pilot":
        iq = p.astype(np.complex64)
    else:
        iq = ((d + 1j * p) / math.sqrt(2.0)).astype(np.complex64)   # QPSK
    return iq, n_samples


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

    prim_chk = {
        ("data", 1): ("26771056", "42646672"), ("data", 2): ("64771737", "43261240"),
        ("data", 63): ("55037136", "06147764"),
        ("pilot", 1): ("26772435", "05133452"), ("pilot", 63): ("25236023", "01076040"),
    }
    for (comp, prn), want in prim_chk.items():
        c = _primary(prn, comp)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"{comp} primary PRN{prn:2d}: {octs(c)} expect {want} [{'OK' if good else 'FAIL'}]")

    for prn, want in {1: ("32063050", "65322167")}.items():
        s = _pilot_secondary(prn)
        good = octs(s) == want and len(s) == 100
        ok = ok and good
        print(f"pilot secondary PRN{prn}: {octs(s)} expect {want} [{'OK' if good else 'FAIL'}]")

    leg_ok = sum(_legendre()) == (WEIL_N - 1) // 2
    ok = ok and leg_ok
    print(f"Legendre(1021) ones={sum(_legendre())} [{'OK' if leg_ok else 'FAIL'}]")

    try:
        from gnss_acq import check_negotiation_fidelity
        # Fidelity on the 1 ms data primary (the acquired code); the 100 ms tiered
        # buffer repeats the primary and would show every ms as a sidelobe.
        ok = check_negotiation_fidelity(
            lambda r: build_b2a_buffer(6, "data", "primary", r)[0],
            chip_rate_hz=CHIP_RATE_HZ, ideal_rate_hz=40.92e6, negotiated_rate_hz=40.96e6,
            label="B2a data primary (1 ms)", min_db=18.0) and ok
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BeiDou B2a channel-task — bit-exact QPSK B2a (tiered data+pilot, "
               "BDS-SIS-ICD-B2a) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=6, required=True,
                 help="BeiDou B2a PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B2A_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .choice("-Component", "--component", options=["both", "data", "pilot"],
                default="both", help="both = QPSK; data = I only; pilot = Q only.")
        .choice("-Loop", "--loop", options=["full", "primary"], default="full",
                help="full = 100 ms tiered (~33 MB @ 40.96 MHz); primary = 1 ms.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=15.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate (negotiated). B2a is ~20 MHz wide; "
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
    iq, n_samples = build_b2a_buffer(args.prn, args.component, args.loop, rate_hz)
    path = write_shm(iq, "bds_b2a")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"bds_b2a prn{args.prn} {args.component}"}
    period = "100 ms tiered" if args.loop == "full" else "1 ms primary"
    info = [f"PRN            : {args.prn}   component {args.component}   loop {args.loop}",
            f"buffer         : {n_samples} samples ({period}, {n_samples*8/1e6:.1f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="BeiDou B2a channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

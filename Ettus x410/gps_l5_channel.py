#!/usr/bin/env python3
"""
gps_l5_channel — GPS L5 channel-task for the X410 engine (expanded mode).

Generates a bit-exact GPS **L5** signal (1176.45 MHz), QPSK at 10.23 Mcps:

    L5I (in-phase, "data")  : 10230-chip primary × NH10 secondary (10 ms)
    L5Q (quadrature, pilot) : 10230-chip primary × NH20 secondary (20 ms)

The whole signal repeats every 20 ms (the NH20 period), so the complete buffer is
small (~6.5 MB at 40.96 MHz) and plays as one seamless expanded loop — no
composite needed. Primary codes are the real IS-GPS-705 XA⊕XB construction with
the per-PRN XB code-advance table; NH10/NH20 are exact. No navigation data (bare
code). --channel picks I, Q, or the equal-power QPSK sum.

Primary XA: x^13+x^12+x^10+x^9+1 (short-cycled to 8190); XB: maximal 8191. Each
PRN = XA ⊕ (XB advanced per IS-GPS-705). Validated in --self-test (XB m-sequence,
per-PRN length/balance/distinctness, first-24-chip check values).

L5 is ~20 MHz wide → use ~40 MHz sample rate (default target 40.96 MHz). See
gps_prn_channel.py for the channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: L5 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gps_l5_channel.py --channel 0 --prn 5 --gain 55 --amplitude 0
    gps_l5_channel.py --channel 1 --prn 5 --component Q          # pilot only
    gps_l5_channel.py --self-test        # real codes + NH + fidelity, no engine
    gps_l5_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

L5_HZ = 1176.45e6
CODE_RATE_HZ = 10_230_000        # 10.23 Mcps
L5_CODE_LEN = 10230              # chips in an L5 primary code (1 ms period)

NH10 = [0, 0, 0, 0, 1, 1, 0, 1, 0, 1]                                  # L5I, 10 ms
NH20 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]    # L5Q, 20 ms

FREQUENCIES = {"GPS L5 (1176.45 MHz)": L5_HZ}
SAMPLE_RATES_MHZ = {"20.48 MHz (min)": 20.48, "40.96 MHz (default)": 40.96,
                    "61.44 MHz (max)": 61.44}

# Per-PRN XB code advance (chips), IS-GPS-705 — index = PRN-1, PRN 1..63.
L5I_XB_ADVANCE = (
    266, 365, 804, 1138, 1509, 1559, 1756, 2084, 2170, 2303,
    2527, 2687, 2930, 3471, 3940, 4132, 4332, 4924, 5343, 5443,
    5641, 5816, 5898, 5918, 5955, 6243, 6345, 6477, 6518, 6875,
    7168, 7187, 7329, 7577, 7720, 7777, 8057,
    5358, 3550, 3412, 819, 4608, 3698, 962, 3001, 4441, 4937,
    3717, 4730, 7291, 2279, 7613, 5723, 7030, 1475, 2593, 2904,
    2056, 2757, 3756, 6205, 5053, 6437,
)
L5Q_XB_ADVANCE = (
    1701, 323, 5292, 2020, 5429, 7136, 1041, 5947, 4315, 148,
    535, 1939, 5206, 5910, 3595, 5135, 6082, 6990, 3546, 1523,
    4548, 4484, 1893, 3961, 7106, 5299, 4660, 276, 4389, 3783,
    1591, 1601, 749, 1387, 1661, 3210, 708,
    4226, 5604, 6375, 3056, 1772, 3662, 4401, 5218, 2838, 6913,
    1685, 1194, 6963, 5001, 6694, 991, 7489, 2441, 639, 2097,
    2498, 6470, 2399, 242, 3768, 1186,
)


# ── L5 primary code (bit-exact XA ⊕ XB, IS-GPS-705) ────────────────────────────

_XA = None
_XB = None


def _build_registers() -> None:
    global _XA, _XB
    if _XA is not None:
        return
    xa = [1] * 13
    ya = []
    for _ in range(L5_CODE_LEN):
        ya.append(xa[12])
        if xa == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]:   # short cycle → reset
            xa = [1] * 13
        else:
            xa = [xa[12] ^ xa[11] ^ xa[9] ^ xa[8]] + xa[0:12]
    xb = [1] * 13
    yb = []
    for _ in range(8191):
        yb.append(xb[12])
        xb = [xb[12] ^ xb[11] ^ xb[7] ^ xb[6] ^ xb[5] ^ xb[3] ^ xb[2] ^ xb[0]] + xb[0:12]
    _XA, _XB = ya, yb


def l5_primary(prn: int, channel: str):
    """One 10230-chip L5 primary code (0/1) for a PRN (1..63) and channel ('I'/'Q')."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    _build_registers()
    adv = (L5I_XB_ADVANCE if channel == "I" else L5Q_XB_ADVANCE)[prn - 1]
    return [_XA[i] ^ _XB[(adv + i) % 8191] for i in range(L5_CODE_LEN)]


# ── Baseband buffer (one seamless-looping 20 ms L5 period) ─────────────────────

def build_l5_buffer(prn: int, component: str, samp_rate_hz: float):
    """Build a complex64 L5 buffer over one 20 ms NH period (loops seamlessly).
    component 'I'/'Q'/'IQ'. Constant-modulus QPSK for 'IQ'. Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.020 * sr))          # 20 ms — full NH20 period

    bi = 1.0 - 2.0 * np.asarray(l5_primary(prn, "I"), dtype=np.float32)
    bq = 1.0 - 2.0 * np.asarray(l5_primary(prn, "Q"), dtype=np.float32)
    nh10 = 1.0 - 2.0 * np.asarray(NH10, dtype=np.float32)
    nh20 = 1.0 - 2.0 * np.asarray(NH20, dtype=np.float32)

    n = np.arange(n_samples, dtype=np.int64)
    gchip = n * CODE_RATE_HZ // sr               # 0 .. 204599
    ms_idx = gchip // L5_CODE_LEN                 # 0 .. 19
    chip = gchip % L5_CODE_LEN                    # 0 .. 10229

    i_val = bi[chip] * nh10[ms_idx % 10]
    q_val = bq[chip] * nh20[ms_idx % 20]

    if component == "I":
        iq = i_val.astype(np.complex64)
    elif component == "Q":
        iq = q_val.astype(np.complex64)
    else:  # IQ — equal-power QPSK, constant modulus
        iq = ((i_val + 1j * q_val) / np.sqrt(2.0)).astype(np.complex64)
    return iq, n_samples


# ── Self-test (real-code validation + NH + fidelity, no engine) ────────────────

def _self_test() -> int:
    from fractions import Fraction
    check24 = {
        ("I", 1): 0o66124275, ("I", 2): 0o24763202,
        ("I", 10): 0o41006103, ("I", 32): 0o30576255,
        ("Q", 1): 0o63131310, ("Q", 2): 0o44165373,
        ("Q", 10): 0o47557674, ("Q", 32): 0o52731266,
    }
    ok = True

    _build_registers()
    xb_ones = sum(_XB)
    good = len(_XB) == 8191 and xb_ones == 4096
    ok = ok and good
    print(f"XB maximal m-sequence: len={len(_XB)} ones={xb_ones} "
          f"(expect 8191/4096) [{'OK' if good else 'FAIL'}]")

    for ch in ("I", "Q"):
        for (cch, prn), want in check24.items():
            if cch != ch:
                continue
            v = 0
            for b in l5_primary(prn, ch)[:24]:
                v = (v << 1) | b
            good = v == want
            ok = ok and good
            print(f"{ch} PRN{prn:2d}: first24={oct(v)} expect={oct(want)} "
                  f"[{'OK' if good else 'FAIL'}]")

    nh_ok = len(NH10) == 10 and len(NH20) == 20
    ok = ok and nh_ok
    print(f"NH10 len={len(NH10)}, NH20 len={len(NH20)} [{'OK' if nh_ok else 'FAIL'}]")

    for samp_mhz in (40.96, 61.44):
        sr = int(round(samp_mhz * 1e6))
        n = 0.020 * sr
        chips = Fraction(int(round(n)) * CODE_RATE_HZ, sr)
        good = n == int(n) and chips == 204600
        ok = ok and good
        print(f"{samp_mhz:g} MHz → {n:.0f} samples/20ms, chips={chips} [{'OK' if good else 'FAIL'}]")

    try:
        import numpy as np
        from gnss_acq import check_negotiation_fidelity

        # A receiver acquires the 1 ms L5 primary (NH resolves the 20 ms secondary),
        # so measure fidelity over ONE primary period — the full 20 ms buffer repeats
        # the primary 20× and would show every ms as a "sidelobe".
        def primary_1ms(rate_hz, prn=5, chan="I"):
            sr = int(round(rate_hz))
            n = int(round(0.001 * sr))
            b = 1.0 - 2.0 * np.asarray(l5_primary(prn, chan), dtype=np.float32)
            idx = (np.arange(n, dtype=np.int64) * CODE_RATE_HZ // sr) % L5_CODE_LEN
            return b[idx].astype(np.complex64)

        ok = check_negotiation_fidelity(
            primary_1ms, chip_rate_hz=CODE_RATE_HZ, ideal_rate_hz=40.92e6,
            negotiated_rate_hz=40.96e6, label="L5 primary (1 ms)", min_db=18.0) and ok
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GPS L5 channel-task — bit-exact QPSK L5 (I data × NH10 + Q pilot × "
               "NH20, real IS-GPS-705 codes) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS L5 PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L5_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .choice("-Component", "--component", options=["IQ", "I", "Q"], default="IQ",
                help="IQ = full QPSK; I = data only; Q = pilot only. Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=15.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate (negotiated). L5 is ~20 MHz wide; "
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
    iq, n_samples = build_l5_buffer(args.prn, args.component, rate_hz)
    path = write_shm(iq, "gps_l5")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"gps_l5 prn{args.prn} {args.component}"}
    info = [f"PRN            : {args.prn}   component {args.component}",
            f"buffer         : {n_samples} samples (20 ms NH period)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="GPS L5 channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

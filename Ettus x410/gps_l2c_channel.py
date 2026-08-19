#!/usr/bin/env python3
"""
gps_l2c_channel — GPS L2C channel-task for the X410 engine (expanded mode).

Generates a bit-exact GPS **L2C** signal (1227.60 MHz): the civil L2 signal, a
chip-by-chip time-multiplex of two 511.5 kcps codes interleaved to 1.023 Mcps
(BPSK-R(1), ~2 MHz wide):

    L2 CM (Civil Moderate) : 10230 chips, 20 ms period  (carries CNAV; here bare)
    L2 CL (Civil Long)     : 767250 chips, 1.5 s period (dataless pilot)

Both from the IS-GPS-200 27-stage Galois register (mask 0o445112474), per-PRN
initial states, validated in --self-test against the official L2C sheet
(init→end after a full period) plus first-24-chip check values.

Loop length (--loop)
────────────────────
  full (default) : one whole CL period = 1.5 s (CM repeats 75×). Bit-exact,
                   complete spectrum — but a large buffer (1.5 s × rate × 8 B ≈
                   123 MB at 10.24 MHz). Heavy on the ARM if several run at once.
  cm             : one CM period = 20 ms (CL truncated). Tiny; identical BPSK-R(1)
                   envelope, but the CL line structure sits at 50 Hz not 0.667 Hz
                   (both unresolvable at practical RBW). Good for envelope checks.

L2C is ~2 MHz wide, so ~10 MHz sample rate is plenty (default target 10.24 MHz =
10 samples/chip after negotiation). See gps_prn_channel.py for the lifecycle.

⚠  RF SAFETY / LEGAL: L2 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gps_l2c_channel.py --channel 0 --prn 5 --gain 55 --amplitude 0
    gps_l2c_channel.py --channel 1 --prn 5 --loop cm --samp_rate 20.48
    gps_l2c_channel.py --self-test        # generator vs sheet + fidelity, no engine
    gps_l2c_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

L2_HZ = 1227.60e6
CM_LEN = 10230                   # L2 CM code length (20 ms at 511.5 kcps)
CL_LEN = 767250                  # L2 CL code length (1.5 s at 511.5 kcps)
CHANNEL_CHIP_RATE = 511_500      # each of CM / CL
COMBINED_CHIP_RATE = 1_023_000   # after chip-by-chip multiplexing
LFSR_MASK = 0o445112474          # IS-GPS-200 L2C feedback polynomial (Galois)

FREQUENCIES = {"GPS L2 (1227.60 MHz)": L2_HZ}
SAMPLE_RATES_MHZ = {"10.24 MHz (default)": 10.24, "20.48 MHz": 20.48, "40.96 MHz": 40.96}

# Per-PRN initial shift-register states (octal), IS-GPS-200, PRN 1..63.
L2CM_INIT = (
    0o742417664, 0o756014035, 0o002747144, 0o066265724, 0o601403471, 0o703232733,
    0o124510070, 0o617316361, 0o047541621, 0o733031046, 0o713512145, 0o024437606,
    0o021264003, 0o230655351, 0o001314400, 0o222021506, 0o540264026, 0o205521705,
    0o064022144, 0o120161274, 0o044023533, 0o724744327, 0o045743577, 0o741201660,
    0o700274134, 0o010247261, 0o713433445, 0o737324162, 0o311627434, 0o710452007,
    0o722462133, 0o050172213, 0o500653703, 0o755077436, 0o136717361, 0o756675453,
    0o435506112, 0o771353753, 0o226107701, 0o022025110, 0o402466344, 0o752566114,
    0o702011164, 0o041216771, 0o047457275, 0o266333164, 0o713167356, 0o060546335,
    0o355173035, 0o617201036, 0o157465571, 0o767360553, 0o023127030, 0o431343777,
    0o747317317, 0o045706125, 0o002744276, 0o060036467, 0o217744147, 0o603340174,
    0o326616775, 0o063240065, 0o111460621,
)
L2CL_INIT = (
    0o624145772, 0o506610362, 0o220360016, 0o710406104, 0o001143345, 0o053023326,
    0o652521276, 0o206124777, 0o015563374, 0o561522076, 0o023163525, 0o117776450,
    0o606516355, 0o003037343, 0o046515565, 0o671511621, 0o605402220, 0o002576207,
    0o525163451, 0o266527765, 0o006760703, 0o501474556, 0o743747443, 0o615534726,
    0o763621420, 0o720727474, 0o700521043, 0o222567263, 0o132765304, 0o746332245,
    0o102300466, 0o255231716, 0o437661701, 0o717047302, 0o222614207, 0o561123307,
    0o240713073, 0o101232630, 0o132525726, 0o315216367, 0o377046065, 0o655351360,
    0o435776513, 0o744242321, 0o024346717, 0o562646415, 0o731455342, 0o723352536,
    0o000013134, 0o011566642, 0o475432222, 0o463506741, 0o617127534, 0o026050332,
    0o733774235, 0o751477772, 0o417631550, 0o052247456, 0o560404163, 0o417751005,
    0o004302173, 0o715005045, 0o001154457,
)


# ── L2C codes (bit-exact 27-stage Galois LFSR, IS-GPS-200) ─────────────────────

def _lfsr_bits(init: int, n: int):
    """n output chips (LSB each) of the L2C register started at `init`."""
    x = init
    out = [0] * n
    for i in range(n):
        out[i] = x & 1
        x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
    return out


def cm_code(prn: int):
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CM_INIT[prn - 1], CM_LEN)


def cl_code(prn: int, n: int = CL_LEN):
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CL_INIT[prn - 1], n)


# ── Baseband buffer (CM/CL time-multiplexed, seamless loop) ────────────────────

def build_l2c_buffer(prn: int, loop: str, samp_rate_hz: float):
    """Build a complex64 L2C baseband buffer (real BPSK, Q=0). loop='full' → one
    1.5 s CL period; loop='cm' → one 20 ms CM period (CL truncated). Returns
    (iq, n_samples)."""
    import numpy as np

    n_cl = CL_LEN if loop == "full" else CM_LEN
    cm = np.asarray(cm_code(prn), dtype=np.int8)
    cl = np.asarray(cl_code(prn, n_cl), dtype=np.int8)

    period_s = n_cl / CHANNEL_CHIP_RATE          # 1.5 s (full) or 20 ms (cm)
    sr = int(round(samp_rate_hz))
    n_samples = int(round(period_s * sr))

    n = np.arange(n_samples, dtype=np.int64)
    gchip = n * COMBINED_CHIP_RATE // sr         # 0 .. 2*n_cl-1
    half = (gchip >> 1)
    is_cl = (gchip & 1) == 1
    bit = np.where(is_cl, cl[half % n_cl], cm[half % CM_LEN])
    iq = (1.0 - 2.0 * bit).astype(np.complex64)
    return iq, n_samples


# ── Self-test (generator vs official sheet + fidelity, no engine) ──────────────

def _self_test() -> int:
    ok = True

    def run(init, steps):
        x = init
        for _ in range(steps):
            x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
        return x

    sheet = [  # (cm_init, cm_end, cl_init, cl_end) — official L2C sheet PRN 159/160
        (0o604055104, 0o425373114, 0o605253024, 0o44547544),
        (0o157065232, 0o427153064, 0o63314262, 0o707116115),
    ]
    for i, (cmi, cme, cli, cle) in enumerate(sheet):
        cm_ok = run(cmi, CM_LEN - 1) == cme
        ok = ok and cm_ok
        print(f"sheet PRN{159+i} CM init→end ({CM_LEN} chips): [{'OK' if cm_ok else 'FAIL'}]")
    cl_ok = run(sheet[0][2], CL_LEN - 1) == sheet[0][3]
    ok = ok and cl_ok
    print(f"sheet PRN159 CL init→end ({CL_LEN} chips): [{'OK' if cl_ok else 'FAIL'}]")

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {("CM", 1): 0o12757036, ("CM", 2): 0o50370043,
              ("CL", 1): 0o24676104, ("CL", 2): 0o20022732}
    for (kind, prn), want in checks.items():
        got = o24(cm_code(prn) if kind == "CM" else cl_code(prn, 24))
        good = got == want
        ok = ok and good
        print(f"{kind} PRN{prn}: first24={oct(got)} expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    cm1 = cm_code(1)
    good_len = len(cm1) == CM_LEN
    ok = ok and good_len
    print(f"CM len={len(cm1)} ones={sum(cm1)} (≈5115) [{'OK' if good_len else 'FAIL'}]")

    # Negotiation fidelity on the 20 ms cm-loop buffer (fast; the acquired code).
    try:
        from gnss_acq import check_negotiation_fidelity
        ok = check_negotiation_fidelity(
            lambda r: build_l2c_buffer(5, "cm", r)[0],
            chip_rate_hz=COMBINED_CHIP_RATE, ideal_rate_hz=10.23e6,
            negotiated_rate_hz=10.24e6, label="L2C (cm loop)", min_db=18.0) and ok
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GPS L2C channel-task — bit-exact CM/CL time-multiplexed civil L2 "
               "signal (IS-GPS-200) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS L2C PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L2_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .choice("-Loop", "--loop", options=["full", "cm"], default="full",
                help="full = 1.5 s CL period (bit-exact, ~123 MB @ 10.24 MHz); "
                     "cm = 20 ms (tiny, same envelope). Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=4.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=10.24, required=True,
                help="Target channel sample rate (negotiated). L2C is ~2 MHz wide; "
                     "10 MHz is plenty. Fixed per run.")
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
    iq, n_samples = build_l2c_buffer(args.prn, args.loop, rate_hz)
    path = write_shm(iq, "gps_l2c")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"gps_l2c prn{args.prn}"}
    period = "1.5 s CL" if args.loop == "full" else "20 ms CM"
    info = [f"PRN            : {args.prn}   loop {args.loop} ({period})",
            f"buffer         : {n_samples} samples ({n_samples*8/1e6:.1f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="GPS L2C channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

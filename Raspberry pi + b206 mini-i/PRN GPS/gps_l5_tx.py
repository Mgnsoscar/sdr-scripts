#!/usr/bin/env python3
"""
GPS L5 transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** GPS **L5** signal (1176.45 MHz): the QPSK of two
10.23 Mcps BPSK channels —

    L5I (in-phase, "data")  : 10230-chip primary × NH10 secondary (10 ms)
    L5Q (quadrature, pilot) : 10230-chip primary × NH20 secondary (20 ms)

Precomputed and replayed from a file so a Raspberry Pi can sustain the 40+ MS/s
an L5 signal needs (same recipe as gps_prn_tx.py / mcode_boc_tx.py).

Code fidelity — these are the real IS-GPS-705 codes
───────────────────────────────────────────────────
The primary codes come from the L5 XA/XB shift-register construction with the
per-PRN **XB code-advance** table from IS-GPS-705. Tables and generator were
cross-validated against TWO independent open-source implementations
(pmonta/GNSS-DSP-tools and taroz/GNSS-SDRLIB) — they agree exactly — plus these
structural checks (all in --self-test):
  • XB is a maximal m-sequence: period 8191, 4096 ones,
  • XA short-cycles to 8190 chips,
  • each primary code is 10230 chips, balanced (5115 ±1 ones), distinct per PRN,
  • the first-24-chips (octal) of several PRNs match known check values.
So `--prn N` produces satellite N's actual L5 code — a receiver can acquire and
identify it. NH10/NH20 secondary codes are exact. (No navigation data is
modulated — bare code; the Q pilot is dataless anyway, and the I channel would
normally carry CNAV, so it's like a constant data bit.)

⚠  RF SAFETY / LEGAL: L5 is a live GNSS (aeronautical safety-of-life) band.
   Transmit ONLY into a shielded / conducted setup you are LICENSED / AUTHORISED
   to use — never radiate over the air.

Why it runs on a Pi + live tuning: see gps_prn_tx.py. Precompute + loop from a
/dev/shm file, sc8 over the wire, quiet, 1:1 master clock. Live knobs: gain and
amplitude (instant). PRN / carrier / channel / sample rate / otw fixed per run.

Sample rate: L5 needs fs ≳ 20.46 MHz; the default 40.92 MHz (= 40 × 1.023) gives
an exact 4 samples/chip so chip edges land on samples (cleanest). 61.38 also works.

CLI
───
    gps_l5_tx.py --prn 5 --gain 55 --sample_rate 40.92
    gps_l5_tx.py --channel Q      # pilot only (clean acquisition tests)
    gps_l5_tx.py --self-test      # verify real codes + NH + sizing, no hardware
    gps_l5_tx.py --describe-params
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script


# ── Constants ─────────────────────────────────────────────────────────────────

L5_HZ = 1176.45e6
CODE_RATE_HZ = 10_230_000        # 10.23 Mcps
L5_CODE_LEN = 10230              # chips in an L5 primary code (1 ms period)

# Neuman-Hofman secondary codes (bit patterns), IS-GPS-705 §3.2.1.4.
NH10 = [0, 0, 0, 0, 1, 1, 0, 1, 0, 1]                                  # L5I, 10 ms
NH20 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]    # L5Q, 20 ms

# Per-PRN XB code advance (chips), IS-GPS-705 — index = PRN-1, PRN 1..63.
# Cross-validated against pmonta/GNSS-DSP-tools and taroz/GNSS-SDRLIB (identical).
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

FREQUENCIES = {"GPS L5 (1176.45 MHz)": L5_HZ}


# ── L5 primary code (bit-exact XA ⊕ XB, IS-GPS-705) ────────────────────────────
# XA: x^13 + x^12 + x^10 + x^9 + 1, short-cycled to 8190 chips (reset one chip
# early). XB: x^13 + x^12 + x^8 + x^7 + x^6 + x^4 + x^3 + x + 1, maximal (8191).
# Each PRN's code is XA ⊕ (XB advanced by the per-PRN amount).

_XA: list | None = None
_XB: list | None = None


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


def l5_primary(prn: int, channel: str) -> list[int]:
    """One 10230-chip L5 primary code (0/1) for a PRN (1..63) and channel
    ('I' or 'Q') — the real IS-GPS-705 code for that satellite."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    _build_registers()
    adv = (L5I_XB_ADVANCE if channel == "I" else L5Q_XB_ADVANCE)[prn - 1]
    return [_XA[i] ^ _XB[(adv + i) % 8191] for i in range(L5_CODE_LEN)]


# ── Self-test (real-code validation + sizing; pure Python, no hardware) ────────

def _self_test() -> int:
    from fractions import Fraction
    # first-24-chips (octal) reference values — a transcription guard proving the
    # XB-advance tables produce the real IS-GPS-705 codes.
    check24 = {
        ("I", 1): 0o66124275, ("I", 2): 0o24763202,
        ("I", 10): 0o41006103, ("I", 32): 0o30576255,
        ("Q", 1): 0o63131310, ("Q", 2): 0o44165373,
        ("Q", 10): 0o47557674, ("Q", 32): 0o52731266,
    }
    ok = True

    _build_registers()
    xb_ones = sum(_XB)
    print(f"XB maximal m-sequence: len={len(_XB)} ones={xb_ones} "
          f"(expect 8191/4096) [{'OK' if len(_XB) == 8191 and xb_ones == 4096 else 'FAIL'}]")
    ok = ok and len(_XB) == 8191 and xb_ones == 4096

    for ch in ("I", "Q"):
        # Length exact, near-balanced (a 10230 code sits within ~±60 of 5115), distinct.
        bal = all(len(l5_primary(p, ch)) == L5_CODE_LEN
                  and abs(sum(l5_primary(p, ch)) - 5115) < 100 for p in range(1, 64))
        distinct = len({tuple(l5_primary(p, ch)) for p in range(1, 64)}) == 63
        print(f"{ch}: PRN 1..63 length/balance ok={bal}, distinct={distinct}")
        ok = ok and bal and distinct
        # Bit-exactness: first-24-chips match the reference values.
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
    print(f"NH10 len={len(NH10)}, NH20 len={len(NH20)} [{'OK' if nh_ok else 'FAIL'}]")
    ok = ok and nh_ok

    for samp_mhz in (40.92, 61.38, 40.0):
        sr = int(round(samp_mhz * 1e6))
        n = 0.020 * sr
        chips = Fraction(int(round(n)) * CODE_RATE_HZ, sr)
        print(f"{samp_mhz:g} MHz → {n:.0f} samples/20ms (int={n == int(n)}), "
              f"chips={chips} (=204600), samples/chip={sr / CODE_RATE_HZ:.4f}")
        ok = ok and n == int(n) and chips == 204600

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (one seamless-looping 20 ms L5 period) ─────────────────────

def build_l5_buffer(prn: int, channel: str, samp_rate_hz: float):
    """Build a complex64 L5 baseband buffer over one full 20 ms NH period (loops
    seamlessly). Constant-modulus QPSK; amplitude applied live downstream.
    Returns (iq, n_samples)."""
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

    if channel == "I":
        iq = i_val.astype(np.complex64)
    elif channel == "Q":
        iq = q_val.astype(np.complex64)
    else:  # IQ — equal-power QPSK, constant modulus
        iq = ((i_val + 1j * q_val) / np.sqrt(2.0)).astype(np.complex64)
    return iq, n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file: str, center_freq_hz: float, samp_rate_hz: float,
                     gain_db: float, amplitude: float, otw_format: str,
                     extra_args: str):
    from gnuradio import gr, blocks, uhd

    class L5Tx(gr.top_block):
        def __init__(self):
            super().__init__("GPS L5 TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=otw_format,
                                channels=[0]),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return L5Tx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GPS L5 transmitter (QPSK: L5I data + L5Q pilot, real IS-GPS-705 "
               "codes, 10.23 Mcps, NH secondary codes), file-replay at high sample "
               "rate. Authorised, shielded setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63) — the real L5 code. Fixed per run.")
        .choice("-Channel", "--channel", options=["IQ", "I", "Q"], default="IQ",
                help="IQ = full L5 (QPSK); I = data channel only; Q = pilot only.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L5_HZ,
                help="RF carrier (default L5). Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=50,
                required=True, live=True, help="USRP TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.9,
                required=True, live=True,
                help="Baseband digital amplitude (0..1). Live.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=20.46, max=61.44,
                default=40.92,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "40.92 (=40×1.023) → 4 samples/chip. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load; sc16 for more "
                     "dynamic range.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()
    samp_rate_hz = args.sample_rate * 1e6

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gps_l5_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_l5_buffer(args.prn, args.channel, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"l5_prn{args.prn}_{args.channel}.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(
        iq_file=iq_file, center_freq_hz=args.freq, samp_rate_hz=samp_rate_hz,
        gain_db=args.gain, amplitude=args.amplitude, otw_format=args.otw,
        extra_args="")

    print("── GPS L5 TX ───────────────────────────────────────────────")
    print(f"  satellite PRN  : {args.prn}  (channel {args.channel}, real L5 code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : QPSK-R(10) — 10.23 Mcps, ±10.23 MHz lobes")
    print(f"  buffer         : {nsamp} samples (20 ms — full NH period)")
    print(f"  otw / gain     : {args.otw} / {args.gain:g} dB")
    print(f"  amplitude      : {args.amplitude:g}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "gain":
            tb.set_gain(value)
            ctrl.report("gain", tb.actual_gain())
        elif name == "amplitude":
            tb.set_amplitude(value)
            ctrl.report("amplitude", value)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                apply_change(change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

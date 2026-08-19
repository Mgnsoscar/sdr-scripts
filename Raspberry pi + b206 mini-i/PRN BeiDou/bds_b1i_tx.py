#!/usr/bin/env python3
"""
BeiDou B1I transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** BeiDou **B1I** signal (1561.098 MHz): BPSK-R(2) —
a 2.046 Mcps, 2046-chip ranging code, 1 ms period (~4 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B1I codes
──────────────────────────────────────────
A balanced Gold code (truncated by its last chip → 2046) from two 11-bit LFSRs
(ICD §4.3):
  G1(X) = 1+X+X^7+X^8+X^9+X^10+X^11
  G2(X) = 1+X+X^2+X^3+X^4+X^5+X^8+X^9+X^11
both initialised 01010101010. The per-SV code is G1(stage 11) ⊕ a XOR of selected
G2 stages (the ICD Table 4-1 "phase assignment", e.g. PRN 1 = 1⊕3, PRN 63 =
3⊕6⊕9). The generator + per-SV tap table match the ICD exactly and are byte-
identical to pmonta/GNSS-DSP-tools' beidou/b1i.py (which is used to acquire live
B1I); --self-test re-checks the codes against embedded reference values.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data / 1 kHz NH secondary code (those ride on the data).

⚠  RF SAFETY / LEGAL: B1I is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
live gain + amplitude. The default 20.46 MHz (= 10×2.046) gives 10 samples/chip.

CLI
───
    bds_b1i_tx.py --prn 6 --gain 55
    bds_b1i_tx.py --self-test
    bds_b1i_tx.py --describe-params
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

B1I_HZ = 1561.098e6
CHIP_RATE_HZ = 2_046_000
CODE_LEN = 2046
G_INIT = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)   # 01010101010, both registers

FREQUENCIES = {"BeiDou B1I (1561.098 MHz)": B1I_HZ}

# Per-SV G2 phase selection (XORed G2 stages, 1-indexed), BDS-SIS-ICD-B1I Table 4-1.
G2_TAPS = (
    (1,3),(1,4),(1,5),(1,6),(1,8),(1,9),(1,10),(1,11),(2,7),(3,4),
    (3,5),(3,6),(3,8),(3,9),(3,10),(3,11),(4,5),(4,6),(4,8),(4,9),
    (4,10),(4,11),(5,6),(5,8),(5,9),(5,10),(5,11),(6,8),(6,9),(6,10),
    (6,11),(8,9),(8,10),(8,11),(9,10),(9,11),(10,11),
    (1,2,7),(1,3,4),(1,3,6),(1,3,8),(1,3,10),(1,3,11),(1,4,5),(1,4,9),
    (1,5,6),(1,5,8),(1,5,10),(1,5,11),(1,6,9),(1,8,9),(1,9,10),(1,9,11),
    (2,3,7),(2,5,7),(2,7,9),(3,4,5),(3,4,9),(3,5,6),(3,5,8),(3,5,10),
    (3,5,11),(3,6,9),
)


# ── B1I ranging code (bit-exact, BDS-SIS-ICD-B1I §4.3) ─────────────────────────

def _g1_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[6] ^ r[7] ^ r[8] ^ r[9] ^ r[10]] + r[0:10]        # taps 1,7,8,9,10,11


def _g2_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[7] ^ r[8] ^ r[10]] + r[0:10]  # 1,2,3,4,5,8,9,11


def b1i_code(prn: int) -> list[int]:
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


# ── Self-test (period + code check values; no hardware) ────────────────────────

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
    print(f"G1 period={per} (expect 2047) [{'OK' if per==2047 else 'FAIL'}]")
    ok = ok and per == 2047

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o31333315, 2: 0o44461070, 6: 0o32442011, 38: 0o67733254, 63: 0o74366441}
    for prn, want in checks.items():
        c = b1i_code(prn)
        got = o24(c)
        good = got == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B1I PRN{prn:2d}: first24={oct(got)} expect={oct(want)} len={len(c)} "
              f"[{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b1i_code(p)) for p in range(1, 11)}) == 10
    print(f"PRN 1..10 distinct: {distinct}")
    ok = ok and distinct

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b1i_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 B1I baseband buffer over one 1 ms code period (loops
    seamlessly). Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))               # 1 ms — one code period
    bipolar = 1.0 - 2.0 * np.asarray(b1i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B1ITx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B1I TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=otw_format, channels=[0]),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def set_amplitude(self, a): self.amp.set_k(a)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return B1ITx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BeiDou B1I transmitter (real BDS-SIS-ICD-B1I ranging code, "
               "BPSK-R(2), 2.046 Mcps), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou SV / ranging-code number (1..63). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B1I_HZ,
                help="RF carrier (default B1I). Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=50,
                required=True, live=True, help="USRP TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.9,
                required=True, live=True, help="Baseband amplitude (0..1). Live.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=5.0, max=61.44,
                default=20.46,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "20.46 (=10×2.046) → 10 samples/chip. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load; sc16 more range.")
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
    tmpdir = tempfile.mkdtemp(prefix="bds_b1i_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b1i_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b1i_prn{args.prn}.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(iq_file, args.freq, samp_rate_hz, args.gain,
                          args.amplitude, args.otw, "")

    print("── BeiDou B1I TX ───────────────────────────────────────────")
    print(f"  SV / code num  : {args.prn}  (real B1I ranging code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BPSK-R(2) — 2.046 Mcps, ~4 MHz wide")
    print(f"  buffer         : {nsamp} samples (1 ms code period)")
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

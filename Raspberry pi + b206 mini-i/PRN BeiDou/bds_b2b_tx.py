#!/usr/bin/env python3
"""
BeiDou B2b (I-component) transmitter for GNU Radio + UHD (B200-mini family).

Generates a **bit-exact** BeiDou **B2b_I** open-service signal (1207.14 MHz):
BPSK-R(10) — a 10.23 Mcps, 10230-chip ranging code, 1 ms period (~20.46 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Note: this is the **Open Service B2b_I** ranging signal. PPP-B2b (precise point
positioning) is a separate correction-data service on the same carrier — not this.

Code fidelity — real BDS-SIS-ICD-B2b codes
──────────────────────────────────────────
Gold code from two 13-bit LFSRs (register 1 all-ones, short-cycled at chip 8190;
register 2 per-PRN), output = stage 13, ICD §5:
  g1 = 1+x+x^9+x^10+x^13     g2 = 1+x^3+x^4+x^6+x^9+x^12+x^13
Validated against the ICD's own check values: generating each code from its
register-2 init reproduces the ICD's first-24 AND last-24 chips (octal) — 53/53
for all PRNs (6..58). --self-test re-checks representative PRNs.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data (bare code). B2b_I has no pilot component or secondary code.

⚠  RF SAFETY / LEGAL: B2b (1207.14 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use — never over air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
live gain + amplitude. Default 40.92 MHz (= 40×1.023) → 4 samples/chip.

CLI
───
    bds_b2b_tx.py --prn 20 --gain 55
    bds_b2b_tx.py --self-test
    bds_b2b_tx.py --describe-params
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

B2B_HZ = 1207.14e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
G1_TAPS = (1, 9, 10, 13)
G2_TAPS = (3, 4, 6, 9, 12, 13)

FREQUENCIES = {"BeiDou B2b (1207.14 MHz)": B2B_HZ}

# Register-2 initial values (stage1..stage13), BDS-SIS-ICD-B2b Table 5-1 (PRN 6..58).
B2B_REG2 = {
    6:"1000110101110", 7:"1000111101110", 8:"1000111111011", 9:"1001100101001",
    10:"1001111011010", 11:"1010000110101", 12:"1010001000100", 13:"1010001010101",
    14:"1010001011011", 15:"1010001011100", 16:"1010010100011", 17:"1010011110111",
    18:"1010100000001", 19:"1010100111110", 20:"1010110101011", 21:"1010110110001",
    22:"1011001010011", 23:"1011001100010", 24:"1011010011000", 25:"1011010110110",
    26:"1011011110010", 27:"1011011111111", 28:"1011100010010", 29:"1011100111100",
    30:"1011110100001", 31:"1011111001000", 32:"1011111010100", 33:"1011111101011",
    34:"1011111110011", 35:"1100001010001", 36:"1100010010100", 37:"1100010110111",
    38:"1100100010001", 39:"1100100011001", 40:"1100110101011", 41:"1100110110001",
    42:"1100111010010", 43:"1101001010101", 44:"1101001110100", 45:"1101011001011",
    46:"1101101010111", 47:"1110000110100", 48:"1110010000011", 49:"1110010001011",
    50:"1110010100011", 51:"1110010101000", 52:"1110100111011", 53:"1110110010111",
    54:"1111001001000", 55:"1111010010100", 56:"1111010011001", 57:"1111011011010",
    58:"1111011111000",
}


# ── B2b_I ranging code (bit-exact, BDS-SIS-ICD-B2b §5) ─────────────────────────

def b2b_code(prn: int) -> list[int]:
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


# ── Self-test (codes vs ICD check values; no hardware) ─────────────────────────

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
        print(f"B2b PRN{prn:2d}: {octs(c)} expect {want} len={len(c)} [{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b2b_code(p)) for p in range(6, 16)}) == 10
    print(f"PRN 6..15 distinct: {distinct}")
    ok = ok and distinct

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b2b_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 B2b_I baseband buffer over one 1 ms code period (loops
    seamlessly). Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b2b_code(prn), dtype=np.int8)   # logic 1→−1, 0→+1
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B2BTx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B2b TX")
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

    return B2BTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BeiDou B2b (I-component) transmitter (real BDS-SIS-ICD-B2b ranging "
               "code, BPSK-R(10), 10.23 Mcps), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=6, max=58, default=6, required=True,
                 help="BeiDou PRN / ranging-code number (6..58). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B2B_HZ,
                help="RF carrier (default B2b). Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=50,
                required=True, live=True, help="USRP TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.9,
                required=True, live=True, help="Baseband amplitude (0..1). Live.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=20.46, max=61.44,
                default=40.92,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "40.92 (=40×1.023) → 4 samples/chip. Fixed per run.")
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
    tmpdir = tempfile.mkdtemp(prefix="bds_b2b_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b2b_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b2b_prn{args.prn}.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(iq_file, args.freq, samp_rate_hz, args.gain,
                          args.amplitude, args.otw, "")

    print("── BeiDou B2b TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B2b_I ranging code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BPSK-R(10) — 10.23 Mcps, ~20.46 MHz wide")
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

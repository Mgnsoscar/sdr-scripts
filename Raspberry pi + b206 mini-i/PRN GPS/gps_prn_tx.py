#!/usr/bin/env python3
"""
GPS-style PRN transmitter for GNU Radio + UHD (Ettus B200-mini family).

Purpose
───────
Transmit a BPSK-modulated GPS C/A Gold code at the L1 carrier, at a high enough
sample rate (40 MS/s by default) to carry a wide, ~20 MHz spreading code — on a
Raspberry Pi, where synthesising IQ at runtime can't keep up.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded/conducted setup (cable + attenuators into a receiver or spectrum
   analyser) that you are LICENSED / AUTHORISED to use. Radiating a PRN over the
   air can jam or spoof real GNSS receivers and is illegal in most places.

How it hits 40 MS/s on a Pi (the three levers)
──────────────────────────────────────────────
1. PRECOMPUTE + LOOP, don't generate at runtime. One full code period is built
   once at startup, written to a small file in /dev/shm (RAM-backed, no SD-card
   wear), and replayed with blocks.file_source(repeat=True). At runtime the CPU
   only DMAs bytes to USB — no per-sample Python/NumPy math, which is what caps a
   live flowgraph at a few MS/s.

2. sc8 OVER THE WIRE. cpu_format=fc32 but otw_format=sc8 halves the USB payload:
   at 40 MS/s that's 80 MB/s instead of 160 MB/s (sc16) — comfortably within a
   Pi's USB3. A PRN is constant-modulus BPSK, so 8-bit I/Q costs nothing that
   matters here (the B200's DAC is 12-bit anyway).

3. NO STATUS UPDATES MID-RUN. UHD's fastpath underflow markers (the "UUUU"
   spam) are disabled, console logging is off, and the task should run with
   PYTHONUNBUFFERED=0 (see configs/tasks.yaml) so stdout block-buffers. Nothing
   is printed once streaming starts — status writes under load actually *cause*
   underflows, so we stay silent.

1:1 master clock
────────────────
master_clock_rate is pinned equal to the sample rate, so UHD runs the AD9361 1:1
with NO FPGA halfband resampling and NO rate coercion — you get exactly the rate
you asked for, so the samples-per-chip and the loop length stay exact. (At 1:1
the only anti-image filtering is the analog TX low-pass, which is fine for a
signal that fills the band.)

Seamless looping
────────────────
At 40 MS/s a 1023-chip code period is an exact integer number of samples
(1.023 Mcps → 40000 samples; 10.23 Mcps → 4000 samples), so the file loops with
no seam. For any other rate the buffer is sized to the smallest whole number of
code periods that is also an integer number of samples (see build_iq_buffer),
so every generated buffer loops perfectly too.

Live tuning (retune while transmitting, via paramkit.live)
──────────────────────────────────────────────────────────
Precomputing removes the old tension between "high rate" and "live tuning" — the
control socket is nearly free, so both work at once:

    gain        → UHD set_gain            (instant)
    amplitude   → multiply_const_cc.set_k (instant — baseband digital scale)
    code_rate   → file_source.open(other) (instant swap between prebuilt files)

The 1.023 and 10.23 Mcps period files are both built at startup, so switching the
spreading-code width mid-run is just a file swap — GNU Radio changes source at the
next work boundary (one brief seam at the swap, then the new file loops cleanly).
PRN index, carrier, sample rate and otw are fixed for a run (change them by
restarting the task).

CLI
───
    gps_prn_tx.py --prn 5 --code_rate 10.23 --gain 55 --amplitude 0.9
    gps_prn_tx.py --self-test        # verify the Gold-code generator, no hardware
    gps_prn_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# UHD/GNU Radio must be quiet BEFORE the libraries load (they read these at
# import). The heavy imports live inside main(), so setting them here takes
# effect. A task env (configs/tasks.yaml) can still override any of them.
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")   # no UHD console logging
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")  # no "UUUU" underflow spam
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")        # skip slow pref scan

# paramkit is pure-Python and always importable (agent puts it on PYTHONPATH).
# NumPy and GNU Radio are imported lazily so --self-test / --describe-params run
# anywhere, including CI and dev boxes without a radio stack.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script


# ── Constants ─────────────────────────────────────────────────────────────────

L1_HZ = 1575.42e6
L2_HZ = 1227.60e6

CODE_LEN = 1023                 # chips in a GPS C/A Gold code period

# The two selectable spreading-code rates ("1 MHz" and "10 MHz" codes), in Mcps.
# 1.023 Mcps is the true GPS C/A rate (~2 MHz null-to-null); 10.23 Mcps is the
# same Gold code clocked 10× (~20 MHz null-to-null). Both divide 40 MS/s into an
# integer-length period, so each loops seamlessly.
CODE_RATES_MCPS = {
    "1 MHz code — 1.023 Mcps (~2 MHz BW)":  1.023,
    "10 MHz code — 10.23 Mcps (~20 MHz BW)": 10.23,
}

FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": L1_HZ,
    "GPS L2 (1227.60 MHz)": L2_HZ,
}

SAMPLE_RATES = {
    "4.092 MHz (Minimum -> Just main lobe + first skirt)": 4.092,
    "20.46 MHz (Default -> Most faithfull representation)": 20.46,
    "61.38 MHz (Maximum -> Captures the widest skirts)": 61.38
}

# GPS ICD-200 Table 3-Ia: G2 code-phase tap pairs (1-indexed) selecting each
# satellite's C/A code. Verified against the ICD "first 10 chips" column for all
# 32 PRNs (see --self-test).
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}

# ICD reference: first 10 chips of each PRN's C/A code, as an octal integer.
# Used only by --self-test to prove the generator matches the standard.
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── GPS C/A Gold-code generation (pure Python, no NumPy) ───────────────────────

def ca_code(prn: int) -> list[int]:
    """Return the 1023-chip GPS C/A Gold code for a PRN (1..32) as a list of 0/1.

    Two 10-stage LFSRs (both seeded all-ones):
      G1: x^10 + x^3 + 1
      G2: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
    The chip is G1's output XOR the sum of two PRN-specific G2 taps.
    """
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1 to 32, got {prn}")
    g1 = [1] * 10
    g2 = [1] * 10
    ta, tb = G2_TAPS[prn]
    out: list[int] = []
    for _ in range(CODE_LEN):
        out.append(g1[9] ^ g2[ta - 1] ^ g2[tb - 1])
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return out


def _self_test() -> int:
    """Verify ca_code() against the ICD reference for all 32 PRNs. Returns 0 on
    success (usable as a process exit code); prints a per-PRN report."""
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
              f"expect={_FIRST10_OCTAL[prn]:#06o} ones={sum(code)} "
              f"[{'OK' if good else 'FAIL'}]")
    print("ALL PRN CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (one seamless-looping code period) ─────────────────────────

def build_iq_buffer(prn: int, chip_rate_hz: float, samp_rate_hz: float):
    """Build a complex64 baseband buffer holding a whole number of C/A code
    periods that is also an exact integer number of samples, so it loops with no
    seam. BPSK: I = ±1 chip, Q = 0. Amplitude is applied live downstream (a
    multiply_const), so the buffer is left at unit magnitude.

    Returns (iq: np.ndarray[complex64], n_samples: int, n_periods: int).
    """
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(chip_rate_hz))
    # Samples per code period as an exact fraction; the number of periods we must
    # tile to reach an integer sample count is its denominator in lowest terms.
    spp = Fraction(sr * CODE_LEN, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator          # == n_periods * (samples per one period)

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code         # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr // sr) % CODE_LEN   # exact zero-order-hold chip mapping
    iq = bipolar[chip_idx].astype(np.complex64)   # Q stays 0 (real BPSK)
    return iq, n_samples, n_periods


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(files_by_mcps: dict, initial_mcps: float, center_freq_hz: float,
                     samp_rate_hz: float, gain_db: float, amplitude: float,
                     otw_format: str, extra_args: str):
    """Construct the GNU Radio top_block. Imported lazily so the module loads
    without a radio stack for --self-test / --describe-params."""
    from gnuradio import gr, blocks, uhd

    class PrnTx(gr.top_block):
        def __init__(self):
            super().__init__("GPS PRN TX")

            # Pin master clock == sample rate → 1:1, no FPGA resampling/coercion.
            # Enlarge the USB send buffer so the host stays ahead at high rate.
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args

            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(
                    cpu_format="fc32", otw_format=otw_format,
                    channels=[0]
                ),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            self.src = blocks.file_source(
                gr.sizeof_gr_complex, files_by_mcps[initial_mcps], repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def swap_code_file(self, path: str) -> None:
            self.src.open(path, True)          # switch at next work boundary

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return PrnTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(
            "GPS-style PRN (C/A Gold code) transmitter."
        )
        .integer(
            "-PRN", "--prn", 
            min=1, max=32, 
            default=1, 
            required=True,
            help="GPS satellite PRN / Gold code index (1 to 32). Fixed per run."
            )
        .number(
            "-Code-rate", "--code_rate", 
            unit="Mcps",
            min=0.1, max=20.0,
            presets=CODE_RATES_MCPS, default=1.023, required=True, live=True,
            help="Spreading-code chip rate."
        )
        .number(
            "-Gain", "--gain", 
            unit="dB", 
            min=0, max=89.75, 
            default=89.75,
            required=True, 
            live=True, 
            help="USRP TX gain."
        )
        .number(
            "-Amplitude", "--amplitude", 
            min=0.0, max=1.0, 
            default=0.11,
            required=True, 
            live=True,
            help="Baseband digital amplitude (0 to 1)."
        )
        .number(
            "-Center-frequency", "--freq", 
            unit="Hz", 
            min=70e6, max=6e9,
            presets=FREQUENCIES, default=L1_HZ,
            help="RF carrier. Fixed per run."
        )
        .number(
            "-Sample-rate", "--samp_rate", 
            unit="MHz", min=1.23, max=61.44,
            default=SAMPLE_RATES["20.46 MHz (Default -> Most faithfull representation)"],
            presets=SAMPLE_RATES,
            help=
                "Host/DAC sample rate; master clock is pinned equal to it "
                "(1:1). Fixed per run."
        )
        .choice(
            "-OTW-format", "--otw", 
            options=["sc8", "sc16"], 
            default="sc8",
            help=
                "Over-the-wire sample format. sc8 halves USB load (needed "
                "for 40+ MS/s on a Pi); sc16 for more dynamic range."
        )
    )


# ── Live-change dispatch ────────────────────────────────────────────────────────

def _apply_live_change(tb, ctrl, files_by_mcps, name: str, value) -> None:
    if name == "gain":
        tb.set_gain(value)
        ctrl.report("gain", tb.actual_gain())
    elif name == "amplitude":
        tb.set_amplitude(value)
        ctrl.report("amplitude", value)
    elif name == "code_rate":
        path = files_by_mcps.get(round(float(value), 6))
        if path is None:
            # A rate other than the two prebuilt ones — reject rather than stall
            # the stream. (Restart the task to change PRN/rate wholesale.)
            ctrl.report("code_rate", "rejected: only 1.023 or 10.23 Mcps live")
            return
        tb.swap_code_file(path)
        ctrl.report("code_rate", value)


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()

    samp_rate_hz = args.samp_rate * 1e6

    # Temp dir for the period files: prefer /dev/shm (RAM-backed → fast, no SD
    # wear); fall back to the default temp location if it's absent.
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gps_prn_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    # Prebuild BOTH selectable code-rate files up front so a live code-rate change
    # is an instant file swap. Keyed by Mcps (rounded) to match the live value.
    files_by_mcps: dict = {}
    for mcps in CODE_RATES_MCPS.values():
        iq, nsamp, nper = build_iq_buffer(args.prn, mcps * 1e6, samp_rate_hz)
        path = os.path.join(tmpdir, f"prn{args.prn}_{mcps:g}Mcps.fc32")
        iq.tofile(path)
        files_by_mcps[round(mcps, 6)] = path
        print(f"[prebuilt] PRN {args.prn} @ {mcps:g} Mcps → {nsamp} samples "
              f"({nper} code period(s)) → {path}")

    initial_mcps = round(float(args.code_rate), 6)
    if initial_mcps not in files_by_mcps:
        # A custom starting rate: build it on demand and use it as the initial file.
        iq, nsamp, nper = build_iq_buffer(args.prn, args.code_rate * 1e6, samp_rate_hz)
        path = os.path.join(tmpdir, f"prn{args.prn}_{args.code_rate:g}Mcps.fc32")
        iq.tofile(path)
        files_by_mcps[initial_mcps] = path
        print(f"[prebuilt] PRN {args.prn} @ {args.code_rate:g} Mcps → {nsamp} samples")

    tb = _build_top_block(
        files_by_mcps=files_by_mcps, initial_mcps=initial_mcps,
        center_freq_hz=args.freq, samp_rate_hz=samp_rate_hz, gain_db=args.gain,
        amplitude=args.amplitude, otw_format=args.otw, extra_args="")

    # Startup banner (the ONLY output during a run — we go silent after start()).
    print("── GPS PRN TX ──────────────────────────────────────────────")
    print(f"  PRN            : {args.prn}")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  code rate      : {args.code_rate:g} Mcps "
          f"(~{2*args.code_rate:g} MHz null-to-null)")
    print(f"  otw / gain     : {args.otw} / {args.gain:g} dB")
    print(f"  amplitude      : {args.amplitude:g}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                _apply_live_change(tb, ctrl, files_by_mcps, change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

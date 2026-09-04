#!/usr/bin/env python3
"""
M-code-like BOC(10,5) transmitter for GNU Radio + UHD (Ettus B200-mini family).

Reproduces the GPS **M-code modulation** — sine-phased **BOC(10,5)**: a ±1 spreading
code at 5.115 Mcps multiplied by a 10.23 MHz square subcarrier — which gives M-code's
characteristic **split spectrum** (two lobes at ±10.23 MHz around the carrier, ~30 MHz
total). Prebuilt once and looped so a Raspberry Pi sustains the rate with no runtime IQ
math (same recipe as gps_l1ca_tx.py).

⚠  NOT the real M-code. The actual military spreading sequence is CLASSIFIED and
   encrypted and cannot be generated here. This uses an UNCLASSIFIED surrogate PRN (a
   GPS C/A Gold code) under the BOC(10,5) subcarrier, so the RF/spectral shape matches
   M-code but the signal is a test surrogate — it cannot be tracked as the real military
   code. Use it for front-end / spectrum / interference testing only.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) and L2 (1227.60 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup (cable + attenuators) you are
   LICENSED / AUTHORISED to use — never radiate over the air.

BOC(10,5) construction
──────────────────────
  s(t) = code(t) · sc(t)
    code(t) : surrogate Gold code (±1) at fc = 5·1.023 = 5.115 Mcps
    sc(t)   : sine-phased square subcarrier at fsub = 10·1.023 = 10.23 MHz
Real baseband (I = ±1, Q = 0), like a BPSK PRN but split by the subcarrier. Because
fsub = 2·fc, the subcarrier is commensurate with the code, so a whole code period is an
integer number of samples and the buffer loops with no seam.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 12 samples/chip, 6 samples/subcarrier-cycle, exact), 1:1 clock;
  • over-the-wire sc8 (constant-modulus BOC loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.
The main lobes sit at ±10.23 MHz; the square subcarrier's harmonics extend further, and at
61.38 MHz (± 30.69) the 3rd-harmonic lobes fall at the band edge rather than folding onto the
main lobes (they DO fold at 40 MS/s) — so 61.38 is the cleanest rate, and the digital filter
below can additionally strip the harmonics to leave just the split main lobes.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid, and the
banner reports the power actually achieved. --gain instead commands the raw SDR gain
(relative), overriding --power. Uncalibrated, there is no dBm scale — use --gain.

Digital passband filter (on the looped buffer — no runtime DSP)
──────────────────────────────────────────────────────────────
An optional steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power. BOC(10,5) is a split
spectrum, so instead of a sidelobe count the passband is set directly as a half-bandwidth in
MHz (a lowpass edge each side of the carrier): the default keeps just the two ±10.23 MHz main
lobes and strips the square-subcarrier harmonics.
  • --filter on/off             enable/disable (live);
  • --passband <MHz>            half-bandwidth kept each side of the carrier (live, presets);
  • --transition <MHz>          skirt steepness — the transition width beyond the edge (live).

CLI
───
    MCode.py --prn 5 --freq 1575.42 --power -30              # calibrated dBm, no filter
    MCode.py --prn 5 --gain 60 --filter on --passband 15.345 # relative gain, main lobes only
    MCode.py --self-test
    MCode.py --describe-params
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time

os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it. Absent it, the script runs uncalibrated (relative gain only). See docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1_mcode"

# Which parameter carries the transmit frequency, so a frequency-dependent calibration chain
# folds --power at the live carrier (see the C/A scripts / docs/calibration-v2.md).
CAL_FREQ_PARAM = "freq"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 12 samples/chip, 6 samples/subcarrier-cycle (exact); clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; BOC is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS BOC(10,5)) ────────────────────────────────────
L1_HZ = 1575.42e6
L2_HZ = 1227.60e6
CODE_LEN = 1023                 # surrogate Gold-code length (chips)
CODE_RATE_HZ = 5_115_000        # BOC(10,5) code rate: 5 × 1.023 Mcps
SUBCARRIER_HZ = 10_230_000      # BOC(10,5) subcarrier: 10 × 1.023 MHz (= 2 × code rate)
SIGNAL_NAME = "M-code BOC(10,5)"
NYQUIST_HZ = SAMP_RATE_HZ / 2.0

FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": L1_HZ / 1e6,
    "GPS L2 (1227.60 MHz)": L2_HZ / 1e6,
}   # presets are in MHz now

# Filter presets: {label: passband half-bandwidth in MHz}. The two main lobes are centred at
# ±10.23 MHz and (with the 5.115 Mcps code) reach out to ~±15.3 MHz, so 15.345 keeps them and
# strips the harmonics; wider keeps more of the square-subcarrier structure.
MIN_PASSBAND_MHZ = 5.0
MAX_PASSBAND_MHZ = 30.69
PASSBAND_PRESETS = {
    "Main split lobes (±15.3 MHz)": 15.345,
    "Lobes + first sidelobes (±20.5 MHz)": 20.46,
    "Full square subcarrier (±30.7 MHz)": 30.69,
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

# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# BOC(10,5) is a SPLIT spectrum: TWO main lobes at ±10.23 MHz (each null-to-null over ±5.115 MHz
# about the subcarrier, so |f| ∈ [5.115, 15.345] MHz), plus square-subcarrier harmonics further
# out. Its power distribution is fixed by ONE measured number — the power spectral DENSITY at the
# main-lobe PEAK (~±9.5 MHz off the carrier), in dBm/Hz (per Hz, NOT per MHz). From that single
# number CAL_POWER_LAWS derives two absolute-power quantities the operator can pick for --power
# (the measured density stays available as a third):
#
#   • Main-lobes integrated power (dBm) = peak_dBm/Hz + 10·log10(BW_ML)   ← BOTH main lobes
#   • Full signal power (dBm)           = peak_dBm/Hz + 10·log10(BW_full) ← widest passband (±Fs/2)
#
# BW_ML / BW_full are effective bandwidths (Hz) = ∫G/G_peak of the sine-BOC(10,5) PSD, over the two
# main lobes (±[5.115,15.345] MHz) and over the widest the passband filter can keep (±30.69 MHz).
# The full-signal power is the WIDEST-passband (worst-case) reading, so it is the safe amplifier-
# limiting quantity: narrowing --passband only lowers the emitted power. No carrier/total quantity
# is offered — the square-subcarrier harmonics the filter strips make it ill-defined here.
# --self-test recomputes both effective bandwidths from the BOC PSD and asserts these literals.
CAL_POWER_LAWS = [
    {"id": "main_lobe_power", "name": "Main-lobes integrated power (both lobes)", "unit": "dBm",
     "in": "density", "out": "abs", "k": 69.5073},     # 10·log10(∫G over ±[5.115,15.345] MHz / G_peak)
    {"id": "full_power", "name": "Full signal power (widest passband)", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.1261},      # 10·log10(∫G over ±30.69 MHz / G_peak)
]

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present (SDR_CALIBRATION_FILE),
    else uncalibrated (relative gain only). Cached so build_script and main agree — and so
    --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Surrogate spreading code (GPS C/A Gold code, pure Python) ──────────────────

def ca_code(prn: int) -> list[int]:
    """1023-chip GPS C/A Gold code for a PRN (1..32) as 0/1 — the unclassified surrogate
    standing in for the classified M-code sequence."""
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1..32, got {prn}")
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


# ── Baseband buffer (one seamless-looping BOC(10,5) period) ────────────────────

def build_boc_buffer(prn: int):
    """The complex64 BOC(10,5) buffer at SAMP_RATE_HZ: surrogate Gold code × sine-phased
    10.23 MHz square subcarrier, sized to a whole number of code periods that is also an
    integer sample count (seamless loop). Unit magnitude (amplitude applied live downstream).
    Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(SAMP_RATE_HZ))
    spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
    n_periods = spp.denominator
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code                       # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * CODE_RATE_HZ // sr) % CODE_LEN    # exact chip mapping
    # Sine-phased square subcarrier: +1 for the first half-period, −1 the next, …
    sub = np.where((n * (2 * SUBCARRIER_HZ) // sr) % 2 == 0, 1.0, -1.0)

    iq = (bipolar[chip_idx] * sub).astype(np.complex64)   # real BPSK×subcarrier
    return iq, n_samples, n_periods


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. `fc_hz` is the −6 dB
    cutoff; `trans_hz` sets the tap count (steeper skirt → more taps). Returns (h, n_taps)."""
    import numpy as np
    m = int(np.ceil(5.5 * SAMP_RATE_HZ / max(trans_hz, 1.0))) | 1     # odd
    m = min(m, (max_taps | 1))
    k = np.arange(m)
    c = (m - 1) / 2.0
    fcn = min(fc_hz / SAMP_RATE_HZ, 0.499)          # never above Nyquist
    h = 2 * fcn * np.sinc(2 * fcn * (k - c))
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n1)
           + 0.14128 * np.cos(4 * np.pi * k / n1) - 0.01168 * np.cos(6 * np.pi * k / n1))
    h = h * win
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, passband_hz: float, trans_hz: float, base_fft=None):
    """Circularly filter the looped BOC buffer to a ±`passband_hz` band. Circular convolution
    (multiply the buffer's DFT by the filter's) keeps the result exactly periodic, so the
    filtered loop has no seam; unity passband gain leaves the kept lobes' power unchanged.
    Pass `base_fft` (= np.fft.fft(base_iq)) to reuse it across live filter changes — the base
    loop is fixed per run, so its DFT need only be computed once, which cuts the per-change CPU
    spike (and the underflows it can cause). Returns (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = float(passband_hz)                           # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                          # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    if base_fft is None:
        base_fft = np.fft.fft(base_iq)
    filtered = np.fft.ifft(base_fft * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test (surrogate Gold code + BOC invariants + filter) ──────────────────

def _self_test() -> int:
    from fractions import Fraction
    ok = True

    # Surrogate Gold code matches the ICD reference for all 32 PRNs.
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

    # BOC(10,5) invariants + seamless sizing at the fixed rate.
    print(f"fsub == 2*fc: {SUBCARRIER_HZ == 2 * CODE_RATE_HZ}")
    sr = int(round(SAMP_RATE_HZ))
    spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
    nn, nper = spp.numerator, spp.denominator
    halfcyc = Fraction(2 * SUBCARRIER_HZ * nn, sr)
    seam = halfcyc.denominator == 1 and halfcyc.numerator % 2 == 0
    ok = ok and seam
    print(f"{SAMP_RATE_HZ/1e6:g} MHz → {nn} samples / {nper} code period(s); "
          f"subcarrier closes: {seam} [{'OK' if seam else 'FAIL'}]")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n, _ = build_boc_buffer(1)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, passband_hz=15.345e6, trans_hz=1.5e6)
    lobes = 10 * np.log10(band(filt, 0, 15.345e6) / band(base, 0, 15.345e6))     # kept
    cut = 10 * np.log10(band(filt, 20e6, 28e6) / max(band(base, 20e6, 28e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(lobes) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main lobes ±15.3 MHz, {taps} taps): kept lobes {lobes:+.3f} dB, "
          f"harmonic band {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} "
          f"[{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    # Calibration law constants: recompute the BOC(10,5) effective bandwidths (both main lobes /
    # widest passband) from the sine-BOC PSD and assert the CAL_POWER_LAWS literals didn't drift.
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def _boc_psd(fv):                       # sine-BOC(10,5): n = 2·fsub/fc = 4 (even)
        a = np.sin(np.pi * fv / (2 * SUBCARRIER_HZ)); c = np.cos(np.pi * fv / (2 * SUBCARRIER_HZ))
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (np.sin(np.pi * fv / CODE_RATE_HZ) * a / (np.pi * fv * c)) ** 2
        return np.nan_to_num(np.where(fv == 0, 0.0, v))
    fv = np.arange(-40e6 + 7, 40e6, 2e3)    # +7 Hz offset dodges the exact f = odd·fsub singularities
    g = _boc_psd(fv); gp = g.max()

    def _ebw(lo, hi):                       # effective bandwidth (Hz) = ∫G / G_peak over |f|∈[lo,hi]
        m = (np.abs(fv) >= lo) & (np.abs(fv) < hi)
        return float(_trapz(g[m], fv[m])) / gp
    ml_k = 10 * np.log10(_ebw(5.115e6, 15.345e6))
    full_k = 10 * np.log10(_ebw(0.0, 30.69e6))
    laws = {l["id"]: l["k"] for l in CAL_POWER_LAWS}
    laws_ok = (abs(laws["main_lobe_power"] - ml_k) < 0.02
               and abs(laws["full_power"] - full_k) < 0.02)
    print(f"calibration: both-main-lobes k={ml_k:.4f} (law {laws['main_lobe_power']}), "
          f"full k={full_k:.4f} (law {laws['full_power']}) [{'OK' if laws_ok else 'FAIL'}]")
    ok = ok and laws_ok
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float,
                     amplitude: float):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack.

    The looped baseband buffer is streamed from RAM by a C++ blocks.vector_source_c
    (repeat=True) — NOT a file_source, and NOT a Python source:
      • file_source streams GIL-free and holds the rate, but swapping it live (open()) races
        GNU Radio internally and THROWS "fread error", which kills the source and silences the
        radio. With no file that is impossible.
      • a Python gr.sync_block has no file either, but its work() runs through the Python
        gateway under the GIL and can't sustain 61.38 Msps on a Pi — it underflows even in
        steady state (no tuning).
    vector_source_c is C++ (GIL-free, no per-sample Python) AND has no file, so it streams as
    smoothly as file_source did, with none of the fread risk. Its one caveat: set_data() is not
    itself locked against work(), so the live filter swap runs it under top-block lock()/
    unlock(), which quiesces the flowgraph for the (brief) duration of the swap — the buffer is
    never freed under a running read. That pause happens only on a filter change; the steady
    stream is untouched."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        # vector_source_c wants a contiguous complex64 buffer; the filtered loop may not be.
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class McodeTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            # repeat=True loops the buffer in C++ (no per-wrap Python), vlen=1, no tags.
            self.src = blocks.vector_source_c(_vec(initial_iq), True, 1, [])
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def swap(self, iq):
            # set_data() reassigns (and frees) the source's buffer with no internal lock, so it
            # must not run while work() is reading. lock()/unlock() quiesces the flowgraph
            # around the swap — the only moment the stream pauses, and only on a filter change.
            data = _vec(iq)
            self.lock()
            try:
                self.src.set_data(data, [])
            finally:
                self.unlock()

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return McodeTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (surrogate Gold code under a 10.23 MHz subcarrier) "
               "— fixed 61.38 MHz / sc8, looped buffer, optional power-preserving digital "
               "passband filter. Surrogate test signal, NOT the classified M-code. Level is "
               "set in dBm via the unit's calibration; uncalibrated it runs on a relative "
               "gain. Authorised, shielded setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=L1_HZ / 1e6, required=True,
                help="RF carrier in MHz (M-code is on L1 = 1575.42 and L2 = 1227.60). "
                     "Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="Surrogate Gold-code index (1..32). Fixed per run.")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain, so "
                     "it preserves what it passes). Live.")
        .number("-Passband", "--passband", unit="MHz",
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=15.345,
                presets=PASSBAND_PRESETS, required=False, live=True,
                help="Passband half-bandwidth kept each side of the carrier (MHz). The main "
                     "split lobes reach ~±15.3 MHz; the default keeps them and strips the "
                     "subcarrier harmonics. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.1, max=8.0, default=1.5,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loop).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    center_freq_hz = args.freq * 1e6

    pmap = power_map()
    amplitude = pmap.amplitude

    # Gain precedence: explicit --gain (raw) > calibrated --power > refuse (uncalibrated).
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz)
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    # Prebuild the unfiltered loop once (PRN is fixed per run); the filter derives from it.
    base_iq, nsamp, nper = build_boc_buffer(args.prn)
    base_fft = {"v": None}      # DFT of the fixed base loop — computed once, reused per change

    # Filter "shape" (the regeneration-requiring params) — mutated by live changes.
    shape = {"on": getattr(args, "filter", "off") == "on",
             "passband_hz": float(getattr(args, "passband", 15.345) or 15.345) * 1e6,
             "trans_hz": float(getattr(args, "transition", 1.5) or 1.5) * 1e6}

    def make_current():
        """The buffer for the current shape: the base loop, or the circularly-filtered loop.
        Returns (iq, info) where info describes the filter for the banner/report."""
        if not shape["on"]:
            return base_iq, {"on": False}
        if base_fft["v"] is None:
            import numpy as np
            base_fft["v"] = np.fft.fft(base_iq)
        filtered, taps, fp = filter_buffer(base_iq, shape["passband_hz"], shape["trans_hz"],
                                           base_fft=base_fft["v"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "trans_hz": shape["trans_hz"]}

    iq0, finfo = make_current()

    tb = _build_top_block(initial_iq=iq0, center_freq_hz=center_freq_hz,
                          gain_db=gain_db, amplitude=amplitude)

    def regenerate():
        """Rebuild the loop for the current filter shape and swap it in atomically (one seam,
        then it loops clean). Runs on the control thread; the flowgraph keeps streaming the old
        buffer until the swap. In-RAM — no file, so the source can never be left dead by a
        read error."""
        iq, info = make_current()
        tb.swap(iq)
        return info

    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def _fmt_band(info):
        if not info.get("on"):
            return "off (full signal)"
        return (f"on — passband ±{info['edge_hz']/1e6:.2f} MHz, "
                f"{info['trans_hz']/1e6:g} MHz transition, {info['taps']} taps")

    print(f"── {SIGNAL_NAME} TX (surrogate) ─────────────────────────────")
    print(f"  surrogate PRN  : {args.prn}  (NOT the classified M-code)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : BOC(10,5) — {CODE_RATE_HZ/1e6:g} Mcps code, "
          f"{SUBCARRIER_HZ/1e6:g} MHz subcarrier (lobes at ±{SUBCARRIER_HZ/1e6:g} MHz)")
    print(f"  buffer         : {nsamp} samples ({nper} code period(s), {nsamp*8/1e6:.2f} MB)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz):.2f} dBm")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  filter         : {_fmt_band(finfo)}")
    print(f"  otw            : {OTW_FORMAT}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "power" and pmap.has_absolute:
            state["gain"] = pmap.gain_for_power(float(value), freq=center_freq_hz)
            if state["rf_on"]:
                tb.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=center_freq_hz), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
            ctrl.report("gain", round(state["gain"], 2))
        elif name == "rf":
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            if on:
                tb.set_amplitude(amplitude)
                tb.set_gain(state["gain"])
            else:
                tb.set_gain(0.0)
                tb.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")
        elif name in ("filter", "passband", "transition"):
            if name == "filter":
                shape["on"] = str(value).strip().lower() in ("on", "1", "true", "yes")
            elif name == "passband":
                shape["passband_hz"] = max(MIN_PASSBAND_MHZ, min(MAX_PASSBAND_MHZ,
                                                                 float(value))) * 1e6
            else:
                shape["trans_hz"] = float(value) * 1e6
            regenerate()
            ctrl.report(name, value)

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

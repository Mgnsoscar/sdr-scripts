#!/usr/bin/env python3
"""
GPS L1 C/A transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a BPSK GPS L1 C/A Gold code (1.023 Mcps) at the L1 carrier (1575.42 MHz),
prebuilt once and looped so a Raspberry Pi sustains the rate with no runtime IQ math.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded/conducted setup (cable + attenuators into a receiver or spectrum analyser)
   you are LICENSED / AUTHORISED to use. Radiating a PRN can jam or spoof real GNSS.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 60 samples/chip, exact), master clock pinned 1:1;
  • over-the-wire sc8 (constant-modulus BPSK loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid (the SDR
gain step and any active-component steps), and the banner reports the power actually
achieved on that grid. --gain instead commands the raw SDR gain (relative), overriding
--power. Uncalibrated, there is no dBm scale — use --gain. (See docs/calibration-v2.md.)

Digital passband filter (ALWAYS ON — on the looped buffer, no runtime DSP)
─────────────────────────────────────────────────────────────────────────
An always-on steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power: if the main lobe measures
−2.5 dBm it reads −2.5 dBm filtered — the filter only removes what's outside the passband. The
passband is ALWAYS an integer number of C/A sidelobes, which is what makes the emitted power a
well-defined function of the sidelobe count (see the calibration note below).
  • --sidelobes <n>             passband keeps the main lobe + n C/A sidelobes, i.e. a
                                ±(n+1)·1.023 MHz band (live, presets by sidelobe count);
  • --transition <MHz>          skirt steepness — the transition width beyond the passband
                                edge (live).
Both are LIVE: changing one rebuilds the (circularly-)filtered loop and swaps it into the
running source; the flowgraph never stops. The loop is streamed from RAM by a C++
blocks.vector_source_c (repeat=True), NOT a file and NOT a Python source: a file_source streams
smoothly but a live swap of it races GNU Radio and THROWS "fread error" (a concurrent open()
resets its length while the old file pointer is mid-loop → a short read → the source thread
dies, radio silent); a Python source has no file but its work() runs under the GIL and can't
hold 61.38 Msps on a Pi, underflowing even untouched. vector_source_c is C++ (GIL-free) with no
file, so it streams as smoothly as file_source with no fread. The filter swap calls set_data()
under top-block lock()/unlock() (set_data isn't internally locked), so the buffer is never freed
under a running read; that pauses the stream only for the swap itself, only on a filter change.

Spectral-density calibration (dBm/Hz at the main-lobe peak → power quantities)
─────────────────────────────────────────────────────────────────────────────
This signal calibrates from ONE measurement — the power spectral DENSITY at the main-lobe PEAK,
in dBm/Hz — because its sinc² shape is fixed. From that single number CAL_POWER_LAWS derives two
absolute-power quantities the operator can pick between for --power (in the calibration editor):
the MAIN-LOBE integrated power, and the FULL signal power passed by the filter (which grows with
the sidelobe count and is the amplifier's LIMITING quantity). See CAL_POWER_LAWS below and
docs/calibration-v2.md §13 (sdr-agent).

CLI
───
    gps_l1ca_tx.py --prn 5 --power -30                       # calibrated dBm (main lobe + 2 sidelobes)
    gps_l1ca_tx.py --prn 5 --gain 60 --sidelobes 5          # relative gain, main + 5 sidelobes
    gps_l1ca_tx.py --self-test        # verify the Gold-code generator (+ filter/laws, if numpy)
    gps_l1ca_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time

# UHD/GNU Radio must be quiet BEFORE the libraries load (they read these at import). The
# heavy imports live inside main(), so setting them here takes effect.
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")   # no UHD console logging
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")  # no "UUUU" underflow spam
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")        # skip slow pref scan

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "GPS L1 C/A (1.023 Mcps)"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain (a
# cable/antenna whose loss varies with frequency) has a --power scale that MOVES with frequency,
# so the map is folded at THIS param's value; --freq is live, so retuning the carrier re-scales
# --power on the fly, and the client folds the range shown in the Run/sequence form to match.
CAL_FREQ_PARAM = "freq"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 60 samples/chip at 1.023 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; BPSK is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L1 C/A) ───────────────────────────────────
CARRIER_HZ = 1575.42e6        # GPS L1 (the --freq default; retunable for bench testing)
CODE_RATE_HZ = 1.023e6        # C/A chip rate (~2 MHz null-to-null)
SIGNAL_NAME = "GPS L1 C/A (1.023 Mcps)"
CODE_LEN = 1023               # chips in a C/A Gold code period
CA_NULL_HZ = 1.023e6          # main-lobe null spacing == the chip rate; sidelobes step by this

FREQUENCIES = {"GPS L1 (1575.42 MHz)": CARRIER_HZ / 1e6}   # presets are in MHz

# Filter presets: {label: number of C/A sidelobes to KEEP}. The passband is the main lobe
# plus that many sidelobes, i.e. a ±(n+1)·1.023 MHz band. Max keeps the band inside ±Fs/2.
MAX_SIDELOBES = 28
DEFAULT_SIDELOBES = 2
SIDELOBE_PRESETS = {
    "Main lobe only (±1.02 MHz)": 0,
    "Main + 1 sidelobe (±2.05 MHz)": 1,
    "Main + 2 sidelobes (±3.07 MHz)": 2,
    "Main + 3 sidelobes (±4.09 MHz)": 3,
    "Main + 5 sidelobes (±6.14 MHz)": 5,
    "Main + 10 sidelobes (±11.25 MHz)": 10,
}


# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# A C/A signal is a BPSK(1) sinc² spectrum, so its whole power distribution is fixed by ONE
# measured number: the power spectral DENSITY at the main-lobe PEAK, in dBm/Hz. From it,
#
#   • Main-lobe integrated power (dBm) = peak_dBm/Hz + 10·log10(Rc · I_ML)      ← CONSTANT
#   • Full signal power (dBm)          = peak_dBm/Hz + 10·log10(Rc · frac(n))    ← tracks --sidelobes
#
# Rc = 1.023e6 Hz (chip rate); I_ML = 0.902823 is the fraction of the signal's total power inside
# the main lobe (±Rc); frac(n) is the fraction inside the filter passband (±(n+1)·Rc), computed
# below by integrating sinc². The full power is measured through the SAME filter the operator
# transmits with, so it IS the amplifier's limiting quantity. NOTE: because 90.3% of a C/A
# signal's power is already in the main lobe, the full power exceeds the main-lobe power by AT
# MOST 0.44 dB — they are EQUAL at n=0 (passband = main lobe) and diverge by only frac(n)/I_ML.

def _sinc2(x: float) -> float:
    """sinc²(x) with sinc(x) = sin(πx)/(πx); the normalized C/A power-spectral shape."""
    if x == 0.0:
        return 1.0
    s = math.sin(math.pi * x) / (math.pi * x)
    return s * s


def _power_fraction_table(nmax: int, step: float = 1e-3) -> tuple:
    """frac(n) = 2·∫₀^(n+1) sinc²(x) dx for n = 0..nmax — the fraction of the signal's total
    power within ±(n+1) chip-rates (the passband for n kept sidelobes). Pure-Python trapezoid
    so the module imports without numpy (numpy is only needed to build the IQ loop)."""
    frac = {}
    acc, prev = 0.0, _sinc2(0.0)
    per = int(round(1.0 / step))                    # samples per unit x; boundaries hit integers
    for i in range(1, (nmax + 1) * per + 1):
        cur = _sinc2(i * step)
        acc += 0.5 * (prev + cur) * step
        prev = cur
        if i % per == 0:                            # x == an integer == (n+1)
            frac[i // per - 1] = 2.0 * acc
    return tuple(frac[n] for n in range(nmax + 1))


_POWER_FRACTION = _power_fraction_table(MAX_SIDELOBES)   # frac(0..MAX_SIDELOBES)
CA_MAIN_LOBE_FRACTION = _POWER_FRACTION[0]               # I_ML ≈ 0.902823


def enbw_mhz(sidelobes: int) -> float:
    """The equivalent-noise bandwidth (MHz) mapping the measured PEAK density to the FULL power
    passed by the filter with `sidelobes` sidelobes: full_dBm = peak_dBm/Hz + 10·log10(enbw·1e6).
    Equals Rc·frac(n); passed live to the power map so the delivered power and the limiting cap
    both track the sidelobe count as it is tuned."""
    n = max(0, min(MAX_SIDELOBES, int(sidelobes)))
    return (CODE_RATE_HZ / 1e6) * _POWER_FRACTION[n]


# The power-quantity conversion laws this signal OFFERS the calibration editor. Both convert the
# measured spectral density (dBm/Hz at the peak) to an absolute power (dBm). The operator picks
# which is --power (and sets the FULL power as the limiting cap) per unit; the chosen law is
# embedded in that unit's calibration doc. Constants are LITERAL (the agent reads CAL_POWER_LAWS
# statically): 60 = 10·log10(1 MHz / 1 Hz); the full-power term adds 10·log10(enbw_mhz); the
# main-lobe k = 10·log10(Rc · I_ML) = 59.654784. `rep` = enbw_mhz(DEFAULT_SIDELOBES) for the
# range read-outs shown before a live --sidelobes is known.
CAL_POWER_LAWS = [
    {"id": "full_power", "name": "Full signal power (filter passband)", "unit": "dBm",
     "in": "density", "out": "abs",
     "k": 60.0, "param": "enbw_mhz", "coeff": 10.0, "ref": 1.0, "rep": 0.988638},
    {"id": "main_lobe_power", "name": "Main-lobe integrated power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 59.654784},
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


# GPS ICD-200 Table 3-Ia: G2 code-phase tap pairs (1-indexed) selecting each PRN's C/A code.
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
# ICD reference: first 10 chips of each PRN's C/A code, octal — used only by --self-test.
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── C/A Gold-code generation (pure Python, no NumPy) ────────────────────────────────

def ca_code(prn: int) -> list[int]:
    """The 1023-chip GPS C/A Gold code for a PRN (1..32) as 0/1. Two 10-stage LFSRs
    (seeded all-ones): G1 = x^10+x^3+1, G2 = x^10+x^9+x^8+x^6+x^3+x^2+1; the chip is G1's
    output XOR two PRN-specific G2 taps."""
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


# ── Baseband buffer (one seamless-looping code period) ──────────────────────────────

def build_iq_buffer(prn: int):
    """The unit-magnitude complex64 C/A buffer at SAMP_RATE_HZ: a whole number of code
    periods that is an exact integer sample count, so it loops with no seam. BPSK: I = ±1,
    Q = 0 (amplitude is applied downstream). Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(SAMP_RATE_HZ))
    cr = int(round(CODE_RATE_HZ))
    spp = Fraction(sr * CODE_LEN, cr)              # samples per code period, exact
    n_periods = spp.denominator
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code                     # 0 → +1, 1 → −1
    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr // sr) % CODE_LEN           # exact zero-order-hold chip mapping
    return bipolar[chip_idx].astype(np.complex64), n_samples, n_periods


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. `fc_hz` is the −6 dB
    cutoff; `trans_hz` sets the tap count (steeper skirt → more taps). Returns (h, n_taps)."""
    import numpy as np
    m = int(np.ceil(5.5 * SAMP_RATE_HZ / max(trans_hz, 1.0))) | 1     # odd
    m = min(m, (max_taps | 1))
    k = np.arange(m)
    c = (m - 1) / 2.0
    fcn = fc_hz / SAMP_RATE_HZ
    h = 2 * fcn * np.sinc(2 * fcn * (k - c))
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n1)
           + 0.14128 * np.cos(4 * np.pi * k / n1) - 0.01168 * np.cos(6 * np.pi * k / n1))
    h = h * win
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, sidelobes: int, trans_hz: float, base_fft=None):
    """Circularly filter the looped C/A buffer to keep the main lobe + `sidelobes` sidelobes.
    Circular convolution (multiply the buffer's DFT by the filter's) keeps the result exactly
    periodic, so the filtered loop has no seam; unity passband gain leaves the kept lobes'
    power unchanged. Pass `base_fft` (= np.fft.fft(base_iq)) to reuse it across live filter
    changes — the base loop is fixed per run, so its DFT need only be computed once, which
    cuts the per-change CPU spike (and the underflows it can cause). Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * CA_NULL_HZ          # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                         # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    if base_fft is None:
        base_fft = np.fft.fft(base_iq)
    filtered = np.fft.ifft(base_fft * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test (Gold code always; filter check when numpy is present) ────────────────

def _self_test() -> int:
    ok = True
    for prn in range(1, 33):
        code = ca_code(prn)
        first10 = 0
        for b in code[:10]:
            first10 = (first10 << 1) | b
        good = (len(code) == CODE_LEN and first10 == _FIRST10_OCTAL[prn] and sum(code) == 512)
        ok = ok and good
        print(f"PRN {prn:2d}: first10={first10:#06o} expect={_FIRST10_OCTAL[prn]:#06o} "
              f"ones={sum(code)} [{'OK' if good else 'FAIL'}]")
    print("Gold code: ALL PRN CHECKS PASSED" if ok else "Gold code: SELF-TEST FAILED")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n, _ = build_iq_buffer(1)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=2, trans_hz=0.5e6)
    main = 10 * np.log10(band(filt, 0, CA_NULL_HZ) / band(base, 0, CA_NULL_HZ))
    kept = 10 * np.log10(band(filt, 2 * CA_NULL_HZ, 3 * CA_NULL_HZ)
                         / band(base, 2 * CA_NULL_HZ, 3 * CA_NULL_HZ))
    cut = 10 * np.log10(band(filt, 10e6, 12e6) / band(base, 10e6, 12e6))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+2 sidelobes, {taps} taps): main lobe {main:+.3f} dB, kept sidelobe "
          f"{kept:+.3f} dB, far sidelobe {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} "
          f"[{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    # Calibration power fractions / laws (pure math — the values baked into CAL_POWER_LAWS).
    fr = _POWER_FRACTION
    mono = all(fr[i] < fr[i + 1] for i in range(len(fr) - 1))
    bounded = 0.9025 < fr[0] < 0.9035 and fr[-1] < 1.0
    # full_power(0) must equal the main-lobe k: with 0 sidelobes the passband IS the main lobe,
    # so the full power passed by the filter is exactly the main-lobe integrated power.
    full0 = 60.0 + 10 * math.log10(enbw_mhz(0))
    laws_ok = mono and bounded and abs(full0 - 59.654784) < 0.01
    print(f"calibration: I_ML={fr[0]:.6f}, frac(max)={fr[-1]:.6f}, full(0)={full0:.4f} dB "
          f"== main-lobe 59.6548 dB, span main→full ≤ {10*math.log10(1/fr[0]):.3f} dB "
          f"[{'OK' if laws_ok else 'FAIL'}]")
    ok = ok and laws_ok
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float, amplitude: float):
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

    class PrnTx(gr.top_block):
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

    return PrnTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} (C/A Gold code) transmitter — fixed 61.38 MHz / sc8, looped "
               "buffer, always-on power-preserving digital passband filter set to an integer "
               "number of sidelobes. Level is set in dBm via the unit's calibration "
               "(spectral density → full / main-lobe power); uncalibrated it runs on a relative gain.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=CARRIER_HZ / 1e6,
                help="RF carrier in MHz (default L1 = 1575.42). Fixed per run.")
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
                 help="GPS satellite PRN / Gold code index (1..32). Fixed per run.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES,
                 default=DEFAULT_SIDELOBES, presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of C/A sidelobes KEPT beside the main "
                      "lobe: a ±(n+1)·1.023 MHz band. The filter is always on (unity passband "
                      "gain). More sidelobes pass more of the signal's power (the full-power "
                      "calibration quantity tracks this). Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.5,
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

    # Filter "shape" (the regeneration-requiring params) — mutated by live changes. The filter
    # is ALWAYS on; only its width (sidelobes) and skirt (transition) vary. Defined before the
    # gain map so the calibration's power laws can read the current equivalent bandwidth.
    shape = {"sidelobes": int(getattr(args, "sidelobes", DEFAULT_SIDELOBES) or 0),
             "trans_hz": float(getattr(args, "transition", 0.5) or 0.5) * 1e6}

    def pwr_params() -> dict:
        """The live keyed-parameter values the calibration's power laws read: the filter's
        equivalent-noise bandwidth, so the FULL-power reading and its limiting cap track the
        sidelobe count (the client folds the --power range at the same value). Harmless when
        the unit is uncalibrated or its laws don't key on it."""
        return {"enbw_mhz": enbw_mhz(shape["sidelobes"])}

    # Gain precedence: explicit --gain (raw) > calibrated --power > refuse (uncalibrated).
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz, params=pwr_params())
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    # Prebuild the unfiltered loop once (PRN is fixed per run); the always-on filter derives from it.
    base_iq, nsamp, nper = build_iq_buffer(args.prn)

    base_fft = {"v": None}      # DFT of the fixed base loop — computed once, reused per change

    def make_current(report=False):
        """The circularly-filtered loop for the current shape (the filter is always on).
        Returns (iq, info) where info describes the filter for the banner/report."""
        if base_fft["v"] is None:
            import numpy as np
            base_fft["v"] = np.fft.fft(base_iq)
        filtered, taps, fp = filter_buffer(base_iq, shape["sidelobes"], shape["trans_hz"],
                                           base_fft=base_fft["v"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}

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

    # Track the held absolute --power target (calibrated mode only) so a live --sidelobes change
    # can re-map the gain: the FULL-power quantity moves with the sidelobe count, so keeping the
    # same delivered power needs a new gain (a main-lobe/relative target is unaffected — calkit
    # handles which, via the embedded law).
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "power": (float(args.power) if (gain_cal is None and pmap.has_absolute
                                             and getattr(args, "power", None) is not None)
                       else None)}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def _fmt_band(info):
        return (f"on — main + {info['sidelobes']} sidelobe(s) "
                f"(±{info['edge_hz']/1e6:.2f} MHz), {info['trans_hz']/1e6:g} MHz transition, "
                f"{info['taps']} taps")

    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  PRN            : {args.prn}")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  code rate      : 1.023 Mcps (~2.046 MHz null-to-null), loop {nsamp} samples")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params()):.2f} dBm")
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
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(float(value), freq=center_freq_hz,
                                                params=pwr_params())
            if state["rf_on"]:
                tb.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=center_freq_hz,
                                                           params=pwr_params()), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            state["power"] = None                      # raw gain takes over the level
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
        elif name in ("sidelobes", "transition"):
            if name == "sidelobes":
                shape["sidelobes"] = max(0, min(MAX_SIDELOBES, int(value)))
            else:
                shape["trans_hz"] = float(value) * 1e6
            regenerate()
            # Widening/narrowing the passband changes the equivalent bandwidth, so a held
            # absolute --power must re-map to keep the delivered power (full-power quantity)
            # constant; the amp's limiting cap moves with it too. calkit no-ops this for a
            # bandwidth-independent (main-lobe) or relative target.
            if name == "sidelobes" and state["power"] is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=center_freq_hz,
                                                    params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(
                    state["gain"], freq=center_freq_hz, params=pwr_params()), 2))
            ctrl.report(name, value)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    # No watchdog: the in-RAM vector_source_c has no file to short-read, so the "fread error"
    # that silently killed the source (and needed a self-heal restart) can't happen. The graph
    # runs until stopped; live changes retune or swap the loop buffer in place.
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

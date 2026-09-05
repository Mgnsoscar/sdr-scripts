#!/usr/bin/env python3
"""
GPS L2C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a **bit-exact** GPS **L2C** signal (1227.60 MHz): the civil signal on L2,
the chip-by-chip time-multiplex of two 511.5 kcps codes interleaved to 1.023 Mcps
(BPSK-R(1), ~2 MHz wide) — L2 CM (10230 chips, 20 ms) and L2 CL (767250 chips,
1.5 s, dataless pilot). Prebuilt once and looped so a Raspberry Pi sustains the
rate with no runtime IQ math (same recipe as gps_l1ca_tx.py).

⚠  RF SAFETY / LEGAL: L2 (1227.60 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never over the air.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 60 samples/combined-chip, exact), master clock 1:1;
  • over-the-wire sc8 (constant-modulus BPSK loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.
L2C is only ~2 MHz wide, but the rate is pinned high (like C/A) so the full sinc²
sidelobe skirt is present for the digital filter below to shape.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid (the SDR
gain step and any active-component steps), and the banner reports the power actually
achieved. --gain instead commands the raw SDR gain (relative), overriding --power.
Uncalibrated, there is no dBm scale — use --gain. (See docs/calibration-v2.md.)

Loop length (--loop)
────────────────────
  full (default) : one whole CL period = 1.5 s (CM repeats 75×). Bit-exact CL phase and
                 complete spectrum. ~736 MB held in RAM at 61.38 MHz (the loop must live in
                 memory to stream at rate), so the host needs the headroom. The live filter
                 stays responsive because it uses a memory-bounded overlap-add convolution (no
                 92-Msample monolithic FFT) and the flowgraph keeps looping the old buffer
                 until the new one is ready.
  cm             : one CM period = 20 ms (CL truncated to its first 10230 chips), ~9.8 MB.
                 The BPSK-R(1) envelope is identical; the CL line structure appears at 50 Hz
                 rather than its true 0.667 Hz (both unresolvable at practical RBW). Pick it
                 when RAM is tight or the true CL phase does not matter.

Digital passband filter (ALWAYS ON — on the looped buffer, no runtime DSP)
─────────────────────────────────────────────────────────────────────────
An always-on steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power: if the main lobe measures
−2.5 dBm it reads −2.5 dBm filtered — the filter only removes what's outside the passband.
L2C's spectrum is the same sinc² as C/A (nulls every 1.023 MHz), so the passband is ALWAYS an
integer number of sidelobes, which is what makes the emitted power a well-defined function of
the sidelobe count (see the calibration note below).
  • --sidelobes <n>             passband keeps the main lobe + n sidelobes, i.e. a
                                ±(n+1)·1.023 MHz band (live, a 0..28 slider).
The skirt transition width is FIXED at 0.05 MHz (not a knob), so the emitted power stays a
well-defined function of the sidelobe count alone. --sidelobes is LIVE: changing it rebuilds
the (circularly-)filtered loop and swaps it into the running source; the flowgraph never stops.

Spectral-density calibration (dBm/Hz at the main-lobe peak → power quantities)
─────────────────────────────────────────────────────────────────────────────
L2C (CM/CL multiplexed to a combined 1.023 Mcps BPSK) is a sinc² power spectrum, so its whole
distribution is fixed by ONE measured number: the power spectral DENSITY at the main-lobe PEAK,
in dBm/Hz. From it CAL_POWER_LAWS derives two absolute-power quantities the operator can pick
between for --power: the MAIN-LOBE integrated power, and the FULL signal power passed by the
filter (which grows with the sidelobe count and is the amplifier's LIMITING quantity). See
CAL_POWER_LAWS below and docs/calibration-v2.md §13.

CLI
───
    gps_l2c_tx.py --prn 5 --power -30                          # calibrated dBm (main + 2 sidelobes)
    gps_l2c_tx.py --prn 5 --gain 60 --sidelobes 5             # relative gain, main + 5 sidelobes
    gps_l2c_tx.py --loop full --prn 5 --power -30             # bit-exact 1.5 s CL (big/slow)
    gps_l2c_tx.py --self-test
    gps_l2c_tx.py --describe-params
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
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l2c"

# Which parameter carries the transmit frequency, so a frequency-dependent calibration chain
# folds --power at the live carrier (see the C/A scripts / docs/calibration-v2.md).
CAL_FREQ_PARAM = "freq"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 60 samples/combined-chip at 1.023 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; BPSK is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L2C) ──────────────────────────────────────
L2_HZ = 1227.60e6                # GPS L2
CM_LEN = 10230                   # L2 CM code length (20 ms at 511.5 kcps)
CL_LEN = 767250                  # L2 CL code length (1.5 s at 511.5 kcps)
CHANNEL_CHIP_RATE = 511_500      # each of CM / CL
COMBINED_CHIP_RATE = 1_023_000   # after chip-by-chip multiplexing (== sinc² null spacing)
SIGNAL_NAME = "GPS L2C"
L2C_NULL_HZ = 1.023e6            # main-lobe null spacing == the chip rate; sidelobes step by this
LFSR_MASK = 0o445112474          # IS-GPS-200 L2C feedback polynomial (Galois)

FREQUENCIES = {"GPS L2 (1227.60 MHz)": L2_HZ / 1e6}   # presets are in MHz now

# Filter passband width, in sidelobes KEPT beside the main lobe: the passband is the main lobe
# plus that many sidelobes, i.e. a ±(n+1)·1.023 MHz band. The maximum keeps the whole passband
# (edge + skirt) inside ±Fs/2: with the fixed 0.05 MHz transition, n = 28 is the largest that fits
# ((28+1)·1.023 + 0.025 = 29.7 MHz < 30.69 = Fs/2).
MAX_SIDELOBES = 28
DEFAULT_SIDELOBES = 2
L2C_NULL_MHZ = 1.023             # sidelobe/main-lobe null spacing (MHz) == the chip rate
SIDELOBE_PRESETS = {
    "Main lobe only (±1.02 MHz)": 0,
    "Main + 1 sidelobe (±2.05 MHz)": 1,
    "Main + 2 sidelobes (±3.07 MHz)": 2,
    "Main + 3 sidelobes (±4.09 MHz)": 3,
    "Main + 5 sidelobes (±6.14 MHz)": 5,
    "Main + 10 sidelobes (±11.25 MHz)": 10,
}

# Filter skirt transition width beyond the passband edge (MHz) — FIXED. The passband is always
# an integer number of sidelobes; the skirt is a constant so the emitted power stays a
# well-defined function of the sidelobe count alone (not the operator's discretion).
TRANSITION_MHZ = 0.05
TRANS_HZ = TRANSITION_MHZ * 1e6


# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# L2C is a BPSK(1) sinc² spectrum, so its whole power distribution is fixed by ONE measured
# number: the power spectral DENSITY at the main-lobe PEAK, in dBm/Hz. From it,
#
#   • Main-lobe integrated power (dBm) = peak_dBm/Hz + 10·log10(Rc · I_ML)      ← CONSTANT
#   • Full signal power (dBm)          = peak_dBm/Hz + 10·log10(Rc · frac(n))    ← tracks --sidelobes
#
# Rc = 1.023e6 Hz (combined chip rate = null spacing); I_ML = 0.902823 is the fraction of the
# signal's total power inside the main lobe (±Rc); frac(n) is the fraction inside the filter
# passband (±(n+1)·Rc), computed below by integrating sinc². The full power is measured through the
# SAME filter the operator transmits with, so it IS the amplifier's limiting quantity. Because
# 90.3% of the power is already in the main lobe, the full power exceeds the main-lobe power by AT
# MOST 0.44 dB — EQUAL at n=0 (passband = main lobe) and diverging only by frac(n)/I_ML. Shape
# identical to the C/A 1.023 signal — so enbw and the main-lobe k match it exactly.

def _sinc2(x: float) -> float:
    """sinc²(x) with sinc(x) = sin(πx)/(πx); the normalized sinc² power-spectral shape."""
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
L2C_MAIN_LOBE_FRACTION = _POWER_FRACTION[0]              # I_ML ≈ 0.902823
I_ML = 0.902823                                         # sinc² power fraction within the main lobe


def enbw_mhz(sidelobes: int) -> float:
    """The equivalent-noise bandwidth (MHz) mapping the measured PEAK density to the FULL power
    passed by the filter with `sidelobes` sidelobes: full_dBm = peak_dBm/Hz + 10·log10(enbw·1e6).
    Equals Rc·frac(n); passed live to the power map so the delivered power and the limiting cap
    both track the sidelobe count as it is tuned."""
    n = max(0, min(MAX_SIDELOBES, int(sidelobes)))
    return (COMBINED_CHIP_RATE / 1e6) * _POWER_FRACTION[n]


# Static enbw_mhz(n) lookup for the GUI. The client's schema extractor is a static AST reader
# (it can't run the sinc² integration above), and the full-power law keys on enbw_mhz — a value
# with no input field — so the schema exposes it as a HIDDEN derived field whose formula is a
# nearest-integer table lookup on --sidelobes. The first element names the source field; the
# rest are enbw_mhz(0..MAX_SIDELOBES). Kept a literal so the extractor can read it; --self-test
# asserts it matches enbw_mhz() so the two can never silently drift.
_ENBW_TABLE_ARGS = [
    "sidelobes",
    0.923588, 0.971788, 0.988638, 0.997168, 1.002311, 1.005749, 1.008208, 1.010054,
    1.011490, 1.012640, 1.013581, 1.014365, 1.015029, 1.015598, 1.016091, 1.016523,
    1.016904, 1.017242, 1.017545, 1.017818, 1.018065, 1.018289, 1.018494, 1.018682,
    1.018854, 1.019014, 1.019161, 1.019298, 1.019426,
]


# The power-quantity conversion laws this signal OFFERS the calibration editor. Both convert the
# measured spectral density (dBm/Hz at the peak) to an absolute power (dBm). The operator picks
# which is --power (and sets the FULL power as the limiting cap) per unit. Constants are LITERAL
# (the agent reads CAL_POWER_LAWS statically): 60 = 10·log10(1 MHz / 1 Hz); the full-power term
# adds 10·log10(enbw_mhz); the main-lobe k = 10·log10(Rc · I_ML) = 59.654784. `rep` =
# enbw_mhz(DEFAULT_SIDELOBES) for the range read-outs shown before a live --sidelobes is known.
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

def _lfsr_bits(init: int, n: int) -> list[int]:
    """n output chips (LSB each) of the L2C register started at `init`."""
    x = init
    out = [0] * n
    for i in range(n):
        out[i] = x & 1
        x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
    return out


def cm_code(prn: int) -> list[int]:
    """L2 CM code (10230 chips, 0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CM_INIT[prn - 1], CM_LEN)


def cl_code(prn: int, n: int = CL_LEN) -> list[int]:
    """L2 CL code (n chips, 0/1) for a PRN (1..63); n<CL_LEN truncates it."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CL_INIT[prn - 1], n)


# ── Baseband buffer (CM/CL time-multiplexed, seamless loop) ────────────────────

def build_l2c_buffer(prn: int, loop: str):
    """The complex64 L2C baseband buffer at SAMP_RATE_HZ (real BPSK, Q=0). loop='cm' →
    one 20 ms CM period (CL truncated); loop='full' → one 1.5 s CL period. Each is a whole
    number of combined chips that is an exact integer sample count, so it loops with no
    seam. Filled in chunks so the full 92-Msample loop needs only its own storage plus a
    small working set (not several int64 copies of it). Returns (iq, n_samples)."""
    import numpy as np

    n_cl = CL_LEN if loop == "full" else CM_LEN
    cm = np.asarray(cm_code(prn), dtype=np.int8)
    cl = np.asarray(cl_code(prn, n_cl), dtype=np.int8)

    period_s = n_cl / CHANNEL_CHIP_RATE          # 1.5 s (full) or 20 ms (cm)
    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(period_s * sr))

    out = np.empty(n_samples, dtype=np.complex64)
    chunk = 1 << 22                              # ~4 M samples/pass → bounded working set
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        n = np.arange(s, e, dtype=np.int64)
        gchip = n * COMBINED_CHIP_RATE // sr     # 0 .. 2*n_cl-1
        half = gchip >> 1
        is_cl = (gchip & 1) == 1
        bit = np.where(is_cl, cl[half % n_cl], cm[half % CM_LEN])
        out[s:e] = (1.0 - 2.0 * bit).astype(np.complex64)
    return out, n_samples


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


def _circular_convolve(x, h):
    """Circular convolution of period len(x) between complex `x` and real FIR `h` (len ≤ len(x)).
    For a short filter on a huge loop (the 1.5 s CL buffer is ~92 M samples at 61.38 MHz) a single
    monolithic DFT would need several GB, so this uses OVERLAP-ADD with a small block FFT and then
    aliases the (M−1)-sample linear-convolution tail back to the head — which is exactly what makes
    the result circular, so the filtered loop still repeats with no seam. Peak memory is one
    complex64 copy of the loop plus O(block), not O(len·16 bytes)."""
    import numpy as np
    n = len(x)
    m = len(h)
    if m >= n:                                        # tiny loop — a direct DFT is fine
        return np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, n)).astype(np.complex64)
    nfft = 1
    while nfft < 4 * m:                               # comfortably larger than the filter
        nfft <<= 1
    step = nfft - (m - 1)                             # samples consumed per block
    hf = np.fft.fft(h, nfft)
    y = np.zeros(n, dtype=np.complex64)               # accumulator (bounded memory)
    for start in range(0, n, step):
        blk = x[start:start + step]
        yb = np.fft.ifft(np.fft.fft(blk, nfft) * hf)  # linear conv of this block (len blk + m − 1)
        yb = yb[:len(blk) + m - 1]
        end = start + len(yb)
        if end <= n:
            y[start:end] += yb
        else:                                         # wrap the tail → circular aliasing
            first = n - start
            y[start:n] += yb[:first]
            y[0:len(yb) - first] += yb[first:]
    return y


def filter_buffer(base_iq, sidelobes: int, trans_hz: float):
    """Circularly filter the looped L2C buffer to keep the main lobe + `sidelobes` sidelobes.
    Circular convolution keeps the result exactly periodic, so the filtered loop has no seam;
    unity passband gain leaves the kept lobes' power unchanged. Returns (filtered_iq, n_taps,
    passband_edge_hz)."""
    fp = (int(sidelobes) + 1) * L2C_NULL_HZ          # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                          # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    return _circular_convolve(base_iq, h), m, fp


# ── Self-test (generator vs official sheet + check values; filter when numpy present) ──

def _self_test() -> int:
    ok = True

    def run(init, steps):
        x = init
        for _ in range(steps):
            x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
        return x

    # Ground truth from the official L2C PRN Code Assignments sheet (PRN 159/160):
    # register run from Initial State reaches End State at the last chip.
    sheet = [  # (cm_init, cm_end, cl_init, cl_end)
        (0o604055104, 0o425373114, 0o605253024, 0o44547544),    # PRN 159
        (0o157065232, 0o427153064, 0o63314262, 0o707116115),    # PRN 160
    ]
    for i, (cmi, cme, cli, cle) in enumerate(sheet):
        cm_ok = run(cmi, CM_LEN - 1) == cme
        ok = ok and cm_ok
        print(f"sheet PRN{159+i} CM init→end ({CM_LEN} chips): [{'OK' if cm_ok else 'FAIL'}]")
    cl_ok = run(sheet[0][2], CL_LEN - 1) == sheet[0][3]
    ok = ok and cl_ok
    print(f"sheet PRN159 CL init→end ({CL_LEN} chips): [{'OK' if cl_ok else 'FAIL'}]")

    # first-24-chip octal check values (transcription guard for the init tables).
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
    len_ok = len(cm1) == CM_LEN
    print(f"CM len={len(cm1)} ones={sum(cm1)} (≈5115) [{'OK' if len_ok else 'FAIL'}]")
    ok = ok and len_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_l2c_buffer(1, "cm")

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=2, trans_hz=TRANS_HZ)
    main = 10 * np.log10(band(filt, 0, L2C_NULL_HZ) / band(base, 0, L2C_NULL_HZ))
    kept = 10 * np.log10(band(filt, 2 * L2C_NULL_HZ, 3 * L2C_NULL_HZ)
                         / band(base, 2 * L2C_NULL_HZ, 3 * L2C_NULL_HZ))
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
    bounded = 0.9025 < fr[0] < 0.9035 and fr[-1] < 1.05
    # full_power(0) must equal the main-lobe k: with 0 sidelobes the passband IS the main lobe,
    # so the full power passed by the filter is exactly the main-lobe integrated power.
    full0 = 60.0 + 10 * math.log10(enbw_mhz(0))
    # The GUI's hidden enbw_mhz derived field is a STATIC literal table (_ENBW_TABLE_ARGS); the
    # runtime computes the same values via enbw_mhz(). Assert they can't have drifted.
    table_ok = (len(_ENBW_TABLE_ARGS) == MAX_SIDELOBES + 2
                and _ENBW_TABLE_ARGS[0] == "sidelobes"
                and all(abs(_ENBW_TABLE_ARGS[1 + n] - enbw_mhz(n)) < 5e-6
                        for n in range(MAX_SIDELOBES + 1)))
    laws_ok = mono and bounded and abs(full0 - 59.654784) < 0.01 and table_ok
    print(f"calibration: I_ML={fr[0]:.6f}, frac(max)={fr[-1]:.6f}, full(0)={full0:.4f} dB "
          f"== main-lobe 59.6548 dB, span main→full ≤ {10*math.log10(1/fr[0]):.3f} dB, "
          f"enbw table {'matches' if table_ok else 'DRIFTED'} "
          f"[{'OK' if laws_ok else 'FAIL'}]")
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

    class L2CTx(gr.top_block):
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

    return L2CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (CM/CL time-multiplexed, real IS-GPS-200 codes, "
               "1.023 Mcps BPSK) — fixed 61.38 MHz / sc8, looped buffer, always-on "
               "power-preserving digital passband filter set to an integer number of sidelobes. "
               "Level is set in dBm via the unit's calibration (spectral density → full / "
               "main-lobe power); uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=L2_HZ / 1e6,
                help="RF carrier in MHz (default L2 = 1227.60). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63) — the real L2C code. Fixed per run.")
        .choice("-Loop", "--loop", options=["full", "cm"], default="full",
                help="full = whole 1.5 s CL period (bit-exact CL phase, complete spectrum; "
                     "~736 MB in RAM at 61.38 MHz); cm = 20 ms CM period (CL truncated; "
                     "~9.8 MB, envelope-correct) for tight RAM.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, step=1,
                 default=DEFAULT_SIDELOBES, presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of sidelobes KEPT beside the main lobe: "
                      "a ±(n+1)·1.023 MHz band. 0 keeps the main lobe only. The filter is always "
                      "on (unity passband gain). More sidelobes pass more of the signal's power "
                      "(the full-power calibration quantity tracks this). Max 28 keeps the band "
                      "inside the sample rate. Live (rebuilds the filtered loop).")
        .derived("-Passband-bandwidth", name="passband_bw_mhz", unit="MHz",
                 formula={"linear": ["sidelobes", 2.046, 2.046]},
                 help="Occupied bandwidth the filter passes at the current sidelobe count: "
                      "2·(n+1)·1.023 MHz (i.e. ±(n+1)·1.023 MHz).")
        .derived("-Full-power-bandwidth", name="enbw_mhz", unit="MHz", hidden=True,
                 formula={"table": _ENBW_TABLE_ARGS},
                 help="Equivalent-noise bandwidth mapping the measured peak density to the full "
                      "in-band power. Feeds the full-power calibration law; not shown.")
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
    # is ALWAYS on; only its width (sidelobes) varies (the skirt transition is fixed). Defined
    # before the gain map so the calibration's power laws can read the current equivalent
    # bandwidth.
    shape = {"sidelobes": int(getattr(args, "sidelobes", DEFAULT_SIDELOBES) or 0),
             "trans_hz": TRANS_HZ}

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

    # Prebuild the unfiltered loop once (PRN/loop are fixed per run); the always-on filter
    # derives from it.
    base_iq, nsamp = build_l2c_buffer(args.prn, args.loop)

    def make_current():
        """The circularly-filtered loop for the current shape (the filter is always on).
        Returns (iq, info) where info describes the filter for the banner/report."""
        filtered, taps, fp = filter_buffer(base_iq, shape["sidelobes"], shape["trans_hz"])
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
    print(f"  satellite PRN  : {args.prn}  (real L2C code, CM+CL)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : BPSK-R(1) — 1.023 Mcps (CM/CL @ 511.5 kcps each)")
    print(f"  loop           : {args.loop} "
          f"({'1.5 s (bit-exact CL)' if args.loop=='full' else '20 ms (CL truncated)'}), "
          f"{nsamp} samples ({nsamp*8/1e6:.1f} MB)")
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
        elif name == "sidelobes":
            shape["sidelobes"] = max(0, min(MAX_SIDELOBES, int(value)))
            regenerate()
            # Widening/narrowing the passband changes the equivalent bandwidth, so a held
            # absolute --power must re-map to keep the delivered power (full-power quantity)
            # constant; the amp's limiting cap moves with it too. calkit no-ops this for a
            # bandwidth-independent (main-lobe) or relative target.
            if state["power"] is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=center_freq_hz,
                                                    params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(
                    state["gain"], freq=center_freq_hz, params=pwr_params()), 2))
            ctrl.report("sidelobes", shape["sidelobes"])

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

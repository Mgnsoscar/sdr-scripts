#!/usr/bin/env python3
"""
GPS L1C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a spectrally-correct GPS **L1C** signal (1575.42 MHz) — the modernized civil
L1 signal — as the in-phase sum of its pilot and data components:

    L1Cp (pilot, 75% power) : Weil code × TMBOC(6,1,4/33) subcarrier × overlay
    L1Cd (data,  25% power) : Weil code × BOC(1,1) subcarrier   (bare code here)

Prebuilt once and looped so a Raspberry Pi sustains the rate with no runtime IQ math
(same recipe as gps_l1ca_tx.py).

Code fidelity — real IS-GPS-800 Weil codes
──────────────────────────────────────────
Both 10230-chip primary codes are the real IS-GPS-800 codes: Legendre (mod 10223)
→ Weil W[k]=L[k]⊕L[k+w] → 7-chip insertion [0110100] at index p. Per-PRN (w,p) for
pilot and data were validated against the official L1C PRN Code Assignments sheet.
The pilot **overlay** (1800-symbol secondary, 18 s period) is the single 11-bit
LFSR of IS-GPS-800; its per-PRN polynomial/init table was cross-checked against the
sheet's L1CO columns.

Full-length overlay WITHOUT a multi-GB file
───────────────────────────────────────────
The overlay is a *slow* code: one ±1 symbol per 10 ms primary period, so the 18 s
tiered pilot is the 10 ms pilot buffer replayed 1800 times with a per-period sign flip.
Rather than precompute 18 s, the flow applies it at runtime:

    pilot loop ─► × ─────────┐
    overlay(1800)─►repeat(N)─┘ ├─► + ─► (amp) ─► USRP
    data loop  ──────────────┘

so the full 18 s signal streams from ~10 MB. `--secondary off` drops it (10 ms primary
loop only; spectrally identical, no secondary sync).

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 60 samples/chip; the BOC(6,1) subcarrier lands 10 samples/
    cycle, exact), master clock pinned 1:1;
  • over-the-wire sc8 (halves USB load; the small peak factor survives 8-bit here);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid, and the
banner reports the power actually achieved. --gain instead commands the raw SDR gain
(relative), overriding --power. Uncalibrated, there is no dBm scale — use --gain.

Digital passband filter (ALWAYS ON — on the looped buffers, no runtime DSP)
──────────────────────────────────────────────────────────────────────────
An always-on steep FIR passband, applied to the PRECOMPUTED loops by CIRCULAR convolution, so
the filtered buffers still loop with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power. The same filter is applied to
the pilot and data component buffers (filtering is linear, and the overlay is a ±1 per-period
sign that commutes with it), so the summed signal is filtered identically. L1C is a split (BOC)
signal — BOC(1,1) lobes at ±1.023 MHz, TMBOC BOC(6,1) lobes at ±6.138 MHz — whose spectral
nulls sit at every 1.023 MHz. The BOC(1,1) core is bounded by the ±2.046 MHz null, so the
passband edge SNAPS TO THE NULLS as: keep the core + `sidelobes` further null-steps.
  • --sidelobes <n>             passband keeps the BOC(1,1) core + n further null-steps
                                (each 1.023 MHz), i.e. a ±(n+2)·1.023 MHz band (live, presets).
                                0 = the BOC(1,1) core only (±2.046 MHz); 5 = the full TMBOC
                                (±7.16 MHz, just outside the BOC(6,1) lobes); 28 = ±30.69 =
                                ±Fs/2 (the whole representable signal).
The skirt transition width is FIXED at 0.05 MHz (not a knob), so the emitted power stays a
well-defined function of the sidelobe count alone. --sidelobes is LIVE: changing it rebuilds
the (circularly-)filtered loops and swaps them into the running sources; the flow never stops.

Spectral-density calibration (dBm/Hz at the main-lobe peak → power quantities)
─────────────────────────────────────────────────────────────────────────────
L1C is a SPLIT (BOC) spectrum; the calibration is on the BOC(1,1) CORE — two main lobes at
±1.023 MHz, bounded by the DC null and the ±2.046 MHz null. Its distribution is fixed by ONE
measured number — the power spectral DENSITY at the (BOC(1,1) core) main-lobe PEAK, in dBm/Hz.
From it CAL_POWER_LAWS derives two absolute-power quantities the operator can pick between for
--power: the CORE main-lobes integrated power, and the FULL signal power passed by the filter
(which grows with the sidelobe count and is the amplifier's LIMITING quantity). See
CAL_POWER_LAWS below and docs/calibration-v2.md §13.

CLI
───
    gps_l1c_tx.py --prn 5 --power -30                          # calibrated dBm (full TMBOC)
    gps_l1c_tx.py --component pilot --secondary full --gain 60 # pilot only, full overlay
    gps_l1c_tx.py --prn 5 --gain 60 --sidelobes 0             # BOC(1,1) core only (±2.05 MHz)
    gps_l1c_tx.py --self-test
    gps_l1c_tx.py --describe-params
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
from paramkit import Script, PowerMap

import math

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1c"

# Which parameter carries the transmit frequency, so a frequency-dependent calibration chain
# folds --power at the live carrier (see the C/A scripts / docs/calibration-v2.md).
CAL_FREQ_PARAM = "freq"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 60 samples/chip; BOC(6,1) subcarrier at 10 samples/cycle (exact)
OTW_FORMAT = "sc8"            # over-the-wire; halves USB load
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L1C) ──────────────────────────────────────
L1_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10
LEG_N = 10223
INSERT = (0, 1, 1, 0, 1, 0, 0)
TMBOC = (1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)
SEC_LEN = 1800
A_PILOT = math.sqrt(0.75)
A_DATA = math.sqrt(0.25)
SIGNAL_NAME = "GPS L1C"

FREQUENCIES = {"GPS L1 (1575.42 MHz)": L1_HZ / 1e6}   # presets are in MHz now

# The L1C code is 1.023 Mcps, so its spectral nulls sit at every 1.023 MHz. The BOC(1,1) core is
# bounded by the ±2.046 MHz (2nd) null, so the passband edge snaps to the nulls as: keep the core
# + `sidelobes` further null-steps → edge at ±(n+2)·1.023 MHz. n=0 keeps the BOC(1,1) core; n=5
# the full TMBOC (just outside the ±6.138 MHz BOC(6,1) lobes, at ±7.16 MHz); n=28 fills ±Fs/2.
L1C_NULL_HZ = 1.023e6
CORE_NULLS = 2                   # the BOC(1,1) core ends at the 2nd null (±2.046 MHz)
MAX_SIDELOBES = 28               # (28+2)·1.023 = 30.69 MHz = Fs/2 — the whole representable signal
DEFAULT_SIDELOBES = 5            # the full TMBOC (±7.16 MHz)
SIDELOBE_PRESETS = {
    "BOC(1,1) core only (±2.05 MHz)": 0,
    "Between the lobes (±4.09 MHz)": 2,
    "Full TMBOC — incl BOC(6,1) lobes (±7.16 MHz)": 5,
    "Wide (±14.32 MHz)": 12,
}

# Filter skirt transition width beyond the passband edge (MHz) — FIXED. The passband edge snaps
# to a spectral null; a constant skirt keeps the emitted power a well-defined function of the
# sidelobe count alone (not the operator's discretion).
TRANSITION_MHZ = 0.05
TRANS_HZ = TRANSITION_MHZ * 1e6

# Per-PRN (Weil index w, insertion index p), IS-GPS-800 (validated vs the sheet).
L1CP_WP = (
    (5111,412), (5109,161), (5108,1), (5106,303), (5103,207), (5101,4971),
    (5100,4496), (5098,5), (5095,4557), (5094,485), (5093,253), (5091,4676),
    (5090,1), (5081,66), (5080,4485), (5069,282), (5068,193), (5054,5211),
    (5044,729), (5027,4848), (5026,982), (5014,5955), (5004,9805), (4980,670),
    (4915,464), (4909,29), (4893,429), (4885,394), (4832,616), (4824,9457),
    (4591,4429), (3706,4771), (5092,365), (4986,9705), (4965,9489), (4920,4193),
    (4917,9947), (4858,824), (4847,864), (4790,347), (4770,677), (4318,6544),
    (4126,6312), (3961,9804), (3790,278), (4911,9461), (4881,444), (4827,4839),
    (4795,4144), (4789,9875), (4725,197), (4675,1156), (4539,4674), (4535,10035),
    (4458,4504), (4197,5), (4096,9937), (3484,430), (3481,5), (3393,355),
    (3175,909), (2360,1622), (1852,6284),
)
L1CD_WP = (
    (5097,181), (5110,359), (5079,72), (4403,1110), (4121,1480), (5043,5034),
    (5042,4622), (5104,1), (4940,4547), (5035,826), (4372,6284), (5064,4195),
    (5084,368), (5048,1), (4950,4796), (5019,523), (5076,151), (3736,713),
    (4993,9850), (5060,5734), (5061,34), (5096,6142), (4983,190), (4783,644),
    (4991,467), (4815,5384), (4443,801), (4769,594), (4879,4450), (4894,9437),
    (4985,4307), (5056,5906), (4921,378), (5036,9448), (4812,9432), (4838,5849),
    (4855,5547), (4904,9546), (4753,9132), (4483,403), (4942,3766), (4813,3),
    (4957,684), (4618,9711), (4669,333), (4969,6124), (5031,10216), (5038,4251),
    (4740,9893), (4073,9884), (4843,4627), (4979,4449), (4867,9798), (4964,985),
    (5025,4272), (4579,126), (4390,10024), (4763,434), (4612,1029), (4784,561),
    (3716,289), (4703,638), (4851,4353),
)
# Pilot overlay: (S1 polynomial octal, S1 initial state octal), IS-GPS-800.
L1CO_PARAMS = (
    (0o5111,0o3266), (0o5421,0o2040), (0o5501,0o1527), (0o5403,0o3307), (0o6417,0o3756), (0o6141,0o3026),
    (0o6351,0o0562), (0o6501,0o0420), (0o6205,0o3415), (0o6235,0o0337), (0o7751,0o0265), (0o6623,0o1230),
    (0o6733,0o2204), (0o7627,0o1440), (0o5667,0o2412), (0o5051,0o3516), (0o7665,0o2761), (0o6325,0o3750),
    (0o4365,0o2701), (0o4745,0o1206), (0o7633,0o1544), (0o6747,0o1774), (0o4475,0o0546), (0o4225,0o2213),
    (0o7063,0o3707), (0o4423,0o2051), (0o6651,0o3650), (0o4161,0o1777), (0o7237,0o3203), (0o4473,0o1762),
    (0o5477,0o2100), (0o6163,0o0571), (0o7223,0o3710), (0o6323,0o3535), (0o7125,0o3110), (0o7035,0o1426),
    (0o4341,0o0255), (0o4353,0o0321), (0o4107,0o3124), (0o5735,0o0572), (0o6741,0o1736), (0o7071,0o3306),
    (0o4563,0o1307), (0o5755,0o3763), (0o6127,0o1604), (0o4671,0o1021), (0o4511,0o2624), (0o4533,0o0406),
    (0o5357,0o0114), (0o5607,0o0077), (0o6673,0o3477), (0o6153,0o1000), (0o7565,0o3460), (0o7107,0o2607),
    (0o6211,0o2057), (0o4321,0o3467), (0o7201,0o0706), (0o4451,0o2032), (0o5411,0o1464), (0o5141,0o0520),
    (0o7041,0o1766), (0o6637,0o3270), (0o4577,0o0341),
)

# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# L1C is a SPLIT (BOC) spectrum; the calibration is on the BOC(1,1) CORE — two main lobes at
# ±1.023 MHz, bounded by the DC null and the ±2.046 MHz null. ~91% of L1C power is BOC(1,1) (all
# data + 29/33 of the pilot); the other ~9% sits in the TMBOC BOC(6,1) lobes at ±6.138 MHz. Its
# distribution is fixed by ONE measured number — the power spectral DENSITY at the (BOC(1,1) core)
# main-lobe PEAK (~±0.76 MHz off the carrier), in dBm/Hz:
#
#   • Main-lobes integrated power (dBm) = peak_dBm/Hz + 10·log10(BW_ML)      ← the CORE lobes (CONST)
#   • Full signal power (dBm)           = peak_dBm/Hz + 10·log10(enbw_mhz)   ← tracks --sidelobes
#
# BW_ML is the effective bandwidth (Hz) = ∫G/G_peak of the TMBOC PSD (0.909·BOC(1,1) +
# 0.091·BOC(6,1)) over the BOC(1,1) core (±2.046 MHz); enbw_mhz is the effective bandwidth (MHz)
# the ±(n+2)·1.023 MHz lowpass passes at the current --sidelobes. The full power INCLUDES the
# BOC(6,1) lobes once --sidelobes reaches them, so it is the safe amplifier-limiting quantity and
# TRACKS --sidelobes (narrowing lowers it). At 0 sidelobes the passband IS the core, so full power
# equals the main-lobes power there. No carrier/total quantity is offered for a BOC signal.
# --self-test recomputes BW_ML and the enbw table from the TMBOC PSD and asserts these literals.

# Effective bandwidth (MHz) the ±(n+2)·1.023 MHz lowpass passes at sidelobes 0..MAX,
# = ∫G(TMBOC)/G_peak over |f| < (n+2)·1.023 MHz. A HIDDEN derived field for the GUI's static AST
# reader (it can't run the PSD integration): a nearest-integer table lookup on --sidelobes — the
# first element names the source field, the rest are enbw_mhz(0..MAX). Kept a LITERAL so the reader
# can read it AND the runtime enbw_mhz() reads the SAME literals; --self-test recomputes them from
# the TMBOC PSD (< 1e-3 MHz) so the two can never silently drift.
_ENBW_TABLE_ARGS = [
    "sidelobes",
    1.669003, 1.748812, 1.809067, 1.841331, 1.945677, 2.027690, 2.041373, 2.049564,
    2.056676, 2.061802, 2.066514, 2.070138, 2.073549, 2.076315, 2.079025, 2.081593,
    2.092184, 2.101983, 2.103927, 2.105382, 2.106726, 2.107872, 2.108972, 2.109935,
    2.110873, 2.111715, 2.112563, 2.113433, 2.117183,     # enbw_mhz(0..28)
]


def enbw_mhz(sidelobes: int) -> float:
    """The equivalent-noise bandwidth (MHz) the ±(n+2)·1.023 MHz lowpass passes at `sidelobes`
    sidelobes: full_dBm = peak_dBm/Hz + 10·log10(enbw·1e6). Reads the baked _ENBW_TABLE_ARGS (the
    same literals the client folds through); --self-test re-derives it from the TMBOC PSD to guard
    drift. Passed live to the power map so the delivered full power and the limiting cap both track
    the sidelobe count as it is tuned."""
    n = max(0, min(MAX_SIDELOBES, int(sidelobes)))
    return float(_ENBW_TABLE_ARGS[1 + n])


# The power-quantity conversion laws this signal OFFERS the calibration editor. Both convert the
# measured spectral density (dBm/Hz at the core peak) to an absolute power (dBm). Constants are
# LITERAL (the agent reads CAL_POWER_LAWS statically): 60 = 10·log10(1 MHz / 1 Hz); the full-power
# term adds 10·log10(enbw_mhz); the main-lobes k = 10·log10(∫G over ±2.046 MHz / G_peak). `rep` =
# enbw_mhz(DEFAULT_SIDELOBES) for the range read-outs shown before a live --sidelobes is known.
CAL_POWER_LAWS = [
    {"id": "full_power", "name": "Full signal power (filter passband)", "unit": "dBm",
     "in": "density", "out": "abs",
     "k": 60.0, "param": "enbw_mhz", "coeff": 10.0, "ref": 1.0, "rep": 2.027690},
    {"id": "main_lobe_power", "name": "Main-lobes integrated power (BOC(1,1) core, both lobes)",
     "unit": "dBm", "in": "density", "out": "abs", "k": 62.2246},   # 10·log10(∫G ±2.046 MHz / G_peak)
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


# ── L1C codes (bit-exact primaries + overlay, IS-GPS-800) ──────────────────────

_LEG: list | None = None


def _legendre() -> list:
    global _LEG
    if _LEG is None:
        qr = {(x * x) % LEG_N for x in range(1, LEG_N)}
        _LEG = [0] + [1 if k in qr else 0 for k in range(1, LEG_N)]
    return _LEG


def _primary(prn: int, component: str) -> list[int]:
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    w, p = (L1CP_WP if component == "pilot" else L1CD_WP)[prn - 1]
    L = _legendre()
    W = [L[k] ^ L[(k + w) % LEG_N] for k in range(LEG_N)]
    return W[0:p - 1] + list(INSERT) + W[p - 1:LEG_N]


def _overlay(prn: int) -> list[int]:
    """The 1800-symbol pilot overlay (0/1), IS-GPS-800 single 11-bit LFSR."""
    poly, init = L1CO_PARAMS[prn - 1]
    p = [((poly // 2) >> i) & 1 for i in range(11)]
    x = [(init >> i) & 1 for i in range(11)]
    c = [0] * SEC_LEN
    for i in range(SEC_LEN):
        c[i] = x[10]
        fb = 0
        for a, b in zip(x, p):
            fb ^= a & b
        x = [fb] + x[:-1]
    return c


# ── Baseband buffers (data + pilot components, one 10 ms primary period) ───────

def build_l1c_components(prn: int):
    """Return (data_buf, pilot_buf, n_samples): the two in-phase component buffers
    (complex64, one 10 ms primary period at SAMP_RATE_HZ), commonly peak-normalised. The
    pilot overlay is applied downstream."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(PRIMARY_MS * 1e-3 * sr))

    pilot = 1 - 2 * np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    data = 1 - 2 * np.asarray(_primary(prn, "data"), dtype=np.int8)
    tmboc = np.asarray(TMBOC, dtype=np.int8)

    n = np.arange(n_samples, dtype=np.int64)
    num = n * CHIP_RATE_HZ
    chip = num // sr
    rem = num - chip * sr
    boc11 = np.where(rem * 2 < sr, 1.0, -1.0)
    boc61 = np.where(((rem * 12) // sr) % 2 == 0, 1.0, -1.0)
    cidx = chip % CODE_LEN
    use61 = tmboc[chip % 33] == 1

    data_s = (A_DATA * data[cidx] * boc11).astype(np.complex128)
    pilot_s = (A_PILOT * pilot[cidx] * np.where(use61, boc61, boc11)).astype(np.complex128)
    norm = max(np.max(np.abs(data_s + pilot_s)), np.max(np.abs(data_s - pilot_s)))
    return (data_s / norm).astype(np.complex64), (pilot_s / norm).astype(np.complex64), n_samples


def overlay_signs(prn: int):
    """The pilot overlay as ±1 complex values (1800), for the runtime multiply."""
    import numpy as np
    return (1.0 - 2.0 * np.asarray(_overlay(prn), dtype=np.float32)).astype(np.complex64)


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


def filter_buffer(base_iq, sidelobes: int, trans_hz: float, base_fft=None):
    """Circularly filter a looped L1C component buffer, passband edge snapped to the null at the
    BOC(1,1) core + `sidelobes` further null-steps (±(n+2)·1.023 MHz). Circular convolution keeps
    the result exactly periodic, so the filtered loop has no seam; unity passband gain leaves the
    kept lobes' power unchanged. Pass `base_fft` (= np.fft.fft(base_iq)) to reuse it across live
    filter changes — each component loop is fixed per run, so its DFT need only be computed once,
    which cuts the per-change CPU spike (and the underflows it can cause). Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + CORE_NULLS) * L1C_NULL_HZ  # flat passband edge, on the (n+2)th null
    fc = fp + trans_hz / 2.0                          # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    if base_fft is None:
        base_fft = np.fft.fft(base_iq)
    filtered = np.fft.ifft(base_fft * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v

    leg = _legendre()
    lok = sum(leg) == (LEG_N - 1) // 2
    print(f"Legendre({LEG_N}) ones={sum(leg)} (expect {(LEG_N-1)//2}) [{'OK' if lok else 'FAIL'}]")
    tok = sum(TMBOC) == 4 and [i for i, x in enumerate(TMBOC) if x] == [0, 4, 6, 29]
    print(f"TMBOC BOC(6,1) chips {[i for i,x in enumerate(TMBOC) if x]} [{'OK' if tok else 'FAIL'}]")
    ok = ok and lok and tok

    prim = {("P", 1): 0o5752067, ("P", 2): 0o70146401, ("P", 63): 0o56350460,
            ("D", 1): 0o77001425, ("D", 2): 0o23342754, ("D", 63): 0o34665654}
    for (kind, prn), want in prim.items():
        c = _primary(prn, "pilot" if kind == "P" else "data")
        good = o24(c) == want and len(c) == CODE_LEN and sum(c) == 5115
        ok = ok and good
        print(f"{'L1Cp' if kind=='P' else 'L1Cd'} PRN{prn:2d}: first24={oct(o24(c))} "
              f"expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    # Overlay: regression guard (from IS-GPS-800 LFSR params) + structure.
    ovl = {1: 0o65550354, 63: 0o7034020}
    for prn, want in ovl.items():
        c = _overlay(prn)
        good = o24(c) == want and len(c) == SEC_LEN and sum(c) == 900
        ok = ok and good
        print(f"overlay PRN{prn:2d}: first24={oct(o24(c))} expect={oct(want)} "
              f"len={len(c)} ones={sum(c)} [{'OK' if good else 'FAIL'}]")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    data_buf, pilot_buf, n = build_l1c_components(1)
    base = data_buf + pilot_buf                       # composite (overlay sign +1 for period 0)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    # Filter each component (as the flow does) then sum — must equal filtering the sum.
    df, taps, fp = filter_buffer(data_buf, sidelobes=5, trans_hz=TRANS_HZ)     # full TMBOC (±7.16)
    pf, _, _ = filter_buffer(pilot_buf, sidelobes=5, trans_hz=TRANS_HZ)
    filt = df + pf
    kept = 10 * np.log10(band(filt, 0, fp) / band(base, 0, fp))
    cut = 10 * np.log10(band(filt, 12e6, 20e6) / max(band(base, 12e6, 20e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (sidelobes=5 → ±{fp/1e6:.2f} MHz, {taps} taps): kept band {kept:+.3f} dB, "
          f"out-of-band {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    # Calibration law constants: recompute the TMBOC effective bandwidths and assert the main-lobes
    # k AND the enbw table (full_power tracks --sidelobes) didn't drift. The BOC(6,1) weight is
    # derived from the actual TMBOC pattern + the pilot power, so a code change can't silently
    # invalidate them.
    _trapz = getattr(np, "trapezoid", None) or np.trapz

    def _boc_psd(fv, fs, fc):                # sine-BOC(fs,fc); n = 2·fs/fc (even for both used here)
        a = np.sin(np.pi * fv / (2 * fs)); c = np.cos(np.pi * fv / (2 * fs))
        top = np.sin(np.pi * fv / fc) if round(2 * fs / fc) % 2 == 0 else np.cos(np.pi * fv / fc)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (top * a / (np.pi * fv * c)) ** 2
        return np.nan_to_num(np.where(fv == 0, 0.0, v))
    fv = np.arange(-40e6 + 7, 40e6, 2e3)
    w61 = (A_PILOT ** 2) * (sum(TMBOC) / len(TMBOC))      # pilot power × BOC(6,1) chip fraction (≈0.091)
    w11 = 1.0 - w61
    g11 = _boc_psd(fv, CHIP_RATE_HZ, CHIP_RATE_HZ); g11 /= _trapz(g11, fv)
    g61 = _boc_psd(fv, 6 * CHIP_RATE_HZ, CHIP_RATE_HZ); g61 /= _trapz(g61, fv)
    gt = w11 * g11 + w61 * g61; gtp = gt.max()

    def _ebw(lo, hi):
        m = (np.abs(fv) >= lo) & (np.abs(fv) < hi)
        return float(_trapz(gt[m], fv[m])) / gtp
    ml_k = 10 * np.log10(_ebw(0.0, 2.046e6))
    laws = {l["id"]: l for l in CAL_POWER_LAWS}
    ml_ok = abs(laws["main_lobe_power"]["k"] - ml_k) < 0.02
    # full_power is KEYED on enbw_mhz; the enbw table must match ∫G over ±(n+2)·1.023 MHz / G_peak.
    tab_ok = (laws["full_power"].get("param") == "enbw_mhz"
              and abs(laws["full_power"]["k"] - 60.0) < 1e-9
              and _ENBW_TABLE_ARGS[0] == "sidelobes"
              and len(_ENBW_TABLE_ARGS) == MAX_SIDELOBES + 2
              and all(abs(_ENBW_TABLE_ARGS[1 + m]
                          - _ebw(0.0, (m + CORE_NULLS) * L1C_NULL_HZ) / 1e6) < 1e-3
                      for m in range(MAX_SIDELOBES + 1)))
    mono = all(_ENBW_TABLE_ARGS[1 + i] < _ENBW_TABLE_ARGS[2 + i] for i in range(MAX_SIDELOBES))
    # 0 sidelobes: the passband IS the core, so full power == main-lobes power there.
    full0 = 60.0 + 10 * math.log10(enbw_mhz(0))
    laws_ok = ml_ok and tab_ok and mono and abs(full0 - ml_k) < 2e-3
    print(f"calibration: BOC(6,1) weight {w61:.4f}, core-lobes k={ml_k:.4f} "
          f"(law {laws['main_lobe_power']['k']}), full(0)={full0:.4f} == core, "
          f"full(max)={60+10*math.log10(enbw_mhz(MAX_SIDELOBES)):.4f} dB, "
          f"enbw table {'matches' if tab_ok else 'DRIFTED'} [{'OK' if laws_ok else 'FAIL'}]")
    ok = ok and laws_ok
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(data_iq, pilot_iq, sec_signs, period_samples,
                     center_freq_hz, gain_db, amplitude):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack.

    Each component loop is streamed from RAM by a C++ blocks.vector_source_c (repeat=True) —
    NOT a file_source, and NOT a Python source:
      • file_source streams GIL-free and holds the rate, but swapping it live (open()) races
        GNU Radio internally and THROWS "fread error", which kills the source and silences the
        radio. With no file that is impossible.
      • a Python gr.sync_block has no file either, but its work() runs under the GIL and can't
        sustain 61.38 Msps on a Pi — it underflows even in steady state.
    vector_source_c is C++ (GIL-free) with no file, so it streams as smoothly as file_source
    did, with none of the fread risk. set_data() has no internal lock against work(), so the
    live filter swap replaces BOTH component buffers under one top-block lock()/unlock() — the
    buffers are never freed under a running read, the summed signal never streams a new
    component against an old one, and the pause is only for the swap, only on a filter
    change."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        # vector_source_c wants a contiguous complex64 buffer; the filtered loop may not be.
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class L1CTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            self.pilot_src = None
            self.data_src = None
            branches = []
            if pilot_iq is not None:
                # repeat=True loops the buffer in C++ (no per-wrap Python), vlen=1, no tags.
                self.pilot_src = blocks.vector_source_c(_vec(pilot_iq), True, 1, [])
                if sec_signs is not None:
                    sec_src = blocks.vector_source_c(list(sec_signs), repeat=True)
                    sec_rep = blocks.repeat(gr.sizeof_gr_complex, int(period_samples))
                    mult = blocks.multiply_cc()
                    self.connect(sec_src, sec_rep, (mult, 1))
                    self.connect(self.pilot_src, (mult, 0))
                    branches.append(mult)
                else:
                    branches.append(self.pilot_src)
            if data_iq is not None:
                self.data_src = blocks.vector_source_c(_vec(data_iq), True, 1, [])
                branches.append(self.data_src)

            self.amp = blocks.multiply_const_cc(amplitude)
            if len(branches) == 1:
                self.connect(branches[0], self.amp)
            else:
                adder = blocks.add_cc()
                for i, b in enumerate(branches):
                    self.connect(b, (adder, i))
                self.connect(adder, self.amp)
            self.connect(self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def swap(self, data_iq, pilot_iq):
            # Replace both component buffers under ONE lock, so the summed signal never streams
            # a new component against an old one. set_data() has no internal lock vs work(), so
            # lock()/unlock() quiesces the flowgraph for the swap — the only moment it pauses,
            # and only on a filter change.
            self.lock()
            try:
                if self.data_src is not None and data_iq is not None:
                    self.data_src.set_data(_vec(data_iq), [])
                if self.pilot_src is not None and pilot_iq is not None:
                    self.pilot_src.set_data(_vec(pilot_iq), [])
            finally:
                self.unlock()

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return L1CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (real IS-GPS-800 Weil codes, TMBOC(6,1,4/33) pilot "
               "+ BOC(1,1) data, full 18 s overlay) — fixed 61.38 MHz / sc8, looped buffers, "
               "always-on power-preserving digital passband filter set to an integer number of "
               "sidelobes. Level is set in dBm via the unit's calibration (spectral density → "
               "full / main-lobes power); uncalibrated it runs on a relative gain. Authorised, "
               "shielded setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=L1_HZ / 1e6,
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
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63). Fixed per run.")
        .choice("-Component", "--component", options=["both", "pilot", "data"], default="both",
                help="both = full L1C (25/75); pilot = L1Cp TMBOC only; data = L1Cd.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="full = apply the 1800-symbol pilot overlay (18 s, runtime multiply); "
                     "off = 10 ms primary loop only.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, step=1,
                 default=DEFAULT_SIDELOBES, presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of null-steps KEPT beyond the BOC(1,1) core: "
                      "a ±(n+2)·1.023 MHz band. 0 keeps the BOC(1,1) core only (±2.046 MHz); 5 the "
                      "full TMBOC (incl. the BOC(6,1) lobes, ±7.16 MHz). The filter is always on "
                      "(unity passband gain). More sidelobes pass more of the signal's power (the "
                      "full-power calibration quantity tracks this). Max 28 fills the band to "
                      "±Fs/2 = ±30.69 MHz. Live (rebuilds the filtered loops).")
        .derived("-Passband-bandwidth", name="passband_bw_mhz", unit="MHz",
                 formula={"linear": ["sidelobes", 2.046, 4.092],
                          # per-sidelobe-count annotation the GUI appends to the readout
                          # (nearest-int lookup on --sidelobes, last entry covers 6..28): e.g.
                          # "14.32 MHz  (full TMBOC)". Past the TMBOC lobes all power is captured.
                          "labels": ["sidelobes",
                                     "BOC(1,1) core",
                                     "core + 1 sidelobe",
                                     "core + 2 sidelobes",
                                     "core + 3 sidelobes",
                                     "incl. BOC(6,1) lobes",
                                     "full TMBOC",
                                     "full signal"]},
                 help="Occupied bandwidth the filter passes at the current sidelobe count: "
                      "2·(n+2)·1.023 MHz (i.e. ±(n+2)·1.023 MHz).")
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

    # Prebuild the unfiltered component loops once (PRN/component fixed per run); the always-on
    # filter derives from them.
    data_base, pilot_base, nsamp = build_l1c_components(args.prn)
    want_data = args.component in ("both", "data")
    want_pilot = args.component in ("both", "pilot")
    # DFT of each fixed component loop — computed once, reused across live filter changes.
    data_fft = {"v": None}
    pilot_fft = {"v": None}

    def make_component(base, cache):
        """Filtered copy of one component buffer for the current shape (the filter is always on),
        reusing the component's cached base DFT across changes."""
        if cache["v"] is None:
            import numpy as np
            cache["v"] = np.fft.fft(base)
        filtered, taps, fp = filter_buffer(base, shape["sidelobes"], shape["trans_hz"],
                                           base_fft=cache["v"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}

    finfo = {"on": True, "edge_hz": (shape["sidelobes"] + CORE_NULLS) * L1C_NULL_HZ,
             "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}
    data_iq0 = pilot_iq0 = None
    if want_data:
        data_iq0, finfo = make_component(data_base, data_fft)
    if want_pilot:
        pilot_iq0, finfo = make_component(pilot_base, pilot_fft)

    sec_signs = overlay_signs(args.prn) if (pilot_iq0 is not None and args.secondary == "full") else None

    tb = _build_top_block(data_iq0, pilot_iq0, sec_signs, nsamp,
                          center_freq_hz, gain_db, amplitude)

    def regenerate():
        """Rebuild the filtered component loops for the current shape and swap them in
        atomically (one seam, then they loop clean). Runs on the control thread; the flow keeps
        streaming the old buffers until the swap. In-RAM — no file, so a source can never be
        left dead by a read error."""
        info = {"on": True, "edge_hz": (shape["sidelobes"] + CORE_NULLS) * L1C_NULL_HZ,
                "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}
        data_iq = pilot_iq = None
        if want_data:
            data_iq, info = make_component(data_base, data_fft)
        if want_pilot:
            pilot_iq, info = make_component(pilot_base, pilot_fft)
        tb.swap(data_iq, pilot_iq)
        return info

    # Track the held absolute --power target (calibrated mode only) so a live --sidelobes change
    # can re-map the gain: the FULL-power quantity moves with the sidelobe count, so keeping the
    # same delivered power needs a new gain (a main-lobes/relative target is unaffected — calkit
    # handles which, via the embedded law).
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "power": (float(args.power) if (gain_cal is None and pmap.has_absolute
                                             and getattr(args, "power", None) is not None)
                       else None)}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def _fmt_band(info):
        return (f"on — core + {info['sidelobes']} null-step(s) "
                f"(±{info['edge_hz']/1e6:.2f} MHz), {info['trans_hz']/1e6:g} MHz transition"
                + (f", {info['taps']} taps" if 'taps' in info else ""))

    sec_desc = "18 s (full overlay)" if sec_signs is not None else "10 ms (primary only)"
    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  satellite PRN  : {args.prn}  (real L1C Weil codes, {args.component})")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : L1Cp TMBOC(6,1,4/33) + L1Cd BOC(1,1), 75/25 power")
    print(f"  period         : {sec_desc}")
    print(f"  primary buffer : {nsamp} samples ({nsamp*8/1e6:.1f} MB/component file)")
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
            # bandwidth-independent (main-lobes) or relative target.
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

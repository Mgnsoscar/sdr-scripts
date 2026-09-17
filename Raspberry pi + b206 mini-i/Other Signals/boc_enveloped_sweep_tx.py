#!/usr/bin/env python3
"""
BOC Enveloped Sweep transmitter for GNU Radio + UHD (Ettus B200-mini family).

The Enveloped Sweep, shaped like a sine-phased **BOC(10,5)** (GPS M-code) spectrum instead of a
sinc². A CONSTANT-ENVELOPE swept tone whose TIME-AVERAGED PSD reproduces M-code's SPLIT spectrum —
two lobes at ±10.23 MHz with a gap at the centre — realised purely by DWELL TIME (no amplitude
taper, no crest-factor penalty). Like enveloped_sweep_tx.py but with the BOC(10,5) target: the
sweep lingers where BOC has power (the two main lobes around ±10.23 MHz) and rushes through the
nulls and the centre gap. See enveloped_sweep_tx.py for the mechanism write-up.

BOC(10,5)  (fixed — this IS the M-code modulation)
──────────────────────────────────────────────────
  subcarrier fsub = 10 × 1.023 = 10.23 MHz   (the split-spectrum lobe centres, ±fsub)
  code rate  fc   =  5 × 1.023 =  5.115 MHz   (the PSD null spacing)
The sine-phased BOC(10,5) PSD (Betz) is  G(f) ∝ [ sin(4a)·sin(a) / (π f·cos(a)) ]²  with
a = π f / (2·fsub); using sin(4a)/cos(a) = 4·sin(a)·(1−2sin²a) it is finite at every lobe peak.
Nulls sit at every 5.115 MHz; the two MAIN lobes span |f| ∈ [5.115, 15.345] MHz (fsub ± fc), and
the 3rd-harmonic lobes sit at ±30.69 MHz (= Fs/2). `--sidelobes n` keeps the main lobes + n
further null-steps, a ±(n+3)·5.115 MHz band (0 → ±15.345, 3 → ±30.69).

How the shape is realised
─────────────────────────
`S(f) = BOC(10,5) PSD` is treated as a dwell density: the instantaneous frequency is driven
through its INVERSE CDF, so the tone dwells ∝ S(f) and the averaged PSD IS the BOC split
spectrum. Swept symmetrically (out and back) so the looped buffer closes with no reset; a tiny
dwell floor lets the tone creep through the nulls and the centre gap (soft, leakage-limited).
Fixed radio setup + always-on unity passband filter + precompute-and-loop, exactly as
enveloped_sweep_tx.py / fm_chirp_tx.py.

The passband is set by TWO knobs (both snap to BOC nulls): --sidelobes extends the OUTER edge (n
null-steps beyond the main lobes), and --inner-sidelobes drops the INNER edge — a bandpass that
notches away the low-power gap between the two lobes (±5.115 MHz), leaving a clean split spectrum.

Calibration REUSES the regular sweep's ("Chirp/Sweep"). Constant-envelope at the same amplitude,
so at a given gain it delivers the IDENTICAL total power — the flat-sweep calibration already
contains it. Offers FULL signal power (dBm) and MAIN-LOBES power (both split lobes; that minus a
fixed BOC offset that tracks --sidelobes). MAIN-LOBES power is EXACT whether or not the centre is
notched (the notch never touches the lobes) — so set --power in Main-lobes power when using
--inner-sidelobes; FULL signal power counts the pre-notch total (the notch discards the ~0.4 dB
inner gap). Uncalibrated it runs on a relative --gain.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a shielded /
   conducted setup (cable + attenuators) you are LICENSED / AUTHORISED to use.

CLI
───
    boc_enveloped_sweep_tx.py --freq 1575.42 --sidelobes 0 --power -30
    boc_enveloped_sweep_tx.py --freq 1575.42 --sidelobes 0 --inner-sidelobes 1 --power -30  # clean split
    boc_enveloped_sweep_tx.py --freq 1575.42 --sidelobes 2 --gain 60      # raw-gain override
    boc_enveloped_sweep_tx.py --self-test        # verify seam closure + BOC shape, no hardware
    boc_enveloped_sweep_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# Quiet UHD/GNU Radio BEFORE the libs load (imported lazily inside main()).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id — the SAME as fm_chirp_tx.py's "Chirp/Sweep". A regular (flat)
# sweep and this BOC-shaped Enveloped Sweep are BOTH constant-envelope at the same baseband
# amplitude, so at a given SDR gain they deliver the IDENTICAL total power — the unit's flat-sweep
# calibration already contains this signal's power vs gain, no separate measurement. The agent
# injects the unit's resolved "Chirp/Sweep" calibration; calkit folds --power through its measured
# density curve exactly as the chirp does. Absent it, the script runs uncalibrated (gain only).
CAL_SIGNAL_ID = "Chirp/Sweep"

# Which parameter carries the transmit frequency (folded there, live).
CAL_FREQ_PARAM = "freq"

# The sweep bandwidth (MHz) the shared "Chirp/Sweep" density is measured at (matches fm_chirp): it
# fixes k = 60 + 10·log10(CAL_MEAS_BW_MHZ) = 70, turning the measured peak density (dBm/Hz) into
# the total signal power: full_dBm = density + 70.
CAL_MEAS_BW_MHZ = 10.0

# Fraction of the BOC(10,5) power inside the two MAIN lobes (|f| ∈ [5.115, 15.345] MHz) relative
# to the total kept out to ±(sidelobes+3)·5.115 MHz: main_lobes_dBm = full_dBm + 10·log10(frac).
# A pure geometric constant of the BOC(10,5) shape and the sidelobe truncation — needs no
# measurement. frac(0) ≈ 0.915 (−0.39 dB: the ±15.345 band also passes the low-power centre gap),
# dropping further as the outer null-steps + the ±30.69 MHz 3rd-harmonic lobes are added. Baked
# literal (static AST reader); --self-test re-derives it from the BOC PSD so the two can't drift.
_MAIN_LOBE_FRAC_ARGS = [
    "sidelobes",
    0.914610, 0.912547, 0.911460, 0.867182,       # sidelobes 0, 1, 2, 3
]
_MAIN_LOBE_FRAC = _MAIN_LOBE_FRAC_ARGS[1:]         # the fractions alone, for the runtime lookup


def main_lobe_frac(sidelobes: int) -> float:
    """The both-main-lobes power fraction for `sidelobes` sidelobes kept (nearest, clamped)."""
    n = max(0, min(len(_MAIN_LOBE_FRAC) - 1, int(sidelobes)))
    return _MAIN_LOBE_FRAC[n]


# Power-quantity conversion laws (ride through the static argspec to the client; the runtime folds
# --power in the base density exactly like the chirp). k = 70 = 60 + 10·log10(CAL_MEAS_BW_MHZ).
# Full signal power is the bandwidth-invariant total (= the flat sweep's total at this gain);
# main-lobes power is that minus the fixed BOC offset, KEYED on --sidelobes via the hidden
# `main_lobe_frac` derived field. `rep` = the value at the default sidelobe count.
CAL_POWER_LAWS = [
    {"id": "full_power", "name": "Full signal power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.0},
    {"id": "main_lobe_power", "name": "Main-lobes power (both split lobes)", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.0,
     "param": "main_lobe_frac", "coeff": 10.0, "ref": 1.0, "rep": 0.914610},
]


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain. GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1) — the amplitude the calibration is measured at.
AMPLITUDE = 0.5
HW_MAX_GAIN_DB = 89.75      # B200-mini TX-gain ceiling


_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration if present (SDR_CALIBRATION_FILE),
    else uncalibrated (relative gain only). Cached so build_script and main share one."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Fixed radio setup + BOC(10,5) constants (NOT parameters) ─────────────────────────
SAMP_RATE_HZ = 61.38e6       # the max; master clock pinned 1:1
OTW_FORMAT = "sc8"           # over-the-wire; halves USB load

CODE_RATE_HZ = 5_115_000     # BOC(10,5) code rate  fc = 5 × 1.023 Mcps (the PSD null spacing)
SUBCARRIER_HZ = 10_230_000   # BOC(10,5) subcarrier fsub = 10 × 1.023 MHz (= 2 × code rate)
BOC_NULL_HZ = CODE_RATE_HZ   # nulls sit at every code rate
MAIN_LOBE_NULLS = 3          # the two main lobes end at the 3rd null (±15.345 MHz)
MAX_SIDELOBES = 3            # (3+3)·5.115 = 30.69 MHz = Fs/2 — the whole representable signal
DEFAULT_SIDELOBES = 0        # main lobes only (±15.345 MHz)
MAX_ROAM_MHZ = 30.69         # ±(sidelobes+3)·5.115 must stay within ±Nyquist (Fs/2)
# The centre gap between the two split lobes is one inner null-step ([0, 5.115] MHz). Notching it
# (a bandpass instead of a lowpass) leaves a clean split spectrum. 0 = keep it, 1 = notch it.
MAX_INNER_SIDELOBES = 1
DEFAULT_INNER_SIDELOBES = 0

# One whole sweep is precomputed into this many samples and looped (the fm_chirp floor, ~2 MB).
BUFFER_SAMPS = 1 << 18

# Tiny dwell floor so the sweep CREEPS through the nulls + centre gap (where the BOC PSD → 0)
# instead of teleporting — keeps the CDF strictly increasing. Sets the realised null floor.
DWELL_FLOOR = 1e-3

# Always-on passband filter (unity gain): passband = the occupied band ±(sidelobes+3)·5.115 MHz
# (snaps to a BOC null), fixed transition skirt. Band-limits + cleans the turnaround; the unity
# passband leaves the BOC shape untouched.
FILTER_TRANSITION_MHZ = 0.05
FILTER_TRANSITION_HZ = FILTER_TRANSITION_MHZ * 1e6

# Named GNSS carriers (same list as fm_chirp_tx.py), in MHz.
FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": 1575.42, "GPS L2 (1227.60 MHz)": 1227.60,
    "GPS L5 (1176.45 MHz)": 1176.45,
    "Galileo E1 (1575.42 MHz)": 1575.42, "Galileo E5a (1176.45 MHz)": 1176.45,
    "Galileo E5b (1207.14 MHz)": 1207.14,
    "Galileo E5 (1191.795 MHz)": 1191.795, "Galileo E6 (1278.75 MHz)": 1278.75,
    "BeiDou B1I (1561.098 MHz)": 1561.098, "BeiDou B1C (1575.42 MHz)": 1575.42,
    "BeiDou B2a (1176.45 MHz)": 1176.45,
    "BeiDou B2b (1207.14 MHz)": 1207.14, "BeiDou B2 (1191.795 MHz)": 1191.795,
    "BeiDou B3 (1268.52 MHz)": 1268.52,
    "GLONASS L1 (1602.0 MHz)": 1602.0, "GLONASS L2 (1246.0 MHz)": 1246.0,
    "GLONASS L3 (1202.025 MHz)": 1202.025,
    "Iridium (1621.25 MHz)": 1621.25,
}


# ── BOC(10,5) PSD + dwell-shaped frequency trajectory ────────────────────────────

def boc_psd(f_hz):
    """Sine-phased BOC(10,5) power spectral density (unnormalised shape). Robust at the removable
    singularities: sin(4a)/cos(a) = 4·sin(a)·(1 − 2·sin²a), so the lobe peaks are finite."""
    import numpy as np
    f = np.asarray(f_hz, dtype=float)
    a = np.pi * f / (2.0 * SUBCARRIER_HZ)
    sa = np.sin(a)
    num = (4.0 * sa * (1.0 - 2.0 * sa * sa)) * sa     # sin(4a)·sin(a)/cos(a), singularity-free
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (num / (np.pi * f)) ** 2
    return np.where(f == 0.0, 0.0, g)                 # split spectrum → a null at DC


def roam_hz(sidelobes: int) -> float:
    """The occupied half-width: the outermost BOC null kept = ±(sidelobes+3)·code_rate."""
    return (int(sidelobes) + MAIN_LOBE_NULLS) * BOC_NULL_HZ


def inner_edge_hz(inner_sidelobes: int) -> float:
    """The passband's INNER edge: 0 (a plain lowpass, centre kept) or ±code_rate (the centre gap
    notched, leaving a clean split spectrum). Snaps to the BOC null at ±5.115 MHz."""
    return max(0, min(MAX_INNER_SIDELOBES, int(inner_sidelobes))) * BOC_NULL_HZ


def check_roam(sidelobes: int) -> None:
    """Raise ValueError if the occupied band would exceed the usable baseband."""
    r = roam_hz(sidelobes) / 1e6
    if r > MAX_ROAM_MHZ + 1e-6:
        raise ValueError(
            f"occupied band ±{r:g} MHz (= (sidelobes+3)·5.115) exceeds the maximum "
            f"±{MAX_ROAM_MHZ:g} MHz — lower --sidelobes.")


def _boc_dwell_freq(sidelobes: int, n: int):
    """Instantaneous frequency f[k] (Hz) for a length-n CONSTANT-ENVELOPE buffer whose
    time-averaged PSD is the BOC(10,5) spectrum, truncated at ±(sidelobes+3)·5.115 MHz and swept
    symmetrically (out and back) so the loop closes. Inverse-CDF of boc_psd → dwell ∝ BOC. Returns
    the zero-mean trajectory."""
    import numpy as np
    r = roam_hz(sidelobes)
    g = np.linspace(-r, r, 40001)
    S = boc_psd(g)
    S = np.maximum(S, DWELL_FLOOR * S.max())          # floor so the CDF is strictly increasing
    C = np.cumsum(S)
    C = (C - C[0]) / (C[-1] - C[0])                   # CDF, 0..1
    half = n // 2
    w = (np.arange(half) + 0.5) / half
    up = np.interp(w, C, g)                            # inverse-CDF: dwell ∝ BOC PSD
    f = np.concatenate([up, up[::-1]])                 # out and back → symmetric, closes at the wrap
    if len(f) < n:
        f = np.concatenate([f, f[-1:]])
    f = f[:n]
    return f - f.mean()                                # exact zero mean → phase closes over the loop


def build_boc_sweep_buffer(sidelobes: int):
    """A complex64 constant-modulus buffer of one whole BOC-dwell sweep that loops with no seam.
    Amplitude is applied live downstream. Returns (iq, freq_trajectory)."""
    import numpy as np
    f = _boc_dwell_freq(sidelobes, BUFFER_SAMPS)
    phase = (2.0 * np.pi / SAMP_RATE_HZ) * np.cumsum(f)
    return np.exp(1j * phase).astype(np.complex64), f


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. Returns (h, n_taps)."""
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
    h = h / h.sum()
    return h.astype(np.float64), m


def filter_buffer(base_iq, width_hz: float, trans_hz: float, inner_hz: float = 0.0):
    """Circularly filter the looped sweep to the passband [inner_hz, width_hz/2] on each side: a
    plain lowpass when `inner_hz == 0`, or a BANDPASS when `inner_hz > 0` that also notches away
    the low-power centre gap between the two BOC lobes (|f| < inner_hz). Built as
    lowpass(outer) − lowpass(inner), so the passband is unity and both edges snap to BOC nulls;
    circular convolution keeps the loop seamless. Returns (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = float(width_hz) / 2.0
    n = len(base_iq)
    h_out, m = _design_lowpass(fp + trans_hz / 2.0, trans_hz, n // 2)
    H = np.fft.fft(h_out, n)
    if inner_hz > 0.0:                              # bandpass = lp(outer) − lp(inner) → notch the centre
        h_in, _ = _design_lowpass(float(inner_hz), trans_hz, n // 2)
        H = H - np.fft.fft(h_in, n)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * H).astype(np.complex64)
    return filtered, m, fp


# ── Self-test: seam + constant envelope + BOC shape + confinement, no hardware ──────

def _self_test() -> int:
    import cmath
    import math
    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — cannot run the self-test)")
        return 0

    ok = True
    sidelobes = 0

    # 1) seamless phase closure
    f = _boc_dwell_freq(sidelobes, BUFFER_SAMPS)
    phase = (2.0 * math.pi / SAMP_RATE_HZ) * np.cumsum(f)
    iq = np.exp(1j * phase)
    expected = 2 * math.pi / SAMP_RATE_HZ * f[0]
    measured = cmath.phase(iq[0] / iq[-1])
    seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
    good = seam_err < 1e-9
    ok = ok and good
    print(f"seam closure : err={seam_err:.2e} rad [{'OK' if good else 'FAIL'}]")

    # 2) constant envelope (pre-filter)
    env = np.abs(iq)
    env_ok = float(np.max(np.abs(env - 1.0))) < 1e-6
    ok = ok and env_ok
    print(f"constant env : max|1-|iq||={np.max(np.abs(env-1.0)):.1e} [{'OK' if env_ok else 'FAIL'}]")

    # 3) the averaged PSD is BOC(10,5): peak in the main lobe (~±10.23), nulls at 5.115 / 15.345,
    #    and a gap at the centre — the split spectrum
    filt, taps, fp = filter_buffer(
        build_boc_sweep_buffer(sidelobes)[0], 2 * roam_hz(sidelobes), FILTER_TRANSITION_HZ)
    X = np.abs(np.fft.fftshift(np.fft.fft(filt))) ** 2
    ff = np.fft.fftshift(np.fft.fftfreq(len(filt), 1.0 / SAMP_RATE_HZ))
    sm = np.convolve(X, np.ones(31) / 31, "same")
    sm = 10 * np.log10(sm / sm.max() + 1e-30)

    def at(fq):
        return float(sm[np.argmin(np.abs(ff - fq))])

    def peak_in(lo, hi):
        m = (np.abs(ff) >= lo) & (np.abs(ff) < hi)
        return float(np.max(sm[m]))

    lobe = peak_in(5.115e6, 15.345e6)        # both main lobes
    centre = at(0.0)                          # split-spectrum gap
    null1 = at(5.115e6)                       # 1st null
    null2 = at(15.345e6)                      # main-lobe outer null
    shape_ok = (lobe > -1.0) and (centre < -10.0) and (null1 < -10.0) and (null2 < -10.0)
    ok = ok and shape_ok
    print(f"BOC shape    : main lobe {lobe:+.1f} dB, centre gap {centre:+.1f} dB, "
          f"nulls {null1:+.1f}/{null2:+.1f} dB [{'OK' if shape_ok else 'FAIL'}]")

    # 4) the sweep stays INSIDE the occupied band ±roam (never dwells outside the filter passband)
    r = roam_hz(sidelobes)
    excess = float(np.max(np.abs(f))) - r
    conf_ok = excess <= 1.0
    ok = ok and conf_ok
    print(f"confined     : max|f|={np.max(np.abs(f))/1e6:.4f} MHz vs roam {r/1e6:.4f} MHz "
          f"(excess {excess:+.2e} Hz) [{'OK' if conf_ok else 'FAIL'}]")

    # 5) the baked main-lobe-fraction table matches the BOC integral (can't silently drift)
    grid = np.linspace(-MAX_ROAM_MHZ * 1e6, MAX_ROAM_MHZ * 1e6, 400001)
    G = boc_psd(grid)
    ml = np.sum(G[(np.abs(grid) >= 5.115e6) & (np.abs(grid) < 15.345e6)])
    frac_ok = True
    for n_sl, baked in enumerate(_MAIN_LOBE_FRAC):
        tot = np.sum(G[np.abs(grid) < (n_sl + MAIN_LOBE_NULLS) * BOC_NULL_HZ])
        frac_ok = frac_ok and abs(ml / tot - baked) < 5e-3
    ok = ok and frac_ok
    print(f"main-lobe tbl: baked ≡ BOC integral (≤5e-3) [{'OK' if frac_ok else 'FAIL'}]")

    # 6) the inner notch (--inner-sidelobes 1) drops the centre gap while keeping the main lobes
    filt_n, _t, _fp = filter_buffer(
        build_boc_sweep_buffer(0)[0], 2 * roam_hz(0), FILTER_TRANSITION_HZ, inner_hz=inner_edge_hz(1))
    Xn = np.abs(np.fft.fftshift(np.fft.fft(filt_n))) ** 2
    smn = np.convolve(Xn, np.ones(31) / 31, "same")
    smn = 10 * np.log10(smn / smn.max() + 1e-30)
    centre_n = float(smn[np.argmin(np.abs(ff))])                     # centre after the notch
    lobe_n = float(np.max(smn[(np.abs(ff) >= 5.115e6) & (np.abs(ff) < 15.345e6)]))
    notch_ok = (centre_n < centre - 10.0) and (lobe_n > -1.0)        # centre far deeper, lobes intact
    ok = ok and notch_ok
    print(f"inner notch  : centre {centre:+.1f}→{centre_n:+.1f} dB, main lobe {lobe_n:+.1f} dB "
          f"[{'OK' if notch_ok else 'FAIL'}]")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float, amplitude: float):
    """The looped sweep is streamed from RAM by a C++ blocks.vector_source_c (repeat=True); a live
    shape change swaps the buffer with set_data() under top-block lock()/unlock(). See
    fm_chirp_tx.py for the full write-up."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class BocSweepTx(gr.top_block):
        def __init__(self):
            super().__init__("BOC Enveloped Sweep TX")
            self._freq_hz = center_freq_hz

            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]),
            )
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self._retune()
            self.usrp.set_gain(gain_db, 0)

            self.src = blocks.vector_source_c(_vec(initial_iq), True, 1, [])
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def _retune(self) -> None:
            self.usrp.set_center_freq(uhd.tune_request(self._freq_hz), 0)

        def set_center_frequency(self, hz: float) -> None:
            self._freq_hz = hz
            self._retune()

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def swap(self, iq) -> None:
            data = _vec(iq)
            self.lock()
            try:
                self.src.set_data(data, [])
            finally:
                self.unlock()

        def actual_freq(self) -> float:
            return self.usrp.get_center_freq(0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return BocSweepTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BOC Enveloped Sweep transmitter — a CONSTANT-ENVELOPE swept tone whose averaged "
               "spectrum is shaped like a sine-phased BOC(10,5) (GPS M-code): a split spectrum "
               "with two lobes at ±10.23 MHz, by dwell time only (no amplitude taper, no "
               "crest-factor penalty). Fixed 61.38 MHz / sc8, looped buffer, always-on unity "
               "passband filter. Reuses the regular sweep's (\"Chirp/Sweep\") calibration and "
               "offers full-signal and main-lobes power (dBm). Uncalibrated it runs on a relative "
               "gain. Authorised, shielded setups only.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration (folded at the current carrier) and snaps to its achievable "
                     "grid; ignored if --gain is given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .number("-Center-frequency", "--freq", unit="MHz", min=1000, max=1800,
                presets=FREQUENCIES, default=1575.42, required=False, live=True,
                help="RF carrier (the sweep centre) in MHz. --power is calibrated here. Live.")
        .number("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, step=1,
                default=DEFAULT_SIDELOBES, required=False, live=True,
                help="How many BOC null-steps to keep beyond the two main lobes (0 = the main "
                     "lobes only, ±15.345 MHz). The occupied band is ±(sidelobes+3)·5.115 MHz; "
                     "3 = ±30.69 MHz (Fs/2). Live (regenerates the sweep).")
        .number("-Inner-sidelobes", "--inner-sidelobes", min=0, max=MAX_INNER_SIDELOBES, step=1,
                default=DEFAULT_INNER_SIDELOBES, required=False, live=True,
                help="Notch away the low-power gap BETWEEN the two main lobes (a bandpass instead "
                     "of a lowpass): 0 = keep the centre; 1 = a clean split spectrum (passband "
                     "starts at ±5.115 MHz). Set --power in Main-lobes power when notching — "
                     "Full-signal power counts the discarded centre. Live (regenerates).")
        .derived("-Main-lobe-fraction", name="main_lobe_frac", hidden=True,
                 formula={"table": _MAIN_LOBE_FRAC_ARGS},
                 help="Fraction of the BOC power inside the two main lobes at the current sidelobe "
                      "count. Feeds the main-lobes-power calibration law; not shown.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                is_rf=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Power edits made while OFF are staged. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()

    center_freq_hz = float(args.freq) * 1e6
    sidelobes = int(args.sidelobes)
    inner = int(getattr(args, "inner_sidelobes", DEFAULT_INNER_SIDELOBES) or 0)
    try:
        check_roam(sidelobes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Current "shape" — mutated by live changes. Defined before the gain fold so the calibration's
    # main-lobes-power law can read the live fraction.
    shape = {"sidelobes": sidelobes, "inner": inner}

    def pwr_params() -> dict:
        """Live keyed-parameter values the calibration's power laws read: the BOC main-lobes
        fraction at the current sidelobe count. Harmless when the calibration doesn't key on it."""
        return {"main_lobe_frac": main_lobe_frac(shape["sidelobes"])}

    pmap = power_map()
    amplitude = pmap.amplitude
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

    def make_current():
        """The buffer for the current shape: the BOC-dwell sweep, band-limited to the occupied
        band ±(sidelobes+3)·5.115 MHz — and, when --inner-sidelobes is set, notched below the
        inner edge (±5.115 MHz) to drop the centre gap. Returns (iq, finfo)."""
        base, _f = build_boc_sweep_buffer(shape["sidelobes"])
        r = roam_hz(shape["sidelobes"])
        inner_hz = inner_edge_hz(shape["inner"])
        filt, taps, fp = filter_buffer(base, 2 * r, FILTER_TRANSITION_HZ, inner_hz=inner_hz)
        return filt, {"taps": taps, "edge_hz": fp, "roam_hz": r, "inner_hz": inner_hz}

    iq, finfo = make_current()

    tb = _build_top_block(
        initial_iq=iq, center_freq_hz=center_freq_hz, gain_db=gain_db, amplitude=amplitude)

    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": center_freq_hz, "power": _target_power}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def regenerate():
        iq, fi = make_current()
        tb.swap(iq)
        return fi

    T_ms = BUFFER_SAMPS / SAMP_RATE_HZ * 1e3

    print("── BOC Enveloped Sweep TX ──────────────────────────────────")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  shape          : BOC(10,5)  (fsub 10.23 MHz, fc 5.115 MHz, "
          f"{sidelobes} sidelobe null-step(s))")
    print(f"  occupied band  : ±{finfo['roam_hz']/1e6:g} MHz (split lobes at ±10.23 MHz)")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  sweep period   : {T_ms:.2f} ms ({BUFFER_SAMPS} samples, loops "
          f"{SAMP_RATE_HZ/BUFFER_SAMPS:.0f} Hz)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params()):.2f} dBm")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    print(f"  filter         : on (always) — passband ±{finfo['edge_hz']/1e6:.2f} MHz "
          f"(= occupied band), {FILTER_TRANSITION_MHZ:g} MHz transition, {finfo['taps']} taps")
    if finfo.get("inner_hz", 0.0) > 0:
        print(f"  inner notch    : centre gap dropped below ±{finfo['inner_hz']/1e6:.3f} MHz "
              f"(clean split spectrum — set --power in Main-lobes power)")
    print(f"  otw            : {OTW_FORMAT}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "freq":
            hz = float(value) * 1e6
            tb.set_center_frequency(hz)
            state["freq"] = hz
            ctrl.report("freq", tb.actual_freq() / 1e6)
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"], params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power",
                                round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))
                else:
                    ctrl.report("power",
                                round(pmap.power_for_gain(state["gain"], freq=state["freq"], params=pwr_params()), 2))
        elif name == "power":
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"], params=pwr_params())
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(
                    tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"], params=pwr_params()), 2))
        elif name == "gain":
            state["power"] = None
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("gain", round(tb.actual_gain(), 2))
            else:
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
            new_sl = int(value)
            try:
                check_roam(new_sl)
            except ValueError:
                ctrl.report(name, value)          # keep the last good shape
                return
            shape["sidelobes"] = new_sl
            regenerate()
            ctrl.report(name, value)
            # A --sidelobes change moves the main-lobes fraction, so re-map a held --power in case
            # the calibration's cap keys on it (a no-op for a total-power cap — constant envelope).
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"], params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power", round(pmap.power_for_gain(
                        tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))
        elif name == "inner_sidelobes":
            # Only the FILTER changes (a notch of the centre gap); the trajectory, the main-lobes
            # fraction and the gain are all unaffected — so just rebuild + swap, no power re-map.
            shape["inner"] = max(0, min(MAX_INNER_SIDELOBES, int(value)))
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

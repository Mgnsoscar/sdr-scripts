#!/usr/bin/env python3
"""
Enveloped Sweep transmitter for GNU Radio + UHD (Ettus B200-mini family).

A CONSTANT-ENVELOPE swept tone whose TIME-AVERAGED power spectrum is shaped like a
sinc² — energy concentrated at the centre, tapering to the band edges — instead of the
flat rectangle a plain chirp paints. It is the power-efficient way to focus energy where
it matters: the tone runs at full amplitude the whole time (like cw_tx / fm_chirp), so
there is no crest-factor penalty, and the shaping is done purely by DWELL TIME — the
sweep lingers near the centre and rushes through the edges.

Why dwell shaping (and NOT an amplitude envelope)
─────────────────────────────────────────────────
At max SDR gain the tone is already pegged at full scale every instant, so an amplitude
envelope could only push the EDGES down (and would waste power). Making the sweep DWELL
longer near the centre instead REDISTRIBUTES the same total power toward the centre —
raising the centre PSD above a flat sweep of the same total power, at the same gain, for
free. The PSD at a frequency is (time spent there) × (amplitude² there); amplitude² is
maxed everywhere, so the only free lever is time.

How the shape is realised (inverse-CDF dwell mapping)
─────────────────────────────────────────────────────
Treat the target spectrum S(f) = sinc²(f / chip_rate) as a probability density over the
occupied band, integrate it to a CDF, and drive the instantaneous frequency through the
INVERSE CDF of a uniform time index. The tone then visits each frequency in proportion
to S(f), so the averaged PSD IS sinc². The sweep is symmetric (out and back), so the
looped buffer closes with no reset splatter. A tiny dwell floor lets the tone creep
through the sinc² nulls rather than teleport across them (so the realised nulls are
soft, ~-27 dB with this buffer, not razor nulls — dwell shaping is leakage-limited; this
is fine for energy shaping).

  chip_rate  — the sinc² width: the FIRST spectral null sits at ±chip_rate (Mcps ≡ MHz),
               exactly like a BPSK/PRN signal at that chip rate. Live (regenerates).
  sidelobes  — how many sinc² sidelobes to keep each side (0 = main lobe only). The band
               occupied is ±(sidelobes+1)·chip_rate. Live (regenerates).

Fixed radio setup (as fm_chirp_tx.py): sample rate 61.38 MHz, over-the-wire sc8, baseband
amplitude 0.5 (the amplitude the calibration is measured at), and an always-on unity-gain
digital passband filter whose passband equals the occupied band (it band-limits the sweep
and cleans the turnaround). None are parameters. The single sweep is precomputed once and
replayed from RAM by a C++ vector_source_c (repeat=True) — the same recipe fm_chirp_tx.py
uses to sustain the rate on a Pi. A live chip_rate / sidelobes change rebuilds one sweep in
RAM and swaps it under the top-block lock (one brief seam at the swap, then it loops clean).

Live tuning (retune while transmitting, via paramkit.live)
──────────────────────────────────────────────────────────
    freq       → UHD tune_request        (instant; re-maps --power at the new carrier)
    power      → dBm → set_gain           (instant; staged while --rf off)
    gain       → raw set_gain             (instant; overrides --power)
    rf         → on/off mute/unmute       (instant — gain AND amplitude to 0 / back)
    chip_rate  → rebuild buffer + swap    ┐ shape changes: regenerate one sweep in RAM and
    sidelobes  → rebuild buffer + swap    ┘ set_data() it under the top-block lock.

Calibration REUSES the regular sweep's ("Chirp/Sweep"). Both are constant-envelope at the same
amplitude, so at a given gain they deliver the IDENTICAL total power — the unit's flat-sweep
calibration already contains this signal's power vs gain, no separate measurement. Two power
quantities are offered: FULL signal power (= the flat sweep's total, dBm) and MAIN-LOBE power
(that minus a fixed sinc² offset that tracks --sidelobes). So knowing the flat sweep's passband
PSD at a gain gives this signal's full and main-lobe power at that gain. --power is set in dBm
via the calibration (folded at the carrier), or a relative --gain uncalibrated. The shape
decides WHERE the power lands, not how much there is.

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a shielded /
   conducted setup (cable + attenuators) you are LICENSED / AUTHORISED to use.

CLI
───
    enveloped_sweep_tx.py --freq 1575.42 --chip-rate 1.023 --sidelobes 3 --power -30
    enveloped_sweep_tx.py --freq 1575.42 --chip-rate 2.0 --gain 60          # raw-gain override
    enveloped_sweep_tx.py --self-test        # verify seam closure + sinc² shape, no hardware
    enveloped_sweep_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# Stable calibration signal id — the SAME as fm_chirp_tx.py's "Chirp/Sweep". A regular
# (flat) sweep and this Enveloped Sweep are BOTH constant-envelope at the same baseband
# amplitude, so at a given SDR gain they deliver the IDENTICAL total power (verified: within
# 0.002 dB). The unit's flat-sweep calibration therefore already contains this signal's power
# vs gain — no separate measurement. The agent injects the unit's resolved "Chirp/Sweep"
# calibration (SDR_CALIBRATION_FILE); calkit folds --power through its MEASURED density curve
# exactly as the chirp does. Absent it, the script runs uncalibrated (relative gain only).
CAL_SIGNAL_ID = "Chirp/Sweep"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# has a --power scale that MOVES with frequency, so the map is folded at THIS param's value —
# and it is live, so retuning the centre re-scales --power on the fly.
CAL_FREQ_PARAM = "freq"

# The sweep bandwidth (MHz) the shared "Chirp/Sweep" density is MEASURED at (matches
# fm_chirp_tx.py). It fixes the constant that turns the measured peak density (dBm/Hz) into
# the total signal power: full_dBm = density + 10·log10(CAL_MEAS_BW_MHZ·1e6) = density + 70.
CAL_MEAS_BW_MHZ = 10.0

# Fraction of a sinc²'s power inside the MAIN LOBE (±chip_rate) relative to the total kept out
# to (sidelobes+1) nulls: main_lobe_dBm = full_dBm + 10·log10(frac). A pure geometric constant
# of the sinc² shape and the sidelobe truncation — needs no measurement. Index by sidelobe
# count 0..SIDELOBES_MAX; the 0-sidelobe case keeps only the main lobe, so frac = 1 (main lobe
# IS the whole kept signal). Baked literal (the client's schema reader is a static AST reader);
# --self-test re-derives it from the sinc² integral so the two can never silently drift.
# The hidden `main_lobe_frac` derived field's table (a nearest-int lookup on --sidelobes). The
# first element names the source field; the rest are the fractions for 0..SIDELOBES_MAX. Kept a
# pure LITERAL so the client's static AST reader can extract it; --self-test re-derives it from
# the sinc² integral so it can never silently drift.
_MAIN_LOBE_FRAC_ARGS = [
    "sidelobes",
    1.000000, 0.950401, 0.934203, 0.926212, 0.921459, 0.918309, 0.916069, 0.914395, 0.913096,
]
_MAIN_LOBE_FRAC = _MAIN_LOBE_FRAC_ARGS[1:]     # the fractions alone, for the runtime lookup


def main_lobe_frac(sidelobes: int) -> float:
    """The main-lobe power fraction for `sidelobes` sidelobes kept (nearest, clamped)."""
    n = max(0, min(len(_MAIN_LOBE_FRAC) - 1, int(sidelobes)))
    return _MAIN_LOBE_FRAC[n]


# Power-quantity conversion laws this signal OFFERS the calibration editor (they ride through
# the static argspec to the client; the runtime folds --power in the base density exactly like
# the chirp). Both convert the shared measured spectral density (dBm/Hz) to an absolute power
# (dBm). Constants are LITERAL (read statically): k = 70 = 60 + 10·log10(CAL_MEAS_BW_MHZ). Full
# signal power is the bandwidth-invariant total (same as the flat sweep's total at this gain);
# main-lobe power is that minus the fixed sinc² offset, KEYED on --sidelobes via the hidden
# `main_lobe_frac` derived field so it tracks the truncation. `rep` = the value at the default
# sidelobe count, for range read-outs shown before a live --sidelobes is known.
CAL_POWER_LAWS = [
    {"id": "full_power", "name": "Full signal power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.0},
    {"id": "main_lobe_power", "name": "Main-lobe power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.0,
     "param": "main_lobe_frac", "coeff": 10.0, "ref": 1.0, "rep": 0.926212},
]


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task parameter:
# the calibration is measured at THIS amplitude, so a unit calibrated at a different
# amplitude no longer matches. calkit detects that at load and runs UNCALIBRATED with a loud
# warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum.
HW_MAX_GAIN_DB = 89.75


# ── Power map: the unit's injected calibration curve if present, else the baked
#    constants above (relative gain only) ─────────────────────────────────────────

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script
    and main share one — and so --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6       # the max; master clock pinned 1:1. A sweep has no chip grid,
                             # so this is simply as wide as the B200 goes.
OTW_FORMAT = "sc8"           # over-the-wire; halves USB load
MAX_ROAM_MHZ = 27.0          # the occupied half-width ±(sidelobes+1)·chip_rate must stay
                             # inside ±Nyquist (±30.69 MHz) with margin

# One whole sweep is precomputed into this many samples and looped (kept as the fm_chirp_tx
# floor — ~2 MB as fc32). For this signal the WHOLE buffer is one continuous sweep (not a
# short period tiled up), so every sample is distinct trajectory.
BUFFER_SAMPS = 1 << 18       # 262144 samples ≈ 2 MB as fc32

# Tiny dwell floor added to the sinc² target so the sweep CREEPS through the nulls (where
# sinc²=0) instead of teleporting — keeps the CDF strictly increasing and the trajectory
# finite. Sets the realised null floor (~10·log10 of this), capped by the buffer length.
DWELL_FLOOR = 1e-3

# Passband filter (always on, not a parameter): unity passband gain, passband = the occupied
# band ±(sidelobes+1)·chip_rate (the outermost null), fixed transition skirt. Band-limits the
# sweep and cleans the turnaround; the unity passband leaves the sinc² shape untouched.
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

# Chip-rate / sidelobe defaults + bounds.
CHIP_RATE_DEFAULT_MCPS = 1.023   # the GPS C/A chip rate; first null at ±1.023 MHz
CHIP_RATE_MIN_MCPS = 0.05
CHIP_RATE_MAX_MCPS = 12.0        # covers the 10.23 Mcps family; runtime still bounds the roam
SIDELOBES_DEFAULT = 3
SIDELOBES_MAX = 8


# ── Sinc² dwell-shaped frequency trajectory ─────────────────────────────────────

def roam_hz(chip_rate_hz: float, sidelobes: int) -> float:
    """The occupied half-width: the outermost sinc² null kept = ±(sidelobes+1)·chip_rate."""
    return (int(sidelobes) + 1) * float(chip_rate_hz)


def check_roam(chip_rate_hz: float, sidelobes: int) -> None:
    """Raise ValueError if the occupied band would exceed the usable baseband."""
    r = roam_hz(chip_rate_hz, sidelobes) / 1e6
    if r > MAX_ROAM_MHZ:
        raise ValueError(
            f"occupied band ±{r:g} MHz (= (sidelobes+1)·chip_rate) exceeds the maximum "
            f"±{MAX_ROAM_MHZ:g} MHz — lower --chip-rate or --sidelobes.")


def _sinc2_dwell_freq(chip_rate_hz: float, sidelobes: int, n: int):
    """Instantaneous frequency f[k] (Hz) for a length-n CONSTANT-ENVELOPE buffer whose
    time-averaged PSD is sinc²(f / chip_rate), truncated at the (sidelobes+1)th null each
    side and swept symmetrically (out and back) so the loop closes with no reset.

    Uses the inverse-CDF (quantile) of the target: dwell density ∝ sinc², so the tone
    lingers at the centre. Returns the zero-mean trajectory (phase then closes over the
    loop, verified in --self-test)."""
    import numpy as np
    r = roam_hz(chip_rate_hz, sidelobes)
    g = np.linspace(-r, r, 20001)
    S = np.sinc(g / chip_rate_hz) ** 2                 # np.sinc(x)=sin(πx)/(πx); nulls at k·chip_rate
    S = np.maximum(S, DWELL_FLOOR * S.max())           # floor so the CDF is strictly increasing
    C = np.cumsum(S)
    C = (C - C[0]) / (C[-1] - C[0])                    # CDF, 0..1
    half = n // 2
    w = (np.arange(half) + 0.5) / half                 # uniform time index across the up-sweep
    up = np.interp(w, C, g)                            # inverse-CDF: dwell ∝ sinc²
    f = np.concatenate([up, up[::-1]])                 # out and back → symmetric, continuous at the wrap
    if len(f) < n:                                     # odd n: hold the turnaround one sample
        f = np.concatenate([f, f[-1:]])
    f = f[:n]
    return f - f.mean()                                # exact zero mean → phase closes over the loop


def build_enveloped_buffer(chip_rate_hz: float, sidelobes: int):
    """A complex64 constant-modulus buffer of one whole sinc²-dwell sweep that loops with no
    seam. Amplitude is applied live downstream. Returns (iq, freq_trajectory)."""
    import numpy as np
    f = _sinc2_dwell_freq(chip_rate_hz, sidelobes, BUFFER_SAMPS)
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
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, width_hz: float, trans_hz: float):
    """Circularly filter the looped sweep to a `width_hz`-wide passband centred on the carrier
    (the filter passes ±width_hz/2). Circular convolution keeps the result exactly periodic,
    so the filtered loop has no seam; unity passband gain leaves the in-band shape unchanged.
    Returns (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = float(width_hz) / 2.0
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test: seam closure + constant envelope + sinc² shape, no hardware ──────────

def _self_test() -> int:
    import cmath
    import math
    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — cannot run the self-test)")
        return 0

    ok = True
    chip_rate_hz, sidelobes = 1.023e6, 3

    # 1) seamless phase closure over the looped buffer
    f = _sinc2_dwell_freq(chip_rate_hz, sidelobes, BUFFER_SAMPS)
    phase = (2.0 * math.pi / SAMP_RATE_HZ) * np.cumsum(f)
    iq = np.exp(1j * phase)
    expected = 2 * math.pi / SAMP_RATE_HZ * f[0]
    measured = cmath.phase(iq[0] / iq[-1])
    seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
    good = seam_err < 1e-9
    ok = ok and good
    print(f"seam closure : err={seam_err:.2e} rad [{'OK' if good else 'FAIL'}]")

    # 2) constant envelope (pre-filter) — dwell shaping, no amplitude taper
    env = np.abs(iq)
    env_ok = float(np.max(np.abs(env - 1.0))) < 1e-6
    ok = ok and env_ok
    print(f"constant env : max|1-|iq||={np.max(np.abs(env-1.0)):.1e} [{'OK' if env_ok else 'FAIL'}]")

    # 3) the averaged PSD is sinc²: first sidelobe ≈ -13.3 dB, nulls present, peak at centre
    filt, taps, fp = filter_buffer(
        build_enveloped_buffer(chip_rate_hz, sidelobes)[0],
        2 * roam_hz(chip_rate_hz, sidelobes), FILTER_TRANSITION_HZ)
    X = np.abs(np.fft.fftshift(np.fft.fft(filt))) ** 2
    ff = np.fft.fftshift(np.fft.fftfreq(len(filt), 1.0 / SAMP_RATE_HZ))
    sm = np.convolve(X, np.ones(15) / 15, "same")
    sm = 10 * np.log10(sm / sm.max() + 1e-30)

    def at(fq):
        return float(sm[np.argmin(np.abs(ff - fq))])

    main = at(0.0)
    side1 = at(1.43 * chip_rate_hz)      # first sinc² sidelobe peak ≈ 1.43·chip_rate
    null1 = at(1.0 * chip_rate_hz)       # first null at ±chip_rate
    shape_ok = (abs(main) < 0.5) and (abs(side1 - (-13.3)) < 3.0) and (null1 < -18.0)
    ok = ok and shape_ok
    print(f"sinc² shape  : main {main:+.1f} dB, 1st sidelobe {side1:+.1f} dB (ideal -13.3), "
          f"1st null {null1:+.1f} dB [{'OK' if shape_ok else 'FAIL'}]")

    # 4) the sweep stays INSIDE the occupied band ±roam — it never dwells outside the filter
    #    passband (the roam confines the trajectory by construction, so no time is wasted).
    r = roam_hz(chip_rate_hz, sidelobes)
    excess = float(np.max(np.abs(f))) - r
    conf_ok = excess <= 1.0                       # within 1 Hz of the edge (sub-Hz mean residual)
    ok = ok and conf_ok
    print(f"confined     : max|f|={np.max(np.abs(f))/1e6:.4f} MHz vs roam {r/1e6:.4f} MHz "
          f"(excess {excess:+.2e} Hz) [{'OK' if conf_ok else 'FAIL'}]")

    # 5) the baked main-lobe-fraction table matches the sinc² integral (can't silently drift)
    grid = np.linspace(-9, 9, 400001); s2 = np.sinc(grid) ** 2
    ml = np.sum(s2[np.abs(grid) < 1])
    frac_ok = True
    for n, baked in enumerate(_MAIN_LOBE_FRAC):
        derived = ml / np.sum(s2[np.abs(grid) < (n + 1)])
        frac_ok = frac_ok and abs(derived - baked) < 1e-3
    ok = ok and frac_ok
    print(f"main-lobe tbl: baked ≡ sinc² integral (≤1e-3) [{'OK' if frac_ok else 'FAIL'}]")

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float, amplitude: float):
    """The looped sweep is streamed from RAM by a C++ blocks.vector_source_c (repeat=True),
    NOT a file_source and NOT a Python source (see fm_chirp_tx.py for the full write-up: a
    file_source races GNU Radio on a live swap, a Python source can't hold 61.38 Msps on a
    Pi; vector_source_c is C++/GIL-free with no file). A live shape change swaps the buffer
    with set_data() under top-block lock()/unlock(), so the stream pauses only for the swap."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class EnvelopedSweepTx(gr.top_block):
        def __init__(self):
            super().__init__("Enveloped Sweep TX")
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

        # ── live setters (called from the main loop, device-safe) ──────────────
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

    return EnvelopedSweepTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Enveloped Sweep transmitter — a CONSTANT-ENVELOPE swept tone whose averaged "
               "spectrum is shaped like a sinc² (energy concentrated at the centre), by dwell "
               "time only (no amplitude taper, no crest-factor penalty). Fixed 61.38 MHz / sc8, "
               "looped buffer, always-on unity passband filter. Reuses the regular sweep's "
               "(\"Chirp/Sweep\") calibration — same total power at a given gain — and offers "
               "full-signal and main-lobe power (dBm). Uncalibrated it runs on a relative gain. "
               "Authorised, shielded setups only.")
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
        .number("-Chip-rate", "--chip-rate", unit="Mcps",
                min=CHIP_RATE_MIN_MCPS, max=CHIP_RATE_MAX_MCPS, default=CHIP_RATE_DEFAULT_MCPS,
                required=False, live=True,
                help="Width of the sinc² spectrum: the first spectral null sits at "
                     "±chip-rate (Mcps ≡ MHz), as for a BPSK/PRN signal at that chip rate. "
                     "Higher = wider. Live (regenerates the sweep).")
        .number("-Sidelobes", "--sidelobes", min=0, max=SIDELOBES_MAX, step=1,
                default=SIDELOBES_DEFAULT, required=False, live=True,
                help="How many sinc² sidelobes to keep each side (0 = the main lobe only). "
                     "The occupied band is ±(sidelobes+1)·chip-rate. Live (regenerates).")
        .derived("-Main-lobe-fraction", name="main_lobe_frac", hidden=True,
                 formula={"table": _MAIN_LOBE_FRAC_ARGS},
                 help="Fraction of the sinc² power inside the main lobe at the current sidelobe "
                      "count. Feeds the main-lobe-power calibration law; not shown.")
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
    chip_rate_hz = float(args.chip_rate) * 1e6
    sidelobes = int(args.sidelobes)
    try:
        check_roam(chip_rate_hz, sidelobes)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Current "shape" (the regeneration-requiring params) — mutated by live changes. Defined
    # before the gain fold so the calibration's main-lobe-power law can read the live fraction.
    shape = {"chip_rate_hz": chip_rate_hz, "sidelobes": sidelobes}

    def pwr_params() -> dict:
        """Live keyed-parameter values the calibration's power laws read: the sinc² main-lobe
        fraction at the current sidelobe count, so a main-lobe-power reading / cap tracks
        --sidelobes. Harmless when the calibration doesn't key on it (the map ignores it)."""
        return {"main_lobe_frac": main_lobe_frac(shape["sidelobes"])}

    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only.
    pmap = power_map()
    amplitude = pmap.amplitude
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:                         # calibrated: the authored absolute --power
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz, params=pwr_params())
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    def make_current():
        """The buffer for the current shape: the sinc²-dwell sweep, band-limited to the
        occupied band ±(sidelobes+1)·chip-rate. Returns (iq, finfo)."""
        base, _f = build_enveloped_buffer(shape["chip_rate_hz"], shape["sidelobes"])
        r = roam_hz(shape["chip_rate_hz"], shape["sidelobes"])
        filt, taps, fp = filter_buffer(base, 2 * r, FILTER_TRANSITION_HZ)
        return filt, {"taps": taps, "edge_hz": fp, "roam_hz": r}

    iq, finfo = make_current()

    tb = _build_top_block(
        initial_iq=iq, center_freq_hz=center_freq_hz, gain_db=gain_db, amplitude=amplitude)

    # Track the live transmit frequency and (in absolute mode) the held target power, so a
    # live retune can re-map --power at the new frequency on a frequency-dependent chain.
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": center_freq_hz, "power": _target_power}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def regenerate():
        iq, fi = make_current()
        tb.swap(iq)                            # atomic in-RAM buffer swap under top-block lock
        return fi

    T_ms = BUFFER_SAMPS / SAMP_RATE_HZ * 1e3

    print("── Enveloped Sweep TX ──────────────────────────────────────")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  shape          : sinc²  (chip-rate {chip_rate_hz/1e6:g} MHz, "
          f"{sidelobes} sidelobe(s)/side)")
    print(f"  occupied band  : ±{finfo['roam_hz']/1e6:g} MHz")
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
            # A frequency-dependent calibration re-scales --power with frequency.
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
        elif name in ("chip_rate", "sidelobes"):
            # A shape change: validate the occupied band, then rebuild + swap. An invalid band
            # is left unapplied (keep the last good shape — the GUI flags it too).
            new_cr = float(value) * 1e6 if name == "chip_rate" else shape["chip_rate_hz"]
            new_sl = int(value) if name == "sidelobes" else shape["sidelobes"]
            try:
                check_roam(new_cr, new_sl)
            except ValueError:
                ctrl.report(name, value)          # keep the last good shape
                return
            shape["chip_rate_hz"] = new_cr
            shape["sidelobes"] = new_sl
            regenerate()
            ctrl.report(name, value)
            # A --sidelobes change moves the main-lobe fraction, so re-map a held --power in case
            # the calibration's cap keys on it (a no-op for a total-power cap — constant envelope).
            if name == "sidelobes" and state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"], params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power", round(pmap.power_for_gain(
                        tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    from paramkit.txhealth import watch_flowgraph
    _health = watch_flowgraph(tb, stop)   # RF-fault: a silent GR halt -> non-zero exit (rf-fault-recovery.md §5.1)
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                apply_change(change.name, change.value)
            time.sleep(0.1)
    finally:
        stop.set()   # an intentional teardown: tb.stop() ends the watcher's tb.wait(), and a set
                     # stop keeps an exception out of the loop a plain crash, not a false RF fault
        ctrl.close()
        tb.stop()
        tb.wait()
    return 1 if _health.faulted else 0


if __name__ == "__main__":
    raise SystemExit(main())

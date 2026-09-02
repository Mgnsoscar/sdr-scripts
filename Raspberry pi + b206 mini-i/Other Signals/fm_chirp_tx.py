#!/usr/bin/env python3
"""
FM-chirp transmitter for GNU Radio + UHD (Ettus B200-mini family).

A sawtooth frequency sweep, precomputed and replayed from RAM so a Raspberry Pi can
sustain a high sample rate (40–60 MS/s) — the same recipe as gps_l1ca_tx.py, applied
to the swept-tone chirp from TriangleChirp.

The instantaneous frequency is  f(t) = sawtooth(t) · (sweep_bw / 2), i.e. it
sweeps ±sweep_bw/2 around the carrier, and the modulating sawtooth repeats at
`sweep_rate`.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable +
   attenuators) on frequencies you are LICENSED / AUTHORISED to use.

Why this reaches 40–60 MS/s on a Pi (see gps_l1ca_tx.py for the full write-up)
────────────────────────────────────────────────────────────────────────────
  1. PRECOMPUTE + LOOP — one whole sweep is built once and replayed with a C++
     blocks.vector_source_c(repeat=True); no per-sample NumPy in a work() block
     (that per-sample synthesis is exactly what capped the old TriangleChirp).
  2. sc8 over the wire — halves USB payload.
  3. Quiet — UHD fastpath/console logging off, and the task runs with
     PYTHONUNBUFFERED=0 (configs/tasks.yaml) so nothing is written mid-stream.
  4. 1:1 master clock — master_clock_rate is pinned to the sample rate, so UHD
     runs the AD9361 with no FPGA resampling and no rate coercion.

Seamless looping of a SWEPT buffer
──────────────────────────────────
A frequency sweep only loops without a seam if the accumulated phase closes at
the wrap. Two things guarantee it here:
  • the buffer holds a whole number of sweep periods (integer samples/period), and
  • the swept frequency is forced to EXACT zero mean, so the phase integrated
    over the buffer returns to its start (verified to ~1e-14 rad; see --self-test).
The tiny mean removed is a fraction of a hertz for the symmetric waveforms, so
the carrier is unaffected.

Live tuning (retune while transmitting, via paramkit.live)
──────────────────────────────────────────────────────────
    freq       → UHD tune_request        (instant)
    power      → dBm → set_gain          (instant — see the USER CALIBRATION block)
    rf         → on/off mute/unmute      (instant — gain AND amplitude to 0 / back)
    bw         → rebuild buffer + swap    ┐ shape changes: regenerate one sweep
    sweep_rate → rebuild buffer + swap    ┘ in RAM and set_data() it under the
                                            top-block lock (one brief seam at the
                                            swap, then it loops clean)
Regeneration runs on the control thread; the flowgraph keeps streaming the old
buffer on GNU Radio's scheduler threads until the swap, so RF never stops.

Fixed radio setup
─────────────────
Sample rate is fixed at 61.38 MHz (the B200's ceiling; a chirp has no chip grid, so this
is simply as wide as the sweep can go), over-the-wire sc8, and baseband amplitude 0.5 (the
amplitude the calibration is measured at). None are parameters. An optional digital passband
filter (--filter/--passband/--transition) brick-wall band-limits the sweep and strips the
sawtooth/square reset splatter; --passband is the TOTAL filter width (the filter passes
±passband/2), so set it a hair above bw to keep the whole sweep.

CLI
───
    fm_chirp_tx.py --freq 1575.42 --bw 20 --rate 200 --power -30
    fm_chirp_tx.py --freq 1575.42 --bw 20 --gain 60 --filter on --passband 22   # clean band-limit
    fm_chirp_tx.py --self-test        # verify seamless phase closure (+ filter), no hardware
    fm_chirp_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only)
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "fm_chirp"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# (a cable/antenna whose loss varies with frequency) has a --power scale that MOVES with
# frequency, so the map is folded at THIS param's value — and it is live, so retuning the
# chirp's centre re-scales --power on the fly. The client folds the --power range shown in
# the Run / sequence form at the same frequency. See calkit / calibration-v2.
CAL_FREQ_PARAM = "freq"

# Sweep bandwidth (MHz) the spectral-density calibration is MEASURED at. It anchors the two
# conversion laws below: one density measurement at this bandwidth yields BOTH total power and
# density-at-any-bandwidth, with no second measurement. If you calibrate at a different sweep
# width, change this and the two constants derived from it (k and ref).
CAL_MEAS_BW_MHZ = 10.0

# Power-quantity conversion laws this signal offers the calibration editor (see
# docs/calibration-v2.md §13 in sdr-agent). A chirp's baseband is CONSTANT-AMPLITUDE, so its
# TOTAL (full-bandwidth) power depends only on gain — widening the sweep spreads the same power
# over more spectrum (density drops), it does NOT add power. So from one spectral-density
# measurement taken at CAL_MEAS_BW_MHZ, both readings below are exact at ANY live sweep width:
#
#   Full-bandwidth power = density + 10·log10(CAL_MEAS_BW_MHZ)     ← CONSTANT (bandwidth-invariant)
#   Spectral density(bw) = density − 10·log10(bw / CAL_MEAS_BW_MHZ) ← tracks the live --bw
#
# `k` is the constant dB the reading adds to the measured value; `param`/`coeff`/`ref` add a
# `coeff·log10(param/ref)` term. Both encode CAL_MEAS_BW_MHZ (10): k = 10·log10(10) = 10 for the
# total-power law; ref = 10 for the density-restatement law. The laws are only OFFERED here; the
# operator picks which is --power's quantity per unit in the editor (and can switch units on the
# --power field between the two, since they differ by exactly 10·log10(bw)). The chosen law is
# embedded in that unit's calibration doc. `rep` is a representative --bw for the range read-outs
# shown before a live --bw is known.
CAL_POWER_LAWS = [
    {"id": "fbw_power", "name": "Full-bandwidth (total) power",
     "in": "density", "out": "abs",
     "k": 10.0, "rep": 10.0},                                # +10·log10(CAL_MEAS_BW_MHZ)
    {"id": "psd_live", "name": "Spectral density (at live sweep bw)",
     "in": "density", "out": "density",
     "param": "bw", "coeff": -10.0, "ref": 10.0, "rep": 10.0},  # −10·log10(bw / CAL_MEAS_BW_MHZ)
]


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75



# ── Power map: the unit's injected calibration curve if present, else the baked
#    constants above (identical to the old single-anchor slope-1 behaviour) ────────

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script and
    main share one — and so --power's schema bounds match the real operating range
    (calibrated → e.g. EIRP; else the baked SDR-port range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6       # the max; master clock pinned 1:1. A chirp has no chip grid,
                             # so this is just "as wide as the B200 goes" for the sweep.
OTW_FORMAT = "sc8"           # over-the-wire; halves USB load
MAX_SWEEP_BW_MHZ = 55.0      # sweep is ±bw/2; keep it inside ±Nyquist (±30.69 MHz) with margin

# ── Constants ─────────────────────────────────────────────────────────────────

WAVEFORMS = ["Sine", "Triangle", "Sawtooth", "Square"]

# Tile the single sweep period up to at least this many samples so the looping source
# wraps infrequently at high rate (whole periods only → still seamless).
MIN_BUFFER_SAMPS = 1 << 18   # 262144 samples ≈ 2 MB as fc32

# Named GNSS carriers (same list as the original TriangleChirp), in MHz.
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

# Filter presets: {label: TOTAL passband width in MHz}. A chirp is broadband by design (it
# sweeps ±bw/2, i.e. bw wide), so the passband is a total width centred on the carrier — the
# filter passes ±passband/2: set it a hair above bw to brick-wall the sweep and strip the
# sawtooth/square reset splatter, or below bw to gate the sweep into a sub-band. Clamped to
# Nyquist (the full 61.38 MHz).
MIN_PASSBAND_MHZ = 1.0
MAX_PASSBAND_MHZ = 61.2
PASSBAND_PRESETS = {
    "20 MHz": 20.0, "30 MHz": 30.0, "40 MHz": 40.0, "56 MHz": 56.0,
}


# ── Modulating waveform (NumPy, for the buffer builder) ────────────────────────

def _mod_waveform(kind: str, p):
    """Value in [-1, 1] for phase p in [0, 1). Zero-mean over a period.
    p is a NumPy array; the result modulates the instantaneous frequency."""
    import numpy as np
    if kind == "Sine":
        return np.sin(2 * np.pi * p)
    if kind == "Triangle":
        return 1.0 - 2.0 * np.abs(2.0 * p - 1.0)          # -1 → +1 → -1
    if kind == "Sawtooth":
        return 2.0 * p - 1.0                               # -1 → +1 (reset)
    if kind == "Square":
        return np.where(np.sin(2 * np.pi * p) >= 0, 1.0, -1.0)
    raise ValueError(f"unknown waveform {kind!r}")


def build_chirp_buffer(waveform: str, sweep_bw_hz: float, sweep_rate_hz: float):
    """Build a complex64 baseband buffer at the fixed SAMP_RATE_HZ holding a whole number
    of sweep periods that loops with no seam. Unit magnitude (amplitude is applied live
    downstream). Returns (iq, samples_per_period, reps, actual_sweep_rate_hz)."""
    import numpy as np

    n_per = max(2, int(round(SAMP_RATE_HZ / sweep_rate_hz)))
    actual_sweep_rate = SAMP_RATE_HZ / n_per       # rate quantised to the grid

    p = np.arange(n_per, dtype=np.float64) / n_per
    freq = _mod_waveform(waveform, p) * (sweep_bw_hz / 2.0)
    freq = freq - freq.mean()                      # exact zero-mean → phase closes
    phase = (2.0 * np.pi / SAMP_RATE_HZ) * np.cumsum(freq)
    period = np.exp(1j * phase).astype(np.complex64)

    reps = max(1, -(-MIN_BUFFER_SAMPS // n_per))   # ceil division
    iq = np.tile(period, reps)
    return iq, n_per, reps, actual_sweep_rate


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
    """Circularly filter the looped chirp buffer to a `width_hz`-wide passband centred on the
    carrier (the filter passes ±width_hz/2). Circular convolution keeps the result exactly
    periodic, so the filtered loop has no seam; unity passband gain leaves the in-band sweep's
    power unchanged. Filtering band-limits the sweep (and removes reset splatter), so the result
    is no longer constant-modulus. Returns (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = float(width_hz) / 2.0
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test: seamless phase closure, no NumPy / no hardware ──────────────────

def _self_test() -> int:
    import cmath
    import math

    def wave(kind, p):
        if kind == "Sine":     return math.sin(2 * math.pi * p)
        if kind == "Triangle": return 1.0 - 2.0 * abs(2.0 * p - 1.0)
        if kind == "Sawtooth": return 2.0 * p - 1.0
        if kind == "Square":   return 1.0 if math.sin(2 * math.pi * p) >= 0 else -1.0
        raise ValueError(kind)

    fs, bw, rate = SAMP_RATE_HZ, 20e6, 200e3
    ok = True
    for kind in WAVEFORMS:
        n = max(2, round(fs / rate))
        freq = [wave(kind, i / n) * (bw / 2.0) for i in range(n)]
        m = sum(freq) / n
        freq = [f - m for f in freq]
        acc, phase = 0.0, []
        for f in freq:
            acc += 2 * math.pi / fs * f
            phase.append(acc)
        iq = [cmath.exp(1j * ph) for ph in phase]
        expected = 2 * math.pi / fs * freq[0]
        measured = cmath.phase(iq[0] / iq[-1])
        seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
        good = seam_err < 1e-9
        ok = ok and good
        print(f"{kind:9s}: n={n} seam_err={seam_err:.2e} rad [{'OK' if good else 'FAIL'}]")
    print("SEAMLESS — ALL WAVEFORMS PASSED" if ok else "SELF-TEST FAILED")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    # Filter a 20 MHz sawtooth sweep with a ±11 MHz passband: the in-band sweep power (out to
    # bw/2 = 10 MHz) survives, the reset splatter beyond the passband is cut, and peak stays safe.
    base, _n, _r, _ar = build_chirp_buffer("Sawtooth", 20e6, 200e3)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, width_hz=22.0e6, trans_hz=1.0e6)   # ±11 MHz edge
    inband = 10 * np.log10(band(filt, 0, 10e6) / band(base, 0, 10e6))
    cut = 10 * np.log10(band(filt, 16e6, 28e6) / max(band(base, 16e6, 28e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(inband) < 0.2 and cut < -30 and peak * AMPLITUDE < 1.0
    print(f"filter (±11 MHz on a 20 MHz sweep, {taps} taps): in-band {inband:+.3f} dB, "
          f"out-of-band {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, lo_offset_hz: float,
                     gain_db: float, amplitude: float):
    """The looped sweep is streamed from RAM by a C++ blocks.vector_source_c (repeat=True),
    NOT a file_source, and NOT a Python source: swapping a file_source live (open()) races GNU
    Radio and THROWS "fread error" (the source dies, radio silent); a Python source has no file
    but its work() runs under the GIL and can't hold 61.38 Msps on a Pi. vector_source_c is C++
    (GIL-free) with no file, so it streams smoothly with no fread risk. A live waveform/filter
    change swaps the buffer with set_data() under top-block lock()/unlock() (set_data has no
    internal lock), so it's never freed under a running read — the stream pauses only for the
    swap, only on a change."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        # vector_source_c wants a contiguous complex64 buffer; the filtered loop may not be.
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class ChirpTx(gr.top_block):
        def __init__(self):
            super().__init__("FM chirp TX")
            self._freq_hz = center_freq_hz
            self._lo_hz = lo_offset_hz

            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]),
            )
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self._retune()
            self.usrp.set_gain(gain_db, 0)

            # repeat=True loops the buffer in C++ (no per-wrap Python), vlen=1, no tags.
            self.src = blocks.vector_source_c(_vec(initial_iq), True, 1, [])
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def _retune(self) -> None:
            self.usrp.set_center_freq(
                uhd.tune_request(self._freq_hz, self._lo_hz), 0)

        def set_center_frequency(self, hz: float) -> None:
            self._freq_hz = hz
            self._retune()

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def swap(self, iq) -> None:
            # set_data() reassigns (and frees) the source's buffer with no internal lock, so it
            # must not run while work() is reading. lock()/unlock() quiesces the flowgraph
            # around the swap — the only moment the stream pauses, and only on a change.
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

    return ChirpTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("FM-chirp transmitter (sawtooth sweep) — fixed 61.38 MHz "
               "/ sc8, looped buffer, optional power-preserving digital passband filter. Level "
               "is set in dBm via the unit's calibration; uncalibrated it runs on a relative "
               "gain. Authorised, shielded setups only.")
        # The sweep band is entered one of two ways; the mode reveals its own fields.
        .choice("-Band-mode", "--band-mode",
                options={"Centre + width": "center_bw", "Start / stop": "start_stop"},
                default="center_bw", required=False,
                help="How the sweep band is entered: a centre carrier + width, or absolute "
                     "start/stop edges (carrier = their midpoint, width = their span). Fixed "
                     "at launch.")
        # ── center_bw mode ──────────────────────────────────────────────────────────
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=1575.42, required=False, live=True,
                show_when={"band_mode": "center_bw"},
                help="RF carrier in MHz. Live.")
        # ── start_stop mode ─────────────────────────────────────────────────────────
        .number("-Start-frequency", "--start", unit="MHz", min=70.0, max=6000.0,
                step=0.01, default=1570.42, required=False, live=True,
                show_when={"band_mode": "start_stop"},
                help="Sweep start — the low RF edge, in MHz. Live.")
        .number("-Stop-frequency", "--stop", unit="MHz", min=70.0, max=6000.0,
                step=0.01, default=1580.42, required=False, live=True,
                show_when={"band_mode": "start_stop"},
                help="Sweep stop — the high RF edge, in MHz. Live.")
        .derived("-Carrier", name="band_center", unit="MHz",
                 formula={"center": ["start", "stop"]}, is_freq=True,
                 show_when={"band_mode": "start_stop"},
                 help="Carrier the LO tunes to = midpoint of start/stop. Power is calibrated "
                      "here.")
        .derived("-Sweep-width", name="band_span", unit="MHz",
                 formula={"span": ["start", "stop"]}, min=0.001, max=MAX_SWEEP_BW_MHZ,
                 show_when={"band_mode": "start_stop"},
                 help="Resulting sweep width = stop − start. Must stay within the SDR's "
                      "maximum sweep width.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration (folded at the current carrier) and snaps to its achievable "
                     "grid; ignored if --gain is given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .number("-Sweep-BW", "--bw", unit="MHz", min=0.001, max=MAX_SWEEP_BW_MHZ, default=20.0,
                required=False, live=True, show_when={"band_mode": "center_bw"},
                help="Peak-to-peak sweep width; f sweeps ±bw/2 around the carrier. Live "
                     "(regenerates the loop).")
        .number("-Sweep-rate", "--rate", unit="kHz", min=0.1, max=5000.0,
                default=200.0, required=True, live=True,
                help="How fast the sweep repeats, in kHz. Live (regenerates the loop).")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain). Set "
                     "the passband a hair above bw to brick-wall the sweep and strip reset "
                     "splatter. Live.")
        .number("-Passband", "--passband", unit="MHz",
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=30.0,
                presets=PASSBAND_PRESETS, required=False, live=True,
                help="Total passband width kept, centred on the carrier (MHz) — the filter passes "
                     "±passband/2. Clamped to Nyquist. ≥ bw keeps the whole sweep; < bw gates it "
                     "into a sub-band. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=8.0, default=1.0,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loop).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Band resolution (center+width ↔ start/stop) ───────────────────────────────────

def resolve_band(band_mode: str, freq, bw, start, stop):
    """Resolve the sweep band from whichever mode is selected into a canonical
    (center_freq_hz, sweep_bw_hz). In 'start_stop' the carrier is the midpoint and the
    width is the span; in 'center_bw' they are given directly. Raises ValueError with a
    clear message on an unusable band (missing edges, zero/negative or over-wide span) —
    the same conditions the GUI flags live via the derived width field."""
    if band_mode == "start_stop":
        if start is None or stop is None:
            raise ValueError("start_stop mode needs both --start and --stop (MHz).")
        lo, hi = sorted((float(start), float(stop)))
        span = hi - lo
        if span <= 0:
            raise ValueError("--start and --stop must differ (the sweep has zero width).")
        if span > MAX_SWEEP_BW_MHZ:
            raise ValueError(
                f"start/stop span {span:g} MHz exceeds the maximum sweep width "
                f"{MAX_SWEEP_BW_MHZ:g} MHz — narrow the start/stop range.")
        return (lo + hi) / 2.0 * 1e6, span * 1e6
    # center_bw (default)
    if freq is None or bw is None:
        raise ValueError("center_bw mode needs --freq and --bw (MHz).")
    return float(freq) * 1e6, float(bw) * 1e6


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    try:
        center_freq_hz, sweep_bw_hz = resolve_band(
            getattr(args, "band_mode", "center_bw"),
            getattr(args, "freq", None), getattr(args, "bw", None),
            getattr(args, "start", None), getattr(args, "stop", None))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present,
    # so you can command a gain directly or measure output power at it.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:                         # calibrated: the authored absolute --power
        # Fold the calibration at the transmit frequency so a frequency-dependent chain maps
        # --power on the right scale (the chirp can retune live — see apply_change).
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz,
                                      params={"bw": sweep_bw_hz / 1e6})
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    # Current "shape" (the regeneration-requiring params) — mutated by live changes.
    # bw_hz comes from resolve_band (either --bw directly, or the start/stop span).
    band_mode = getattr(args, "band_mode", "center_bw")
    shape = {"waveform": "Sawtooth",              # fixed — the only sweep shape this script emits
             "bw_hz": sweep_bw_hz,
             "rate_hz": args.rate * 1e3,               # --rate is in kHz
             "filter_on": getattr(args, "filter", "off") == "on",
             "width_hz": float(getattr(args, "passband", 30.0) or 30.0) * 1e6,  # total width
             "trans_hz": float(getattr(args, "transition", 1.0) or 1.0) * 1e6}

    def pwr_params():
        """Live parameter values a power-quantity bridge may key on (docs/calibration-v2.md
        §13). The full-bandwidth-power law reads the current sweep bandwidth in MHz, so a
        bridged --power tracks --bw as it is tuned. Harmless when the calibration uses no
        bridge (the map ignores it)."""
        return {"bw": shape["bw_hz"] / 1e6}

    def make_current():
        """The buffer for the current shape (the base sweep, optionally band-limited).
        Returns (iq, n_per, reps, actual_rate, finfo)."""
        base, n_per, reps, actual = build_chirp_buffer(
            shape["waveform"], shape["bw_hz"], shape["rate_hz"])
        if not shape["filter_on"]:
            return base, n_per, reps, actual, {"on": False}
        filt, taps, fp = filter_buffer(base, shape["width_hz"], shape["trans_hz"])
        return filt, n_per, reps, actual, {"on": True, "taps": taps, "edge_hz": fp,
                                           "trans_hz": shape["trans_hz"]}

    iq, n_per, reps, actual_rate, finfo = make_current()

    tb = _build_top_block(
        initial_iq=iq, center_freq_hz=center_freq_hz,
        lo_offset_hz=0.0, gain_db=gain_db, amplitude=amplitude)

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    # Track the live transmit frequency (Hz) and (in absolute mode) the held target power, so
    # a live retune can re-map --power at the new frequency on a frequency-dependent chain.
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": center_freq_hz, "power": _target_power,
             "start": getattr(args, "start", None), "stop": getattr(args, "stop", None)}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def regenerate() -> float:
        iq, _n, _r, actual, _fi = make_current()
        tb.swap(iq)                            # atomic in-RAM buffer swap under top-block lock
        return actual

    def _fmt_band(info):
        if not info.get("on"):
            return "off (full sweep)"
        return (f"on — passband ±{info['edge_hz']/1e6:.2f} MHz, "
                f"{info['trans_hz']/1e6:g} MHz transition, {info['taps']} taps")

    print("── FM chirp TX ─────────────────────────────────────────────")
    print(f"  band mode      : {band_mode}")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    if band_mode == "start_stop":
        _lo, _hi = sorted((float(args.start), float(args.stop)))
        print(f"  sweep edges    : {_lo:g} … {_hi:g} MHz (midpoint carrier)")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  waveform       : Sawtooth")
    print(f"  sweep bw       : {sweep_bw_hz/1e6:g} MHz (±{sweep_bw_hz/2e6:g} MHz)")
    print(f"  sweep rate     : requested {args.rate:g} kHz, "
          f"got {actual_rate/1e3:.3f} kHz ({n_per} samples/period × {reps} reps)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz, params=pwr_params()):.2f} dBm")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:                       # e.g. calibration amplitude != this
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")   # script's fixed amplitude
    print(f"  filter         : {_fmt_band(finfo)}")
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
            # A frequency-dependent calibration re-scales --power with frequency, so re-map
            # the held target power at the new frequency — delivered power stays as requested.
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power",
                                round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"],
                                                          params=pwr_params()), 2))
                else:
                    ctrl.report("power",
                                round(pmap.power_for_gain(state["gain"], freq=state["freq"], params=pwr_params()), 2))
        elif name == "power":
            # power/gain edits are staged into state["gain"] and only reach the radio
            # when RF is on; the --rf toggle mutes/restores gain AND amplitude. Folded at
            # the current transmit frequency (frequency-dependent chains).
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                params=pwr_params())
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(
                    tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"], params=pwr_params()), 2))
        elif name == "gain":
            # A raw gain overrides the dBm mapping — drop any held target power so a later
            # frequency change doesn't re-map back to it.
            state["power"] = None
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("gain", round(tb.actual_gain(), 2))
            else:
                ctrl.report("gain", round(state["gain"], 2))
        elif name in ("start", "stop"):
            # start_stop mode: a live edge move re-derives the carrier (midpoint) and the
            # sweep width (span), then retunes + regenerates. An invalid span (zero or
            # over-wide) is left unapplied — the GUI already flags it via the derived width.
            state[name] = float(value)
            if state.get("start") is not None and state.get("stop") is not None:
                try:
                    c_hz, bw_hz = resolve_band(
                        "start_stop", None, None, state["start"], state["stop"])
                except ValueError:
                    ctrl.report(name, value)      # keep the last good band
                    return
                tb.set_center_frequency(c_hz)
                state["freq"] = c_hz
                # re-map the held target power at the new midpoint (freq-dependent chain)
                if state.get("power") is not None:
                    state["gain"] = pmap.gain_for_power(state["power"], freq=c_hz,
                                                        params=pwr_params())
                    if state["rf_on"]:
                        tb.set_gain(state["gain"])
                shape["bw_hz"] = bw_hz
                regenerate()
            ctrl.report(name, value)
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
        elif name in ("bw", "rate", "filter", "passband", "transition"):
            if name == "bw":
                shape["bw_hz"] = value * 1e6
            elif name == "rate":
                shape["rate_hz"] = value * 1e3            # --rate is in kHz
            elif name == "filter":
                shape["filter_on"] = str(value).strip().lower() in ("on", "1", "true", "yes")
            elif name == "passband":
                shape["width_hz"] = max(MIN_PASSBAND_MHZ, min(MAX_PASSBAND_MHZ,
                                                              float(value))) * 1e6
            else:  # transition
                shape["trans_hz"] = float(value) * 1e6
            actual = regenerate()
            ctrl.report("rate" if name == "rate" else name,
                        actual / 1e3 if name == "rate" else value)
            # A --bw change re-scales --power when the calibration's reported reading is the
            # full-bandwidth-power law (which keys on --bw): re-map the held target so the
            # delivered reported power stays as requested. No-op when --power isn't held or the
            # calibration uses no bw-keyed bridge (pwr_params is then ignored).
            if name == "bw" and state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"],
                                                    params=pwr_params())
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power", round(pmap.power_for_gain(
                        tb.actual_gain(), freq=state["freq"], params=pwr_params()), 2))
                else:
                    ctrl.report("power", round(pmap.power_for_gain(
                        state["gain"], freq=state["freq"], params=pwr_params()), 2))

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

#!/usr/bin/env python3
"""
Fixed-bandwidth FM-chirp transmitter for GNU Radio + UHD (Ettus B200-mini family).

A sawtooth frequency sweep of a FIXED width (set by SWEEP_BW_MHZ below), precomputed
and replayed from RAM so a Raspberry Pi sustains the sample rate — the same recipe as
gps_l1ca_tx.py / fm_chirp_tx.py, but with the sweep bandwidth baked in so the emitted
spectral density is constant and the signal can be calibrated on its own.

Why a dedicated per-bandwidth script?
─────────────────────────────────────
A chirp's power spectral density is (roughly) total_power / sweep_bandwidth, so the
density — and therefore the calibration that maps a commanded gain/power to a real
dBm — changes with the sweep width. Fixing the width per script gives each one a
stable density and its own CAL_SIGNAL_ID, so a unit can be calibrated for THIS
bandwidth independently of the others. The sweep width is therefore NOT a parameter;
the only width control left is the optional digital passband FILTER.

The instantaneous frequency is  f(t) = sawtooth(t) · (SWEEP_BW/2), i.e. it sweeps
±SWEEP_BW/2 around the carrier, and the modulating sawtooth repeats at `--rate`.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable +
   attenuators) on frequencies you are LICENSED / AUTHORISED to use.

Why this reaches the sample rate on a Pi (see gps_l1ca_tx.py for the full write-up)
──────────────────────────────────────────────────────────────────────────────────
  1. PRECOMPUTE + LOOP — one whole sweep is built once and replayed with a C++
     blocks.vector_source_c(repeat=True); no per-sample NumPy in a work() block.
  2. sc8 over the wire — halves USB payload.
  3. Quiet — UHD fastpath/console logging off.
  4. 1:1 master clock — master_clock_rate pinned to the sample rate; no FPGA resampling.

Seamless looping of a SWEPT buffer
──────────────────────────────────
A frequency sweep only loops without a seam if the accumulated phase closes at the
wrap. Two things guarantee it: the buffer holds a whole number of sweep periods
(integer samples/period), and the swept frequency is forced to EXACT zero mean, so
the phase integrated over the buffer returns to its start (see --self-test).

Live tuning (retune while transmitting, via paramkit.live)
──────────────────────────────────────────────────────────
    freq       → UHD tune_request        (instant)
    power      → dBm → set_gain          (instant — folded at the carrier)
    rf         → on/off mute/unmute      (instant — gain AND amplitude to 0 / back)
    rate       → rebuild buffer + swap    ┐ shape changes: regenerate one sweep in RAM
    filter     → rebuild buffer + swap    │ and set_data() it under the top-block lock
    passband   → rebuild buffer + swap    │ (one brief seam at the swap, then it loops
    transition → rebuild buffer + swap    ┘ clean); RF never stops.

Fixed radio setup
─────────────────
Sample rate 61.38 MHz (the B200's ceiling), over-the-wire sc8, baseband amplitude 0.5
(the amplitude the calibration is measured at). None are parameters. An optional digital
passband filter (--filter/--passband/--transition) brick-walls the sweep and strips the
sawtooth reset splatter; --passband is the TOTAL filter width (it passes ±passband/2), so
set it a hair above the sweep width to keep the whole sweep, or below it to gate a sub-band.

CLI
───
    fm_chirp_1MHz_tx.py --freq 1575.42 --rate 200 --power -30
    fm_chirp_1MHz_tx.py --freq 1575.42 --gain 60 --filter on --passband 1.2   # clean band-limit
    fm_chirp_1MHz_tx.py --self-test        # verify seamless phase closure (+ filter), no hardware
    fm_chirp_1MHz_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# ── The one thing that differs between the fixed-bandwidth chirp scripts ─────────────
# The sweep is ±SWEEP_BW_MHZ/2 around the carrier. It is NOT a parameter (a fixed width
# is what makes this signal's spectral density — and its calibration — stable). Every
# other per-bandwidth value (the calibration id, the filter defaults) derives from it,
# so sibling scripts differ from this one by this single line.
SWEEP_BW_MHZ = 1.0
SWEEP_BW_HZ = SWEEP_BW_MHZ * 1e6

# Stable calibration signal id — unique per bandwidth so each width calibrates on its own.
# When a task sets SDR_CAL_SIGNAL_ID to this value the agent injects this unit's resolved
# calibration (SDR_CALIBRATION_FILE); calkit reads it and --power maps through the unit's
# MEASURED curve at its real operating plane. Absent it, the script runs uncalibrated.
CAL_SIGNAL_ID = f"fm_chirp_{SWEEP_BW_MHZ:g}MHz"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# has a --power scale that MOVES with frequency, so the map is folded at THIS param's value
# — and it is live, so retuning the carrier re-scales --power on the fly.
CAL_FREQ_PARAM = "freq"


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task parameter:
# the calibration is measured at THIS amplitude, so a unit calibrated at a different
# amplitude no longer matches. calkit detects that at load and runs UNCALIBRATED with a
# loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum.
HW_MAX_GAIN_DB = 89.75


# ── Power map: the unit's injected calibration curve if present, else uncalibrated ──

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
SAMP_RATE_HZ = 61.38e6       # the max; master clock pinned 1:1
OTW_FORMAT = "sc8"           # over-the-wire; halves USB load

# Tile the single sweep period up to at least this many samples so the looping source
# wraps infrequently at high rate (whole periods only → still seamless).
MIN_BUFFER_SAMPS = 1 << 18   # 262144 samples ≈ 2 MB as fc32

# ── Filter bounds + defaults (defaults derive from the fixed sweep width) ────────────
MIN_PASSBAND_MHZ = 0.1
MAX_PASSBAND_MHZ = 61.2                                   # clamped to Nyquist
_DEF_PASSBAND_MHZ = round(min(MAX_PASSBAND_MHZ, SWEEP_BW_MHZ * 1.2), 3)   # a hair above the sweep
_DEF_TRANSITION_MHZ = round(min(8.0, max(0.05, SWEEP_BW_MHZ * 0.1)), 3)   # steepness scaled to width

# Named GNSS carriers, in MHz.
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


# ── Swept buffer (one seamless-looping sawtooth period) ──────────────────────────

def build_chirp_buffer(sweep_rate_hz: float):
    """Build a complex64 baseband buffer at SAMP_RATE_HZ holding a whole number of sawtooth
    sweep periods (±SWEEP_BW_HZ/2) that loops with no seam. Unit magnitude (amplitude is
    applied live downstream). Returns (iq, samples_per_period, reps, actual_sweep_rate_hz)."""
    import numpy as np

    n_per = max(2, int(round(SAMP_RATE_HZ / sweep_rate_hz)))
    actual_sweep_rate = SAMP_RATE_HZ / n_per       # rate quantised to the grid

    p = np.arange(n_per, dtype=np.float64) / n_per
    freq = (2.0 * p - 1.0) * (SWEEP_BW_HZ / 2.0)   # sawtooth: -1 → +1 (reset)
    freq = freq - freq.mean()                      # exact zero-mean → phase closes at the wrap
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
    power unchanged. Returns (filtered_iq, n_taps, passband_edge_hz)."""
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

    fs, bw, rate = SAMP_RATE_HZ, SWEEP_BW_HZ, 200e3
    n = max(2, round(fs / rate))
    freq = [(2.0 * (i / n) - 1.0) * (bw / 2.0) for i in range(n)]   # sawtooth
    mean = sum(freq) / n
    freq = [f - mean for f in freq]
    acc, phase = 0.0, []
    for f in freq:
        acc += 2 * math.pi / fs * f
        phase.append(acc)
    iq = [cmath.exp(1j * ph) for ph in phase]
    expected = 2 * math.pi / fs * freq[0]
    measured = cmath.phase(iq[0] / iq[-1])
    seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
    ok = seam_err < 1e-9
    print(f"Sawtooth {bw/1e6:g} MHz: n={n} seam_err={seam_err:.2e} rad [{'OK' if ok else 'FAIL'}]")
    print("SEAMLESS" if ok else "SELF-TEST FAILED")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, _n, _r, _ar = build_chirp_buffer(200e3)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / fs))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    # Passband a hair above the sweep: the in-band sweep power survives, the reset splatter
    # beyond the passband edge is cut, and the peak stays safe.
    pb = _DEF_PASSBAND_MHZ * 1e6
    tr = _DEF_TRANSITION_MHZ * 1e6
    filt, taps, fp = filter_buffer(base, pb, tr)
    inband = 10 * np.log10(band(filt, 0, bw * 0.4) / max(band(base, 0, bw * 0.4), 1e-30))
    oob_lo = pb / 2.0 + 2.0 * tr
    oob_hi = min(oob_lo + bw, 0.49 * fs)
    cut = 10 * np.log10(band(filt, oob_lo, oob_hi) / max(band(base, oob_lo, oob_hi), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(inband) < 0.3 and cut < -25 and peak * AMPLITUDE < 1.0
    print(f"filter (±{pb/2e6:g} MHz on {bw/1e6:g} MHz sweep, {taps} taps): in-band {inband:+.3f} dB, "
          f"out-of-band {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float, amplitude: float):
    """The looped sweep is streamed from RAM by a C++ blocks.vector_source_c (repeat=True),
    NOT a file_source, and NOT a Python source: swapping a file_source live (open()) races GNU
    Radio and THROWS "fread error" (the source dies, radio silent); a Python source has no file
    but its work() runs under the GIL and can't hold 61.38 Msps on a Pi. vector_source_c is C++
    (GIL-free) with no file, so it streams smoothly with no fread risk. A live filter change swaps
    the buffer with set_data() under top-block lock()/unlock() (set_data has no internal lock), so
    it's never freed under a running read — the stream pauses only for the swap, only on a change."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        # vector_source_c wants a contiguous complex64 buffer; the filtered loop may not be.
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class ChirpTx(gr.top_block):
        def __init__(self):
            super().__init__(f"FM chirp {SWEEP_BW_MHZ:g} MHz TX")
            self._freq_hz = center_freq_hz

            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]),
            )
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(self._freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            # repeat=True loops the buffer in C++ (no per-wrap Python), vlen=1, no tags.
            self.src = blocks.vector_source_c(_vec(initial_iq), True, 1, [])
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def set_center_frequency(self, hz: float) -> None:
            self._freq_hz = hz
            self.usrp.set_center_freq(uhd.tune_request(hz), 0)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def swap(self, iq) -> None:
            # set_data() reassigns (and frees) the source's buffer with no internal lock, so it
            # must not run while work() is reading. lock()/unlock() quiesces the flowgraph around
            # the swap — the only moment the stream pauses, and only on a change.
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
        Script(f"FM-chirp transmitter — fixed {SWEEP_BW_MHZ:g} MHz sawtooth sweep, 61.38 MHz / "
               "sc8, looped buffer, optional power-preserving digital passband filter. The sweep "
               "width is fixed (so this signal calibrates on its own); level is set in dBm via the "
               "unit's calibration, or a relative gain uncalibrated. Authorised, shielded setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=1575.42, required=True, live=True,
                help="RF carrier in MHz; the sweep occupies ±"
                     f"{SWEEP_BW_MHZ/2:g} MHz around it. Live.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration (folded at the carrier) and snaps to its achievable grid; "
                     "ignored if --gain is given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .number("-Sweep-rate", "--rate", unit="kHz", min=0.1, max=5000.0,
                default=200.0, required=True, live=True,
                help="How fast the sweep repeats, in kHz. Live (regenerates the loop).")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain). Set the "
                     "passband a hair above the sweep width to brick-wall the sweep and strip reset "
                     "splatter. Live.")
        .number("-Passband", "--passband", unit="MHz",
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=_DEF_PASSBAND_MHZ,
                required=False, live=True,
                help="Total passband width kept, centred on the carrier (MHz) — the filter passes "
                     f"±passband/2. ≥ {SWEEP_BW_MHZ:g} MHz keeps the whole sweep; less gates it into "
                     "a sub-band. Clamped to Nyquist. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=8.0,
                default=_DEF_TRANSITION_MHZ, required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the steepness "
                     "knob. Live (rebuilds the filtered loop).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    center_freq_hz = float(args.freq) * 1e6

    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:                         # calibrated: the authored absolute --power
        # Fold the calibration at the transmit frequency so a frequency-dependent chain maps
        # --power on the right scale (the chirp can retune live — see apply_change).
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    # Current "shape" (the regeneration-requiring params) — mutated by live changes. The
    # sweep width is fixed (SWEEP_BW_HZ), so it is not part of the shape.
    shape = {"rate_hz": args.rate * 1e3,              # --rate is in kHz
             "filter_on": getattr(args, "filter", "off") == "on",
             "width_hz": float(getattr(args, "passband", _DEF_PASSBAND_MHZ)
                               or _DEF_PASSBAND_MHZ) * 1e6,
             "trans_hz": float(getattr(args, "transition", _DEF_TRANSITION_MHZ)
                               or _DEF_TRANSITION_MHZ) * 1e6}

    def make_current():
        """The buffer for the current shape (the base sweep, optionally band-limited).
        Returns (iq, n_per, reps, actual_rate, finfo)."""
        base, n_per, reps, actual = build_chirp_buffer(shape["rate_hz"])
        if not shape["filter_on"]:
            return base, n_per, reps, actual, {"on": False}
        filt, taps, fp = filter_buffer(base, shape["width_hz"], shape["trans_hz"])
        return filt, n_per, reps, actual, {"on": True, "taps": taps, "edge_hz": fp,
                                           "trans_hz": shape["trans_hz"]}

    iq, n_per, reps, actual_rate, finfo = make_current()

    tb = _build_top_block(initial_iq=iq, center_freq_hz=center_freq_hz,
                          gain_db=gain_db, amplitude=amplitude)

    # RF on/off state + the gain RF-on applies. Track the live transmit frequency and (in
    # absolute mode) the held target power, so a live retune can re-map --power at the new
    # frequency on a frequency-dependent chain.
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": center_freq_hz, "power": _target_power}
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
        return (f"on — passband ±{info['edge_hz']/1e6:.3f} MHz, "
                f"{info['trans_hz']/1e6:g} MHz transition, {info['taps']} taps")

    print(f"── FM chirp {SWEEP_BW_MHZ:g} MHz TX ────────────────────────────────")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  sweep          : Sawtooth, {SWEEP_BW_MHZ:g} MHz wide (±{SWEEP_BW_MHZ/2:g} MHz)")
    print(f"  sweep rate     : requested {args.rate:g} kHz, "
          f"got {actual_rate/1e3:.3f} kHz ({n_per} samples/period × {reps} reps)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz):.2f} dBm")
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
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power",
                                round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"]), 2))
                else:
                    ctrl.report("power",
                                round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
        elif name == "power":
            # power/gain edits are staged into state["gain"] and only reach the radio when RF
            # is on; folded at the current transmit frequency (frequency-dependent chains).
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"]), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
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
        elif name in ("rate", "filter", "passband", "transition"):
            if name == "rate":
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

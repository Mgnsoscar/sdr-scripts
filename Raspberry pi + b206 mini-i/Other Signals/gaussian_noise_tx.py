#!/usr/bin/env python3
"""
Band-limited Gaussian-noise transmitter for GNU Radio + UHD (Ettus B200-mini family).

What it makes
─────────────
A continuous, NON-REPEATING complex Gaussian-noise carrier, digitally band-limited by a
steep FIR bandpass applied on the host before the samples reach the radio. Useful as a
calibrated wideband noise source / band-limited jammer into a shielded, conducted setup.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable + attenuators
   into a receiver or spectrum analyser) on frequencies you are LICENSED / AUTHORISED to
   use. Radiating broadband noise can jam anything nearby and is illegal in most places.

Why streaming (the gps_l2p_tx.py method), not prebuild-and-loop
───────────────────────────────────────────────────────────────
The other Pi scripts precompute one seamless period and replay it with
blocks.file_source(repeat=True). Noise has no period: a looped noise buffer would repeat
every buffer — a line spectrum of copies, not noise. So this uses the same real-time
streaming path gps_l2p_tx.py uses for the (unloopable) week-long P-code:

  • a PRODUCER thread generates fresh complex Gaussian samples in blocks, band-limits each
    block, and writes them into a named pipe (FIFO) in /dev/shm;
  • a GNU Radio blocks.file_source(repeat=False) reads that FIFO into uhd.usrp_sink — the
    fleet's proven device path, no custom block, no loop. The FIFO + the radio's own
    buffering ride out scheduler jitter.
Because every block is freshly drawn (one RNG advanced continuously), the stream never
repeats — it is true non-looping noise, not a replayed buffer.

The steep digital bandpass, applied ON the buffers
──────────────────────────────────────────────────
The band-limiting is a windowed-sinc (Blackman-Harris) FIR bandpass — a lowpass prototype
of width `--bandwidth`, frequency-shifted to `--offset` from the carrier — applied by
OVERLAP-SAVE across the block boundaries, so the filtered stream is seam-free (bit-for-bit
identical to filtering the whole never-ending stream at once; `--self-test` proves it).
`--transition` sets the steepness (narrower → more taps → sharper skirts). This is a
DIGITAL filter on the IQ, independent of the master clock; there is no analog-filter knob.

Throughput — this is RUNTIME DSP, so the sample rate is CPU-bound
────────────────────────────────────────────────────────────────
A non-looping stream can't be pre-filtered, so the noise draw + FFT filter run live in the
producer, and the FFT is the limiter (and, being sequential overlap-save, single-threaded —
the white_noise_tx.py sibling has no filter and parallelises across cores for far higher
rates). A fast desktop sustains ~10–15 MS/s; a Raspberry Pi (slower numpy) sustains only low
single-digit MS/s. Past that the FIFO starves and the radio underflows ("U…"), so the
defaults are modest. Measure YOUR unit's real ceiling with `--dry-run` — it runs the ACTUAL
producer through a real pipe (not a bare generate loop), so its ×real-time verdict predicts
hardware — and keep headroom (UHD's USB + GNU Radio threads also want CPU): if it's not
comfortably above 1×, drop `--samp_rate`, widen `--transition` (fewer taps), or narrow
`--bandwidth`.

Level / calibration (same plumbing as the PRN scripts)
──────────────────────────────────────────────────────
The producer emits UNIT-POWER noise (E|z|² = 1, mean-square-normalised by the filter's
energy so the level doesn't drift with bandwidth); the flowgraph's multiply_const applies
the fixed baseband amplitude, and --power/--gain map to the SDR gain via the unit's
injected calibration exactly as in the PRN scripts. AMPLITUDE is 0.25 (not 0.5): Gaussian
noise has a high crest factor, so the RMS is kept ~12 dB below full scale to leave headroom
and keep DAC clipping negligible (P(|z| clips) ≈ 1e-7). --power is average (RMS) power.

CLI
───
    gaussian_noise_tx.py --freq 1575.42e6 --bandwidth 2 --samp_rate 4 --gain 60
    gaussian_noise_tx.py --freq 1227.60e6 --bandwidth 3 --offset 1 --samp_rate 8 --power -30
    gaussian_noise_tx.py --self-test        # validate the filter + streaming (no hardware)
    gaussian_noise_tx.py --dry-run --samp_rate 8 --gain 60   # measure this host's max rate
    gaussian_noise_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# Quiet UHD/GNU Radio BEFORE the libs load (imported lazily inside main()).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")   # no "UUUU" underflow spam
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

try:
    import numpy as np                       # core to generation; kept optional so
except ImportError:                          # --describe-params works without it
    np = None

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the agent
# injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads it and
# --power maps through the unit's MEASURED curve at its real operating plane (e.g. EIRP).
# Absent it, the script runs uncalibrated (relative gain only). See docs/calibration.md.
CAL_SIGNAL_ID = "gaussian_noise"

# Which parameter carries the transmit frequency, for a frequency-dependent calibration
# chain (a cable/antenna whose loss varies with frequency): --power is folded at THIS
# param's value, and it is live, so retuning re-scales --power on the fly.
CAL_FREQ_PARAM = "freq"

# ── RF chain limits (mirrors the PRN scripts) ───────────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script obeys)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# Fixed baseband amplitude the calibration is measured at. 0.25 (not 0.5) leaves crest-
# factor headroom for Gaussian noise so DAC clipping is negligible. NOT a user control:
# calkit runs UNCALIBRATED with a warning if a unit was calibrated at a different value.
AMPLITUDE = 0.25

# Named GNSS carriers (same list as fm_chirp_tx), in Hz.
FREQUENCIES = {
    "GPS L1": 1575.42e6, "GPS L2": 1227.60e6, "GPS L5": 1176.45e6,
    "Galileo E1": 1575.42e6, "Galileo E5a": 1176.45e6, "Galileo E5b": 1207.14e6,
    "Galileo E5": 1191.795e6, "Galileo E6": 1278.75e6,
    "BeiDou B1I": 1561.098e6, "BeiDou B1C": 1575.42e6, "BeiDou B2a": 1176.45e6,
    "BeiDou B2b": 1207.14e6, "BeiDou B2": 1191.795e6, "BeiDou B3": 1268.52e6,
    "GLONASS L1": 1602.0e6, "GLONASS L2": 1246.0e6, "GLONASS L3": 1202.025e6,
    "Iridium": 1621.25e6,
}

# Windowed-sinc taps ≈ WINDOW_TAP_K · Fs / transition. Blackman-Harris → ~5.5 (steep skirts,
# ~92 dB stopband). Capped so a tiny --transition can't ask for an unbounded filter.
WINDOW_TAP_K = 5.5
MAX_TAPS = 8191
# Target producer block duration; the block size is the next power of two ≥ this many
# samples (bigger → fewer FFTs, more latency). The dry-run reports sustained throughput.
TARGET_BLOCK_SECONDS = 0.02

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present, else uncalibrated
    (relative-gain-only). Cached so build_script and main agree."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ═══════════════════════════════════════════════════════════════════════════════════
# Digital bandpass — numpy-only windowed-sinc FIR + stateful overlap-save streaming
# ═══════════════════════════════════════════════════════════════════════════════════

def design_bandpass(fs: float, bw_hz: float, offset_hz: float, trans_hz: float):
    """Complex FIR bandpass taps (complex64): a Blackman-Harris windowed-sinc lowpass of
    half-width ``bw_hz/2``, frequency-shifted to ``offset_hz``. ``trans_hz`` sets the
    transition width → number of taps. Unity passband gain. Returns (h, n_taps)."""
    m = int(np.ceil(WINDOW_TAP_K * fs / max(trans_hz, 1.0)))
    m = min(m | 1, MAX_TAPS | 1)                       # force odd, cap length
    n = np.arange(m)
    c = (m - 1) / 2.0
    fcn = (bw_hz / 2.0) / fs                            # lowpass cutoff (cycles/sample)
    h = 2 * fcn * np.sinc(2 * fcn * (n - c))            # ideal lowpass impulse response
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * n / n1)
           + 0.14128 * np.cos(4 * np.pi * n / n1)
           - 0.01168 * np.cos(6 * np.pi * n / n1))      # Blackman-Harris
    h = h * win
    h = h / np.sum(h)                                   # unity DC (→ passband) gain
    h = h * np.exp(1j * 2 * np.pi * (offset_hz / fs) * (n - c))   # shift to the offset
    return h.astype(np.complex64), m


def _fft_size(fs: float, m: int) -> int:
    """The overlap-save FFT size: a power of two near TARGET_BLOCK_SECONDS of samples and
    comfortably larger than the filter, so the valid output per FFT (N−M+1) is ≈ N —
    keeping the transform efficient (a size ≈ 2× the block wastes half the FFT)."""
    target = max(int(TARGET_BLOCK_SECONDS * fs), 4 * m)
    n = 1
    while n < target:
        n <<= 1
    return n


class _OverlapSave:
    """Stateful overlap-save FIR run at its efficient operating point: one length-N FFT per
    call yields L = N−M+1 valid, seam-free output samples (feed exactly L new samples in)."""

    def __init__(self, h, n_fft: int):
        self.M = len(h)
        self.N = n_fft
        self.L = n_fft - self.M + 1                      # valid samples per transform
        self.H = np.fft.fft(h.astype(np.complex128), n_fft)
        self.tail = np.zeros(self.M - 1, dtype=np.complex128)   # last M-1 input samples

    def process(self, x):
        """Filter one block of exactly L complex samples → L filtered samples."""
        seg = np.empty(self.N, dtype=np.complex128)
        seg[: self.M - 1] = self.tail
        seg[self.M - 1 :] = x                            # (M-1) + L == N, no zero-pad
        y = np.fft.ifft(np.fft.fft(seg) * self.H)
        self.tail = x[-(self.M - 1):]                    # carry history for the next block
        return y[self.M - 1 :]


class NoiseSource:
    """Continuous band-limited unit-power complex-Gaussian source. Each pull draws fresh
    samples (the RNG advances → never repeats) and band-limits them with the overlap-save
    filter, so consecutive blocks join seamlessly."""

    def __init__(self, fs: float, bw_hz: float, offset_hz: float, trans_hz: float,
                 seed=None):
        if np is None:
            raise RuntimeError("numpy is required to generate noise")
        self.h, self.n_taps = design_bandpass(fs, bw_hz, offset_hz, trans_hz)
        self.osf = _OverlapSave(self.h, _fft_size(fs, self.n_taps))
        self.block = self.osf.L                          # new samples generated per pull
        # Mean-square normalisation: white unit-power in → the filter scales power by
        # sum|h|²; divide it back out so E|z|²=1 regardless of bandwidth (a constant, so
        # the noise stays stationary and the level doesn't jump between blocks).
        self.norm = np.float32(1.0 / np.sqrt(np.sum(np.abs(self.h.astype(np.complex128)) ** 2)))
        self.rng = np.random.default_rng(seed)

    def next_block(self):
        """The next `self.block` band-limited unit-power complex64 samples. float32 draws
        (half the RNG cost of float64) — the FFT is the limiter, not the noise draw."""
        n = self.block
        white = np.empty(n, dtype=np.complex64)
        white.real = self.rng.standard_normal(n, dtype=np.float32)
        white.imag = self.rng.standard_normal(n, dtype=np.float32)
        white *= np.float32(0.70710678)                  # unit power: var(I)+var(Q)=1
        return (self.osf.process(white) * self.norm).astype(np.complex64)


# ── parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Band-limited Gaussian-noise transmitter — a continuous, non-repeating "
               "complex-noise carrier band-limited by a steep digital FIR bandpass on the "
               "host. Level is set in dBm via the unit's calibration; uncalibrated it runs "
               "on a relative gain. Transmit only into an authorised, shielded setup.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="RF carrier. Live.")
        .number("-Bandwidth", "--bandwidth", unit="MHz", min=0.01, max=56.0, default=1.0,
                required=True,
                help="Passband WIDTH of the digital bandpass (MHz): the noise fills "
                     "±bandwidth/2 around the passband centre. Fixed per run.")
        .number("-Offset", "--offset", unit="MHz", min=-28.0, max=28.0, default=0.0,
                required=False,
                help="Passband CENTRE offset from the carrier (MHz); 0 = centred on the "
                     "carrier. Use it to place the noise band off-centre. Fixed per run.")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=10.0, default=0.25,
                required=False,
                help="Transition width of the bandpass skirts (MHz) — the steepness knob. "
                     "Narrower = sharper edges = more taps (more CPU). Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=1.0, max=61.44, default=2.0,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). Must be "
                     "> the passband edges. RUNTIME-DSP LIMITED: the producer filters live, "
                     "so a Pi sustains only a few MS/s — check with --dry-run and lower this "
                     "if it underflows. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE (average/RMS) power at the delivered plane (dBm). Bounds "
                     "track the unit's calibration when present. Ignored if --gain is "
                     "given. Live.")
        .number("-LO-offset", "--lo_offset", unit="MHz", min=-30.0, max=30.0, default=0.0,
                live=True, help="LO offset to push LO leakage out of band. Live.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load (helps at high "
                     "MS/s on a Pi); sc16 for more dynamic range.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the baseband amplitude to 0; ON restores "
                     "it. Live.")
        .number("-Duration", "--duration", unit="s", min=0.0, max=604800.0, default=0.0,
                required=False, help="Stop after this many seconds. 0 = run until stopped.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing "
                     "the dBm calibration. When given, overrides --power. Live.")
    )


# ── self-test: validate the filter + streaming (no hardware) ────────────────────────

def _self_test() -> int:
    if np is None:
        print("numpy required for --self-test", file=sys.stderr)
        return 2
    fs, bw, off, trans = 20.46e6, 8e6, 0.0, 1.0e6
    src = NoiseSource(fs, bw, off, trans, seed=0)
    print(f"filter: {src.n_taps} taps, block {src.block} samples, "
          f"FFT {src.osf.N}, sum|h|²={np.sum(np.abs(src.h.astype(np.complex128))**2):.4f}")

    # 1) overlap-save is seam-free: streamed blocks == one-shot convolution of the same input.
    rng = np.random.default_rng(1)
    L = src.osf.L
    total = 5 * L
    x = (rng.standard_normal(total) + 1j * rng.standard_normal(total)) / np.sqrt(2.0)
    osf = _OverlapSave(src.h, src.osf.N)
    ys = [osf.process(x[i:i + L]) for i in range(0, total, L)]
    y_stream = np.concatenate(ys)
    y_ref = np.convolve(x, src.h.astype(np.complex128))[:total]
    seam = float(np.max(np.abs(y_stream - y_ref)))
    print(f"overlap-save continuity: max|stream − oneshot| = {seam:.2e}  "
          f"[{'OK' if seam < 1e-6 else 'FAIL'}]")

    # 2) steep band-limiting: in-band power ≫ out-of-band power.
    seg = y_ref[src.n_taps:src.n_taps + 65536]
    f = np.fft.fftshift(np.fft.fftfreq(seg.size, 1.0 / fs))
    psd = np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    edge = bw / 2.0
    inb = psd[np.abs(f) <= edge - trans].mean()
    outb = psd[np.abs(f) >= edge + trans].mean()
    supp = 10 * np.log10(inb / outb)
    print(f"stopband suppression: {supp:.1f} dB below passband  "
          f"[{'OK' if supp > 40 else 'FAIL'}]")

    # 3) unit output power (level independent of bandwidth).
    p = float(np.mean(np.abs(np.concatenate([src.next_block() for _ in range(4)])) ** 2))
    print(f"output power: E|z|² = {p:.3f} (target 1.0)  [{'OK' if 0.8 < p < 1.25 else 'FAIL'}]")

    ok = seam < 1e-6 and supp > 40 and 0.8 < p < 1.25
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── entry point ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    dry_run = "--dry-run" in sys.argv[1:]
    if dry_run:                                   # not a paramkit param — strip before parse
        sys.argv = [a for a in sys.argv if a != "--dry-run"]

    script = build_script()
    args = script.parse()

    if np is None:
        print("numpy is required to run", file=sys.stderr)
        return 2

    samp_rate_hz = float(args.samp_rate) * 1e6
    bw_hz = float(args.bandwidth) * 1e6
    offset_hz = float(getattr(args, "offset", 0.0) or 0.0) * 1e6
    trans_hz = float(getattr(args, "transition", 1.0) or 1.0) * 1e6

    # Passband must fit inside the complex baseband (±Fs/2).
    hi_edge = abs(offset_hz) + bw_hz / 2.0
    if hi_edge >= samp_rate_hz / 2.0:
        print(f"error: passband edge {hi_edge/1e6:.3f} MHz exceeds ±Fs/2 "
              f"({samp_rate_hz/2e6:.3f} MHz) — widen --samp_rate or narrow the band.",
              file=sys.stderr)
        return 2

    pmap = power_map()
    amplitude = pmap.amplitude

    # gain precedence: explicit --gain > calibrated --power > persisted fallback > refuse.
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=float(args.freq))
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    src = NoiseSource(samp_rate_hz, bw_hz, offset_hz, trans_hz)

    print("── Gaussian-noise TX ────────────────────────────────────────")
    print(f"  carrier        : {float(args.freq)/1e6:.3f} MHz  (LO offset {args.lo_offset:g} MHz)")
    print(f"  sample rate    : {samp_rate_hz/1e6:g} MHz (1:1 master clock)")
    lo, hi = (offset_hz - bw_hz / 2.0) / 1e6, (offset_hz + bw_hz / 2.0) / 1e6
    print(f"  noise band     : {args.bandwidth:g} MHz wide, centred {args.offset:g} MHz off "
          f"carrier  (baseband {lo:+.3f}…{hi:+.3f} MHz)")
    print(f"  filter         : {src.n_taps} taps, {args.transition:g} MHz transition "
          f"(block {src.block} samples)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": float(args.freq), "power": (args.power if (pmap.has_absolute and
                                                                gain_cal is None) else None),
             "stop": False}

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())
    signal.signal(signal.SIGINT, lambda *_: stop_evt.set())

    ctrl = script.live_control(args)

    def apply_live(tb=None):
        """Drain live edits into `state` (freq/power/gain/rf). With a top_block, push
        freq/gain/rf to the radio too. Returns True if the radio level changed."""
        changed = False
        for ch in ctrl.drain():
            if ch.name == "freq":
                state["freq"] = float(ch.value)
                if tb is not None:
                    tb.set_center_frequency(state["freq"])
                    ctrl.report("freq", tb.actual_freq())
                if state.get("power") is not None:      # re-map --power at the new frequency
                    state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
                    changed = True
            elif ch.name == "lo_offset":
                if tb is not None:
                    tb.set_lo_offset(float(ch.value) * 1e6)
                ctrl.report("lo_offset", float(ch.value))
            elif ch.name == "power" and pmap.has_absolute:
                state["power"] = float(ch.value)
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
                changed = True
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
            elif ch.name == "gain":
                state["power"] = None                   # raw gain drops any held target power
                state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(ch.value)))
                changed = True
                ctrl.report("gain", round(state["gain"], 2))
            elif ch.name == "rf":
                state["rf_on"] = str(ch.value).strip().lower() in ("on", "1", "true", "yes")
                changed = True
                ctrl.report("rf", "on" if state["rf_on"] else "off")
        return changed

    duration = float(getattr(args, "duration", 0.0) or 0.0)
    deadline = (time.monotonic() + duration) if duration > 0 else None

    if dry_run:
        return _dry_run(src, samp_rate_hz, stop_evt)

    # ── real hardware: usrp_sink fed by a FIFO (the gps_l2p_tx.py path) ──
    import fcntl
    import tempfile
    from gnuradio import gr, blocks, uhd

    tmpdir = tempfile.mkdtemp(prefix="noise_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    fifo_path = os.path.join(tmpdir, "iq.fifo")
    os.mkfifo(fifo_path)

    def producer_fifo():
        try:
            fd = os.open(fifo_path, os.O_WRONLY)          # blocks until file_source opens read end
        except OSError:
            return
        try:
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 1 << 20)  # ~1 MB pipe buffer (best effort)
        except (OSError, AttributeError):
            pass
        try:
            while not stop_evt.is_set() and not state["stop"]:
                os.write(fd, memoryview(src.next_block()))  # zero-copy band-limited noise
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    prod = threading.Thread(target=producer_fifo, daemon=True)
    prod.start()

    class _NoiseTx(gr.top_block):
        def __init__(self):
            super().__init__("Gaussian noise TX")
            self._freq_hz = float(args.freq)
            self._lo_hz = float(args.lo_offset) * 1e6
            dev = (f"master_clock_rate={samp_rate_hz:.0f},"
                   "num_send_frames=512,send_frame_size=16000")
            extra = os.environ.get("SDR_UHD_ARGS", "")
            if extra:
                dev += "," + extra
            self.usrp = uhd.usrp_sink(
                dev, uhd.stream_args(cpu_format="fc32", otw_format=args.otw, channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            self._retune()
            self.usrp.set_gain(state["gain"], 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, fifo_path, repeat=False)
            self.amp = blocks.multiply_const_cc(amplitude if state["rf_on"] else 0.0)
            self.connect(self.src, self.amp, self.usrp)

        def _retune(self):
            self.usrp.set_center_freq(uhd.tune_request(self._freq_hz, self._lo_hz), 0)

        def set_center_frequency(self, hz):
            self._freq_hz = hz
            self._retune()

        def set_lo_offset(self, hz):
            self._lo_hz = hz
            self._retune()

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def actual_freq(self):
            return self.usrp.get_center_freq(0)

    tb = _NoiseTx()                                       # opens the read end → producer unblocks
    tb.start()

    last = (state["gain"], state["rf_on"])
    try:
        while not stop_evt.is_set():
            if apply_live(tb) and (state["gain"], state["rf_on"]) != last:
                tb.set_gain(state["gain"])
                tb.set_amplitude(amplitude if state["rf_on"] else 0.0)
                last = (state["gain"], state["rf_on"])
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    finally:
        state["stop"] = True
        stop_evt.set()
        try:
            tb.stop(); tb.wait()
        except Exception:      # noqa: BLE001
            pass
        prod.join(timeout=1.0)
        ctrl.close()
        try:
            os.remove(fifo_path); os.rmdir(tmpdir)
        except OSError:
            pass
    print("Gaussian-noise TX stopped.")
    return 0


def _dry_run(src, samp_rate_hz, stop_evt) -> int:
    """No radio, but HONEST: run the REAL producer (generate + overlap-save filter + the
    zero-copy write) into a REAL pipe, drained by a reader modelling the radio, and measure
    the rate actually delivered. The filter is sequential (overlap-save state), so this is
    single-threaded — its ceiling is the runtime FFT. Its verdict predicts hardware, unlike a
    bare drain loop."""
    import queue
    import fcntl
    print("[dry-run] no UHD; real producer (generate + filter) → real pipe, measuring the "
          "sustained delivered rate …")
    r, w = os.pipe()
    try:
        fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 1 << 20)
    except (OSError, AttributeError):
        pass
    done = threading.Event()

    def writer():
        try:
            while not done.is_set():
                os.write(w, memoryview(src.next_block()))
        except (OSError, BrokenPipeError):
            pass
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    t_warm = time.perf_counter() + 0.6
    while time.perf_counter() < t_warm and not stop_evt.is_set():
        os.read(r, 1 << 20)
    t0 = time.perf_counter(); nbytes = 0
    t_end = t0 + 2.5
    while time.perf_counter() < t_end and not stop_evt.is_set():
        b = os.read(r, 1 << 20)
        if not b:
            break
        nbytes += len(b)
    dt = time.perf_counter() - t0

    done.set()
    try:
        os.close(r)
    except OSError:
        pass
    wt.join(timeout=1.0)
    try:
        os.close(w)
    except OSError:
        pass

    msps = (nbytes / 8) / dt / 1e6 if dt > 0 else 0.0     # complex64 = 8 bytes/sample
    ratio = msps / (samp_rate_hz / 1e6) if samp_rate_hz > 0 else 0.0
    if ratio >= 2.0:
        verdict = "comfortable headroom — should stream cleanly"
    elif ratio >= 1.3:
        verdict = ("MARGINAL — likely to underflow on hardware; lower --samp_rate, widen "
                   "--transition, or narrow --bandwidth")
    else:
        verdict = ("TOO SLOW — will underflow; lower --samp_rate, widen --transition, or "
                   "narrow --bandwidth")
    print(f"[dry-run] sustained ~{msps:.1f} Msps through the pipe = {ratio:.1f}x the "
          f"{samp_rate_hz/1e6:g} MHz sample rate")
    print(f"[dry-run] {verdict}")
    print("[dry-run] note: UHD's USB + GNU Radio threads also need CPU on the real unit, so "
          "keep headroom — run comfortably below this ceiling, not right at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

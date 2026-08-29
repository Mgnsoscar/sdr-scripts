#!/usr/bin/env python3
"""
Full-band white Gaussian-noise transmitter for GNU Radio + UHD (Ettus B200-mini family).

What it makes
─────────────
A continuous, NON-REPEATING complex white-Gaussian-noise carrier that fills the whole
±Fs/2 baseband — no digital filtering. The only band-limiting is the AD9361's own analog
reconstruction filter (which UHD sets from the sample rate); there is no digital OR analog
filter knob. Useful as a calibrated WIDEBAND noise source / jammer into a shielded setup.

This is the un-filtered sibling of gaussian_noise_tx.py (which band-limits each block with a
steep FIR). Dropping the filter removes the per-block FFT — the only work left is the noise
draw — so it sustains a much higher sample rate on the same hardware. Use this when you want
the widest flat noise band; use gaussian_noise_tx.py when you need a shaped/narrow band.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable + attenuators
   into a receiver or spectrum analyser) on frequencies you are LICENSED / AUTHORISED to
   use. Radiating broadband noise can jam anything nearby and is illegal in most places.

Why streaming (the gps_l2p_tx.py method), not prebuild-and-loop
───────────────────────────────────────────────────────────────
Noise has no period: a looped buffer would repeat every buffer — a line spectrum of copies,
not noise. So this uses the same real-time streaming path gps_l2p_tx.py uses for the
(unloopable) week-long P-code:

  • a PRODUCER thread draws fresh complex-Gaussian samples in blocks (the RNG advances → the
    stream never repeats — true white noise, not a replayed buffer) and writes them into a
    named pipe (FIFO) in /dev/shm;
  • a GNU Radio blocks.file_source(repeat=False) reads that FIFO into uhd.usrp_sink — the
    fleet's proven device path, no custom block, no loop. The FIFO + the radio's own
    buffering ride out scheduler jitter.

Throughput — runtime-generated, parallelised across cores
─────────────────────────────────────────────────────────
The only per-sample cost is the Gaussian draw (no FFT), and numpy's draw releases the GIL,
so the producer runs several worker threads that draw independent blocks on separate cores
— for noise, block order doesn't matter, so this is embarrassingly parallel — and a single
writer feeds them to the radio with a zero-copy write. Workers default to (cores − 1); set
SDR_NOISE_WORKERS to override (fewer can stream more smoothly if the radio's threads starve).

It is still CPU-bound: past what the host sustains, the FIFO starves and the radio underflows
("U…"). Measure YOUR unit's real ceiling with `--dry-run` — it runs the ACTUAL producer
through a real pipe drained by a reader (not a bare generate loop), so its number predicts
hardware — and keep headroom (UHD's USB + GNU Radio threads also want CPU): run comfortably
below the ×real-time it reports, not right at it.

Level / calibration (same plumbing as the PRN scripts)
──────────────────────────────────────────────────────
The producer emits UNIT-POWER noise (E|z|² = 1); the flowgraph's multiply_const applies the
fixed baseband amplitude, and --power/--gain map to the SDR gain via the unit's injected
calibration exactly as in the PRN scripts. AMPLITUDE is 0.25 (not 0.5): Gaussian noise has a
high crest factor, so the RMS is kept ~12 dB below full scale to leave headroom and keep DAC
clipping negligible. --power is average (RMS) power.

CLI
───
    white_noise_tx.py --freq 1575.42e6 --samp_rate 8 --gain 60
    white_noise_tx.py --freq 1227.60e6 --samp_rate 12 --power -30
    white_noise_tx.py --self-test        # validate the noise + streaming (no hardware)
    white_noise_tx.py --dry-run --samp_rate 20 --gain 60   # measure this host's max rate
    white_noise_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import queue
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
CAL_SIGNAL_ID = "white_noise"

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

# Target producer block duration; the block size is the next power of two ≥ this many
# samples (bigger → fewer/larger writes, more latency). The dry-run reports throughput.
TARGET_BLOCK_SECONDS = 0.02

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present, else uncalibrated
    (relative-gain-only). Cached so build_script and main agree."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── white-noise source (no filter — just a continuous Gaussian draw) ────────────────

def _block_len(fs: float) -> int:
    """Producer block size: next power of two ≥ TARGET_BLOCK_SECONDS of samples (floor 16k)."""
    target = max(int(TARGET_BLOCK_SECONDS * fs), 1 << 14)
    n = 1
    while n < target:
        n <<= 1
    return n


class NoiseSource:
    """Unit-power complex white-Gaussian generator. `draw(rng)` produces one independent
    block; consecutive/parallel blocks are independent white noise, so they join with no
    seam (there's no filter state to carry — which is also why generation parallelises)."""

    def __init__(self, fs: float, seed=None):
        if np is None:
            raise RuntimeError("numpy is required to generate noise")
        self.block = _block_len(fs)
        self._rng = np.random.default_rng(seed)          # for next_block() / the self-test

    def draw(self, rng):
        """One block of `self.block` unit-power complex64 samples, drawn with `rng` (float32
        draws — the draw is the only cost, so it sustains a high rate)."""
        n = self.block
        z = np.empty(n, dtype=np.complex64)
        z.real = rng.standard_normal(n, dtype=np.float32)
        z.imag = rng.standard_normal(n, dtype=np.float32)
        z *= np.float32(0.70710678)                      # unit power: var(I)+var(Q)=1
        return z

    def next_block(self):
        return self.draw(self._rng)


def _n_workers() -> int:
    """Number of generator worker threads. numpy's Gaussian draw releases the GIL, so N
    threads draw ~N× faster on N cores — the key to sustaining a wide band. Defaults to
    (cores − 1) to leave a core for UHD's USB + GNU Radio threads; override with the
    SDR_NOISE_WORKERS env var."""
    env = os.environ.get("SDR_NOISE_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - 1)


class _ParallelProducer:
    """N worker threads draw independent noise blocks into a bounded queue; a single consumer
    pulls finished blocks (order is irrelevant for noise) and writes them to the radio — so
    the CPU-bound generation runs across cores while the write to the FIFO stays serial."""

    def __init__(self, src: "NoiseSource", n_workers: int, prebuffer: int = 8):
        self.src = src
        self.q: "queue.Queue" = queue.Queue(maxsize=max(2, prebuffer))
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._work, daemon=True)
                         for _ in range(max(1, n_workers))]

    def _work(self):
        rng = np.random.default_rng()                    # OS entropy → independent per worker
        while not self._stop.is_set():
            z = self.src.draw(rng)
            while not self._stop.is_set():
                try:
                    self.q.put(z, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def start(self):
        for t in self._threads:
            t.start()

    def get(self, timeout: float = 0.5):
        return self.q.get(timeout=timeout)

    def stop(self):
        self._stop.set()
        try:
            while True:
                self.q.get_nowait()                      # unblock any worker stuck on put()
        except queue.Empty:
            pass
        for t in self._threads:
            t.join(timeout=1.0)


# ── parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Full-band white Gaussian-noise transmitter — a continuous, non-repeating "
               "complex-noise carrier filling ±Fs/2 (no digital or analog filter knob). "
               "Level is set in dBm via the unit's calibration; uncalibrated it runs on a "
               "relative gain. Transmit only into an authorised, shielded setup.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="RF carrier. Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=1.0, max=61.44, default=8.0,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). The noise "
                     "fills the whole ±Fs/2, so this sets the noise BANDWIDTH. Runtime-"
                     "generated, so a Pi caps at a host-dependent rate — check with --dry-run "
                     "and lower it if it underflows. Fixed per run.")
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


# ── self-test: validate the noise + streaming (no hardware) ─────────────────────────

def _self_test() -> int:
    if np is None:
        print("numpy required for --self-test", file=sys.stderr)
        return 2
    src = NoiseSource(20e6, seed=0)
    print(f"block {src.block} samples")
    z = np.concatenate([src.next_block() for _ in range(3)])

    power = float(np.mean(np.abs(z) ** 2))
    vi, vq = float(np.var(z.real)), float(np.var(z.imag))
    xcorr = float(abs(np.mean(z.real * z.imag)))
    print(f"unit power: E|z|² = {power:.3f} (target 1.0)  "
          f"[{'OK' if 0.9 < power < 1.1 else 'FAIL'}]")
    print(f"I/Q balance: var(I)={vi:.3f} var(Q)={vq:.3f}, |corr|={xcorr:.3f}  "
          f"[{'OK' if 0.45 < vi < 0.55 and 0.45 < vq < 0.55 and xcorr < 0.02 else 'FAIL'}]")

    # whiteness: power is flat across the band (split into 8 sub-bands, compare).
    seg = z[: 1 << 16]
    psd = np.abs(np.fft.fft(seg)) ** 2
    bands = psd.reshape(8, -1).mean(axis=1)
    flat_db = 10 * np.log10(bands.max() / bands.min())
    print(f"whiteness: sub-band spread = {flat_db:.2f} dB (flat if small)  "
          f"[{'OK' if flat_db < 1.5 else 'FAIL'}]")

    # non-repeating: two fresh blocks differ.
    a, b = src.next_block(), src.next_block()
    repeats = bool(np.array_equal(a, b))
    print(f"non-repeating: consecutive blocks differ  [{'OK' if not repeats else 'FAIL'}]")

    ok = (0.9 < power < 1.1 and 0.45 < vi < 0.55 and 0.45 < vq < 0.55
          and xcorr < 0.02 and flat_db < 1.5 and not repeats)
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

    src = NoiseSource(samp_rate_hz)
    n_workers = _n_workers()

    print("── White-noise TX ───────────────────────────────────────────")
    print(f"  carrier        : {float(args.freq)/1e6:.3f} MHz  (LO offset {args.lo_offset:g} MHz)")
    print(f"  sample rate    : {samp_rate_hz/1e6:g} MHz (1:1 master clock)")
    print(f"  noise band     : full ±{samp_rate_hz/2e6:g} MHz (white; shaped only by the "
          f"AD9361 analog filter)")
    print(f"  generator      : {n_workers} worker thread(s)  (SDR_NOISE_WORKERS to override), "
          f"block {src.block} samples")
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
        return _dry_run(src, samp_rate_hz, n_workers, stop_evt)

    # ── real hardware: usrp_sink fed by a FIFO (the gps_l2p_tx.py path) ──
    import fcntl
    import tempfile
    from gnuradio import gr, blocks, uhd

    tmpdir = tempfile.mkdtemp(prefix="wnoise_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    fifo_path = os.path.join(tmpdir, "iq.fifo")
    os.mkfifo(fifo_path)

    producer = _ParallelProducer(src, n_workers)

    def writer_fifo():
        try:
            fd = os.open(fifo_path, os.O_WRONLY)          # blocks until file_source opens read end
        except OSError:
            return
        try:
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 1 << 20)  # ~1 MB pipe buffer (best effort)
        except (OSError, AttributeError):
            pass
        producer.start()                                  # workers draw noise across cores
        try:
            while not stop_evt.is_set() and not state["stop"]:
                try:
                    z = producer.get(timeout=0.5)
                except queue.Empty:
                    continue
                os.write(fd, memoryview(z))               # zero-copy raw bytes → FIFO
        except (BrokenPipeError, OSError):
            pass
        finally:
            producer.stop()
            try:
                os.close(fd)
            except OSError:
                pass

    prod = threading.Thread(target=writer_fifo, daemon=True)
    prod.start()

    class _NoiseTx(gr.top_block):
        def __init__(self):
            super().__init__("White noise TX")
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
    print("White-noise TX stopped.")
    return 0


def _dry_run(src, samp_rate_hz, n_workers, stop_evt) -> int:
    """No radio, but HONEST: run the REAL parallel producer into a REAL pipe, drained by a
    reader (modelling the radio consuming), and measure the rate actually delivered through
    the pipe — generation + the zero-copy write + thread contention all included. That's why
    its verdict predicts hardware, where a bare 'how fast can I generate' loop does not."""
    import fcntl
    print(f"[dry-run] no UHD; {n_workers} generator worker(s) → real pipe, measuring the "
          f"sustained delivered rate …")
    r, w = os.pipe()
    try:
        fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 1 << 20)
    except (OSError, AttributeError):
        pass
    producer = _ParallelProducer(src, n_workers)
    done = threading.Event()

    def writer():
        producer.start()
        try:
            while not done.is_set():
                try:
                    z = producer.get(timeout=0.3)
                except queue.Empty:
                    continue
                os.write(w, memoryview(z))
        except (OSError, BrokenPipeError):
            pass

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()

    # warm up (workers spin up, pipe fills), then measure the sustained drained rate.
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
        os.close(r)                                       # unblocks a blocked writer (EPIPE)
    except OSError:
        pass
    producer.stop()
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
        verdict = "MARGINAL — likely to underflow on hardware; lower --samp_rate or add workers"
    else:
        verdict = "TOO SLOW — will underflow; lower --samp_rate (or raise SDR_NOISE_WORKERS)"
    print(f"[dry-run] sustained ~{msps:.1f} Msps through the pipe = {ratio:.1f}x the "
          f"{samp_rate_hz/1e6:g} MHz sample rate")
    print(f"[dry-run] {verdict}")
    print("[dry-run] note: UHD's USB + GNU Radio threads also need CPU on the real unit, so "
          "keep headroom — run comfortably below this ceiling, not right at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

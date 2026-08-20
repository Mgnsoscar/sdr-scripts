#!/usr/bin/env python3
"""
x410_engine — a persistent, multi-channel IQ-replay engine for the Ettus USRP
X410, controlled over a local socket.

Why this exists
───────────────
The X410 has four TX channels, but UHD lets only ONE process own the device at a
time — so you can't launch four independent transmitter scripts, one per channel.
This engine is that single owner: it opens the X410 once (all channels) and then
takes per-channel commands over a Unix socket. "Tasks" (start signal X on channel
2 for 5 minutes) become commands, not device-claiming processes, so they can run
and overlap across channels while only the engine ever touches UHD.

The agent launches the engine once; short-lived "channel-task" clients (see
engine_client.py) then drive individual channels. Sequences/timelines are just
scheduled channel commands — the timeline stays data, not hard-coded here.

Per-channel sample rates (the whole point of the redesign)
──────────────────────────────────────────────────────────
A GNSS scene mixes a wide signal (Galileo E5 AltBOC ≈ 51 MHz occupied → ~61 MS/s)
with narrow ones (GPS L1 C/A ≈ 2 MHz → happy at ~8 MS/s) at the same time. A
single device-wide rate would force every channel up to the widest one, streaming
~4×61 MS/s of mostly-oversampled narrow signal — exactly the host/DMA bandwidth
the X410's modest ARM can't spare.

So the engine drives the radio through UHD's **MultiUSRP** API (the same API the
validated x410_cw_tx.py uses), with **one TX streamer per channel** and an
independent `set_tx_rate()` per channel. The narrow channels stream at their own
low rate; only the wide channel pays for its width. One process still owns the
device — the four streamers are just four views of it.

Each channel runs a **replay thread**: it loops one precomputed complex64 buffer
(a whole number of code periods, so it wraps seamlessly, just like the Pi
file-replay scripts) straight into `tx_streamer.send()`. No per-sample DSP in the
hot path — the buffer is built once, on the client side, at the negotiated rate.

Master clock (fixed, stock)
───────────────────────────
The engine sets ONE master clock rate at launch and never changes it. Only stock
X410 clocks are assumed (default 245.76 MHz), so the achievable per-channel rates
are the divisors master/N — a known discrete set (61.44, 40.96, 30.72, 20.48,
10.24, 8.192, 5.12, … MHz at 245.76). Because GNSS code periods are
whole-millisecond-based and these rates are integer samples-per-ms, the replay
buffers still loop with no seam.

Two-phase rate negotiation (build the buffer at the rate you'll actually stream)
────────────────────────────────────────────────────────────────────────────────
A channel-task must build its IQ at the exact rate the channel will stream. It
doesn't guess: it asks the engine.

  1. → {"cmd":"configure","channel":0,"owner":"task-abc","target_rate_hz":8.192e6}
     ← {"ok":true,"actual_rate_hz":8192000.0,"master_clock_hz":245760000.0}
  2. the task builds its buffer at actual_rate_hz into /dev/shm
  3. → {"cmd":"load","channel":0,"owner":"task-abc","spec":{...ChannelSpec...}}

Channel chain modes (kept deliberately light on the ARM)
────────────────────────────────────────────────────────
Every mode reduces to a "playlist" — a few device-rate period-blocks plus a
selector sequence (one index per period) that's looped. The expansion happens
once, at load, in NumPy; the hot loop only DMAs finished blocks, never per-sample
DSP. Streaming blocks[selectors[k]] in order is byte-identical to one fully-baked
buffer, so nothing about fidelity is traded for the RAM saving.

  • "expanded"  device-rate IQ file → one block, one selector      (lightest; pure DMA)
  • "pcode"     chip-rate IQ file, each chip repeated `interp`×     (long m-seqs, e.g. GLONASS P)
                (np.repeat) → one device-rate block
  • "tiered"    primary-period IQ × a ±1 overlay → two blocks       (single-component + overlay)
                (+primary, −primary), selectors from the overlay
  • "composite" the channel-task supplies the distinct period-blocks and the       (L1C / L5 /
                selector sequence directly — for multi-component signals            E1 / B1C)
                (pilot+data sums, per-component overlays) that aren't one
                block × one ±1 overlay. Reproduces the full-length signal
                (e.g. L1C's 18 s overlay) from a handful of blocks.
  • "tone"      a GENERATED continuous-phase CW at a baseband offset — no buffer    (drifting CW)
                at all. `set` drifts tone_hz over time (phase-continuous via an
                accumulator), so a 20-minute frequency ramp costs nothing.

The on-air handshake (fits the agent's pre-roll)
────────────────────────────────────────────────
A channel-task is started ~10 s before on-air with amplitude 0, streaming zeros
(the streamer stays fed, so there's no start glitch). At the on-air instant a
`set{amplitude, gain}` flips it live. Amplitude is a digital scale applied once
per change (O(N) rescale of the base buffer), so the hot loop stays a pure send;
gain and freq are instant UHD calls.

Control protocol (JSON object per line, over a Unix stream socket)
──────────────────────────────────────────────────────────────────
  → {"cmd":"hello"}                        → master clock, channel count, per-ch state
  → {"cmd":"configure","channel":0,"owner":"task-abc","target_rate_hz":8.192e6}
  → {"cmd":"acquire","channel":0,"owner":"task-abc"}
  → {"cmd":"load","channel":0,"owner":"task-abc","spec":{...ChannelSpec...}}
  → {"cmd":"set","channel":0,"owner":"task-abc","amplitude":0.9,"gain_db":40}
  → {"cmd":"set","channel":0,"owner":"task-abc","freq_hz":1.57542e9}
  → {"cmd":"release","channel":0,"owner":"task-abc"}
  → {"cmd":"status"}                        → per-channel owner/signal/amp/freq/rate
  → {"cmd":"benchmark","seconds":5}         → underflow count at the current scene
  ← {"ok":true, ...}   or   {"ok":false,"error":"..."}

CLI
───
    x410_engine.py --master_clock 245.76 --socket /tmp/x410.sock   # run the engine
    x410_engine.py --benchmark 5 --bench_rates 61.44,8.192,8.192,8.192  # underflow probe
    x410_engine.py --self-test        # protocol + state machine, no UHD/hardware
    x410_engine.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Keep UHD quiet by default (benchmark re-enables fastpath to count underflows).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script


NUM_CHANNELS = 4
MODES = ("expanded", "tiered", "pcode", "composite", "tone")

# Host (CPU) sample formats. "fc32" keeps the proven complex64 path (UHD converts
# to the wire format on send). The integer formats build samples ALREADY in the
# wire layout, so send() is a memcpy with no per-sample conversion — and are
# REQUIRED for an 8-bit wire, since this UHD build registers no fc32→sc8 converter.
CPU_FORMATS = ("fc32", "sc16", "sc8")
_INT_FULL_SCALE = {"sc16": 32767.0, "sc8": 127.0}     # digital full-scale per format

# Streaming is chunked by TIME, not by one transport packet: sending ~10 ms per
# send() (bounded) instead of one ~2000-sample packet cuts the Python/UHD call rate
# by 10–50× at high sample rates, which is what keeps the ARM ahead of the DAC.
DEFAULT_SEND_MS = 10.0
MAX_SEND_SAMPS = 1 << 18                               # cap a single send (latency)


def resolve_cpu_format(cpu: str, otw: str) -> str:
    """The host CPU format to use. 'auto' keeps fc32 for a 16-bit wire (unchanged,
    proven) and matches the wire for 8-bit (fc32→sc8 has no converter here)."""
    if cpu in (None, "", "auto"):
        return "fc32" if otw == "sc16" else otw
    return cpu

# Stock X410 master clock rates (Hz). The engine only ever sets one of these; the
# achievable per-channel rates are the integer divisors of the chosen clock.
STOCK_MASTER_CLOCKS_HZ = (245.76e6, 250.0e6, 500.0e6)
DEFAULT_MASTER_CLOCK_HZ = 245.76e6

# How closely UHD must honour a requested per-channel rate before we treat it as a
# mismatch (the buffer would then be built for the wrong rate and not loop cleanly).
RATE_TOLERANCE_HZ = 1.0


def achievable_rate(master_clock_hz: float, target_rate_hz: float) -> float:
    """The nearest rate the device can actually produce for `target_rate_hz`, given
    a fixed master clock: master / N for the integer N that lands closest. This is
    the pure-arithmetic prediction; the real device value is read back after
    set_tx_rate and is what a channel-task must build its buffer for."""
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    n = max(1, round(master_clock_hz / target_rate_hz))
    return master_clock_hz / n


# ── Channel spec (what a channel-task hands the engine to load) ────────────────

@dataclass
class ChannelSpec:
    """A signal to play on one channel. `mode` selects how the device-rate playback
    is built (see module docstring). Any IQ files must already exist on disk, built
    at the channel's negotiated rate (see the `configure` command).

    Internally every mode becomes a "playlist" = (a few period-blocks, a per-period
    selector sequence that's looped). expanded/pcode are one block; tiered and
    composite pick among blocks per period."""
    mode: str                       # "expanded" | "tiered" | "pcode" | "composite"
    freq_hz: float                  # channel carrier
    gain_db: float = 50.0
    amplitude: float = 0.0          # digital scale 0..1; start muted, raise at on-air
    label: str = ""                 # human tag for status, e.g. "gps_l1_ca prn1"
    # expanded / pcode
    iq_file: str = ""               # device-rate IQ (expanded) or chip-rate IQ (pcode)
    interp: int = 1                 # pcode: samples/chip (np.repeat factor); else 1
    # tiered
    primary_file: str = ""          # device-rate primary-period IQ
    secondary: List[int] = field(default_factory=list)   # ±1 overlay, one per primary period
    period_samples: int = 0         # samples per primary period (overlay repeat factor)
    # composite — multi-component signals (pilot+data sums, per-component overlays).
    # The channel-task precomputes the few DISTINCT period-blocks (each one whole
    # primary period, equal length) and a selector sequence that names, per period,
    # which block to play. Streaming blocks[selectors[k]] in order and looping the
    # sequence is byte-identical to a single fully-baked buffer — without holding
    # the whole (e.g. 18 s) thing in RAM.
    block_files: List[str] = field(default_factory=list)   # device-rate period blocks
    selectors: List[int] = field(default_factory=list)     # index into block_files, per period
    # tone — a GENERATED continuous-phase CW at a baseband offset (no buffer). The
    # channel-task drifts tone_hz over time (e.g. a slow x→y frequency ramp); the
    # engine's LO stays at freq_hz, so the emitted frequency is freq_hz + tone_hz.
    tone_hz: float = 0.0
    # Optional MANUAL analog-LO anchor (Hz). When >0 the tune to freq_hz pins the RF
    # LO here and reaches freq_hz with the digital DUC/NCO instead — so a wide CW
    # sweep can move the tone across the NCO window with the analog LO held fixed
    # (no synth relock, no settle glitch). 0 = ordinary automatic tuning.
    rf_freq_hz: float = 0.0

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not (0.0 <= self.amplitude <= 1.0):
            raise ValueError(f"amplitude must be 0..1, got {self.amplitude}")
        if self.freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
        if self.mode == "tone":
            return                          # a generated tone needs no files
        if self.mode == "expanded":
            if not self.iq_file:
                raise ValueError("expanded mode needs iq_file")
        elif self.mode == "pcode":
            if not self.iq_file:
                raise ValueError("pcode mode needs iq_file (chip-rate IQ)")
            if self.interp < 1:
                raise ValueError("pcode mode needs interp >= 1 (samples/chip)")
        elif self.mode == "tiered":
            if not self.primary_file:
                raise ValueError("tiered mode needs primary_file")
            if not self.secondary:
                raise ValueError("tiered mode needs a non-empty secondary overlay")
            if self.period_samples < 1:
                raise ValueError("tiered mode needs period_samples >= 1")
            if any(s not in (-1, 1) for s in self.secondary):
                raise ValueError("secondary overlay values must be ±1")
        elif self.mode == "composite":
            if not self.block_files:
                raise ValueError("composite mode needs at least one block_file")
            if not self.selectors:
                raise ValueError("composite mode needs a non-empty selector sequence")
            if any(not (0 <= s < len(self.block_files)) for s in self.selectors):
                raise ValueError("every selector must index into block_files")

    def files(self) -> List[str]:
        return [f for f in (self.iq_file, self.primary_file) if f] + list(self.block_files)


def build_playlist(spec: ChannelSpec):
    """Turn a validated ChannelSpec into a device-rate PLAYLIST: a list of distinct
    complex64 period-blocks plus a selector sequence (one index per period, looped).
    Streaming blocks[selectors[k]] in order, wrapping the sequence, reproduces the
    signal exactly — including full-length overlays — without materialising the
    whole thing. All per-mode expansion happens here, once, in NumPy; the replay
    hot loop only ever DMAs finished blocks. Shared by the real backend and tests.

    Returns (blocks: List[np.ndarray complex64 of equal length], selectors: List[int]).
    """
    import numpy as np

    def _read(path: str):
        buf = np.fromfile(path, dtype=np.complex64)
        if buf.size == 0:
            raise ValueError(f"IQ file is empty or not complex64: {path}")
        return buf

    if spec.mode == "expanded":
        return [_read(spec.iq_file)], [0]

    if spec.mode == "pcode":
        chips = _read(spec.iq_file)
        block = np.repeat(chips, int(spec.interp)) if spec.interp > 1 else chips
        return [np.ascontiguousarray(block)], [0]

    if spec.mode == "tiered":
        # Two blocks (+primary, −primary); the ±1 overlay picks one per period.
        primary = _read(spec.primary_file)
        if primary.size != spec.period_samples:
            raise ValueError(
                f"tiered primary has {primary.size} samples but period_samples="
                f"{spec.period_samples}")
        blocks = [np.ascontiguousarray(primary),
                  np.ascontiguousarray(-primary)]
        selectors = [0 if s == 1 else 1 for s in spec.secondary]
        return blocks, selectors

    # composite: the channel-task already built the distinct period-blocks and the
    # selector sequence; just read them and check they're equal-length.
    blocks = [np.ascontiguousarray(_read(f)) for f in spec.block_files]
    n = blocks[0].size
    for i, b in enumerate(blocks):
        if b.size != n:
            raise ValueError(
                f"composite block {i} has {b.size} samples, expected {n} "
                f"(all period-blocks must be the same length)")
    return blocks, list(spec.selectors)


# ── Radio backends ─────────────────────────────────────────────────────────────

class RadioBackend:
    """Interface the Engine drives. UhdRadio is the real one; MockRadio backs
    --self-test so the protocol/state machine run with no UHD or hardware."""
    master_clock_hz: float = DEFAULT_MASTER_CLOCK_HZ

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def configure(self, ch: int, target_rate_hz: float) -> float: ...
    def load(self, ch: int, spec: ChannelSpec) -> None: ...
    def clear(self, ch: int) -> None: ...
    def set_amplitude(self, ch: int, a: float) -> None: ...
    def set_freq(self, ch: int, hz: float, rf_freq_hz: float = None) -> None: ...
    def set_gain(self, ch: int, db: float) -> None: ...
    def set_tone_hz(self, ch: int, hz: float) -> None: ...
    def actual_freq(self, ch: int) -> float: ...
    def actual_rate(self, ch: int) -> float: ...
    def underflows(self, ch: int) -> int: ...
    def benchmark(self, seconds: float, rates_hz: List[float]) -> Dict[str, Any]: ...


class MockRadio(RadioBackend):
    """Records calls; validates the files it's told to load exist and that a
    channel is configured before load. No DSP, no device."""
    def __init__(self, master_clock_hz: float, channels: int):
        self.master_clock_hz = master_clock_hz
        self.channels = channels
        self.calls: List[str] = []
        self._freq = [0.0] * channels
        self._rate = [0.0] * channels

    def start(self): self.calls.append("start")
    def stop(self): self.calls.append("stop")

    def configure(self, ch, target_rate_hz):
        rate = achievable_rate(self.master_clock_hz, target_rate_hz)
        self._rate[ch] = rate
        self.calls.append(f"configure ch{ch} {rate:.0f}")
        return rate

    def load(self, ch, spec):
        if self._rate[ch] <= 0:
            raise ValueError(f"channel {ch} not configured (set a rate first)")
        for f in spec.files():
            if not os.path.exists(f):
                raise FileNotFoundError(f"IQ file not found: {f}")
        self._freq[ch] = spec.freq_hz
        self.calls.append(f"load ch{ch} {spec.mode} {spec.label}")

    def clear(self, ch): self.calls.append(f"clear ch{ch}")
    def set_amplitude(self, ch, a): self.calls.append(f"amp ch{ch} {a}")
    def set_freq(self, ch, hz, rf_freq_hz=None):
        self._freq[ch] = hz
        tag = f"freq ch{ch} {hz}" + (f" rf={rf_freq_hz}" if rf_freq_hz else "")
        self.calls.append(tag)
    def set_gain(self, ch, db): self.calls.append(f"gain ch{ch} {db}")
    def set_tone_hz(self, ch, hz): self.calls.append(f"tone ch{ch} {hz}")
    def actual_freq(self, ch): return self._freq[ch]
    def actual_rate(self, ch): return self._rate[ch]
    def underflows(self, ch): return 0

    def benchmark(self, seconds, rates_hz):
        return {"seconds": seconds, "rates_hz": rates_hz, "underflows": 0,
                "note": "mock backend"}


class UhdRadio(RadioBackend):
    """Real X410 backend: one MultiUSRP, one TX streamer per channel, each replaying
    a precomputed complex64 buffer from its own thread at its own rate. UHD is
    imported here so the module still loads for --self-test / --describe-params."""

    def __init__(self, device_args: str, master_clock_hz: float, channels: int,
                 otw: str, cpu: str = "auto", send_ms: float = DEFAULT_SEND_MS):
        import numpy as np
        import uhd
        self.np = np
        self.uhd = uhd
        self.channels = channels
        self.otw = otw
        self.cpu = resolve_cpu_format(cpu, otw)   # host format (fc32 / sc16 / sc8)
        self.send_s = max(0.001, send_ms / 1e3)   # target seconds of samples per send

        self.usrp = uhd.usrp.MultiUSRP(device_args or "type=x4xx")
        self.usrp.set_master_clock_rate(master_clock_hz)
        # Read back what the device actually locked to — stock clocks should match,
        # but the negotiated rates must divide the REAL clock, not the requested one.
        self.master_clock_hz = float(self.usrp.get_master_clock_rate())

        self._chan = [_ReplayChannel(self, ch) for ch in range(channels)]

    # ── backend interface ─────────────────────────────────────────────────────
    def start(self):
        for c in self._chan:
            c.start()

    def stop(self):
        for c in self._chan:
            c.stop()

    def configure(self, ch, target_rate_hz):
        return self._chan[ch].configure(target_rate_hz)

    def _tune_request(self, target_hz, rf_freq_hz=None):
        """A TuneRequest for target_hz. With a positive rf_freq_hz the analog RF LO
        is pinned there (MANUAL policy) and the residual is reached by the digital
        DUC/NCO — so sweeping target_hz while rf_freq_hz stays fixed moves only the
        NCO, with no analog synth relock. Otherwise UHD tunes automatically."""
        types = self.uhd.types
        if rf_freq_hz and rf_freq_hz > 0:
            tr = types.TuneRequest(float(target_hz))
            tr.rf_freq_policy = types.TuneRequestPolicy.manual
            tr.rf_freq = float(rf_freq_hz)
            tr.dsp_freq_policy = types.TuneRequestPolicy.auto
            return tr
        return types.TuneRequest(float(target_hz))

    def load(self, ch, spec):
        self.usrp.set_tx_freq(self._tune_request(spec.freq_hz, spec.rf_freq_hz), ch)
        self.usrp.set_tx_gain(spec.gain_db, ch)
        if spec.mode == "tone":
            self._chan[ch].load_tone(spec.tone_hz, spec.amplitude)
        else:
            blocks, selectors = build_playlist(spec)   # NumPy expansion, once
            self._chan[ch].load(blocks, selectors, spec.amplitude)

    def clear(self, ch):
        self._chan[ch].mute_idle()

    def set_amplitude(self, ch, a): self._chan[ch].set_amplitude(float(a))
    def set_freq(self, ch, hz, rf_freq_hz=None):
        self.usrp.set_tx_freq(self._tune_request(hz, rf_freq_hz), ch)
    def set_gain(self, ch, db): self.usrp.set_tx_gain(float(db), ch)
    def set_tone_hz(self, ch, hz): self._chan[ch].set_tone_hz(float(hz))
    def actual_freq(self, ch): return self.usrp.get_tx_freq(ch)
    def actual_rate(self, ch): return self._chan[ch].rate_hz
    def underflows(self, ch): return self._chan[ch]._underflows

    def benchmark(self, seconds: float, rates_hz: List[float]) -> Dict[str, Any]:
        """Stream a synthetic buffer per channel at the given per-channel rates for
        `seconds`, counting UHD underflow ('U') markers with the fastpath enabled.
        Mirrors the real replay path (one streamer/thread per channel)."""
        import tempfile
        np = self.np
        rates = [achievable_rate(self.master_clock_hz, r) for r in rates_hz][:self.channels]
        # A throwaway buffer per channel (10 ms of a low tone) at each rate.
        for ch, rate in enumerate(rates):
            n = max(1, int(round(rate * 0.01)))
            t = np.arange(n) / rate
            buf = (0.2 * np.exp(2j * np.pi * 1e3 * t)).astype(np.complex64)
            self.usrp.set_tx_freq(self.uhd.types.TuneRequest(1.5e9), ch)
            self._chan[ch].configure(rate)
            self._chan[ch].load([buf], [0], 0.2)

        os.environ["UHD_LOG_FASTPATH_DISABLE"] = "0"
        tmp = tempfile.TemporaryFile(mode="w+")
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            self.start()
            time.sleep(max(0.5, seconds))
            self.stop()
        finally:
            os.dup2(saved, 2); os.close(saved)
        tmp.seek(0)
        text = tmp.read()
        tmp.close()
        return {"seconds": seconds, "channels": len(rates),
                "rates_hz": rates, "master_clock_hz": self.master_clock_hz,
                "underflows": text.count("U")}


class _ReplayChannel:
    """One TX channel: a UHD streamer plus a thread that loops the current PLAYLIST
    into it. A playlist is a list of period-blocks + a selector sequence naming which
    block plays each period; streaming blocks[selectors[k]] in order (looping the
    sequence) reproduces the full signal. Playlist swaps and amplitude changes are
    done by replacing references under a lock, so the hot send loop never allocates
    and never blocks on DSP."""

    def __init__(self, radio: "UhdRadio", ch: int):
        self.radio = radio
        self.ch = ch
        self.rate_hz = 0.0
        self._lock = threading.Lock()
        self._blocks = None            # list[complex64] unit-amplitude period-blocks
        self._scaled = None            # list[complex64] blocks×amp — what's streamed
        self._selectors = None         # list[int] index into blocks, per period
        self._amp = 0.0
        self._tone = False             # generated-CW mode (no buffer)
        self._tone_hz = 0.0            # baseband tone frequency (live, drifts)
        self._tone_phase = 0.0         # phase accumulator → continuous phase across changes
        self._tone_w = None            # cached angular step; step-vector rebuilt when it changes
        self._tone_step = None         # cached exp(1j·w·n) per-sample phasor ramp (complex64)
        self._streamer = None
        self._spp = 0
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._async_thread: Optional[threading.Thread] = None
        self._underflows = 0           # TX underflows seen on this channel (async metadata)
        self._underflow_warned = 0.0   # monotonic time of last throttled underflow warning

    # ── configuration ─────────────────────────────────────────────────────────
    def configure(self, target_rate_hz: float) -> float:
        """Set this channel's TX rate and return the actual rate UHD locked to (what
        the buffer must be built for). The TX streamer is created ONCE per channel
        and reused across re-configures: on the X410's RFNoC graph a second
        get_tx_stream() on the same channel would fail with 'Attempting to reconnect
        input port!', and changing the rate needs only set_tx_rate() anyway."""
        u = self.radio
        u.usrp.set_tx_rate(float(target_rate_hz), self.ch)
        self.rate_hz = float(u.usrp.get_tx_rate(self.ch))
        if self._streamer is None:
            st_args = u.uhd.usrp.StreamArgs(u.cpu, u.otw)   # host format == wire when
            st_args.channels = [self.ch]                    # integer → memcpy send
            self._streamer = u.usrp.get_tx_stream(st_args)
            self._spp = int(self._streamer.get_max_num_samps())
        return self.rate_hz

    # ── host sample format (fc32 complex64, or interleaved int16/int8) ──────────
    def _host_dtype(self):
        """numpy dtype for one sample in the streamer's CPU format. Integer formats
        use a 2-field struct so each element is one interleaved-I/Q sample and UHD
        reads the right sample count."""
        np = self.radio.np
        cpu = self.radio.cpu
        if cpu == "fc32":
            return np.complex64
        idt = np.int16 if cpu == "sc16" else np.int8
        return np.dtype([("re", idt), ("im", idt)])

    def _to_host(self, cfloat):
        """A unit-scale complex64 buffer → a contiguous (1, N) array in the streamer's
        CPU format, ready to send with no conversion (integer) or the proven fc32
        path. Integer output is rounded and clipped to full-scale."""
        np = self.radio.np
        cpu = self.radio.cpu
        n = int(cfloat.size)
        if cpu == "fc32":
            return np.ascontiguousarray(cfloat, dtype=np.complex64).reshape(1, n)
        full = _INT_FULL_SCALE[cpu]
        idt = np.int16 if cpu == "sc16" else np.int8
        out = np.empty((1, n), dtype=np.dtype([("re", idt), ("im", idt)]))
        out["re"][0] = np.clip(np.rint(cfloat.real * full), -full, full).astype(idt)
        out["im"][0] = np.clip(np.rint(cfloat.imag * full), -full, full).astype(idt)
        return out

    def _host_zeros(self, n):
        return self.radio.np.zeros((1, int(n)), dtype=self._host_dtype())

    # ── playlist / amplitude ──────────────────────────────────────────────────
    def _scale(self, blocks, amp):
        """Scale the unit-amplitude complex64 blocks by amp and convert each ONCE to
        the host format — so the hot send loop only slices and memcpys, never
        converts."""
        if amp:
            return [self._to_host(b * amp) for b in blocks]
        return [self._host_zeros(b.size) for b in blocks]

    def load(self, blocks, selectors, amplitude: float) -> None:
        np = self.radio.np
        blocks = [np.ascontiguousarray(b, dtype=np.complex64).reshape(-1) for b in blocks]
        with self._lock:
            self._tone = False
            self._blocks = blocks                 # unit-amplitude complex64 (re-scale)
            self._selectors = list(selectors)
            self._amp = float(amplitude)
            self._scaled = self._scale(blocks, self._amp)   # host-format, streamed
        self.start()   # ensure the thread is running (streams zeros while amp==0)

    def load_tone(self, tone_hz: float, amplitude: float) -> None:
        """Switch to generated-CW mode: a continuous-phase tone at `tone_hz` off the
        LO. tone_hz is live (see set_tone_hz) so a task can drift it over time."""
        with self._lock:
            self._tone = True
            self._tone_hz = float(tone_hz)
            self._amp = float(amplitude)
            self._tone_w = None          # force the step-vector to rebuild for this load
            self._tone_step = None
            self._blocks = self._scaled = self._selectors = None
        self.start()

    def set_tone_hz(self, hz: float) -> None:
        with self._lock:
            self._tone_hz = float(hz)      # phase stays continuous via the accumulator

    def set_amplitude(self, a: float) -> None:
        with self._lock:
            self._amp = float(a)
            if self._tone or self._blocks is None:
                return
            self._scaled = self._scale(self._blocks, self._amp)

    def mute_idle(self) -> None:
        """Release a channel: mute and drop its playlist/tone. The thread keeps
        running and feeds zeros so the DUC stays fed, ready for reuse."""
        with self._lock:
            self._amp = 0.0
            self._tone = False
            self._blocks = None
            self._scaled = None
            self._selectors = None

    @staticmethod
    def _make_tone_step(np, w: float, ramp) -> "Any":
        """The per-sample phasor ramp exp(1j·w·n), n in [0,spp), as complex64. Cached
        by the caller and rebuilt only when the angular step w changes, so the tone
        hot path avoids a full np.exp on every chunk."""
        return np.exp(1j * w * ramp).astype(np.complex64)

    # ── streaming thread ──────────────────────────────────────────────────────
    def start(self) -> None:
        if self._streamer is None or self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name=f"x410-ch{self.ch}", daemon=True)
        self._thread.start()
        # Best-effort async monitor so TX underflows are counted and surfaced
        # instead of vanishing (the engine disables UHD's fastpath 'U' markers).
        self._async_thread = threading.Thread(
            target=self._monitor_async, name=f"x410-ch{self.ch}-async", daemon=True)
        self._async_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._async_thread is not None:
            self._async_thread.join(timeout=1.0)
            self._async_thread = None
        if self._streamer is not None:
            self._send_eob()

    def _monitor_async(self) -> None:
        """Drain the streamer's async message queue, counting TX underflows and
        printing a throttled warning. Best-effort: if the UHD async API isn't shaped
        as expected (e.g. a mock backend), it exits quietly and streaming continues."""
        uhd = self.radio.uhd
        try:
            md = uhd.types.TXAsyncMetadata()
            codes = uhd.types.TXMetadataEventCode
            underflow_codes = {codes.underflow, getattr(codes, "underflow_in_packet",
                                                        codes.underflow)}
        except Exception:
            return
        while self._running.is_set():
            try:
                if not self._streamer.recv_async_msg(md, 0.1):
                    continue
            except Exception:
                return
            if md.event_code in underflow_codes:
                self._underflows += 1
                now = time.monotonic()
                if now - self._underflow_warned > 1.0:
                    self._underflow_warned = now
                    print(f"[engine] ch{self.ch} TX underflow (total {self._underflows}) "
                          f"— host can't sustain {self.rate_hz/1e6:.3f} MS/s on this "
                          f"channel", file=sys.stderr, flush=True)

    def _run(self) -> None:
        np = self.radio.np
        md = self.radio.uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = False
        # Chunk by time: send ~send_s worth per call (bounded), so the call rate stays
        # low even at wide rates. This is the main lever against underflows on the ARM.
        rate0 = self.rate_hz or 1.0
        send_len = int(min(MAX_SEND_SAMPS, max(self._spp, round(rate0 * self.radio.send_s))))
        zeros = self._host_zeros(send_len)
        ramp = np.arange(send_len, dtype=np.float64)
        sel_i = 0        # position in the selector sequence
        samp_i = 0       # position within the current block
        while self._running.is_set():
            with self._lock:
                tone = self._tone
                blocks = self._scaled
                selectors = self._selectors
                tone_hz = self._tone_hz
                amp = self._amp
                rate = self.rate_hz or 1.0   # read fresh: a re-configure (reused
                                             # channel) changes the negotiated rate,
                                             # and the tone frequency must track it.
            if tone:
                # Generated CW: continuous phase across frequency (and rate) changes
                # via the phase accumulator, so a drifting tone_hz produces no phase
                # glitches. The per-sample phasor ramp exp(1j·w·n) is cached and only
                # rebuilt when w changes — the hot path is one scalar exp, one array
                # multiply and one host-format pack over a whole send_len chunk.
                w = 2.0 * np.pi * tone_hz / rate
                if w != self._tone_w or self._tone_step is None or self._tone_step.size != send_len:
                    self._tone_w = w
                    self._tone_step = self._make_tone_step(np, w, ramp)
                scal = np.complex64(amp * np.exp(1j * self._tone_phase))
                chunk = scal * self._tone_step               # complex64 × complex64
                self._streamer.send(self._to_host(chunk), md)
                md.start_of_burst = False
                self._tone_phase = (self._tone_phase + w * send_len) % (2.0 * np.pi)
                continue
            if not blocks or not selectors:
                self._streamer.send(zeros, md)
                md.start_of_burst = False
                sel_i = samp_i = 0
                continue
            if sel_i >= len(selectors):
                sel_i = 0
            buf = blocks[selectors[sel_i]]      # host-format (1, n), pre-converted
            n = buf.shape[1]
            if samp_i >= n:
                samp_i = 0
            end = min(samp_i + send_len, n)     # never cross a block edge, so looping
            self._streamer.send(buf[:, samp_i:end], md)   # is a contiguous-view send
            md.start_of_burst = False
            if end >= n:                        # block done → advance the selector
                samp_i = 0
                sel_i = (sel_i + 1) % len(selectors)
            else:
                samp_i = end

    def _send_eob(self) -> None:
        md = self.radio.uhd.types.TXMetadata()
        md.start_of_burst = False
        md.end_of_burst = True
        md.has_time_spec = False
        try:
            self._streamer.send(self._host_zeros(1), md)
        except Exception:
            pass


# ── Replay-block backend (FPGA-DRAM playback, host-rate-independent) ───────────
#
# The streaming backend feeds every sample from the ARM in real time, so it
# underflows once the per-channel rate outruns the ARM (≈10 MS/s and up). The
# Replay backend removes the ARM from the sample path entirely: each signal's loop
# is uploaded to FPGA DRAM ONCE (a non-real-time write — fc32→sc16 conversion
# happens here, off the RF deadline, at full fidelity), then the RFNoC Replay block
# streams it to the radio, looping, straight from DRAM. Nothing per-sample touches
# the host, so 61.44 MS/s is as cheap as 1 MS/s and underflows are structural-
# impossible. This is the correct high-rate path; the streaming backend stays the
# default for lower rates and live drifting CW (which a static DRAM loop can't do).

def bake_signal_loop(spec: "ChannelSpec"):
    """Materialise a buffered spec into ONE contiguous complex64 loop — the full
    selector sequence concatenated once. Looping it in the Replay block reproduces
    the signal exactly (the same guarantee the streaming playlist gives, baked once
    into DRAM instead of held in RAM). Not for tone mode."""
    import numpy as np
    blocks, selectors = build_playlist(spec)
    loop = np.concatenate([blocks[s] for s in selectors]) if len(selectors) > 1 \
        else blocks[selectors[0]]
    return np.ascontiguousarray(loop, dtype=np.complex64)


def bake_tone_loop(rate_hz: float, tone_hz: float, min_samps: int = 4096,
                   max_samps: int = 1 << 20):
    """A short complex64 CW loop at baseband tone_hz, sized to hold a whole number of
    cycles so it loops seamlessly (DC → a constant buffer). For a STATIC tone on the
    Replay backend; a drifting sweep re-bakes per step (better on the stream backend)."""
    import numpy as np
    if abs(tone_hz) < 1e-3 or rate_hz <= 0:
        return np.ones(min_samps, dtype=np.complex64)
    samples_per_cycle = rate_hz / abs(tone_hz)
    cycles = max(1, int(round(min_samps / samples_per_cycle)))   # ≥ min_samps long
    n = int(round(cycles * samples_per_cycle))
    n = max(min_samps, min(n, max_samps))
    t = np.arange(n, dtype=np.float64)
    return np.exp(2j * np.pi * tone_hz * t / rate_hz).astype(np.complex64)


def align_loop(loop_c64, samples_per_word: int, np):
    """Tile the loop to a whole number of DRAM words (still seamless) so its sc16 byte
    count is word-aligned for the Replay block. Returns the (possibly tiled) loop."""
    import math
    if samples_per_word <= 1 or loop_c64.size % samples_per_word == 0:
        return loop_c64
    reps = samples_per_word // math.gcd(loop_c64.size, samples_per_word)
    return np.tile(loop_c64, reps)


class RfnocRadio(RadioBackend):
    """Replay-block backend: one RfnocGraph, per channel a Radio(+DUC) fed by a
    Replay block that loops a DRAM buffer. Implements the same RadioBackend contract
    as UhdRadio, so the Engine drives it unchanged. Amplitude is applied when baking
    (0 = not playing = silent); gain and freq are live on the radio. tone mode bakes
    a static CW loop (re-baked on tone change).

    This path is exercised only on hardware; the RFNoC topology (block names, port
    counts, DUC presence) varies by FPGA image, so discovery is dynamic and every
    device call is logged with the step name to make bench iteration pinpointable."""

    def __init__(self, device_args: str, master_clock_hz: float, channels: int,
                 dram_mb: float = 0.0, **_ignored):
        import numpy as np
        import uhd
        self.np = np
        self.uhd = uhd
        self.channels = channels
        self._dram_mb = dram_mb

        self._log("opening RFNoC graph…")
        self.graph = uhd.rfnoc.RfnocGraph(device_args or "type=x4xx")
        self._radios, self._ducs, self._replay, self._map = self._discover(channels)
        self.master_clock_hz = float(master_clock_hz)
        self._apply_master_clock()

        # Per-channel state.
        self._rate = [0.0] * channels
        self._freq = [0.0] * channels
        self._amp = [0.0] * channels
        self._tone_hz = [0.0] * channels
        self._mode = [None] * channels
        self._loop = [None] * channels          # complex64 unit loop (re-bake on amp)
        self._playing = [False] * channels
        self._errors = [0] * channels
        self._up_streamer = [None] * channels

        self._word = self._replay_word_size()
        self._mem = self._replay_mem_size()
        self._region = self._mem // max(1, channels)   # DRAM slice per channel
        self._wire_graph()
        self.graph.commit()
        self._log(f"graph committed: {channels} ch, DRAM {self._mem/1e6:.0f} MB "
                  f"({self._region/1e6:.0f} MB/ch), word {self._word} B")

    # ── logging ────────────────────────────────────────────────────────────────
    @staticmethod
    def _log(msg: str) -> None:
        print(f"[engine/replay] {msg}", file=sys.stderr, flush=True)

    def _step(self, name: str, fn):
        """Run a device call, logging the step so a hardware failure names exactly
        which RFNoC call to look at."""
        try:
            return fn()
        except Exception as ex:
            self._log(f"FAILED at '{name}': {type(ex).__name__}: {ex}")
            raise

    # ── discovery / wiring (topology-dependent — logged for bench iteration) ────
    def _discover(self, channels: int):
        g = self.graph
        radio_ids = list(g.find_blocks("Radio"))
        duc_ids = list(g.find_blocks("DUC"))
        replay_ids = list(g.find_blocks("Replay"))
        self._log(f"blocks: radios={radio_ids} ducs={duc_ids} replay={replay_ids}")
        if not radio_ids:
            raise RuntimeError("no Radio blocks in the FPGA image")
        if not replay_ids:
            raise RuntimeError("no Replay block in the FPGA image — the Replay "
                               "backend needs a DRAM/Replay-capable image (e.g. X4_200)")
        radios = [self.uhd.rfnoc.RadioControl(g.get_block(i)) for i in radio_ids]
        ducs = [self.uhd.rfnoc.DucBlockControl(g.get_block(i)) for i in duc_ids] \
            if duc_ids else []
        replay = self.uhd.rfnoc.ReplayBlockControl(g.get_block(replay_ids[0]))

        # Flatten radio TX channels (each Radio block has get_num_input_ports TX
        # ports) into a global channel list, pairing each with a DUC (if present).
        chan_map = []
        for r_idx, (rid, rc) in enumerate(zip(radio_ids, radios)):
            n_tx = int(rc.get_num_input_ports())
            for p in range(n_tx):
                duc_idx = len(chan_map) if ducs else None
                chan_map.append({"radio": r_idx, "radio_id": rid, "radio_port": p,
                                 "duc": (duc_idx if ducs and duc_idx < len(ducs) else None),
                                 "replay_port": len(chan_map)})
        if len(chan_map) < channels:
            raise RuntimeError(f"image exposes {len(chan_map)} TX channels, need {channels}")
        chan_map = chan_map[:channels]
        for ch, m in enumerate(chan_map):
            self._log(f"ch{ch} → radio {m['radio_id']} port {m['radio_port']}, "
                      f"duc {m['duc']}, replay port {m['replay_port']}")
        self._replay_id = replay_ids[0]
        self._duc_ids = duc_ids
        return radios, ducs, replay, chan_map

    def _wire_graph(self):
        """Connect, per channel: upload-streamer → Replay in, and Replay out → (DUC) →
        Radio. Recording and playback share the Replay block; the streamer only writes
        DRAM (once, non-real-time)."""
        g = self.graph
        uhd = self.uhd
        for ch, m in enumerate(self._map):
            rport = m["replay_port"]
            sa = uhd.usrp.StreamArgs("fc32", "sc16")   # host complex → sc16 into DRAM
            sa.channels = [0]
            st = self._step(f"ch{ch} create_tx_streamer",
                            lambda: g.create_tx_streamer(1, sa))
            self._up_streamer[ch] = st
            self._step(f"ch{ch} connect streamer→replay[{rport}]",
                       lambda: g.connect(st, 0, self._replay_id, rport))
            if m["duc"] is not None:
                did = self._duc_ids[m["duc"]]
                self._step(f"ch{ch} connect replay[{rport}]→duc",
                           lambda: g.connect(self._replay_id, rport, did, 0))
                self._step(f"ch{ch} connect duc→radio",
                           lambda: g.connect(did, 0, m["radio_id"], m["radio_port"]))
            else:
                self._step(f"ch{ch} connect replay[{rport}]→radio",
                           lambda: g.connect(self._replay_id, rport,
                                             m["radio_id"], m["radio_port"]))

    def _apply_master_clock(self):
        # Prefer the motherboard controller (sets the device master clock the radios
        # run from); fall back to the radio's own rate. Best-effort — some images
        # fix the clock at load, in which case we just read what's there.
        try:
            mbc = self.graph.get_mb_controller(0)
            if hasattr(mbc, "set_master_clock_rate"):
                mbc.set_master_clock_rate(self.master_clock_hz)
        except Exception:
            pass
        for rc in self._radios:
            try:
                self.master_clock_hz = float(rc.get_rate())
                break
            except Exception:
                pass

    def _replay_word_size(self):
        for name in ("get_word_size", "get_mem_word_size"):
            fn = getattr(self._replay, name, None)
            if fn:
                try:
                    return int(fn())
                except Exception:
                    pass
        return 8       # X4xx DRAM word is 8 bytes (2× sc16 samples)

    def _replay_mem_size(self):
        fn = getattr(self._replay, "get_mem_size", None)
        try:
            return int(fn()) if fn else (1 << 31)
        except Exception:
            return 1 << 31

    # ── RadioBackend contract ──────────────────────────────────────────────────
    def start(self):
        pass   # nothing streams until a channel plays

    def stop(self):
        for ch in range(self.channels):
            self._stop_play(ch)

    def configure(self, ch, target_rate_hz):
        m = self._map[ch]
        rate = float(target_rate_hz)
        if m["duc"] is not None:
            duc = self._ducs[m["duc"]]
            self._step(f"ch{ch} duc.set_input_rate",
                       lambda: duc.set_input_rate(rate, 0))
            self._rate[ch] = float(self._step(f"ch{ch} duc.get_input_rate",
                                              lambda: duc.get_input_rate(0)))
        else:
            # No DUC: the channel streams at the radio rate (its divisors).
            self._rate[ch] = achievable_rate(self.master_clock_hz, rate)
        return self._rate[ch]

    def load(self, ch, spec):
        self.set_freq(ch, spec.freq_hz)
        self.set_gain(ch, spec.gain_db)
        self._mode[ch] = spec.mode
        if spec.mode == "tone":
            self._tone_hz[ch] = spec.tone_hz
            self._loop[ch] = bake_tone_loop(self._rate[ch], spec.tone_hz)
        else:
            self._loop[ch] = bake_signal_loop(spec)      # complex64 unit loop
        self._amp[ch] = float(spec.amplitude)
        self._apply(ch)                                   # upload + play (or stay silent)

    def clear(self, ch):
        self._stop_play(ch)
        self._loop[ch] = None
        self._mode[ch] = None
        self._amp[ch] = 0.0

    def set_amplitude(self, ch, a):
        self._amp[ch] = float(a)
        self._apply(ch)             # re-bake at the new amplitude (0 ⇒ silence)

    def set_freq(self, ch, hz, rf_freq_hz=None):
        m = self._map[ch]
        rc = self._radios[m["radio"]]
        tr = self.uhd.types.TuneRequest(float(hz))
        if rf_freq_hz and rf_freq_hz > 0:
            tr.rf_freq_policy = self.uhd.types.TuneRequestPolicy.manual
            tr.rf_freq = float(rf_freq_hz)
            tr.dsp_freq_policy = self.uhd.types.TuneRequestPolicy.auto
        self._step(f"ch{ch} radio.set_tx_frequency",
                   lambda: rc.set_tx_frequency(float(hz), m["radio_port"]))
        self._freq[ch] = float(hz)

    def set_gain(self, ch, db):
        m = self._map[ch]
        rc = self._radios[m["radio"]]
        self._step(f"ch{ch} radio.set_tx_gain",
                   lambda: rc.set_tx_gain(float(db), m["radio_port"]))

    def set_tone_hz(self, ch, hz):
        if self._mode[ch] != "tone":
            return
        self._tone_hz[ch] = float(hz)
        self._loop[ch] = bake_tone_loop(self._rate[ch], float(hz))
        self._apply(ch)             # re-bake + re-upload (brief gap; drift → use stream)

    def actual_freq(self, ch):
        return self._freq[ch]

    def actual_rate(self, ch):
        return self._rate[ch]

    def underflows(self, ch):
        return self._errors[ch]     # DRAM playback can't underflow; counts play errors

    def benchmark(self, seconds, rates_hz):
        return {"seconds": seconds, "rates_hz": rates_hz,
                "master_clock_hz": self.master_clock_hz, "underflows": 0,
                "note": "replay backend — FPGA-DRAM playback, host-rate-independent"}

    # ── DRAM upload / play ─────────────────────────────────────────────────────
    def _apply(self, ch):
        """(Re)bake the loop at the current amplitude and start looping it from DRAM.
        amplitude 0 ⇒ stop playing (silent). Called on load and on amplitude change."""
        if self._loop[ch] is None:
            return
        if self._amp[ch] <= 0.0:
            self._stop_play(ch)
            return
        np = self.np
        loop = (self._loop[ch] * self._amp[ch]).astype(np.complex64)
        loop = align_loop(loop, max(1, self._word // 4), np)   # sc16 = 4 B/sample
        nbytes = loop.size * 4                                  # sc16 bytes in DRAM
        if nbytes > self._region:
            raise ValueError(
                f"channel {ch} loop is {nbytes/1e6:.1f} MB but the per-channel DRAM "
                f"budget is {self._region/1e6:.1f} MB — lower the rate or shorten the "
                f"loop (or use --backend stream for this signal)")
        self._stop_play(ch)
        self._upload(ch, loop)
        self._start_play(ch, nbytes)

    def _upload(self, ch, loop_c64):
        rport = self._map[ch]["replay_port"]
        offset = ch * self._region
        nbytes = loop_c64.size * 4                              # sc16 in DRAM
        rep, st, uhd = self._replay, self._up_streamer[ch], self.uhd
        self._step(f"ch{ch} replay.record",
                   lambda: rep.record(offset, nbytes, rport))
        # Upload as one burst; the fc32→sc16 conversion happens here, off the RF
        # deadline, so it costs nothing in real time and keeps full sc16 fidelity.
        md = uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = True
        md.has_time_spec = False
        self._step(f"ch{ch} upload send()",
                   lambda: st.send(loop_c64.reshape(1, loop_c64.size), md))
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if int(rep.get_record_fullness(rport)) >= nbytes:
                    break
            except Exception:
                break
            time.sleep(0.01)
        self._log(f"ch{ch} uploaded {nbytes/1e3:.1f} kB to DRAM offset {offset}")

    def _start_play(self, ch, nbytes):
        rport = self._map[ch]["replay_port"]
        offset = ch * self._region
        rep, uhd = self._replay, self.uhd
        self._step(f"ch{ch} replay.set_play_type(sc16)",
                   lambda: rep.set_play_type("sc16", rport))
        self._step(f"ch{ch} replay.config_play",
                   lambda: rep.config_play(offset, nbytes, rport))
        try:
            self._step(f"ch{ch} replay.play(repeat)",
                       lambda: rep.play(offset, nbytes, rport,
                                        uhd.types.TimeSpec(0.0), True))
        except TypeError:
            # Older signature without repeat/time_spec — set repeat separately.
            if hasattr(rep, "set_play_repeat"):
                rep.set_play_repeat(True, rport)
            self._step(f"ch{ch} replay.play",
                       lambda: rep.play(offset, nbytes, rport))
        self._playing[ch] = True

    def _stop_play(self, ch):
        if not self._playing[ch]:
            return
        try:
            self._replay.stop(self._map[ch]["replay_port"])
        except Exception as ex:
            self._errors[ch] += 1
            self._log(f"ch{ch} replay.stop failed: {ex}")
        self._playing[ch] = False


# ── Engine: per-channel ownership + state, delegating to a backend ─────────────

class Engine:
    def __init__(self, backend: RadioBackend, channels: int = NUM_CHANNELS):
        self.backend = backend
        self.channels = channels
        self._lock = threading.RLock()
        self._owner: List[Optional[str]] = [None] * channels
        self._signal: List[Optional[str]] = [None] * channels
        self._amp: List[float] = [0.0] * channels
        self._freq: List[float] = [0.0] * channels
        self._rate: List[float] = [0.0] * channels

    @property
    def master_clock_hz(self) -> float:
        return self.backend.master_clock_hz

    def _check(self, ch: int, owner: Optional[str], need_owner: bool = True) -> None:
        if not (0 <= ch < self.channels):
            raise ValueError(f"channel must be 0..{self.channels - 1}, got {ch}")
        if need_owner and self._owner[ch] is not None and self._owner[ch] != owner:
            raise PermissionError(
                f"channel {ch} is owned by {self._owner[ch]!r}, not {owner!r}")

    def acquire(self, ch: int, owner: str) -> None:
        with self._lock:
            if not (0 <= ch < self.channels):
                raise ValueError(f"channel must be 0..{self.channels - 1}")
            if self._owner[ch] is not None and self._owner[ch] != owner:
                raise PermissionError(f"channel {ch} already owned by {self._owner[ch]!r}")
            self._owner[ch] = owner

    def release(self, ch: int, owner: str) -> None:
        with self._lock:
            self._check(ch, owner)
            self.backend.clear(ch)
            self._owner[ch] = None
            self._signal[ch] = None
            self._amp[ch] = 0.0

    def configure(self, ch: int, owner: str, target_rate_hz: float) -> float:
        with self._lock:
            self._check(ch, owner)
            actual = self.backend.configure(ch, target_rate_hz)   # may raise first
            self._rate[ch] = actual
            if self._owner[ch] is None:
                self._owner[ch] = owner            # implicit acquire once it succeeds
            return actual

    def load(self, ch: int, owner: str, spec: ChannelSpec) -> None:
        spec.validate()
        with self._lock:
            self._check(ch, owner)
            if self._rate[ch] <= 0:
                raise ValueError(
                    f"channel {ch} not configured — send 'configure' before 'load'")
            if self._owner[ch] is None:
                self._owner[ch] = owner
            self.backend.load(ch, spec)
            self._signal[ch] = spec.label or spec.mode
            self._amp[ch] = spec.amplitude
            self._freq[ch] = spec.freq_hz

    def set(self, ch: int, owner: str, *, amplitude=None, freq_hz=None,
            gain_db=None, tone_hz=None, rf_freq_hz=None) -> None:
        with self._lock:
            self._check(ch, owner)
            if amplitude is not None:
                self.backend.set_amplitude(ch, amplitude); self._amp[ch] = amplitude
            if freq_hz is not None:
                self.backend.set_freq(ch, freq_hz, rf_freq_hz); self._freq[ch] = freq_hz
            if gain_db is not None:
                self.backend.set_gain(ch, gain_db)
            if tone_hz is not None:
                self.backend.set_tone_hz(ch, tone_hz)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"master_clock_hz": self.master_clock_hz, "channels": [
                {"channel": ch, "owner": self._owner[ch], "signal": self._signal[ch],
                 "amplitude": self._amp[ch], "freq_hz": self._freq[ch],
                 "rate_hz": self._rate[ch], "underflows": self.backend.underflows(ch)}
                for ch in range(self.channels)]}

    def benchmark(self, seconds: float, rates_hz: List[float]) -> Dict[str, Any]:
        return self.backend.benchmark(seconds, rates_hz)


# ── Control server (JSON object per line over a Unix stream socket) ────────────

class ControlServer:
    def __init__(self, engine: Engine, socket_path: str):
        self.engine = engine
        self.socket_path = socket_path
        self._stop = threading.Event()
        self._srv: Optional[socket.socket] = None

    def _dispatch(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        cmd = msg.get("cmd")
        e = self.engine
        try:
            if cmd == "hello":
                return {"ok": True, **e.status()}
            if cmd == "status":
                return {"ok": True, **e.status()}
            if cmd == "benchmark":
                rates = msg.get("rates_hz") or [msg.get("rate_hz", 61.44e6)]
                return {"ok": True, **e.benchmark(float(msg.get("seconds", 5)),
                                                  [float(r) for r in rates])}
            ch = int(msg["channel"])
            owner = msg.get("owner", "")
            if cmd == "acquire":
                e.acquire(ch, owner); return {"ok": True}
            if cmd == "release":
                e.release(ch, owner); return {"ok": True}
            if cmd == "configure":
                actual = e.configure(ch, owner, float(msg["target_rate_hz"]))
                return {"ok": True, "actual_rate_hz": actual,
                        "master_clock_hz": e.master_clock_hz}
            if cmd == "load":
                e.load(ch, owner, ChannelSpec(**msg["spec"])); return {"ok": True}
            if cmd == "set":
                e.set(ch, owner, amplitude=msg.get("amplitude"),
                      freq_hz=msg.get("freq_hz"), gain_db=msg.get("gain_db"),
                      tone_hz=msg.get("tone_hz"), rf_freq_hz=msg.get("rf_freq_hz"))
                return {"ok": True}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except KeyError as ex:
            return {"ok": False, "error": f"missing field: {ex}"}
        except (ValueError, PermissionError, FileNotFoundError) as ex:
            # Expected client-facing rejections (bad rate, wrong owner, load before
            # configure, missing IQ file, …). These are normal feedback to the
            # caller, so reply with the message and no stack trace — no traceback
            # noise for a routine no.
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
        except Exception as ex:
            # Any *unexpected* backend/UHD error (RuntimeError from UHD, …) becomes
            # an error reply — never let it kill the serve thread and drop the
            # connection, or one bad command would take the whole engine down for
            # every channel. Log it with a traceback since it's genuinely unexpected.
            import traceback
            print(f"[engine] command {cmd!r} failed: {type(ex).__name__}: {ex}",
                  file=sys.stderr)
            traceback.print_exc()
            return {"ok": False, "error": f"{type(ex).__name__}: {ex}"}

    def _serve_conn(self, conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as ex:
                    reply = {"ok": False, "error": f"bad JSON: {ex}"}
                else:
                    reply = self._dispatch(msg)
                f.write((json.dumps(reply) + "\n").encode()); f.flush()

    def serve_forever(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(self.socket_path)
        self._srv.listen(8)
        self._srv.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._srv:
            self._srv.close()
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass


# ── Self-test (protocol + state machine, MockRadio, no UHD) ────────────────────

def _self_test() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}· {msg}")
        ok = ok and cond

    # achievable_rate arithmetic
    check(achievable_rate(245.76e6, 8e6) == 245.76e6 / 31, "achievable_rate picks nearest divisor")
    check(achievable_rate(245.76e6, 61.44e6) == 61.44e6, "achievable_rate exact when it divides")

    # ChannelSpec validation
    try:
        ChannelSpec(mode="bogus", freq_hz=1e9).validate(); check(False, "bad mode rejected")
    except ValueError:
        check(True, "bad mode rejected")
    try:
        ChannelSpec(mode="pcode", freq_hz=1e9, iq_file="x", interp=0).validate()
        check(False, "pcode interp<1 rejected")
    except ValueError:
        check(True, "pcode interp<1 rejected")
    try:
        ChannelSpec(mode="tiered", freq_hz=1e9, primary_file="x",
                    secondary=[1, 2], period_samples=10).validate()
        check(False, "non-±1 secondary rejected")
    except ValueError:
        check(True, "non-±1 secondary rejected")

    # build_playlist (needs NumPy; skip gracefully if absent)
    try:
        import numpy as np
        tmpd = tempfile.mkdtemp()
        chipf = os.path.join(tmpd, "chips.fc32")
        np.array([1 + 0j, -1 + 0j, 1 + 0j], dtype=np.complex64).tofile(chipf)
        blk, sel = build_playlist(ChannelSpec(mode="pcode", freq_hz=1e9, iq_file=chipf, interp=4))
        check(len(blk) == 1 and sel == [0] and blk[0].size == 12
              and blk[0][0] == 1 and blk[0][4] == -1, "pcode np.repeat expands chips")
        primf = os.path.join(tmpd, "prim.fc32")
        np.array([1 + 0j, 1j], dtype=np.complex64).tofile(primf)
        blk, sel = build_playlist(ChannelSpec(mode="tiered", freq_hz=1e9, primary_file=primf,
                                              secondary=[1, -1, 1], period_samples=2))
        check(len(blk) == 2 and sel == [0, 1, 0] and blk[1][0] == -1 and blk[1][1] == -1j,
              "tiered → [+primary,−primary] + selectors from overlay")

        # composite: two blocks + a selector sequence, and the EXACTNESS guarantee —
        # the streamed playlist equals a single fully-baked buffer, byte for byte.
        b0 = os.path.join(tmpd, "b0.fc32"); b1 = os.path.join(tmpd, "b1.fc32")
        B0 = np.array([1 + 0j, 2 + 0j, 3 + 0j], dtype=np.complex64)
        B1 = np.array([-1 + 0j, -2 + 0j, -3 + 0j], dtype=np.complex64)
        B0.tofile(b0); B1.tofile(b1)
        selectors = [0, 1, 1, 0]
        blocks, sel = build_playlist(ChannelSpec(
            mode="composite", freq_hz=1e9, block_files=[b0, b1], selectors=selectors))
        streamed = np.concatenate([blocks[s] for s in sel])           # what the thread emits
        baked = np.concatenate([[B0, B1][s] for s in selectors])      # a single big buffer
        check(np.array_equal(streamed, baked), "composite playlist == fully-baked buffer (exact)")
        # unequal-length blocks are rejected
        bshort = os.path.join(tmpd, "bs.fc32")
        np.array([1 + 0j], dtype=np.complex64).tofile(bshort)
        try:
            build_playlist(ChannelSpec(mode="composite", freq_hz=1e9,
                                       block_files=[b0, bshort], selectors=[0, 1]))
            check(False, "unequal-length composite blocks rejected")
        except ValueError:
            check(True, "unequal-length composite blocks rejected")
    except ImportError:
        check(True, "build_playlist skipped (no NumPy here)")

    # _ReplayChannel reuses its TX streamer across re-configures (a second
    # get_tx_stream on the same channel fails with 'reconnect input port' on the
    # X410 RFNoC graph — re-running a task must not recreate the streamer).
    try:
        import types as _types
        import numpy as _np
        _n = {"streams": 0}

        class _FakeStreamer:
            def get_max_num_samps(self): return 2000
            def send(self, b, m): pass

        class _FakeUsrp:
            def __init__(self): self.rate = 0.0
            def set_tx_rate(self, r, c): self.rate = float(r)
            def get_tx_rate(self, c): return self.rate
            def get_tx_stream(self, a): _n["streams"] += 1; return _FakeStreamer()

        class _FakeRadio:
            np = _np
            otw = "sc16"
            cpu = "fc32"
            send_s = 0.01
            def __init__(self): self.usrp = _FakeUsrp()
            uhd = _types.SimpleNamespace(usrp=_types.SimpleNamespace(
                StreamArgs=lambda a, b: _types.SimpleNamespace(channels=[])))

        rc = _ReplayChannel(_FakeRadio(), 0)
        rc.configure(40.96e6)
        mid = rc.configure(20.48e6)                 # rate change → same streamer
        rc.configure(40.96e6)
        check(_n["streams"] == 1 and mid == 20.48e6 and rc.rate_hz == 40.96e6,
              "_ReplayChannel reuses TX streamer across re-configures")
    except ImportError:
        check(True, "streamer-reuse check skipped (no NumPy here)")

    # Host-format packing: fc32 stays complex64; sc16/sc8 pack to interleaved-int
    # samples (one struct element per sample) with correct scale, clipping and a
    # zero-conversion (1, N) shape — this is what makes send() a memcpy and what an
    # 8-bit wire needs (no fc32→sc8 converter in this UHD build).
    try:
        import types as _types
        import numpy as _np

        def _chan(cpu):
            fr = _types.SimpleNamespace(np=_np, cpu=cpu, otw=cpu,
                                        uhd=None, send_s=0.01)
            return _ReplayChannel(fr, 0)

        c = _np.array([0.5 - 0.25j, -1.0 + 1.0j, 2.0 + 0j], dtype=_np.complex64)

        h32 = _chan("fc32")._to_host(c)
        check(h32.dtype == _np.complex64 and h32.shape == (1, 3)
              and _np.allclose(h32[0], c), "fc32 host stays complex64 (1,N)")

        h16 = _chan("sc16")._to_host(c)
        check(h16.shape == (1, 3) and h16.dtype.names == ("re", "im")
              and h16["re"][0, 0] == round(0.5 * 32767)
              and h16["im"][0, 0] == round(-0.25 * 32767)
              and h16["re"][0, 2] == 32767,             # +2.0 clips to full-scale
              "sc16 host packs interleaved int16 with scale + clip")

        h8 = _chan("sc8")._to_host(c)
        check(h8.dtype.names == ("re", "im") and h8["re"].dtype == _np.int8
              and h8["re"][0, 1] == -127 and h8["im"][0, 1] == 127,
              "sc8 host packs interleaved int8 with clip to ±127")

        z = _chan("sc8")._host_zeros(4)
        check(z.shape == (1, 4) and int(z["re"][0, 0]) == 0 and int(z["im"][0, 3]) == 0,
              "host zeros are silent in the wire format")
    except ImportError:
        check(True, "host-format check skipped (no NumPy here)")

    # Replay-backend baking (no hardware): the DRAM loop is the exact concatenation
    # of the streaming playlist, the tone loop holds whole cycles (seamless), and
    # word-alignment tiles without altering the waveform.
    try:
        import numpy as _np
        import tempfile as _tf
        d = _tf.mkdtemp()
        f0 = os.path.join(d, "e.fc32")
        sig = _np.array([1 + 0j, 1j, -1 + 0j, -1j, 2 + 0j, 3 + 0j], dtype=_np.complex64)
        sig.tofile(f0)
        loop = bake_signal_loop(ChannelSpec(mode="expanded", freq_hz=1e9, iq_file=f0))
        check(_np.array_equal(loop, sig) and loop.dtype == _np.complex64,
              "bake_signal_loop expands a spec to its exact loop")

        fa = os.path.join(d, "a.fc32"); fb = os.path.join(d, "b.fc32")
        A = _np.array([1 + 0j, 2 + 0j], dtype=_np.complex64)
        Bb = _np.array([-1 + 0j, -2 + 0j], dtype=_np.complex64)
        A.tofile(fa); Bb.tofile(fb)
        cloop = bake_signal_loop(ChannelSpec(mode="composite", freq_hz=1e9,
                                             block_files=[fa, fb], selectors=[0, 1, 1, 0]))
        check(_np.array_equal(cloop, _np.concatenate([A, Bb, Bb, A])),
              "bake_signal_loop composite == concatenated selector sequence")

        dc = bake_tone_loop(4e6, 0.0)
        check(_np.allclose(dc, 1.0), "bake_tone_loop DC → constant carrier")
        tl = bake_tone_loop(4e6, 1e6)          # 4 samples/cycle, whole cycles
        w = 2.0 * _np.pi * 1e6 / 4e6
        fmeas = _np.mean(_np.angle(tl[1:] * _np.conj(tl[:-1]))) / (2 * _np.pi) * 4e6
        seam = _np.angle(tl[0] * _np.conj(tl[-1]))    # wrap step ≈ one sample step
        check(abs(fmeas - 1e6) < 1.0 and abs(abs(seam) - w) < 1e-3,
              "bake_tone_loop holds the tone and loops seamlessly")

        aligned = align_loop(_np.ones(3, dtype=_np.complex64), 2, _np)
        check(aligned.size % 2 == 0
              and align_loop(_np.ones(4, dtype=_np.complex64), 2, _np).size == 4,
              "align_loop tiles to a whole DRAM word, leaves aligned loops alone")
    except ImportError:
        check(True, "replay-bake check skipped (no NumPy here)")

    # Generated-tone math (mirrors the _run hot path): the emitted baseband
    # frequency equals tone_hz regardless of the negotiated sample rate, and the
    # phase accumulator keeps the signal continuous when tone_hz changes mid-drift.
    try:
        import numpy as _np

        def _emit(rate_hz, tone_hz, spp=257, nchunks=12, phase0=0.0):
            ramp = _np.arange(spp, dtype=_np.float64)
            w = 2.0 * _np.pi * tone_hz / rate_hz
            step = _ReplayChannel._make_tone_step(_np, w, ramp)   # production builder
            phase = phase0
            out = []
            for _ in range(nchunks):
                scal = _np.complex64(_np.exp(1j * phase))
                out.append(scal * step)
                phase = (phase + w * spp) % (2.0 * _np.pi)
            return _np.concatenate(out), phase

        def _measure_hz(x, rate_hz):
            d = _np.angle(x[1:] * _np.conj(x[:-1]))               # per-sample phase step
            return float(_np.mean(d) / (2.0 * _np.pi) * rate_hz)

        f_hi, _ = _emit(40.96e6, 12345.0)
        f_lo, _ = _emit(2.048e6, 12345.0)
        check(abs(_measure_hz(f_hi, 40.96e6) - 12345.0) < 1.0 and
              abs(_measure_hz(f_lo, 2.048e6) - 12345.0) < 1.0,
              "generated-tone frequency == tone_hz, independent of sample rate")

        # continuity across a tone_hz change: end one chunk-run, carry the phase into
        # a new frequency, and confirm the seam has no phase discontinuity. The
        # accumulator carries the phase forward, so the seam sample still advances by
        # the OLD per-sample step (the new frequency applies from the next sample) —
        # a smooth continuation with no jump, which is the whole point of the phasor.
        a, ph = _emit(2.048e6, 3000.0, nchunks=4)
        b, _ = _emit(2.048e6, -4000.0, nchunks=4, phase0=ph)
        w_old = 2.0 * _np.pi * 3000.0 / 2.048e6
        seam = _np.angle(b[0] * _np.conj(a[-1])) - w_old
        check(abs((seam + _np.pi) % (2.0 * _np.pi) - _np.pi) < 1e-6,
              "tone phase stays continuous across a frequency change")
    except ImportError:
        check(True, "tone-frequency check skipped (no NumPy here)")

    # End-to-end over a real Unix socket with the mock backend.
    tmp = tempfile.mkdtemp()
    iqf = os.path.join(tmp, "sig.fc32")
    open(iqf, "wb").write(b"\x00" * 64)          # a file that "exists"
    sock = os.path.join(tmp, "eng.sock")
    eng = Engine(MockRadio(DEFAULT_MASTER_CLOCK_HZ, NUM_CHANNELS))
    srv = ControlServer(eng, sock)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    for _ in range(50):
        if os.path.exists(sock):
            break
        time.sleep(0.02)

    def call(msg):
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.connect(sock)
        f = c.makefile("rwb")
        f.write((json.dumps(msg) + "\n").encode()); f.flush()
        reply = json.loads(f.readline()); c.close(); return reply

    hello = call({"cmd": "hello"})
    check(hello["ok"] and hello["master_clock_hz"] == DEFAULT_MASTER_CLOCK_HZ,
          "hello reports master clock")
    check(call({"cmd": "acquire", "channel": 0, "owner": "A"})["ok"], "acquire ch0 by A")
    check(not call({"cmd": "acquire", "channel": 0, "owner": "B"})["ok"],
          "second owner B rejected on ch0")
    # load before configure is rejected
    check(not call({"cmd": "load", "channel": 0, "owner": "A",
                    "spec": {"mode": "expanded", "freq_hz": 1.57542e9, "iq_file": iqf,
                             "amplitude": 0.0}})["ok"],
          "load before configure rejected")
    r = call({"cmd": "configure", "channel": 0, "owner": "A", "target_rate_hz": 8e6})
    check(r["ok"] and abs(r["actual_rate_hz"] - 245.76e6 / 31) < 1,
          "configure returns negotiated rate")
    r = call({"cmd": "load", "channel": 0, "owner": "A",
              "spec": {"mode": "expanded", "freq_hz": 1.57542e9, "iq_file": iqf,
                       "amplitude": 0.0, "label": "gps_l1_ca prn1"}})
    check(r["ok"], "load expanded on ch0 after configure")
    check(not call({"cmd": "set", "channel": 0, "owner": "B", "amplitude": 0.5})["ok"],
          "set by wrong owner rejected")
    check(call({"cmd": "set", "channel": 0, "owner": "A", "amplitude": 0.9,
                "gain_db": 40})["ok"], "on-air set{amplitude,gain} by owner A")
    # implicit acquire via configure on a fresh channel + pcode load
    check(call({"cmd": "configure", "channel": 1, "owner": "C",
                "target_rate_hz": 10.23e6})["ok"], "configure ch1 (implicit acquire by C)")
    r = call({"cmd": "load", "channel": 1, "owner": "C",
              "spec": {"mode": "pcode", "freq_hz": 1.602e9, "iq_file": iqf,
                       "interp": 4, "label": "glonass_p"}})
    check(r["ok"], "load pcode on ch1")
    # composite load over the socket (two blocks + selectors)
    check(call({"cmd": "configure", "channel": 3, "owner": "D",
                "target_rate_hz": 20.48e6})["ok"], "configure ch3 for composite")
    r = call({"cmd": "load", "channel": 3, "owner": "D",
              "spec": {"mode": "composite", "freq_hz": 1.57542e9,
                       "block_files": [iqf, iqf], "selectors": [0, 1, 1, 0],
                       "label": "l1c prn5"}})
    check(r["ok"], "load composite on ch3")
    st = call({"cmd": "status"})
    owners = {c["channel"]: c["owner"] for c in st["channels"]}
    rates = {c["channel"]: c["rate_hz"] for c in st["channels"]}
    check(owners[0] == "A" and owners[1] == "C" and owners[2] is None,
          "status reflects owners")
    check(rates[0] > 0 and rates[1] > 0 and rates[2] == 0, "status reflects per-channel rates")
    check(call({"cmd": "release", "channel": 0, "owner": "A"})["ok"], "release ch0")
    check(call({"cmd": "status"})["channels"][0]["owner"] is None, "ch0 free after release")
    check(not call({"cmd": "configure", "channel": 2, "owner": "A",
                    "target_rate_hz": 0})["ok"], "non-positive rate rejected")
    check(not call({"cmd": "load", "channel": 2, "owner": "A",
                    "spec": {"mode": "expanded", "freq_hz": 1e9,
                             "iq_file": "/no/such/file"}})["ok"],
          "load on unconfigured/ missing-file channel rejected")
    # tone mode: generated CW needs no files, and tone_hz drifts via set
    check(call({"cmd": "configure", "channel": 2, "owner": "T",
                "target_rate_hz": 4e6})["ok"], "configure ch2 for tone")
    r = call({"cmd": "load", "channel": 2, "owner": "T",
              "spec": {"mode": "tone", "freq_hz": 1.57542e9, "tone_hz": 0.0,
                       "amplitude": 0.0, "label": "cw"}})
    check(r["ok"], "load tone on ch2 (no files)")
    check(call({"cmd": "set", "channel": 2, "owner": "T", "tone_hz": 1234.0})["ok"],
          "drift tone_hz via set")
    # manual-LO (rf_freq_hz) tune passes through — the analog LO is pinned while the
    # emitted frequency is reached by the DUC/NCO (wide CW sweep with no synth relock).
    eng.backend.calls.clear()
    check(call({"cmd": "set", "channel": 2, "owner": "T", "freq_hz": 1.6e9,
                "rf_freq_hz": 1.58e9})["ok"], "set with manual rf_freq_hz accepted")
    check(any("rf=1.58" in c or "rf=1580000000" in c for c in eng.backend.calls),
          "manual rf_freq_hz reaches the backend tune")
    srv.stop()

    print("ALL ENGINE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ─────────────────────────────────────────────

def build_script() -> Script:
    clocks = {f"{c/1e6:g} MHz": c / 1e6 for c in STOCK_MASTER_CLOCKS_HZ}  # values in MHz
    return (
        Script("Persistent multi-channel IQ-replay engine for the USRP X410. Owns "
               "the device; channel-tasks drive channels (each at its own sample "
               "rate) over a socket.")
        .number("-Master-clock", "--master_clock", unit="MHz", min=100.0, max=500.0,
                presets=clocks, default=DEFAULT_MASTER_CLOCK_HZ / 1e6,
                help="Device master clock (stock rates only). Fixed for the engine's "
                     "lifetime; per-channel rates are its integer divisors.")
        .integer("-Channels", "--channels", min=1, max=4, default=4,
                 help="Number of TX channels to open.")
        .text("-Socket", "--socket", default="/tmp/x410_engine.sock",
              help="Unix socket path the control clients connect to.")
        .text("-Device-args", "--device_args", default="type=x4xx",
              help="UHD device args, e.g. 'type=x4xx,addr=192.168.10.2'.")
        .choice("-Backend", "--backend",
                options={"stream": "host streams every sample (default; simple, "
                                   "supports live drifting CW; underflows ≳10 MS/s)",
                         "replay": "FPGA-DRAM Replay block loops each signal — "
                                   "host-rate-independent, no underflows at any rate"},
                default="stream",
                help="Playback engine. 'replay' uploads each signal to FPGA DRAM once "
                     "and loops it from there, so wide rates (61.44 MS/s) never "
                     "underflow. Needs a Replay-capable FPGA image.")
        .choice("-OTW-format", "--otw",
                options={"sc16": "16-bit (default, full range)",
                         "sc8": "8-bit (halves the wire rate — helps at ≥10 MS/s)"},
                default="sc16", help="Over-the-wire sample format (stream backend).")
        .choice("-CPU-format", "--cpu",
                options={"auto": "match the wire (fc32 for sc16, sc8 for sc8)",
                         "fc32": "complex float host (only valid with sc16 wire)",
                         "sc16": "int16 host — memcpy send, no conversion",
                         "sc8": "int8 host — memcpy send (required for an sc8 wire)"},
                default="auto",
                help="Host sample format. Matching it to the wire makes send() a "
                     "memcpy (no per-sample conversion). An sc8 wire REQUIRES sc8/sc16 "
                     "host — this UHD build has no fc32→sc8 converter.")
        .number("-Send-ms", "--send_ms", unit="ms", min=1.0, max=100.0,
                default=DEFAULT_SEND_MS,
                help="Milliseconds of samples per send() call. Larger = fewer calls = "
                     "less ARM overhead (fewer underflows), at the cost of live-tune "
                     "latency. 10 ms is a good default.")
        .number("-Benchmark", "--benchmark", unit="s", min=0.0, max=120.0, default=0.0,
                help="If >0, run an underflow probe for this many seconds (per-channel "
                     "streamers at --bench_rates) and exit.")
        .text("-Benchmark-rates", "--bench_rates", default="61.44,8.192,8.192,8.192",
              help="Comma-separated per-channel rates in MHz for --benchmark.")
    )


def _parse_rates_mhz(s: str) -> List[float]:
    return [float(x) * 1e6 for x in str(s).split(",") if x.strip()]


def build_backend(args, master_clock_hz):
    """Construct the selected backend. 'replay' uses the FPGA Replay block; 'stream'
    (default) uses the host-streaming path."""
    if args.backend == "replay":
        return RfnocRadio(args.device_args, master_clock_hz, args.channels)
    return UhdRadio(args.device_args, master_clock_hz, args.channels, args.otw,
                    args.cpu, args.send_ms)


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    args = build_script().parse()
    master_clock_hz = args.master_clock * 1e6

    if args.benchmark and args.benchmark > 0:
        rates = _parse_rates_mhz(args.bench_rates)
        radio = build_backend(args, master_clock_hz)
        print(f"[benchmark] {args.channels} ch, master {radio.master_clock_hz/1e6:g} MHz, "
              f"rates {[f'{r/1e6:g}' for r in rates]} MHz, {args.benchmark:g}s…", flush=True)
        result = radio.benchmark(args.benchmark, rates)
        print(json.dumps(result, indent=2))
        u = result.get("underflows", 0)
        print("RESULT:", "clean (no underflows)" if u == 0 else f"{u} underflow marker(s)")
        return 0

    radio = build_backend(args, master_clock_hz)
    engine = Engine(radio, channels=args.channels)
    server = ControlServer(engine, args.socket)

    radio.start()
    print("── X410 engine ─────────────────────────────────────────────")
    print(f"  device         : {args.device_args}")
    print(f"  master clock   : {radio.master_clock_hz/1e6:g} MHz  ({args.channels} channels)")
    if args.backend == "replay":
        print("  backend        : replay (FPGA-DRAM loop — no host-rate limit)")
    else:
        print(f"  backend        : stream — host {radio.cpu} → wire {args.otw}"
              f"{'  (memcpy send)' if radio.cpu == args.otw else ''}, "
              f"{args.send_ms:g} ms/send")
    print(f"  control socket : {args.socket}")
    print("  all channels idle (muted); waiting for channel-tasks…")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    import signal as _signal
    stop = threading.Event()
    _signal.signal(_signal.SIGTERM, lambda *_: stop.set())
    _signal.signal(_signal.SIGINT, lambda *_: stop.set())
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        while not stop.is_set():
            time.sleep(0.2)
    finally:
        server.stop()
        radio.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
Every mode reduces to "produce one device-rate complex64 buffer, then replay it";
the expansion happens once, at load, in NumPy — never per sample in the hot loop:

  • "expanded"  device-rate IQ file → replay as-is                 (lightest; pure DMA)
  • "tiered"    primary-period IQ × a ±1 secondary overlay, one    (L1C / B1C)
                sign per primary period → built into one buffer
  • "pcode"     chip-rate IQ file, each chip repeated `interp`×     (long m-seqs, e.g. GLONASS P)
                (np.repeat) → device-rate buffer

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
MODES = ("expanded", "tiered", "pcode")

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
    """A signal to play on one channel. `mode` selects how the device-rate buffer
    is built (see module docstring). Any IQ files must already exist on disk, built
    at the channel's negotiated rate (see the `configure` command)."""
    mode: str                       # "expanded" | "tiered" | "pcode"
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

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not (0.0 <= self.amplitude <= 1.0):
            raise ValueError(f"amplitude must be 0..1, got {self.amplitude}")
        if self.freq_hz <= 0:
            raise ValueError("freq_hz must be positive")
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

    def files(self) -> List[str]:
        return [f for f in (self.iq_file, self.primary_file) if f]


def build_base_buffer(spec: ChannelSpec):
    """Turn a validated ChannelSpec into ONE device-rate complex64 buffer, ready to
    replay. All the per-mode expansion happens here, once, in NumPy — the replay
    hot loop only ever streams the finished buffer. Shared by the real backend and
    tests so the two never diverge."""
    import numpy as np

    def _read(path: str):
        buf = np.fromfile(path, dtype=np.complex64)
        if buf.size == 0:
            raise ValueError(f"IQ file is empty or not complex64: {path}")
        return buf

    if spec.mode == "expanded":
        return _read(spec.iq_file)

    if spec.mode == "pcode":
        chips = _read(spec.iq_file)
        return np.repeat(chips, int(spec.interp)) if spec.interp > 1 else chips

    # tiered: one primary period, sign-flipped per secondary chip, concatenated.
    primary = _read(spec.primary_file)
    if primary.size != spec.period_samples:
        raise ValueError(
            f"tiered primary has {primary.size} samples but period_samples="
            f"{spec.period_samples}")
    signs = np.asarray(spec.secondary, dtype=np.float32)
    # (period, 1) × (n_periods,) → (n_periods, period) → flat: primary tiled and
    # sign-applied per period, without a Python loop.
    return (primary[None, :] * signs[:, None]).astype(np.complex64).reshape(-1)


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
    def set_freq(self, ch: int, hz: float) -> None: ...
    def set_gain(self, ch: int, db: float) -> None: ...
    def actual_freq(self, ch: int) -> float: ...
    def actual_rate(self, ch: int) -> float: ...
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
    def set_freq(self, ch, hz): self._freq[ch] = hz; self.calls.append(f"freq ch{ch} {hz}")
    def set_gain(self, ch, db): self.calls.append(f"gain ch{ch} {db}")
    def actual_freq(self, ch): return self._freq[ch]
    def actual_rate(self, ch): return self._rate[ch]

    def benchmark(self, seconds, rates_hz):
        return {"seconds": seconds, "rates_hz": rates_hz, "underflows": 0,
                "note": "mock backend"}


class UhdRadio(RadioBackend):
    """Real X410 backend: one MultiUSRP, one TX streamer per channel, each replaying
    a precomputed complex64 buffer from its own thread at its own rate. UHD is
    imported here so the module still loads for --self-test / --describe-params."""

    def __init__(self, device_args: str, master_clock_hz: float, channels: int,
                 otw: str):
        import numpy as np
        import uhd
        self.np = np
        self.uhd = uhd
        self.channels = channels
        self.otw = otw

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

    def load(self, ch, spec):
        base = build_base_buffer(spec)          # NumPy expansion, once
        self.usrp.set_tx_freq(self.uhd.types.TuneRequest(spec.freq_hz), ch)
        self.usrp.set_tx_gain(spec.gain_db, ch)
        self._chan[ch].load(base, spec.amplitude)

    def clear(self, ch):
        self._chan[ch].mute_idle()

    def set_amplitude(self, ch, a): self._chan[ch].set_amplitude(float(a))
    def set_freq(self, ch, hz):
        self.usrp.set_tx_freq(self.uhd.types.TuneRequest(float(hz)), ch)
    def set_gain(self, ch, db): self.usrp.set_tx_gain(float(db), ch)
    def actual_freq(self, ch): return self.usrp.get_tx_freq(ch)
    def actual_rate(self, ch): return self._chan[ch].rate_hz

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
            self._chan[ch].load(buf, 0.2)

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
    """One TX channel: a UHD streamer plus a thread that loops the current buffer
    into it. Buffer swaps and amplitude changes are done by replacing references
    under a lock, so the hot send loop never allocates and never blocks on DSP."""

    def __init__(self, radio: "UhdRadio", ch: int):
        self.radio = radio
        self.ch = ch
        self.rate_hz = 0.0
        self._lock = threading.Lock()
        self._scaled = None            # complex64 buffer actually streamed (base×amp)
        self._base = None              # unit-amplitude buffer (kept for rescale)
        self._amp = 0.0
        self._streamer = None
        self._spp = 0
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── configuration ─────────────────────────────────────────────────────────
    def configure(self, target_rate_hz: float) -> float:
        """Set this channel's TX rate and (re)build its streamer for that rate.
        Returns the actual rate UHD locked to — what the buffer must be built for."""
        u = self.radio
        u.usrp.set_tx_rate(float(target_rate_hz), self.ch)
        self.rate_hz = float(u.usrp.get_tx_rate(self.ch))
        st_args = u.uhd.usrp.StreamArgs("fc32", u.otw)
        st_args.channels = [self.ch]
        self._streamer = u.usrp.get_tx_stream(st_args)
        self._spp = int(self._streamer.get_max_num_samps())
        return self.rate_hz

    # ── buffer / amplitude ────────────────────────────────────────────────────
    def load(self, base, amplitude: float) -> None:
        np = self.radio.np
        base = np.ascontiguousarray(base, dtype=np.complex64)
        with self._lock:
            self._base = base
            self._amp = float(amplitude)
            self._scaled = (base * self._amp).astype(np.complex64) if self._amp else \
                np.zeros_like(base)
        self.start()   # ensure the thread is running (streams zeros while amp==0)

    def set_amplitude(self, a: float) -> None:
        np = self.radio.np
        with self._lock:
            self._amp = float(a)
            if self._base is None:
                return
            self._scaled = (self._base * self._amp).astype(np.complex64) if self._amp \
                else np.zeros_like(self._base)

    def mute_idle(self) -> None:
        """Release a channel: mute and drop its buffer. The thread keeps running and
        feeds zeros so the DUC stays fed (no device error), ready for the next load."""
        with self._lock:
            self._amp = 0.0
            self._base = None
            self._scaled = None

    # ── streaming thread ──────────────────────────────────────────────────────
    def start(self) -> None:
        if self._streamer is None or self._running.is_set():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run, name=f"x410-ch{self.ch}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._streamer is not None:
            self._send_eob()

    def _run(self) -> None:
        np = self.radio.np
        md = self.radio.uhd.types.TXMetadata()
        md.start_of_burst = True
        md.end_of_burst = False
        md.has_time_spec = False
        spp = self._spp
        zeros = np.zeros((1, spp), dtype=np.complex64)
        idx = 0
        while self._running.is_set():
            with self._lock:
                buf = self._scaled
            if buf is None or buf.size == 0:
                self._streamer.send(zeros, md)
                md.start_of_burst = False
                idx = 0
                continue
            n = buf.size
            if idx >= n:
                idx = 0
            end = min(idx + spp, n)             # variable chunk at the wrap boundary,
            chunk = buf[idx:end]                # so looping never allocates/concats
            self._streamer.send(
                np.ascontiguousarray(chunk.reshape(1, chunk.size)), md)
            md.start_of_burst = False
            idx = 0 if end >= n else end

    def _send_eob(self) -> None:
        np = self.radio.np
        md = self.radio.uhd.types.TXMetadata()
        md.start_of_burst = False
        md.end_of_burst = True
        md.has_time_spec = False
        try:
            self._streamer.send(np.zeros((1, 1), dtype=np.complex64), md)
        except Exception:
            pass


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
            if self._owner[ch] is None:
                self._owner[ch] = owner            # implicit acquire on configure
            actual = self.backend.configure(ch, target_rate_hz)
            self._rate[ch] = actual
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
            gain_db=None) -> None:
        with self._lock:
            self._check(ch, owner)
            if amplitude is not None:
                self.backend.set_amplitude(ch, amplitude); self._amp[ch] = amplitude
            if freq_hz is not None:
                self.backend.set_freq(ch, freq_hz); self._freq[ch] = freq_hz
            if gain_db is not None:
                self.backend.set_gain(ch, gain_db)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {"master_clock_hz": self.master_clock_hz, "channels": [
                {"channel": ch, "owner": self._owner[ch], "signal": self._signal[ch],
                 "amplitude": self._amp[ch], "freq_hz": self._freq[ch],
                 "rate_hz": self._rate[ch]}
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
                      freq_hz=msg.get("freq_hz"), gain_db=msg.get("gain_db"))
                return {"ok": True}
            return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        except (KeyError,) as ex:
            return {"ok": False, "error": f"missing field: {ex}"}
        except (ValueError, PermissionError, FileNotFoundError, TypeError) as ex:
            return {"ok": False, "error": str(ex)}

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

    # build_base_buffer (needs NumPy; skip gracefully if absent)
    try:
        import numpy as np
        tmpd = tempfile.mkdtemp()
        chipf = os.path.join(tmpd, "chips.fc32")
        np.array([1 + 0j, -1 + 0j, 1 + 0j], dtype=np.complex64).tofile(chipf)
        buf = build_base_buffer(ChannelSpec(mode="pcode", freq_hz=1e9, iq_file=chipf, interp=4))
        check(buf.size == 12 and buf[0] == 1 and buf[4] == -1, "pcode np.repeat expands chips")
        primf = os.path.join(tmpd, "prim.fc32")
        np.array([1 + 0j, 1j], dtype=np.complex64).tofile(primf)
        tb = build_base_buffer(ChannelSpec(mode="tiered", freq_hz=1e9, primary_file=primf,
                                           secondary=[1, -1], period_samples=2))
        check(tb.size == 4 and tb[2] == -1 and tb[3] == -1j, "tiered applies ±1 per period")
    except ImportError:
        check(True, "build_base_buffer skipped (no NumPy here)")

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
    srv.stop()

    print("ALL ENGINE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ─────────────────────────────────────────────

def build_script() -> Script:
    clocks = {f"{c/1e6:g} MHz": c for c in STOCK_MASTER_CLOCKS_HZ}
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
        .choice("-OTW-format", "--otw",
                options={"sc16": "16-bit (default, full range)",
                         "sc8": "8-bit (halves the internal stream)"},
                default="sc16", help="Over-the-wire sample format.")
        .number("-Benchmark", "--benchmark", unit="s", min=0.0, max=120.0, default=0.0,
                help="If >0, run an underflow probe for this many seconds (per-channel "
                     "streamers at --bench_rates) and exit.")
        .text("-Benchmark-rates", "--bench_rates", default="61.44,8.192,8.192,8.192",
              help="Comma-separated per-channel rates in MHz for --benchmark.")
    )


def _parse_rates_mhz(s: str) -> List[float]:
    return [float(x) * 1e6 for x in str(s).split(",") if x.strip()]


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    args = build_script().parse()
    master_clock_hz = args.master_clock * 1e6

    if args.benchmark and args.benchmark > 0:
        rates = _parse_rates_mhz(args.bench_rates)
        radio = UhdRadio(args.device_args, master_clock_hz, args.channels, args.otw)
        print(f"[benchmark] {args.channels} ch, master {radio.master_clock_hz/1e6:g} MHz, "
              f"rates {[f'{r/1e6:g}' for r in rates]} MHz, {args.benchmark:g}s…", flush=True)
        result = radio.benchmark(args.benchmark, rates)
        print(json.dumps(result, indent=2))
        u = result.get("underflows", 0)
        print("RESULT:", "clean (no underflows)" if u == 0 else f"{u} underflow marker(s)")
        return 0

    radio = UhdRadio(args.device_args, master_clock_hz, args.channels, args.otw)
    engine = Engine(radio, channels=args.channels)
    server = ControlServer(engine, args.socket)

    radio.start()
    print("── X410 engine ─────────────────────────────────────────────")
    print(f"  device         : {args.device_args}")
    print(f"  master clock   : {radio.master_clock_hz/1e6:g} MHz  ({args.channels} channels, {args.otw})")
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

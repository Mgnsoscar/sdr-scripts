#!/usr/bin/env python3
"""
x410_engine — a persistent, multi-channel GNU Radio playback engine for the
Ettus USRP X410, controlled over a local socket.

Why this exists
───────────────
The X410 has four TX channels, but UHD lets only ONE process own the device at a
time — so you can't launch four independent transmitter scripts, one per channel.
This engine is that single owner: it opens the X410 once with all channels, then
takes per-channel commands over a Unix socket. "Tasks" (start signal X on channel
2 for 5 minutes) become commands, not device-claiming processes, so they can run
and overlap across channels while only the engine ever touches UHD.

The agent launches the engine once; short-lived "channel-task" clients (see
engine_client.py) then drive individual channels. Sequences/timelines are just
scheduled channel commands — the timeline stays data, not hard-coded here.

Channel chain modes (kept deliberately light on the X410's ARM)
───────────────────────────────────────────────────────────────
Each channel is  <source> → amp(multiply_const) → sink[ch]  where the amp block
and the sink connection are PERMANENT (so amplitude/mute/retune are live setters,
no reconfiguration) and only the <source> is rebuilt on load, under lock/unlock:

  • "expanded"  file_source(device-rate IQ) → amp                 (lightest; pure DMA)
  • "tiered"    file_source(primary) ┐                            (L1C / B1C, as before)
                vector_source(2ndary)→repeat(period)→┘ multiply_cc → amp
  • "pcode"     file_source(chip-rate IQ) → repeat(samp/chip) → amp   (long m-seqs, e.g. GLONASS P)

Signal swaps happen at task boundaries (seconds/minutes apart), so the brief
flowgraph lock during a rebuild is acceptable; the other channels resume intact.

Scene sample rate
─────────────────
The engine runs at ONE device sample rate per session (a "scene"), set at launch.
All channel buffers must be built for that rate. Pick it to match the signals in
the scene (don't oversample narrow signals to a needlessly high rate — that only
bloats buffers). Restart the engine to change the scene rate.

Control protocol (JSON object per line, over a Unix stream socket)
──────────────────────────────────────────────────────────────────
  → {"cmd":"acquire","channel":0,"owner":"task-abc"}
  → {"cmd":"load","channel":0,"owner":"task-abc","spec":{...ChannelSpec...}}
  → {"cmd":"set","channel":0,"owner":"task-abc","amplitude":0.9}
  → {"cmd":"set","channel":0,"owner":"task-abc","freq_hz":1.57542e9,"gain_db":40}
  → {"cmd":"release","channel":0,"owner":"task-abc"}
  → {"cmd":"status"}                       → per-channel owner/signal/amp/freq
  → {"cmd":"benchmark","seconds":5}        → underflow count at the scene rate
  ← {"ok":true, ...}   or   {"ok":false,"error":"..."}

CLI
───
    x410_engine.py --samp_rate 20.44 --socket /tmp/x410.sock   # run the engine
    x410_engine.py --benchmark 5 --samp_rate 61.44             # underflow probe, no socket
    x410_engine.py --self-test        # protocol + state machine, no GNU Radio/hardware
    x410_engine.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Keep UHD quiet by default (benchmark re-enables fastpath to count underflows).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script


NUM_CHANNELS = 4
MODES = ("expanded", "tiered", "pcode")


# ── Channel spec (what a channel-task hands the engine to load) ────────────────

@dataclass
class ChannelSpec:
    """A signal to play on one channel. `mode` selects the chain (see module
    docstring). Buffers must already exist on disk at the engine's scene rate."""
    mode: str                       # "expanded" | "tiered" | "pcode"
    freq_hz: float                  # channel carrier
    gain_db: float = 50.0
    amplitude: float = 0.9
    label: str = ""                 # human tag for status, e.g. "gps_l1_ca prn1"
    # expanded / pcode
    iq_file: str = ""               # device-rate IQ (expanded) or chip-rate IQ (pcode)
    interp: int = 1                 # pcode: samples/chip (repeat factor); else 1
    # tiered
    primary_file: str = ""          # device-rate primary-period IQ
    secondary: List[int] = field(default_factory=list)   # ±1 overlay, one per primary period
    period_samples: int = 0         # samples per primary period (repeat factor for the overlay)

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


# ── Radio backends ─────────────────────────────────────────────────────────────

class RadioBackend:
    """Interface the Engine drives. UhdRadio is the real one; MockRadio backs
    --self-test so the protocol/state machine run with no GNU Radio or hardware."""
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def load(self, ch: int, spec: ChannelSpec) -> None: ...
    def clear(self, ch: int) -> None: ...
    def set_amplitude(self, ch: int, a: float) -> None: ...
    def set_freq(self, ch: int, hz: float) -> None: ...
    def set_gain(self, ch: int, db: float) -> None: ...
    def actual_freq(self, ch: int) -> float: ...
    def benchmark(self, seconds: float) -> Dict[str, Any]: ...


class MockRadio(RadioBackend):
    """Records calls; validates the files it's told to load exist. No DSP."""
    def __init__(self, samp_rate_hz: float, channels: int):
        self.samp_rate_hz = samp_rate_hz
        self.channels = channels
        self.calls: List[str] = []
        self._freq = [0.0] * channels

    def start(self): self.calls.append("start")
    def stop(self): self.calls.append("stop")

    def load(self, ch, spec):
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
    def benchmark(self, seconds):
        return {"seconds": seconds, "underflows": 0, "note": "mock backend"}


class UhdRadio(RadioBackend):
    """Real X410 backend: one gr.top_block owning a 4-channel uhd.usrp_sink.

    Per channel a permanent multiply_const (amp) feeds the sink; only the source
    ahead of it is rebuilt on load(), under top_block lock/unlock. GNU Radio is
    imported here so the module loads for --self-test / --describe-params."""

    def __init__(self, device_args: str, samp_rate_hz: float, channels: int,
                 otw: str):
        from gnuradio import gr, blocks, uhd
        self.gr, self.blocks, self.uhd = gr, blocks, uhd
        self.samp_rate_hz = samp_rate_hz
        self.channels = channels

        self.tb = gr.top_block("x410-engine")
        args = device_args or "type=x4xx"
        self.usrp = uhd.usrp_sink(
            args, uhd.stream_args(cpu_format="fc32", otw_format=otw,
                                  channels=list(range(channels))))
        self.usrp.set_samp_rate(samp_rate_hz)

        # Permanent per-channel amp → sink[ch]; sources are attached on load.
        self._amp = []
        self._src_edges: List[List[tuple]] = [[] for _ in range(channels)]
        for ch in range(channels):
            amp = blocks.multiply_const_cc(0.0)         # start muted
            self.tb.connect(amp, (self.usrp, ch))
            self._amp.append(amp)
            self._attach_idle(ch)

    # ── chain construction ────────────────────────────────────────────────────
    def _attach_idle(self, ch: int) -> None:
        """Silent source so every channel always has something feeding its amp."""
        z = self.blocks.null_source(self.gr.sizeof_gr_complex)
        self.tb.connect(z, self._amp[ch])
        self._src_edges[ch] = [(z, self._amp[ch])]

    def _teardown_src(self, ch: int) -> None:
        for edge in self._src_edges[ch]:
            self.tb.disconnect(*edge)
        self._src_edges[ch] = []

    def _build_src(self, ch: int, spec: ChannelSpec):
        """Build the source chain for `spec` and connect its head to amp[ch].
        Records every edge so it can be torn down on the next load."""
        b, gr = self.blocks, self.gr
        edges: List[tuple] = []
        if spec.mode == "expanded":
            src = b.file_source(gr.sizeof_gr_complex, spec.iq_file, repeat=True)
            edges.append((src, self._amp[ch]))
        elif spec.mode == "pcode":
            src = b.file_source(gr.sizeof_gr_complex, spec.iq_file, repeat=True)
            rep = b.repeat(gr.sizeof_gr_complex, int(spec.interp))
            edges += [(src, rep), (rep, self._amp[ch])]
        elif spec.mode == "tiered":
            prim = b.file_source(gr.sizeof_gr_complex, spec.primary_file, repeat=True)
            sec = b.vector_source_c([complex(s, 0) for s in spec.secondary], repeat=True)
            rep = b.repeat(gr.sizeof_gr_complex, int(spec.period_samples))
            mult = b.multiply_cc()
            edges += [(sec, rep), (prim, (mult, 0)), (rep, (mult, 1)),
                      (mult, self._amp[ch])]
        for e in edges:
            self.tb.connect(*e)
        self._src_edges[ch] = edges

    # ── backend interface ─────────────────────────────────────────────────────
    def start(self): self.tb.start()

    def stop(self):
        self.tb.stop()
        self.tb.wait()

    def load(self, ch, spec):
        for f in spec.files():
            if not os.path.exists(f):
                raise FileNotFoundError(f"IQ file not found: {f}")
        self.tb.lock()
        try:
            self._teardown_src(ch)
            self._build_src(ch, spec)
            self.usrp.set_center_freq(self.uhd.tune_request(spec.freq_hz), ch)
            self.usrp.set_gain(spec.gain_db, ch)
            self._amp[ch].set_k(spec.amplitude)
        finally:
            self.tb.unlock()

    def clear(self, ch):
        self.tb.lock()
        try:
            self._amp[ch].set_k(0.0)
            self._teardown_src(ch)
            self._attach_idle(ch)
        finally:
            self.tb.unlock()

    def set_amplitude(self, ch, a): self._amp[ch].set_k(float(a))
    def set_freq(self, ch, hz): self.usrp.set_center_freq(self.uhd.tune_request(float(hz)), ch)
    def set_gain(self, ch, db): self.usrp.set_gain(float(db), ch)
    def actual_freq(self, ch): return self.usrp.get_center_freq(ch)

    def benchmark(self, seconds: float) -> Dict[str, Any]:
        """Stream a `repeat`+`multiply`-exercising chain on all channels at the
        scene rate for `seconds`, counting UHD underflow ('U') markers by
        capturing stderr with the fastpath re-enabled. Real, if slightly hacky."""
        import tempfile
        b, gr = self.blocks, self.gr
        # Build a throwaway top_block that mirrors the heaviest chain (pcode:
        # file/vector → repeat → multiply_const) on every channel.
        tb = gr.top_block("x410-bench")
        usrp = self.uhd.usrp_sink(
            "type=x4xx", self.uhd.stream_args(cpu_format="fc32", otw_format="sc16",
                                              channels=list(range(self.channels))))
        usrp.set_samp_rate(self.samp_rate_hz)
        for ch in range(self.channels):
            vs = b.vector_source_c([complex((i % 7) - 3, 0) for i in range(10230)],
                                   repeat=True)
            rep = b.repeat(gr.sizeof_gr_complex, 6)
            amp = b.multiply_const_cc(0.2)
            tb.connect(vs, rep, amp, (usrp, ch))
            usrp.set_center_freq(self.uhd.tune_request(1.5e9), ch)

        # Capture stderr (UHD prints 'U' per underflow when fastpath is enabled).
        os.environ["UHD_LOG_FASTPATH_DISABLE"] = "0"
        tmp = tempfile.TemporaryFile(mode="w+")
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        try:
            tb.start()
            time.sleep(max(0.5, seconds))
            tb.stop(); tb.wait()
        finally:
            os.dup2(saved, 2); os.close(saved)
        tmp.seek(0)
        text = tmp.read()
        tmp.close()
        underflows = text.count("U")
        return {"seconds": seconds, "channels": self.channels,
                "samp_rate_hz": self.samp_rate_hz, "underflows": underflows}


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

    def load(self, ch: int, owner: str, spec: ChannelSpec) -> None:
        spec.validate()
        with self._lock:
            self._check(ch, owner)
            if self._owner[ch] is None:
                self._owner[ch] = owner            # implicit acquire on load
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
            return {"channels": [
                {"channel": ch, "owner": self._owner[ch], "signal": self._signal[ch],
                 "amplitude": self._amp[ch], "freq_hz": self._freq[ch]}
                for ch in range(self.channels)]}

    def benchmark(self, seconds: float) -> Dict[str, Any]:
        return self.backend.benchmark(seconds)


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
            if cmd == "status":
                return {"ok": True, **e.status()}
            if cmd == "benchmark":
                return {"ok": True, **e.benchmark(float(msg.get("seconds", 5)))}
            ch = int(msg["channel"])
            owner = msg.get("owner", "")
            if cmd == "acquire":
                e.acquire(ch, owner); return {"ok": True}
            if cmd == "release":
                e.release(ch, owner); return {"ok": True}
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


# ── Self-test (protocol + state machine, MockRadio, no GNU Radio) ──────────────

def _self_test() -> int:
    import tempfile
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'}· {msg}")
        ok = ok and cond

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

    # End-to-end over a real Unix socket with the mock backend.
    tmp = tempfile.mkdtemp()
    iqf = os.path.join(tmp, "sig.fc32")
    open(iqf, "wb").write(b"\x00" * 64)          # a file that "exists"
    sock = os.path.join(tmp, "eng.sock")
    eng = Engine(MockRadio(20.44e6, NUM_CHANNELS))
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

    check(call({"cmd": "acquire", "channel": 0, "owner": "A"})["ok"], "acquire ch0 by A")
    check(not call({"cmd": "acquire", "channel": 0, "owner": "B"})["ok"],
          "second owner B rejected on ch0")
    r = call({"cmd": "load", "channel": 0, "owner": "A",
              "spec": {"mode": "expanded", "freq_hz": 1.57542e9, "iq_file": iqf,
                       "amplitude": 0.9, "label": "gps_l1_ca prn1"}})
    check(r["ok"], "load expanded on ch0")
    check(not call({"cmd": "set", "channel": 0, "owner": "B", "amplitude": 0.5})["ok"],
          "set by wrong owner rejected")
    check(call({"cmd": "set", "channel": 0, "owner": "A", "amplitude": 0.0})["ok"],
          "mute by owner A")
    r = call({"cmd": "load", "channel": 1, "owner": "C",
              "spec": {"mode": "pcode", "freq_hz": 1.602e9, "iq_file": iqf,
                       "interp": 4, "label": "glonass_p"}})
    check(r["ok"], "load pcode on ch1 (implicit acquire by C)")
    st = call({"cmd": "status"})
    owners = {c["channel"]: c["owner"] for c in st["channels"]}
    check(owners[0] == "A" and owners[1] == "C" and owners[2] is None,
          "status reflects owners")
    check(call({"cmd": "release", "channel": 0, "owner": "A"})["ok"], "release ch0")
    check(call({"cmd": "status"})["channels"][0]["owner"] is None, "ch0 free after release")
    check(not call({"cmd": "load", "channel": 2, "owner": "A",
                    "spec": {"mode": "expanded", "freq_hz": 1e9,
                             "iq_file": "/no/such/file"}})["ok"],
          "missing IQ file rejected")
    srv.stop()

    print("ALL ENGINE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ─────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Persistent multi-channel GNU Radio playback engine for the USRP "
               "X410. Owns the device; channel-tasks drive channels over a socket.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=0.2, max=250.0,
                default=20.44,
                help="Scene sample rate (device rate for all channels). Match it to "
                     "the signals in the scene. Fixed for the engine's lifetime.")
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
                help="If >0, run an underflow probe for this many seconds (all "
                     "channels, repeat+multiply chain) and exit — no socket.")
    )


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    args = build_script().parse()
    samp_rate_hz = args.samp_rate * 1e6

    if args.benchmark and args.benchmark > 0:
        radio = UhdRadio(args.device_args, samp_rate_hz, args.channels, args.otw)
        print(f"[benchmark] {args.channels} ch × {args.samp_rate:g} MHz, "
              f"{args.benchmark:g}s, repeat+multiply chain…", flush=True)
        result = radio.benchmark(args.benchmark)
        print(json.dumps(result, indent=2))
        u = result.get("underflows", 0)
        print("RESULT:", "clean (no underflows)" if u == 0 else f"{u} underflow marker(s)")
        return 0

    radio = UhdRadio(args.device_args, samp_rate_hz, args.channels, args.otw)
    engine = Engine(radio, channels=args.channels)
    server = ControlServer(engine, args.socket)

    radio.start()
    print("── X410 engine ─────────────────────────────────────────────")
    print(f"  device         : {args.device_args}")
    print(f"  scene rate     : {args.samp_rate:g} MHz  ({args.channels} channels, {args.otw})")
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

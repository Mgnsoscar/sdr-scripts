#!/usr/bin/env python3
"""
engine_client — thin client for x410_engine.

A channel-task uses EngineClient to drive one channel of the running engine:
configure its rate, load a signal, tune/mute/unmute, and release it on exit. It's
a tiny JSON-over-Unix-socket wrapper — no UHD, no device access — so it stays
lightweight and importable anywhere (including the system python the tasks run under).

The two-phase rate handshake
────────────────────────────
A channel-task must build its IQ at the exact rate the channel will stream. It
doesn't guess — it negotiates:

    with EngineClient("/tmp/x410_engine.sock") as eng:
        eng.acquire(2, owner="task-42")
        actual = eng.configure(2, owner="task-42", target_rate_hz=8.192e6)
        iq_path = build_my_iq(actual)             # ← build at the rate UHD gave
        eng.load(2, owner="task-42", spec={
            "mode": "expanded", "freq_hz": 1.57542e9,
            "iq_file": iq_path, "amplitude": 0.0,   # muted; raise at on-air
            "label": "gps_l1_ca prn1"})
        ...                                        # pre-roll, then at on-air:
        eng.set(2, owner="task-42", amplitude=0.9, gain_db=45)
        ...                                        # run for the task's duration
        eng.release(2, owner="task-42")            # (channel_session does this for you)

As a CLI (handy for manual poking / debugging)
──────────────────────────────────────────────
    engine_client.py --socket /tmp/x410_engine.sock hello
    engine_client.py --socket /tmp/x410_engine.sock status
    engine_client.py --socket /tmp/x410_engine.sock configure --channel 0 --target_rate_hz 8.192e6 --owner cli
    engine_client.py --socket /tmp/x410_engine.sock set --channel 0 --amplitude 0
    engine_client.py --socket /tmp/x410_engine.sock benchmark --seconds 5 --rates 61.44,8.192,8.192,8.192
"""
from __future__ import annotations

import argparse
import json
import socket
from contextlib import contextmanager
from typing import Any, Dict, List, Optional


class EngineError(RuntimeError):
    """The engine replied {"ok": false, "error": ...}."""


def nearest_achievable_rate(master_clock_hz: float, target_rate_hz: float) -> float:
    """Pure-arithmetic prediction of the rate the device will produce for
    `target_rate_hz` (master / N, nearest N). Handy for sizing a buffer before the
    round-trip, but `configure()` returns the authoritative value — always build
    for that."""
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    n = max(1, round(master_clock_hz / target_rate_hz))
    return master_clock_hz / n


class EngineClient:
    """One connection to the engine's Unix socket."""

    def __init__(self, socket_path: str, timeout: float = 5.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._f = None

    # ── connection ────────────────────────────────────────────────────────────
    def connect(self) -> "EngineClient":
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.socket_path)
        self._sock = s
        self._f = s.makefile("rwb")
        return self

    def close(self) -> None:
        try:
            if self._f:
                self._f.close()
            if self._sock:
                self._sock.close()
        finally:
            self._f = self._sock = None

    def __enter__(self): return self.connect()
    def __exit__(self, *exc): self.close()

    # ── raw request ───────────────────────────────────────────────────────────
    def request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        if self._f is None:
            self.connect()
        self._f.write((json.dumps(msg) + "\n").encode())
        self._f.flush()
        line = self._f.readline()
        if not line:
            raise EngineError("engine closed the connection")
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise EngineError(reply.get("error", "unknown engine error"))
        return reply

    # ── typed commands ────────────────────────────────────────────────────────
    def hello(self) -> Dict[str, Any]:
        return self.request({"cmd": "hello"})

    def status(self) -> Dict[str, Any]:
        return self.request({"cmd": "status"})

    def benchmark(self, seconds: float = 5.0,
                  rates_hz: Optional[List[float]] = None) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"cmd": "benchmark", "seconds": seconds}
        if rates_hz:
            msg["rates_hz"] = rates_hz
        return self.request(msg)

    def acquire(self, channel: int, owner: str) -> None:
        self.request({"cmd": "acquire", "channel": channel, "owner": owner})

    def release(self, channel: int, owner: str) -> None:
        self.request({"cmd": "release", "channel": channel, "owner": owner})

    def configure(self, channel: int, owner: str, target_rate_hz: float) -> float:
        """Set the channel's TX rate; returns the ACTUAL rate UHD locked to. Build
        the channel's IQ buffer for exactly this returned rate."""
        r = self.request({"cmd": "configure", "channel": channel, "owner": owner,
                          "target_rate_hz": target_rate_hz})
        return float(r["actual_rate_hz"])

    def load(self, channel: int, owner: str, spec: Dict[str, Any]) -> None:
        self.request({"cmd": "load", "channel": channel, "owner": owner, "spec": spec})

    def set(self, channel: int, owner: str, *, amplitude: Optional[float] = None,
            freq_hz: Optional[float] = None, gain_db: Optional[float] = None) -> None:
        msg: Dict[str, Any] = {"cmd": "set", "channel": channel, "owner": owner}
        if amplitude is not None:
            msg["amplitude"] = amplitude
        if freq_hz is not None:
            msg["freq_hz"] = freq_hz
        if gain_db is not None:
            msg["gain_db"] = gain_db
        self.request(msg)


@contextmanager
def channel_session(socket_path: str, channel: int, owner: str,
                    target_rate_hz: float, build_spec):
    """The exact lifecycle a channel-task wants, with the rate handshake built in:
    acquire → configure → (task builds IQ at the negotiated rate) → load muted →
    guaranteed release on exit.

    `build_spec(actual_rate_hz) -> spec_dict` is called with the rate UHD locked
    to; it builds the IQ (e.g. into /dev/shm) and returns the load spec. The spec's
    amplitude should be 0 — raise it at the on-air moment via the yielded client:

        def build(rate):
            path = synth_iq(rate)                 # build at the real rate
            return {"mode": "expanded", "freq_hz": 1.57542e9,
                    "iq_file": path, "amplitude": 0.0, "label": "gps_l1_ca prn1"}

        with channel_session(sock, 2, "task-42", 8.192e6, build) as eng:
            eng.set(2, "task-42", amplitude=0.9, gain_db=45)   # on-air
            time.sleep(duration)                  # or block until SIGTERM
        # channel muted + released here, even on exception / SIGTERM
    """
    client = EngineClient(socket_path).connect()
    try:
        client.acquire(channel, owner)
        actual = client.configure(channel, owner, target_rate_hz)
        spec = build_spec(actual)
        client.load(channel, owner, spec)
        yield client
    finally:
        try:
            client.release(channel, owner)
        except Exception:
            pass
        client.close()


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main() -> int:
    p = argparse.ArgumentParser(description="Talk to a running x410_engine.")
    p.add_argument("--socket", default="/tmp/x410_engine.sock")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hello")
    sub.add_parser("status")
    b = sub.add_parser("benchmark")
    b.add_argument("--seconds", type=float, default=5.0)
    b.add_argument("--rates", default="", help="comma-separated per-channel rates in MHz")
    for name in ("acquire", "release"):
        s = sub.add_parser(name)
        s.add_argument("--channel", type=int, required=True)
        s.add_argument("--owner", required=True)
    c = sub.add_parser("configure")
    c.add_argument("--channel", type=int, required=True)
    c.add_argument("--owner", required=True)
    c.add_argument("--target_rate_hz", type=float, required=True)
    s = sub.add_parser("set")
    s.add_argument("--channel", type=int, required=True)
    s.add_argument("--owner", default="cli")
    s.add_argument("--amplitude", type=float)
    s.add_argument("--freq_hz", type=float)
    s.add_argument("--gain_db", type=float)
    args = p.parse_args()

    with EngineClient(args.socket) as eng:
        if args.cmd == "hello":
            print(json.dumps(eng.hello(), indent=2))
        elif args.cmd == "status":
            print(json.dumps(eng.status(), indent=2))
        elif args.cmd == "benchmark":
            rates = [float(x) * 1e6 for x in args.rates.split(",") if x.strip()] or None
            print(json.dumps(eng.benchmark(args.seconds, rates), indent=2))
        elif args.cmd == "acquire":
            eng.acquire(args.channel, args.owner); print("ok")
        elif args.cmd == "release":
            eng.release(args.channel, args.owner); print("ok")
        elif args.cmd == "configure":
            print(f"actual_rate_hz = {eng.configure(args.channel, args.owner, args.target_rate_hz):.0f}")
        elif args.cmd == "set":
            eng.set(args.channel, args.owner, amplitude=args.amplitude,
                    freq_hz=args.freq_hz, gain_db=args.gain_db); print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

#!/usr/bin/env python3
"""
engine_client — thin client for x410_engine.

A channel-task uses EngineClient to drive one channel of the running engine:
acquire it, load a signal, tune/mute/unmute, and release it on exit. It's a tiny
JSON-over-Unix-socket wrapper — no GNU Radio, no device access — so it stays
lightweight and importable anywhere.

As a library
────────────
    from engine_client import EngineClient
    with EngineClient("/tmp/x410_engine.sock") as eng:
        eng.acquire(2, owner="task-42")
        eng.load(2, owner="task-42", spec={
            "mode": "expanded", "freq_hz": 1.57542e9,
            "iq_file": "/dev/shm/gps_l1_ca_prn1.fc32", "amplitude": 0.9,
            "label": "gps_l1_ca prn1"})
        ...                                  # run for the task's duration
        eng.release(2, owner="task-42")      # (a ChannelSession does this for you)

As a CLI (handy for manual poking / debugging)
──────────────────────────────────────────────
    engine_client.py --socket /tmp/x410_engine.sock status
    engine_client.py --socket /tmp/x410_engine.sock set --channel 0 --amplitude 0
    engine_client.py --socket /tmp/x410_engine.sock benchmark --seconds 5
"""
from __future__ import annotations

import argparse
import json
import socket
from contextlib import contextmanager
from typing import Any, Dict, Optional


class EngineError(RuntimeError):
    """The engine replied {"ok": false, "error": ...}."""


class EngineClient:
    """One short-lived TCP-less connection to the engine's Unix socket."""

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
    def status(self) -> Dict[str, Any]:
        return self.request({"cmd": "status"})

    def benchmark(self, seconds: float = 5.0) -> Dict[str, Any]:
        return self.request({"cmd": "benchmark", "seconds": seconds})

    def acquire(self, channel: int, owner: str) -> None:
        self.request({"cmd": "acquire", "channel": channel, "owner": owner})

    def release(self, channel: int, owner: str) -> None:
        self.request({"cmd": "release", "channel": channel, "owner": owner})

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
def channel_session(socket_path: str, channel: int, owner: str, spec: Dict[str, Any]):
    """Acquire a channel, load `spec`, and guarantee release on exit — the exact
    lifecycle a channel-task wants. Yields the connected EngineClient so the task
    can retune / adjust amplitude while it runs.

        with channel_session(sock, 2, "task-42", spec) as eng:
            time.sleep(duration)            # or block until the agent kills us
        # channel muted + released here, even on exception / SIGTERM
    """
    client = EngineClient(socket_path).connect()
    try:
        client.acquire(channel, owner)
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
    sub.add_parser("status")
    b = sub.add_parser("benchmark"); b.add_argument("--seconds", type=float, default=5.0)
    for name in ("acquire", "release"):
        s = sub.add_parser(name); s.add_argument("--channel", type=int, required=True)
        s.add_argument("--owner", required=True)
    s = sub.add_parser("set")
    s.add_argument("--channel", type=int, required=True)
    s.add_argument("--owner", default="cli")
    s.add_argument("--amplitude", type=float)
    s.add_argument("--freq_hz", type=float)
    s.add_argument("--gain_db", type=float)
    args = p.parse_args()

    with EngineClient(args.socket) as eng:
        if args.cmd == "status":
            print(json.dumps(eng.status(), indent=2))
        elif args.cmd == "benchmark":
            print(json.dumps(eng.benchmark(args.seconds), indent=2))
        elif args.cmd == "acquire":
            eng.acquire(args.channel, args.owner); print("ok")
        elif args.cmd == "release":
            eng.release(args.channel, args.owner); print("ok")
        elif args.cmd == "set":
            eng.set(args.channel, args.owner, amplitude=args.amplitude,
                    freq_hz=args.freq_hz, gain_db=args.gain_db); print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

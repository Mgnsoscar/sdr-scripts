"""
paramkit.live — retune a running script's parameters over a local control socket.

A script marks parameters ``live=True`` (see Script.number/integer/…), then in
its main loop drains and applies pending changes::

    args = script.parse()
    ctrl = script.live_control(args)          # opens the control socket, if any

    sdr.set_freq(args.freq); sdr.set_gain(args.gain)
    while running:
        for change in ctrl.drain():           # applied on THIS thread — device-safe
            if change.name == "freq":
                ctrl.report("freq", sdr.set_freq(change.value))   # report the value
            elif change.name == "gain":                            # the device took
                ctrl.report("gain", sdr.set_gain(change.value))
        process(sdr.read())

The host (the sdr-agent) sets ``SDR_CTRL_SOCK`` to a per-run Unix-socket path
and connects to it to push updates. The wire protocol is newline-delimited JSON:

    → {"op": "set", "values": {"freq": 1.01e8, "gain": 40}, "wait": 1.0}
    ← {"ok": true, "accepted": {...}, "rejected": {...}, "applied": {...},
       "pending": ["gain"]}
    → {"op": "get"}
    ← {"ok": true, "current": {...}, "applied": {...}, "live": [...]}

Design notes
------------
* **Thread-safe, device-safe.** The socket thread only validates incoming values
  and records them; the script's own loop is the only place that touches the
  device (via drain()). Nothing here calls into user code off-thread.
* **Authoritative values.** ``set`` blocks (up to ``wait`` seconds) until the
  script has drained and reported each accepted change, so the caller learns the
  value the device actually took (e.g. a gain quantised to the nearest step).
  Anything not reported in time comes back in ``pending`` with the requested
  value; the caller can poll ``get`` for the settled value.
* **No socket ⇒ no-op.** Run standalone (no ``SDR_CTRL_SOCK``, no live params)
  and drain() simply returns nothing, so the same script works on the CLI.
"""
from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from .params import CHOICE, FLAG, INTEGER, NUMBER, Param

CTRL_SOCK_ENV = "SDR_CTRL_SOCK"


class Change(NamedTuple):
    """One pending parameter change handed to the script by drain()."""
    name: str
    value: Any


class LiveControl:
    """Runtime control channel for a script's live parameters. Create via
    ``Script.live_control(args)`` rather than directly."""

    def __init__(self, params: Sequence[Param], args: Any,
                 socket_path: Optional[str] = None):
        self._live: Dict[str, Param] = {p.name: p for p in params if getattr(p, "live", False)}
        self._cond = threading.Condition(threading.RLock())
        self._current: Dict[str, Any] = {}     # last accepted (validated) value
        self._applied: Dict[str, Any] = {}     # last value the script reported back
        self._applied_stamp: Dict[str, int] = {}   # bumped on each report()
        self._pending: List[Change] = []
        self._closed = False
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._path: Optional[str] = None

        for name in self._live:
            v = getattr(args, name, None)
            self._current[name] = v
            self._applied[name] = v
            self._applied_stamp[name] = 0

        path = socket_path or os.environ.get(CTRL_SOCK_ENV)
        if path and self._live:
            self._serve(path)

    # ── Script-facing API (called from the main loop) ─────────────────────────

    def drain(self) -> List[Change]:
        """Return the parameter changes received since the last drain, in order,
        and clear them. Call this once per loop iteration and apply each change to
        the device on this thread. Returns [] when nothing changed / no channel."""
        with self._cond:
            changes, self._pending = self._pending, []
        return changes

    def report(self, name: str, value: Any) -> None:
        """Record the value the device actually took for a parameter. Call this
        right after applying a drained change so a waiting ``set`` learns the real
        (possibly quantised) value. Optional but recommended."""
        with self._cond:
            self._applied[name] = value
            self._applied_stamp[name] = self._applied_stamp.get(name, 0) + 1
            self._cond.notify_all()

    def value(self, name: str) -> Any:
        """The latest accepted value for a live parameter (thread-safe)."""
        with self._cond:
            return self._current.get(name)

    def is_live(self, name: str) -> bool:
        return name in self._live

    @property
    def live_names(self) -> List[str]:
        return list(self._live)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass

    def __enter__(self) -> "LiveControl":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Socket server ─────────────────────────────────────────────────────────

    def _serve(self, path: str) -> None:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(path)
        s.listen(8)
        s.settimeout(0.5)   # so the accept loop can notice close()
        self._sock = s
        self._path = path
        self._thread = threading.Thread(
            target=self._accept_loop, name="paramkit-live", daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(10.0)
        try:
            f = conn.makefile("rwb")
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    req = json.loads(raw)
                except ValueError:
                    self._send(f, {"ok": False, "error": "invalid JSON"})
                    continue
                self._send(f, self._dispatch(req if isinstance(req, dict) else {}))
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    @staticmethod
    def _send(f, obj: Dict[str, Any]) -> None:
        f.write((json.dumps(obj) + "\n").encode("utf-8"))
        f.flush()

    def _dispatch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        op = req.get("op")
        if op == "get":
            with self._cond:
                return {"ok": True, "current": dict(self._current),
                        "applied": dict(self._applied), "live": list(self._live)}
        if op == "set":
            values = req.get("values")
            wait = req.get("wait", 1.0)
            return self._do_set(values if isinstance(values, dict) else {}, wait)
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _do_set(self, values: Dict[str, Any], wait: Any) -> Dict[str, Any]:
        accepted: Dict[str, Any] = {}
        rejected: Dict[str, str] = {}
        for name, raw in values.items():
            p = self._live.get(name)
            if p is None:
                rejected[name] = "not a live parameter"
                continue
            try:
                accepted[name] = _coerce(p, raw)
            except ValueError as exc:
                rejected[name] = str(exc)

        with self._cond:
            start_stamp = {n: self._applied_stamp.get(n, 0) for n in accepted}
            for name, val in accepted.items():
                self._current[name] = val
                self._pending.append(Change(name, val))
            self._cond.notify_all()

        try:
            wait_s = float(wait)
        except (TypeError, ValueError):
            wait_s = 0.0

        applied: Dict[str, Any] = {}
        pending: List[str] = []
        if accepted:
            deadline = time.monotonic() + max(0.0, wait_s)
            with self._cond:
                for name in accepted:
                    while (self._applied_stamp.get(name, 0) <= start_stamp[name]
                           and time.monotonic() < deadline):
                        self._cond.wait(timeout=max(0.0, deadline - time.monotonic()))
                    if self._applied_stamp.get(name, 0) > start_stamp[name]:
                        applied[name] = self._applied[name]
                    else:
                        pending.append(name)
                        applied[name] = accepted[name]   # best-known (the request)

        return {"ok": not rejected, "accepted": accepted, "rejected": rejected,
                "applied": applied, "pending": pending}


# ── Validation / coercion ─────────────────────────────────────────────────────

def _coerce(p: Param, raw: Any) -> Any:
    """Validate + coerce an incoming JSON value against a Param's schema. Raises
    ValueError with a human message on anything out of range / wrong type."""
    if p.kind == FLAG:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)

    if p.kind in (NUMBER, INTEGER):
        val: Any = None
        if isinstance(raw, str):
            for pr in p.presets:                       # accept a preset key/label
                if raw == pr.key or raw.lower() == pr.label.lower():
                    val = pr.value
                    break
            if val is None:
                try:
                    val = float(raw)
                except ValueError:
                    raise ValueError(f"'{raw}' is not a number")
        elif isinstance(raw, bool):
            raise ValueError("expected a number, got a boolean")
        elif isinstance(raw, (int, float)):
            val = raw
        else:
            raise ValueError("expected a number")
        val = int(val) if p.kind == INTEGER else float(val)
        unit = f" {p.unit}" if p.unit else ""
        if p.min is not None and val < p.min:
            raise ValueError(f"{val}{unit} is below the minimum {p.min}{unit}")
        if p.max is not None and val > p.max:
            raise ValueError(f"{val}{unit} is above the maximum {p.max}{unit}")
        return val

    if p.kind == CHOICE:
        s = str(raw)
        if p.choices and s not in p.choices:
            raise ValueError(f"'{s}' is not one of {p.choices}")
        return s

    return str(raw)   # TEXT

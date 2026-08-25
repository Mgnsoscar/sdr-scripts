#!/usr/bin/env python3
"""
channel_task — shared lifecycle for X410 engine channel-tasks.

Every GNSS channel-task does the same dance: connect to the engine, acquire a
channel, negotiate its rate, build IQ at that rate, load it muted, forward live
amplitude/gain/freq changes, and release on stop. This module factors that out so
each signal script only supplies its parameter schema and a `build` callback.

    from channel_task import run_channel, write_shm

    def build(args, rate_hz):
        iq = my_synthesis(args.prn, rate_hz)        # build at the negotiated rate
        path = write_shm(iq, "gps_l2c")
        spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
                "amplitude": args.amplitude, "iq_file": path, "label": f"l2c prn{args.prn}"}
        return spec, [path], [f"PRN {args.prn}"]     # spec, files-to-delete, banner lines

    def main():
        script = build_script()
        args = script.parse()
        return run_channel(script, args, build, title="GPS L2C")

The engine copies IQ into RAM at load, so the /dev/shm files are deleted right
after the load returns.

Requires the task's parameter schema to include: --channel, --samp_rate, --gain,
--amplitude (live), --freq (live), --engine_socket, --owner. Live freq/gain/
amplitude are forwarded to the engine automatically; a task needing extra live
behaviour (e.g. shape reloads) can pass an `on_live` hook.
"""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
from typing import Callable, List, Optional

from engine_client import EngineClient, EngineError


def write_shm(iq, prefix: str) -> str:
    """Write a complex64 buffer to a unique /dev/shm file (tempdir fallback)."""
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".fc32", dir=shm)
    os.close(fd)
    iq.tofile(path)
    return path


def connect_engine(socket_path: str, attempts: int = 20) -> EngineClient:
    """Connect to the engine, retrying briefly (it may still be coming up when the
    agent launches a channel-task alongside it)."""
    last = None
    for _ in range(attempts):
        try:
            return EngineClient(socket_path).connect()
        except OSError as exc:
            last = exc
            time.sleep(0.25)
    raise SystemExit(f"could not reach engine at {socket_path}: {last}")


def run_channel(script, args, build: Callable, *, title: str = "channel-task",
                on_live: Optional[Callable] = None) -> int:
    """Drive one engine channel through the standard lifecycle.

    build(args, actual_rate_hz) -> (spec_dict, cleanup_paths, info_lines):
        spec_dict     the load spec (mode-specific: expanded/composite/pcode).
        cleanup_paths /dev/shm files to unlink after load (engine copies to RAM).
        info_lines    extra 'label : value' strings for the startup banner.

    on_live(change, ctx) -> bool (optional): handle a live change the standard
        forwarder doesn't (return True if handled). ctx exposes eng, ch, owner,
        ctrl, args, and rebuild(spec) for tasks that reload.
    """
    ch = args.channel
    owner = args.owner or f"ch{ch}-{os.getpid()}"

    eng = connect_engine(args.engine_socket)
    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)

        spec, cleanup, info = build(args, actual_rate)

        print(f"── {title} ".ljust(60, "─"))
        print(f"  engine channel : {ch}   owner {owner}")
        print(f"  carrier        : {args.freq/1e6:.3f} MHz")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        for line in info:
            print(f"  {line}")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        print("─" * 60)
        sys.stdout.flush()

        try:
            eng.load(ch, owner, spec)
        finally:
            for p in cleanup:
                try:
                    os.unlink(p)     # engine copied the IQ into RAM at load
                except OSError:
                    pass

        ctrl = script.live_control(args)

        class _Ctx:
            pass
        ctx = _Ctx()
        ctx.eng, ctx.ch, ctx.owner, ctx.ctrl, ctx.args = eng, ch, owner, ctrl, args

        def _forward(change) -> None:
            name, value = change.name, change.value
            if name == "amplitude":
                eng.set(ch, owner, amplitude=value); ctrl.report("amplitude", value)
            elif name == "gain":
                eng.set(ch, owner, gain_db=value); ctrl.report("gain", value)
            elif name == "freq":
                eng.set(ch, owner, freq_hz=value); ctrl.report("freq", value)
            elif on_live is not None:
                on_live(change, ctx)

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        while not stop.is_set():
            for change in ctrl.drain():
                try:
                    _forward(change)
                except EngineError as exc:
                    print(f"[warn] live {change.name}={change.value} rejected: {exc}",
                          flush=True)
            time.sleep(0.1)
        ctrl.close()
    finally:
        # Best-effort cleanup: the connection may already be gone (the engine
        # crashed, or the socket dropped), so tolerate any error here — a failed
        # release must never mask the real exit or raise a secondary
        # BrokenPipeError over the original traceback.
        try:
            eng.release(ch, owner)
        except Exception:
            pass
        try:
            eng.close()
        except Exception:
            pass
    return 0

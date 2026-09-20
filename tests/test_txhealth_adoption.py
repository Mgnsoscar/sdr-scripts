"""
RF-fault Phase 1 (§5.1): the CLEAN-set RPi transmit scripts adopt paramkit.txhealth.watch_flowgraph
(a silent GR halt → a non-zero exit the agent sees), while the repeat=False FIFO stagers are
EXCLUDED — their normal end-of-stream returns from tb.wait() with stop unset and would read as a
false fault; they rely on the agent's Layer-2 log-scan watchdog instead.

Review fix #7: the watcher declares a fault whenever tb.wait() RETURNS with ``stop`` still UNSET.
An adopter's teardown (``finally: ctrl.close(); tb.stop(); tb.wait()``) used to run tb.stop()
WITHOUT setting ``stop`` first — so an ordinary Python exception escaping the main loop (a UHD
RuntimeError on a retune, a ValueError from a fold, any script bug) made the watcher's tb.wait()
return with ``stop`` unset and the crash was MISREPORTED as an RF fault (a false FAULT_MARKER; the
agent then skipped its crash-restart path and an auto-restart policy burned its budget relaunching
a deterministic traceback). Every adopter now sets ``stop`` BEFORE tb.stop() on the teardown path,
which the static check pins per file and the behavioural test proves on the real cw_tx.py.
"""
import ast
import importlib.util
import io
import os
import pathlib
import sys
import threading
import types

import pytest

RPI = pathlib.Path(__file__).resolve().parent.parent / "Raspberry pi + b206 mini-i"

# repeat=False, producer-fed FIFO + a --duration deadline → a normal EOF halts the graph on its own.
FIFO_EXCLUDED = {"gps_l1p_tx.py", "gps_l2p_tx.py", "white_noise_tx.py", "gaussian_noise_tx.py"}


def _rpi_sources():
    return list(RPI.rglob("*.py"))


def _adopters():
    return [p for p in _rpi_sources() if "watch_flowgraph(tb, stop)" in p.read_text()]


def test_clean_set_adopted_the_done_watcher_uniformly():
    adopted = []
    for p in _rpi_sources():
        text = p.read_text()
        if "watch_flowgraph(tb, stop)" in text:
            adopted.append(p.name)
            # each adopter wires it fully: the import, the call after tb.start(), the gated return
            assert "from paramkit.txhealth import watch_flowgraph" in text, p.name
            assert "return 1 if _health.faulted else 0" in text, p.name
    # the exact clean-set inventory (continuous / repeat=True scripts, incl. the incident fm_chirp)
    assert len(adopted) == 30, sorted(adopted)


def test_fifo_stagers_are_excluded():
    for p in _rpi_sources():
        if p.name in FIFO_EXCLUDED:
            assert "watch_flowgraph" not in p.read_text(), \
                f"{p.name}: a repeat=False FIFO stager must NOT adopt the done-watcher (false fault)"


def test_mock_scripts_have_no_flowgraph_watcher():
    # The FakeRadio mocks run no GNU Radio top block, so there is no tb.wait() to watch.
    for p in _rpi_sources():
        if p.name.startswith("mock_"):
            assert "watch_flowgraph" not in p.read_text(), p.name


# ── review fix #7: the teardown sets `stop` BEFORE tb.stop() ─────────────────────────────────

def _method_calls(stmts):
    """The `<obj>.<method>()` expression-statements in a statement list, in order."""
    out = []
    for s in stmts:
        if (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                and isinstance(s.value.func, ast.Attribute)
                and isinstance(s.value.func.value, ast.Name)):
            out.append(f"{s.value.func.value.id}.{s.value.func.attr}")
    return out


def _teardown_calls(path):
    """The ordered method calls of the `finally:` that tears the flowgraph down (the one holding
    tb.stop()), found by AST so a re-indented / re-commented teardown still counts."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and node.finalbody:
            calls = _method_calls(node.finalbody)
            if "tb.stop" in calls:
                found.append(calls)
    assert len(found) == 1, (path.name, found)
    return found[0]


def test_every_adopter_sets_stop_before_tb_stop_on_teardown():
    adopters = _adopters()
    assert len(adopters) == 30
    for p in adopters:
        calls = _teardown_calls(p)
        assert "stop.set" in calls, f"{p.name}: the teardown never sets `stop` — a crash out of the " \
                                    f"loop would read as a flowgraph fault ({calls})"
        assert calls.index("stop.set") < calls.index("tb.stop"), \
            f"{p.name}: `stop.set()` must precede `tb.stop()` on the teardown path ({calls})"
        assert "tb.wait" in calls and calls.index("tb.stop") < calls.index("tb.wait"), (p.name, calls)


# ── the behavioural proof: a crash in the loop is NOT an RF fault ─────────────────────────────

def _find_agent():
    cands = []
    if os.environ.get("SDR_AGENT_PATH"):
        cands.append(pathlib.Path(os.environ["SDR_AGENT_PATH"]))
    cands += [p / "sdr-agent" for p in pathlib.Path(__file__).resolve().parents]
    for c in cands:
        if (c / "paramkit" / "txhealth.py").is_file():
            return c
    return None


# A stand-in `gnuradio` package (the same shape tests/test_cw_drift.py runs the real scripts
# against as a subprocess): every block is an inert object, and the top block's wait() BLOCKS
# until stop() — modelling a real flowgraph, so the watcher only sees wait() return when the
# graph is stopped (or halts on its own).
_FAKE_GNURADIO = '''
class _Obj:
    def __init__(self, *a, **k): self._gain = 0.0
    def __getattr__(self, name): return lambda *a, **k: None
    def get_gain(self, *a): return self._gain
    def set_gain(self, g, *a): self._gain = float(g)

class _Mod:
    def __getattr__(self, name):
        if name.isupper(): return name
        return _Obj

import threading as _threading
class _TopBlock:
    def __init__(self, *a, **k): self._done = _threading.Event()
    def connect(self, *a): pass
    def start(self): pass
    def stop(self): self._done.set()
    def wait(self): self._done.wait()

gr = _Mod(); gr.top_block = _TopBlock
analog = _Mod(); blocks = _Mod(); uhd = _Mod()
'''


def _fake_gnuradio_module():
    mod = types.ModuleType("gnuradio")
    exec(_FAKE_GNURADIO, mod.__dict__)
    return mod


@pytest.fixture
def txhealth_env(monkeypatch):
    """sdr-agent's paramkit on the path + the fake gnuradio installed; returns the txhealth module."""
    agent = _find_agent()
    if agent is None:
        pytest.skip("sdr-agent (paramkit) not found; set SDR_AGENT_PATH")
    monkeypatch.syspath_prepend(str(agent))
    monkeypatch.setitem(sys.modules, "gnuradio", _fake_gnuradio_module())
    monkeypatch.delenv("SDR_CTRL_SOCK", raising=False)       # live control stays an inert no-op
    monkeypatch.delenv("SDR_CALIBRATION_FILE", raising=False)
    import paramkit.txhealth as txhealth
    return txhealth


def _capturing_watcher(monkeypatch, txhealth):
    """Wrap watch_flowgraph so the test can reach the Watcher main() keeps as a local, with the
    marker routed to a private buffer (the watcher thread prints there, not to the test's stdout)."""
    real = txhealth.watch_flowgraph
    seen = {"watcher": None, "out": io.StringIO()}

    def wrapped(tb, stop, **kw):
        kw.setdefault("stream", seen["out"])
        w = real(tb, stop, **kw)
        seen["watcher"] = w
        return w
    monkeypatch.setattr(txhealth, "watch_flowgraph", wrapped)
    return seen


def test_a_crash_in_the_main_loop_is_a_plain_crash_not_an_rf_fault(monkeypatch, txhealth_env):
    """Drive the REAL cw_tx.py main() (fake gnuradio, no radio) and make a live retune blow up
    inside the loop: the exception must propagate (an ordinary crash exit) while the watcher
    stays un-faulted and prints NO health marker — the finally sets `stop` before tb.stop(), so
    the watcher reads the resulting tb.wait() return as an intentional stop."""
    txhealth = txhealth_env
    import paramkit.live as live
    script = RPI / "Other Signals" / "cw_tx.py"
    spec = importlib.util.spec_from_file_location("cw_tx_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # --gain: no calibration needed; sys.argv is what Script.parse() reads.
    monkeypatch.setattr(sys, "argv", [str(script), "--freq", "1227.6", "--power", "-30", "--gain", "60"])
    # main() installs SIGTERM/SIGINT handlers — keep pytest's own (a no-op stub in the module).
    monkeypatch.setattr(mod, "signal", types.SimpleNamespace(
        SIGTERM=15, SIGINT=2, signal=lambda *a, **k: None))
    # The fault injection: the FIRST drain hands the loop a live retune whose value can't be
    # parsed (`float("not-a-number")` inside apply_change → ValueError) — the "a retune blew up"
    # class of ordinary script exception.
    monkeypatch.setattr(live.LiveControl, "drain",
                        lambda self: [live.Change("freq", "not-a-number")])
    seen = _capturing_watcher(monkeypatch, txhealth)

    with pytest.raises(ValueError):
        mod.main()                                   # (a) the exception propagates — a plain crash

    w = seen["watcher"]
    assert w is not None, "cw_tx.py never started the done-watcher"
    w.join(timeout=5.0)                              # the finally's tb.stop() ended its tb.wait()
    assert not w._thread.is_alive()
    assert w.faulted is False                        # (b) not an RF fault …
    assert txhealth.FAULT_MARKER not in seen["out"].getvalue()   # … and no marker for the watchdog


def _script_shaped_run(txhealth, *, set_stop_on_teardown: bool):
    """The adopters' exact loop + teardown shape, inline, with a loop body that raises — the
    before/after control for the test above: WITHOUT `stop.set()` in the finally (the pre-fix
    shape) the crash reads as a flowgraph fault; WITH it (the shipped shape) it does not."""
    gr = sys.modules["gnuradio"].gr
    tb = gr.top_block()
    stop = threading.Event()
    out = io.StringIO()
    tb.start()
    health = txhealth.watch_flowgraph(tb, stop, stream=out)
    try:
        try:
            while not stop.is_set():
                raise RuntimeError("UHD: retune failed")     # any exception out of the loop
        finally:
            if set_stop_on_teardown:
                stop.set()
            tb.stop()
            tb.wait()
    except RuntimeError:
        pass
    health.join(timeout=5.0)
    return health.faulted, out.getvalue()


def test_the_teardown_shape_decides_the_verdict(txhealth_env):
    txhealth = txhealth_env
    faulted, out = _script_shaped_run(txhealth, set_stop_on_teardown=False)
    assert faulted is True and txhealth.FAULT_MARKER in out          # the pre-fix false fault
    faulted, out = _script_shaped_run(txhealth, set_stop_on_teardown=True)
    assert faulted is False and txhealth.FAULT_MARKER not in out     # the shipped shape

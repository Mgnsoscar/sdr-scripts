"""
RF-fault Phase 1 (§5.1): the CLEAN-set RPi transmit scripts adopt paramkit.txhealth.watch_flowgraph
(a silent GR halt → a non-zero exit the agent sees), while the repeat=False FIFO stagers are
EXCLUDED — their normal end-of-stream returns from tb.wait() with stop unset and would read as a
false fault; they rely on the agent's Layer-2 log-scan watchdog instead.
"""
import pathlib

RPI = pathlib.Path(__file__).resolve().parent.parent / "Raspberry pi + b206 mini-i"

# repeat=False, producer-fed FIFO + a --duration deadline → a normal EOF halts the graph on its own.
FIFO_EXCLUDED = {"gps_l1p_tx.py", "gps_l2p_tx.py", "white_noise_tx.py", "gaussian_noise_tx.py"}


def _rpi_sources():
    return list(RPI.rglob("*.py"))


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

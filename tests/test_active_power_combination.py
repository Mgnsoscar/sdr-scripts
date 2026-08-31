"""No-hardware end-to-end check of the ACTIVE-component power path.

Requesting a calibrated delivered ``--power`` must resolve to the right COMBINATION of SDR
gain (commanded by the transmit script) and attenuation (a one-shot the agent fires on the
control task) — together they deliver the requested power. This drives the real mock scripts
(mock_sdr_tx.py, mock_atten.py) with a resolved calibration artifact and checks the two
against the agent resolver, without any radio or attenuator connected.

The SDR is −40..0 dBm (1 dB gain grid) and the active component is an AD-USB1AR36G95-shaped
step attenuator (0..95 dB, 0.25 dB) ⇒ an effective −135..0 dBm.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ── locate the agent checkout (paramkit + the calibration resolver) for a dev run ────
def _find_agent():
    cands = []
    if os.environ.get("SDR_AGENT_PATH"):
        cands.append(Path(os.environ["SDR_AGENT_PATH"]))
    here = Path(__file__).resolve()
    cands += [p / "sdr-agent" for p in here.parents]
    for c in cands:
        if (c / "paramkit").is_dir() and (c / "agent" / "calibration.py").is_file():
            return c
    return None


_AGENT = _find_agent()
if _AGENT is None:
    pytest.skip("sdr-agent (paramkit + resolver) not found; set SDR_AGENT_PATH",
                allow_module_level=True)
sys.path.insert(0, str(_AGENT))

from agent.calibration import resolve   # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "Other Signals"
_MOCK_SDR = _SCRIPTS / "mock_sdr_tx.py"
_MOCK_ATTEN = _SCRIPTS / "mock_atten.py"

# SDR: 1 dB gain ⇒ 1 dB power over 0..40 dB (−40..0 dBm), then a 0..95/0.25 attenuator.
SDR_POINTS = [(0, -40.0), (40, 0.0)]


def _doc():
    return {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": 0.0, "max_gain_db": 40.0, "gain_step_db": 1.0},
            "operating_plane": "atten_out",
            "planes": {
                "sdr_output": {"type": "measured", "quantity": "power"},
                "atten_out": {"type": "derived", "from": "sdr_output", "delta_db": 0.0,
                              "control": {"task": "mock_atten", "param": "attenuation",
                                          "sense": "attenuation", "min_db": 0.0,
                                          "max_db": 95.0, "step_db": 0.25, "engage_pct": 0.0}},
            },
        },
        "signals": {"mock": {"curves": {"sdr_output": {"points": [
            {"gain_db": g, "power_dbm": p} for g, p in SDR_POINTS]}}}},
        "defaults": {"amplitude": 0.5},
    }


def _sdr_curve(gain_db: float) -> float:
    """Delivered power at the SDR/operating plane for a commanded gain (linear over
    SDR_POINTS) — the attenuator's baseline is 0 dB, so this is P_base(gain)."""
    (g0, p0), (g1, p1) = SDR_POINTS
    return p0 + (p1 - p0) * (gain_db - g0) / (g1 - g0)


@pytest.fixture(scope="module")
def resolved():
    return resolve(_doc(), None, "mock", {})


@pytest.fixture(scope="module")
def artifact_file(resolved, tmp_path_factory):
    path = tmp_path_factory.mktemp("cal") / "artifact.json"
    path.write_text(json.dumps(resolved.to_public_dict()))
    return path


def _run(script: Path, args, artifact=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    if artifact is not None:
        env["SDR_CALIBRATION_FILE"] = str(artifact)
    out = subprocess.run([sys.executable, str(script), *args],
                         capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 0, f"{script.name} failed: {out.stderr}\n{out.stdout}"
    return out.stdout


def _result(stdout: str, key: str) -> float:
    m = re.search(rf"{key}=(-?\d+(?:\.\d+)?)", stdout)
    assert m, f"no {key} in output: {stdout!r}"
    return float(m.group(1))


def test_extended_range_is_the_spec_example(resolved):
    assert resolved.min_power_dbm == pytest.approx(-135.0)
    assert resolved.max_power_dbm == pytest.approx(0.0)


@pytest.mark.parametrize("requested", [0.0, -20.0, -40.0, -55.25, -100.0, -134.75, -135.0])
def test_sdr_gain_and_attenuation_combine_to_deliver_the_power(requested, resolved, artifact_file):
    # What the resolver says the two devices should do for this request.
    want = resolved.realize(requested)
    want_gain = want["sdr_gain_db"]
    want_atten = want["settings"][0]["value"]
    achieved = want["power_dbm"]

    # 1) The mock SDR, given the injected calibration, commands exactly that SDR gain.
    sdr_out = _run(_MOCK_SDR, ["--power", f"{requested:g}"], artifact=artifact_file)
    assert _result(sdr_out, "gain_db") == pytest.approx(want_gain)

    # 2) The mock attenuator, run one-shot with that value, applies exactly that attenuation.
    att_out = _run(_MOCK_ATTEN, ["--attenuation", f"{want_atten:g}"])
    assert _result(att_out, "attenuation_db") == pytest.approx(want_atten)

    # 3) The COMBINATION is physically right: SDR-plane power at the commanded gain, minus the
    #    attenuation the attenuator applies, equals the achievable delivered power.
    delivered = _sdr_curve(want_gain) - want_atten
    assert delivered == pytest.approx(achieved)
    assert delivered == pytest.approx(resolved.snap_power(requested))


def test_agent_auto_fires_the_real_attenuator_oneshot(tmp_path, monkeypatch):
    """The full automatic path: the agent's ProcessManager, asked to launch a calibrated
    transmit task at −100 dBm, fires the REAL mock attenuator one-shot (no running task) with
    the resolved value, before the transmit — reading the value back from its task log."""
    import asyncio
    from agent import process_manager as pm
    from agent.models import TaskConfig
    from agent.process_manager import ProcessManager

    doc_path = tmp_path / "calibration.json"
    doc_path.write_text(json.dumps(_doc()))
    (tmp_path / "defaults.yaml").write_text("types: {}\n")
    (tmp_path / "components.yaml").write_text("components: {}\n")
    monkeypatch.setattr(pm._agentcfg, "CALIBRATION_DOC", doc_path)
    monkeypatch.setattr(pm._agentcfg, "CALIBRATION_DEFAULTS", tmp_path / "defaults.yaml")
    monkeypatch.setattr(pm._agentcfg, "CALIBRATION_COMPONENTS", tmp_path / "components.yaml")

    env = {"PYTHONPATH": str(_AGENT)}
    tasks = {
        "mock_sdr": TaskConfig(name="mock_sdr", command=["python3", str(_MOCK_SDR)],
                               working_dir=str(_SCRIPTS),
                               env={**env, "SDR_CAL_SIGNAL_ID": "mock"}),
        "mock_atten": TaskConfig(name="mock_atten", command=["python3", str(_MOCK_ATTEN)],
                                 working_dir=str(_SCRIPTS), env=dict(env)),
    }
    mgr = ProcessManager(tasks, tmp_path, "unit-a")

    # The agent resolves --attenuation from the control task's argspec and fires it one-shot.
    assert mgr._active_flag("mock_atten", "attenuation") == "--attenuation"
    asyncio.run(mgr._precommand_active("mock_sdr", -100.0))

    logtext = mgr._get("mock_atten").log.current.read_text()
    assert "attenuation_db=60" in logtext, logtext        # −100 dBm ⇒ SDR floor + 60 dB atten


def test_sdr_first_keeps_the_attenuator_at_rest_in_the_sdrs_own_range(resolved):
    # From 0 down to −40 the SDR carries it and the attenuator stays at 0 dB.
    for p, g in [(0.0, 40.0), (-20.0, 20.0), (-40.0, 0.0)]:
        r = resolved.realize(p)
        assert r["sdr_gain_db"] == pytest.approx(g)
        assert r["settings"][0]["value"] == pytest.approx(0.0)
    # Below the SDR floor the SDR pins at min gain and the attenuator fills the rest.
    deep = resolved.realize(-100.0)
    assert deep["sdr_gain_db"] == pytest.approx(0.0)
    assert deep["settings"][0]["value"] == pytest.approx(60.0)

"""No-hardware check that mock_cw_tx.py commands the right SDR gain for a calibrated dBm ``--power``
and stays a faithful stand-in for the real cw_tx.py.

A CW tone is a single dBm quantity (no power-quantity laws), so --power maps straight through the
unit's measured dBm curve at the transmit frequency. This drives the real mock script with a
resolved calibration and asserts the mapping, a raw --gain override, the uncalibrated refusal, and
that the mock's static calibration surface matches cw_tx.py (drift guard).

No radio, no agent process — just the script + the resolver.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


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

from agent.calibration import resolve                 # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "Other Signals"
_MOCK = _SCRIPTS / "mock_cw_tx.py"
_REAL = _SCRIPTS / "cw_tx.py"

# A clean 1:1 dBm curve for the test doc: dBm = gain − 120 over a 0..80 dB / 0.25 dB grid.
GAIN_MIN, GAIN_MAX, GAIN_STEP = 0.0, 80.0, 0.25


def _doc():
    return {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": GAIN_MIN, "max_gain_db": GAIN_MAX,
                            "gain_step_db": GAIN_STEP},
            "operating_plane": "sdr_output",
            "planes": {"sdr_output": {"type": "measured", "quantity": "power"}},
        },
        "signals": {"cw_tone": {
            "measurement": {"quantity": "power", "unit": "dBm"},
            "curves": {"sdr_output": {"interp": "linear", "points": [
                {"gain_db": GAIN_MIN, "power_dbm": -120.0},
                {"gain_db": GAIN_MAX, "power_dbm": -40.0}]}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": 0.5},
    }


@pytest.fixture(scope="module")
def artifact_file(tmp_path_factory):
    import json
    art = resolve(_doc(), None, "cw_tone").to_public_dict()
    path = tmp_path_factory.mktemp("cal") / "cw.json"
    path.write_text(json.dumps(art))
    return path


def _run(args, artifact=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    if artifact is not None:
        env["SDR_CALIBRATION_FILE"] = str(artifact)
    out = subprocess.run([sys.executable, str(_MOCK), *args, "--once"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 0, f"mock failed: {out.stderr}\n{out.stdout}"
    return out.stdout


def _gain(stdout):
    m = re.search(r"gain_db=(-?\d+(?:\.\d+)?)", stdout)
    assert m, f"no gain_db in output: {stdout!r}"
    return float(m.group(1))


def _expected_gain(power_dbm):
    g = power_dbm + 120.0                              # dBm = gain − 120
    g = round(g / GAIN_STEP) * GAIN_STEP
    return max(GAIN_MIN, min(GAIN_MAX, round(g, 6)))


@pytest.mark.parametrize("power_dbm,want_gain", [
    (-120.0, 0.0), (-100.0, 20.0), (-70.0, 50.0), (-40.0, 80.0),
])
def test_dbm_power_maps_to_the_curve_gain(power_dbm, want_gain, artifact_file):
    out = _run(["--freq", "1575.42e6", "--power", f"{power_dbm:g}"], artifact=artifact_file)
    assert _gain(out) == pytest.approx(want_gain)
    assert _gain(out) == pytest.approx(_expected_gain(power_dbm))


def test_raw_gain_override(artifact_file):
    # --power is required by the CW schema (matching cw_tx.py); a raw --gain overrides it.
    out = _run(["--freq", "1575.42e6", "--power", "-60", "--gain", "55"], artifact=artifact_file)
    assert _gain(out) == pytest.approx(55.0)


def test_mock_matches_the_real_cw_calibration_surface():
    from agent.argspec import extract_params
    real = extract_params(_REAL.read_text(encoding="utf-8"))
    mock = extract_params(_MOCK.read_text(encoding="utf-8"))
    assert mock["calibration_signal"] == real["calibration_signal"] == "cw_tone"
    assert mock["calibration_freq_param"] == real["calibration_freq_param"] == "freq"
    # CW carries no power-quantity laws (a tone is a single dBm quantity).
    assert not mock.get("calibration_power_laws")
    assert mock.get("calibration_power_laws") == real.get("calibration_power_laws")

    def surface(d):
        keep = ("freq", "power", "gain", "rf")
        return {p["dest"]: (p.get("unit"), p.get("kind")) for p in d["params"] if p["dest"] in keep}
    assert surface(mock) == surface(real)


def test_uncalibrated_power_is_refused():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, str(_MOCK), "--freq", "1575.42e6",
                          "--power", "-60", "--once"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 2
    assert "not calibrated" in out.stderr

"""No-hardware check that mock_gps_ca_code_1.023Mcps_tx.py commands the right SDR gain for a
calibrated ``--power`` — the quantity path the client's power card drives — and that it stays a
faithful stand-in for the real GPS C/A script.

The client always SENDS ``--power`` in the calibration's BASE (measured) quantity — here the peak
spectral density (dBm/Hz) — converting from whatever quantity you controlled on the form (main-lobe
power, full signal power) via the signal's declared laws. This drives the real mock script with a
resolved calibration and asserts:

  • a base --power maps to the gain the calibration curve dictates (density = gain − 200 here);
  • MAIN-LOBE power is sidelobe-invariant (constant law) → same gain at any --sidelobes;
  • holding a fixed FULL signal power needs LESS base density as the passband widens (its enbw
    grows), so the base the client sends falls with the sidelobe count;
  • a raw --gain override bypasses the mapping;
  • the mock's static calibration surface matches the real gps_ca_code_1.023Mcps.py (drift guard).

No radio, no attenuator, no agent process — just the script + the resolver + the shared law math.
"""
import math
import os
import re
import subprocess
import sys
import importlib.util
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
from paramkit.power_law import parse_law              # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "PRN GPS"
_MOCK = _SCRIPTS / "mock_gps_ca_code_1.023Mcps_tx.py"
_REAL = _SCRIPTS / "gps_ca_code_1.023Mcps.py"

# Load the mock module (for its enbw_mhz(n)) so the test folds full power exactly as it does.
_spec = importlib.util.spec_from_file_location("mock_gps_ca", _MOCK)
_mock_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mock_mod)
enbw_mhz = _mock_mod.enbw_mhz

# A clean 1:1 density curve for the test doc: density = gain − 200 over a 0..80 dB / 0.25 dB grid.
GAIN_MIN, GAIN_MAX, GAIN_STEP = 0.0, 80.0, 0.25
FULL = {"id": "full_power", "in": "density", "out": "abs",
        "k": 60.0, "param": "enbw_mhz", "coeff": 10.0, "ref": 1.0, "rep": 0.988638}
MAIN = {"id": "main_lobe_power", "in": "density", "out": "abs", "k": 59.654784}


def _doc():
    return {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": GAIN_MIN, "max_gain_db": GAIN_MAX,
                            "gain_step_db": GAIN_STEP},
            "operating_plane": "sdr_output",
            "planes": {"sdr_output": {
                "type": "measured", "quantity": "spectral density",
                # A density measurement is capped through a law that returns dBm; set high so it
                # never binds across the test's gain range (keeps the mapping clean).
                "limiting": {"kind": "law", "law": FULL, "max_dbm": 20.0}}},
        },
        "signals": {"GPS C/A (1.023 Mcps)": {
            "measurement": {"quantity": "spectral density", "unit": "dBm/Hz"},
            "curves": {"sdr_output": {"interp": "linear", "points": [
                {"gain_db": GAIN_MIN, "power_dbm": -200.0},
                {"gain_db": GAIN_MAX, "power_dbm": -120.0}]}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": 0.5},
    }


@pytest.fixture(scope="module")
def artifact_file(tmp_path_factory):
    import json
    art = resolve(_doc(), None, "GPS C/A (1.023 Mcps)").to_public_dict()
    path = tmp_path_factory.mktemp("cal") / "gps_ca.json"
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


def _expected_gain(base_density):
    """Gain the curve dictates for a base measured density (density = gain − 200), snapped to the
    0.25 dB grid and clamped — computed independently of the script."""
    g = base_density + 200.0
    g = round(g / GAIN_STEP) * GAIN_STEP
    return max(GAIN_MIN, min(GAIN_MAX, round(g, 6)))


# ── base --power maps to the curve's gain ───────────────────────────────────────

@pytest.mark.parametrize("base_density,want_gain", [
    (-200.0, 0.0), (-180.0, 20.0), (-150.0, 50.0), (-120.0, 80.0),
])
def test_base_power_maps_to_the_curve_gain(base_density, want_gain, artifact_file):
    out = _run(["--prn", "1", "--sidelobes", "5", "--power", f"{base_density:g}"],
               artifact=artifact_file)
    assert _gain(out) == pytest.approx(want_gain)
    assert _gain(out) == pytest.approx(_expected_gain(base_density))


# ── main-lobe power is sidelobe-invariant → same gain at any passband width ──────

@pytest.mark.parametrize("sidelobes", [0, 5, 12, 28])
def test_main_lobe_power_gives_the_same_gain_across_sidelobes(sidelobes, artifact_file):
    main_dbm = -100.0                          # operator controls in main-lobe power
    base = main_dbm - parse_law(MAIN).delta_db({})     # constant law, no keyed param
    out = _run(["--prn", "1", "--sidelobes", str(sidelobes), "--power", f"{base:g}"],
               artifact=artifact_file)
    assert base == pytest.approx(-100.0 - 59.654784)
    assert _gain(out) == pytest.approx(_expected_gain(base))


# ── holding a fixed FULL power → less base density as the passband widens ────────

def test_full_power_base_falls_as_sidelobes_grow(artifact_file):
    full_dbm = -70.0                           # held full signal power (filter passband)
    law = parse_law(FULL)
    bases, gains = {}, {}
    for n in (0, 5, 28):
        base = full_dbm - law.delta_db({"enbw_mhz": enbw_mhz(n)})
        bases[n] = base
        gains[n] = _gain(_run(["--prn", "1", "--sidelobes", str(n), "--power", f"{base:g}"],
                              artifact=artifact_file))
        assert base == pytest.approx(full_dbm - (60.0 + 10 * math.log10(enbw_mhz(n))))
        assert gains[n] == pytest.approx(_expected_gain(base))
    # more sidelobes ⇒ larger enbw ⇒ lower base density needed to hold the same full power
    assert bases[0] > bases[5] > bases[28]
    # full(0) uses the SAME delta as the main-lobe law (passband == main lobe at 0 sidelobes)
    assert bases[0] == pytest.approx(full_dbm - parse_law(MAIN).delta_db({}), abs=1e-4)


def test_raw_gain_override(artifact_file):
    out = _run(["--prn", "1", "--sidelobes", "5", "--gain", "55"], artifact=artifact_file)
    assert _gain(out) == pytest.approx(55.0)


# ── the mock stays a faithful stand-in for the real GPS C/A (drift guard) ────────

def test_mock_matches_the_real_gps_ca_calibration_surface():
    from agent.argspec import extract_params
    real = extract_params(_REAL.read_text(encoding="utf-8"))
    mock = extract_params(_MOCK.read_text(encoding="utf-8"))
    assert mock["calibration_signal"] == real["calibration_signal"] == "GPS C/A (1.023 Mcps)"
    assert mock["calibration_freq_param"] == real["calibration_freq_param"] == "freq"
    assert mock["calibration_power_laws"] == real["calibration_power_laws"]

    def surface(d):
        keep = ("power", "gain", "freq", "prn", "sidelobes", "passband_bw_mhz", "enbw_mhz", "rf")
        return {p["dest"]: (p.get("unit"), p.get("hidden"), p.get("formula"), p.get("kind"))
                for p in d["params"] if p["dest"] in keep}
    assert surface(mock) == surface(real)


def test_uncalibrated_power_is_refused():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, str(_MOCK), "--prn", "1", "--sidelobes", "5",
                          "--power", "-120", "--once"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 2
    assert "not calibrated" in out.stderr

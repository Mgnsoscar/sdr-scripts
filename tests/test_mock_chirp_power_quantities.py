"""No-hardware check that mock_fm_chirp_tx.py commands the right SDR gain for a calibrated
``--power`` — the quantity path the client's power card drives.

The client always SENDS ``--power`` in the calibration's BASE (measured) quantity, converting
from whatever quantity you were controlling on the form (spectral density, total power, dBm/Hz)
via the signal's declared laws. This drives the real mock script with a resolved calibration and
asserts:

  • a base --power maps to the gain the calibration curve dictates (density = gain − 150 here);
  • TOTAL power is bandwidth-invariant, so the same total → the same gain at any sweep width;
  • a fixed live SPECTRAL DENSITY needs more gain as the sweep widens (its base density rises);
  • a raw --gain override bypasses the mapping.

No radio, no attenuator, no agent process — just the script + the resolver + the shared law math.
"""
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


# ── locate the agent checkout (paramkit + the calibration resolver) ─────────────────
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

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "Other Signals"
_MOCK = _SCRIPTS / "mock_fm_chirp_tx.py"
_REAL = _SCRIPTS / "fm_chirp_tx.py"

# The SDR is measured in spectral density (dBm/MHz) at the 10 MHz reference sweep: density =
# gain − 150 over a 0..70 dB / 0.5 dB grid. This mirrors the mock's own sample calibration.
GAIN_MIN, GAIN_MAX, GAIN_STEP = 0.0, 70.0, 0.5
CAL_MEAS_BW_MHZ = 10.0
FBW = {"id": "fbw_power", "in": "density", "out": "abs", "k": 10.0, "rep": 10.0}
PSD = {"id": "psd_live", "in": "density", "out": "density",
       "param": "bw", "coeff": -10.0, "ref": 10.0, "rep": 10.0}


def _doc():
    return {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": GAIN_MIN, "max_gain_db": GAIN_MAX,
                            "gain_step_db": GAIN_STEP},
            "operating_plane": "sdr_output",
            "planes": {"sdr_output": {
                "type": "measured", "quantity": "spectral density",
                "limiting": {"kind": "law", "law": FBW, "max_dbm": 0.0}}},
        },
        "signals": {"fm_chirp": {
            "measurement": {"quantity": "spectral density", "unit": "dBm/MHz"},
            "curves": {"sdr_output": {"interp": "linear", "points": [
                {"gain_db": GAIN_MIN, "power_dbm": -150.0},
                {"gain_db": GAIN_MAX, "power_dbm": -80.0}]}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": 0.5},
    }


@pytest.fixture(scope="module")
def artifact_file(tmp_path_factory):
    import json
    art = resolve(_doc(), None, "fm_chirp").to_public_dict()
    path = tmp_path_factory.mktemp("cal") / "fm_chirp.json"
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


def _expected_gain(base_density_dbm_per_mhz):
    """The gain the curve dictates for a base measured density, snapped to the 0.5 dB grid and
    clamped — computed independently of the script (density = gain − 150)."""
    g = base_density_dbm_per_mhz + 150.0
    g = round(g / GAIN_STEP) * GAIN_STEP
    return max(GAIN_MIN, min(GAIN_MAX, round(g, 6)))


def _base_for(quantity_law, operator_value, bw_mhz):
    """The BASE measured density the client sends when the operator sets ``operator_value`` in
    ``quantity_law`` at sweep width ``bw_mhz`` (base = value − the law's delta over measured)."""
    law = parse_law(quantity_law)
    return operator_value - law.delta_db({"bw": bw_mhz})


# ── base --power maps to the curve's gain ───────────────────────────────────────

@pytest.mark.parametrize("base_density,want_gain", [
    (-120.0, 30.0), (-130.0, 20.0), (-95.0, 55.0), (-150.0, 0.0), (-80.0, 70.0),
])
def test_base_power_maps_to_the_curve_gain(base_density, want_gain, artifact_file):
    out = _run(["--freq", "1575.42", "--bw", "20", "--rate", "200",
                "--power", f"{base_density:g}"], artifact=artifact_file)
    assert _gain(out) == pytest.approx(want_gain)
    assert _gain(out) == pytest.approx(_expected_gain(base_density))


# ── total power is bandwidth-invariant → same gain at any sweep width ────────────

@pytest.mark.parametrize("bw", [5.0, 10.0, 20.0, 40.0])
def test_total_power_quantity_gives_the_same_gain_across_bandwidth(bw, artifact_file):
    total_dbm = -110.0                      # what the operator sets controlling in total power
    base = _base_for(FBW, total_dbm, bw)    # what the client sends (base measured density)
    out = _run(["--freq", "1575.42", "--bw", f"{bw:g}", "--rate", "200",
                "--power", f"{base:g}"], artifact=artifact_file)
    # total = base + 10 → base = −120 for every bw → gain 30 for every bw
    assert base == pytest.approx(-120.0)
    assert _gain(out) == pytest.approx(30.0)


# ── a fixed spectral density needs more gain as the sweep widens ────────────────

def test_spectral_density_quantity_tracks_bandwidth(artifact_file):
    density_dbm_per_mhz = -120.0            # held live spectral density (psd_live)
    gains = {}
    for bw in (10.0, 20.0, 40.0):
        base = _base_for(PSD, density_dbm_per_mhz, bw)   # base = D + 10·log10(bw/10)
        out = _run(["--freq", "1575.42", "--bw", f"{bw:g}", "--rate", "200",
                    "--power", f"{base:g}"], artifact=artifact_file)
        gains[bw] = _gain(out)
        assert base == pytest.approx(density_dbm_per_mhz + 10 * math.log10(bw / CAL_MEAS_BW_MHZ))
        assert gains[bw] == pytest.approx(_expected_gain(base))
    # more bandwidth ⇒ higher base density ⇒ more gain to hold the same per-MHz density
    assert gains[10.0] < gains[20.0] < gains[40.0]
    assert gains[10.0] == pytest.approx(30.0)            # at the reference sweep, base = D


# ── a raw --gain override bypasses the mapping ──────────────────────────────────

def test_raw_gain_override(artifact_file):
    out = _run(["--freq", "1575.42", "--bw", "20", "--rate", "200", "--gain", "55"],
               artifact=artifact_file)
    assert _gain(out) == pytest.approx(55.0)


# ── the mock stays a faithful stand-in for the real chirp (drift guard) ─────────

def test_mock_matches_the_real_chirp_calibration_surface():
    # The client renders the mock's Run/tune form from the SAME static schema reader that reads
    # the real chirp, so the calibration surface (signal id, fold frequency, power laws, and the
    # power-quantity-relevant params) must match — else the mock would drive a different form.
    from agent.argspec import extract_params
    real = extract_params(_REAL.read_text(encoding="utf-8"))
    mock = extract_params(_MOCK.read_text(encoding="utf-8"))
    assert mock["calibration_signal"] == real["calibration_signal"] == "fm_chirp"
    assert mock["calibration_freq_param"] == real["calibration_freq_param"]
    assert mock["calibration_power_laws"] == real["calibration_power_laws"]

    def surface(d):
        keep = ("band_mode", "freq", "start", "stop", "band_center", "band_span",
                "power", "gain", "bw")
        return {p["dest"]: (p.get("unit"), p.get("show_when"), p.get("provides"),
                            p.get("is_freq"), p.get("kind"))
                for p in d["params"] if p["dest"] in keep}
    assert surface(mock) == surface(real)


def test_uncalibrated_power_is_refused():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, str(_MOCK), "--freq", "1575.42", "--bw", "20",
                          "--rate", "200", "--power", "-120", "--once"],
                         capture_output=True, text=True, env=env, timeout=30)
    assert out.returncode == 2
    assert "not calibrated" in out.stderr

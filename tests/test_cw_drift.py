"""cw_drift_tx.py — a CW tone drifting hundreds of MHz over a very long time (no hardware).

The drift is a baseband NCO within a window of the sample rate; a span wider than the window is
swept window by window with analog-LO hops (the X410 cw_channel planner, ported). Calibrated
power is re-folded at the live frequency as the tone moves, and the sweep is checked at start
for stretches where the calibration's ceiling sits below the request. These tests exercise the
pure planner math + the schema the client sees (the static argspec reader), plus the script's
own --self-test. No radio, no agent process.
"""
import importlib.util
import json
import os
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

from agent.argspec import extract_params                  # noqa: E402
from agent.calibration import resolve                     # noqa: E402
from paramkit import PowerMap                             # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "Other Signals"
_DRIFT = _SCRIPTS / "cw_drift_tx.py"
_TONE = _SCRIPTS / "cw_tx.py"


def _load():
    spec = importlib.util.spec_from_file_location("cw_drift_tx", _DRIFT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cwd = _load()

S, E, D = 1600e6, 1300e6, 10800.0          # 300 MHz DOWN over three hours


# ── a cw_tone calibration with a frequency-dependent source flatness ─────────

def _doc(bias=None):
    doc = {
        "schema_version": 1, "unit_type": "broadcaster",
        "chain": {
            "gain_limits": {"min_gain_db": 0.0, "max_gain_db": 80.0, "gain_step_db": 0.25},
            "operating_plane": "sdr_output",
            "limits": [{"plane": "sdr_output", "max_dbm": -35.0, "reason": "test cap"}],
            "planes": {"sdr_output": {"type": "measured", "quantity": "power"}},
        },
        "signals": {"cw_tone": {
            "measurement": {"quantity": "power", "unit": "dBm"},
            "curves": {"sdr_output": {"interp": "linear", "points": [
                {"gain_db": 0.0, "power_dbm": -120.0},
                {"gain_db": 80.0, "power_dbm": -40.0}]}},
            "center_freq_hz": 1575.42e6}},
        "defaults": {"amplitude": 0.5},
    }
    if bias:
        doc["source_bias"] = {"power_by_freq": [[float(f), float(d)] for f, d in bias]}
    return doc


def _pmap(doc):
    art = resolve(doc, None, "cw_tone").to_public_dict()
    return PowerMap.from_artifact(art, 0.5)


# ── the drift law ────────────────────────────────────────────────────────────

def test_drift_freq_modes():
    f = cwd.drift_freq
    assert f(0, S, E, D, "once") == S and f(D, S, E, D, "once") == E
    assert f(2 * D, S, E, D, "once") == E                            # holds at the end
    assert f(D / 2, S, E, D, "once") == pytest.approx(0.5 * (S + E))   # linear
    assert f(2 * D, S, E, D, "loop") == pytest.approx(S)             # wraps
    assert f(D, S, E, D, "pingpong") == E and f(2 * D, S, E, D, "pingpong") == pytest.approx(S)
    assert f(5, S, S, D, "once") == S                                # no end ⇒ static


# ── the LO-hop planner ───────────────────────────────────────────────────────

def _walk(start, end, rate, mode="once", step_s=1.0, duration=D):
    """Simulate the loop: (max |offset| / half_window, hop count) over one drift."""
    half = cwd.half_window_hz(rate)
    lo = cwd.initial_lo(start, end, rate)
    hops, worst = 0, 0.0
    t = 0.0
    while t <= duration * (2.0 if mode == "pingpong" else 1.0) + 1e-9:
        f = cwd.drift_freq(t, start, end, duration, mode)
        new_lo = cwd.plan_lo(f, lo, half)
        if new_lo != lo:
            hops += 1
            lo = new_lo
        worst = max(worst, abs(f - lo) / half)
        t += step_s
    return worst, hops


@pytest.mark.parametrize("rate_mhz,expected_hops", [(2.0, 214), (10.0, 42)])
def test_wide_sweep_hops_once_per_window_and_keeps_the_nco_inside(rate_mhz, expected_hops):
    rate = rate_mhz * 1e6
    assert cwd.is_wide(S, E, rate)
    worst, hops = _walk(S, E, rate)
    assert worst <= 1.0 + 1e-9                          # the NCO never leaves ±half_window
    assert hops == cwd.hop_count(S, E, rate) == expected_hops
    window = 2 * cwd.half_window_hz(rate)
    assert abs(hops - abs(E - S) / window) <= 1.0       # ≈ span / window
    # The LO grid is centred on the sweep: first and last windows sit symmetrically.
    lo0 = cwd.initial_lo(S, E, rate)
    lo1 = cwd.plan_lo(E, lo0, cwd.half_window_hz(rate))
    assert abs((lo0 + lo1) / 2 - 0.5 * (S + E)) < 1e-6


def test_an_upward_sweep_hops_the_same_as_a_downward_one():
    assert cwd.hop_count(E, S, 2e6) == cwd.hop_count(S, E, 2e6) == 214
    worst, hops = _walk(E, S, 2e6)
    assert worst <= 1.0 + 1e-9 and hops == 214


def test_narrow_drift_never_hops_and_sits_on_the_centre():
    s, e, rate = 1575.42e6, 1575.43e6, 2e6
    assert not cwd.is_wide(s, e, rate) and cwd.hop_count(s, e, rate) == 0
    assert cwd.initial_lo(s, e, rate) == pytest.approx(0.5 * (s + e))
    worst, hops = _walk(s, e, rate)
    assert hops == 0 and worst < 0.01
    # Exactly one window wide still fits (no hops); a hair more needs them.
    w = 2 * cwd.half_window_hz(rate)
    assert not cwd.is_wide(s, s + w, rate) and cwd.is_wide(s, s + w + 1.0, rate)


def test_loop_and_pingpong_stay_inside_the_window_across_the_wrap():
    # loop: the jump back to start crosses ~214 windows in one plan_lo call
    worst, hops = _walk(S, E, 2e6, mode="loop", duration=600.0)
    assert worst <= 1.0 + 1e-9 and hops >= 214
    worst, hops = _walk(S, E, 2e6, mode="pingpong", duration=600.0)
    assert worst <= 1.0 + 1e-9 and 2 * 214 - 1 <= hops <= 2 * 214 + 1


# ── the calibrated gain follows the tone ─────────────────────────────────────

def test_gain_refolds_every_refold_step_and_tracks_the_flatness():
    assert cwd.REFOLD_STEP_HZ == 250e3
    assert cwd.needs_refold(1500.25e6, 1500e6) and not cwd.needs_refold(1500.2e6, 1500e6)
    # A 6 dB flatness rise between 1.3 and 1.6 GHz: the gain for a held −70 dBm drops 6 dB
    # from the start of the sweep to its end (bias is normalised to 0 at the rep frequency).
    pm = _pmap(_doc(bias=[(1.3e9, -11.0), (1.6e9, -5.0)]))
    g_start = pm.gain_for_power(-70.0, freq=S)
    g_end = pm.gain_for_power(-70.0, freq=E)
    assert g_end - g_start == pytest.approx(6.0, abs=0.25)
    assert pm.power_for_gain(g_end, freq=E) == pytest.approx(-70.0, abs=0.13)


def test_coverage_gaps_name_the_stretch_where_the_ceiling_bites():
    # A 15 dB dip in the source flatness around 1.2 GHz: the top of the range at the start
    # frequency can't be delivered there — the gap covers the dip and nothing else.
    pm = _pmap(_doc(bias=[(1.0e9, -20.0), (1.2e9, -20.0), (1.4e9, -5.0), (1.6e9, -5.0)]))
    top = pm.max_power_dbm
    gaps = cwd.coverage_gaps(pm, top, 1600e6, 1150e6)
    assert len(gaps) == 1
    f_lo, f_hi, worst = gaps[0]
    assert f_lo <= 1.2e9 <= f_hi and f_hi < 1.4e9
    assert worst < top - 10.0                              # ≈ 15 dB short at the dip
    # A level the whole sweep can deliver has no gap; uncalibrated is silent.
    assert cwd.coverage_gaps(pm, top - 20.0, 1600e6, 1150e6) == []
    assert cwd.coverage_gaps(PowerMap.uncalibrated(0.0, 80.0, 0.5), -30.0, S, E) == []


def test_fmt_rate_reads_in_the_right_units():
    assert cwd.fmt_rate(-300e6 / 10800.0) == "-27.78 kHz/s (-100 MHz/h)"
    assert cwd.fmt_rate(10e3 / 1200.0) == "8.333 Hz/s (30 kHz/h)"


# ── the schema the client sees (static argspec) + the shared calibration signal ──

def test_schema_surface_and_shared_calibration_signal():
    drift = extract_params(_DRIFT.read_text(encoding="utf-8"))
    tone = extract_params(_TONE.read_text(encoding="utf-8"))
    assert drift["calibration_signal"] == tone["calibration_signal"] == "cw_tone"
    assert drift["calibration_freq_param"] == tone["calibration_freq_param"] == "freq"
    assert not drift.get("calibration_power_laws")       # a tone is a single dBm quantity
    by = {p["dest"]: p for p in drift["params"]}
    assert set(by) >= {"freq", "freq_end", "duration", "drift", "sample_rate", "hop_blank",
                       "power", "rf", "restart", "gain"}
    assert by["duration"]["max"] == 7 * 86400 and by["duration"]["unit"] == "s"
    assert by["freq_end"]["max"] == 6e9 and by["freq_end"]["unit"] == "Hz"
    assert by["hop_blank"]["live"] and by["hop_blank"]["unit"] == "ms"
    assert by["rf"]["is_rf"] and by["rf"]["live"]
    assert by["power"]["live"] and by["power"]["unit"] == "dBm"
    vals = [p.get("value") if isinstance(p, dict) else (p[1] if isinstance(p, (list, tuple)) else p)
            for p in by["sample_rate"]["presets"]]
    assert 10.0 in vals and 20.0 in vals and 40.0 not in vals
    tone_by = {p["dest"]: p for p in tone["params"]}
    assert tone_by["rf"]["is_rf"]                          # the pure tone marks its gate too


def test_self_test_and_describe_params_run_without_a_radio():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_AGENT) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, str(_DRIFT), "--self-test"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "SELF-TEST OK" in out.stdout and "214 LO hops" in out.stdout
    out = subprocess.run([sys.executable, str(_DRIFT), "--describe-params"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stderr
    names = {p.get("name") or p.get("dest") for p in json.loads(out.stdout)["params"]}
    assert {"freq", "freq_end", "duration", "hop_blank", "restart"} <= names

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
import time
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


def test_the_attenuator_split_is_pinned_across_the_drift():
    # A chain with a programmable attenuator (insertion −4.5 dB, engaged from mid-gain — the
    # SDR is held at the engagement threshold and the attenuator absorbs the rest, so the split
    # it picks follows the flatness): the agent positions it at the START carrier; the drift
    # must fold only the SDR gain from there on (PowerMap pinned fold), never re-pick the split
    # — and the coverage check folds the same way.
    d = _doc(bias=[(1.3e9, -3.0), (1.6e9, 3.0)])
    d["chain"]["planes"]["atten_out"] = {
        "type": "derived", "from": "sdr_output", "delta_db": -4.5,
        "control": {"task": "atten_set", "param": "attenuation", "sense": "attenuation",
                    "min_db": 0.0, "max_db": 95.0, "step_db": 0.25, "engage_pct": 50.0}}
    d["chain"]["operating_plane"] = "atten_out"
    pm = _pmap(d)
    pin = pm.pinned_applied(-100.0, freq=S)
    assert pin is not None and pin < 0.0                  # engaged: a real attenuation
    for f in (S, 1500e6, 1400e6, E):
        g = pm.gain_for_power(-100.0, freq=f, applied_db=pin)
        assert abs(pm.power_for_gain(g, freq=f, applied_db=pin) - (-100.0)) <= 0.13
    assert cwd.coverage_gaps(pm, -100.0, S, E, applied_db=pin) == []
    # Without the pin the realization hops to another attenuation along the sweep (the SDR
    # stays at the threshold, so the flatness lands in the attenuator) — the very mismatch
    # pinning prevents.
    assert any(pm.pinned_applied(-100.0, freq=f) != pin for f in (1500e6, 1400e6, E))


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
    # --duration is entered in MINUTES (7 days = 10080), default 10 min; the drift law runs in
    # seconds behind the boundary (the banner test below proves the ×60).
    assert by["duration"]["unit"] == "min" and by["duration"]["max"] == 7 * 24 * 60
    assert by["duration"]["default"] == 10.0 and by["duration"]["min"] == 0.1
    # Both carriers are entered in MHz (like the PRN scripts' -Center-frequency); the planner
    # works in Hz behind the boundary. Presets + defaults are MHz too, on both scripts.
    for dest in ("freq", "freq_end"):
        assert by[dest]["unit"] == "MHz" and (by[dest]["min"], by[dest]["max"]) == (70.0, 6000.0)
        vals = [p.get("value") if isinstance(p, dict) else (p[1] if isinstance(p, (list, tuple)) else p)
                for p in by[dest]["presets"]]
        assert 1575.42 in vals and all(70.0 <= v <= 6000.0 for v in vals)
    assert by["freq"]["default"] == 1575.42 and by["freq_end"]["default"] == 1575.43
    assert {p["dest"]: p for p in tone["params"]}["freq"]["unit"] == "MHz"
    assert by["hop_blank"]["live"] and by["hop_blank"]["unit"] == "ms"
    assert by["rf"]["is_rf"] and by["rf"]["live"]
    assert by["power"]["live"] and by["power"]["unit"] == "dBm"
    vals = [p.get("value") if isinstance(p, dict) else (p[1] if isinstance(p, (list, tuple)) else p)
            for p in by["sample_rate"]["presets"]]
    assert 10.0 in vals and 20.0 in vals and 40.0 not in vals
    tone_by = {p["dest"]: p for p in tone["params"]}
    assert tone_by["rf"]["is_rf"]                          # the pure tone marks its gate too


# A stand-in `gnuradio` package so the REAL scripts' main() runs to their start banner with no
# radio: the banner prints the frequencies main() derived from the MHz args, so it proves the
# MHz → Hz boundary (a wrong scale would print 0.001600 MHz, or trip the planner).
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
    # wait() BLOCKS until stop() — modelling a real flowgraph (it returns when the graph halts or is
    # stopped), so paramkit.txhealth.watch_flowgraph doesn't read an instant return as a false halt.
    def __init__(self, *a, **k): self._done = _threading.Event()
    def connect(self, *a): pass
    def start(self): pass
    def stop(self): self._done.set()
    def wait(self): self._done.wait()

gr = _Mod(); gr.top_block = _TopBlock
analog = _Mod(); blocks = _Mod(); uhd = _Mod()
'''


def _banner(tmp_path, script, args):
    pkg = tmp_path / "gnuradio"
    pkg.mkdir(exist_ok=True)
    (pkg / "__init__.py").write_text(_FAKE_GNURADIO)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(_AGENT), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen([sys.executable, str(script), *args], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = []
    try:
        while True:                                   # the banner ends with a full rule line
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
            if line.strip() and set(line.strip()) == {"─"} and len(lines) > 1:
                break
        time.sleep(0.5)                               # let it reach the loop (SIGTERM handler)
        proc.terminate()                              # SIGTERM → the loop exits cleanly
        _out, err = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
    # The banner is the deliverable. A clean exit (0) is the norm; −15 means the SIGTERM landed
    # before the handler was installed (a slow box) — still not a crash, which shows as a
    # traceback / a non-zero argparse or script error.
    assert proc.returncode in (0, -15) and "Traceback" not in err, "".join(lines) + err
    return "".join(lines)


def test_the_tone_scripts_take_mhz_and_run_in_hz(tmp_path):
    out = _banner(tmp_path, _DRIFT, ["--freq", "1600", "--freq_end", "1300", "--duration",
                                     "180", "--sample_rate", "2", "--power", "-30",
                                     "--gain", "60", "--rf", "on"])
    assert "1600.000000 → 1300.000000 MHz over 180 min" in out and "WIDE — 300.000 MHz" in out
    assert "214 analog-LO hops per pass" in out                 # the planner saw a 300 MHz span
    # --duration is MINUTES: 300 MHz over 180 min = 3 h → −27.78 kHz/s (an unscaled 180 s would
    # read −1.667 MHz/s).
    assert "-27.78 kHz/s (-100 MHz/h)" in out
    out = _banner(tmp_path, _DRIFT, ["--freq", "1575.42", "--freq_end", "1575.43", "--duration",
                                     "10", "--sample_rate", "2", "--power", "-30", "--gain", "60"])
    assert "1575.420000 → 1575.430000 MHz over 10 min" in out and "narrow" in out
    out = _banner(tmp_path, _TONE, ["--freq", "1227.6", "--power", "-30", "--gain", "60"])
    assert "tone           : 1227.600000 MHz" in out


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

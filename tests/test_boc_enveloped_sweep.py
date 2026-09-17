"""boc_enveloped_sweep_tx.py — the Enveloped Sweep shaped like a sine-phased BOC(10,5) (GPS
M-code) spectrum: a constant-envelope swept tone whose time-averaged PSD is the split spectrum
(two lobes at ±10.23 MHz, a gap at the centre), by dwell time only. Tests: the BOC PSD, the
dwell shaping (split spectrum, constant envelope, seam), confinement to ±roam, the reused
calibration + full / main-lobes laws, and the schema the client sees.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


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

_SCRIPTS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "Other Signals"
_BOC = _SCRIPTS / "boc_enveloped_sweep_tx.py"


def _load():
    spec = importlib.util.spec_from_file_location("boc_enveloped_sweep_tx", _BOC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bs = _load()


def _psd_db(sidelobes, smooth=31):
    base, _f = bs.build_boc_sweep_buffer(sidelobes)
    filt, _t, _fp = bs.filter_buffer(base, 2 * bs.roam_hz(sidelobes), bs.FILTER_TRANSITION_HZ)
    X = np.abs(np.fft.fftshift(np.fft.fft(filt))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(filt), 1.0 / bs.SAMP_RATE_HZ))
    X = np.convolve(X, np.ones(smooth) / smooth, "same")
    return f, 10 * np.log10(X / X.max() + 1e-30)


def _at(f, P, fq):
    return float(P[np.argmin(np.abs(f - fq))])


# ── the analytic BOC(10,5) PSD ────────────────────────────────────────────────────

def test_boc_psd_is_the_split_spectrum():
    f = np.linspace(-31e6, 31e6, 400001)
    G = bs.boc_psd(f)
    assert np.all(np.isfinite(G))                          # no NaN at the removable singularities
    G = G / G.max()
    # peak sits in the main lobe around ±10.23 MHz, not at the centre
    peak_f = abs(f[np.argmax(G)])
    assert 8e6 < peak_f < 13e6
    # deep null at the code rate and at the main-lobe outer edge; a gap at DC (split spectrum)
    assert 10 * np.log10(G[np.argmin(np.abs(f - 5.115e6))] + 1e-30) < -20
    assert 10 * np.log10(G[np.argmin(np.abs(f - 15.345e6))] + 1e-30) < -20
    assert 10 * np.log10(G[np.argmin(np.abs(f))] + 1e-30) < -20


# ── the schema the client sees + the reused calibration ──────────────────────────

def test_argspec_surface_and_reuse():
    spec = extract_params(_BOC.read_text(encoding="utf-8"))
    assert spec["calibration_signal"] == "Chirp/Sweep"     # reuses the flat sweep's calibration
    assert spec["calibration_freq_param"] == "freq"

    by = {p["dest"]: p for p in spec["params"]}
    assert {"power", "gain", "freq", "sidelobes", "rf"} <= set(by)
    assert "chip_rate" not in by                            # BOC(10,5) rates are fixed
    assert by["sidelobes"]["max"] == bs.MAX_SIDELOBES
    assert by["rf"].get("is_rf") is True
    assert by["main_lobe_frac"]["hidden"] is True

    laws = {l["id"]: l for l in spec["calibration_power_laws"]}
    assert set(laws) == {"full_power", "main_lobe_power"}


def test_power_laws_evaluate():
    from paramkit.power_law import parse_law
    import math
    laws = {l["id"]: l for l in extract_params(_BOC.read_text(encoding="utf-8"))["calibration_power_laws"]}
    assert parse_law(laws["full_power"]).delta_db({}) == pytest.approx(70.0)
    main = parse_law(laws["main_lobe_power"])
    d0 = main.delta_db({"main_lobe_frac": bs.main_lobe_frac(0)})
    assert d0 == pytest.approx(70.0 + 10 * math.log10(bs.main_lobe_frac(0)))
    assert d0 < 70.0                                        # main lobes below the full signal
    # the M-code note: full(0) sits ~0.39 dB above the main lobes (the ±15.345 band passes the gap)
    assert 70.0 - d0 == pytest.approx(0.39, abs=0.05)


def test_main_lobe_frac_matches_the_boc_integral():
    f = np.linspace(-bs.MAX_ROAM_MHZ * 1e6, bs.MAX_ROAM_MHZ * 1e6, 400001)
    G = bs.boc_psd(f)
    ml = np.sum(G[(np.abs(f) >= 5.115e6) & (np.abs(f) < 15.345e6)])
    for sl in range(0, bs.MAX_SIDELOBES + 1):
        tot = np.sum(G[np.abs(f) < (sl + bs.MAIN_LOBE_NULLS) * bs.BOC_NULL_HZ])
        assert bs.main_lobe_frac(sl) == pytest.approx(ml / tot, abs=5e-3)


# ── the dwell shaping: constant envelope, seam, the realised split spectrum ───────

def test_constant_envelope_and_seam():
    iq, _f = bs.build_boc_sweep_buffer(0)
    assert float(np.max(np.abs(np.abs(iq) - 1.0))) < 1e-5
    f = bs._boc_dwell_freq(0, bs.BUFFER_SAMPS)
    phase = (2 * np.pi / bs.SAMP_RATE_HZ) * np.cumsum(f)
    iq = np.exp(1j * phase)
    seam = abs(((np.angle(iq[0] / iq[-1]) - 2 * np.pi / bs.SAMP_RATE_HZ * f[0] + np.pi)
                % (2 * np.pi)) - np.pi)
    assert seam < 1e-9


def test_realised_psd_is_the_split_spectrum():
    f, P = _psd_db(0)
    m = (np.abs(f) >= 5.115e6) & (np.abs(f) < 15.345e6)
    assert float(np.max(P[m])) > -1.0                      # the main lobes carry the peak
    assert _at(f, P, 0.0) < -10.0                          # centre gap (split spectrum)
    assert _at(f, P, 5.115e6) < -10.0                      # 1st null
    assert _at(f, P, 15.345e6) < -10.0                     # main-lobe outer null


def test_sidelobes_set_the_occupied_band():
    assert bs.roam_hz(0) == pytest.approx(15.345e6)        # main lobes only
    assert bs.roam_hz(3) == pytest.approx(30.69e6)         # + 3 null-steps = Fs/2


def test_check_roam_guard():
    bs.check_roam(3)                                        # ±30.69 MHz — the max, no raise
    with pytest.raises(ValueError):
        bs.check_roam(4)                                   # ±35.8 MHz — over Nyquist


def test_sweep_stays_inside_the_occupied_band():
    for sl in (0, 1, 3):
        f = bs._boc_dwell_freq(sl, bs.BUFFER_SAMPS)
        assert float(np.max(np.abs(f))) <= bs.roam_hz(sl) + 1.0


# ── subprocess smoke ──────────────────────────────────────────────────────────────

def _run(*args):
    env = dict(os.environ, PYTHONPATH=str(_AGENT))
    return subprocess.run([sys.executable, str(_BOC), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_self_test_subprocess():
    r = _run("--self-test")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELF-TEST OK" in r.stdout


def test_describe_params_subprocess():
    r = _run("--describe-params")
    assert r.returncode == 0, r.stdout + r.stderr
    names = {p.get("name") for p in json.loads(r.stdout).get("params", [])}
    assert {"sidelobes", "freq", "power"} <= names

"""enveloped_sweep_tx.py — a constant-envelope swept tone whose time-averaged PSD is a
sinc², shaped by DWELL TIME only (no amplitude taper, no crest-factor penalty). These
tests exercise the pure DSP (inverse-CDF dwell → sinc² spectrum, constant envelope, seam
closure), the occupied-band guard, and the schema the client sees (the static argspec
reader) + the script's own --self-test. No radio, no agent process.
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
_SWEEP = _SCRIPTS / "enveloped_sweep_tx.py"


def _load():
    spec = importlib.util.spec_from_file_location("enveloped_sweep_tx", _SWEEP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


es = _load()


# ── helpers ─────────────────────────────────────────────────────────────────────

def _psd_db(chip_rate_hz, sidelobes, smooth=15):
    """Filtered, normalised power spectrum (dB) + the frequency axis (Hz)."""
    base, _f = es.build_enveloped_buffer(chip_rate_hz, sidelobes)
    filt, _t, _fp = es.filter_buffer(base, 2 * es.roam_hz(chip_rate_hz, sidelobes),
                                     es.FILTER_TRANSITION_HZ)
    X = np.abs(np.fft.fftshift(np.fft.fft(filt))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(filt), 1.0 / es.SAMP_RATE_HZ))
    X = np.convolve(X, np.ones(smooth) / smooth, "same")
    return f, 10 * np.log10(X / X.max() + 1e-30)


def _at(f, P, fq):
    return float(P[np.argmin(np.abs(f - fq))])


# ── the schema the client sees (static argspec) ──────────────────────────────────

def test_argspec_surface():
    spec = extract_params(_SWEEP.read_text(encoding="utf-8"))
    assert spec["calibration_signal"] == "Enveloped Sweep"
    assert spec["calibration_freq_param"] == "freq"
    assert not spec.get("calibration_power_laws")          # single dBm quantity, no laws (v1)

    by = {p["dest"]: p for p in spec["params"]}
    assert set(by) == {"power", "gain", "freq", "chip_rate", "sidelobes", "rf"}
    assert by["chip_rate"]["unit"] == "Mcps"
    assert by["chip_rate"]["default"] == pytest.approx(1.023)
    assert by["chip_rate"]["min"] == pytest.approx(es.CHIP_RATE_MIN_MCPS)
    assert by["chip_rate"]["max"] == pytest.approx(es.CHIP_RATE_MAX_MCPS)
    assert by["sidelobes"]["default"] == es.SIDELOBES_DEFAULT
    assert by["sidelobes"]["max"] == es.SIDELOBES_MAX
    assert by["freq"]["unit"] == "MHz"
    # every knob is live; rf is flagged as the RF gate
    assert all(by[d]["live"] for d in by)
    assert by["rf"].get("is_rf") is True


# ── constant envelope + seamless loop (dwell shaping, not amplitude) ──────────────

def test_constant_envelope():
    iq, _f = es.build_enveloped_buffer(1.023e6, 3)
    assert float(np.max(np.abs(np.abs(iq) - 1.0))) < 1e-5   # unit modulus everywhere


def test_seam_closes():
    f = es._sinc2_dwell_freq(1.023e6, 3, es.BUFFER_SAMPS)
    phase = (2 * np.pi / es.SAMP_RATE_HZ) * np.cumsum(f)
    iq = np.exp(1j * phase)
    expected = 2 * np.pi / es.SAMP_RATE_HZ * f[0]
    measured = np.angle(iq[0] / iq[-1])
    seam = abs(((measured - expected + np.pi) % (2 * np.pi)) - np.pi)
    assert seam < 1e-9


# ── the averaged PSD is a sinc² ───────────────────────────────────────────────────

def test_sinc2_shape():
    cr = 1.023e6
    f, P = _psd_db(cr, 3)
    assert abs(_at(f, P, 0.0)) < 0.6                        # peak at the centre
    assert _at(f, P, 1.43 * cr) == pytest.approx(-13.3, abs=3.0)   # 1st sidelobe ≈ -13.3 dB
    assert _at(f, P, cr) < -18.0                            # 1st null (soft) well below the main lobe


def test_chip_rate_sets_the_null_position():
    # the first null sits at ±chip_rate for whatever chip rate is set
    for cr in (1.023e6, 2.0e6):
        f, P = _psd_db(cr, 3)
        assert _at(f, P, cr) < -18.0                        # a null at chip_rate
        assert _at(f, P, 0.5 * cr) > -6.0                   # still inside the main lobe → high


def test_sidelobes_set_the_occupied_band():
    assert es.roam_hz(1.023e6, 0) == pytest.approx(1.023e6)         # main lobe only
    assert es.roam_hz(1.023e6, 3) == pytest.approx(4 * 1.023e6)     # +3 sidelobes → ±4·chip


def test_wider_chip_rate_widens_the_spectrum():
    def w3(cr):
        f, P = _psd_db(cr, 3)
        m = P >= -3.0
        return f[m].max() - f[m].min()
    assert w3(2.0e6) > 1.6 * w3(1.0e6)                      # ~doubling chip rate ~doubles the width


# ── occupied-band guard ───────────────────────────────────────────────────────────

def test_check_roam_guard():
    es.check_roam(1.023e6, 3)                               # ±4.09 MHz — fine, no raise
    with pytest.raises(ValueError):
        es.check_roam(10.0e6, 3)                            # ±40 MHz — over the ±27 MHz max


# ── subprocess smoke: --self-test and --describe-params ──────────────────────────

def _run(*args):
    env = dict(os.environ, PYTHONPATH=str(_AGENT))
    return subprocess.run([sys.executable, str(_SWEEP), *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_self_test_subprocess():
    r = _run("--self-test")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SELF-TEST OK" in r.stdout


def test_describe_params_subprocess():
    r = _run("--describe-params")
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(r.stdout)
    names = {p.get("name") for p in doc.get("params", [])}
    assert {"chip_rate", "sidelobes", "freq", "power"} <= names

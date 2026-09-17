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
    assert {"power", "gain", "freq", "sidelobes", "inner_sidelobes", "rf"} <= set(by)
    assert "chip_rate" not in by                            # BOC(10,5) rates are fixed
    assert by["sidelobes"]["max"] == bs.MAX_SIDELOBES
    assert by["inner_sidelobes"]["max"] == bs.MAX_INNER_SIDELOBES and by["inner_sidelobes"]["default"] == 0
    assert by["rf"].get("is_rf") is True
    assert by["main_lobe_frac"]["hidden"] is True
    assert by["deliver_frac"]["hidden"] is True            # full-power notch key (hidden derived)

    laws = {l["id"]: l for l in spec["calibration_power_laws"]}
    assert set(laws) == {"full_power", "main_lobe_power"}
    assert laws["full_power"]["param"] == "deliver_frac"   # full power tracks the inner notch


def test_power_laws_evaluate():
    from paramkit.power_law import parse_law
    import math
    laws = {l["id"]: l for l in extract_params(_BOC.read_text(encoding="utf-8"))["calibration_power_laws"]}
    full = parse_law(laws["full_power"])
    # centre kept (--inner-sidelobes 1, deliver_frac 1.0) → the bandwidth-invariant constant-envelope
    # total, density + 70
    assert full.delta_db({"deliver_frac": bs.deliver_frac(1)}) == pytest.approx(70.0)
    main = parse_law(laws["main_lobe_power"])
    d0 = main.delta_db({"main_lobe_frac": bs.main_lobe_frac(0)})
    assert d0 == pytest.approx(70.0 + 10 * math.log10(bs.main_lobe_frac(0)))
    assert d0 < 70.0                                        # main lobes below the un-notched full signal
    # the M-code note: un-notched full(0) sits ~0.39 dB above the main lobes (the ±15.345 band
    # passes the low-power centre gap)
    assert 70.0 - d0 == pytest.approx(0.39, abs=0.05)


def test_full_power_is_exact_under_the_inner_notch():
    """--inner-sidelobes 0 notches the centre gap; Full-signal power reports the DELIVERED total
    (the discarded gap is subtracted via `deliver_frac`); at 0 sidelobes the notched signal IS the
    two main lobes, so full == main-lobes exactly. --inner-sidelobes 1 keeps the centre (frac 1.0)."""
    from paramkit.power_law import parse_law
    import math
    laws = {l["id"]: l for l in extract_params(_BOC.read_text(encoding="utf-8"))["calibration_power_laws"]}
    full = parse_law(laws["full_power"])
    main = parse_law(laws["main_lobe_power"])
    assert bs.deliver_frac(1) == 1.0                       # inner 1 → centre kept → nothing lost
    assert bs.deliver_frac(0) < 1.0                        # inner 0 → notched → the centre gap discarded
    # notched full = density + 70 + 10·log10(deliver_frac) — the centre gap is accounted for
    dn = full.delta_db({"deliver_frac": bs.deliver_frac(0)})
    assert dn == pytest.approx(70.0 + 10 * math.log10(bs.deliver_frac(0)))
    assert dn < 70.0                                        # delivered below the un-notched total
    # at 0 sidelobes the notched signal IS the two main lobes → full == main-lobes exactly
    d_main0 = main.delta_db({"main_lobe_frac": bs.main_lobe_frac(0)})
    assert dn == pytest.approx(d_main0, abs=1e-6)
    # the ~0.4 dB discarded gap is the same order as the un-notched full-vs-lobes gap
    assert -0.45 < 10 * math.log10(bs.deliver_frac(0)) < -0.30


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


# ── the inner notch (a bandpass that drops the centre gap) ───────────────────────

def _psd_db_filtered(base, width_hz, inner_hz):
    filt, _t, _fp = bs.filter_buffer(base, width_hz, bs.FILTER_TRANSITION_HZ, inner_hz=inner_hz)
    X = np.abs(np.fft.fftshift(np.fft.fft(filt))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(filt), 1.0 / bs.SAMP_RATE_HZ))
    X = np.convolve(X, np.ones(31) / 31, "same")
    return f, 10 * np.log10(X / X.max() + 1e-30)


def test_inner_edge_snaps_to_the_code_rate_null():
    assert bs.inner_edge_hz(0) == pytest.approx(5.115e6)    # inner 0 → notch below the 1st null
    assert bs.inner_edge_hz(1) == 0.0                       # inner 1 → no notch — a plain lowpass


def test_inner_notch_drops_the_centre_gap_but_keeps_the_lobes():
    base = bs.build_boc_sweep_buffer(0)[0]
    width = 2 * bs.roam_hz(0)
    f0, P0 = _psd_db_filtered(base, width, 0.0)                       # lowpass: centre kept
    f1, P1 = _psd_db_filtered(base, width, bs.inner_edge_hz(0))       # bandpass (inner 0): centre notched
    c0 = float(P0[np.argmin(np.abs(f0))])
    c1 = float(P1[np.argmin(np.abs(f1))])
    assert c1 < c0 - 20.0                                             # centre far deeper after the notch
    # the two main lobes are untouched
    m = (np.abs(f1) >= 5.115e6) & (np.abs(f1) < 15.345e6)
    assert float(np.max(P1[m])) > -1.0


def test_main_lobe_frac_is_notch_independent():
    # the notch never touches the main lobes, so main_lobe_power (keyed on --sidelobes) is unchanged
    assert bs.main_lobe_frac(0) == pytest.approx(0.914610, abs=1e-6)


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

"""Tests for calkit.PowerMap — the transmit script's calibration consumer.

Run from the repo root:  PYTHONPATH=. pytest tests/test_calkit.py
"""
import json

import pytest

from paramkit import PowerMap, CALIBRATION_FILE_ENV


# ── Baked fallback reproduces the old single-anchor slope-1 model ────────────────

GAIN_AT_MAX = 89.75
OUTPUT_POWER = -20.0
MAX_DELIVERED = OUTPUT_POWER            # cable/amp = 0
MIN_DELIVERED = MAX_DELIVERED - GAIN_AT_MAX


def _baked():
    return PowerMap.from_linear(0.0, GAIN_AT_MAX, MIN_DELIVERED, MAX_DELIVERED,
                                amplitude=0.8)


def test_baked_is_slope_one_through_the_anchor():
    m = _baked()
    assert m.source == "baked defaults"
    assert m.gain_for_power(-20.0) == pytest.approx(89.75)      # anchor
    assert m.gain_for_power(-30.0) == pytest.approx(79.75)      # 1 dB/dB
    assert m.power_for_gain(50.0) == pytest.approx(-20.0 - (89.75 - 50.0))


def test_baked_clamps_up_and_down():
    m = _baked()
    assert m.gain_for_power(0.0) == pytest.approx(89.75)        # above max → ceiling
    assert m.power_for_gain(89.75) == pytest.approx(-20.0)      # never above anchor
    assert m.gain_for_power(-500.0) == pytest.approx(0.0)       # below floor → min gain
    assert m.max_power_dbm == pytest.approx(-20.0)
    assert m.min_power_dbm == pytest.approx(MIN_DELIVERED)


# ── Artifact-backed map ──────────────────────────────────────────────────────────

def _artifact():
    return {
        "schema_version": 1,
        "signal_id": "gps_l1_mcode",
        "operating_plane": "antenna_eirp",
        "quantity": "EIRP",
        "amplitude": 0.8,
        "min_gain_db": 0.0,
        "max_gain_db": 74.0,
        # amplifier_output + 4.2 (cable -1.8, antenna +6.0), a deliberate kink 70→74
        "curve": [[40, -1.8], [50, 8.2], [60, 18.2], [70, 26.2], [74, 28.2]],
    }


def test_from_artifact_interpolates_and_labels():
    m = PowerMap.from_artifact(_artifact(), fallback_amplitude=0.5)
    assert m.source == "calibration file"
    assert m.label == "EIRP, at antenna_eirp"
    assert m.amplitude == 0.8
    assert m.max_gain_db == pytest.approx(74.0)
    # exact at a measured point
    assert m.power_for_gain(60) == pytest.approx(18.2)
    # across the kink: at 72 dB, halfway 26.2→28.2 = 27.2
    assert m.power_for_gain(72) == pytest.approx(27.2)


def test_from_artifact_inversion_round_trips():
    m = PowerMap.from_artifact(_artifact(), fallback_amplitude=0.8)
    for eirp in (0.0, 18.2, 27.2):
        assert m.power_for_gain(m.gain_for_power(eirp)) == pytest.approx(eirp, abs=1e-9)


def test_from_artifact_clamps_to_ceiling():
    m = PowerMap.from_artifact(_artifact(), fallback_amplitude=0.8)
    assert m.gain_for_power(999) == pytest.approx(74.0)         # never past the ceiling
    assert m.max_power_dbm == pytest.approx(28.2)


def test_amplitude_falls_back_when_artifact_omits_it():
    art = _artifact()
    art["amplitude"] = None
    m = PowerMap.from_artifact(art, fallback_amplitude=0.8)
    assert m.amplitude == 0.8


def test_non_monotonic_curve_rejected():
    art = _artifact()
    art["curve"] = [[40, -1.8], [50, -1.8]]                     # flat → not invertible
    with pytest.raises(ValueError):
        PowerMap.from_artifact(art, fallback_amplitude=0.8)


# ── load(): env selects artifact vs baked ────────────────────────────────────────

def test_load_without_env_returns_baked(monkeypatch):
    monkeypatch.delenv(CALIBRATION_FILE_ENV, raising=False)
    m = PowerMap.load(_baked())
    assert m.source == "baked defaults"


def test_load_with_env_reads_artifact(tmp_path, monkeypatch):
    p = tmp_path / "resolved.json"
    p.write_text(json.dumps(_artifact()))
    monkeypatch.setenv(CALIBRATION_FILE_ENV, str(p))
    m = PowerMap.load(_baked())
    assert m.source == "calibration file"
    assert m.label == "EIRP, at antenna_eirp"
    assert m.gain_for_power(18.2) == pytest.approx(60.0)


def test_load_uses_baked_amplitude_when_artifact_omits(tmp_path, monkeypatch):
    art = _artifact()
    art["amplitude"] = None
    p = tmp_path / "resolved.json"
    p.write_text(json.dumps(art))
    monkeypatch.setenv(CALIBRATION_FILE_ENV, str(p))
    m = PowerMap.load(PowerMap.from_linear(0.0, 89.75, MIN_DELIVERED, MAX_DELIVERED,
                                           amplitude=0.8))
    assert m.amplitude == 0.8


def test_load_with_broken_file_raises(tmp_path, monkeypatch):
    p = tmp_path / "resolved.json"
    p.write_text("{ not json ")
    monkeypatch.setenv(CALIBRATION_FILE_ENV, str(p))
    with pytest.raises(ValueError):
        PowerMap.load(_baked())

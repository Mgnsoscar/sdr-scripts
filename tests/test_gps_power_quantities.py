"""The GPS PRN scripts' spectral-density calibration surface — the quantity path the client's
power card drives. For each script this checks, through the SAME static reader the client uses
(agent.argspec) and the SAME law evaluator (paramkit.power_law), that:

  • the calibration signal id + frequency param are what the client scopes/folds on;
  • the density→power laws parse and evaluate (a measured peak density in dBm/Hz maps to the
    published absolute-power constant);
  • the constants match the spectrum (drift guard): for the BPSK signals the carrier (total)
    power is exactly 10·log10(1/I_ML) ≈ 0.444 dB above the main lobe; for the BOC signals the
    full-passband power sits above the main-lobe(s) power.

No radio and no agent process — the script text + the shared law math (each script's own
``--self-test`` separately re-derives the constants from its PSD)."""
import os
import sys
from pathlib import Path

import pytest


def _find_agent():
    if os.environ.get("SDR_AGENT_PATH"):
        cands = [Path(os.environ["SDR_AGENT_PATH"])]
    else:
        cands = [p / "sdr-agent" for p in Path(__file__).resolve().parents]
    for c in cands:
        if (c / "paramkit").is_dir() and (c / "agent" / "argspec.py").is_file():
            return c
    return None


_AGENT = _find_agent()
if _AGENT is None:
    pytest.skip("sdr-agent (paramkit + argspec) not found; set SDR_AGENT_PATH",
                allow_module_level=True)
sys.path.insert(0, str(_AGENT))

from agent.argspec import extract_params            # noqa: E402
from paramkit.power_law import parse_law            # noqa: E402

_GPS = Path(__file__).resolve().parents[1] / "Raspberry pi + b206 mini-i" / "PRN GPS"

# Per script: (signal id, freq param or None, {law id: k}, kind). The k values are pinned here so
# a change to a script's baked constant fails loudly until this test is updated too.
_BPSK_10 = {"main_lobe_power": 69.654784, "carrier_power": 70.098756}   # Rc = 10.23e6
_BPSK_1 = {"main_lobe_power": 59.654784, "carrier_power": 60.098756}    # Rc = 1.023e6
CASES = {
    "gps_l5_tx.py":  ("gps_l5", "freq", _BPSK_10, "bpsk"),
    "gps_l2c_tx.py": ("gps_l2c", "freq", _BPSK_1, "bpsk"),
    "gps_l1p_tx.py": ("gps_l1_p", None, _BPSK_10, "bpsk"),
    "gps_l2p_tx.py": ("gps_l2_p", None, _BPSK_10, "bpsk"),
    "MCode.py":      ("gps_l1_mcode", "freq",
                      {"main_lobe_power": 69.5073, "full_power": 70.1261}, "boc"),
    "gps_l1c_tx.py": ("gps_l1c", "freq",
                      {"main_lobe_power": 62.2246, "full_power": 63.2576}, "boc"),
}

I_ML = 0.902823
import math                                              # noqa: E402


@pytest.mark.parametrize("fname", list(CASES))
def test_gps_calibration_surface(fname):
    sig, freq, ks, kind = CASES[fname]
    p = extract_params((_GPS / fname).read_text(encoding="utf-8"))
    assert p.get("calibration_signal") == sig
    assert p.get("calibration_freq_param") == freq

    laws = {l["id"]: l for l in (p.get("calibration_power_laws") or [])}
    assert set(laws) == set(ks), f"{fname}: law ids {set(laws)} != {set(ks)}"

    for lid, k in ks.items():
        law = laws[lid]
        assert law["in"] == "density" and law["out"] == "abs"   # density → dBm
        assert law["unit"] == "dBm"
        # Evaluate through the real law engine: a peak density D (dBm/Hz) → D + k (a constant law).
        parsed = parse_law(law)
        assert parsed.rep_delta_db() == pytest.approx(k, abs=1e-4)
        D = -170.0                                              # a representative peak dBm/Hz
        assert D + parsed.rep_delta_db() == pytest.approx(D + k, abs=1e-4)


def test_bpsk_carrier_is_the_main_lobe_plus_the_sinc_tail():
    # For every BPSK signal, carrier (total) power = main-lobe power + 10·log10(1/I_ML) ≈ +0.444 dB.
    gap = 10 * math.log10(1.0 / I_ML)
    for fname, (_sig, _f, ks, kind) in CASES.items():
        if kind != "bpsk":
            continue
        assert ks["carrier_power"] - ks["main_lobe_power"] == pytest.approx(gap, abs=2e-3), fname


def test_boc_full_power_exceeds_the_main_lobes_and_has_no_carrier():
    for fname, (_sig, _f, ks, kind) in CASES.items():
        if kind != "boc":
            continue
        assert "carrier_power" not in ks                        # not offered for a BOC signal
        assert ks["full_power"] > ks["main_lobe_power"]         # widest passband ≥ the main lobes
        # …but only modestly (a split spectrum keeps most power in the main lobes): M-code ~0.6 dB,
        # L1C ~1.0 dB (its BOC(6,1) lobes add power outside the BOC(1,1) core).
        assert ks["full_power"] - ks["main_lobe_power"] < 1.5

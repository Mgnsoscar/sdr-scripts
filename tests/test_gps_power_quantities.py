"""The GPS PRN scripts' spectral-density calibration surface — the quantity path the client's
power card drives. For each script this checks, through the SAME static reader the client uses
(agent.argspec) and the SAME law evaluator (paramkit.power_law), that:

  • the calibration signal id + frequency param are what the client scopes/folds on;
  • exactly two power quantities are offered — the MAIN-LOBE integrated power and the FULL signal
    power — and NO carrier/total quantity is offered by name (the operator asked for the carrier
    to be dropped everywhere: the measured PSD in dBm/Hz + main-lobe + full-signal power only);
  • the density→power laws parse and evaluate (a measured peak density in dBm/Hz maps to the
    published absolute-power constant);
  • the constants match the spectrum (drift guard):
      - filtered BPSK (L5, L2C): full_power is KEYED on the filter's equivalent-noise bandwidth
        (enbw_mhz, a table lookup on --sidelobes) so it TRACKS the live filter, and it MEETS the
        main-lobe power when 0 sidelobes are kept (passband == main lobe);
      - streamed BPSK (L1P, L2P): full_power is the fixed TOTAL signal power, exactly
        10·log10(1/I_ML) ≈ 0.444 dB above the main lobe;
      - BOC (M-code, L1C): full_power (widest passband) sits modestly above the main lobe(s).

No radio and no agent process — the script text + the shared law math (each script's own
``--self-test`` separately re-derives the constants from its PSD)."""
import math
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

I_ML = 0.902823                                     # sinc² power fraction inside the main lobe

# Per script: signal id, freq param (or None), the family, and the expected law constants.
#   main_k  = the main-lobe integrated power k (a pure-constant law everywhere).
#   full_k  = the full-power k for the CONSTANT-full-power families (streamed BPSK / BOC).
#             The FILTERED-BPSK family has no fixed full_k — its full_power is keyed on enbw_mhz
#             (k = 60.0, i.e. 10·log10(MHz/Hz)); it is checked against the enbw table instead.
CASES = {
    "gps_l5_tx.py":  dict(sig="gps_l5",       freq="freq", kind="bpsk_filtered", main_k=69.654784),
    "gps_l2c_tx.py": dict(sig="gps_l2c",      freq="freq", kind="bpsk_filtered", main_k=59.654784),
    "gps_l1p_tx.py": dict(sig="gps_l1_p",     freq=None,   kind="bpsk_stream",
                          main_k=69.654784, full_k=70.098756),
    "gps_l2p_tx.py": dict(sig="gps_l2_p",     freq=None,   kind="bpsk_stream",
                          main_k=69.654784, full_k=70.098756),
    "MCode.py":      dict(sig="gps_l1_mcode", freq="freq", kind="boc",
                          main_k=69.5073, full_k=70.1261),
    "gps_l1c_tx.py": dict(sig="gps_l1c",      freq="freq", kind="boc",
                          main_k=62.2246, full_k=63.2576),
}


def _extract(fname):
    return extract_params((_GPS / fname).read_text(encoding="utf-8"))


def _laws(p):
    return {l["id"]: l for l in (p.get("calibration_power_laws") or [])}


def _enbw_table(p):
    """The static enbw_mhz(sidelobes) lookup the filtered-BPSK scripts publish as a hidden
    derived field — the values the client folds full_power through as --sidelobes moves."""
    for pr in p["params"]:
        if pr.get("dest") == "enbw_mhz" and pr.get("kind") == "derived":
            tbl = (pr.get("formula") or {}).get("table")
            assert tbl and tbl[0] == "sidelobes", f"enbw_mhz table keys on {tbl[0]!r}"
            return [float(v) for v in tbl[1:]]
    return None


@pytest.mark.parametrize("fname", list(CASES))
def test_gps_calibration_surface(fname):
    c = CASES[fname]
    p = _extract(fname)
    assert p.get("calibration_signal") == c["sig"]
    assert p.get("calibration_freq_param") == c["freq"]

    laws = _laws(p)
    # Exactly the two quantities the operator asked for — and no carrier quantity anywhere.
    assert set(laws) == {"full_power", "main_lobe_power"}, f"{fname}: {set(laws)}"
    assert "carrier_power" not in laws
    for law in laws.values():
        assert law["in"] == "density" and law["out"] == "abs"   # density → dBm
        assert law["unit"] == "dBm"

    # Main-lobe power: a pure-constant density→dBm law at the pinned k.
    main = parse_law(laws["main_lobe_power"])
    assert main.params() == []
    assert main.rep_delta_db() == pytest.approx(c["main_k"], abs=1e-4)

    full = parse_law(laws["full_power"])
    if c["kind"] == "bpsk_filtered":
        # KEYED on the filter's equivalent-noise bandwidth so it tracks --sidelobes live.
        assert full.params() == ["enbw_mhz"]
        assert laws["full_power"]["k"] == pytest.approx(60.0)     # 10·log10(MHz/Hz)
        tbl = _enbw_table(p)
        assert tbl is not None and len(tbl) >= 3
        assert all(tbl[i] < tbl[i + 1] for i in range(len(tbl) - 1))   # more sidelobes → more BW
        # 0 sidelobes: the passband IS the main lobe, so full_power == main-lobe power.
        assert full.delta_db({"enbw_mhz": tbl[0]}) == pytest.approx(c["main_k"], abs=2e-3)
        # widest passband keeps more than the main lobe → strictly above it.
        assert full.delta_db({"enbw_mhz": tbl[-1]}) > c["main_k"]
    else:
        # A fixed constant (streamed BPSK total, or BOC widest-passband) — the amp-limit reading.
        assert full.params() == []
        assert full.rep_delta_db() == pytest.approx(c["full_k"], abs=1e-4)


def test_no_carrier_quantity_in_any_script():
    for fname in CASES:
        laws = _laws(_extract(fname))
        assert "carrier_power" not in laws, f"{fname} still offers a carrier quantity"
        assert set(laws) == {"full_power", "main_lobe_power"}, fname


def test_streamed_bpsk_full_is_the_total_signal_power():
    # For the filterless P-code streams, the full (total) power = main-lobe power + 10·log10(1/I_ML)
    # ≈ +0.444 dB (the fraction of power outside the main lobe).
    gap = 10 * math.log10(1.0 / I_ML)
    for fname, c in CASES.items():
        if c["kind"] != "bpsk_stream":
            continue
        assert c["full_k"] - c["main_k"] == pytest.approx(gap, abs=2e-3), fname


def test_boc_full_power_exceeds_the_main_lobes():
    for fname, c in CASES.items():
        if c["kind"] != "boc":
            continue
        assert c["full_k"] > c["main_k"]              # widest passband ≥ the main lobes
        # …but only modestly (a split spectrum keeps most power in the main lobes): M-code ~0.6 dB,
        # L1C ~1.0 dB (its BOC(6,1) lobes add power outside the BOC(1,1) core).
        assert c["full_k"] - c["main_k"] < 1.5


def test_filtered_bpsk_full_power_tracks_sidelobes():
    # The whole point of the filtered-BPSK scheme: full_power keys on enbw_mhz, a derived table
    # lookup on --sidelobes, so the operator's "full signal power" reading and its amp-limit cap
    # move with the live filter width. Check the law + table wire up to a monotonic reading.
    for fname, c in CASES.items():
        if c["kind"] != "bpsk_filtered":
            continue
        p = _extract(fname)
        full = parse_law(_laws(p)["full_power"])
        tbl = _enbw_table(p)
        deltas = [full.delta_db({"enbw_mhz": v}) for v in tbl]
        assert all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1)), fname
        assert deltas[0] == pytest.approx(c["main_k"], abs=2e-3), fname

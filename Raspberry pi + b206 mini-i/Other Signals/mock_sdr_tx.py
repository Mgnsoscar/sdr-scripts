#!/usr/bin/env python3
"""
mock_sdr_tx — a NO-HARDWARE, one-shot SDR transmit mock for exercising the
power-calibration path (especially ACTIVE components) without any radio.

It never touches UHD / GNU Radio and never transmits. It just maps a requested
``--power`` (dBm) to the SDR gain it *would* command — reading the per-unit
calibration the agent injects (env ``SDR_CALIBRATION_FILE``) exactly like the real
transmit scripts — then prints that gain and exits. Pair it with a step attenuator
declared as an ACTIVE component in the calibration chain and the agent will set the
attenuator (a one-shot) first, so the SDR gain here + the attenuation together
deliver the requested power.

It prints one machine-readable line for tests/tooling:

    RESULT gain_db=<g> power_dbm=<p> source=<calibrated|uncalibrated>

CLI
───
    mock_sdr_tx.py --power -100            # dBm through the injected calibration
    mock_sdr_tx.py --gain 20              # raw SDR gain (dB), bypassing calibration
    mock_sdr_tx.py --describe-params      # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import logging
import os
import sys

# Make paramkit + calkit importable both on a unit (scripts flattened one level under
# BASE_DIR, next to paramkit/) and in the dev repo. Insert candidate roots; whichever
# holds paramkit wins. (PYTHONPATH is also honoured, e.g. pointing at the agent checkout.)
_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from paramkit import Script, PowerMap

# Stable calibration signal id — set a task's SDR_CAL_SIGNAL_ID to this and the agent
# injects this unit's resolved calibration for it (active components included).
CAL_SIGNAL_ID = "mock"

# RF-chain limits used ONLY as the uncalibrated fallback (no baked dBm scale).
MIN_GAIN_DB = 0.0
MAX_GAIN_DB = 89.75
AMPLITUDE = 0.5

log = logging.getLogger("mock_sdr_tx")


def power_map() -> PowerMap:
    """The unit's injected calibration curve if present (SDR_CALIBRATION_FILE), else an
    uncalibrated relative-gain-only map."""
    return PowerMap.load(PowerMap.uncalibrated(MIN_GAIN_DB, MAX_GAIN_DB, AMPLITUDE))


def build_script() -> Script:
    return (
        Script("Mock SDR transmitter (NO HARDWARE) — maps --power to the SDR gain it would "
               "command and exits. Exercises the calibrated-power path, active components "
               "included. Transmits nothing.")
        .number("-Frequency", "--freq", unit="Hz", min=1e6, max=6e9, default=1575.42e6,
                help="Informational carrier for the report line. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False,
                help="ABSOLUTE delivered power (dBm). Mapped to SDR gain through this unit's "
                     "calibration (active components extend the range). Ignored if --gain given.")
        .number("-Gain", "--gain", unit="dB", min=MIN_GAIN_DB, max=MAX_GAIN_DB,
                required=False,
                help="RELATIVE power: raw SDR gain (dB), bypassing calibration. Overrides --power.")
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = build_script().parse()
    pmap = power_map()

    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:                          # raw bench override
        gain_db = max(MIN_GAIN_DB, min(MAX_GAIN_DB, float(gain_cal)))
        power_dbm = pmap.power_for_gain(gain_db) if pmap.has_absolute else None
    elif pmap.has_absolute:                           # calibrated: absolute --power → gain
        if getattr(args, "power", None) is None:
            print("error: --power (or --gain) is required", file=sys.stderr)
            return 2
        gain_db = pmap.gain_for_power(float(args.power))
        power_dbm = pmap.power_for_gain(gain_db)
    else:                                             # uncalibrated: need a relative gain
        print("error: not calibrated for this signal — pass --gain (raw dB), not --power",
              file=sys.stderr)
        return 2

    log.info("── mock SDR (no hardware) ──────────────────────────────────")
    log.info("  signal id      : %s", CAL_SIGNAL_ID)
    log.info("  carrier (info) : %.3f MHz", args.freq / 1e6)
    if getattr(args, "power", None) is not None and gain_cal is None:
        log.info("  power (target) : %g dBm  (%s)", args.power, pmap.label)
    log.info("  → SDR gain     : %.2f dB   (radio.set_gain, not really)", gain_db)
    if power_dbm is not None:
        log.info("  delivered      : %.2f dBm at the operating plane (SDR alone)", power_dbm)
    log.info("  calibration    : %s", pmap.source)
    log.info("────────────────────────────────────────────────────────────")
    # One machine-readable line for tests / tooling.
    print("RESULT gain_db=%.6g power_dbm=%s source=%s"
          % (gain_db, ("%.6g" % power_dbm) if power_dbm is not None else "na",
             "calibrated" if pmap.has_absolute else "uncalibrated"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

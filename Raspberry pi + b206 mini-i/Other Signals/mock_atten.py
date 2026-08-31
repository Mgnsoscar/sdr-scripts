#!/usr/bin/env python3
"""
mock_atten — a NO-HARDWARE, one-shot mock of a step attenuator's control task.

It stands in for the real ``ad_usb1ar36g95_atten.py`` so the ACTIVE-component path
can be exercised end to end without any attenuator connected. It parses
``--attenuation`` (dB), logs the value it *would* set, and exits — exactly the
one-shot shape the agent fires when a calibrated ``--power`` is requested on a
transmit task that lists this task as its active component's control task.

Declare it as the ``control.task`` (param ``attenuation``) on a derived plane in the
calibration chain; the agent resolves ``--attenuation`` from this script's schema and
runs ``mock_atten.py --attenuation <N>`` before the transmit task emits.

It prints one machine-readable line for tests/tooling:

    RESULT attenuation_db=<a>

CLI
───
    mock_atten.py --attenuation 60.25
    mock_atten.py --describe-params        # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import logging
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from paramkit import Script

# The device's range/resolution (matches the AD-USB1AR36G95: 0–95 dB, 0.25 dB steps).
MIN_DB = 0.0
MAX_DB = 95.0
STEP_DB = 0.25

log = logging.getLogger("mock_atten")


def build_script() -> Script:
    return (
        Script("Mock step attenuator (NO HARDWARE) — logs the attenuation it would set and "
               "exits. Stands in for a real USB attenuator's control task.")
        .number("-Attenuation", "--attenuation", unit="dB", min=MIN_DB, max=MAX_DB,
                default=0.0,
                help="Attenuation to apply (dB). One-shot: set and exit (the real hardware "
                     "holds the setting).")
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = build_script().parse()
    att = max(MIN_DB, min(MAX_DB, float(args.attenuation)))
    log.info("mock attenuator: set %.2f dB (no hardware) — would hold until changed", att)
    print("RESULT attenuation_db=%.6g" % att)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

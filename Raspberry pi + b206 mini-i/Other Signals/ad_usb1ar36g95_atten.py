#!/usr/bin/env python3
"""
ad_usb1ar36g95_atten — set the attenuation of an AD-USB1AR36G95 1-channel USB step
attenuator (DC–36 GHz, 0–95 dB, 0.25 dB steps), then exit.

This is a ONE-SHOT control task: it opens the device, applies ``--attenuation`` (dB),
optionally reads the device's acknowledgement, and exits — the hardware holds the
setting. Declare it as an ACTIVE component's control task on a derived plane in the
calibration chain (control param: ``attenuation``); the agent then runs this
automatically, just before a calibrated transmit task emits, so the operator only ever
thinks about the delivered ``--power`` and never has to command the attenuator by hand.

    ┌──────────────────────────────────────────────────────────────────────────┐
    │ ⚠ WIRE PROTOCOL — VERIFY AGAINST THE DEVICE MANUAL.                        │
    │ This device is a USB-serial attenuator; the exact command string and       │
    │ serial settings (baud, terminator, whether it echoes) vary by firmware.    │
    │ Everything device-specific is isolated in _SERIAL_* below and              │
    │ _wire_command() — adjust those two if your unit expects a different form.  │
    │ Use --dry-run to see the command without opening the port.                  │
    └──────────────────────────────────────────────────────────────────────────┘

CLI
───
    ad_usb1ar36g95_atten.py --attenuation 60.25
    ad_usb1ar36g95_atten.py --attenuation 30 --port /dev/ttyUSB0
    ad_usb1ar36g95_atten.py --attenuation 12 --dry-run     # print the command, send nothing
    ad_usb1ar36g95_atten.py --describe-params              # paramkit JSON schema for the GUI
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

log = logging.getLogger("ad_usb1ar36g95")

# ── Device characteristics ──────────────────────────────────────────────────────
MIN_DB = 0.0
MAX_DB = 95.0
STEP_DB = 0.25

# ── Serial settings (⚠ verify against the manual) ───────────────────────────────
_SERIAL_BAUD = 115200
_SERIAL_TIMEOUT_S = 1.0
# USB-serial bridges these devices commonly enumerate as. Used only to auto-pick a
# port when --port isn't given; override with --port for anything else.
_KNOWN_USB_SERIAL = (
    (0x0403, None),   # FTDI
    (0x10C4, None),   # Silicon Labs CP210x
    (0x1A86, None),   # QinHeng CH340/CH341
)


def _wire_command(att_db: float) -> bytes:
    """The bytes sent to the attenuator for ``att_db`` dB. ⚠ The default is the common
    ASCII form (the value, two decimals, CR-terminated); adjust to your unit's manual.
    Snapped to the device's 0.25 dB grid so an off-grid request can't be sent."""
    snapped = round(att_db / STEP_DB) * STEP_DB
    return f"{snapped:.2f}\r".encode("ascii")


def _autodetect_port() -> "str | None":
    """First USB-serial port that looks like a known bridge, or None."""
    try:
        from serial.tools import list_ports
    except Exception:  # noqa: BLE001 — pyserial missing; caller reports it
        return None
    for p in list_ports.comports():
        for vid, pid in _KNOWN_USB_SERIAL:
            if p.vid == vid and (pid is None or p.pid == pid):
                return p.device
    return None


def _set_attenuation(att_db: float, port: "str | None") -> None:
    """Open the device and apply ``att_db``. Raises RuntimeError with a clear message on any
    problem (no pyserial, no port, write/ack failure) so the agent logs a useful line."""
    try:
        import serial  # pyserial
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pyserial is not installed (pip install pyserial) — cannot reach the attenuator"
        ) from exc

    dev = port or _autodetect_port()
    if not dev:
        raise RuntimeError(
            "no attenuator serial port found — pass --port (e.g. /dev/ttyUSB0)")

    cmd = _wire_command(att_db)
    with serial.Serial(dev, _SERIAL_BAUD, timeout=_SERIAL_TIMEOUT_S) as ser:
        ser.reset_input_buffer()
        ser.write(cmd)
        ser.flush()
        # Read any acknowledgement the device echoes (best-effort — some are silent).
        try:
            ack = ser.readline().decode("ascii", "replace").strip()
        except Exception:  # noqa: BLE001
            ack = ""
    log.info("  port           : %s @ %d baud", dev, _SERIAL_BAUD)
    log.info("  sent           : %r", cmd)
    if ack:
        log.info("  device ack     : %s", ack)


def build_script() -> Script:
    return (
        Script("AD-USB1AR36G95 1-channel USB step attenuator (0–95 dB, 0.25 dB). One-shot: "
               "set the attenuation and exit. Used as an active-component control task.")
        .number("-Attenuation", "--attenuation", unit="dB", min=MIN_DB, max=MAX_DB,
                default=0.0,
                help="Attenuation to apply (dB), snapped to the 0.25 dB grid. Set and exit; "
                     "the hardware holds the setting.")
        .text("-Port", "--port", default="",
              help="Serial port of the attenuator (e.g. /dev/ttyUSB0). Empty = auto-detect.")
        .flag("-DryRun", "--dry-run",
              help="Print the command that would be sent and exit without opening the port.")
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = build_script().parse()
    att = max(MIN_DB, min(MAX_DB, float(args.attenuation)))
    snapped = round(att / STEP_DB) * STEP_DB

    log.info("── AD-USB1AR36G95 attenuator ───────────────────────────────")
    log.info("  attenuation    : %.2f dB (snapped to %.2f dB grid)", att, snapped)

    if getattr(args, "dry_run", False):
        log.info("  DRY RUN — would send %r; not opening the port.", _wire_command(att))
        print("RESULT attenuation_db=%.6g dry_run=1" % snapped)
        return 0

    try:
        _set_attenuation(att, (getattr(args, "port", "") or None))
    except RuntimeError as exc:
        log.error("  ✗ %s", exc)
        print("RESULT attenuation_db=%.6g error=1" % snapped, file=sys.stderr)
        return 1
    log.info("────────────────────────────────────────────────────────────")
    print("RESULT attenuation_db=%.6g" % snapped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

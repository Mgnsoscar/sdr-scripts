"""
paramkit — friendly, self-documenting command-line parameters for SDR scripts.

Wrap argparse in a small declarative container that also exposes a rich schema
(units, ranges, named presets, choices) so a GUI can build proper input widgets
instead of a bare text box.

    from paramkit import Script

    script = (
        Script("My transmitter")
        .number("-f", "--freq", unit="Hz", min=70e6, max=6e9,
                presets={"WiFi ch1 (2.412 GHz)": 2.412e9}, required=True)
        .number("-g", "--gain", unit="dB", min=0, max=89, default=40)
        .flag("-v", "--verbose")
    )
    args = script.parse()

See paramkit/README.md for the full guide.
"""
from .params import (
    CHOICE,
    FLAG,
    INTEGER,
    NUMBER,
    TEXT,
    Param,
    Preset,
    Script,
    slug,
)
from .live import CTRL_SOCK_ENV, Change, LiveControl

__all__ = [
    "Script",
    "Param",
    "Preset",
    "slug",
    "NUMBER",
    "INTEGER",
    "TEXT",
    "CHOICE",
    "FLAG",
    "LiveControl",
    "Change",
    "CTRL_SOCK_ENV",
]

__version__ = "0.1.0"

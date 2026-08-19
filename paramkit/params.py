"""
paramkit.params — the core of the parameter toolkit.

Define a script's command-line parameters declaratively, get a normal argparse
CLI for free, and get a rich, JSON-serialisable *schema* that a GUI can turn into
proper input widgets (dropdowns, bounded number fields, checkboxes).

Why this exists
───────────────
Raw argparse is fine for a CLI but tells a GUI almost nothing: it can't express a
value's unit, its valid range, or a set of human-named presets. So a GUI can only
offer a bare text box and hope the operator types something sensible.

paramkit adds that missing metadata without giving up the CLI. A frequency
parameter, for example, can declare a valid range (70 MHz–6 GHz), a unit (Hz), and
named presets ("WiFi ch1" → 2.412e9). On the command line you can pass either a raw
number (`--freq 2.412e9`) or a preset key (`--freq wifi_ch1`); in a GUI the presets
become a dropdown and the range bounds a validated number field.

Quick start
───────────
    from paramkit import Script

    FREQS = {
        "WiFi ch1 (2.412 GHz)": 2.412e9,
        "ISM 2.4 GHz":          2.4e9,
        "GPS L1":               1.57542e9,
    }

    script = (
        Script("Linear chirp transmitter.")
        .number("-f", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQS, required=True, help="Center frequency.")
        .number("-g", "--gain", unit="dB", min=0, max=89, default=40,
                help="TX gain.")
        .flag("-v", "--verbose", help="Verbose logging.")
    )

    args = script.parse()        # a normal argparse.Namespace, presets resolved
    print(args.freq, args.gain)  # e.g. 2412000000.0 40.0

Two extra abilities the CLI gains for free:
  * `--describe-params` prints the JSON schema and exits (how a host/GUI can
    discover the parameters at runtime).
  * preset keys and raw values are interchangeable on the command line.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union


# ── Presets ──────────────────────────────────────────────────────────────────

@dataclass
class Preset:
    """
    A named value for a parameter.

    key    : short CLI-friendly token (e.g. "wifi_ch1"). Auto-derived from the
             label if you pass presets as a plain {label: value} mapping.
    label  : human-facing name shown in a GUI (e.g. "WiFi ch1 (2.412 GHz)").
    value  : the actual value the script receives when this preset is chosen.
    description : optional longer explanation for tooltips.
    """
    key: str
    label: str
    value: Any
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"key": self.key, "label": self.label, "value": self.value}
        if self.description:
            d["description"] = self.description
        return d


def slug(text: str) -> str:
    """Turn a human label into a short CLI token: 'WiFi ch1 (2.4G)' → 'wifi_ch1_2_4g'."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower())
    return s.strip("_") or "preset"


# Presets may be given as {label: value}, [(label, value), ...], or [Preset, ...].
PresetsInput = Union[Mapping[str, Any], Sequence[Any], None]


def _normalise_presets(presets: PresetsInput) -> List[Preset]:
    if not presets:
        return []
    out: List[Preset] = []
    seen: set[str] = set()

    def _add(key: str, label: str, value: Any, description: str = "") -> None:
        base, i = key, 2
        while key in seen:                      # keep keys unique
            key = f"{base}_{i}"; i += 1
        seen.add(key)
        out.append(Preset(key=key, label=label, value=value, description=description))

    if isinstance(presets, Mapping):
        for label, value in presets.items():
            _add(slug(label), str(label), value)
    else:
        for item in presets:
            if isinstance(item, Preset):
                _add(item.key or slug(item.label), item.label, item.value, item.description)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                label, value = item
                _add(slug(label), str(label), value)
            else:
                raise TypeError(
                    "each preset must be a Preset, a (label, value) pair, or use a "
                    f"{{label: value}} mapping — got {item!r}"
                )
    return out


# ── Parameter spec ───────────────────────────────────────────────────────────

# The declared kinds. Each maps to a widget hint the GUI can honour.
NUMBER = "number"   # float; range + presets → dropdown-with-custom + bounded field
INTEGER = "integer" # int; same as number but whole values
TEXT = "text"       # free string
CHOICE = "choice"   # one of a fixed set of strings → dropdown
FLAG = "flag"       # boolean switch → checkbox


@dataclass
class Param:
    """One declared parameter. You normally create these via Script's builder
    methods (number/integer/text/choice/flag) rather than directly."""
    name: str                       # dest / canonical name
    flags: List[str]                # CLI flags, e.g. ["-f", "--freq"]; empty = positional
    kind: str                       # one of NUMBER/INTEGER/TEXT/CHOICE/FLAG
    help: str = ""
    unit: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None
    presets: List[Preset] = field(default_factory=list)
    default: Any = None
    required: bool = False
    multiple: bool = False          # accept more than one value (argparse nargs="+")
    live: bool = False              # tunable while the script runs (see Script.live_control)

    @property
    def display_name(self) -> str:
        long = [f for f in self.flags if f.startswith("--")]
        return (long[0] if long else (self.flags[0] if self.flags else self.name)).lstrip("-")

    def to_dict(self) -> Dict[str, Any]:
        """The JSON-serialisable schema entry a GUI consumes."""
        return {
            "name": self.name,
            "flags": list(self.flags),
            "kind": self.kind,
            "help": self.help,
            "unit": self.unit,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
            "presets": [p.to_dict() for p in self.presets],
            "default": self.default,
            "required": self.required,
            "multiple": self.multiple,
            "live": self.live,
        }


# ── Value resolution (preset key OR raw value) + range checking ──────────────

def _make_resolver(param: Param, cast: Callable[[str], Any]) -> Callable[[str], Any]:
    """Build the argparse `type` callable for a numeric parameter.

    Accepts either a preset key/label or a raw number, then enforces min/max.
    Raising argparse.ArgumentTypeError gives a clean CLI error message.
    """
    by_key = {p.key: p.value for p in param.presets}
    by_label = {p.label.lower(): p.value for p in param.presets}
    unit = f" {param.unit}" if param.unit else ""

    def resolve(token: str) -> Any:
        s = token.strip()
        if s in by_key:
            value = by_key[s]
        elif s.lower() in by_label:
            value = by_label[s.lower()]
        else:
            try:
                value = cast(s)
            except (TypeError, ValueError):
                known = ", ".join(by_key) if by_key else "a number"
                raise argparse.ArgumentTypeError(
                    f"'{token}' is not valid for {param.display_name} "
                    f"(expected {known})"
                )
        if param.min is not None and value < param.min:
            raise argparse.ArgumentTypeError(
                f"{param.display_name} {value}{unit} is below the minimum "
                f"{param.min}{unit}"
            )
        if param.max is not None and value > param.max:
            raise argparse.ArgumentTypeError(
                f"{param.display_name} {value}{unit} is above the maximum "
                f"{param.max}{unit}"
            )
        return value

    return resolve


# ── The container ────────────────────────────────────────────────────────────

class Script:
    """
    A friendly, self-documenting wrapper around argparse.

    Build it up with the chainable builder methods, then call parse() in your
    script's main(). The same declaration powers both the CLI and the GUI schema.
    """

    def __init__(self, description: str = "", *, epilog: str = "",
                 add_describe_flag: bool = True):
        self.description = description
        self.epilog = epilog
        self._params: List[Param] = []
        self._by_name: Dict[str, Param] = {}
        self._add_describe_flag = add_describe_flag

    # ── Builder methods ──────────────────────────────────────────────────────

    def _add(self, param: Param) -> "Script":
        if param.name in self._by_name:
            raise ValueError(f"duplicate parameter name: {param.name!r}")
        # Guard against presets that fall outside the declared range — a
        # definition-time mistake we'd rather catch loudly than ship.
        for p in param.presets:
            if isinstance(p.value, (int, float)):
                if param.min is not None and p.value < param.min:
                    raise ValueError(
                        f"preset {p.label!r}={p.value} is below min {param.min} "
                        f"for {param.display_name}")
                if param.max is not None and p.value > param.max:
                    raise ValueError(
                        f"preset {p.label!r}={p.value} is above max {param.max} "
                        f"for {param.display_name}")
        self._params.append(param)
        self._by_name[param.name] = param
        return self

    @staticmethod
    def _derive_name(flags: Sequence[str], name: Optional[str]) -> tuple[str, List[str]]:
        flags = list(flags)
        if name is None:
            longs = [f for f in flags if f.startswith("--")]
            src = longs[0] if longs else (flags[0] if flags else "")
            name = src.lstrip("-").replace("-", "_")
        if not name:
            raise ValueError("a parameter needs at least one flag or an explicit name")
        return name, flags

    def number(self, *flags: str, name: Optional[str] = None, help: str = "",
               unit: str = "", min: Optional[float] = None, max: Optional[float] = None,
               step: Optional[float] = None, presets: PresetsInput = None,
               default: Optional[float] = None, required: bool = False,
               multiple: bool = False, live: bool = False) -> "Script":
        """A floating-point parameter with optional unit, range, and named presets.

        Pass live=True to mark it tunable while the script runs — a GUI can then
        offer a control that applies changes mid-run (the script reads updates via
        Script.live_control)."""
        n, flags = self._derive_name(flags, name)
        return self._add(Param(
            name=n, flags=flags, kind=NUMBER, help=help, unit=unit, min=min, max=max,
            step=step, presets=_normalise_presets(presets), default=default,
            required=required, multiple=multiple, live=live,
        ))

    def integer(self, *flags: str, name: Optional[str] = None, help: str = "",
                unit: str = "", min: Optional[int] = None, max: Optional[int] = None,
                step: Optional[int] = None, presets: PresetsInput = None,
                default: Optional[int] = None, required: bool = False,
                multiple: bool = False, live: bool = False) -> "Script":
        """A whole-number parameter. Like number() but values are ints. See number()
        for the live= flag."""
        n, flags = self._derive_name(flags, name)
        return self._add(Param(
            name=n, flags=flags, kind=INTEGER, help=help, unit=unit, min=min, max=max,
            step=step, presets=_normalise_presets(presets), default=default,
            required=required, multiple=multiple, live=live,
        ))

    def choice(self, *flags: str, options: Sequence[str], name: Optional[str] = None,
               help: str = "", default: Optional[str] = None,
               required: bool = False, live: bool = False) -> "Script":
        """A string parameter restricted to a fixed set of options (a GUI dropdown).
        See number() for the live= flag."""
        n, flags = self._derive_name(flags, name)
        opts = [str(o) for o in options]
        if default is not None and str(default) not in opts:
            raise ValueError(f"default {default!r} is not one of the options {opts}")
        return self._add(Param(
            name=n, flags=flags, kind=CHOICE, help=help, choices=opts,
            default=default, required=required, live=live,
        ))

    def text(self, *flags: str, name: Optional[str] = None, help: str = "",
             default: Optional[str] = None, required: bool = False,
             multiple: bool = False, live: bool = False) -> "Script":
        """A free-form string parameter. See number() for the live= flag."""
        n, flags = self._derive_name(flags, name)
        return self._add(Param(
            name=n, flags=flags, kind=TEXT, help=help, default=default,
            required=required, multiple=multiple, live=live,
        ))

    def flag(self, *flags: str, name: Optional[str] = None, help: str = "",
             default: bool = False, live: bool = False) -> "Script":
        """A boolean on/off switch (a GUI checkbox). Present on the CLI ⇒ True.
        See number() for the live= flag."""
        n, flags = self._derive_name(flags, name)
        return self._add(Param(
            name=n, flags=flags, kind=FLAG, help=help, default=bool(default), live=live,
        ))

    # ── argparse construction ────────────────────────────────────────────────

    def build_parser(self) -> argparse.ArgumentParser:
        """Return a configured argparse.ArgumentParser (useful for testing/help)."""
        parser = argparse.ArgumentParser(
            description=self.description, epilog=self.epilog,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        if self._add_describe_flag:
            parser.add_argument(
                "--describe-params", action="store_true",
                help="print this script's parameter schema as JSON and exit",
            )
        for p in self._params:
            self._add_to_parser(parser, p)
        return parser

    def _add_to_parser(self, parser: argparse.ArgumentParser, p: Param) -> None:
        kwargs: Dict[str, Any] = {"help": self._help_text(p)}
        is_optional = bool(p.flags) and p.flags[0].startswith("-")
        if is_optional:
            kwargs["dest"] = p.name

        if p.kind == FLAG:
            kwargs["action"] = "store_true"
            kwargs["default"] = bool(p.default)
        else:
            if p.kind == NUMBER:
                kwargs["type"] = _make_resolver(p, float)
            elif p.kind == INTEGER:
                kwargs["type"] = _make_resolver(p, lambda s: int(s, 0))
            elif p.kind == CHOICE:
                kwargs["choices"] = p.choices
                kwargs["type"] = str
            else:  # TEXT
                kwargs["type"] = str

            if p.multiple:
                kwargs["nargs"] = "+"
            if p.default is not None:
                kwargs["default"] = p.default
            if is_optional and p.required:
                kwargs["required"] = True
            if p.unit and p.kind in (NUMBER, INTEGER):
                kwargs["metavar"] = p.unit.upper()

        flags = p.flags if p.flags else [p.name]
        parser.add_argument(*flags, **kwargs)

    def _help_text(self, p: Param) -> str:
        bits = [p.help] if p.help else []
        if p.unit:
            bits.append(f"[{p.unit}]")
        if p.min is not None or p.max is not None:
            lo = "" if p.min is None else f"{p.min:g}"
            hi = "" if p.max is None else f"{p.max:g}"
            bits.append(f"(range {lo}..{hi})")
        if p.presets:
            keys = ", ".join(pr.key for pr in p.presets)
            bits.append(f"(presets: {keys})")
        return " ".join(bits)

    # ── Parsing ──────────────────────────────────────────────────────────────

    def parse(self, argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """
        Parse arguments (defaults to sys.argv). Preset keys/labels are resolved to
        their values and ranges are enforced. If --describe-params is present, the
        schema is printed as JSON and the process exits 0 (before any real work).
        """
        parser = self.build_parser()
        raw = list(sys.argv[1:] if argv is None else argv)
        # Handle --describe-params BEFORE parse_args so a GUI can fetch the schema
        # without supplying required arguments (which argparse would otherwise
        # demand first). It's still declared on the parser so it shows in --help.
        if self._add_describe_flag and "--describe-params" in raw:
            print(self.to_json())
            parser.exit(0)
        args = parser.parse_args(raw)
        if hasattr(args, "describe_params"):
            del args.describe_params
        return args

    # ── Live control (retune while running) ───────────────────────────────────

    def live_control(self, args: Any, *, socket_path: Optional[str] = None):
        """Open a control channel for this script's live parameters, seeded from
        the parsed ``args``. Returns a paramkit.live.LiveControl — call drain() in
        your main loop to apply changes on your own thread. With no socket (no
        SDR_CTRL_SOCK and no explicit path) it's an inert no-op, so the same script
        still runs fine on the CLI."""
        from .live import LiveControl
        return LiveControl(self._params, args, socket_path=socket_path)

    # ── Schema (for GUIs / hosts) ────────────────────────────────────────────

    def describe(self) -> Dict[str, Any]:
        """Return the JSON-serialisable schema describing every parameter."""
        return {
            "description": self.description,
            "params": [p.to_dict() for p in self._params],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.describe(), indent=indent)

    @property
    def params(self) -> List[Param]:
        return list(self._params)

# paramkit

Friendly, self-documenting command-line parameters for SDR scripts.

`paramkit` wraps `argparse` in a small declarative container. You still get a
normal CLI, but you *also* get a rich, JSON-serialisable **schema** — units,
value ranges, named presets, choices — so a GUI can build proper input widgets
(dropdowns, bounded number fields, checkboxes) instead of a bare text box.

## Why

Raw `argparse` tells a GUI almost nothing: it can't express that `--freq` is in
Hz, must be between 70 MHz and 6 GHz, or that "WiFi ch1" means `2.412e9`. So a GUI
can only offer a text box. `paramkit` carries that metadata alongside the CLI, so
the *same* declaration drives both.

You don't have to register a task per value: declare a frequency once with named
presets, and the operator can pick "WiFi ch1" from a dropdown *or* type any raw
frequency.

## Quick start

```python
from paramkit import Script

FREQUENCIES = {
    "WiFi ch1 (2.412 GHz)": 2.412e9,
    "ISM 2.4 GHz":          2.4e9,
    "GPS L1 (1.575 GHz)":   1.57542e9,
}

script = (
    Script("Transmit a single tone.")
    .number("-f", "--freq", unit="Hz", min=70e6, max=6e9,
            presets=FREQUENCIES, required=True, help="Center frequency.")
    .number("-g", "--gain", unit="dB", min=0, max=89, default=40, help="TX gain.")
    .number("-d", "--duration", unit="s", min=0, max=3600, default=10)
    .choice("--antenna", options=["TX/RX", "RX2"], default="TX/RX")
    .flag("-v", "--verbose")
)

args = script.parse()          # a normal argparse.Namespace, presets resolved
print(args.freq, args.gain)    # e.g. 2412000000.0 40.0
```

On the command line, a preset key and a raw value are interchangeable:

```
tone_tx.py --freq wifi_ch1_2_412_ghz -g 40      # preset key
tone_tx.py --freq 2.412e9 -g 40                 # raw value
tone_tx.py -f 7e9                               # error: above the maximum 6e9 Hz
tone_tx.py --help
tone_tx.py --describe-params                    # prints the JSON schema, exits
```

A complete runnable example lives in [`examples/tone_tx.py`](../examples/tone_tx.py).

## Builder methods

Every builder takes one or more flags first (e.g. `"-f", "--freq"`), then keyword
options. They return the `Script` so you can chain.

| Method | Widget hint | Key options |
| --- | --- | --- |
| `number(...)`  | bounded float field (+ preset dropdown) | `unit, min, max, step, presets, default, required, multiple` |
| `integer(...)` | bounded int field (+ preset dropdown)   | same as `number` (values are ints; accepts `0x..`/`0o..`) |
| `choice(...)`  | dropdown                                | `options, default, required` |
| `text(...)`    | text field                              | `default, required, multiple` |
| `flag(...)`    | checkbox                                | `default` |

Notes:
- The parameter's name (the `args.` attribute) is derived from the first `--long`
  flag, or pass `name=` explicitly.
- `min`/`max` are enforced at parse time with a clear CLI error.
- A preset whose value falls outside `min`/`max` is rejected when you *define* the
  script — a loud, early failure instead of a confusing one later.

## Presets

Presets are named values. Three input forms are accepted:

```python
.number("-f", "--freq", presets={"WiFi ch1 (2.412 GHz)": 2.412e9})           # mapping
.number("-f", "--freq", presets=[("WiFi ch1", 2.412e9), ("GPS L1", 1.575e9)]) # pairs
.number("-f", "--freq", presets=[Preset("wifi1", "WiFi ch1", 2.412e9)])       # explicit
```

Each preset has:
- **key** — a short CLI token (auto-slugged from the label, e.g. `wifi_ch1_2_412_ghz`),
- **label** — what a GUI shows,
- **value** — what the script receives.

On the CLI you may pass the key *or* the exact label *or* a raw value.

## The schema (for GUIs / hosts)

`script.describe()` returns a dict (and `to_json()` the JSON) shaped like:

```json
{
  "description": "Transmit a single tone.",
  "params": [
    {
      "name": "freq", "flags": ["-f", "--freq"], "kind": "number",
      "unit": "Hz", "min": 70000000.0, "max": 6000000000.0, "step": null,
      "choices": null, "required": true, "default": null, "multiple": false,
      "help": "Center frequency.",
      "presets": [
        {"key": "wifi_ch1_2_412_ghz", "label": "WiFi ch1 (2.412 GHz)", "value": 2412000000.0}
      ]
    }
  ]
}
```

`kind` is one of `number`, `integer`, `text`, `choice`, `flag` — a direct hint for
which widget to render. A host can fetch this at runtime by running the script
with `--describe-params` (it prints the JSON and exits before doing any real
work).

## Live parameters (retune while running)

Mark a parameter `live=True` and a host can change it *while the script runs* —
no restart. It shows up as `"live": true` in the schema, and the script applies
updates from its own main loop, so device access stays single-threaded:

```python
s = (Script("capture")
     .number("-f", "--freq", unit="Hz", min=70e6, max=6e9, default=100e6, live=True)
     .integer("-g", "--gain", unit="dB", min=0, max=49, default=30, live=True))
args = s.parse()
ctrl = s.live_control(args)          # opens the control socket, if the host set one

sdr.set_freq(args.freq); sdr.set_gain(args.gain)
while running:
    for change in ctrl.drain():                    # applied on THIS thread
        if change.name == "freq":
            ctrl.report("freq", sdr.set_freq(change.value))   # report the value
        elif change.name == "gain":                            # the device took
            ctrl.report("gain", sdr.set_gain(change.value))
    process(sdr.read())
```

* `drain()` returns the changes received since the last call (validated against
  the schema — out-of-range values are rejected before they ever reach you) and
  is a no-op when the script is run straight from the CLI, so the same script
  still works without a host.
* `report(name, value)` sends back the value the device actually took (e.g. a
  gain quantised to the nearest step) so the UI reflects reality.

The agent (sdr-agent) provisions a per-run Unix socket via the `SDR_CTRL_SOCK`
env var and exposes `POST /tasks/{name}/params` to push updates to it.

## Status & roadmap

This is the first cut: the library, the frequency example, and tests
(`tests/test_paramkit.py`, run with `python3 tests/test_paramkit.py` — no pytest
needed). Not yet wired into the agent or GUI. Planned next steps:

1. **Agent introspection** — have `GET /scripts/{name}/params` return this richer
   schema (either by running `python <script> --describe-params`, or by extending
   the existing static AST parser in `agent/argspec.py` to recognise paramkit).
2. **GUI widgets** — teach the task editor to render preset dropdowns, bounded
   number fields, and unit labels from the schema.
3. **Packaging/deploy** — ship `paramkit` onto the Pi so `scripts/` can import it.

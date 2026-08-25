# `gps_l1ca_tx.py` — a complete walkthrough

A transmit script that radiates a **GPS L1 C/A** signal from a Raspberry Pi driving
an Ettus B200-mini-family SDR (over UHD / GNU Radio). This document explains, end to
end, how the script is built: the signal it produces and the DSP behind it, how it
plugs into **paramkit** for its CLI/GUI schema and live tuning, and how it plugs into
the **per-unit power calibration** system so you can dial the transmit level in
absolute dBm.

> ⚠ **RF safety / legal.** L1 (1575.42 MHz) is a live GNSS band. Transmit only into a
> shielded / conducted setup (cable + attenuators into a receiver or analyser) that
> you are licensed / authorised to use. Radiating a PRN over the air can jam or spoof
> real GNSS receivers and is illegal in most jurisdictions.

---

## 1. The big picture

The script does four separable things, and the rest of this document takes them one at
a time:

| Stage | What happens | Where in the file |
|-------|--------------|-------------------|
| **Signal design** | Decide what GPS L1 C/A *is* — carrier, code, modulation | constants + `ca_code()` |
| **Baseband synthesis** | Build one seamless, loopable buffer of IQ samples | `build_iq_buffer()` |
| **Radio flowgraph** | Stream that buffer to the SDR at the right rate/gain | `_build_top_block()` |
| **Control surface** | Expose parameters (CLI + GUI), calibrate power, tune live | `build_script()`, `power_map()`, `_apply_live_change()`, `main()` |

Two design goals shape everything:

1. **A Raspberry Pi cannot synthesise GNSS IQ at runtime.** L1 C/A needs a few MHz of
   bandwidth; the M-code sibling needs 40–60 MS/s. Python/NumPy per-sample math can't
   keep up, so the script **precomputes one buffer and loops it** — at runtime the CPU
   only DMAs bytes to USB.
2. **You should set power in dBm, not in the SDR's arbitrary gain number.** That
   requires knowing how gain maps to real output power, which is exactly what the
   calibration system provides (§6).

---

## 2. The signal: GPS L1 C/A

### 2.1 What it is

GPS L1 C/A ("Coarse/Acquisition") is the open civilian GPS signal. Physically it is:

- A **carrier** at **1575.42 MHz** (`CARRIER_HZ`).
- **BPSK-modulated** by a **1023-chip Gold code** (`CODE_LEN = 1023`) clocked at
  **1.023 Mcps** (`CODE_RATE_MCPS`). Each satellite uses a different code, indexed by
  its **PRN** (1–32).
- (In a real satellite, also a 50 bps navigation data bit-stream XOR'd onto the code.
  This script transmits the bare spreading code — a clean, un-modulated-by-data PRN,
  which is what you want for front-end / spectrum / acquisition testing.)

The code repeats every **1 ms** (1023 chips ÷ 1.023 Mcps). Because it's a ±1 sequence
at ~1 MHz, the transmitted spectrum is a **sinc²** centred on L1 with its first nulls
at ±1.023 MHz — the familiar ~2 MHz main lobe.

### 2.2 Gold codes and the two LFSRs

The C/A code is a **Gold code**: the XOR of two 10-stage maximal-length LFSR sequences
(`G1` and `G2`), each 1023 chips long. Gold codes are chosen because different PRNs
have low cross-correlation — receivers can separate satellites sharing the band.

`ca_code(prn)` implements the GPS ICD-200 definition directly (pure Python, no NumPy,
so it runs anywhere):

```python
g1 = [1] * 10          # both registers seeded all-ones
g2 = [1] * 10
ta, tb = G2_TAPS[prn]  # the PRN-specific phase-selector tap pair
for _ in range(CODE_LEN):
    out.append(g1[9] ^ g2[ta - 1] ^ g2[tb - 1])   # chip = G1 out XOR two G2 taps
    fb1 = g1[2] ^ g1[9]                                # G1: x^10 + x^3 + 1
    fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]  # G2: x^10+x^9+x^8+x^6+x^3+x^2+1
    g1 = [fb1] + g1[:9]                                 # shift, feedback into stage 0
    g2 = [fb2] + g2[:9]
```

The satellite-specific part is **`G2_TAPS[prn]`** — the ICD-200 Table 3-Ia phase
selector. Instead of physically delaying G2, the standard picks the modulo-2 sum of
**two G2 stages**; that pair *is* the per-satellite code phase.

### 2.3 Proving it's correct — `--self-test`

`_self_test()` regenerates all 32 PRNs and checks each against two invariants the ICD
guarantees, with **no hardware and no radio stack**:

- The **first 10 chips** as an octal integer match `_FIRST10_OCTAL[prn]` (the ICD's
  published reference column).
- The code is **balanced**: exactly 512 ones out of 1023 (a maximal-length property).

```
$ gps_l1ca_tx.py --self-test
PRN  1: first10=0o1440 expect=0o1440 ones=512 [OK]
...
ALL PRN CHECKS PASSED
```

This is the script's unit test: if the Gold-code generator is ever broken, this fails
loudly before anything touches a radio.

---

## 3. Creating the baseband: one seamless, loopable buffer

Runtime synthesis is off the table (§1), so `build_iq_buffer(prn, chip_rate, samp_rate)`
builds **one buffer that loops with no discontinuity**, written once to a file and
replayed forever.

### 3.1 The seam problem

If you generate exactly one 1023-chip code period, the number of *samples* it occupies
is `samp_rate × 1023 / chip_rate`, which is usually **not an integer**. Loop a
non-integer-length buffer and you get a fractional-sample jump at the wrap — a spectral
splatter every millisecond.

The fix is to tile **just enough** code periods that the total is an exact integer
number of samples. That count is the **denominator of the sample-count fraction in
lowest terms**:

```python
from fractions import Fraction
spp = Fraction(sr * CODE_LEN, cr)   # samples per ONE code period, exact
n_periods = spp.denominator          # tile this many periods → integer samples
n_samples = spp.numerator            # == n_periods × samples-per-period
```

At the default 20.46 MHz over 1.023 Mcps, one period is exactly 20 460 samples
(20.46/1.023 = 20 samples per chip × 1023 chips), so `n_periods = 1`. At an awkward
rate it tiles a few periods; the loop is still seamless.

### 3.2 Chips → samples (zero-order hold)

With the sample count fixed, each output sample takes the value of whatever chip is
active at that instant — an exact integer **zero-order hold**, no floating-point drift:

```python
bipolar = 1.0 - 2.0 * code            # map code bits: 0 → +1, 1 → −1
n = np.arange(n_samples, dtype=np.int64)
chip_idx = (n * cr // sr) % CODE_LEN  # which chip is live at sample n (integer math)
iq = bipolar[chip_idx].astype(np.complex64)   # I = ±1, Q = 0  → real BPSK
```

BPSK here is **real**: the imaginary part stays zero. Amplitude is deliberately *not*
baked into this buffer — it's applied downstream by a `multiply_const` block (§4) so
that RF on/off and the calibration amplitude can change it live without rebuilding.

### 3.3 Where the buffer lives

`main()` writes the buffer to a temp file under **`/dev/shm`** (RAM-backed → fast, and
no SD-card wear on the Pi), falling back to the default temp dir if `/dev/shm` is
absent. An `atexit` hook removes it on exit.

---

## 4. The radio flowgraph

`_build_top_block()` constructs a three-block GNU Radio graph, imported lazily so the
module still loads for `--self-test` / `--describe-params` on a machine with no UHD:

```
file_source(repeat=True) → multiply_const_cc(amplitude) → uhd.usrp_sink
   (the precomputed          (baseband level;              (the SDR)
    IQ, looping)              set live by RF on/off)
```

Three performance levers keep a Pi ahead of the DAC:

1. **Precompute + loop** (§3): the CPU only moves bytes at runtime.
2. **`sc8` over the wire.** `cpu_format="fc32"` but `otw_format="sc8"` halves the USB
   payload. A constant-modulus BPSK PRN loses nothing meaningful at 8-bit I/Q. (`--otw`
   lets you pick `sc16` for more dynamic range if the link can sustain it.)
3. **Silence under load.** UHD console + fastpath logging and GR pref-scanning are
   disabled via env vars set *before* the libraries import (top of file), and nothing
   is printed once `start()` is called — status writes under load themselves *cause*
   underflows.

**1:1 master clock.** `master_clock_rate` is pinned equal to the sample rate, so UHD
runs the AD9361 with no FPGA resampling and no rate coercion — you get exactly the rate
you asked for, which is what keeps samples-per-chip and the loop length exact:

```python
args = f"master_clock_rate={samp_rate_hz:.0f},num_send_frames=512,send_frame_size=16000"
```

The `top_block` also exposes small **live setters** — `set_gain()`, `set_amplitude()`,
`actual_gain()` — that the control loop (§5, §6) calls to retune a running transmission.

---

## 5. How it relates to paramkit

[paramkit](../../paramkit/) is the project's parameter toolkit. One declaration powers
**three** things: the argparse CLI, the JSON schema a GUI renders, and the live-tune
channel. The script never touches argparse directly.

### 5.1 Declaring parameters

`build_script()` builds a `Script` with a fluent builder. Each call adds one typed
parameter with rich metadata (unit, min/max, presets, `live`, `required`):

```python
Script("GPS L1 C/A … transmitter.")
  .integer("-PRN", "--prn", min=1, max=32, default=1, required=True, help="…")
  .number ("-Power", "--power", unit="dBm", min=…, max=…, required=False, live=True, help="…")
  .number ("-Sample-rate", "--samp_rate", unit="MHz", presets=SAMPLE_RATES, …)
  .choice ("-OTW-format", "--otw", options=["sc8","sc16"], default="sc8", …)
  .choice ("-RF", "--rf", options=["on","off"], default="on", live=True, …)
  .number ("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB, required=False, live=True, …)
```

- **`unit=`, `min=`, `max=`, `presets=`** give a GUI enough to render a proper widget
  (a spinbox with bounds, a dropdown of named sample rates) instead of a raw text box.
- **`live=True`** marks a parameter as retunable while the task runs (§5.3). Here:
  `--power`, `--rf`, `--gain`.
- **`required=False` on `--power`** and **no default on `--gain`** are load-bearing for
  the power modes — see §6.4.

### 5.2 Two ways the schema is read

There are two independent paths that turn this declaration into a schema, and it's
worth knowing which is which:

- **Runtime** — `gps_l1ca_tx.py --describe-params` prints the schema as JSON by
  actually importing and calling `build_script()`. Because the code runs, `--power`'s
  bounds reflect the *live* `power_map()` (including any injected calibration).
- **Static** — the agent's and client's `argspec.py` parse the **source with `ast`**,
  without executing it, to offer the schema for the GUI even when the script can't be
  run (e.g. in the Library, off-unit). Static extraction can't evaluate
  `round(power_map().min_power_dbm, 2)`, so it reports `min/max = None` and the real
  bounds are filled in later from the unit's `/calibration` (§6.5). Static extraction
  **also reads the `CAL_SIGNAL_ID` module constant** and surfaces it as the schema's
  `calibration_signal` — that's how the client knows to wire the task's env (§6.5).

### 5.3 Live tuning

`main()` opens a live channel with `script.live_control(args)` and, in its loop,
drains change events and dispatches them:

```python
ctrl = script.live_control(args)
while not stop.is_set():
    for change in ctrl.drain():
        _apply_live_change(tb, ctrl, state, pmap, amplitude, change.name, change.value)
    time.sleep(0.1)
```

Only `live=True` params arrive here. `ctrl.report(name, value)` sends back the value
the device *actually* took (e.g. the power for the gain the SDR quantised to), which
the GUI's live-tune dialog displays. With no control socket set (plain CLI), the
channel is an inert no-op, so the same script still runs fine from a shell.

---

## 6. How it relates to calibration

This is the part that makes `--power` mean *real* dBm. Full design lives in the
**sdr-agent** repo at `docs/calibration.md`.

### 6.1 The problem calibration solves

The SDR's `set_gain(x)` takes an abstract 0–89.75 dB number. What that produces at your
antenna depends on *this unit's* hardware, cabling, and amplifier — properties that
can't be baked into a script shared across units. Calibration is how each unit teaches
the script its own gain→power reality.

### 6.2 Two sources of truth: `PowerMap`

The script never converts dBm↔gain by hand. It asks a **`PowerMap`** (from
[`paramkit/calkit.py`](../../paramkit/calkit.py)), obtained once and cached:

```python
def power_map() -> PowerMap:
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
    return _PMAP
```

`PowerMap.load()` returns one of two maps:

- **Measured (preferred).** If the environment variable `SDR_CALIBRATION_FILE` points
  at an artifact (the agent injects it — §6.5), the map is the unit's **measured
  gain→power curve**, interpolated piecewise-linearly, reading at the unit's real
  **operating plane** (e.g. EIRP at the antenna).
- **Baked fallback.** With no artifact, it builds a straight slope-1 line from the
  **USER CALIBRATION constants** at the top of the file (`OUTPUT_POWER_DBM`,
  `GAIN_AT_MAX_DB`, cable/amp offsets, `AMPLITUDE`). This is byte-identical to the
  script's original single-anchor behaviour, so an uncalibrated unit works exactly as
  before.

Either way the script calls the same two methods:

- `pmap.gain_for_power(dbm)` → the SDR gain to command (clamped to the unit's gain
  limits — it never extrapolates past the safety ceiling).
- `pmap.power_for_gain(gain)` → the delivered power for a gain the SDR actually took
  (used to report back what really happened after quantisation).

### 6.3 Opting in: `CAL_SIGNAL_ID`

```python
CAL_SIGNAL_ID = "gps_l1_ca"
```

This stable id is the signal's key into a unit's calibration document. A **task** opts
in by setting the environment variable `SDR_CAL_SIGNAL_ID` to this value; the agent then
resolves that signal's calibration for the unit and injects the artifact. The value is
also what the FleetView task editor stamps automatically when it sees the script
declares a `calibration_signal` (§6.5) — so you rarely set it by hand.

### 6.4 `--power` vs `--gain`: absolute vs relative

The script offers **two power modes**, and the parameter declarations encode the rule
"relative wins":

| | `--power` (absolute) | `--gain` (relative) |
|---|---|---|
| Units | dBm at the operating plane | raw SDR TX gain, dB |
| Bounds | from the active `PowerMap` | 0 … `HW_MAX_GAIN_DB` |
| Declared | `required=False`, has a default | **no default** |
| Meaning of presence | the normal path | its presence **selects relative mode and overrides `--power`** |

`main()` resolves them:

```python
gain_cal = getattr(args, "gain", None)
gain_db = float(gain_cal) if gain_cal is not None else pmap.gain_for_power(args.power)
```

So `--gain 60` commands gain 60 directly (bypassing dBm), while `--power -30` maps
−30 dBm through the calibration to a gain. `--gain` doubles as the **calibration knob**:
sweep it while measuring output on a spectrum analyser to fill in the baked constants,
or to build the measured curve you upload to the unit.

### 6.5 The full chain: from the file to a slider in FleetView

Putting §5 and §6 together, here's how a request for absolute power actually flows:

```
Script source ──(static ast parse)──► calibration_signal = "gps_l1_ca"
                                              │
FleetView task editor ── stamps env: SDR_CAL_SIGNAL_ID=gps_l1_ca ──► tasks.yaml
                                              │
Run / Live-tune / Sequence view ── reads env, calls GET /calibration ──► agent
                                              │
Agent resolves this unit's measured curve, writes artifact, sets
SDR_CALIBRATION_FILE ──► the running script's PowerMap.load() picks it up
                                              │
--power bounds + slider now read in the unit's real dBm (e.g. EIRP)
```

If the unit is **uncalibrated** for this signal, `/calibration` has no bounds, the GUI
offers **relative power only** (`--gain`), and the script falls back to the baked map —
nothing breaks, you just don't get calibrated dBm until the unit is measured.

### 6.6 RF on/off, with staged levels

`--rf on|off` (live) mutes and restores the output. The subtlety: a power or gain edit
made **while RF is OFF** must not silently reach the air. `_apply_live_change()` keeps a
`state` dict — `{"rf_on": bool, "gain": float}` — so edits are *staged* into
`state["gain"]` and only pushed to the hardware when RF is on:

```python
if name == "power":
    state["gain"] = pmap.gain_for_power(float(value))   # stage
    if state["rf_on"]:
        tb.set_gain(state["gain"])                       # …apply only if live
        ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain()), 2))
elif name == "rf":
    if on:
        tb.set_amplitude(amplitude); tb.set_gain(state["gain"])  # restore staged level
    else:
        tb.set_gain(0.0); tb.set_amplitude(0.0)                  # mute gain AND baseband
```

Muting drops **both** the SDR gain and the baseband amplitude to zero (belt and
braces); turning RF back on restores the staged gain and the calibration amplitude.

---

## 7. CLI reference

```
gps_l1ca_tx.py --prn 5 --power -30 --samp_rate 20.46   # absolute dBm (calibrated)
gps_l1ca_tx.py --prn 5 --gain 60                       # relative: raw SDR gain
gps_l1ca_tx.py --self-test                             # verify Gold codes, no hardware
gps_l1ca_tx.py --describe-params                       # paramkit JSON schema for the GUI
```

| Flag | Meaning | Live? |
|------|---------|:-----:|
| `--prn` | GPS satellite PRN / Gold-code index, 1–32 | — |
| `--power` | Absolute output power (dBm) at the operating plane | ✓ |
| `--gain` | Raw SDR TX gain (dB); overrides `--power` (relative mode) | ✓ |
| `--samp_rate` | Host/DAC sample rate (MHz); master clock pinned equal | — |
| `--otw` | Over-the-wire sample format (`sc8` / `sc16`) | — |
| `--rf` | RF output on/off | ✓ |

---

## 8. Calibrating a unit (workflow)

1. Deploy the script and create a task for it; the task editor stamps
   `SDR_CAL_SIGNAL_ID=gps_l1_ca` automatically.
2. Expose `--gain` and, on a spectrum analyser (conducted, into attenuators), sweep the
   raw gain, recording **gain → measured power** at your operating plane.
3. Enter those points in the unit's **Calibration** panel (FleetView) as the measured
   curve for signal `gps_l1_ca`, with the plane topology and safety ceilings. The agent
   validates it before storing.
4. From then on, `--power` on this unit reads and is bounded in real dBm; the GUI offers
   the absolute/relative toggle. Uncalibrated units keep working on the baked fallback.

---

## 9. File map

| Symbol | Role |
|--------|------|
| `CARRIER_HZ`, `CODE_RATE_MCPS`, `CODE_LEN` | fixed L1 C/A signal constants |
| `G2_TAPS`, `_FIRST10_OCTAL` | ICD-200 per-PRN phase selectors + self-test reference |
| `ca_code(prn)` | the two-LFSR Gold-code generator |
| `_self_test()` | verify all 32 PRNs against the ICD, no hardware |
| `build_iq_buffer()` | seamless, loopable BPSK baseband buffer |
| `_build_top_block()` | the GNU Radio flowgraph |
| `CAL_SIGNAL_ID`, `power_map()` | calibration opt-in id + active `PowerMap` |
| `build_script()` | the paramkit parameter declaration (CLI + GUI schema) |
| `_apply_live_change()` | live-tune dispatch (power/gain/RF, with staging) |
| `main()` | wire it all together, stream, and drain live changes |

# sdr-scripts — Claude working notes

The **transmit scripts** the SDR agent runs on each box — GNSS/PRN generators and test signals
for two platforms: **`Ettus x410/`** (a channel-based engine: `x410_engine.py` +
`*_channel.py`) and **`Raspberry pi + b206 mini-i/`** (standalone `*_tx.py` scripts, grouped by
constellation: `PRN GPS`, `PRN Galileo`, `PRN BeiDou`, `PRN GLONASS`, `Other Signals`). Part of a
three-repo system: **`sdr-agent`** (runs these + resolves calibration), **`sdr-client`** (GUI),
**`sdr-scripts`** (this).

## Environment + tests
Scripts import `paramkit` from **`sdr-agent`**, so put it on `PYTHONPATH`:
```bash
pip3 install numpy pytest
PYTHONPATH=/path/to/sdr-agent python3 -m pytest -q          # ~10 tests (no radio needed)
# a generator's built-in spectral self-test (no hardware):
PYTHONPATH=/path/to/sdr-agent python3 "Raspberry pi + b206 mini-i/PRN GPS/gps_ca_code_1.023Mcps.py" --self-test
```
The tests drive the mock scripts (`mock_sdr_tx.py`, `mock_atten.py`) against the agent's resolver
to check the calibrated-power path end-to-end without hardware.

## Calibration hooks a script declares (read by the agent + surfaced in the client)
- `SDR_CAL_SIGNAL_ID` (env) — which calibration signal this task belongs to. The client scopes a
  signal's law picker by this id (one signal's laws must not appear for another).
- `CAL_POWER_LAWS` — power-quantity conversion **laws** the signal offers (e.g. spectral density
  → full-bandwidth power), each with `in`/`out` families (`abs`/`density`) and an affine
  log10 form. A **limiting** law must return dBm (`out: abs`). Shared evaluation lives in
  `sdr-agent/paramkit/power_law.py` (mirrored to `sdr-client/state/power_law.py`).
  - `restates_measurement: True` (optional, per law) — this law RE-EXPRESSES the measured
    reading itself, not a distinct quantity (e.g. a chirp's live spectral density restating the
    density measured at a fixed reference sweep). The Run/tune form then drops the raw measured
    quantity from the operator's "control in" choices and offers the live restatement instead —
    so a bandwidth-frozen measured density doesn't sit confusingly beside its live twin. It is an
    explicit opt-out (never inferred from unit/family), so a same-unit but genuinely different
    reading (main-lobe vs total-in-band power, both dBm) is unaffected. Any extra law key rides
    through the agent (`argspec` copies laws verbatim) to the client; no agent bump needed.
- `--self-test` — a no-hardware spectral-density check some generators implement.

## Current state — mock PRN + CW for the headless test unit: COMPLETE (branch `claude/hold-step-phase-0-wwwxf7`, cross-repo)
Two new NO-HARDWARE mock transmitters so the local, headless integration unit (sdr-agent
`deploy/run_local.sh`) has one armable mock per signal family — a **PRN**, a **chirp** and a **CW** —
and nothing else. Both mirror the real script's parameter schema, `CAL_SIGNAL_ID`, `CAL_FREQ_PARAM`
and `CAL_POWER_LAWS` verbatim (a `FakeRadio` LOGS the SDR gain it *would* command; no UHD/GNU Radio),
exactly like the pre-existing `mock_fm_chirp_tx.py`, so the client renders the SAME power card and
`--power` folds through `PowerMap` at the same math:
- **`PRN GPS/mock_gps_ca_code_1.023Mcps_tx.py`** — mock of `gps_ca_code_1.023Mcps.py`. Carries the C/A
  spectral-density surface: `--power`/`--gain`/`--freq`/`--prn`/`--sidelobes` + the derived
  `passband_bw_mhz` and the HIDDEN `enbw_mhz` table, and the `full_power` (keyed on `enbw_mhz`) +
  `main_lobe_power` laws. The sinc² power-fraction math (`enbw_mhz(n)`) is the SAME pure-Python code
  the real script runs (Gold code / IQ loop / filter DSP / GNU Radio dropped); `--self-test` asserts
  the baked enbw table still matches `enbw_mhz()` so they can't drift. Live `--sidelobes` re-maps a
  held full/absolute power.
- **`Other Signals/mock_cw_tx.py`** — mock of `cw_tx.py`. A single dBm quantity (no power laws):
  `--freq`/`--power`/`--rf`/`--gain`, calibrated dBm folded at the live carrier.
Both add `--once` / `--self-test` / `--make-sample-calibration` (like `mock_fm_chirp_tx.py`). The
generic `mock_tx.py`/`mock_sdr_tx.py` (signal id `mock`) stay — they're test infrastructure
(`tests/test_active_power_combination.py`) — but the `mock` SIGNAL is dropped from the sample unit.
Tests: `tests/test_mock_gps_ca_power_quantities.py`, `tests/test_mock_cw.py` (each drives the mock
against the resolver + a drift guard: mock argspec surface == the real script's). Agent side (sample
calibration = the three signals, `run_local.sh` task wiring): see `sdr-agent` CLAUDE.md +
`docs/local-integration-run.md`.

## Current state — L2C + L5 sidelobes slider: COMPLETE (branch `claude/l1c-sidelobes-slider`)
`gps_l2c_tx.py` + `gps_l5_tx.py` `--sidelobes` render as a SLIDER now: dropped `presets=
SIDELOBE_PRESETS` (a numeric field with presets renders as a preset DROPDOWN; with none it's a
spinbox + range rail, like the C/A scripts) and removed the now-unused `SIDELOBE_PRESETS` dict.
These are BPSK (nulls at the chip rate — L2C 1.023 MHz, L5 10.23 MHz), so a sidelobe count already
means WHOLE sidelobes (the L1C half-cut issue was BOC-only); only the widget changed — no physics,
enbw, label, or max change (L2C max 28, L5 max 2). MCode keeps its preset dropdown (not requested).
Scripts-only; the client already renders a presetless integer as a slider. Tests:
`tests/test_gps_power_quantities.py` (`test_l2c_l5_sidelobes_render_as_a_slider`).

## Current state — L1C sidelobes: whole-sidelobe slider + Tune-form bandwidth readout: COMPLETE (branch `claude/l1c-sidelobes-slider`, cross-repo)
`gps_l1c_tx.py` `--sidelobes` reworked so a count means WHOLE sidelobes each side (never a half-cut
outer lobe), rendered as a SLIDER, capped at 13, with the operator's requested per-count labels.
- **Whole sidelobes:** L1C's PSD ZEROS sit at the EVEN multiples of 1.023 MHz (±2.046, ±4.092, …);
  the odd multiples are lobe PEAKS. The old edge `(n+2)·1.023` landed on a peak for odd n (cutting
  the outer lobe in half). New edge = `(n+1)·2.046 MHz` (`SIDELOBE_STEP_HZ = 2·L1C_NULL_HZ`), the
  (n+1)th even null — one whole sidelobe per step. `filter_buffer`/`finfo`/`regenerate` + the
  `--self-test` enbw check all use it; the enbw table was recomputed for the new edges (14 entries,
  0..13) and re-derived in `--self-test` (< 1e-3). `main_lobe_power` k (62.2246) + `full(0)==core`
  unchanged (n=0 is still the ±2.046 core). `full_power` `rep` → enbw_mhz(5) = 2.066514.
- **Slider:** dropped `presets=SIDELOBE_PRESETS` — a numeric field with presets renders as a preset
  DROPDOWN in the client (`param_form._widget_for`); with no presets it's a spinbox + range rail
  (the slider), exactly like the C/A scripts. `MAX_SIDELOBES=13` (±28.64 MHz), `DEFAULT=5`.
- **Labels** on `passband_bw_mhz` (formula `linear [sidelobes, 4.092, 4.092]` = 4.092·n + 4.092 MHz):
  the exact per-count text — `BOC(1,1) core` at 0; `+ N`; `+ BOC(6,1) core` once fully contained
  (n≥3, edge ±8.18 > the ±7.16 lobe edge); `+ 1` for the BOC(6,1) first sidelobe (n≥9, ±18.41 MHz).
- **Tune-form bandwidth readout (client):** `sdr-client/ui/live_tune_dialog.py` `_prepare_specs` now
  RENDERS a visible derived field whose formula reads only LIVE knobs (like `passband_bw_mhz` off the
  live `--sidelobes`) read-only, instead of routing every non-live spec to fold context — so the
  bandwidth tracks the slider while retuning. Hidden derived fields (a law's `enbw_mhz` key) stay
  context. Only the RPi `gps_l1c_tx.py` carries this surface (the Ettus channel + fixed-bw variants
  don't). Tests: scripts `tests/test_gps_power_quantities.py` (slider/no-presets, labels); client
  `tests/test_live_tune_power.py` (visible derived readout renders + tracks the live source).

## Current state — chirp measured density re-anchored to dBm/Hz: COMPLETE (branch `claude/chirp-density-dbm-hz`)
The FM-chirp/sweep signal's MEASURED spectral density is now anchored in **dBm/Hz** (per Hz — the
operator enters the analyzer's dBm/Hz reading directly), matching the GPS scripts' convention;
previously the `CAL_POWER_LAWS` assumed the measured density was dBm/MHz, so entering a dBm/Hz
measurement gave total-power / ceiling readings 60 dB off. Scripts-only (`fm_chirp_tx.py` +
`mock_fm_chirp_tx.py`) — the client/agent read the measured unit + laws from the doc/argspec and
were already generic. Changes to `CAL_POWER_LAWS`: (1) `psd_live` (the default control quantity)
unit `dBm/MHz` → `dBm/Hz`, same live restatement (`coeff −10, ref 10`); (2) `fbw_power` (total
power, dBm) `k` `10.0` → `70.0` (= 60 + 10·log10(CAL_MEAS_BW_MHZ): total = density(dBm/Hz) +
10·log10(10 MHz in Hz), bandwidth-invariant); (3) the old per-Hz secondary view `psd_hz` (k −60) is
DROPPED — only the two quantities remain, the live density (`dBm/Hz`) and total power (`dBm`); no
redundant dBm/MHz view (owner decision, matching the GPS scripts' density + total-power pattern).
The mock's `--make-sample-calibration` + the guard test re-express the sample curve per
Hz (density = gain − 210, was gain − 150 dBm/MHz; local `fbw` k → 70) so the physical unit is
unchanged, just relabeled. Docs: agent `docs/calibration-v2.md` §13 example now says dBm/Hz. Tests:
`tests/test_mock_chirp_power_quantities.py` (base density maps to the curve gain; total power
bw-invariant at base + 70; a held per-Hz density needs more gain as the sweep widens; drift guard
mock ≡ real laws). The fixed-bandwidth `fm_chirp_<n>MHz_tx.py` variants + the Ettus channel carry no
`CAL_POWER_LAWS`, so they're untouched.

## Current state — GPS PRN spectral-density calibration: COMPLETE (branch `claude/gps-calibration`)
The six remaining `PRN GPS` scripts got the C/A spectral-density calibration surface (measured =
peak PSD in **dBm/Hz**; `CAL_POWER_LAWS` convert it to absolute-power quantities the operator picks
for `--power`). Scripts-only — laws ride through `argspec` verbatim; no agent/client/capability change.
**Owner decided (this revision): NO carrier/total-signal quantity on any signal.** Every script offers
exactly the measured PSD (dBm/Hz) + `main_lobe_power` + `full_power` — nothing else.
- **Filtered BPSK — `gps_l5` (Rc 10.23), `gps_l2c` (Rc 1.023):** the always-on-filter C/A treatment
  (ported verbatim from `gps_ca_code_*.py`). The digital passband filter is ALWAYS on (fixed 0.05 MHz
  transition; the `--filter on/off` and `--transition` knobs are GONE), width = `--sidelobes`.
  `main_lobe_power` (k = 10·log10(Rc·I_ML): 69.654784 / 59.654784) is a CONSTANT; `full_power`
  (k = 60 + 10·log10(`enbw_mhz`)) is KEYED on the filter's equivalent-noise bandwidth (a hidden
  `enbw_mhz` derived field, `{"table": ["sidelobes", …]}` — the 10.23 / 1.023 enbw tables) so the
  full-power reading + its amp-limit cap TRACK the live filter; `pwr_params()` supplies the live
  `enbw_mhz`, and a live `--sidelobes` change re-maps a held `--power`. `full_power(0 sidelobes)`
  == `main_lobe_power` (passband == main lobe). `CAL_FREQ_PARAM="freq"`.
- **Streamed BPSK — `gps_l1p`/`gps_l2p` (Rc 10.23, streamed/filterless, fixed carrier → no
  `CAL_FREQ_PARAM`):** no filter, so `full_power` is the fixed TOTAL signal power (k = 10·log10(Rc)
  = 70.098756, ≈ +0.444 dB above the main lobe) — a conservative, bandwidth-independent amp-limit
  reading. `main_lobe_power` k = 69.654784.
- **BOC — `MCode` BOC(10,5), `gps_l1c` TMBOC (BOC(1,1) core):** now the SAME always-on-filter,
  live-tracking treatment as the BPSK filtered signals. Both dropped `--filter on/off` /
  `--transition` (and M-code's continuous `--passband`) and expose a discrete `--sidelobes` count
  (0 = the main lobe(s); n = keep n further spectral-null steps), the passband edge snapping to the
  BOC PSD nulls: M-code `--sidelobes` 0..3 → edge ±(n+3)·5.115 MHz (0 = both split lobes ±15.345,
  3 = ±30.69 = Fs/2); L1C `--sidelobes` 0..28 → edge ±(n+2)·1.023 MHz (0 = the BOC(1,1) core ±2.046,
  5 = full TMBOC ±7.16, 28 = ±30.69). `main_lobe_power` stays a CONSTANT (both main lobes /
  the BOC(1,1) core, k = 69.5073 / 62.2246); `full_power` is now KEYED on an `enbw_mhz` table (∫ the
  sine-BOC PSD out to the live edge, baked literal + re-derived in `--self-test`) so it tracks
  `--sidelobes` and re-maps a held `--power`, exactly like the BPSK path. `full_power(0)` ==
  `main_lobe_power` for L1C (edge = the core); for M-code it sits ~0.39 dB above (the ±15.345 lowpass
  also passes the low-power DC gap between the split lobes). No carrier quantity.
- Every constant is baked as a literal AND re-derived in each script's `--self-test` (∫sinc² for
  BPSK, ∫BOC for BOC) so it can't silently drift; the filtered scripts' baked `enbw_mhz` tables are
  re-derived from the PSD in `--self-test` (< 1e-3 MHz) so they can't drift from the runtime. Guard
  test: `tests/test_gps_power_quantities.py` — argspec extracts each surface; laws evaluate via the
  real `paramkit.power_law`; asserts NO `carrier_power` anywhere; every FILTERED signal's
  `full_power` keyed on `enbw_mhz` + monotonic + meets the main lobe(s) at 0 sidelobes (M-code just
  above, DC gap); streamed-BPSK full = main + 0.444 dB.
- Measurement is **dBm/Hz** everywhere (per Hz, not per MHz), per the owner's request.

Prior work this branch (packaging-standalone base): `Raspberry pi + b206 mini-i/Other Signals/mock_fm_chirp_tx.py` — a
NO-HARDWARE stand-in for `fm_chirp_tx.py`. Same calibration surface (identical param schema,
`CAL_SIGNAL_ID="fm_chirp"`, `CAL_FREQ_PARAM`, `CAL_POWER_LAWS`), so the client renders the SAME
power card (density / total-power / dBm-per-Hz) and DEPENDS ON row and drives it like the real
chirp; but it imports no UHD/GNU Radio and a `FakeRadio` only LOGS the SDR gain it would command
NO-HARDWARE stand-in for `fm_chirp_tx.py`. Same calibration surface (identical param schema,
`CAL_SIGNAL_ID="fm_chirp"`, `CAL_FREQ_PARAM`, `CAL_POWER_LAWS`), so the client renders the SAME
power card (density / total-power / dBm-per-Hz) and DEPENDS ON row and drives it like the real
chirp; but it imports no UHD/GNU Radio and a `FakeRadio` only LOGS the SDR gain it would command
(mirrors `mock_tx.py`/`mock_sdr_tx.py`). `--power` maps to gain via `PowerMap.gain_for_power`
folded at the carrier + sweep bw, exactly as the real chirp — so the gain it prints is the ground
truth for "did my power quantity map right". Runs under the agent (`SDR_CALIBRATION_FILE`
injected) or standalone (`--calibration <artifact>`, `--make-sample-calibration <out>` builds+
resolves a representative density calibration — needs sdr-agent on PYTHONPATH). Extra flags:
`--once` (print gain + a `RESULT gain_db=… power_dbm=… source=…` line and exit), `--self-test`.
Because the client reads params via a STATIC AST reader (`agent/argspec.py`), the schema + CAL
constants live verbatim in the file (an import wouldn't be seen); `tests/test_mock_chirp_power_
quantities.py` guards against drift from `fm_chirp_tx.py` and checks the gain maps right across
power quantities (total power bw-invariant; a held density needs more gain as the sweep widens).

Prior work this branch: `fm_chirp_tx.py`'s `band_span` derived field (start/stop mode) declares
`provides="bw"` — so in start/stop mode the client folds the calibration power laws at the actual
sweep span (stop − start), not the stale hidden `--bw`. The runtime transmit fold was already
correct (`resolve_band` → `sweep_bw_hz`); this fixed only the client display fold. `provides` is a
new `paramkit` `.derived()` kwarg (the bandwidth analogue of `is_freq`), extracted by the
drift-guarded `argspec` and honored by `sdr-client` `param_form._live_params`.

Prior work this branch: `fm_chirp_tx.py` dropped the `--filter` / `--passband` / `--transition`
parameters — the digital passband filter is now ALWAYS on, its passband always equals the sweep
bandwidth (tracks `--bw`, the filter passes ±bw/2), with a fixed `FILTER_TRANSITION_MHZ = 0.05`
skirt. `make_current` always filters; live tuning drops those knobs (a `--bw` change re-derives the
passband automatically). `--self-test` still exercises the filter internally.

Prior work this branch: `fm_chirp_tx.py` marks its two spectral-density laws (`psd_live`,
`psd_hz`) with `restates_measurement: True` and leads with `psd_live`, so the Run/tune form drops
the raw fixed-bandwidth measured density (a bw-invariant "total power − 10 dB" in disguise) from
the "control in" picker and offers the live density + total power instead — fixing a two-densities
confusion. Total power (`fbw_power`) is a distinct reading and stays on offer. Honored by
`sdr-client/ui/param_form.py` (`_power_views`); no agent change (argspec passes law keys through).

Prior work this branch: the GPS C/A transmit scripts were consolidated to exactly two, named by
chip rate, with generic (band-agnostic) calibration ids and both L1+L2 carrier presets —
`Raspberry pi + b206 mini-i/PRN GPS/gps_ca_code_1.023Mcps.py` and `gps_ca_code_10.23Mcps.py`
(the 10.23 script got the full spectral-density calibration treatment: enbw table, max sidelobes,
`--self-test`). The broader per-signal calibration-UI redesign it supports is complete across the
three repos — see `sdr-client/docs/calibration-ui-redesign.md`.

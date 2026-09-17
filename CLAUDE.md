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

## Current state — BOC Enveloped Sweep (BOC(10,5) / M-code shaped): COMPLETE (branch `claude/system-familiarization-f5mezz`, scripts-only)
New `Other Signals/boc_enveloped_sweep_tx.py` — the Enveloped Sweep with the shape of a sine-phased
**BOC(10,5)** (GPS M-code) spectrum instead of a sinc². Same dwell-shaping engine, different target
`S(f)`: a constant-envelope swept tone whose averaged PSD reproduces M-code's SPLIT spectrum — two
lobes at ±10.23 MHz with a gap at the centre — by dwell time (the tone lingers on the two main
lobes, rushes through the nulls + centre gap). Fixed BOC(10,5) rates (like `PRN GPS/MCode.py`):
subcarrier fsub 10.23 MHz, code rate fc 5.115 MHz.
- **`boc_psd(f)`** — the sine-BOC(10,5) PSD `∝ [sin(4a)sin(a)/(πf·cos(a))]²`, a=πf/(2·fsub), made
  singularity-free via `sin(4a)/cos(a)=4·sin(a)(1−2sin²a)` so the lobe peaks (0/0 at ±fsub) are
  finite; a null at DC. Verified: peak ±10.23, deep nulls at 5.115 / 15.345, gap at 0.
- **`_boc_dwell_freq`** — inverse-CDF of `boc_psd` (out-and-back, zero-mean → seamless loop), the
  same core as the sinc² script. `--sidelobes` 0..3 keeps the two main lobes + n further BOC
  null-steps → ±(sidelobes+3)·5.115 MHz (0 → ±15.345, 3 → ±30.69 = Fs/2), matching MCode's null
  snapping. **Confined to ±roam** by construction (`--self-test` asserts `max|f| ≤ roam`; measured
  it stops just inside the outermost null). No `--chip-rate` (BOC rates are fixed); params are
  `--freq`/`--power`/`--gain`/`--sidelobes`/`--inner-sidelobes`/`--rf`, all live.
- **`--inner-sidelobes` (0..1) — the INNER filter (owner ask "filter away the inner sidelobes").**
  The passband gets an inner edge as well as the outer one: `filter_buffer(inner_hz=)` becomes a
  BANDPASS (`lowpass(outer) − lowpass(inner)`) that notches the low-power centre GAP between the two
  split lobes (|f| < ±5.115 MHz), leaving a clean split spectrum (`--self-test`: centre −25 → −163 dB,
  lobes intact). 0 = keep the centre (a plain lowpass); 1 = notch it (the 1 inner null-step; the main
  lobe starts at ±5.115 so ≥2 would eat it). Only the FILTER + the delivered-total READING change —
  the trajectory, the main-lobes fraction and the SDR gain are all untouched, so a live change just
  rebuilds + swaps (no gain re-map).
  **Calibration under the notch — `full_power` is now EXACT (deliver_frac).** `main_lobe_power` (the
  two lobes) was always notch-independent; `full_power` used to stay the PRE-notch constant-envelope
  total (density + 70), over-reading by the discarded ~0.39 dB gap. Now a hidden **`deliver_frac`**
  derived field (fraction of in-band power that SURVIVES the notch: 1.0 with the centre kept, else
  `1 − centre-gap/total`) keys the `full_power` law, so full = `density + 70 + 10·log10(deliver_frac)`
  = the DELIVERED total, and a `--power` set in it is delivered exactly. **`deliver_frac` is keyed on
  `--inner-sidelobes` ALONE** — a plain `table` over an existing field, exactly like `main_lobe_frac`,
  so NO agent/client/version change (a true 2-param sidelobes×notch fold would need a cross-repo
  `eval_formula` recursion; not worth it). The centre-gap fraction of the total barely moves with
  `--sidelobes` (0.0854 → 0.0810 across 0..3, a **0.02 dB** spread « the 0.25 dB gain grid), and at
  `--sidelobes 0` the notched signal IS exactly the two lobes, so `deliver_frac(notch, 0 sidelobes)
  == main_lobe_frac(0)` (`0.914610`): full power is **EXACT at sidelobes 0** (full == main-lobes there)
  and within ~0.02 dB above. No-notch (`deliver_frac 1.0`) is byte-identical to before (density + 70).
  `--self-test` re-derives the notch value from the BOC integral so the baked constant can't drift.
- **Calibration** — reuses `"Chirp/Sweep"` (constant-envelope → identical total power) with the
  `full_power` (k=70, keyed on `deliver_frac`) + `main_lobe_power` laws; the `main_lobe_frac` table is
  the BOTH-main-lobes fraction (|f|∈[5.115,15.345] MHz) of the BOC integral, keyed on `--sidelobes`:
  frac(0)=0.9146 (−0.39 dB — the ±15.345 band also passes the low-power centre gap, exactly MCode's
  note), down to frac(3)=0.867. Both baked literals re-derived from the BOC PSD in `--self-test`.
- `argspec`/`ramp` untouched (drift guard intact); no agent/client change (the laws + hidden keys ride
  through the static `argspec`, folded by the existing `power_law`/`eval_formula`). Tests:
  `tests/test_boc_enveloped_sweep.py` (BOC PSD split spectrum + finite peaks; argspec + reuse +
  the `--inner-sidelobes` param + `deliver_frac`/`main_lobe_frac` hidden; laws evaluate incl. the
  0.39 dB full-vs-main gap; `full_power` exact under the notch (== main-lobes at sl 0); main_lobe_frac
  ≡ BOC integral + notch-independent; constant envelope + seam; realised split spectrum; sidelobes set
  the band; the inner notch drops the centre gap but keeps the lobes; roam guard; confinement;
  `--self-test`/`--describe-params`). Suite 83 → 98.

## Current state — Enveloped Sweep (sinc² dwell-shaped constant-envelope sweep): COMPLETE (branch `claude/system-familiarization-f5mezz`, scripts-only)
New `Other Signals/enveloped_sweep_tx.py`. A SEQUENTIAL swept tone whose TIME-AVERAGED PSD is
shaped like a sinc² — energy concentrated at the centre — realised purely by DWELL TIME, not an
amplitude envelope. Constant-envelope (full amplitude every instant, ~0 dB PAPR), so it delivers
full power; the shaping redistributes WHERE the power lands, raising the centre PSD above a flat
sweep at the same gain (owner-verified reasoning: at max gain amplitude² is already maxed, so
only dwell time is free — `PSD(f) ∝ dwell(f)`). Design conversation settled: sequential (not a
simultaneous multitone — that pays 11–23 dB crest factor for the same bell); dwell shaping (not
an amplitude taper — that only lowers the edges and wastes power); sinc² only for v1.
- **Mechanism** — `_sinc2_dwell_freq` drives the instantaneous frequency through the INVERSE CDF
  of `S(f)=sinc²(f/chip_rate)`, so the tone dwells ∝ S(f). Swept symmetrically (out-and-back) so
  the looped buffer closes with no reset (`--self-test` seam err ~6e-11); a tiny `DWELL_FLOOR`
  lets the tone creep through the nulls (soft nulls ~-22 dB, leakage-limited — deeper needs a
  longer buffer + lower floor, diminishing ~1 dB/doubling; owner chose to keep the 2 MB buffer).
  Reuses the `fm_chirp` precompute-and-loop + always-on unity passband filter + `vector_source_c`
  hot-swap. **The sweep is CONFINED to ±roam by construction** (the inverse-CDF maps all time into
  the occupied band), so it never dwells outside the filter passband — `--self-test` asserts
  `max|f| ≤ roam`.
- **Params** — `--chip-rate` (Mcps ≡ MHz): the FIRST sinc² null sits at ±chip-rate (BPSK/PRN
  convention); `--sidelobes` truncates the band to ±(sidelobes+1)·chip-rate; `--freq`/`--power`/
  `--gain`/`--rf` + both shape knobs are live (a shape change rebuilds one sweep + swaps under the
  top-block lock). Occupied band guarded to ≤ ±27 MHz (`check_roam`).
- **Calibration REUSES the flat sweep's** — `CAL_SIGNAL_ID = "Chirp/Sweep"`. Both signals are
  constant-envelope at amplitude 0.5, so at a given gain they deliver the IDENTICAL total power
  (verified within 0.002 dB) — the unit's flat-sweep calibration already contains this signal's
  power vs gain, no separate measurement. `CAL_POWER_LAWS` offer **`full_power`** (k=70, = the
  flat sweep's bandwidth-invariant total = `fm_chirp`'s `fbw_power`) and **`main_lobe_power`**
  (full + a fixed sinc² offset KEYED on `--sidelobes` via a hidden `main_lobe_frac` derived-field
  table — −0.22 dB at 1 sl … −0.40 dB at 8 sl; frac(0)=1). So the flat sweep's passband PSD at a
  gain gives this signal's full and main-lobe power at that gain. Runtime folds `--power` in the
  base density exactly like the chirp (`pwr_params()` supplies the live `main_lobe_frac`). The
  `main_lobe_frac` table is a baked literal (static AST reader) re-derived from the sinc² integral
  in `--self-test` so it can't drift.
- `argspec`/`ramp` untouched (drift guard intact); no agent/client change (laws + hidden key ride
  through `argspec`; the client renders the same power card as the GPS density signals). NOT yet
  wired into `run_local.sh`/the sample unit and no mock stand-in (candidates for a follow-up).
  Tests: `tests/test_enveloped_sweep.py` (argspec surface + reuse; constant envelope + seam;
  sinc² shape incl. −13.3 dB 1st sidelobe; chip-rate sets the null; sidelobes set the band;
  confinement to ±roam; the full/main-lobe laws evaluate; main_lobe_frac ≡ sinc² integral;
  `--self-test`/`--describe-params` subprocesses). Suite 69 → 83.

## Current state — `cw_drift_tx.py --duration` in MINUTES: COMPLETE (branch `claude/cw-drift-wide`, scripts-only)
Owner ask (after the MHz change): the drift's duration in minutes, not seconds. `--duration` now declares
`unit="min"`, `min=0.1` (6 s), `max=MAX_DURATION_MIN` (7 days = 10080), `default=10.0` (was 600 s);
`main()` scales once (`duration_s = args.duration × 60`) and the drift law / rate / progress line keep
running in seconds (`drift_freq`, `fmt_rate` untouched; the banner reads `over 180 min · −27.78 kHz/s
(−100 MHz/h)`). No agent/client change (a plain numeric field with a `min` unit label). A saved task
that still passes the old seconds value drifts 60× too slowly — re-enter it in minutes. Test:
`tests/test_cw_drift.py` (schema unit/range/default; the fake-`gnuradio` banner test asserts the
rate, which only comes out right if the minutes were scaled). Suite unchanged at 69.

## Current state — the CW scripts take their frequencies in MHz: COMPLETE (branch `claude/cw-drift-wide`, cross-repo seed)
Owner ask: the CW scripts' frequency parameters in MHz (they were the only RPi calibrated signals
still in Hz; the PRN/chirp scripts' `-Center-frequency` is MHz). `cw_tx.py` `--freq`, `cw_drift_tx.py`
`--freq` + `--freq_end`, and the mock `mock_cw_tx.py` `--freq` now declare `unit="MHz"`, `min=70`,
`max=6000`, defaults `1575.42` (/ `1575.43`), and the `FREQUENCIES` preset dict is MHz. Each `main()`
scales ONCE at the boundary (`× 1e6`) — the fold, the planner math (`plan_lo`/`hop_count`/…, all Hz)
and the radio are untouched; a live `--freq` tune scales the same way and reports back in MHz. The
agent needs no change: its carrier derivation (`tune_log.freq_hz_of`, 1.27.2) scales by the DECLARED
unit, so the attenuator / export / `SDR_CAL_FREQ_HZ` follow automatically. Cross-repo: the agent's
sample seed (`sdr-agent/deploy/make_sample_sequences.py` → `sequences.json`) launches `mock_cw` with
`--freq 1575.42` (was `1575420000`, which the MHz schema would refuse as > 6000). Existing saved tasks
/ sequences that launch a CW script with a Hz value must be re-entered in MHz (the schema refuses
the old value loudly rather than silently mis-tuning). The X410 `cw_channel.py` and the other RPi
`Other Signals` scripts (noise/comb/mock_sdr) still take Hz — untouched. Tests: `tests/test_mock_cw.py`
(`--freq 1300` vs `1600` on a biased chain differ by the flatness → the mock scales MHz→Hz before
folding; the surface guard pins unit/min/max/default/presets), `tests/test_cw_drift.py` (schema in
MHz; a fake `gnuradio` package lets the REAL `cw_drift_tx.py` / `cw_tx.py` `main()` run to their
banner, which prints the Hz the MHz args became — `1600.000000 → 1300.000000 MHz`, 214 hops). Suite
67 → 69.

## Current state — CW drift over hundreds of MHz / days (`cw_drift_tx.py` rewrite): COMPLETE (branch `claude/cw-drift-wide`, scripts-only)
Owner ask: a CW that drifts hundreds of MHz over a very long time; one script for the plain single
tone and one for the drift. The split already existed (`Other Signals/cw_tx.py` = the pure tone,
`cw_drift_tx.py` = the drift), but the drift script capped the duration at 20 min, REFUSED any span
wider than the baseband window (`drift range >= samp_rate` → error, so a few MHz on a Pi) and folded
`--power` at the START frequency only. Rewritten (scripts-only; no agent/client change — the agent's
static `argspec` reads the new schema as-is):
- **Wide sweeps via LO hops** — the X410 `cw_channel.py` planner ported (`plan_lo`, `SWEEP_MARGIN` 0.7):
  the software NCO carries the tone within a window of `0.7·sample_rate`; at a window edge the analog LO
  hops one window and the NCO wraps, BLANKED (`--hop_blank`, 20 ms default, live) to hide the synth
  relock. `half_window_hz` / `is_wide` / `initial_lo` (a window grid centred on the sweep) /
  `hop_count` (closed form) are pure helpers. 300 MHz at 2 MHz = 214 hops (one 20 ms blank per ~50 s of
  a 3 h drift); at 10 MHz = 42. A narrow drift (span ≤ one window) is unchanged: LO fixed at the centre,
  fully continuous. `--duration` max 20 min → **7 days** (`MAX_DURATION_S`); sample-rate presets 1/2/5/
  10/20 MHz (the 40 MHz preset dropped — a Pi can't stream it; the hardware max 61.44 stays accepted).
- **Calibrated power tracks the drift** — the held `--power` is re-folded through the calibration at the
  LIVE frequency every `REFOLD_STEP_HZ` (250 kHz) of movement (`needs_refold` → `gain_for_power(freq=f)`,
  the gain applied only when it changes; raw `--gain` is never re-folded). `coverage_gaps(pmap, power,
  start, end)` samples the sweep at start and the banner names each stretch where the ceiling sits below
  the request (`⚠ POWER … can't be delivered over X–Y MHz`; the gain clamps there — safe). A progress
  line every 5 min (`drift @ … MHz (N % of the span) · gain · LO hops so far`).
- **The SDR/attenuator split is PINNED across the drift** (needs `sdr-agent` ≥ 1.27.2 on the unit —
  `PowerMap.pinned_applied` / `gain_for_power(..., applied_db=)`). The agent positions a programmable
  attenuator ONCE, at the launch carrier (the start frequency, from `CAL_FREQ_PARAM`); as the tone
  moves the script folds only the SDR gain with that attenuation pinned (`state["applied"]`), never
  re-realizing — re-picking the split at every new frequency would hop the ASSUMED attenuation by
  whole steps while the physical attenuator stayed put (a 14.25 → 12.75 dB walk over a 300 MHz drift
  on a frequency-dependent chain, i.e. up to 1.5 dB of power error). A live `--power` tune re-picks
  the split at the START frequency (where the agent repositions the attenuator for it) and folds the
  SDR gain at the live frequency with it pinned; `coverage_gaps(..., applied_db=)` checks the sweep
  the same way; the banner names the pinned attenuation. No attenuator ⇒ `applied` is None, the plain
  fold. Test: `tests/test_cw_drift.py::test_the_attenuator_split_is_pinned_across_the_drift` (an
  engaged −4.5 dB-insertion attenuator: the pinned fold holds −100 dBm within 0.13 dB over the sweep
  while the per-frequency realization would pick a different attenuation). Suite 66 → 67.
- **Shared calibration signal** — `CAL_SIGNAL_ID` `"cw_drift"` → **`"cw_tone"`** (same as `cw_tx.py`): at
  any instant the drift IS a pure CW at one frequency at the same `AMPLITUDE`, so the same measured
  curve applies and one calibration serves both; the unit's source-flatness / cable tables supply the
  frequency dependence. A unit that had a separate `cw_drift` signal calibrated keeps it unused.
  `CAL_FREQ_PARAM` stays `freq` (the client folds the shown range at the start; the script re-folds).
- `--rf` is marked `is_rf=True` on BOTH real CW scripts (the mock already had it), so the client's RF
  auto-gating / run-log muting recognise the gate by the marker, not only the on/off convention.
- `--self-test` re-derives the planner over a 300 MHz sweep at 2 + 10 MHz (offset within ±half, hop
  count == `hop_count`). Tests: `tests/test_cw_drift.py` (drift modes; wide-sweep walk at 2/10 MHz — NCO
  never leaves the window, hops per window, symmetric grid, up == down; narrow never hops; loop/pingpong
  wraps; refold cadence + a 6 dB flatness rise moves the gain 6 dB; coverage gaps name a 15 dB dip and
  stay silent for a deliverable level / uncalibrated; `fmt_rate`; the argspec surface incl. the shared
  signal id, 7-day max, presets; `--self-test` + `--describe-params` subprocesses). Suite 55 → 66.
  Not verified on hardware (no radio here): the hop path is the proven `set_center_freq` + `set_tone`
  sequence, mute-wrapped.

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

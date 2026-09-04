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

## Current state — GPS PRN spectral-density calibration: COMPLETE (branch `claude/gps-calibration`)
The six remaining `PRN GPS` scripts got the C/A spectral-density calibration surface (measured =
peak PSD in **dBm/Hz**; `CAL_POWER_LAWS` convert it to absolute-power quantities the operator picks
for `--power`). Scripts-only — laws ride through `argspec` verbatim; no agent/client/capability change.
- **BPSK (single main lobe):** `gps_l5` (Rc 10.23), `gps_l2c` (Rc 1.023), `gps_l1p`/`gps_l2p`
  (Rc 10.23, streamed/filterless, fixed carrier → no `CAL_FREQ_PARAM`). Two CONSTANT laws:
  `main_lobe_power` (k = 10·log10(Rc·I_ML)) and `carrier_power` (total, k = 10·log10(Rc), ≈ +0.444 dB
  above the main lobe). Bandwidth-independent, so carrier is the safe amp-limiting quantity (a filter
  only lowers emitted power). `CAL_FREQ_PARAM="freq"` on l5/l2c (they have `--freq`).
- **BOC (two main lobes):** `MCode` BOC(10,5) — both main lobes ±[5.115,15.345] MHz; `gps_l1c`
  TMBOC — the BOC(1,1) **core** (both ±1.023 MHz lobes) per owner's choice. Each offers
  `main_lobe_power` (both lobes) + `full_power` (widest-passband, the amp limit — M-code to ±30.69,
  L1C to ±30.69 incl. the BOC(6,1) lobes). No carrier quantity (subcarrier harmonics the filter
  strips make it ill-defined). Constants computed by integrating the sine-BOC PSD (Betz).
- Every constant is baked as a literal AND re-derived in each script's `--self-test` (∫sinc² for
  BPSK, ∫BOC for BOC) so it can't silently drift. Guard test:
  `tests/test_gps_power_quantities.py` (argspec extracts each surface; laws evaluate via the real
  `paramkit.power_law`; BPSK carrier = main + 0.444 dB; BOC full > main, no carrier).
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

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

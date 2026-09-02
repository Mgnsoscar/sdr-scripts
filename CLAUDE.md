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
- `--self-test` — a no-hardware spectral-density check some generators implement.

## Current state
Latest work this branch: the GPS C/A transmit scripts were consolidated to exactly two, named by
chip rate, with generic (band-agnostic) calibration ids and both L1+L2 carrier presets —
`Raspberry pi + b206 mini-i/PRN GPS/gps_ca_code_1.023Mcps.py` and `gps_ca_code_10.23Mcps.py`
(the 10.23 script got the full spectral-density calibration treatment: enbw table, max sidelobes,
`--self-test`). The broader per-signal calibration-UI redesign it supports is complete across the
three repos — see `sdr-client/docs/calibration-ui-redesign.md`.

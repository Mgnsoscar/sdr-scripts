# Ettus X410 — signal engine & channel-tasks

GNSS (and CW / chirp) signal generation for the **Ettus/NI USRP X410**, driven by
the SDR agent + FleetView client. Unlike the Raspberry-Pi units — where each script
owns the radio and transmits one signal — the X410 has **four TX channels but UHD
lets only one process own the device at a time.** So this folder is built around a
different model:

> **One persistent engine owns UHD. Short-lived "channel-tasks" drive individual
> channels over a socket.**

A "task" becomes a *channel command*, not a device-claiming process, so up to four
signals run and overlap across the four RF ports while only the engine ever touches
UHD.

```
                         ┌─────────────────────────────────────────┐
   FleetView / agent     │            x410_engine (1 process)       │
        │  tasks         │   owns UHD · 4 channels · 1 per RF port  │
        ▼                │                                          │
  gps_prn_channel ──┐    │   ch0  ← replay/generate ← RF0           │
  gal_e1_channel  ──┼──► │   ch1  ← replay/generate ← RF1   socket  │
  bds_b1c_channel ──┤    │   ch2  ← replay/generate ← RF2  control  │
  cw_channel      ──┘    │   ch3  ← replay/generate ← RF3           │
   (channel-tasks)       └─────────────────────────────────────────┘
```

Every code is reproduced **bit-exact** against its ICD/reference check values, and
every signal ships a `--self-test` that also measures its **acquisition fidelity
under the engine's rate negotiation** — so each script proves, with no hardware,
that a receiver will still acquire it.

---

## The pieces

| File | Role |
|---|---|
| `x410_engine.py` | The persistent device owner. Opens one MultiUSRP, one TX streamer + sample rate **per channel**, and replays/generates each channel from its own thread. Speaks a line-JSON control protocol over a Unix socket. |
| `engine_client.py` | Thin client for the engine socket (`acquire`/`configure`/`load`/`set`/`release`) + a `channel_session` helper. No UHD. |
| `channel_task.py` | Shared channel-task lifecycle (connect → acquire → configure → build IQ → load muted → forward live tune changes → release). Each signal script supplies only its code-gen + a `build()` callback. |
| `gnss_acq.py` | Hardware-free acquisition checks (correlation peak-to-sidelobe, cross-code isolation, negotiation-fidelity) used by every `--self-test`. |
| `*_channel.py` | The signal scripts — one per GNSS signal, plus `cw_channel` and `fm_chirp_channel`. |

---

## Why per-channel sample rates

A real scene mixes a **wide** signal (Galileo E5 AltBOC, ~51 MHz → 61.44 MS/s)
with **narrow** ones (GPS L1 C/A, ~2 MHz) at the same time. A single device-wide
rate would force every channel up to the widest one — ~4×61 MS/s of mostly-
oversampled narrow signal, which the X410's modest ARM can't spare. So the engine
drives each channel through its **own TX streamer at its own rate**: the narrow
channels stream at a low rate, only the wide one pays for its width.

Because the master clock is fixed (stock, 245.76 MHz default), the achievable
per-channel rates are its integer divisors (61.44, 40.96, 30.72, 20.48, 10.24,
8.192, …). A channel-task asks for a target rate and the engine returns the
**actual** rate it locked to; the task then builds its IQ for exactly that rate
(two-phase negotiation). GNSS codes are ms-periodic, so they still loop seamlessly,
and the sub-sample chip-edge jitter from a non-integer samples/chip costs a
fraction of a dB in acquisition (measured in each `--self-test`).

---

## Playback modes

Every signal lowers to one of these engine modes:

| Mode | What it is | Signals |
|---|---|---|
| **expanded** | one device-rate buffer, replayed | C/A, P(Y), L2C, L5, M-code, B1I/B2b/B3I/B2a, E5, PRS, AltBOC, chirp, GLONASS |
| **composite** | a few distinct period-blocks + a per-period selector sequence; streams the full (e.g. 18 s overlay) signal from a handful of blocks, byte-identical to a fully-baked buffer | L1C, B1C, E1, E6 |
| **tone** | a **generated** continuous-phase CW at a baseband offset — no buffer; the frequency can be drifted live, and a wide sweep hops the LO under a small baseband | CW / drift-CW |

---

## Running on the X410

1. **Deploy** these scripts + `paramkit` onto the unit (FleetView Library deploy,
   or the `sdr-agent` X410 install). Scripts land flat in the agent's `scripts/`
   dir alongside `paramkit/`, so `from paramkit import Script`, `from engine_client
   import …` and `from channel_task import …` all resolve.

2. **Start the engine once** (a long-running task in `tasks.yaml`, or by hand):

   ```sh
   python3 x410_engine.py --master_clock 245.76 --socket /tmp/x410_engine.sock
   ```

   It opens the device, sets the master clock, and waits — all channels idle/muted.

3. **Run channel-tasks** against the engine socket, one per signal. Each is an
   ordinary agent task. Example (GPS L1 C/A on RF0):

   ```sh
   python3 gps_prn_channel.py --channel 0 --prn 5 --code_rate 1.023 \
       --freq 1.57542e9 --samp_rate 20.46 --gain 45 --amplitude 0
   ```

### The pre-roll / on-air handshake

Building IQ (and, for GLONASS SF, synthesising the P-code) takes time, and you
usually want several channels to go live at a precise instant. So each channel-task
follows this pattern, which fits the client's timeline:

- **Start ~10 s early with `--amplitude 0`.** The task acquires its channel,
  negotiates the rate, builds its IQ, and loads it **muted** — the engine streams
  zeros so the channel stays fed and glitch-free.
- **At on-air, a timeline tune-step raises `amplitude` (and/or `gain`).** These are
  paramkit `live` params, forwarded to the engine, so the existing live-tune UI and
  sequence tune-steps work unchanged. The signal appears cleanly at the on-air
  instant.

`freq`, `gain`, `amplitude` are live on every signal. The CW task adds a live
`restart` trigger (re-run the drift from the start).

### Underflow monitoring

The engine normally suppresses UHD's fastpath `U` markers, so a struggling
channel used to fail silently. Each channel now runs a lightweight async monitor:
TX underflows are **counted per channel and exposed in `status`** (the
`underflows` field), and a throttled `[engine] chN TX underflow …` line is logged
to stderr. If a channel underflows, it's asking for more samples/s than the ARM
can generate — lower that channel's `--samp_rate`, or move a wide signal off a
crowded scene. The generated `tone` mode caches its per-sample phasor ramp (only
rebuilt when the frequency changes), so a pure or drifting CW now streams cleanly
at the wide rates too.

---

## Signal catalog

All frequencies are the real carriers; `--samp_rate` is a *target* the engine
negotiates to the nearest supported rate.

| Script | Signal(s) | Mode | Key params |
|---|---|---|---|
| `gps_prn_channel` | GPS **L1 C/A**, **L1 P(Y)**, **L2 P(Y)** | expanded | `--prn 1..32`, `--code_rate 1.023\|10.23`, `--freq` L1/L2 |
| `gps_l1c_channel` | GPS **L1C** (QMBOC, 18 s overlay) | composite | `--prn 1..63`, `--component both\|pilot\|data`, `--secondary full\|off` |
| `gps_l2c_channel` | GPS **L2C** (CM/CL) | expanded | `--prn 1..63`, `--loop full\|cm` |
| `gps_l5_channel` | GPS **L5** (QPSK, NH10/NH20) | expanded | `--prn 1..63`, `--component IQ\|I\|Q` |
| `mcode_channel` | GPS **M-code** BOC(10,5) *(surrogate)* | expanded | `--prn 1..32`, `--freq` L1/L2 |
| `bds_b1i_channel` | BeiDou **B1I** BPSK-R(2) | expanded | `--prn 1..63` |
| `bds_b2a_channel` | BeiDou **B2a** QPSK tiered | expanded | `--prn 1..63`, `--component`, `--loop full\|primary` |
| `bds_b2b_channel` | BeiDou **B2b_I** BPSK-R(10) | expanded | `--prn 6..58` |
| `bds_b3i_channel` | BeiDou **B3I** BPSK-R(10) | expanded | `--prn 1..63` |
| `bds_b1c_channel` | BeiDou **B1C** QMBOC (18 s overlay) | composite | `--prn 1..63`, `--component`, `--secondary` |
| `gal_e1_channel` | Galileo **E1** CBOC(6,1,1/11) | composite | `--svid 1..50`, `--secondary full\|off` |
| `gal_e5_channel` | Galileo **E5a / E5b** QPSK | expanded | `--band E5a\|E5b`, `--svid 1..50`, `--component IQ\|I\|Q` |
| `gal_e6_channel` | Galileo **E6** BPSK(5) | composite | `--svid 1..50`, `--secondary full\|off` |
| `gal_prs_channel` | Galileo **PRS** E1-A/E6-A *(surrogate)* | expanded | `--band E1A\|E6A` |
| `gal_e5_altboc_channel` | Galileo **E5 AltBOC(15,10)** (both sidebands) | expanded | `--svid 1..50` |
| `glonass_of_channel` | GLONASS **L1OF/L2OF** (FDMA C/A) | expanded | `--band L1\|L2`, `--mode channel\|band`, `--k -7..6` |
| `glonass_sf_channel` | GLONASS **L1SF/L2SF** (FDMA P-code) | expanded | `--band L1\|L2`, `--mode channel\|band`, `--k -7..6` |
| `iridium_stl_channel` | **Iridium / STL** DQPSK bursts *(surrogate)* | expanded | `--freq` (STL band), `--payload_symbols`, `--burst_period`, `--frames` |
| `fm_chirp_channel` | FM chirp (swept tone) | expanded | `--freq`, `--bw`, `--rate`, `--waveform` |
| `cw_channel` | **CW** tone + optional slow drift (wide spans hop the LO) | tone | `--freq`, `--freq_end`, `--duration ≤1200`, `--drift once\|loop\|pingpong`, `--restart` |

**Surrogates:** M-code, PRS and Iridium/STL reproduce the correct RF/spectral (and,
for Iridium, burst-timing) shape over an unclassified stand-in — the real sequences
/ payloads are classified or proprietary. Iridium/STL is a burst DQPSK waveform, not
a PRN. GPS "P(Y)" here is the C/A Gold code clocked at 10.23 Mcps (a wideband
surrogate), *not* the encrypted P/Y. GLONASS SF is the **real** public P-code
(reverse-engineered, unencrypted).

### CW / drift-CW

A pure tone, or a tone that drifts from a start to an end frequency over up to
20 minutes — built on the engine's generated `tone` mode, so the drift is
continuous-phase and costs no buffer however slow it is.

```sh
# pure CW at L1 (leave --freq_end unset)
cw_channel.py --channel 0 --freq 1575.42e6 --gain 45 --amplitude 0

# narrow drift 1575.42 → 1575.43 MHz over 20 min, restartable from a tune-step
cw_channel.py --channel 1 --freq 1575.42e6 --freq_end 1575.43e6 \
    --duration 1200 --drift once --samp_rate 2.048

# WIDE sweep 1600 → 1545 MHz over 15 min — the LO hops, baseband stays small
cw_channel.py --channel 0 --freq 1600e6 --freq_end 1545e6 \
    --duration 900 --drift once --samp_rate 5.0
```

The drift begins at on-air (first `amplitude > 0`); the live `restart` flag re-runs
it from the start.

**Narrow vs. wide sweeps** — a baseband tone can only occupy ±`samp_rate`/2, so the
task picks the regime automatically:

- **Narrow** (span ≤ one baseband window): the LO stays fixed at the drift centre
  and the whole sweep is carried in baseband — perfectly continuous, no retunes.
- **Wide** (span bigger than a window, e.g. the 55 MHz `1600→1545`): the hardware
  **LO hops** in whole-window steps and the baseband tone fills in between. A hop
  moves the LO and the baseband by equal-and-opposite amounts, so the *emitted*
  frequency stays continuous across it — only the LO's own retune settle is a brief
  transient. So a 55 MHz sweep needs only a few-MHz `--samp_rate` (one hop window),
  not 55 MS/s. Hop count is `span ÷ window`, so a higher `--samp_rate` = fewer hops
  (e.g. 55 MHz at 5 MS/s ≈ 16 hops, ~1/min). Keep the rate in the ARM's clean range
  (~5 MS/s is solid; 10 MS/s jitters occasionally).

---

## Testing (no hardware)

- `python3 <script>.py --self-test` — validates the codes bit-exact against their
  ICD/reference check values and measures negotiation fidelity. Runs anywhere with
  NumPy; the engine/UHD are never imported.
- `python3 <script>.py --describe-params` — prints the paramkit JSON schema the
  client renders the form from.
- `python3 x410_engine.py --self-test` — protocol + state machine + playlist/tone
  generation, with a mock radio (no UHD).

---

## First hardware session

The real UHD path (`MultiUSRP` per-channel streamers) is exercised only on the
device. In order:

1. **Benchmark the ARM.** The decisive question — can it sustain a mixed 4-channel
   scene through the Python replay threads?

   ```sh
   python3 x410_engine.py --benchmark 10 --bench_rates 61.44,8.192,8.192,8.192
   ```

   A clean (0-underflow) result confirms the per-channel-replay approach at that
   scene; underflows tell you the real rate ceiling.

2. **One signal into a cable + attenuator.** Start the engine, run one channel-task,
   and confirm a receiver/analyzer acquires it.

3. **Set the real TX gain range.** Every script currently caps `--gain` at **65 dB**
   as a placeholder — set it to the X410/ZBX's actual TX gain range once confirmed.

⚠ **RF safety:** most of these are live GNSS bands. Transmit **only** into a
shielded / conducted setup (cable + attenuators into a receiver or analyzer) that
you are licensed / authorised to use — never over the air.

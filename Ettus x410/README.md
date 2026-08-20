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
| **tone** | a **generated** continuous-phase CW at a baseband offset — no buffer; the frequency can be drifted live, and a wide sweep pins the analog LO and moves the hardware NCO | CW / drift-CW |

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

   **Two playback backends (`--backend stream | replay`).** This is the main
   throughput decision:

   - **`stream`** (default) — the ARM streams every sample in real time. Simple,
     supports live drifting CW, and the only path that needs the format/throughput
     knobs below. It underflows once a per-channel rate outruns the ARM (≈10 MS/s+).
   - **`replay`** — each signal's loop is uploaded to **FPGA DRAM once** and the
     RFNoC **Replay block** streams it to the radio, looping, straight from DRAM.
     Nothing per-sample touches the host, so **61.44 MS/s is as cheap as 1 MS/s and
     underflows are structurally impossible**. The one-time upload does the
     `fc32→sc16` conversion off the RF deadline, so you keep full 16-bit fidelity
     with no host real-time cost. This is the right path for wide signals (E5
     AltBOC, etc.). Needs a Replay-capable FPGA image (e.g. the stock `X4_200`).

     ```sh
     # first confirm the loaded FPGA image actually has a Replay/DRAM block:
     python3 x410_engine.py --list_rfnoc
     #   → lists every RFNoC block; needs a line containing "Replay"

     python3 x410_engine.py --backend replay --master_clock 245.76
     ```

     **Requires a Replay-capable FPGA image.** Not every X410 image includes a
     Replay/DRAM block (some show only Radio + DUC blocks). If `--list_rfnoc` shows
     no Replay block, load one that has it — download the stock images and flash the
     default, then re-check:

     ```sh
     uhd_images_downloader                       # fetch stock X410 images
     uhd_image_loader --args "type=x4xx"         # flash (X4_200 includes Replay)
     uhd_usrp_probe | grep -i replay             # confirm a Replay block is present
     ```

     If your image genuinely can't provide a Replay block and reflashing isn't an
     option, the wide-rate signal has to stay on `--backend stream` (and we revisit
     an int16-host memcpy path to push the stream ceiling instead).

     Notes: `--otw`/`--cpu`/`--send_ms` don't apply (there's no host sample loop).
     Digital `amplitude` acts as play/mute (0 = not playing); set the level with
     `gain`. A **static** CW works (a baked loop); a **drifting** CW re-bakes per
     step, so run sweeps on `--backend stream`. Each channel gets an equal slice of
     DRAM; a loop bigger than its slice is rejected (lower the rate or shorten it).
     The engine logs the discovered RFNoC topology and every device call under
     `[engine/replay]`, so a first-run mismatch on your image points to the exact
     block/port to adjust.

   **Wire / host sample format & throughput** (`--otw`, `--cpu`, `--send_ms`;
   `stream` backend only). At high per-channel rates (≳10 MS/s) the ARM can
   underflow. Three levers, all about moving fewer bytes and doing less per-sample
   work — or just switch to `--backend replay` above:

   - `--otw sc8` — 8-bit over the wire, half the data rate of the default `sc16`.
     The single biggest win at wide rates (and what fixed the same problem on the
     Pi). **An 8-bit wire needs an 8-bit (or 16-bit) host format** — this UHD build
     has no `fc32→sc8` converter, so `--otw sc8` with the default `fc32` host
     errors (`Cannot find a conversion routine … fc32 → sc8_chdr`). `--cpu auto`
     handles this: it selects an `sc8` host for an `sc8` wire automatically.
   - `--cpu` — the host sample format. `auto` (default) keeps the proven `fc32`
     path for `sc16` and matches the wire for `sc8`. Setting `--cpu sc16`/`sc8`
     builds the samples **already in the wire layout**, so `send()` is a pure
     memcpy — no per-sample conversion on the ARM. IQ is packed into the wire
     format once, at load, not on every send.
   - `--send_ms` — how many ms of samples go out per `send()` call (default 10).
     Bigger = far fewer Python/UHD calls per second (the real ARM cost) at the
     price of live-tune latency.

   A good high-rate recipe: `--otw sc8 --cpu auto --send_ms 20`. Underflows are
   reported per channel in the log and in `status` (see *Underflow monitoring*).

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
| `cw_channel` | **CW** tone + optional slow drift (wide spans sweep on the NCO) | tone | `--freq`, `--freq_end`, `--duration ≤1200`, `--drift once\|loop\|pingpong`, `--lo_span`, `--hop_blank`, `--restart` |

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

# WIDE sweep 1600 → 1545 MHz over 15 min — swept on the hardware NCO
cw_channel.py --channel 0 --freq 1600e6 --freq_end 1545e6 \
    --duration 900 --drift once --samp_rate 2.048 --lo_span 60
```

The drift begins at on-air (first `amplitude > 0`); the live `restart` flag re-runs
it from the start.

**Narrow vs. wide sweeps** — a *software* baseband tone can only occupy
±`samp_rate`/2, so the task picks the regime automatically:

- **Narrow** (span ≤ one baseband window): the LO stays fixed at the drift centre
  and the whole sweep is carried in the software baseband tone — perfectly
  continuous, no retunes.
- **Wide** (span bigger than a window, e.g. the 55 MHz `1600→1545`): the sweep is
  carried by the **hardware DUC/NCO**. The analog LO is *pinned* (a manual-LO tune)
  and the digital NCO moves the carrier across a ±`lo_span`/2 window — a purely
  digital frequency change, so no analog synth relock, no settle glitch, and a flat
  amplitude across the window (this is what removes the old ~8 dB droop and the
  "tone jumps forward then snaps back" transient, which was a software-baseband vs.
  analog-LO race). The analog LO only re-anchors when the tone leaves the window;
  set `--lo_span` ≥ the span and it never moves at all — a fully glitchless sweep.
  Those rare re-anchors are blanked (`--hop_blank`, default 5 ms) to hide the
  retune. A wide sweep is a near-DC signal, so a **low** `--samp_rate` (1–2 MHz) is
  ideal and keeps ARM load minimal.

  `--lo_span` must stay within the channel's NCO/analog range (start ~40–60 MHz and
  widen it on the bench until the tone can't reach the sweep edges, then back off).

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

   A clean (0-underflow) result confirms the per-channel streaming approach at that
   scene; underflows tell you the real rate ceiling of the `stream` backend. The
   benchmark uses the same `--otw`/`--cpu`/`--send_ms` path as live tasks, so re-run
   it with the format you plan to use — e.g. add `--otw sc8 --send_ms 20` to see how
   much headroom 8-bit + larger sends buys at ≥10 MS/s.

   If a scene needs rates the ARM can't stream (e.g. a 61.44 MS/s channel), use
   **`--backend replay`** — FPGA-DRAM playback is host-rate-independent, so it has no
   underflow ceiling at all.

2. **One signal into a cable + attenuator.** Start the engine, run one channel-task,
   and confirm a receiver/analyzer acquires it.

3. **Set the real TX gain range.** Every script currently caps `--gain` at **65 dB**
   as a placeholder — set it to the X410/ZBX's actual TX gain range once confirmed.

⚠ **RF safety:** most of these are live GNSS bands. Transmit **only** into a
shielded / conducted setup (cable + attenuators into a receiver or analyzer) that
you are licensed / authorised to use — never over the air.

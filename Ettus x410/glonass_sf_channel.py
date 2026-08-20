#!/usr/bin/env python3
"""
glonass_sf_channel — GLONASS L1SF / L2SF (FDMA high-accuracy "P-code") channel-task.

Transmit the legacy GLONASS high-accuracy ranging signal — L1SF (~1602 MHz) or
L2SF (~1246 MHz) — on one X410 engine channel, in either of two modes:

  • channel : one satellite on its FDMA frequency channel k (−7…+6). BPSK of the
    P-code at 5.11 Mcps; the engine carrier is tuned to f_k.
  • band    : the whole FDMA band at once — all 14 channels summed at their
    frequency offsets around the band centre.

The GLONASS P-code is officially undocumented but NOT encrypted: it was publicly
reverse-engineered in 1989 and civilian dual-frequency receivers have tracked it
on L2 since. So — unlike GPS P(Y) — it is reproduced BIT-EXACT from its public
definition; no encrypted or classified content. FDMA, shared code (same as
L1OF/L2OF): every SV uses the SAME P-code, told apart by carrier frequency:
    L1SF:  f_k = 1602 MHz + k · 0.5625 MHz     (k = −7 … +6)
    L2SF:  f_k = 1246 MHz + k · 0.4375 MHz     (k = −7 … +6)

P-code: a 25-stage LFSR, G(x)=1+x³+x²⁵ (taps 25 & 3, output stage 25, all-ones
seed), a maximal m-sequence truncated/reset every 1 s (5 110 000 chips). Applied
via ZOH at the negotiated rate (mode "expanded"). Note the 1 s buffers are large
(~82 MB/channel, ~164 MB/band at these rates) and the code takes a couple of
seconds to synthesise — start the task a few seconds before on-air.

⚠  RF SAFETY / LEGAL: L1 (~1602 MHz) and L2 (~1246 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup you are LICENSED / AUTHORISED
   to use — never over the air.

CLI
───
    glonass_sf_channel.py --channel 0 --band L1 --mode channel --k 0 --gain 55
    glonass_sf_channel.py --channel 1 --band L1 --mode band --samp_rate 20.48
    glonass_sf_channel.py --self-test        # P-code primitivity + FDMA plan, no engine
    glonass_sf_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

CHIP_RATE_HZ = 5.11e6          # GLONASS P-code chip rate (10 × C/A)
CODE_LEN = 5_110_000           # truncated to 1 s (reset each second)
LFSR_DEG = 25                  # 25-stage register, G(x)=1+x³+x²⁵
K_MIN, K_MAX = -7, 6

BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6},
}

SAMPLE_RATES_MHZ = {"10.24 MHz (default)": 10.24, "20.48 MHz (band)": 20.48}


# ── P-code generation ──────────────────────────────────────────────────────────

def _pcode_into(buf) -> None:
    """Fill buf (length up to CODE_LEN) with the P-code 0/1 chips. 25-stage
    Fibonacci LFSR, taps 25 & 3, output stage 25, all-ones seed, truncated."""
    state = (1 << LFSR_DEG) - 1
    mask = (1 << LFSR_DEG) - 1
    for i in range(len(buf)):
        buf[i] = (state >> 24) & 1                  # stage 25 output
        fb = ((state >> 24) ^ (state >> 2)) & 1     # taps 25 and 3
        state = ((state << 1) | fb) & mask


def channel_freq(band: str, k: int) -> float:
    b = BANDS[band]
    return b["base"] + k * b["spacing"]


# ── Baseband buffers (ZOH at the negotiated rate) ──────────────────────────────

def build_channel_buffer(samp_rate_hz: float):
    """One-channel real BPSK P-code buffer for one full 1 s period (seamless on
    reset). Returns (iq, n_samples). Real → I, Q=0."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    chips = np.empty(CODE_LEN, dtype=np.int8)
    _pcode_into(chips)                               # ~2 s, once
    bip = (1.0 - 2.0 * chips).astype(np.float32)
    n_samples = int(round(1.0 * sr))                 # 1 s
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * cr // sr) % CODE_LEN
    return bip[chip].astype(np.complex64), n_samples


def build_band_buffer(band: str, samp_rate_hz: float):
    """Full-band composite: all 14 channels of the P-code summed at their frequency
    offsets around the band centre, each with a distinct cyclic code phase. One 1 s
    buffer, built in time-chunks to bound memory. Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    spacing = BANDS[band]["spacing"]

    chips = np.empty(CODE_LEN, dtype=np.int8)
    _pcode_into(chips)
    bip = (1.0 - 2.0 * chips).astype(np.float32)

    n_samples = int(round(1.0 * sr))                 # 1 s
    ks = list(range(K_MIN, K_MAX + 1))
    shift_of = {k: ((k - K_MIN) * 401_887) % CODE_LEN for k in ks}   # distinct phases

    comp = np.empty(n_samples, dtype=np.complex64)
    CHUNK = 2_000_000
    for start in range(0, n_samples, CHUNK):
        end = min(start + CHUNK, n_samples)
        idx = np.arange(start, end, dtype=np.int64)
        chip = (idx * cr // sr)                       # ZOH chip index
        t = idx / sr
        acc = np.zeros(end - start, dtype=np.complex64)
        for k in ks:
            code_k = bip[(chip + shift_of[k]) % CODE_LEN]
            acc += code_k * np.exp(1j * 2 * np.pi * k * spacing * t).astype(np.complex64)
        comp[start:end] = acc
    peak = float(np.max(np.abs(comp))) or 1.0
    return (comp / peak).astype(np.complex64), n_samples


# ── Self-test (P-code primitivity + FDMA plan; no huge buffer) ─────────────────

def _self_test() -> int:
    POLY = (1 << 25) | (1 << 3) | (1 << 0)
    ORDER = (1 << 25) - 1                            # 33554431 = 31·601·1801
    FACTORS = (31, 601, 1801)

    def mulmod(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> 25) & 1:
                a ^= POLY
        return r

    def powmod(base, e):
        r = 1
        while e:
            if e & 1:
                r = mulmod(r, base)
            base = mulmod(base, base)
            e >>= 1
        return r

    primitive = (powmod(2, ORDER) == 1 and 31 * 601 * 1801 == ORDER
                 and all(powmod(2, ORDER // p) != 1 for p in FACTORS))
    seg = bytearray(300_000)
    _pcode_into(seg)
    frac = sum(seg) / len(seg)
    bal_ok = 0.48 < frac < 0.52
    ok = primitive and bal_ok
    print(f"P-code poly 1+x^3+x^25 primitive: {primitive} (period {ORDER}) "
          f"[{'OK' if primitive else 'FAIL'}]")
    print(f"balance (300k segment): {frac*100:.1f}% ones [{'OK' if bal_ok else 'FAIL'}]")
    print(f"period: {CODE_LEN} chips = {CODE_LEN/CHIP_RATE_HZ:.3f} s @ "
          f"{CHIP_RATE_HZ/1e6:g} Mcps")
    for band in ("L1", "L2"):
        lo, hi = channel_freq(band, K_MIN), channel_freq(band, K_MAX)
        print(f"{band}SF plan: k {K_MIN}..{K_MAX}  {lo/1e6:.4f}..{hi/1e6:.4f} MHz")
    print("ALL GLONASS P-CODE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1SF/L2SF (FDMA P-code) channel-task — one channel k or the "
               "whole FDMA band, on one X410 engine channel. Public P-code, bit-exact.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1SF (~1602 MHz) or L2SF (~1246 MHz). Fixed per run.")
        .choice("-Mode", "--mode", options=["channel", "band"], default="channel",
                help="channel = one SV on its FDMA frequency k; band = all 14 "
                     "channels summed. Fixed per run.")
        .integer("-Channel-k", "--k", min=K_MIN, max=K_MAX, default=0,
                 help="FDMA channel number k (−7..+6). Used in 'channel' mode; sets "
                      "the carrier f_k. Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9, live=True,
                help="RF carrier (auto-set from band/mode/k). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=8.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=10.24, required=True,
                help="Target channel sample rate (negotiated). P-code is ~10 MHz "
                     "wide. NB: 1 s buffers scale with rate (~82 MB @ 10.24 MHz). "
                     "Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=55,
                required=True, live=True, help="Channel TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.0,
                required=True, live=True,
                help="Digital amplitude 0..1. Start at 0 for a pre-roll and raise "
                     "on-air via a tune-step; or set >0 to transmit on load. Live.")
        .text("-Engine-socket", "--engine_socket", default="/tmp/x410_engine.sock",
              help="Unix socket of the running x410_engine.")
        .text("-Owner", "--owner", default="",
              help="Channel ownership tag (default: auto from channel + PID).")
    )


def build(args, rate_hz):
    if args.mode == "channel":
        iq, n_samples = build_channel_buffer(rate_hz)
        detail = f"channel k={args.k}  f_k={channel_freq(args.band, args.k)/1e6:.4f} MHz"
    else:
        iq, n_samples = build_band_buffer(args.band, rate_hz)
        detail = f"band (14 channels summed around {BANDS[args.band]['base']/1e6:g} MHz)"
    path = write_shm(iq, "glonass_sf")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"glonass_sf {args.band} {args.mode}"}
    info = [f"band / mode    : {args.band}SF  {detail}",
            f"buffer         : {n_samples} samples (1 s, {n_samples*8/1e6:.0f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    args.freq = channel_freq(args.band, args.k) if args.mode == "channel" \
        else BANDS[args.band]["base"]
    return run_channel(script, args, build, title="GLONASS SF channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

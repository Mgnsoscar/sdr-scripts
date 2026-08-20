#!/usr/bin/env python3
"""
glonass_of_channel — GLONASS L1OF / L2OF (FDMA open signal) channel-task.

Transmit the legacy open GLONASS signal — L1OF (~1602 MHz) or L2OF (~1246 MHz) —
on one X410 engine channel, in either of two modes:

  • channel : one satellite on its own FDMA frequency channel k (−7…+6). BPSK of
    the 511-chip C/A code at 0.511 Mcps; the engine carrier is tuned to f_k.
  • band    : the whole FDMA band at once — all 14 channels summed at their
    frequency offsets around the band centre, as a wideband receiver sees them.

GLONASS is FDMA, not CDMA: EVERY satellite uses the SAME 511-chip C/A code, and
satellites are told apart by CARRIER FREQUENCY:
    L1OF:  f_k = 1602 MHz + k · 0.5625 MHz     (k = −7 … +6)
    L2OF:  f_k = 1246 MHz + k · 0.4375 MHz     (k = −7 … +6)
so the per-satellite selector is the channel number k, not a code index.

C/A code: a 9-stage LFSR, G(x)=1+x⁵+x⁹, output at stage 7, all-ones seed — a
maximal m-sequence of period 511 (1 ms at 0.511 Mcps). Same code for every SV and
both bands. Applied via ZOH at the engine's negotiated rate (mode "expanded").

⚠  RF SAFETY / LEGAL: L1 (~1602 MHz) and L2 (~1246 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup you are LICENSED / AUTHORISED
   to use — never over the air.

CLI
───
    glonass_of_channel.py --channel 0 --band L1 --mode channel --k 0 --gain 55
    glonass_of_channel.py --channel 1 --band L1 --mode band
    glonass_of_channel.py --self-test        # C/A m-sequence + fidelity, no engine
    glonass_of_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

CHIP_RATE_HZ = 0.511e6         # GLONASS C/A chip rate
CODE_LEN = 511                 # C/A code length (chips) — 1 ms period
K_MIN, K_MAX = -7, 6           # FDMA channel numbers

BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6},
}

SAMPLE_RATES_MHZ = {"10.24 MHz (channel)": 10.24, "12.288 MHz (band)": 12.288,
                    "20.48 MHz": 20.48}


# ── C/A ranging code (pure Python) ─────────────────────────────────────────────

def glonass_ca():
    """The 511-chip GLONASS C/A ranging code (0/1). 9-stage LFSR, G(x)=1+x⁵+x⁹
    (feedback stages 5 and 9), output stage 7, all-ones seed."""
    reg = [1] * 9
    out = []
    for _ in range(CODE_LEN):
        out.append(reg[6])                 # stage 7 output
        fb = reg[4] ^ reg[8]               # taps at stages 5 and 9
        reg = [fb] + reg[:8]
    return out


def channel_freq(band: str, k: int) -> float:
    b = BANDS[band]
    return b["base"] + k * b["spacing"]


# ── Baseband buffers (ZOH at the negotiated rate) ──────────────────────────────

def build_channel_buffer(samp_rate_hz: float):
    """One-channel real BPSK C/A buffer (baseband code at DC; the engine carrier is
    already at f_k). Whole number of 1 ms code periods, seamless. Returns
    (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    spp = Fraction(sr * CODE_LEN, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator
    bip = (1.0 - 2.0 * np.asarray(glonass_ca(), dtype=np.int8)).astype(np.float32)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * cr // sr) % CODE_LEN
    return bip[chip].astype(np.complex64), n_samples, n_periods


def build_band_buffer(band: str, samp_rate_hz: float):
    """Full-band composite: all 14 channels summed at their frequency offsets
    (k·spacing) around the band centre, each with a distinct cyclic code phase. Two
    code periods (2 ms) so every offset completes a whole number of cycles. Returns
    (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    spacing = BANDS[band]["spacing"]
    bip = (1.0 - 2.0 * np.asarray(glonass_ca(), dtype=np.int8)).astype(np.float32)

    n_samples = int(round(0.002 * sr))               # 2 ms window
    idx = np.arange(n_samples, dtype=np.int64)
    chip = (idx * cr // sr)
    t = idx / sr
    comp = np.zeros(n_samples, dtype=np.complex64)
    for k in range(K_MIN, K_MAX + 1):
        shift = ((k - K_MIN) * 37) % CODE_LEN        # distinct fixed code phase
        code_k = bip[(chip + shift) % CODE_LEN]
        comp += code_k * np.exp(1j * 2 * np.pi * k * spacing * t).astype(np.complex64)
    peak = float(np.max(np.abs(comp))) or 1.0
    return (comp / peak).astype(np.complex64), n_samples


# ── Self-test (C/A m-sequence + fidelity, no engine) ───────────────────────────

def _self_test() -> int:
    code = glonass_ca()
    ones = sum(code)
    len_ok = len(code) == CODE_LEN
    bal_ok = ones == 256
    reg, seen, steps = [1] * 9, set(), 0
    for _ in range(CODE_LEN + 5):
        s = tuple(reg)
        if s in seen:
            break
        seen.add(s)
        reg = [reg[4] ^ reg[8]] + reg[:8]
        steps += 1
    max_ok = steps == CODE_LEN
    ok = len_ok and bal_ok and max_ok
    print(f"C/A code: len={len(code)} ones={ones}/255 maximal(period={steps}) "
          f"[{'OK' if ok else 'FAIL'}]")
    for band in ("L1", "L2"):
        lo, hi = channel_freq(band, K_MIN), channel_freq(band, K_MAX)
        print(f"{band}OF plan: k {K_MIN}..{K_MAX}  {lo/1e6:.4f}..{hi/1e6:.4f} MHz "
              f"(spacing {BANDS[band]['spacing']/1e6:g} MHz)")

    try:
        from gnss_acq import check_negotiation_fidelity
        # The 511-chip C/A is an m-sequence: ideal sidelobes are exactly −1 → ~54 dB,
        # so ZOH's ~9 dB drop still leaves a superb ~44 dB peak. The absolute
        # peak/sidelobe (min_db) is the real criterion; allow the large relative loss.
        ok = check_negotiation_fidelity(
            lambda r: build_channel_buffer(r)[0], chip_rate_hz=CHIP_RATE_HZ,
            ideal_rate_hz=10.22e6, negotiated_rate_hz=10.24e6, label="OF C/A",
            min_db=35.0, max_loss_db=12.0) and ok
        # band composite builds and is bounded to unit magnitude
        import numpy as np
        iq, n = build_band_buffer("L1", 12.288e6)
        band_ok = bool(np.max(np.abs(iq)) <= 1.0 + 1e-6)
        ok = ok and band_ok
        print(f"L1 band composite @12.288 MHz: {n} samples, peak≤1={band_ok} "
              f"[{'OK' if band_ok else 'FAIL'}]")
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL GLONASS OF CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1OF/L2OF (FDMA open C/A) channel-task — one channel k or "
               "the whole FDMA band, on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1OF (~1602 MHz) or L2OF (~1246 MHz). Fixed per run.")
        .choice("-Mode", "--mode", options=["channel", "band"], default="channel",
                help="channel = one SV on its FDMA frequency k; band = all 14 "
                     "channels summed. Fixed per run.")
        .integer("-Channel-k", "--k", min=K_MIN, max=K_MAX, default=0,
                 help="FDMA channel number k (−7..+6). Used in 'channel' mode; sets "
                      "the carrier f_k. Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9, live=True,
                help="RF carrier (auto-set from band/mode/k). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=4.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=12.288, required=True,
                help="Target channel sample rate (negotiated). ~10 MHz per channel; "
                     "~12 MHz spans the band. Fixed per run.")
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
        iq, n_samples, n_periods = build_channel_buffer(rate_hz)
        detail = f"channel k={args.k}  f_k={channel_freq(args.band, args.k)/1e6:.4f} MHz"
    else:
        iq, n_samples = build_band_buffer(args.band, rate_hz)
        detail = f"band (14 channels summed around {BANDS[args.band]['base']/1e6:g} MHz)"
    path = write_shm(iq, "glonass_of")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"glonass_of {args.band} {args.mode}"}
    info = [f"band / mode    : {args.band}OF  {detail}",
            f"buffer         : {n_samples} samples ({n_samples*8/1e6:.1f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    # Carrier: f_k in channel mode, band centre in band mode.
    args.freq = channel_freq(args.band, args.k) if args.mode == "channel" \
        else BANDS[args.band]["base"]
    return run_channel(script, args, build, title="GLONASS OF channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

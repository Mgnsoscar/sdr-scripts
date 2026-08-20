#!/usr/bin/env python3
"""
gal_prs_channel — Galileo PRS (E1-A / E6-A) SPECTRAL-SURROGATE channel-task.

The Galileo Public Regulated Service (PRS) — components E1-A and E6-A — uses
ENCRYPTED, CLASSIFIED spreading codes that are not published and not reproduced
here. This task transmits a SPECTRAL SURROGATE: the correct PRS *modulation*
(cosine-phased BOC, chip and sub-carrier rates, carrier) driven by a public
m-sequence standing in for the secret code. It reproduces the PRS spectral
footprint for conducted front-end / filter / band-occupancy testing, and is
noise-like to a real PRS receiver — it carries none of the PRS code.

    E1-A : BOC_cos(15, 2.5) @ 1575.42 MHz   (sub 15 MHz, chip 2.5 Mcps)
    E6-A : BOC_cos(10, 5)   @ 1278.75 MHz   (sub 10 MHz, chip 5 Mcps)

Surrogate code: a degree-14 maximal-length m-sequence (16383 chips, balanced),
dense enough that its line spectrum reads as the continuous PRS PSD envelope.
NOT the PRS code. The cosine-BOC sub-carrier and code are applied via ZOH at the
engine's negotiated rate (mode "expanded").

⚠  RF SAFETY / LEGAL: E1/E6 are live GNSS bands. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gal_prs_channel.py --channel 0 --band E1A --gain 55 --amplitude 0
    gal_prs_channel.py --channel 1 --band E6A --samp_rate 40.96
    gal_prs_channel.py --self-test        # surrogate m-seq + BOC pattern, no engine
    gal_prs_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

MHZ = 1e6
BANDS = {
    "E1A": {"carrier": 1575.42e6, "sub_hz": 15 * MHZ, "chip_hz": 2.5 * MHZ,
            "native_sr": 61.44e6, "label": "BOC_cos(15, 2.5)"},
    "E6A": {"carrier": 1278.75e6, "sub_hz": 10 * MHZ, "chip_hz": 5 * MHZ,
            "native_sr": 40.96e6, "label": "BOC_cos(10, 5)"},
}

LFSR_DEGREE = 14
LFSR_TAPS = (14, 5, 3, 1)      # primitive polynomial x^14+x^5+x^3+x+1

SAMPLE_RATES_MHZ = {"40.96 MHz (E6A)": 40.96, "61.44 MHz (E1A default)": 61.44}


# ── Surrogate code + cosine-BOC ────────────────────────────────────────────────

def surrogate_code(degree: int = LFSR_DEGREE, taps=LFSR_TAPS):
    """Maximal-length m-sequence of length 2^degree − 1 (0/1), Fibonacci LFSR — a
    public stand-in for the classified PRS ranging code."""
    state = 1
    length = (1 << degree) - 1
    out = []
    for _ in range(length):
        out.append(state & 1)
        fb = 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (fb << (degree - 1))
    return out


def build_prs_buffer(band: str, samp_rate_hz: float, degree: int = LFSR_DEGREE):
    """Complex64 buffer of ONE surrogate-code period × cosine-BOC sub-carrier, built
    by ZOH at any rate. Real (single BOC channel) → I, Q = 0. Because 2.5/5 Mcps
    don't divide the stock clock, an exact seamless loop would need up to ~125 code
    periods (~400 MB); the surrogate is noise-like, so we loop one period (~3 MB)
    and accept the sub-sample seam at the wrap — spectrally negligible for a
    surrogate. Returns (iq, n_samples)."""
    import numpy as np

    b = BANDS[band]
    sr = int(round(samp_rate_hz))
    chip = int(round(b["chip_hz"]))
    sub = int(round(b["sub_hz"]))

    code = np.asarray(surrogate_code(degree), dtype=np.int8)
    L = code.size
    bip = (1 - 2 * code).astype(np.float32)

    n_samples = int(round(L * sr / chip))    # one code period (~6.5 ms)

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * chip // sr) % L
    # cosine-phased BOC: sign(cos 2π·sub·t) → quarter-period pattern +,−,−,+
    q = (n * 4 * sub // sr) % 4
    boc = np.where(q % 3 == 0, 1.0, -1.0).astype(np.float32)   # q∈{0,3}→+1, {1,2}→−1
    s = (bip[chip_idx] * boc).astype(np.float32)

    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = s
    iq.imag = 0.0
    return iq, n_samples


# ── Self-test (surrogate + modulation; there's no PRS code to check) ───────────

def _self_test() -> int:
    ok = True
    code = surrogate_code()
    period_ok = len(code) == (1 << LFSR_DEGREE) - 1
    balance_ok = sum(code) == (1 << (LFSR_DEGREE - 1))

    state, seen, steps = 1, set(), 0
    for _ in range(1 << LFSR_DEGREE):
        if state in seen:
            break
        seen.add(state)
        fb = 0
        for t in LFSR_TAPS:
            fb ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (fb << (LFSR_DEGREE - 1))
        steps += 1
    maximal = steps == (1 << LFSR_DEGREE) - 1
    good = period_ok and balance_ok and maximal
    ok = ok and good
    print(f"surrogate m-seq: len={len(code)} ones={sum(code)} maximal={maximal} "
          f"[{'OK' if good else 'FAIL'}]")

    # cosine-BOC sign pattern over one sub-carrier period at 4 samples/period.
    try:
        import numpy as np
        # E1A at 60 MHz: 4 samples per 15 MHz sub-carrier period → pattern +,−,−,+
        n = np.arange(8, dtype=np.int64)
        q = (n * 4 * 15 // 60) % 4
        boc = np.where(q % 3 == 0, 1, -1)
        pat_ok = list(boc[:4]) == [1, -1, -1, 1] and boc.sum() == 0
        ok = ok and pat_ok
        print(f"cosine-BOC pattern (4/period): {list(boc[:4])} DC-free={boc.sum()==0} "
              f"[{'OK' if pat_ok else 'FAIL'}]")
        # buffer builds and is unit-magnitude
        iq, n_s = build_prs_buffer("E1A", 61.44e6)
        mag_ok = bool(np.all(np.abs(iq) == 1.0))
        ok = ok and mag_ok
        print(f"E1A buffer @61.44 MHz: {n_s} samples ({n_s*8/1e6:.1f} MB), unit-mag={mag_ok} "
              f"[{'OK' if mag_ok else 'FAIL'}]")
    except ImportError:
        print("buffer/BOC: skipped (no NumPy here)")

    print("ALL PRS SURROGATE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo PRS (E1-A/E6-A) spectral-surrogate channel-task — correct "
               "cosine-BOC modulation over a public m-sequence (NOT the classified "
               "PRS code) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .choice("-Band", "--band", options=["E1A", "E6A"], default="E1A",
                help="E1A → BOC_cos(15,2.5) @ 1575.42 MHz; E6A → BOC_cos(10,5) @ "
                     "1278.75 MHz. Sets the carrier. Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=20.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=61.44, required=True,
                help="Target channel sample rate (negotiated). E1A ~61 MHz, E6A "
                     "~41 MHz to span the BOC lobes. Fixed per run.")
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
    iq, n_samples = build_prs_buffer(args.band, rate_hz)
    path = write_shm(iq, "gal_prs")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path, "label": f"gal_prs {args.band}"}
    info = [f"band           : {args.band}  {BANDS[args.band]['label']} (SURROGATE)",
            f"buffer         : {n_samples} samples ({n_samples*8/1e6:.1f} MB, 1 code period)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    args.freq = BANDS[args.band]["carrier"]      # carrier is set by the band
    return run_channel(script, args, build, title="Galileo PRS surrogate channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

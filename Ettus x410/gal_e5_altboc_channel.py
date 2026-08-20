#!/usr/bin/env python3
"""
gal_e5_altboc_channel — Galileo E5 AltBOC(15,10) channel-task (expanded mode).

Transmit the FULL wideband Galileo E5 signal — both sidebands at once — as a
single constant-envelope AltBOC(15,10) waveform centred on the E5 carrier
(1191.795 MHz). E5a (data E5a-I + pilot E5a-Q) forms the lower sideband
(−15.345 MHz → 1176.45 MHz) and E5b (E5b-I + E5b-Q) the upper sideband
(+15.345 MHz → 1207.14 MHz), exactly as a real Galileo satellite radiates them,
using the real 10.23 Mcps tiered ranging codes (OS SIS ICD v2.2). The two halves
can still be received independently as QPSK at 1176.45 / 1207.14 MHz.

This is the widest GNSS signal here (~51 MHz occupied → ~61.44 MHz sample rate) —
the reason the X410 engine drives per-channel rates. The E5 code generation
(primaries + tiered secondaries) is shared with gal_e5_channel; this module adds
the AltBOC 8-PSK modulation.

AltBOC(15,10), ICD §2.3.1 (Eq. 6): a constant-envelope 8-PSK signal. Each output
sample is exp(j·π/4·k), where k is a pure look-up (ICD Table 7) from the four
tiered code chip signs and the sub-period index iTs ∈ {0..7} of the 15.345 MHz
sub-carrier — chip and sub-period indices computed by ZOH at the negotiated rate.
The full repeating period is 100 ms → a ~49 MB buffer at 61.44 MHz, one seamless
expanded loop.

⚠  RF SAFETY / LEGAL: E5 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gal_e5_altboc_channel.py --channel 0 --svid 1 --gain 55 --amplitude 0
    gal_e5_altboc_channel.py --self-test        # codes + AltBOC table, no engine
    gal_e5_altboc_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm
# The E5 code machinery (primaries + tiered secondaries) is shared verbatim.
from gal_e5_channel import primary_code, secondary_code, PRIMARY_LEN, CHIP_RATE_HZ


# ── Constants ─────────────────────────────────────────────────────────────────

E5_HZ = 1191.795e6             # E5 band centre (between E5a and E5b)
SUBCARRIER_HZ = 15.345e6       # AltBOC side-band sub-carrier rate (15×1.023)
EPOCHS_PER_LOOP = 100          # 100 primary epochs (1 ms) = 100 ms tiered period

FREQUENCIES = {"Galileo E5 (1191.795 MHz)": E5_HZ}
SAMPLE_RATES_MHZ = {"40.96 MHz (marginal)": 40.96, "61.44 MHz (default)": 61.44}

# AltBOC 8-PSK phase-state look-up table (ICD Table 7). Rows = sub-period index
# iTs 0..7; columns = 8·(e_aI>0) + 4·(e_bI>0) + 2·(e_aQ>0) + 1·(e_bQ>0); value = k
# in 1..8 giving s = exp(j·π/4·k).
ALTBOC_K = [
    [5, 4, 4, 3, 6, 3, 1, 2, 6, 5, 7, 2, 7, 8, 8, 1],
    [5, 4, 8, 3, 2, 3, 1, 2, 6, 5, 7, 6, 7, 4, 8, 1],
    [1, 4, 8, 7, 2, 3, 1, 2, 6, 5, 7, 6, 3, 4, 8, 5],
    [1, 8, 8, 7, 2, 3, 1, 6, 2, 5, 7, 6, 3, 4, 4, 5],
    [1, 8, 8, 7, 2, 7, 5, 6, 2, 1, 3, 6, 3, 4, 4, 5],
    [1, 8, 4, 7, 6, 7, 5, 6, 2, 1, 3, 2, 3, 8, 4, 5],
    [5, 8, 4, 3, 6, 7, 5, 6, 2, 1, 3, 2, 7, 8, 4, 1],
    [5, 4, 4, 3, 6, 7, 5, 2, 6, 1, 3, 2, 7, 8, 8, 1],
]


# ── Baseband buffer (constant-envelope AltBOC, one 100 ms period, ZOH) ─────────

def build_altboc_buffer(svid: int, samp_rate_hz: float):
    """Complex64 constant-envelope AltBOC(15,10) buffer over a whole number of
    100 ms E5 periods (seamless). Each sample = exp(j·π/4·k) from the ICD Table-7
    look-up. Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    sc = int(round(SUBCARRIER_HZ))
    chips_per_loop = PRIMARY_LEN * EPOCHS_PER_LOOP        # 1_023_000
    spp = Fraction(sr * chips_per_loop, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator

    def tiered(component, channel, band):
        prim = np.asarray(primary_code(component, svid), dtype=np.int8)
        sec = np.asarray(secondary_code(band, channel, svid), dtype=np.int8)
        overlay = sec[np.arange(EPOCHS_PER_LOOP) % len(sec)]
        chips = (prim[None, :] ^ overlay[:, None]).reshape(-1)
        return (1 - 2 * chips).astype(np.int8)           # ±1

    e_aI = tiered("E5a-I", "data", "E5a")
    e_aQ = tiered("E5a-Q", "pilot", "E5a")
    e_bI = tiered("E5b-I", "data", "E5b")
    e_bQ = tiered("E5b-Q", "pilot", "E5b")

    idx = np.arange(n_samples, dtype=np.int64)
    chip_of = (idx * cr) // sr
    col = (((e_aI[chip_of] > 0).astype(np.int64) << 3)
           | ((e_bI[chip_of] > 0).astype(np.int64) << 2)
           | ((e_aQ[chip_of] > 0).astype(np.int64) << 1)
           | ((e_bQ[chip_of] > 0).astype(np.int64)))
    iTs = ((idx * (8 * sc)) // sr) % 8
    ktab = np.asarray(ALTBOC_K, dtype=np.int64)
    phase = (math.pi / 4.0) * ktab[iTs, col]
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.cos(phase).astype(np.float32)
    iq.imag = np.sin(phase).astype(np.float32)           # |iq| = 1 (const env)
    return iq, n_samples, n_periods


# ── Self-test (AltBOC table vs direct formula + codes + const-envelope) ────────

def _self_test() -> int:
    import cmath
    ok = True

    # AltBOC table vs the ICD's independent direct sub-carrier formula (Eq. 3–4,
    # Table 6) over all 128 (iTs × quadruple) cases.
    r2 = math.sqrt(2.0)
    AS = [(r2 + 1) / 2, 0.5, -0.5, (-r2 - 1) / 2, (-r2 - 1) / 2, -0.5, 0.5, (r2 + 1) / 2]
    AP = [(-r2 + 1) / 2, 0.5, -0.5, (r2 - 1) / 2, (r2 - 1) / 2, -0.5, 0.5, (-r2 + 1) / 2]
    maxerr = 0.0
    for iTs in range(8):
        for c in range(16):
            eaI = 1 if (c >> 3) & 1 else -1
            ebI = 1 if (c >> 2) & 1 else -1
            eaQ = 1 if (c >> 1) & 1 else -1
            ebQ = 1 if (c >> 0) & 1 else -1
            edaI, edaQ = eaQ * ebI * ebQ, eaI * ebI * ebQ
            edbI, edbQ = ebQ * eaI * eaQ, ebI * eaI * eaQ
            scS, scSd = AS[iTs], AS[(iTs - 2) % 8]
            scP, scPd = AP[iTs], AP[(iTs - 2) % 8]
            s = ((eaI + 1j * eaQ) * (scS - 1j * scSd)
                 + (ebI + 1j * ebQ) * (scS + 1j * scSd)
                 + (edaI + 1j * edaQ) * (scP - 1j * scPd)
                 + (edbI + 1j * edbQ) * (scP + 1j * scPd)) / (2 * r2)
            maxerr = max(maxerr, abs(s - cmath.exp(1j * ALTBOC_K[iTs][c] * math.pi / 4)))
    table_ok = maxerr < 1e-9
    ok = ok and table_ok
    print(f"AltBOC table vs direct formula: max|Δ|={maxerr:.2e} over 128 cases "
          f"[{'OK' if table_ok else 'FAIL'}]")

    # Spot-check the shared E5 codes are the real ones (via first-24 chips).
    codes_ok = True
    for comp, want in (("E5a-I", 0x3CEA9D), ("E5b-Q", 0xE49AF0)):
        first24 = 0
        for b in primary_code(comp, 1)[:24]:
            first24 = (first24 << 1) | b
        codes_ok = codes_ok and first24 == want
    ok = ok and codes_ok
    print(f"shared E5 codes (E5a-I/E5b-Q SVID1 first-24): [{'OK' if codes_ok else 'FAIL'}]")

    # Constant envelope: every sample magnitude is 1.
    try:
        import numpy as np
        iq, n, per = build_altboc_buffer(1, 61.44e6)
        env_ok = bool(np.allclose(np.abs(iq), 1.0, atol=1e-5))
        ok = ok and env_ok
        print(f"const-envelope buffer @61.44 MHz: {n} samples ({n*8/1e6:.1f} MB), "
              f"|iq|=1 everywhere={env_ok} [{'OK' if env_ok else 'FAIL'}]")
    except ImportError:
        print("buffer: skipped (no NumPy here)")

    print("ALL E5 AltBOC CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo E5 AltBOC(15,10) channel-task — the full wideband E5 signal "
               "(both sidebands, constant-envelope 8-PSK, real OS SIS ICD codes) on "
               "one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-SVID", "--svid", min=1, max=50, default=1, required=True,
                 help="Galileo SVID (1..50). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=E5_HZ, required=True, live=True,
                help="RF carrier (E5 band centre 1191.795 MHz). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=40.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=61.44, required=True,
                help="Target channel sample rate (negotiated). AltBOC occupies "
                     "~51 MHz (both ±15.345 MHz sidebands) → 61.44 MHz. Fixed per run.")
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
    iq, n_samples, n_periods = build_altboc_buffer(args.svid, rate_hz)
    path = write_shm(iq, "gal_e5_altboc")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"gal_e5_altboc svid{args.svid}"}
    info = [f"SVID           : {args.svid}   (AltBOC(15,10), both sidebands, const-env)",
            f"buffer         : {n_samples} samples ({n_periods}×100 ms, {n_samples*8/1e6:.1f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="Galileo E5 AltBOC channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

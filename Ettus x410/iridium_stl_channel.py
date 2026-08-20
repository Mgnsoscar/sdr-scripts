#!/usr/bin/env python3
"""
iridium_stl_channel — Iridium/STL burst SPECTRAL-SURROGATE channel-task (X410).

Reproduces the RF/temporal footprint of an Iridium downlink burst in the STL
(Satellite Time & Location) band — DQPSK bursts at 25 ksps with realistic burst
length and idle gaps (TDMA duty cycle), RRC pulse-shaped — over a SURROGATE random
payload.

What this is (and is not)
─────────────────────────
STL (Satelles/Iridium) is a commercial PNT service whose burst PAYLOAD is
proprietary and encrypted; it is not published and is not reproduced here. STL is
also NOT a PRN/CDMA GNSS signal — it rides on Iridium's TDMA/FDMA L-band bursts.
This task therefore transmits a SPECTRAL SURROGATE: the correct Iridium physical-
layer modulation (BPSK preamble + differential-QPSK payload, 25 ksps, RRC shaping,
bursted with realistic idle gaps) driven by a PUBLIC pseudo-random payload standing
in for the encrypted STL content. It reproduces the STL-band spectral + burst
footprint for conducted front-end / filter / band-occupancy / RFI testing, and is
noise-like to a real STL receiver — it carries none of the STL data.

The Iridium PHY (25 ksps DQPSK bursts, ~41.67 kHz channel raster, 1616–1626.5 MHz)
is the public reverse-engineered layer (gr-iridium / Iridium Toolkit); STL rides on
the simplex/broadcast channels near 1626 MHz.

Each burst = `preamble` BPSK symbols (a tone for receiver sync) + `payload` DQPSK
symbols, one burst per `burst_period` (the ~90 ms Iridium frame → the TDMA duty).
The buffer holds several frames of independent random payload and loops (expanded
mode). Needs only a low sample rate — one channel is ~35 kHz wide.

⚠  RF SAFETY / LEGAL: the Iridium band is licensed. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    iridium_stl_channel.py --channel 0 --freq 1626.270833e6 --gain 45 --amplitude 0
    iridium_stl_channel.py --self-test        # modulation + duty + occupancy, no engine
    iridium_stl_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

SYMBOL_RATE_HZ = 25_000        # Iridium burst symbol rate (25 ksps)
CHANNEL_SPACING_HZ = 41_667    # FDMA channel raster (reference only)

# STL rides on the Iridium simplex/broadcast channels near 1626 MHz.
FREQUENCIES = {
    "Iridium Ring Alert (1626.270833 MHz)": 1626.270833e6,
    "Iridium/STL simplex band centre (1626.25 MHz)": 1626.25e6,
    "Iridium duplex band centre (1621.25 MHz)": 1621.25e6,
}
SAMPLE_RATES_MHZ = {"1.024 MHz (default)": 1.024, "2.048 MHz": 2.048, "0.512 MHz": 0.512}


# ── Burst synthesis (BPSK preamble + DQPSK payload, RRC-shaped) ────────────────

def _rrc(t, beta: float):
    """Root-raised-cosine impulse response, `t` in symbol periods (array)."""
    import numpy as np
    out = np.zeros_like(t, dtype=np.float64)
    z = np.isclose(t, 0.0)
    out[z] = 1.0 - beta + 4.0 * beta / np.pi
    if beta > 0:
        sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        out[sing] = (beta / np.sqrt(2.0)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    else:
        sing = np.zeros_like(t, dtype=bool)
    m = ~(z | sing)
    tm = t[m]
    num = np.sin(np.pi * tm * (1 - beta)) + 4 * beta * tm * np.cos(np.pi * tm * (1 + beta))
    den = np.pi * tm * (1 - (4 * beta * tm) ** 2)
    out[m] = num / den
    return out


def _burst_symbols(preamble_n: int, payload_n: int, rng):
    """One burst's complex unit symbols: a BPSK alternating preamble (a sync tone)
    followed by a differentially-encoded QPSK random payload."""
    import numpy as np
    pre = np.where(np.arange(preamble_n) % 2 == 0, 1.0, -1.0).astype(np.complex128)
    d = rng.integers(0, 4, size=payload_n)               # 2 bits/symbol → 0..3
    phase = np.cumsum(d) * (np.pi / 2.0)                  # DQPSK: differential phase
    pay = np.exp(1j * phase)
    return np.concatenate([pre, pay])


def build_iridium_buffer(samp_rate_hz: float, preamble_n: int, payload_n: int,
                         burst_period_ms: float, num_frames: int, rolloff: float,
                         seed: int):
    """Complex64 buffer of `num_frames` Iridium-style bursts (one per burst_period),
    RRC-shaped at the symbol rate and separated by idle gaps. Returns
    (iq, n_samples, burst_ms, duty)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    spsym = samp_rate_hz / SYMBOL_RATE_HZ                 # samples per symbol (fractional ok)
    frame_samps = int(round(burst_period_ms * 1e-3 * samp_rate_hz))
    total = frame_samps * num_frames
    out = np.zeros(total, dtype=np.complex128)
    span = 8                                              # RRC pulse half-span, symbols

    for fr in range(num_frames):
        syms = _burst_symbols(preamble_n, payload_n, rng)
        base = fr * frame_samps
        for k, s in enumerate(syms):
            center = base + k * spsym
            i0 = max(0, int(np.floor(center - span * spsym)))
            i1 = min(total, int(np.ceil(center + span * spsym)) + 1)
            idx = np.arange(i0, i1)
            out[idx] += s * _rrc((idx - center) / spsym, rolloff)

    peak = float(np.max(np.abs(out))) or 1.0
    iq = (out * (0.9 / peak)).astype(np.complex64)        # headroom under full scale
    burst_ms = (preamble_n + payload_n) / SYMBOL_RATE_HZ * 1e3
    duty = burst_ms / burst_period_ms
    return iq, total, burst_ms, duty


# ── Self-test (modulation + duty + occupied bandwidth; no engine) ──────────────

def _self_test() -> int:
    ok = True

    # RRC symmetry + peak.
    try:
        import numpy as np
        t = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5])
        r = _rrc(t, 0.4)
        rrc_ok = (abs(r[0] - (1 - 0.4 + 4 * 0.4 / np.pi)) < 1e-9
                  and np.allclose(r[1], r[2]) and np.allclose(r[3], r[4]))
        ok = ok and rrc_ok
        print(f"RRC: peak={r[0]:.4f} symmetric={bool(np.allclose(r[1],r[2]))} "
              f"[{'OK' if rrc_ok else 'FAIL'}]")

        # DQPSK symbols are unit magnitude; preamble is ±1.
        syms = _burst_symbols(64, 256, np.random.default_rng(1))
        mag_ok = np.allclose(np.abs(syms), 1.0)
        ok = ok and mag_ok
        print(f"symbols: {len(syms)} unit-magnitude={mag_ok} [{'OK' if mag_ok else 'FAIL'}]")

        # Build a buffer; check the duty cycle and the occupied bandwidth.
        iq, n, burst_ms, duty = build_iridium_buffer(1.024e6, 64, 256, 90.0, 4, 0.4, 7)
        # duty: fraction of samples with signal energy above a small threshold.
        active = float(np.mean(np.abs(iq) > 0.02))
        duty_ok = abs(active - duty) < 0.06
        ok = ok and duty_ok
        print(f"buffer: {n} samples, burst {burst_ms:.1f} ms, duty {duty*100:.1f}% "
              f"(measured active {active*100:.1f}%) [{'OK' if duty_ok else 'FAIL'}]")

        # Occupied bandwidth: ~99% of energy within ±(1+beta)/2·Rs ≈ ±17.5 kHz.
        one = iq[:int(round(1.024e6 * 90e-3))]           # one frame
        S = np.abs(np.fft.fftshift(np.fft.fft(one))) ** 2
        f = np.fft.fftshift(np.fft.fftfreq(len(one), 1.0 / 1.024e6))
        inband = S[np.abs(f) <= 25e3].sum() / S.sum()
        bw_ok = inband > 0.98
        ok = ok and bw_ok
        print(f"occupancy: {inband*100:.1f}% of energy within ±25 kHz "
              f"(25 ksps RRC) [{'OK' if bw_ok else 'FAIL'}]")
    except ImportError:
        print("synthesis: skipped (no NumPy here)")

    print("ALL IRIDIUM/STL SURROGATE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Iridium/STL burst spectral-surrogate channel-task — realistic DQPSK "
               "bursts (25 ksps, RRC, TDMA duty) over a PUBLIC random payload (NOT "
               "the proprietary STL content) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1626.270833e6, required=True, live=True,
                help="Channel carrier in the Iridium/STL band. Live (retunes).")
        .integer("-Payload-symbols", "--payload_symbols", min=16, max=4000, default=256,
                 help="DQPSK payload symbols per burst. Fixed per run.")
        .integer("-Preamble-symbols", "--preamble_symbols", min=8, max=512, default=64,
                 help="BPSK preamble (sync tone) symbols per burst. Fixed per run.")
        .number("-Burst-period", "--burst_period", unit="ms", min=1.0, max=1000.0,
                default=90.0, help="Time between bursts (Iridium frame ≈ 90 ms) — "
                     "sets the TDMA duty cycle. Fixed per run.")
        .integer("-Frames", "--frames", min=1, max=64, default=8,
                 help="Independent random-payload frames in the looped buffer (more "
                      "= less obvious repetition). Fixed per run.")
        .number("-Rolloff", "--rolloff", min=0.05, max=1.0, default=0.4,
                help="RRC pulse-shaping roll-off. Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=0.2, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=1.024, required=True,
                help="Target channel sample rate (negotiated). A channel is ~35 kHz "
                     "wide; ~1 MHz is plenty. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=45,
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
    iq, n, burst_ms, duty = build_iridium_buffer(
        rate_hz, args.preamble_symbols, args.payload_symbols, args.burst_period,
        args.frames, args.rolloff, seed=1)
    path = write_shm(iq, "iridium_stl")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path, "label": "iridium_stl"}
    info = [f"burst          : {args.preamble_symbols}+{args.payload_symbols} sym "
            f"({burst_ms:.1f} ms) every {args.burst_period:g} ms → {duty*100:.0f}% duty",
            f"buffer         : {n} samples ({args.frames} frames, {n*8/1e6:.1f} MB) "
            f"(SURROGATE payload)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="Iridium/STL surrogate channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
gnss_acq — hardware-free acquisition checks for X410 channel-task IQ buffers.

Every channel-task builds its baseband at a rate the engine *negotiates* from a
fixed master clock, so the realized rate is rarely an exact integer multiple of
the chip rate (e.g. GPS C/A wants 20.46 MHz but a 245.76 MHz clock only offers
20.48). This module measures whether that still yields a signal a GNSS receiver
would acquire — by correlating one code period against a clean replica and
reporting the acquisition peak-to-sidelobe ratio, the code-phase lag, and (for
CDMA constellations) cross-code isolation.

Pure NumPy, no UHD, no hardware — so it runs in CI and in each task's --self-test.
"""
from __future__ import annotations

import numpy as np


def circ_corr_mag(s, r):
    """|circular cross-correlation| of two equal-ish length complex buffers, via
    FFT. Truncates to the shorter length (both are one seamless code period)."""
    n = min(len(s), len(r))
    return np.abs(np.fft.ifft(np.fft.fft(s[:n]) * np.conj(np.fft.fft(r[:n]))))


def peak_to_sidelobe_db(iq, replica, samples_per_chip, main_lobe_chips=1.5):
    """Acquisition quality: 20·log10(peak / worst sidelobe), excluding the full
    ±`main_lobe_chips` correlation triangle around the peak (which is
    samples_per_chip wide when oversampled — a too-small guard would measure the
    peak against its own skirt). Returns (db, peak_lag)."""
    c = circ_corr_mag(iq, replica)
    n = len(c)
    lag = int(np.argmax(c))
    peak = c[lag]
    guard = int(np.ceil(main_lobe_chips * samples_per_chip))
    mask = np.ones(n, bool)
    for d in range(-guard, guard + 1):
        mask[(lag + d) % n] = False
    return 20.0 * np.log10(peak / c[mask].max()), lag


def cross_isolation_db(iq_a, replica_b):
    """Cross-correlation peak of code A against replica B, relative to A's own auto
    peak (dB). For well-chosen CDMA codes this is strongly negative (e.g. GPS Gold
    codes ≈ −24 dB), i.e. no false lock on the wrong code."""
    auto = circ_corr_mag(iq_a, iq_a).max()
    cross = circ_corr_mag(iq_a, replica_b).max()
    return 20.0 * np.log10(cross / auto)


def check_negotiation_fidelity(build_fn, chip_rate_hz, ideal_rate_hz,
                               negotiated_rate_hz, *, min_db=22.0,
                               max_loss_db=1.0, label=""):
    """Assert a negotiated (non-integer-samples/chip) rate still acquires as well as
    the ideal integer-samples/chip rate.

    `build_fn(rate_hz) -> np.ndarray[complex64]` builds one seamless code period at
    a sample rate. Prints a one-line report and returns True/False:

      • negotiated peak-to-sidelobe ≥ `min_db`, and
      • within `max_loss_db` of the ideal-rate peak-to-sidelobe.
    """
    ideal = np.asarray(build_fn(ideal_rate_hz))
    negot = np.asarray(build_fn(negotiated_rate_hz))
    db_i, _ = peak_to_sidelobe_db(ideal, ideal, ideal_rate_hz / chip_rate_hz)
    db_n, lag = peak_to_sidelobe_db(negot, negot, negotiated_rate_hz / chip_rate_hz)
    loss = db_i - db_n
    ok = (db_n >= min_db) and (loss <= max_loss_db)
    tag = f"{label} " if label else ""
    print(f"{tag}fidelity: ideal {ideal_rate_hz/1e6:g}MHz={db_i:.2f}dB  "
          f"negotiated {negotiated_rate_hz/1e6:g}MHz={db_n:.2f}dB  "
          f"(loss {loss:.2f}dB, lag {lag}) [{'OK' if ok else 'FAIL'}]")
    return ok

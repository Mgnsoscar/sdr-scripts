#!/usr/bin/env python3
"""
M-code-like BOC(10,5) transmitter for GNU Radio + UHD (Ettus B200-mini family).

Reproduces the GPS **M-code modulation** — sine-phased **BOC(10,5)**: a ±1
spreading code at 5.115 Mcps multiplied by a 10.23 MHz square subcarrier — which
gives M-code's characteristic **split spectrum** (two lobes at ±10.23 MHz around
the carrier, ~30 MHz total). Precomputed and replayed from a file so a Raspberry
Pi can sustain the 40–60 MS/s needed to carry it (same recipe as gps_l1ca_tx.py).

⚠  NOT the real M-code. The actual military spreading sequence is CLASSIFIED and
   encrypted and cannot be generated here. This uses an UNCLASSIFIED surrogate
   PRN (a GPS C/A Gold code) under the BOC(10,5) subcarrier, so the RF/spectral
   shape matches M-code but the signal is a test surrogate — it is not, and
   cannot be, tracked as the real military code. Use it for front-end / spectrum
   / interference testing only.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) and L2 (1227.60 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup (cable + attenuators) you are
   LICENSED / AUTHORISED to use — never radiate over the air.

BOC(10,5) construction
──────────────────────
  s(t) = code(t) · sc(t)
    code(t) : surrogate Gold code (±1) at fc = 5·1.023 = 5.115 Mcps
    sc(t)   : sine-phased square subcarrier at fsub = 10·1.023 = 10.23 MHz
Real baseband (I = ±1, Q = 0), like a BPSK PRN but split by the subcarrier.
Because fsub = 2·fc, the subcarrier is exactly commensurate with the code, so
whenever a whole code period is an integer number of samples the buffer loops
with no seam (40 MS/s → 8000 samples/period; 60 → 12000).

Sample-rate note: the main lobes sit at ±10.23 MHz, so fs ≥ ~40 MHz captures
them. The square subcarrier's 3rd harmonic lobes are at ±30.69 MHz; at 40 MS/s
those alias near the main lobes, so **60 MS/s is noticeably cleaner** for
BOC(10,5). A true M-code simulator would band-limit; this ships the full square
subcarrier (what M-code actually transmits).

Why it runs on a Pi, and live tuning: see gps_l1ca_tx.py. In short — precompute +
loop from a /dev/shm file, sc8 over the wire, quiet, 1:1 master clock. Level set
in dBm (--power) with a live RF on/off (--rf); see the USER CALIBRATION block. PRN
/ carrier / sample rate / otw are fixed per run (BOC(10,5)'s code and subcarrier
rates are fixed by definition).

CLI
───
    mcode_boc_tx.py --prn 5 --freq 1.57542e9 --power -30 --sample_rate 60
    mcode_boc_tx.py --self-test        # verify the Gold code + sizing, no hardware
    mcode_boc_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve. Absent it, the baked
# USER CALIBRATION constants below are used (unchanged behaviour). See the agent's
# docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1_mcode"


# ═══════════════════════════════════════════════════════════════════════════════
# USER CALIBRATION — FALLBACK CONSTANTS (used only when the unit has no calibration)
# ═══════════════════════════════════════════════════════════════════════════════
# You set the transmit level in dBm. That only works if the script knows how the
# SDR's gain maps to real output power, which you establish once with a spectrum
# analyser: leave AMPLITUDE at the value below, run with --power at its maximum
# (that commands GAIN_AT_MAX_DB), measure the actual output power at the SDR RF
# port, and put that number in OUTPUT_POWER_DBM. From that anchor the script maps
# any requested power to a gain (1 dB gain ≈ 1 dB power, across the B200's linear
# range). CABLE_LOSS_DB / AMPLIFIER_GAIN_DB describe the RF chain AFTER the port,
# so the number you dial in is the power delivered at the far end.

OUTPUT_POWER_DBM = -20.0    # max output (dBm) at GAIN_AT_MAX_DB and AMPLITUDE — MEASURE THIS
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands
CABLE_LOSS_DB = 0.0         # cabling insertion loss after the SDR port (positive dB)
AMPLIFIER_GAIN_DB = 0.0     # external amplifier gain after the SDR port (positive dB)

# Fixed baseband digital amplitude (0..1). NOT a user control: OUTPUT_POWER_DBM is
# calibrated at THIS amplitude, so changing it invalidates the dBm↔gain mapping —
# if you change it, re-measure OUTPUT_POWER_DBM at GAIN_AT_MAX_DB.
AMPLITUDE = 0.8

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75

# Derived delivered-power limits (computed — do not edit).
MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) that puts `delivered_dbm` at the far end of the RF chain, clamped
    to [0, GAIN_AT_MAX_DB] so it can never exceed the calibrated maximum."""
    port_dbm = float(delivered_dbm) + CABLE_LOSS_DB - AMPLIFIER_GAIN_DB
    gain = GAIN_AT_MAX_DB + (port_dbm - OUTPUT_POWER_DBM)
    return max(0.0, min(GAIN_AT_MAX_DB, gain))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) for an actual hardware gain — to report what the radio
    really settled on after quantisation."""
    port_dbm = OUTPUT_POWER_DBM - (GAIN_AT_MAX_DB - float(gain_db))
    return port_dbm - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB


# ── Constants ─────────────────────────────────────────────────────────────────

L1_HZ = 1575.42e6
L2_HZ = 1227.60e6

CODE_LEN = 1023                 # surrogate Gold-code length (chips)
CODE_RATE_HZ = 5_115_000        # BOC(10,5) code rate: 5 × 1.023 Mcps
SUBCARRIER_HZ = 10_230_000      # BOC(10,5) subcarrier: 10 × 1.023 MHz (= 2 × code rate)

FREQUENCIES = {
    "GPS L1 (1575.42 MHz)": L1_HZ,
    "GPS L2 (1227.60 MHz)": L2_HZ,
}

# GPS ICD-200 Table 3-Ia G2 tap pairs (1-indexed) — the surrogate spreading code.
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── Surrogate spreading code (GPS C/A Gold code, pure Python) ──────────────────

def ca_code(prn: int) -> list[int]:
    """1023-chip GPS C/A Gold code for a PRN (1..32) as 0/1 — the unclassified
    surrogate standing in for the classified M-code sequence."""
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1..32, got {prn}")
    g1 = [1] * 10
    g2 = [1] * 10
    ta, tb = G2_TAPS[prn]
    out: list[int] = []
    for _ in range(CODE_LEN):
        out.append(g1[9] ^ g2[ta - 1] ^ g2[tb - 1])
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return out


def _self_test() -> int:
    from fractions import Fraction
    ok = True

    # Surrogate Gold code matches the ICD reference for all 32 PRNs.
    for prn in range(1, 33):
        code = ca_code(prn)
        first10 = 0
        for b in code[:10]:
            first10 = (first10 << 1) | b
        good = (len(code) == CODE_LEN and first10 == _FIRST10_OCTAL[prn]
                and sum(code) == 512)
        ok = ok and good
        print(f"PRN {prn:2d}: first10={first10:#06o} "
              f"expect={_FIRST10_OCTAL[prn]:#06o} [{'OK' if good else 'FAIL'}]")

    # BOC(10,5) invariants + seamless sizing at a few rates.
    print(f"fsub == 2*fc: {SUBCARRIER_HZ == 2 * CODE_RATE_HZ}")
    for samp_mhz in (40.0, 50.0, 60.0):
        sr = int(round(samp_mhz * 1e6))
        spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
        n, nper = spp.numerator, spp.denominator
        halfcyc = Fraction(2 * SUBCARRIER_HZ * n, sr)
        seam = halfcyc.denominator == 1 and halfcyc.numerator % 2 == 0
        ok = ok and seam
        print(f"{samp_mhz:g} MHz → {n} samples / {nper} code period(s); "
              f"subcarrier closes: {seam}")
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (one seamless-looping BOC(10,5) period) ────────────────────

def build_boc_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 baseband BOC(10,5) buffer: surrogate Gold code × sine-
    phased 10.23 MHz square subcarrier, sized to a whole number of code periods
    that is also an integer number of samples (seamless loop). Unit magnitude
    (amplitude applied live downstream). Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    spp = Fraction(sr * CODE_LEN, CODE_RATE_HZ)
    n_periods = spp.denominator
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code                       # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * CODE_RATE_HZ // sr) % CODE_LEN    # exact chip mapping
    # Sine-phased square subcarrier: +1 for the first half-period, −1 the next, …
    sub = np.where((n * (2 * SUBCARRIER_HZ) // sr) % 2 == 0, 1.0, -1.0)

    iq = (bipolar[chip_idx] * sub).astype(np.complex64)   # real BPSK×subcarrier
    return iq, n_samples, n_periods


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file: str, center_freq_hz: float, samp_rate_hz: float,
                     gain_db: float, amplitude: float, otw_format: str,
                     extra_args: str):
    from gnuradio import gr, blocks, uhd

    class McodeTx(gr.top_block):
        def __init__(self):
            super().__init__("M-code BOC(10,5) TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=otw_format,
                                channels=[0]),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return McodeTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("M-code-like BOC(10,5) transmitter (surrogate Gold code under a "
               "10.23 MHz subcarrier), file-replay at high sample rate. Surrogate "
               "test signal — not the classified M-code. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="Surrogate Gold-code index (1..32). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L1_HZ, required=True,
                help="RF carrier (M-code is on L1 and L2). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(MIN_DELIVERED_DBM, 2), max=round(MAX_DELIVERED_DBM, 2),
                default=round(MAX_DELIVERED_DBM, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=30.0, max=61.44,
                default=40.0,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "≥40 MHz for the ±10.23 MHz lobes; 60 is cleaner. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load (needed at these "
                     "rates on a Pi); sc16 for more dynamic range.")
        .choice("-RF", "--rf", options=["on", "off"], default="on",
                required=False, live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them. Change the power (or the "
                     "calibration gain) while OFF and it takes effect when you turn ON.")
    )
    # ── CALIBRATION KNOB (normally commented OUT) ───────────────────────────────
    # Uncomment to expose a raw TX-gain slider (dB) so you can measure output power
    # vs gain on a spectrum analyser and fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB
    # above. While present it OVERRIDES --power (whichever you touch last wins).
    # s = s.number(
    #     "-Cal-gain", "--gain", unit="dB",
    #     min=0, max=HW_MAX_GAIN_DB, default=HW_MAX_GAIN_DB,
    #     required=False, live=True,
    #     help="CALIBRATION ONLY — set SDR TX gain directly, bypassing the dBm "
    #          "mapping. Comment out again for normal dBm operation.")
    return s


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()
    samp_rate_hz = args.sample_rate * 1e6

    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else the baked constants above (identical to the old single-anchor behaviour).
    pmap = PowerMap.load(PowerMap.from_linear(
        min_gain_db=0.0, max_gain_db=GAIN_AT_MAX_DB,
        min_power_dbm=MIN_DELIVERED_DBM, max_power_dbm=MAX_DELIVERED_DBM,
        amplitude=AMPLITUDE))
    amplitude = pmap.amplitude

    # A raw calibration gain (the normally-commented --gain knob) overrides the dBm
    # mapping when present, so you can measure output power at a chosen gain.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else pmap.gain_for_power(args.power)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="mcode_boc_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp, nper = build_boc_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"mcode_prn{args.prn}_{args.sample_rate:g}MHz.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(
        iq_file=iq_file, center_freq_hz=args.freq, samp_rate_hz=samp_rate_hz,
        gain_db=gain_db, amplitude=amplitude, otw_format=args.otw,
        extra_args="")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── M-code-like BOC(10,5) TX (surrogate) ────────────────────")
    print(f"  surrogate PRN  : {args.prn}  (NOT the classified M-code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BOC(10,5) — {CODE_RATE_HZ/1e6:g} Mcps code, "
          f"{SUBCARRIER_HZ/1e6:g} MHz subcarrier (lobes at ±{SUBCARRIER_HZ/1e6:g} MHz)")
    print(f"  buffer         : {nsamp} samples ({nper} code period(s))")
    print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  otw            : {args.otw}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        # power/gain edits are staged into state["gain"] and only reach the radio
        # when RF is on; the --rf toggle mutes/restores gain AND amplitude.
        if name == "power":
            state["gain"] = pmap.gain_for_power(float(value))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain()), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("gain", round(tb.actual_gain(), 2))
            else:
                ctrl.report("gain", round(state["gain"], 2))
        elif name == "rf":
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            if on:
                tb.set_amplitude(amplitude)
                tb.set_gain(state["gain"])
            else:
                tb.set_gain(0.0)
                tb.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                apply_change(change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
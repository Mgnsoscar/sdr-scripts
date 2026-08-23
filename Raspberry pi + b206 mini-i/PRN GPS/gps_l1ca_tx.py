#!/usr/bin/env python3
"""
GPS L1 C/A transmitter for GNU Radio + UHD (Ettus B200-mini family).

Purpose
───────
Transmit a BPSK-modulated GPS L1 C/A Gold code (1.023 Mcps) at the L1 carrier
(1575.42 MHz), at a high enough sample rate to carry the ~2 MHz spreading code —
on a Raspberry Pi, where synthesising IQ at runtime can't keep up.

This is one of three scripts split out from the old combined gps_prn_tx.py:
    gps_l1ca_tx.py   — GPS L1 C/A          (this file, 1.023 Mcps @ L1)
    gps_l1p_tx.py    — GPS L1 P-code       (10.23 Mcps @ L1)
    gps_l2p_tx.py    — GPS L2 P-code       (10.23 Mcps @ L2)

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded/conducted setup (cable + attenuators into a receiver or spectrum
   analyser) that you are LICENSED / AUTHORISED to use. Radiating a PRN over the
   air can jam or spoof real GNSS receivers and is illegal in most places.

Setting the level in dBm (not raw gain)
───────────────────────────────────────
You dial the transmit level in absolute power with --power (dBm at the delivered
plane), not the SDR's arbitrary gain number. That works because the script is told
how the SDR's gain maps to real output power, two ways:
  • Per-unit calibration (preferred). A task that sets SDR_CAL_SIGNAL_ID to
    CAL_SIGNAL_ID gets this unit's MEASURED gain→power curve injected at
    $SDR_CALIBRATION_FILE; --power then reads through that curve at the unit's real
    operating plane (e.g. EIRP). See the agent's docs/calibration.md.
  • Baked fallback. With no calibration, the USER CALIBRATION constants below define
    a single-anchor slope-1 line (1 dB gain ≈ 1 dB power) — the original behaviour.
--gain instead commands the SDR's raw TX gain directly (RELATIVE power), bypassing
the dBm mapping; when given it overrides --power.

How it hits high rates on a Pi (the three levers)
─────────────────────────────────────────────────
1. PRECOMPUTE + LOOP, don't generate at runtime. One whole number of code periods
   is built once at startup, written to /dev/shm (RAM-backed, no SD-card wear), and
   replayed with blocks.file_source(repeat=True). At runtime the CPU only DMAs
   bytes to USB — no per-sample Python/NumPy math.
2. sc8 OVER THE WIRE. cpu_format=fc32 but otw_format=sc8 halves the USB payload. A
   PRN is constant-modulus BPSK, so 8-bit I/Q costs nothing that matters here.
3. NO STATUS UPDATES MID-RUN. UHD fastpath markers off, logging off; nothing is
   printed once streaming starts (status writes under load *cause* underflows).

1:1 master clock
────────────────
master_clock_rate is pinned equal to the sample rate, so UHD runs the AD9361 1:1
with no FPGA resampling and no rate coercion — you get exactly the rate you asked
for, so samples-per-chip and the loop length stay exact.

CLI
───
    gps_l1ca_tx.py --prn 5 --power -30 --samp_rate 20.46   # absolute dBm (calibrated)
    gps_l1ca_tx.py --prn 5 --gain 60                       # relative: raw SDR gain
    gps_l1ca_tx.py --self-test        # verify the Gold-code generator, no hardware
    gps_l1ca_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# UHD/GNU Radio must be quiet BEFORE the libraries load (they read these at
# import). The heavy imports live inside main(), so setting them here takes
# effect. A task env (configs/tasks.yaml) can still override any of them.
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")   # no UHD console logging
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")  # no "UUUU" underflow spam
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")        # skip slow pref scan

# paramkit is pure-Python and always importable (agent puts it on PYTHONPATH).
# NumPy and GNU Radio are imported lazily so --self-test / --describe-params run
# anywhere, including CI and dev boxes without a radio stack.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the baked USER CALIBRATION constants below are used
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1_ca"


# ═══════════════════════════════════════════════════════════════════════════════
# USER CALIBRATION — FALLBACK CONSTANTS (used only when the unit has no calibration)
# ═══════════════════════════════════════════════════════════════════════════════
# You set the transmit level in dBm. That only works if the script knows how the
# SDR's gain maps to real output power, which you establish once with a spectrum
# analyser:
#
#   1. Leave AMPLITUDE at the value below and run with --power at its maximum
#      (that commands GAIN_AT_MAX_DB).
#   2. Measure the actual output power at the SDR's RF port on your analyser.
#   3. Put that measured number in OUTPUT_POWER_DBM.
#
# From that single anchor the script maps any requested power to a gain — 1 dB of
# gain ≈ 1 dB of output power, which holds across the B200's linear range.

# Max output power (dBm) measured at the SDR RF port when the gain is GAIN_AT_MAX_DB
# and the baseband amplitude is AMPLITUDE (below). Your calibration anchor.
OUTPUT_POWER_DBM = -20.0

# The TX gain (dB) that produced OUTPUT_POWER_DBM. Also the HARD CEILING: the script
# never commands a gain above this, so it can never exceed OUTPUT_POWER_DBM at the
# port. (B200-mini TX gain range is 0..89.75 dB.)
GAIN_AT_MAX_DB = 89.75

# RF chain BETWEEN the SDR port and the plane where you want the power delivered
# (e.g. after a cable run and/or an external PA). Both in dB, both positive:
#   CABLE_LOSS_DB     — insertion loss of the cabling (lowers delivered power)
#   AMPLIFIER_GAIN_DB — gain of an external amplifier (raises delivered power)
# Set either to 0 if you don't have that element. Together they shift the delivered
# power and set the maximum dBm the --power field will accept.
CABLE_LOSS_DB = 0.0
AMPLIFIER_GAIN_DB = 0.0

# Fixed baseband digital amplitude (0..1). NOT a user control: OUTPUT_POWER_DBM is
# calibrated at THIS amplitude, so changing it invalidates the dBm↔gain mapping.
# If you change it, you MUST re-measure OUTPUT_POWER_DBM at GAIN_AT_MAX_DB.
AMPLITUDE = 0.8

# ── Derived power limits (computed from the calibration above — do not edit) ─────
# Delivered power = port power − cable loss + amplifier gain. The chain delivers the
# most at GAIN_AT_MAX_DB and the least at gain 0 (the 1:1 gain→power slope).
MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB (your chosen operating ceiling). The calibration gain knob
# sweeps the full 0..HW_MAX_GAIN_DB range so you can find and measure the gain that
# produces your clean maximum power.
HW_MAX_GAIN_DB = 89.75


_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else the baked constants above. Cached, so build_script
    and main share one — and so --power's schema bounds match the real operating
    range (calibrated → e.g. EIRP; else the baked SDR-port range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
    return _PMAP


# ── Signal constants (fixed for this script — it IS GPS L1 C/A) ─────────────────

CARRIER_HZ = 1575.42e6         # GPS L1
CODE_RATE_MCPS = 1.023         # true GPS C/A chip rate (~2 MHz null-to-null)
SIGNAL_NAME = "GPS L1 C/A"

CODE_LEN = 1023                # chips in a GPS C/A Gold code period

SAMPLE_RATES = {
    "4.092 MHz (Minimum -> Just main lobe + first skirt)": 4.092,
    "20.46 MHz (Default -> Most faithful representation)": 20.46,
    "61.38 MHz (Maximum -> Captures the widest skirts)": 61.38,
}
DEFAULT_SAMPLE_RATE = 20.46

# GPS ICD-200 Table 3-Ia: G2 code-phase tap pairs (1-indexed) selecting each
# satellite's C/A code. Verified against the ICD "first 10 chips" column (--self-test).
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}

# ICD reference: first 10 chips of each PRN's C/A code, as an octal integer.
# Used only by --self-test to prove the generator matches the standard.
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── GPS C/A Gold-code generation (pure Python, no NumPy) ───────────────────────

def ca_code(prn: int) -> list[int]:
    """Return the 1023-chip GPS C/A Gold code for a PRN (1..32) as a list of 0/1.

    Two 10-stage LFSRs (both seeded all-ones):
      G1: x^10 + x^3 + 1
      G2: x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
    The chip is G1's output XOR the sum of two PRN-specific G2 taps.
    """
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1 to 32, got {prn}")
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
    """Verify ca_code() against the ICD reference for all 32 PRNs. Returns 0 on
    success (usable as a process exit code); prints a per-PRN report."""
    ok = True
    for prn in range(1, 33):
        code = ca_code(prn)
        first10 = 0
        for b in code[:10]:
            first10 = (first10 << 1) | b
        good = (len(code) == CODE_LEN and first10 == _FIRST10_OCTAL[prn]
                and sum(code) == 512)
        ok = ok and good
        print(f"PRN {prn:2d}: first10={first10:#06o} "
              f"expect={_FIRST10_OCTAL[prn]:#06o} ones={sum(code)} "
              f"[{'OK' if good else 'FAIL'}]")
    print("ALL PRN CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (one seamless-looping code period) ─────────────────────────

def build_iq_buffer(prn: int, chip_rate_hz: float, samp_rate_hz: float):
    """Build a complex64 baseband buffer holding a whole number of code periods
    that is also an exact integer number of samples, so it loops with no seam.
    BPSK: I = ±1 chip, Q = 0. Baseband amplitude is applied downstream (a
    multiply_const), so the buffer is left at unit magnitude.

    Returns (iq: np.ndarray[complex64], n_samples: int, n_periods: int).
    """
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(chip_rate_hz))
    # Samples per code period as an exact fraction; the number of periods we must
    # tile to reach an integer sample count is its denominator in lowest terms.
    spp = Fraction(sr * CODE_LEN, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator          # == n_periods * (samples per one period)

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code         # 0 → +1, 1 → −1

    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr // sr) % CODE_LEN   # exact zero-order-hold chip mapping
    iq = bipolar[chip_idx].astype(np.complex64)   # Q stays 0 (real BPSK)
    return iq, n_samples, n_periods


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(code_file: str, center_freq_hz: float, samp_rate_hz: float,
                     gain_db: float, amplitude: float, otw_format: str,
                     extra_args: str):
    """Construct the GNU Radio top_block. Imported lazily so the module loads
    without a radio stack for --self-test / --describe-params."""
    from gnuradio import gr, blocks, uhd

    class PrnTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")

            # Pin master clock == sample rate → 1:1, no FPGA resampling/coercion.
            # Enlarge the USB send buffer so the host stays ahead at high rate.
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args

            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(
                    cpu_format="fc32", otw_format=otw_format,
                    channels=[0]
                ),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            self.src = blocks.file_source(gr.sizeof_gr_complex, code_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)   # calibration amplitude
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)          # baseband digital scale (used by RF on/off)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return PrnTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script(
            f"{SIGNAL_NAME} (C/A Gold code) transmitter. Level is set in dBm; edit "
            "the USER CALIBRATION block at the top of the script to match your rig."
        )
        .integer(
            "-PRN", "--prn",
            min=1, max=32,
            default=1,
            required=True,
            help="GPS satellite PRN / Gold code index (1 to 32). Fixed per run."
        )
        .number(
            "-Power", "--power",
            unit="dBm",
            min=round(power_map().min_power_dbm, 2),
            max=round(power_map().max_power_dbm, 2),
            default=round(power_map().max_power_dbm, 2),
            required=False, live=True,
            help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                 "unit's calibration when present (e.g. EIRP), else the baked "
                 "SDR-port scale. Ignored if --gain is given (relative wins). Live."
        )
        .number(
            "-Sample-rate", "--samp_rate",
            unit="MHz", min=1.23, max=61.44,
            default=DEFAULT_SAMPLE_RATE,
            presets=SAMPLE_RATES,
            help="Host/DAC sample rate; master clock is pinned equal to it (1:1). "
                 "Fixed per run."
        )
        .choice(
            "-OTW-format", "--otw",
            options=["sc8", "sc16"],
            default="sc8",
            help="Over-the-wire sample format. sc8 halves USB load (needed for "
                 "high MS/s on a Pi); sc16 for more dynamic range."
        )
        .choice(
            "-RF", "--rf",
            options=["on", "off"], default="on", required=False, live=True,
            help="RF output on/off. OFF mutes the signal (gain AND baseband "
                 "amplitude to 0); ON restores them. Change the power (or the "
                 "calibration gain) while OFF and it takes effect when you turn ON."
        )
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power).
        # This is also the calibration knob — set it while measuring output vs gain
        # on a spectrum analyser to fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB above.
        .number(
            "-Gain", "--gain", unit="dB",
            min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
            help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                 "bypassing the dBm calibration. When given, overrides --power. Live."
        )
    )
    return s


# ── Live-change dispatch ────────────────────────────────────────────────────────

def _apply_live_change(tb, ctrl, state, pmap, amplitude, name: str, value) -> None:
    """Live-tune dispatch. `state` carries the RF on/off flag and the gain that RF-on
    should apply, so power/gain edits made while RF is OFF are staged and reach the
    hardware only when RF is switched ON. `pmap` maps dBm ↔ gain through the active
    calibration (the unit's measured curve, or the baked fallback)."""
    if name == "power":
        # dBm → gain via the calibration; stage it, apply to the radio only if RF is on.
        state["gain"] = pmap.gain_for_power(float(value))
        if state["rf_on"]:
            tb.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain()), 2))
        else:
            ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
    elif name == "gain":
        # Relative / calibration knob: raw TX gain (dB), bypassing the dBm mapping.
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
            tb.set_amplitude(amplitude)   # restore staged level, then unmute gain
            tb.set_gain(state["gain"])
        else:
            tb.set_gain(0.0)
            tb.set_amplitude(0.0)
        ctrl.report("rf", "on" if on else "off")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()

    samp_rate_hz = args.samp_rate * 1e6

    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else the baked constants above (identical to the old single-anchor behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude

    # A raw --gain (relative / calibration knob) overrides the dBm mapping when
    # present, so you can command a gain directly or measure output power at it.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else pmap.gain_for_power(args.power)

    # Temp dir for the period file: prefer /dev/shm (RAM-backed → fast, no SD wear);
    # fall back to the default temp location if it's absent.
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gps_l1ca_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp, nper = build_iq_buffer(args.prn, CODE_RATE_MCPS * 1e6, samp_rate_hz)
    code_file = os.path.join(tmpdir, f"prn{args.prn}_{CODE_RATE_MCPS:g}Mcps.fc32")
    iq.tofile(code_file)
    print(f"[prebuilt] PRN {args.prn} @ {CODE_RATE_MCPS:g} Mcps → {nsamp} samples "
          f"({nper} code period(s)) → {code_file}")

    tb = _build_top_block(
        code_file=code_file, center_freq_hz=CARRIER_HZ, samp_rate_hz=samp_rate_hz,
        gain_db=gain_db, amplitude=amplitude, otw_format=args.otw, extra_args="")

    # RF on/off state + the gain that RF-on applies. Starting with --rf off builds
    # the flow muted; power/gain edits made while OFF are staged into state["gain"]
    # and reach the radio only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    # Startup banner (the ONLY output during a run — we go silent after start()).
    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  PRN            : {args.prn}")
    print(f"  carrier        : {CARRIER_HZ/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  code rate      : {CODE_RATE_MCPS:g} Mcps "
          f"(~{2*CODE_RATE_MCPS:g} MHz null-to-null)")
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

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                _apply_live_change(tb, ctrl, state, pmap, amplitude,
                                   change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

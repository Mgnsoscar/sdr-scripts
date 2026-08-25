#!/usr/bin/env python3
"""
BeiDou B1I transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** BeiDou **B1I** signal (1561.098 MHz): BPSK-R(2) —
a 2.046 Mcps, 2046-chip ranging code, 1 ms period (~4 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B1I codes
──────────────────────────────────────────
A balanced Gold code (truncated by its last chip → 2046) from two 11-bit LFSRs
(ICD §4.3):
  G1(X) = 1+X+X^7+X^8+X^9+X^10+X^11
  G2(X) = 1+X+X^2+X^3+X^4+X^5+X^8+X^9+X^11
both initialised 01010101010. The per-SV code is G1(stage 11) ⊕ a XOR of selected
G2 stages (the ICD Table 4-1 "phase assignment", e.g. PRN 1 = 1⊕3, PRN 63 =
3⊕6⊕9). The generator + per-SV tap table match the ICD exactly and are byte-
identical to pmonta/GNSS-DSP-tools' beidou/b1i.py (which is used to acquire live
B1I); --self-test re-checks the codes against embedded reference values.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data / 1 kHz NH secondary code (those ride on the data).

⚠  RF SAFETY / LEGAL: B1I is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
level set in dBm (--power) with a live RF on/off (--rf). The default 20.46 MHz (= 10×2.046) gives 10 samples/chip.

CLI
───
    bds_b1i_tx.py --prn 6 --power -30
    bds_b1i_tx.py --self-test
    bds_b1i_tx.py --describe-params
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
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the baked USER CALIBRATION constants below are used. See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "bds_b1i"


# ═══════════════════════════════════════════════════════════════════════════════
# USER CALIBRATION — MEASURE THESE ONCE, THEN EDIT THE VALUES BELOW
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

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED (baked levels) with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75

# Derived delivered-power limits (computed — do not edit).
MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB


_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else the baked constants above (identical to the old single-anchor
    slope-1 behaviour). Cached, so build_script and main share one — and so --power's schema
    bounds match the real operating range (calibrated → e.g. EIRP; else the baked SDR-port
    range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
    return _PMAP


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) for a requested delivered power, through the active calibration (the
    unit's measured curve when present, else the baked anchor). Every caller keeps working."""
    return power_map().gain_for_power(float(delivered_dbm))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) an actual hardware gain produces, through the active map."""
    return power_map().power_for_gain(float(gain_db))


# ── Constants ─────────────────────────────────────────────────────────────────

B1I_HZ = 1561.098e6
CHIP_RATE_HZ = 2_046_000
CODE_LEN = 2046
G_INIT = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)   # 01010101010, both registers

FREQUENCIES = {"BeiDou B1I (1561.098 MHz)": B1I_HZ}

# Per-SV G2 phase selection (XORed G2 stages, 1-indexed), BDS-SIS-ICD-B1I Table 4-1.
G2_TAPS = (
    (1,3),(1,4),(1,5),(1,6),(1,8),(1,9),(1,10),(1,11),(2,7),(3,4),
    (3,5),(3,6),(3,8),(3,9),(3,10),(3,11),(4,5),(4,6),(4,8),(4,9),
    (4,10),(4,11),(5,6),(5,8),(5,9),(5,10),(5,11),(6,8),(6,9),(6,10),
    (6,11),(8,9),(8,10),(8,11),(9,10),(9,11),(10,11),
    (1,2,7),(1,3,4),(1,3,6),(1,3,8),(1,3,10),(1,3,11),(1,4,5),(1,4,9),
    (1,5,6),(1,5,8),(1,5,10),(1,5,11),(1,6,9),(1,8,9),(1,9,10),(1,9,11),
    (2,3,7),(2,5,7),(2,7,9),(3,4,5),(3,4,9),(3,5,6),(3,5,8),(3,5,10),
    (3,5,11),(3,6,9),
)


# ── B1I ranging code (bit-exact, BDS-SIS-ICD-B1I §4.3) ─────────────────────────

def _g1_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[6] ^ r[7] ^ r[8] ^ r[9] ^ r[10]] + r[0:10]        # taps 1,7,8,9,10,11


def _g2_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[7] ^ r[8] ^ r[10]] + r[0:10]  # 1,2,3,4,5,8,9,11


def b1i_code(prn: int) -> list[int]:
    """The 2046-chip B1I ranging code (0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    taps = G2_TAPS[prn - 1]
    g1 = list(G_INIT)
    g2 = list(G_INIT)
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        g2_out = 0
        for t in taps:
            g2_out ^= g2[t - 1]
        out[i] = g1[10] ^ g2_out
        g1 = _g1_step(g1)
        g2 = _g2_step(g2)
    return out


# ── Self-test (period + code check values; no hardware) ────────────────────────

def _self_test() -> int:
    ok = True

    seen, r = {}, list(G_INIT)
    per = None
    for i in range(3000):
        t = tuple(r)
        if t in seen:
            per = i - seen[t]
            break
        seen[t] = i
        r = _g1_step(r)
    print(f"G1 period={per} (expect 2047) [{'OK' if per==2047 else 'FAIL'}]")
    ok = ok and per == 2047

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o31333315, 2: 0o44461070, 6: 0o32442011, 38: 0o67733254, 63: 0o74366441}
    for prn, want in checks.items():
        c = b1i_code(prn)
        got = o24(c)
        good = got == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B1I PRN{prn:2d}: first24={oct(got)} expect={oct(want)} len={len(c)} "
              f"[{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b1i_code(p)) for p in range(1, 11)}) == 10
    print(f"PRN 1..10 distinct: {distinct}")
    ok = ok and distinct

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b1i_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 B1I baseband buffer over one 1 ms code period (loops
    seamlessly). Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))               # 1 ms — one code period
    bipolar = 1.0 - 2.0 * np.asarray(b1i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B1ITx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B1I TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=otw_format, channels=[0]),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def set_amplitude(self, a): self.amp.set_k(a)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return B1ITx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("BeiDou B1I transmitter (real BDS-SIS-ICD-B1I ranging code, "
               "BPSK-R(2), 2.046 Mcps), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou SV / ranging-code number (1..63). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B1I_HZ,
                help="RF carrier (default B1I). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(power_map().min_power_dbm, 2),
                max=round(power_map().max_power_dbm, 2),
                default=round(power_map().max_power_dbm, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=5.0, max=61.44,
                default=20.46,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "20.46 (=10×2.046) → 10 samples/chip. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load; sc16 more range.")
        .choice("-RF", "--rf", options=["on", "off"], default="on",
                required=False, live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them. Change the power (or the "
                     "calibration gain) while OFF and it takes effect when you turn ON.")
    )
    # RELATIVE power (also the calibration knob): the SDR's raw TX gain (dB), bypassing the
    # dBm mapping. No default, so its PRESENCE selects relative mode and OVERRIDES --power.
    # Set it while measuring output vs gain on a spectrum analyser to fill in
    # OUTPUT_POWER_DBM / GAIN_AT_MAX_DB above.
    s = s.number(
        "-Gain", "--gain", unit="dB",
        min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
        help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing the dBm "
             "calibration. When given, overrides --power. Live.")
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
    # A raw calibration gain (the normally-commented --gain knob) overrides the dBm
    # mapping when present, so you can measure output power at a chosen gain.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else gain_for_power(args.power)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b1i_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b1i_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b1i_prn{args.prn}.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(iq_file, args.freq, samp_rate_hz, gain_db,
                          AMPLITUDE, args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── BeiDou B1I TX ───────────────────────────────────────────")
    print(f"  SV / code num  : {args.prn}  (real B1I ranging code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BPSK-R(2) — 2.046 Mcps, ~4 MHz wide")
    print(f"  buffer         : {nsamp} samples (1 ms code period)")
    print(f"  power (target) : {args.power:g} dBm delivered "
          f"(cable −{CABLE_LOSS_DB:g} dB, amp +{AMPLIFIER_GAIN_DB:g} dB)")
    print(f"  → gain         : {gain_db:.2f} dB (max {power_map().max_gain_db:g}), "
          f"amplitude {AMPLITUDE:g}")
    print(f"  calibration    : {power_map().source}")
    if power_map().warning:                # calibration measured at another amplitude
        print(f"  ⚠ CALIBRATION  : {power_map().warning}")
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
            state["gain"] = gain_for_power(float(value))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(power_for_gain(tb.actual_gain()), 2))
            else:
                ctrl.report("power", round(power_for_gain(state["gain"]), 2))
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
                tb.set_amplitude(AMPLITUDE)
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

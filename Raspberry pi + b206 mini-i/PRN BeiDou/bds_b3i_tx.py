#!/usr/bin/env python3
"""
BeiDou B3I transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** BeiDou **B3I** signal (1268.52 MHz): BPSK-R(10) —
a 10.23 Mcps, 10230-chip ranging code, 1 ms period (~20.46 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B3I codes
──────────────────────────────────────────
The ranging code is the modulo-2 sum of two 13-bit LFSR sequences (ICD §4.3):
  G1(X) = X^13+X^4+X^3+X+1        — init all-ones, short-cycled to 8190 chips
                                     (reset when the register is 1111111111100)
  G2(X) = X^13+X^12+X^10+X^9+X^7+X^6+X^5+X+1  — period 8191, per-SV initial phase
code = G1 ⊕ G2, output tap = stage 13, 10230 chips.

Validated three ways: (1) the G2 register convention reproduces the ICD Table 4-1
shift-count→phase for all 63 SVs; (2) G1/G2 periods are exactly 8190/8191 as the
ICD states; (3) the whole generator is byte-identical to pmonta/GNSS-DSP-tools'
beidou/b3i.py, and the per-SV G2 initial phases are the ICD Table 4-1 values.
--self-test re-checks the codes against embedded reference values.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data or the 1 kHz NH secondary code (those ride on the data, not
the ranging code) is applied.

⚠  RF SAFETY / LEGAL: B3I is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
level set in dBm (--power) with a live RF on/off (--rf). PRN / carrier / sample rate / otw fixed per run. The
default 40.92 MHz (= 40×1.023) gives an exact 4 samples/chip.

CLI
───
    bds_b3i_tx.py --prn 6 --power -30
    bds_b3i_tx.py --self-test
    bds_b3i_tx.py --describe-params
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
from paramkit import Script


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

B3_HZ = 1268.52e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
G1_RESET = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0)   # short-cycle state

FREQUENCIES = {"BeiDou B3I (1268.52 MHz)": B3_HZ}

# Per-SV G2 initial phase (13-bit, stage1..stage13), BDS-SIS-ICD-B3I Table 4-1.
# Validated against the ICD (shift-count→phase, 63/63) and pmonta/GNSS-DSP-tools.
G2_INIT = (
    "1010111111111", "1111000101011", "1011110001010", "1111111111011",
    "1100100011111", "1001001100100", "1111111010010", "1110111111101",
    "1010000000010", "0010000011011", "1110101110000", "0010110011110",
    "0110010010101", "0111000100110", "1000110001001", "1110001111100",
    "0010011000101", "0000011101100", "1000101010111", "0001011011110",
    "0010000101101", "0010110001010", "0001011001111", "0011001100010",
    "0011101001000", "0100100101001", "1011011010011", "1010111100010",
    "0001011110101", "0111111111111", "0110110001111", "1010110001001",
    "1001010101011", "1100110100101", "1101001011101", "1111101110100",
    "0010101100111", "1110100010000", "1101110010000", "1101011001110",
    "1000000110100", "0101111011001", "0110110111100", "1101001110001",
    "0011100100010", "0101011000101", "1001111100110", "1111101001000",
    "0000101001001", "1000010101100", "1111001001100", "0100110001111",
    "0000000011000", "1000000000100", "0011010100110", "1011001000110",
    "0111001111000", "0010111001010", "1100111110110", "1001001000101",
    "0111000100000", "0011001000010", "0010001001110",
)


# ── B3I ranging code (bit-exact, BDS-SIS-ICD-B3I §4.3) ─────────────────────────

def _g1_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[2] ^ r[3] ^ r[12]] + r[0:12]              # taps 1,3,4,13


def _g2_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[4] ^ r[5] ^ r[6] ^ r[8] ^ r[9] ^ r[11] ^ r[12]] + r[0:12]  # 1,5,6,7,9,10,12,13


def b3i_code(prn: int) -> list[int]:
    """The 10230-chip B3I ranging code (0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    g1 = [1] * 13
    g2 = [int(c) for c in G2_INIT[prn - 1]]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        out[i] = g1[12] ^ g2[12]
        g1 = [1] * 13 if tuple(g1) == G1_RESET else _g1_step(g1)
        g2 = _g2_step(g2)
    return out


# ── Self-test (periods + code check values; no hardware) ───────────────────────

def _self_test() -> int:
    ok = True

    def period(step, init, reset=None):
        seen, r = {}, list(init)
        for i in range(9000):
            t = tuple(r)
            if t in seen:
                return i - seen[t]
            seen[t] = i
            r = [1] * 13 if (reset and t == reset) else step(r)
        return None

    g1p = period(_g1_step, [1]*13, G1_RESET)
    g2p = period(_g2_step, [1]*13)
    print(f"G1 period={g1p} (expect 8190), G2 period={g2p} (expect 8191) "
          f"[{'OK' if g1p==8190 and g2p==8191 else 'FAIL'}]")
    ok = ok and g1p == 8190 and g2p == 8191

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o51340, 2: 0o12700750, 6: 0o66330754, 30: 0o7411}
    for prn, want in checks.items():
        c = b3i_code(prn)
        got = o24(c)
        good = got == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B3I PRN{prn:2d}: first24={oct(got)} expect={oct(want)} len={len(c)} "
              f"[{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b3i_code(p)) for p in range(1, 11)}) == 10
    print(f"PRN 1..10 distinct: {distinct}")
    ok = ok and distinct

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b3i_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 B3I baseband buffer over one 1 ms code period (loops
    seamlessly). Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))               # 1 ms — one code period
    bipolar = 1.0 - 2.0 * np.asarray(b3i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B3ITx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B3I TX")
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

    return B3ITx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("BeiDou B3I transmitter (real BDS-SIS-ICD-B3I ranging code, "
               "BPSK-R(10), 10.23 Mcps), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou SV / ranging-code number (1..63). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B3_HZ,
                help="RF carrier (default B3). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(MIN_DELIVERED_DBM, 2), max=round(MAX_DELIVERED_DBM, 2),
                default=round(MAX_DELIVERED_DBM, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=20.46, max=61.44,
                default=40.92,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "40.92 (=40×1.023) → 4 samples/chip. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load; sc16 more range.")
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
    # A raw calibration gain (the normally-commented --gain knob) overrides the dBm
    # mapping when present, so you can measure output power at a chosen gain.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else gain_for_power(args.power)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b3i_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b3i_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b3i_prn{args.prn}.fc32")
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

    print("── BeiDou B3I TX ───────────────────────────────────────────")
    print(f"  SV / code num  : {args.prn}  (real B3I ranging code)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BPSK-R(10) — 10.23 Mcps, ~20.46 MHz wide")
    print(f"  buffer         : {nsamp} samples (1 ms code period)")
    print(f"  power (target) : {args.power:g} dBm delivered "
          f"(cable −{CABLE_LOSS_DB:g} dB, amp +{AMPLIFIER_GAIN_DB:g} dB)")
    print(f"  → gain         : {gain_db:.2f} dB (max {GAIN_AT_MAX_DB:g}), "
          f"amplitude {AMPLITUDE:g}")
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

#!/usr/bin/env python3
"""
BeiDou B2b (I-component) transmitter for GNU Radio + UHD (B200-mini family).

Generates a **bit-exact** BeiDou **B2b_I** open-service signal (1207.14 MHz):
BPSK-R(10) — a 10.23 Mcps, 10230-chip ranging code, 1 ms period (~20.46 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Note: this is the **Open Service B2b_I** ranging signal. PPP-B2b (precise point
positioning) is a separate correction-data service on the same carrier — not this.

Code fidelity — real BDS-SIS-ICD-B2b codes
──────────────────────────────────────────
Gold code from two 13-bit LFSRs (register 1 all-ones, short-cycled at chip 8190;
register 2 per-PRN), output = stage 13, ICD §5:
  g1 = 1+x+x^9+x^10+x^13     g2 = 1+x^3+x^4+x^6+x^9+x^12+x^13
Validated against the ICD's own check values: generating each code from its
register-2 init reproduces the ICD's first-24 AND last-24 chips (octal) — 53/53
for all PRNs (6..58). --self-test re-checks representative PRNs.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data (bare code). B2b_I has no pilot component or secondary code.

⚠  RF SAFETY / LEGAL: B2b (1207.14 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use — never over air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
level set in dBm (--power) with a live RF on/off (--rf). Default 40.92 MHz (= 40×1.023) → 4 samples/chip.

CLI
───
    bds_b2b_tx.py --prn 20 --power -30
    bds_b2b_tx.py --self-test
    bds_b2b_tx.py --describe-params
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

B2B_HZ = 1207.14e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
G1_TAPS = (1, 9, 10, 13)
G2_TAPS = (3, 4, 6, 9, 12, 13)

FREQUENCIES = {"BeiDou B2b (1207.14 MHz)": B2B_HZ}

# Register-2 initial values (stage1..stage13), BDS-SIS-ICD-B2b Table 5-1 (PRN 6..58).
B2B_REG2 = {
    6:"1000110101110", 7:"1000111101110", 8:"1000111111011", 9:"1001100101001",
    10:"1001111011010", 11:"1010000110101", 12:"1010001000100", 13:"1010001010101",
    14:"1010001011011", 15:"1010001011100", 16:"1010010100011", 17:"1010011110111",
    18:"1010100000001", 19:"1010100111110", 20:"1010110101011", 21:"1010110110001",
    22:"1011001010011", 23:"1011001100010", 24:"1011010011000", 25:"1011010110110",
    26:"1011011110010", 27:"1011011111111", 28:"1011100010010", 29:"1011100111100",
    30:"1011110100001", 31:"1011111001000", 32:"1011111010100", 33:"1011111101011",
    34:"1011111110011", 35:"1100001010001", 36:"1100010010100", 37:"1100010110111",
    38:"1100100010001", 39:"1100100011001", 40:"1100110101011", 41:"1100110110001",
    42:"1100111010010", 43:"1101001010101", 44:"1101001110100", 45:"1101011001011",
    46:"1101101010111", 47:"1110000110100", 48:"1110010000011", 49:"1110010001011",
    50:"1110010100011", 51:"1110010101000", 52:"1110100111011", 53:"1110110010111",
    54:"1111001001000", 55:"1111010010100", 56:"1111010011001", 57:"1111011011010",
    58:"1111011111000",
}


# ── B2b_I ranging code (bit-exact, BDS-SIS-ICD-B2b §5) ─────────────────────────

def b2b_code(prn: int) -> list[int]:
    """The 10230-chip B2b_I ranging code (0/1) for a PRN (6..58)."""
    if prn not in B2B_REG2:
        raise ValueError(f"PRN must be 6..58, got {prn}")
    r1 = [1] * 13
    r2 = [int(c) for c in B2B_REG2[prn]]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        if i == RESET_CHIP:
            r1 = [1] * 13
        out[i] = r1[12] ^ r2[12]
        f1 = 0
        for t in G1_TAPS:
            f1 ^= r1[t - 1]
        r1 = [f1] + r1[:12]
        f2 = 0
        for t in G2_TAPS:
            f2 ^= r2[t - 1]
        r2 = [f2] + r2[:12]
    return out


# ── Self-test (codes vs ICD check values; no hardware) ─────────────────────────

def _self_test() -> int:
    ok = True

    def octs(bits):
        f = l = 0
        for b in bits[:24]:
            f = (f << 1) | b
        for b in bits[-24:]:
            l = (l << 1) | b
        return "%08o" % f, "%08o" % l

    checks = {6: ("42471422", "44530033"), 7: ("42071026", "63454537"),
              58: ("70100474", "31701764")}
    for prn, want in checks.items():
        c = b2b_code(prn)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B2b PRN{prn:2d}: {octs(c)} expect {want} len={len(c)} [{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b2b_code(p)) for p in range(6, 16)}) == 10
    print(f"PRN 6..15 distinct: {distinct}")
    ok = ok and distinct

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b2b_buffer(prn: int, samp_rate_hz: float):
    """Build a complex64 B2b_I baseband buffer over one 1 ms code period (loops
    seamlessly). Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b2b_code(prn), dtype=np.int8)   # logic 1→−1, 0→+1
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B2BTx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B2b TX")
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

    return B2BTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("BeiDou B2b (I-component) transmitter (real BDS-SIS-ICD-B2b ranging "
               "code, BPSK-R(10), 10.23 Mcps), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=6, max=58, default=6, required=True,
                 help="BeiDou PRN / ranging-code number (6..58). Fixed per run.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B2B_HZ,
                help="RF carrier (default B2b). Fixed per run.")
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
    tmpdir = tempfile.mkdtemp(prefix="bds_b2b_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b2b_buffer(args.prn, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b2b_prn{args.prn}.fc32")
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

    print("── BeiDou B2b TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B2b_I ranging code)")
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

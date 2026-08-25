#!/usr/bin/env python3
"""
BeiDou B2a transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** BeiDou **B2a** signal (1176.45 MHz): QPSK — a data
component (I) and a pilot component (Q), equal power, each BPSK(10) at 10.23 Mcps.
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B2a codes
──────────────────────────────────────────
Both components use TIERED codes (primary ⊕ secondary), ICD §5:
  data  : Gold primary (10230) ⊕ fixed 5-chip secondary "00010"
  pilot : Gold primary (10230) ⊕ truncated-Weil secondary (100 chips, per-PRN)
Primary codes come from two 13-bit LFSRs (register 1 all-ones, short-cycled at
chip 8190; register 2 per-PRN), output = stage 13:
  data  g1 = 1+x+x^5+x^11+x^13,      g2 = 1+x^3+x^5+x^9+x^11+x^12+x^13
  pilot g1 = 1+x^3+x^6+x^7+x^13,     g2 = 1+x+x^5+x^7+x^8+x^12+x^13
Every table here was validated against the ICD's own check values: generating
each code from its register-2 init / (w,p) reproduces the ICD's first-24 AND
last-24 chips (octal) — 63/63 for the data primary, pilot primary, and pilot
secondary. --self-test re-checks representative PRNs.

Logic level (ICD Table 4-3): 1 → −1.0, 0 → +1.0. Data on I (phase 0), pilot on Q
(phase 90°), power 1:1 → constant-modulus QPSK.

⚠  RF SAFETY / LEGAL: B2a (1176.45 MHz, shared with GPS L5) is a live GNSS band.
   Transmit ONLY into a shielded / conducted setup you are LICENSED / AUTHORISED
   to use — never over the air.

Loop length (--loop):
  full (default) : one full 100 ms tiered period (pilot secondary is 100 ms; data
                   secondary 5 ms divides it). Bit-exact, complete spectrum.
                   ~33 MB at 40.92 MHz.
  primary        : one 1 ms primary period (no secondary cycling). Small; the
                   BPSK(10) envelope is identical.

Sample rate default 40.92 MHz (=40×1.023 → 4 samples/chip). 1:1 master clock;
sc8; level set in dBm (--power) with a live RF on/off (--rf). See gps_l5_tx.py for the engine.

CLI
───
    bds_b2a_tx.py --prn 6 --power -30
    bds_b2a_tx.py --prn 6 --component pilot --loop primary
    bds_b2a_tx.py --self-test
    bds_b2a_tx.py --describe-params
"""
from __future__ import annotations

import math
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

B2A_HZ = 1176.45e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
WEIL_N = 1021
DATA_SEC = (0, 0, 0, 1, 0)                # fixed 5-chip data secondary "00010"

DATA_G1 = (1, 5, 11, 13)
DATA_G2 = (3, 5, 9, 11, 12, 13)
PILOT_G1 = (3, 6, 7, 13)
PILOT_G2 = (1, 5, 7, 8, 12, 13)

FREQUENCIES = {"BeiDou B2a (1176.45 MHz)": B2A_HZ}

# Register-2 initial values (stage1..stage13), BDS-SIS-ICD-B2a Tables 5-2 / 5-3.
B2A_DATA_REG2 = (
    "1000000100101", "1000000110100", "1000010101101", "1000101001111", "1000101010101", "1000110101110",
    "1000111101110", "1000111111011", "1001100101001", "1001111011010", "1010000110101", "1010001000100",
    "1010001010101", "1010001011011", "1010001011100", "1010010100011", "1010011110111", "1010100000001",
    "1010100111110", "1010110101011", "1010110110001", "1011001010011", "1011001100010", "1011010011000",
    "1011010110110", "1011011110010", "1011011111111", "1011100010010", "1011100111100", "1011110100001",
    "1011111001000", "1011111010100", "1011111101011", "1011111110011", "1100001010001", "1100010010100",
    "1100010110111", "1100100010001", "1100100011001", "1100110101011", "1100110110001", "1100111010010",
    "1101001010101", "1101001110100", "1101011001011", "1101101010111", "1110000110100", "1110010000011",
    "1110010001011", "1110010100011", "1110010101000", "1110100111011", "1110110010111", "1111001001000",
    "1111010010100", "1111010011001", "1111011011010", "1111011111000", "1111011111111", "1111110110101",
    "0010000000010", "1101111110101", "0001111010010",
)
B2A_PILOT_REG2 = (
    "1000000100101", "1000000110100", "1000010101101", "1000101001111", "1000101010101", "1000110101110",
    "1000111101110", "1000111111011", "1001100101001", "1001111011010", "1010000110101", "1010001000100",
    "1010001010101", "1010001011011", "1010001011100", "1010010100011", "1010011110111", "1010100000001",
    "1010100111110", "1010110101011", "1010110110001", "1011001010011", "1011001100010", "1011010011000",
    "1011010110110", "1011011110010", "1011011111111", "1011100010010", "1011100111100", "1011110100001",
    "1011111001000", "1011111010100", "1011111101011", "1011111110011", "1100001010001", "1100010010100",
    "1100010110111", "1100100010001", "1100100011001", "1100110101011", "1100110110001", "1100111010010",
    "1101001010101", "1101001110100", "1101011001011", "1101101010111", "1110000110100", "1110010000011",
    "1110010001011", "1110010100011", "1110010101000", "1110100111011", "1110110010111", "1111001001000",
    "1111010010100", "1111010011001", "1111011011010", "1111011111000", "1111011111111", "1111110110101",
    "1010010000110", "0010111111000", "0001101010101",
)
# Pilot secondary (truncated Weil): per-PRN (phase w, truncation point p), Table 5-4.
B2A_PILOT_SEC = (
    (123,138), (55,570), (40,351), (139,77), (31,885), (175,247),
    (350,413), (450,180), (478,3), (8,26), (73,17), (97,172),
    (213,30), (407,1008), (476,646), (4,158), (15,170), (47,99),
    (163,53), (280,179), (322,925), (353,114), (375,10), (510,584),
    (332,60), (7,3), (13,684), (16,263), (18,545), (25,22),
    (50,546), (81,190), (118,303), (127,234), (132,38), (134,822),
    (164,57), (177,668), (208,697), (249,93), (276,18), (349,66),
    (439,318), (477,133), (498,98), (88,70), (155,132), (330,26),
    (3,354), (21,58), (84,41), (111,182), (128,944), (153,205),
    (197,23), (199,1), (214,792), (256,641), (265,83), (291,7),
    (324,111), (326,96), (340,92),
)


# ── B2a primary codes (bit-exact, BDS-SIS-ICD-B2a §5.2.1) ──────────────────────

def _primary(prn: int, component: str) -> list[int]:
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    if component == "data":
        g1t, g2t, reg2 = DATA_G1, DATA_G2, B2A_DATA_REG2[prn - 1]
    else:
        g1t, g2t, reg2 = PILOT_G1, PILOT_G2, B2A_PILOT_REG2[prn - 1]
    r1 = [1] * 13
    r2 = [int(c) for c in reg2]
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        if i == RESET_CHIP:
            r1 = [1] * 13
        out[i] = r1[12] ^ r2[12]
        f1 = 0
        for t in g1t:
            f1 ^= r1[t - 1]
        r1 = [f1] + r1[:12]
        f2 = 0
        for t in g2t:
            f2 ^= r2[t - 1]
        r2 = [f2] + r2[:12]
    return out


_LEG: list | None = None


def _legendre() -> list:
    global _LEG
    if _LEG is None:
        qr = {(x * x) % WEIL_N for x in range(1, WEIL_N)}
        _LEG = [0] + [1 if k in qr else 0 for k in range(1, WEIL_N)]
    return _LEG


def _pilot_secondary(prn: int) -> list[int]:
    w, p = B2A_PILOT_SEC[prn - 1]
    L = _legendre()
    W = [L[k] ^ L[(k + w) % WEIL_N] for k in range(WEIL_N)]
    return [W[(n + p - 1) % WEIL_N] for n in range(100)]


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

    # (component, prn): (first24, last24) from the ICD.
    prim_chk = {
        ("data", 1): ("26771056", "42646672"), ("data", 2): ("64771737", "43261240"),
        ("data", 63): ("55037136", "06147764"),
        ("pilot", 1): ("26772435", "05133452"), ("pilot", 63): ("25236023", "01076040"),
    }
    for (comp, prn), want in prim_chk.items():
        c = _primary(prn, comp)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"{comp} primary PRN{prn:2d}: {octs(c)} expect {want} [{'OK' if good else 'FAIL'}]")

    sec_chk = {1: ("32063050", "65322167")}
    for prn, want in sec_chk.items():
        s = _pilot_secondary(prn)
        good = octs(s) == want and len(s) == 100
        ok = ok and good
        print(f"pilot secondary PRN{prn}: {octs(s)} expect {want} [{'OK' if good else 'FAIL'}]")

    leg_ok = sum(_legendre()) == (WEIL_N - 1) // 2
    print(f"Legendre(1021) ones={sum(_legendre())} (expect {(WEIL_N-1)//2}) [{'OK' if leg_ok else 'FAIL'}]")
    ok = ok and leg_ok

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (tiered QPSK, seamless loop) ───────────────────────────────

def build_b2a_buffer(prn: int, component: str, loop: str, samp_rate_hz: float):
    """Build a complex64 B2a baseband buffer. component: 'both'|'data'|'pilot'.
    loop: 'full' (100 ms tiered) | 'primary' (1 ms). Returns (iq, n_samples)."""
    import numpy as np

    pd = np.asarray(_primary(prn, "data"), dtype=np.int8)
    pp = np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    sd = np.asarray(DATA_SEC, dtype=np.int8)
    sp = np.asarray(_pilot_secondary(prn), dtype=np.int8)

    n_periods = 100 if loop == "full" else 1
    sr = int(round(samp_rate_hz))
    n_samples = int(round(n_periods * CODE_LEN / CHIP_RATE_HZ * sr))

    n = np.arange(n_samples, dtype=np.int64)
    gc = n * CHIP_RATE_HZ // sr
    m = gc // CODE_LEN                    # primary-period index
    c = gc % CODE_LEN                     # chip within primary
    d_bit = pd[c] ^ sd[m % 5]
    p_bit = pp[c] ^ sp[m % 100]
    d = 1.0 - 2.0 * d_bit                 # logic 1→−1, 0→+1
    p = 1.0 - 2.0 * p_bit

    if component == "data":
        iq = d.astype(np.complex64)
    elif component == "pilot":
        iq = p.astype(np.complex64)
    else:
        iq = ((d + 1j * p) / math.sqrt(2.0)).astype(np.complex64)   # QPSK
    return iq, n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B2ATx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B2a TX")
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

    return B2ATx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("BeiDou B2a transmitter (real BDS-SIS-ICD-B2a codes, QPSK: data + "
               "pilot, tiered Gold/Weil, 10.23 Mcps), file-replay. Authorised, "
               "shielded setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou PRN / ranging-code number (1..63). Fixed per run.")
        .choice("-Component", "--component", options=["both", "data", "pilot"],
                default="both",
                help="both = QPSK (data I + pilot Q); data or pilot = one channel.")
        .choice("-Loop", "--loop", options=["full", "primary"], default="full",
                help="full = 100 ms tiered (bit-exact, ~33 MB); primary = 1 ms "
                     "(small, envelope-correct).")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B2A_HZ,
                help="RF carrier (default B2a). Fixed per run.")
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
    tmpdir = tempfile.mkdtemp(prefix="bds_b2a_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_b2a_buffer(args.prn, args.component, args.loop, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"b2a_prn{args.prn}_{args.component}_{args.loop}.fc32")
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

    print("── BeiDou B2a TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B2a codes, {args.component})")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : QPSK (BPSK(10) data + pilot), 10.23 Mcps")
    print(f"  loop           : {args.loop}")
    print(f"  buffer         : {nsamp} samples ({nsamp*8/1e6:.1f} MB)")
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

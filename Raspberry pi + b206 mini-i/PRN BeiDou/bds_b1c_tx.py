#!/usr/bin/env python3
"""
BeiDou B1C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a spectrally-correct BeiDou **B1C** signal (1575.42 MHz) — BeiDou's
modernized civil L1 signal — as the QMBOC composite of its data and pilot
components. Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B1C Weil codes
───────────────────────────────────────────────
Primaries (data + pilot): truncated Weil (Legendre mod N=10243), 10230 chips.
Pilot secondary: truncated Weil (Legendre mod N=3607), 1800 chips, 18 s period.
Per-PRN (w,p) for all three validated against the ICD's own check values — each
reproduces the ICD's first-24 AND last-24 chips (octal), 63/63. --self-test
re-checks representative PRNs.

Full-length secondary WITHOUT a multi-GB file
─────────────────────────────────────────────
The pilot secondary is a *slow overlay*: one ±1 chip per 10 ms primary period, so
the 18 s tiered code is just the 10 ms pilot buffer replayed 1800 times with a
per-period sign flip. Rather than precompute 18 s of samples (~7 GB), the flow
applies the secondary at runtime with stock blocks:

    pilot_file ─► × ─────────┐
    sec(1800) ─► repeat(N) ─┘ ├─► + ─► (amp) ─► USRP
    data_file ───────────────┘

so the full bit-exact 18 s signal streams from ~8 MB of buffers. `--secondary off`
drops it (10 ms primary loop only; spectrally identical, no secondary sync).

Modulation — QMBOC(6,1,4/33), ICD §4.2 (data:pilot power = 1:3)
──────────────────────────────────────────────────────────────
BOC(6,1) is in phase *quadrature* with BOC(1,1) (power 29:4):
    I = ½·D·C_data·BOC(1,1)  −  √(1/11)·C_pilot·BOC(6,1)
    Q =                          √(29/44)·C_pilot·BOC(1,1)
Logic level (ICD): 0 → +1, 1 → −1. No navigation data (bare code).

⚠  RF SAFETY / LEGAL: B1C (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup you are LICENSED / AUTHORISED to use.

Sample rate default 49.104 MHz (=48×1.023 → 4 samples per BOC(6,1) half-period).
1:1 master clock; sc8; level set in dBm (--power) with a live RF on/off (--rf).

CLI
───
    bds_b1c_tx.py --prn 19 --power -30
    bds_b1c_tx.py --prn 19 --component pilot --secondary full
    bds_b1c_tx.py --self-test
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

B1C_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10                  # primary period (ms) = one secondary chip
WEIL_N = 10243                   # primary Weil Legendre prime
SEC_N = 3607                     # secondary Weil Legendre prime
SEC_LEN = 1800                   # pilot secondary length (chips) → 18 s period

A_DATA = 0.5
A_PILOT_A = math.sqrt(29 / 44)   # pilot BOC(1,1), on Q
A_PILOT_B = math.sqrt(1 / 11)    # pilot BOC(6,1), on I (quadrature)

FREQUENCIES = {"BeiDou B1C (1575.42 MHz)": B1C_HZ}

# Per-PRN (Weil phase w, truncation point p), BDS-SIS-ICD-B1C Tables 5-2/5-3/5-4.
B1C_DATA_WP = (
    (2678,699), (4802,694), (958,7318), (859,2127), (3843,715), (2232,6682),
    (124,7850), (4352,5495), (1816,1162), (1126,7682), (1860,6792), (4800,9973),
    (2267,6596), (424,2092), (4192,19), (4333,10151), (2656,6297), (4148,5766),
    (243,2359), (1330,7136), (1593,1706), (1470,2128), (882,6827), (3202,693),
    (5095,9729), (2546,1620), (1733,6805), (4795,534), (4577,712), (1627,1929),
    (3638,5355), (2553,6139), (3646,6339), (1087,1470), (1843,6867), (216,7851),
    (2245,1162), (726,7659), (1966,1156), (670,2672), (4130,6043), (53,2862),
    (4830,180), (182,2663), (2181,6940), (2006,1645), (1080,1582), (2288,951),
    (2027,6878), (271,7701), (915,1823), (497,2391), (139,2606), (3693,822),
    (2054,6403), (4342,239), (3342,442), (2592,6769), (1007,2560), (310,2502),
    (4203,5072), (455,7268), (4318,341),
)
B1C_PILOT_WP = (
    (796,7575), (156,2369), (4198,5688), (3941,539), (1374,2270), (1338,7306),
    (1833,6457), (2521,6254), (3175,5644), (168,7119), (2715,1402), (4408,5557),
    (3160,5764), (2796,1073), (459,7001), (3594,5910), (4813,10060), (586,2710),
    (1428,1546), (2371,6887), (2285,1883), (3377,5613), (4965,5062), (3779,1038),
    (4547,10170), (1646,6484), (1430,1718), (607,2535), (2118,1158), (4709,526),
    (1149,7331), (3283,5844), (2473,6423), (1006,6968), (3670,1280), (1817,1838),
    (771,1989), (2173,6468), (740,2091), (1433,1581), (2458,1453), (3459,6252),
    (2155,7122), (1205,7711), (413,7216), (874,2113), (2463,1095), (1106,1628),
    (1590,1713), (3873,6102), (4026,6123), (4272,6070), (3556,1115), (128,8047),
    (1200,6795), (130,2575), (4494,53), (1871,1729), (3073,6388), (4386,682),
    (4098,5565), (1923,7160), (1176,2277),
)
B1C_SEC_WP = (
    (269,1889), (1448,1268), (1028,1593), (1324,1186), (822,1239), (5,1930),
    (155,176), (458,1696), (310,26), (959,1344), (1238,1271), (1180,1182),
    (1288,1381), (334,1604), (885,1333), (1362,1185), (181,31), (1648,704),
    (838,1190), (313,1646), (750,1385), (225,113), (1477,860), (309,1656),
    (108,1921), (1457,1173), (149,1928), (322,57), (271,150), (576,1214),
    (1103,1148), (450,1458), (399,1519), (241,1635), (1045,1257), (164,1687),
    (513,1382), (687,1514), (422,1), (303,1583), (324,1806), (495,1664),
    (725,1338), (780,1111), (367,1706), (882,1543), (631,1813), (37,228),
    (647,2871), (1043,2884), (24,1823), (120,75), (134,11), (136,63),
    (158,1937), (214,22), (335,1768), (340,1526), (661,1402), (889,1445),
    (929,1680), (1002,1290), (1149,1245),
)


# ── B1C Weil codes (bit-exact, BDS-SIS-ICD-B1C §5.2) ───────────────────────────

_LEG: dict = {}


def _legendre(N: int) -> list:
    if N not in _LEG:
        qr = {(x * x) % N for x in range(1, N)}
        _LEG[N] = [0] + [1 if k in qr else 0 for k in range(1, N)]
    return _LEG[N]


def _weil(w: int, p: int, N: int, length: int) -> list[int]:
    L = _legendre(N)
    W = [L[k] ^ L[(k + w) % N] for k in range(N)]
    return [W[(n + p - 1) % N] for n in range(length)]


def _primary(prn: int, component: str) -> list[int]:
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    w, p = (B1C_DATA_WP if component == "data" else B1C_PILOT_WP)[prn - 1]
    return _weil(w, p, WEIL_N, CODE_LEN)


def _secondary(prn: int) -> list[int]:
    """The 1800-chip pilot secondary code (0/1)."""
    w, p = B1C_SEC_WP[prn - 1]
    return _weil(w, p, SEC_N, SEC_LEN)


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

    chk = {
        ("data", 1): ("53773116", "42711657"), ("data", 63): ("27571255", "47160627"),
        ("pilot", 1): ("71676756", "13053205"), ("pilot", 63): ("03210227", "56250500"),
    }
    for (comp, prn), want in chk.items():
        c = _primary(prn, comp)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"{comp} primary PRN{prn:2d}: {octs(c)} expect {want} [{'OK' if good else 'FAIL'}]")

    sec_chk = {1: ("27516364", "67377026")}
    for prn, want in sec_chk.items():
        s = _secondary(prn)
        good = octs(s) == want and len(s) == SEC_LEN
        ok = ok and good
        print(f"pilot secondary PRN{prn}: {octs(s)} expect {want} len={len(s)} [{'OK' if good else 'FAIL'}]")

    for N, exp in ((WEIL_N, 5121), (SEC_N, 1803)):
        got = sum(_legendre(N))
        print(f"Legendre({N}) ones={got} (expect {exp}) [{'OK' if got==exp else 'FAIL'}]")
        ok = ok and got == exp

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffers (data + pilot components, one 10 ms primary period) ───────

def build_b1c_components(prn: int, samp_rate_hz: float):
    """Return (data_buf, pilot_buf, n_samples): the two QMBOC component buffers
    (complex64, one 10 ms primary period each), commonly peak-normalised so that
    data ± pilot never clips. The pilot secondary is applied downstream."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(PRIMARY_MS * 1e-3 * sr))

    cd = 1.0 - 2.0 * np.asarray(_primary(prn, "data"), dtype=np.int8)
    cp = 1.0 - 2.0 * np.asarray(_primary(prn, "pilot"), dtype=np.int8)

    n = np.arange(n_samples, dtype=np.int64)
    num = n * CHIP_RATE_HZ
    chip = num // sr
    rem = num - chip * sr
    boc11 = np.where(rem * 2 < sr, 1.0, -1.0)
    boc61 = np.where(((rem * 12) // sr) % 2 == 0, 1.0, -1.0)
    cidx = chip % CODE_LEN
    cdv, cpv = cd[cidx], cp[cidx]

    data = (A_DATA * cdv * boc11).astype(np.complex128)               # on I
    pilot = (-A_PILOT_B * cpv * boc61) + 1j * (A_PILOT_A * cpv * boc11)
    # normalise for the worst case (secondary flips the pilot sign each period)
    norm = max(np.max(np.abs(data + pilot)), np.max(np.abs(data - pilot)))
    return (data / norm).astype(np.complex64), (pilot / norm).astype(np.complex64), n_samples


def secondary_signs(prn: int):
    """The pilot secondary as ±1 complex values (1800), for the runtime multiply."""
    import numpy as np
    return (1.0 - 2.0 * np.asarray(_secondary(prn), dtype=np.float32)).astype(np.complex64)


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(data_file, pilot_file, sec_signs, period_samples,
                     center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class B1CTx(gr.top_block):
        def __init__(self):
            super().__init__("BeiDou B1C TX")
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

            branches = []
            if pilot_file:
                pilot_src = blocks.file_source(gr.sizeof_gr_complex, pilot_file, repeat=True)
                if sec_signs is not None:
                    # Apply the slow secondary: one ±1 held for each primary period.
                    sec_src = blocks.vector_source_c(list(sec_signs), repeat=True)
                    sec_rep = blocks.repeat(gr.sizeof_gr_complex, int(period_samples))
                    mult = blocks.multiply_cc()
                    self.connect(sec_src, sec_rep, (mult, 1))
                    self.connect(pilot_src, (mult, 0))
                    branches.append(mult)
                else:
                    branches.append(pilot_src)
            if data_file:
                branches.append(blocks.file_source(gr.sizeof_gr_complex, data_file, repeat=True))

            self.amp = blocks.multiply_const_cc(amplitude)
            if len(branches) == 1:
                self.connect(branches[0], self.amp)
            else:
                adder = blocks.add_cc()
                for i, b in enumerate(branches):
                    self.connect(b, (adder, i))
                self.connect(adder, self.amp)
            self.connect(self.amp, self.usrp)

        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def set_amplitude(self, a): self.amp.set_k(a)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return B1CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("BeiDou B1C transmitter (real BDS-SIS-ICD-B1C Weil codes, "
               "QMBOC(6,1,4/33): data + pilot, full 18 s secondary), file-replay. "
               "Authorised, shielded setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou PRN / ranging-code number (1..63). Fixed per run.")
        .choice("-Component", "--component", options=["both", "data", "pilot"],
                default="both",
                help="both = full QMBOC; data = BOC(1,1) data only; pilot = QMBOC pilot.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="full = apply the 1800-chip pilot secondary (18 s, bit-exact, "
                     "runtime multiply); off = 10 ms primary loop only.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B1C_HZ,
                help="RF carrier (default B1C). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(MIN_DELIVERED_DBM, 2), max=round(MAX_DELIVERED_DBM, 2),
                default=round(MAX_DELIVERED_DBM, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=15.0, max=61.44,
                default=49.104,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "49.104 (=48×1.023) aligns the BOC(6,1) subcarrier. Fixed per run.")
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
    tmpdir = tempfile.mkdtemp(prefix="bds_b1c_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    data_buf, pilot_buf, nsamp = build_b1c_components(args.prn, samp_rate_hz)

    data_file = pilot_file = None
    if args.component in ("both", "data"):
        data_file = os.path.join(tmpdir, "data.fc32")
        data_buf.tofile(data_file)
    if args.component in ("both", "pilot"):
        pilot_file = os.path.join(tmpdir, "pilot.fc32")
        pilot_buf.tofile(pilot_file)

    sec_signs = None
    if pilot_file and args.secondary == "full":
        sec_signs = secondary_signs(args.prn)

    tb = _build_top_block(data_file, pilot_file, sec_signs, nsamp,
                          args.freq, samp_rate_hz, gain_db, AMPLITUDE,
                          args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    sec_desc = ("18 s (full secondary)" if sec_signs is not None
                else "10 ms (primary only)")
    print("── BeiDou B1C TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B1C Weil codes, {args.component})")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : QMBOC(6,1,4/33) — BOC(1,1) data + pilot, BOC(6,1) quad")
    print(f"  period         : {sec_desc}")
    print(f"  primary buffer : {nsamp} samples ({nsamp*8/1e6:.1f} MB/file)")
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

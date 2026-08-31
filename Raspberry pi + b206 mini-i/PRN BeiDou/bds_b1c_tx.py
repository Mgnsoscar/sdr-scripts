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
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only). See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "bds_b1c"


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75



_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else it runs uncalibrated — a relative gain only (no baked
    slope-1 behaviour). Cached, so build_script and main share one — and so --power's schema
    bounds match the real operating range (calibrated → e.g. EIRP; else the baked SDR-port
    range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) for a requested delivered power, through the active calibration (the
    unit's measured curve when present, else the baked anchor). Every caller keeps working."""
    return power_map().gain_for_power(float(delivered_dbm))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) an actual hardware gain produces, through the active map."""
    return power_map().power_for_gain(float(gain_db))


# ── Constants ─────────────────────────────────────────────────────────────────

B1C_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10                  # primary period (ms) = one secondary chip
WEIL_N = 10243                   # primary Weil Legendre prime
SEC_N = 3607                     # secondary Weil Legendre prime
SEC_LEN = 1800                   # pilot secondary length (chips) → 18 s period
SIGNAL_NAME = "BeiDou B1C"

A_DATA = 0.5
A_PILOT_A = math.sqrt(29 / 44)   # pilot BOC(1,1), on Q
A_PILOT_B = math.sqrt(1 / 11)    # pilot BOC(6,1), on I (quadrature)

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6          # 60 samp/chip; BOC(6,1) subcarrier at 10 samples/cycle (exact)
OTW_FORMAT = "sc8"              # over-the-wire; halves USB load

# The B1C code is 1.023 Mcps, so its spectral nulls sit at every 1.023 MHz. The filter's
# passband edge snaps to these nulls: --nulls n → edge at ±n·1.023 MHz (on the null between
# lobes). The BOC(1,1) core is bounded by the n=2 null; the BOC(6,1) lobes by the n=7 null.
B1C_NULL_HZ = 1.023e6
MAX_NULLS = 30
NULL_PRESETS = {
    "BOC(1,1) core (±2.05 MHz)": 2,
    "Between the lobes (±4.09 MHz)": 4,
    "Include BOC(6,1) lobes (±7.16 MHz)": 7,
    "Wide (±14.32 MHz)": 14,
}

FREQUENCIES = {"BeiDou B1C (1575.42 MHz)": B1C_HZ / 1e6}   # presets in MHz

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

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    data_buf, pilot_buf, n = build_b1c_components(1)
    base = data_buf + pilot_buf                       # composite (secondary sign +1 for period 0)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    # Filter each component (as the flow does) then sum — must equal filtering the sum.
    df, taps, fp = filter_buffer(data_buf, nulls=7, trans_hz=0.5e6)   # full QMBOC (±7.16 MHz)
    pf, _, _ = filter_buffer(pilot_buf, nulls=7, trans_hz=0.5e6)
    filt = df + pf
    kept = 10 * np.log10(band(filt, 0, fp) / band(base, 0, fp))
    cut = 10 * np.log10(band(filt, 12e6, 20e6) / max(band(base, 12e6, 20e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (nulls=7 → ±{fp/1e6:.2f} MHz, {taps} taps): kept band {kept:+.3f} dB, "
          f"out-of-band {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffers (data + pilot components, one 10 ms primary period) ───────

def build_b1c_components(prn: int):
    """Return (data_buf, pilot_buf, n_samples): the two QMBOC component buffers
    (complex64, one 10 ms primary period each at the fixed SAMP_RATE_HZ), commonly
    peak-normalised so that data ± pilot never clips. The pilot secondary is applied
    downstream."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
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


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. Returns (h, n_taps)."""
    import numpy as np
    m = int(np.ceil(5.5 * SAMP_RATE_HZ / max(trans_hz, 1.0))) | 1     # odd
    m = min(m, (max_taps | 1))
    k = np.arange(m)
    c = (m - 1) / 2.0
    fcn = min(fc_hz / SAMP_RATE_HZ, 0.499)          # never above Nyquist
    h = 2 * fcn * np.sinc(2 * fcn * (k - c))
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n1)
           + 0.14128 * np.cos(4 * np.pi * k / n1) - 0.01168 * np.cos(6 * np.pi * k / n1))
    h = h * win
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, nulls: int, trans_hz: float):
    """Circularly filter a looped B1C component buffer, passband edge snapped to the nth null
    (±`nulls`·1.023 MHz). Circular convolution keeps the result exactly periodic (seam-free
    loop); unity passband gain leaves the kept lobes' power unchanged. Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = int(nulls) * B1C_NULL_HZ
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(data_file, pilot_file, sec_signs, period_samples,
                     center_freq_hz, gain_db, amplitude):
    """Two file sources (data + pilot QMBOC components) summed after the pilot's slow
    secondary; holds references so each component buffer can be swapped live."""
    from gnuradio import gr, blocks, uhd

    class B1CTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]),
            )
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            self.pilot_src = None
            self.data_src = None
            branches = []
            if pilot_file:
                self.pilot_src = blocks.file_source(gr.sizeof_gr_complex, pilot_file, repeat=True)
                if sec_signs is not None:
                    # Apply the slow secondary: one ±1 held for each primary period.
                    sec_src = blocks.vector_source_c(list(sec_signs), repeat=True)
                    sec_rep = blocks.repeat(gr.sizeof_gr_complex, int(period_samples))
                    mult = blocks.multiply_cc()
                    self.connect(sec_src, sec_rep, (mult, 1))
                    self.connect(self.pilot_src, (mult, 0))
                    branches.append(mult)
                else:
                    branches.append(self.pilot_src)
            if data_file:
                self.data_src = blocks.file_source(gr.sizeof_gr_complex, data_file, repeat=True)
                branches.append(self.data_src)

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

        def swap_pilot(self, path):
            if self.pilot_src is not None:
                self.pilot_src.open(path, True)

        def swap_data(self, path):
            if self.data_src is not None:
                self.data_src.open(path, True)

        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return B1CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (real BDS-SIS-ICD-B1C Weil codes, QMBOC(6,1,4/33): "
               "data + pilot, full 18 s secondary) — fixed 61.38 MHz / sc8, looped buffers, "
               "optional power-preserving digital passband filter. Level is set in dBm via the "
               "unit's calibration; uncalibrated it runs on a relative gain. Authorised, "
               "shielded setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=B1C_HZ / 1e6,
                help="RF carrier in MHz (default B1C = 1575.42). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="BeiDou PRN / ranging-code number (1..63). Fixed per run.")
        .choice("-Component", "--component", options=["both", "data", "pilot"], default="both",
                help="both = full QMBOC; data = BOC(1,1) data only; pilot = QMBOC pilot.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="full = apply the 1800-chip pilot secondary (18 s, bit-exact, runtime "
                     "multiply); off = 10 ms primary loop only.")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffers (unity passband gain, so "
                     "it preserves what it passes). Live.")
        .integer("-Nulls", "--nulls", min=1, max=MAX_NULLS, default=7,
                 presets=NULL_PRESETS, required=False, live=True,
                 help="Passband edge, as the null it snaps to: ±n·1.023 MHz, right on the "
                      "null between lobes. n=2 keeps the BOC(1,1) core, n=7 the full QMBOC "
                      "(incl. the BOC(6,1) lobes). Live (rebuilds the filtered loops).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.5,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loops).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()
    center_freq_hz = args.freq * 1e6

    pmap = power_map()
    amplitude = pmap.amplitude

    # Gain precedence: explicit --gain (raw) > calibrated --power > refuse (uncalibrated).
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=center_freq_hz)
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    # Prebuild the unfiltered component loops once (PRN/component fixed per run).
    data_base, pilot_base, nsamp = build_b1c_components(args.prn)
    want_data = args.component in ("both", "data")
    want_pilot = args.component in ("both", "pilot")

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b1c_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "nulls": int(getattr(args, "nulls", 7) or 7),
             "trans_hz": float(getattr(args, "transition", 0.5) or 0.5) * 1e6}

    def make_component(base):
        if not shape["on"]:
            return base, {"on": False}
        filtered, taps, fp = filter_buffer(base, shape["nulls"], shape["trans_hz"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp, "trans_hz": shape["trans_hz"]}

    finfo = {"on": shape["on"], "edge_hz": shape["nulls"] * B1C_NULL_HZ,
             "trans_hz": shape["trans_hz"]}
    data_file = pilot_file = None
    if want_data:
        iq, finfo = make_component(data_base)
        data_file = write_buffer(iq)
    if want_pilot:
        iq, finfo = make_component(pilot_base)
        pilot_file = write_buffer(iq)

    sec_signs = secondary_signs(args.prn) if (pilot_file and args.secondary == "full") else None
    box = {"data": data_file, "pilot": pilot_file}

    tb = _build_top_block(data_file, pilot_file, sec_signs, nsamp,
                          center_freq_hz, gain_db, amplitude)

    def regenerate():
        info = {"on": shape["on"], "edge_hz": shape["nulls"] * B1C_NULL_HZ,
                "trans_hz": shape["trans_hz"]}
        if want_data:
            iq, info = make_component(data_base)
            new = write_buffer(iq)
            tb.swap_data(new)
            old, box["data"] = box["data"], new
            try:
                os.unlink(old)
            except OSError:
                pass
        if want_pilot:
            iq, info = make_component(pilot_base)
            new = write_buffer(iq)
            tb.swap_pilot(new)
            old, box["pilot"] = box["pilot"], new
            try:
                os.unlink(old)
            except OSError:
                pass
        return info

    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def _fmt_band(info):
        if not info.get("on"):
            return "off (full signal)"
        return (f"on — passband ±{info['edge_hz']/1e6:.2f} MHz, "
                f"{info['trans_hz']/1e6:g} MHz transition"
                + (f", {info['taps']} taps" if 'taps' in info else ""))

    sec_desc = ("18 s (full secondary)" if sec_signs is not None else "10 ms (primary only)")
    print(f"── {SIGNAL_NAME} TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B1C Weil codes, {args.component})")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : QMBOC(6,1,4/33) — BOC(1,1) data + pilot, BOC(6,1) quad")
    print(f"  period         : {sec_desc}")
    print(f"  primary buffer : {nsamp} samples ({nsamp*8/1e6:.1f} MB/component file)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=center_freq_hz):.2f} dBm")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  filter         : {_fmt_band(finfo)}")
    print(f"  otw            : {OTW_FORMAT}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "power" and pmap.has_absolute:
            state["gain"] = pmap.gain_for_power(float(value), freq=center_freq_hz)
            if state["rf_on"]:
                tb.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=center_freq_hz), 2))
        elif name == "gain":
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
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
        elif name in ("filter", "nulls", "transition"):
            if name == "filter":
                shape["on"] = str(value).strip().lower() in ("on", "1", "true", "yes")
            elif name == "nulls":
                shape["nulls"] = max(1, min(MAX_NULLS, int(value)))
            else:
                shape["trans_hz"] = float(value) * 1e6
            regenerate()
            ctrl.report(name, value)

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

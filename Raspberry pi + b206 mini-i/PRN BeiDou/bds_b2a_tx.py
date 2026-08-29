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
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only). See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "bds_b2a"


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

B2A_HZ = 1176.45e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
WEIL_N = 1021
DATA_SEC = (0, 0, 0, 1, 0)                # fixed 5-chip data secondary "00010"
SIGNAL_NAME = "BeiDou B2a"

DATA_G1 = (1, 5, 11, 13)
DATA_G2 = (3, 5, 9, 11, 12, 13)
PILOT_G1 = (3, 6, 7, 13)
PILOT_G2 = (1, 5, 7, 8, 12, 13)

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6          # 6 samples/chip at 10.23 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"              # over-the-wire; halves USB load

# Filter: BPSK(10) is a sinc² with nulls every 10.23 MHz. --sidelobes n keeps the main
# lobe + n sidelobes (±(n+1)·10.23 MHz). At 61.38 MHz (±30.69) n=2 is the whole signal.
B2A_NULL_HZ = 10.23e6
MAX_SIDELOBES = 2
SIDELOBE_PRESETS = {
    "Main lobe only (±10.23 MHz)": 0,
    "Main + 1 sidelobe (±20.46 MHz)": 1,
    "Main + 2 sidelobes (±30.69 MHz, ≈ full)": 2,
}

FREQUENCIES = {"BeiDou B2a (1176.45 MHz)": B2A_HZ / 1e6}   # presets in MHz

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

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_b2a_buffer(1, "both", "primary")

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=1, trans_hz=1.0e6)
    main = 10 * np.log10(band(filt, 0, B2A_NULL_HZ) / band(base, 0, B2A_NULL_HZ))
    cut = 10 * np.log10(band(filt, 24e6, 30e6) / max(band(base, 24e6, 30e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+1 sidelobe, {taps} taps): main lobe {main:+.3f} dB, far sidelobe "
          f"{cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (tiered QPSK, seamless loop) ───────────────────────────────

def build_b2a_buffer(prn: int, component: str, loop: str):
    """Build a complex64 B2a baseband buffer at the fixed SAMP_RATE_HZ. component:
    'both'|'data'|'pilot'. loop: 'full' (100 ms tiered) | 'primary' (1 ms). Returns
    (iq, n_samples)."""
    import numpy as np

    pd = np.asarray(_primary(prn, "data"), dtype=np.int8)
    pp = np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    sd = np.asarray(DATA_SEC, dtype=np.int8)
    sp = np.asarray(_pilot_secondary(prn), dtype=np.int8)

    n_periods = 100 if loop == "full" else 1
    sr = int(round(SAMP_RATE_HZ))
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


def filter_buffer(base_iq, sidelobes: int, trans_hz: float):
    """Circularly filter the looped B2a buffer to keep the main lobe + `sidelobes` sidelobes
    (±(n+1)·10.23 MHz). Circular convolution keeps the result exactly periodic (seam-free
    loop); unity passband gain leaves the kept lobes' power unchanged. Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * B2A_NULL_HZ
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class B2ATx(gr.top_block):
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
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def set_amplitude(self, a): self.amp.set_k(a)
        def swap_file(self, path): self.src.open(path, True)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return B2ATx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (real BDS-SIS-ICD-B2a codes, QPSK: data + pilot, "
               "tiered Gold/Weil, 10.23 Mcps) — fixed 61.38 MHz / sc8, looped buffer, "
               "optional power-preserving digital passband filter. Level is set in dBm via "
               "the unit's calibration; uncalibrated it runs on a relative gain. Authorised, "
               "shielded setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=B2A_HZ / 1e6,
                help="RF carrier in MHz (default B2a = 1176.45). Fixed per run.")
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
                help="both = QPSK (data I + pilot Q); data or pilot = one channel.")
        .choice("-Loop", "--loop", options=["full", "primary"], default="full",
                help="full = 100 ms tiered (bit-exact, ~49 MB); primary = 1 ms "
                     "(small, envelope-correct).")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain, so "
                     "it preserves what it passes). Live.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, default=1,
                 presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of sidelobes KEPT beside the main lobe: "
                      "a ±(n+1)·10.23 MHz band. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.1, max=8.0, default=1.0,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loop).")
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

    base_iq, nsamp = build_b2a_buffer(args.prn, args.component, args.loop)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b2a_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "sidelobes": int(getattr(args, "sidelobes", 1) or 0),
             "trans_hz": float(getattr(args, "transition", 1.0) or 1.0) * 1e6}

    def make_current():
        if not shape["on"]:
            return base_iq, {"on": False}
        filtered, taps, fp = filter_buffer(base_iq, shape["sidelobes"], shape["trans_hz"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "trans_hz": shape["trans_hz"]}

    iq0, finfo = make_current()
    box = {"file": write_buffer(iq0)}

    tb = _build_top_block(box["file"], center_freq_hz, gain_db, amplitude)

    def regenerate():
        iq, info = make_current()
        new_file = write_buffer(iq)
        tb.swap_file(new_file)
        old, box["file"] = box["file"], new_file
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
                f"{info['trans_hz']/1e6:g} MHz transition, {info['taps']} taps")

    print(f"── {SIGNAL_NAME} TX ───────────────────────────────────────────")
    print(f"  PRN            : {args.prn}  (real B2a codes, {args.component})")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : QPSK (BPSK(10) data + pilot), 10.23 Mcps")
    print(f"  loop           : {args.loop}, {nsamp} samples ({nsamp*8/1e6:.1f} MB)")
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
        elif name in ("filter", "sidelobes", "transition"):
            if name == "filter":
                shape["on"] = str(value).strip().lower() in ("on", "1", "true", "yes")
            elif name == "sidelobes":
                shape["sidelobes"] = max(0, min(MAX_SIDELOBES, int(value)))
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

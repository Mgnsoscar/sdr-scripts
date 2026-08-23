#!/usr/bin/env python3
"""
GPS L1C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a spectrally-correct GPS **L1C** signal (1575.42 MHz) — the modernized
civil L1 signal — as the in-phase sum of its pilot and data components:

    L1Cp (pilot, 75% power) : Weil code × TMBOC(6,1,4/33) subcarrier × overlay
    L1Cd (data,  25% power) : Weil code × BOC(1,1) subcarrier   (bare code here)

Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real IS-GPS-800 Weil codes
──────────────────────────────────────────
Both 10230-chip primary codes are the real IS-GPS-800 codes: Legendre (mod 10223)
→ Weil W[k]=L[k]⊕L[k+w] → 7-chip insertion [0110100] at index p. Per-PRN (w,p) for
pilot and data were validated against the official L1C PRN Code Assignments sheet.
The pilot **overlay** (1800-symbol secondary, 18 s period) is the single 11-bit
LFSR of IS-GPS-800; its per-PRN polynomial/init table was cross-checked against
the sheet's L1CO columns (matches 14/14 where the sheet enumerates them, PRN
159–172) — note the overlay has no published *code* check-value, so unlike the
primaries it is table-validated + community-verified (pmonta/GNSS-DSP-tools),
not first/last-chip verified.

Full-length overlay WITHOUT a multi-GB file
───────────────────────────────────────────
The overlay is a *slow* code: one ±1 symbol per 10 ms primary period, so the 18 s
tiered pilot is the 10 ms pilot buffer replayed 1800 times with a per-period sign
flip. Rather than precompute 18 s (~7 GB), the flow applies it at runtime:

    pilot_file ─► × ─────────┐
    overlay(1800)─►repeat(N)─┘ ├─► + ─► (amp) ─► USRP
    data_file  ──────────────┘

so the full 18 s signal streams from ~8 MB. `--secondary off` drops it (10 ms
primary loop only; spectrally identical, no secondary sync).

⚠  RF SAFETY / LEGAL: L1 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Sample rate default 49.104 MHz (=48×1.023 → BOC(6,1) half-chips land on samples).
1:1 master clock; sc8. Level set in dBm (--power) with a live RF on/off (--rf);
see the USER CALIBRATION block. No navigation data (bare code).

CLI
───
    gps_l1c_tx.py --prn 5 --power -30
    gps_l1c_tx.py --component pilot --secondary full
    gps_l1c_tx.py --self-test
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
# (e.g. EIRP). Absent it, the baked USER CALIBRATION constants below are used
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1c"


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


# ── Power map: the unit's injected calibration curve if present, else the baked
#    constants above (identical to the old single-anchor slope-1 behaviour) ────────

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else the baked constants above. Cached, so build_script and
    main share one — and so --power's schema bounds match the real operating range
    (calibrated → e.g. EIRP; else the baked SDR-port range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
    return _PMAP


# ── Constants ─────────────────────────────────────────────────────────────────

L1_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10
LEG_N = 10223
INSERT = (0, 1, 1, 0, 1, 0, 0)
TMBOC = (1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0)
SEC_LEN = 1800
A_PILOT = math.sqrt(0.75)
A_DATA = math.sqrt(0.25)

FREQUENCIES = {"GPS L1 (1575.42 MHz)": L1_HZ}

# Per-PRN (Weil index w, insertion index p), IS-GPS-800 (validated vs the sheet).
L1CP_WP = (
    (5111,412), (5109,161), (5108,1), (5106,303), (5103,207), (5101,4971),
    (5100,4496), (5098,5), (5095,4557), (5094,485), (5093,253), (5091,4676),
    (5090,1), (5081,66), (5080,4485), (5069,282), (5068,193), (5054,5211),
    (5044,729), (5027,4848), (5026,982), (5014,5955), (5004,9805), (4980,670),
    (4915,464), (4909,29), (4893,429), (4885,394), (4832,616), (4824,9457),
    (4591,4429), (3706,4771), (5092,365), (4986,9705), (4965,9489), (4920,4193),
    (4917,9947), (4858,824), (4847,864), (4790,347), (4770,677), (4318,6544),
    (4126,6312), (3961,9804), (3790,278), (4911,9461), (4881,444), (4827,4839),
    (4795,4144), (4789,9875), (4725,197), (4675,1156), (4539,4674), (4535,10035),
    (4458,4504), (4197,5), (4096,9937), (3484,430), (3481,5), (3393,355),
    (3175,909), (2360,1622), (1852,6284),
)
L1CD_WP = (
    (5097,181), (5110,359), (5079,72), (4403,1110), (4121,1480), (5043,5034),
    (5042,4622), (5104,1), (4940,4547), (5035,826), (4372,6284), (5064,4195),
    (5084,368), (5048,1), (4950,4796), (5019,523), (5076,151), (3736,713),
    (4993,9850), (5060,5734), (5061,34), (5096,6142), (4983,190), (4783,644),
    (4991,467), (4815,5384), (4443,801), (4769,594), (4879,4450), (4894,9437),
    (4985,4307), (5056,5906), (4921,378), (5036,9448), (4812,9432), (4838,5849),
    (4855,5547), (4904,9546), (4753,9132), (4483,403), (4942,3766), (4813,3),
    (4957,684), (4618,9711), (4669,333), (4969,6124), (5031,10216), (5038,4251),
    (4740,9893), (4073,9884), (4843,4627), (4979,4449), (4867,9798), (4964,985),
    (5025,4272), (4579,126), (4390,10024), (4763,434), (4612,1029), (4784,561),
    (3716,289), (4703,638), (4851,4353),
)
# Pilot overlay: (S1 polynomial octal, S1 initial state octal), IS-GPS-800.
L1CO_PARAMS = (
    (0o5111,0o3266), (0o5421,0o2040), (0o5501,0o1527), (0o5403,0o3307), (0o6417,0o3756), (0o6141,0o3026),
    (0o6351,0o0562), (0o6501,0o0420), (0o6205,0o3415), (0o6235,0o0337), (0o7751,0o0265), (0o6623,0o1230),
    (0o6733,0o2204), (0o7627,0o1440), (0o5667,0o2412), (0o5051,0o3516), (0o7665,0o2761), (0o6325,0o3750),
    (0o4365,0o2701), (0o4745,0o1206), (0o7633,0o1544), (0o6747,0o1774), (0o4475,0o0546), (0o4225,0o2213),
    (0o7063,0o3707), (0o4423,0o2051), (0o6651,0o3650), (0o4161,0o1777), (0o7237,0o3203), (0o4473,0o1762),
    (0o5477,0o2100), (0o6163,0o0571), (0o7223,0o3710), (0o6323,0o3535), (0o7125,0o3110), (0o7035,0o1426),
    (0o4341,0o0255), (0o4353,0o0321), (0o4107,0o3124), (0o5735,0o0572), (0o6741,0o1736), (0o7071,0o3306),
    (0o4563,0o1307), (0o5755,0o3763), (0o6127,0o1604), (0o4671,0o1021), (0o4511,0o2624), (0o4533,0o0406),
    (0o5357,0o0114), (0o5607,0o0077), (0o6673,0o3477), (0o6153,0o1000), (0o7565,0o3460), (0o7107,0o2607),
    (0o6211,0o2057), (0o4321,0o3467), (0o7201,0o0706), (0o4451,0o2032), (0o5411,0o1464), (0o5141,0o0520),
    (0o7041,0o1766), (0o6637,0o3270), (0o4577,0o0341),
)


# ── L1C codes (bit-exact primaries + overlay, IS-GPS-800) ──────────────────────

_LEG: list | None = None


def _legendre() -> list:
    global _LEG
    if _LEG is None:
        qr = {(x * x) % LEG_N for x in range(1, LEG_N)}
        _LEG = [0] + [1 if k in qr else 0 for k in range(1, LEG_N)]
    return _LEG


def _primary(prn: int, component: str) -> list[int]:
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    w, p = (L1CP_WP if component == "pilot" else L1CD_WP)[prn - 1]
    L = _legendre()
    W = [L[k] ^ L[(k + w) % LEG_N] for k in range(LEG_N)]
    return W[0:p - 1] + list(INSERT) + W[p - 1:LEG_N]


def _overlay(prn: int) -> list[int]:
    """The 1800-symbol pilot overlay (0/1), IS-GPS-800 single 11-bit LFSR."""
    poly, init = L1CO_PARAMS[prn - 1]
    p = [((poly // 2) >> i) & 1 for i in range(11)]
    x = [(init >> i) & 1 for i in range(11)]
    c = [0] * SEC_LEN
    for i in range(SEC_LEN):
        c[i] = x[10]
        fb = 0
        for a, b in zip(x, p):
            fb ^= a & b
        x = [fb] + x[:-1]
    return c


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v

    leg = _legendre()
    lok = sum(leg) == (LEG_N - 1) // 2
    print(f"Legendre({LEG_N}) ones={sum(leg)} (expect {(LEG_N-1)//2}) [{'OK' if lok else 'FAIL'}]")
    tok = sum(TMBOC) == 4 and [i for i, x in enumerate(TMBOC) if x] == [0, 4, 6, 29]
    print(f"TMBOC BOC(6,1) chips {[i for i,x in enumerate(TMBOC) if x]} [{'OK' if tok else 'FAIL'}]")
    ok = ok and lok and tok

    prim = {("P", 1): 0o5752067, ("P", 2): 0o70146401, ("P", 63): 0o56350460,
            ("D", 1): 0o77001425, ("D", 2): 0o23342754, ("D", 63): 0o34665654}
    for (kind, prn), want in prim.items():
        c = _primary(prn, "pilot" if kind == "P" else "data")
        good = o24(c) == want and len(c) == CODE_LEN and sum(c) == 5115
        ok = ok and good
        print(f"{'L1Cp' if kind=='P' else 'L1Cd'} PRN{prn:2d}: first24={oct(o24(c))} "
              f"expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    # Overlay: regression guard (from IS-GPS-800 LFSR params) + structure.
    ovl = {1: 0o65550354, 63: 0o7034020}
    for prn, want in ovl.items():
        c = _overlay(prn)
        good = o24(c) == want and len(c) == SEC_LEN and sum(c) == 900
        ok = ok and good
        print(f"overlay PRN{prn:2d}: first24={oct(o24(c))} expect={oct(want)} "
              f"len={len(c)} ones={sum(c)} [{'OK' if good else 'FAIL'}]")

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffers (data + pilot components, one 10 ms primary period) ───────

def build_l1c_components(prn: int, samp_rate_hz: float):
    """Return (data_buf, pilot_buf, n_samples): the two in-phase component buffers
    (complex64, one 10 ms primary period), commonly peak-normalised. The pilot
    overlay is applied downstream."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(PRIMARY_MS * 1e-3 * sr))

    pilot = 1 - 2 * np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    data = 1 - 2 * np.asarray(_primary(prn, "data"), dtype=np.int8)
    tmboc = np.asarray(TMBOC, dtype=np.int8)

    n = np.arange(n_samples, dtype=np.int64)
    num = n * CHIP_RATE_HZ
    chip = num // sr
    rem = num - chip * sr
    boc11 = np.where(rem * 2 < sr, 1.0, -1.0)
    boc61 = np.where(((rem * 12) // sr) % 2 == 0, 1.0, -1.0)
    cidx = chip % CODE_LEN
    use61 = tmboc[chip % 33] == 1

    data_s = (A_DATA * data[cidx] * boc11).astype(np.complex128)
    pilot_s = (A_PILOT * pilot[cidx] * np.where(use61, boc61, boc11)).astype(np.complex128)
    norm = max(np.max(np.abs(data_s + pilot_s)), np.max(np.abs(data_s - pilot_s)))
    return (data_s / norm).astype(np.complex64), (pilot_s / norm).astype(np.complex64), n_samples


def overlay_signs(prn: int):
    """The pilot overlay as ±1 complex values (1800), for the runtime multiply."""
    import numpy as np
    return (1.0 - 2.0 * np.asarray(_overlay(prn), dtype=np.float32)).astype(np.complex64)


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(data_file, pilot_file, sec_signs, period_samples,
                     center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class L1CTx(gr.top_block):
        def __init__(self):
            super().__init__("GPS L1C TX")
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

    return L1CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("GPS L1C transmitter (real IS-GPS-800 Weil codes, TMBOC(6,1,4/33) "
               "pilot + BOC(1,1) data, full 18 s overlay), file-replay. Authorised, "
               "shielded setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63). Fixed per run.")
        .choice("-Component", "--component", options=["both", "pilot", "data"],
                default="both",
                help="both = full L1C (25/75); pilot = L1Cp TMBOC only; data = L1Cd.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="full = apply the 1800-symbol pilot overlay (18 s, runtime "
                     "multiply); off = 10 ms primary loop only.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L1_HZ,
                help="RF carrier (default L1). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(power_map().min_power_dbm, 2),
                max=round(power_map().max_power_dbm, 2),
                default=round(power_map().max_power_dbm, 2), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given (relative wins). Live.")
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
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power).
        # This is also the calibration knob — set it while measuring output vs gain on
        # a spectrum analyser to fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB above.
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. Live.")
    )
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
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present,
    # so you can command a gain directly or measure output power at it.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else pmap.gain_for_power(args.power)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gps_l1c_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    data_buf, pilot_buf, nsamp = build_l1c_components(args.prn, samp_rate_hz)

    data_file = pilot_file = None
    if args.component in ("both", "data"):
        data_file = os.path.join(tmpdir, "data.fc32")
        data_buf.tofile(data_file)
    if args.component in ("both", "pilot"):
        pilot_file = os.path.join(tmpdir, "pilot.fc32")
        pilot_buf.tofile(pilot_file)

    sec_signs = None
    if pilot_file and args.secondary == "full":
        sec_signs = overlay_signs(args.prn)

    tb = _build_top_block(data_file, pilot_file, sec_signs, nsamp,
                          args.freq, samp_rate_hz, gain_db, amplitude,
                          args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    sec_desc = "18 s (full overlay)" if sec_signs is not None else "10 ms (primary only)"
    print("── GPS L1C TX ──────────────────────────────────────────────")
    print(f"  satellite PRN  : {args.prn}  (real L1C Weil codes, {args.component})")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : L1Cp TMBOC(6,1,4/33) + L1Cd BOC(1,1), 75/25 power")
    print(f"  period         : {sec_desc}")
    print(f"  primary buffer : {nsamp} samples ({nsamp*8/1e6:.1f} MB/file)")
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

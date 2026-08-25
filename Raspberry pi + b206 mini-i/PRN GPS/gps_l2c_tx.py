#!/usr/bin/env python3
"""
GPS L2C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** GPS **L2C** signal (1227.60 MHz): the civil signal on
L2, which is the chip-by-chip time-multiplex of two codes at 511.5 kcps each,
interleaved to 1.023 Mcps (BPSK-R(1), ~2 MHz wide) —

    L2 CM (Civil Moderate) : 10230 chips, 20 ms period   (carries CNAV; here bare)
    L2 CL (Civil Long)     : 767250 chips, 1.5 s period  (dataless pilot)

Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real IS-GPS-200 codes
─────────────────────────────────────
Both codes come from the IS-GPS-200 27-stage Galois shift register (feedback mask
0o445112474, output = LSB), each started from its per-PRN initial state. The
generator was validated against the OFFICIAL "L2C PRN Code Assignments" sheet:
running the register from the sheet's Initial State for a full code period lands
exactly on the sheet's End State — confirmed for all 52 listed PRNs (CM) and
spot-checked for CL. Per-PRN initial states (PRN 1..63) are the IS-GPS-200 values
(cross-checked against pmonta/GNSS-DSP-tools); --self-test re-verifies them.

⚠  RF SAFETY / LEGAL: L2 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Loop length (--loop)
────────────────────
  full (default) : one whole CL period = 1.5 s (CM repeats 75×). Bit-exact,
                   complete spectrum. Large file (scales with sample rate).
  cm             : one CM period = 20 ms (CL truncated to its first 10230 chips).
                   Small/fast; the BPSK-R(1) envelope is identical, but the CL
                   line structure appears at 50 Hz instead of its true 0.667 Hz
                   (both unresolvable at practical RBW). Good for envelope checks.

Sample rate: L2C is ~2 MHz wide, so the default is 10.23 MHz (= 10 × 1.023, an
exact 10 samples/chip). At full-loop that's ~123 MB in /dev/shm; raising the rate
raises the file size proportionally (1.5 s × fs × 8 bytes). Master clock is pinned
equal to the sample rate (1:1). Level set in dBm (--power) with a live RF on/off
(--rf); see the USER CALIBRATION block.

CLI
───
    gps_l2c_tx.py --prn 5 --power -30
    gps_l2c_tx.py --prn 5 --loop cm --sample_rate 40.92   # small envelope version
    gps_l2c_tx.py --self-test
    gps_l2c_tx.py --describe-params
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
# (e.g. EIRP). Absent it, the baked USER CALIBRATION constants below are used
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l2c"


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

L2_HZ = 1227.60e6
CM_LEN = 10230                   # L2 CM code length (20 ms at 511.5 kcps)
CL_LEN = 767250                  # L2 CL code length (1.5 s at 511.5 kcps)
CHANNEL_CHIP_RATE = 511_500      # each of CM / CL
COMBINED_CHIP_RATE = 1_023_000   # after chip-by-chip multiplexing
LFSR_MASK = 0o445112474          # IS-GPS-200 L2C feedback polynomial (Galois)

FREQUENCIES = {"GPS L2 (1227.60 MHz)": L2_HZ}

# Per-PRN initial shift-register states (octal), IS-GPS-200, PRN 1..63.
L2CM_INIT = (
    0o742417664, 0o756014035, 0o002747144, 0o066265724, 0o601403471, 0o703232733,
    0o124510070, 0o617316361, 0o047541621, 0o733031046, 0o713512145, 0o024437606,
    0o021264003, 0o230655351, 0o001314400, 0o222021506, 0o540264026, 0o205521705,
    0o064022144, 0o120161274, 0o044023533, 0o724744327, 0o045743577, 0o741201660,
    0o700274134, 0o010247261, 0o713433445, 0o737324162, 0o311627434, 0o710452007,
    0o722462133, 0o050172213, 0o500653703, 0o755077436, 0o136717361, 0o756675453,
    0o435506112, 0o771353753, 0o226107701, 0o022025110, 0o402466344, 0o752566114,
    0o702011164, 0o041216771, 0o047457275, 0o266333164, 0o713167356, 0o060546335,
    0o355173035, 0o617201036, 0o157465571, 0o767360553, 0o023127030, 0o431343777,
    0o747317317, 0o045706125, 0o002744276, 0o060036467, 0o217744147, 0o603340174,
    0o326616775, 0o063240065, 0o111460621,
)
L2CL_INIT = (
    0o624145772, 0o506610362, 0o220360016, 0o710406104, 0o001143345, 0o053023326,
    0o652521276, 0o206124777, 0o015563374, 0o561522076, 0o023163525, 0o117776450,
    0o606516355, 0o003037343, 0o046515565, 0o671511621, 0o605402220, 0o002576207,
    0o525163451, 0o266527765, 0o006760703, 0o501474556, 0o743747443, 0o615534726,
    0o763621420, 0o720727474, 0o700521043, 0o222567263, 0o132765304, 0o746332245,
    0o102300466, 0o255231716, 0o437661701, 0o717047302, 0o222614207, 0o561123307,
    0o240713073, 0o101232630, 0o132525726, 0o315216367, 0o377046065, 0o655351360,
    0o435776513, 0o744242321, 0o024346717, 0o562646415, 0o731455342, 0o723352536,
    0o000013134, 0o011566642, 0o475432222, 0o463506741, 0o617127534, 0o026050332,
    0o733774235, 0o751477772, 0o417631550, 0o052247456, 0o560404163, 0o417751005,
    0o004302173, 0o715005045, 0o001154457,
)


# ── L2C codes (bit-exact 27-stage Galois LFSR, IS-GPS-200) ─────────────────────

def _lfsr_bits(init: int, n: int) -> list[int]:
    """n output chips (LSB each) of the L2C register started at `init`."""
    x = init
    out = [0] * n
    for i in range(n):
        out[i] = x & 1
        x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
    return out


def cm_code(prn: int) -> list[int]:
    """L2 CM code (10230 chips, 0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CM_INIT[prn - 1], CM_LEN)


def cl_code(prn: int, n: int = CL_LEN) -> list[int]:
    """L2 CL code (n chips, 0/1) for a PRN (1..63); n<CL_LEN truncates it."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    return _lfsr_bits(L2CL_INIT[prn - 1], n)


# ── Self-test (generator vs official sheet + check values; no hardware) ────────

def _self_test() -> int:
    ok = True

    def run(init, steps):
        x = init
        for _ in range(steps):
            x = (x >> 1) ^ ((x & 1) * LFSR_MASK)
        return x

    # Ground truth from the official L2C PRN Code Assignments sheet (PRN 159/160):
    # register run from Initial State reaches End State at the last chip.
    sheet = [  # (cm_init, cm_end, cl_init, cl_end)
        (0o604055104, 0o425373114, 0o605253024, 0o44547544),    # PRN 159
        (0o157065232, 0o427153064, 0o63314262, 0o707116115),    # PRN 160
    ]
    for i, (cmi, cme, cli, cle) in enumerate(sheet):
        cm_ok = run(cmi, CM_LEN - 1) == cme
        ok = ok and cm_ok
        print(f"sheet PRN{159+i} CM init→end ({CM_LEN} chips): [{'OK' if cm_ok else 'FAIL'}]")
    cl_ok = run(sheet[0][2], CL_LEN - 1) == sheet[0][3]
    ok = ok and cl_ok
    print(f"sheet PRN159 CL init→end ({CL_LEN} chips): [{'OK' if cl_ok else 'FAIL'}]")

    # first-24-chip octal check values (transcription guard for the init tables).
    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {("CM", 1): 0o12757036, ("CM", 2): 0o50370043,
              ("CL", 1): 0o24676104, ("CL", 2): 0o20022732}
    for (kind, prn), want in checks.items():
        got = o24(cm_code(prn) if kind == "CM" else cl_code(prn, 24))
        good = got == want
        ok = ok and good
        print(f"{kind} PRN{prn}: first24={oct(got)} expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    # Structural: lengths and rough balance.
    cm1 = cm_code(1)
    print(f"CM len={len(cm1)} ones={sum(cm1)} (≈5115) [{'OK' if len(cm1)==CM_LEN else 'FAIL'}]")
    ok = ok and len(cm1) == CM_LEN

    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (CM/CL time-multiplexed, seamless loop) ────────────────────

def build_l2c_buffer(prn: int, loop: str, samp_rate_hz: float):
    """Build a complex64 L2C baseband buffer (real BPSK, Q=0). loop='full' → one
    1.5 s CL period; loop='cm' → one 20 ms CM period (CL truncated). Returns
    (iq, n_samples)."""
    import numpy as np

    n_cl = CL_LEN if loop == "full" else CM_LEN
    cm = np.asarray(cm_code(prn), dtype=np.int8)
    cl = np.asarray(cl_code(prn, n_cl), dtype=np.int8)

    period_s = n_cl / CHANNEL_CHIP_RATE          # 1.5 s (full) or 20 ms (cm)
    sr = int(round(samp_rate_hz))
    n_samples = int(round(period_s * sr))

    n = np.arange(n_samples, dtype=np.int64)
    gchip = n * COMBINED_CHIP_RATE // sr         # 0 .. 2*n_cl-1
    half = (gchip >> 1)
    is_cl = (gchip & 1) == 1
    bit = np.where(is_cl, cl[half % n_cl], cm[half % CM_LEN])
    iq = (1.0 - 2.0 * bit).astype(np.complex64)
    return iq, n_samples


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class L2CTx(gr.top_block):
        def __init__(self):
            super().__init__("GPS L2C TX")
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

    return L2CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("GPS L2C transmitter (CM/CL time-multiplexed, real IS-GPS-200 "
               "codes, 1.023 Mcps BPSK), file-replay. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63) — the real L2C code. Fixed per run.")
        .choice("-Loop", "--loop", options=["full", "cm"], default="full",
                help="full = whole 1.5 s CL period (bit-exact, big file); "
                     "cm = 20 ms CM period (CL truncated; small, envelope-correct).")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L2_HZ,
                help="RF carrier (default L2). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(power_map().min_power_dbm, 2),
                max=round(power_map().max_power_dbm, 2),
                default=round(power_map().max_power_dbm, 2), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given (relative wins). Live.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=5.0, max=61.44,
                default=10.23,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "L2C is ~2 MHz wide; 10.23 → 10 samples/chip. Fixed per run.")
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
    tmpdir = tempfile.mkdtemp(prefix="gps_l2c_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp = build_l2c_buffer(args.prn, args.loop, samp_rate_hz)
    iq_file = os.path.join(tmpdir, f"l2c_prn{args.prn}_{args.loop}.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(iq_file, args.freq, samp_rate_hz, gain_db,
                          amplitude, args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── GPS L2C TX ──────────────────────────────────────────────")
    print(f"  satellite PRN  : {args.prn}  (real L2C code, CM+CL)")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  modulation     : BPSK-R(1) — 1.023 Mcps (CM/CL @ 511.5 kcps each)")
    print(f"  loop           : {args.loop} "
          f"({'1.5 s (bit-exact)' if args.loop=='full' else '20 ms (CL truncated)'})")
    print(f"  buffer         : {nsamp} samples ({nsamp*8/1e6:.1f} MB)")
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

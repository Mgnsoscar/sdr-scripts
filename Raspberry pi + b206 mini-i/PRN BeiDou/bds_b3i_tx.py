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
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only). See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "bds_b3i"


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

B3_HZ = 1268.52e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
G1_RESET = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0)   # short-cycle state
SIGNAL_NAME = "BeiDou B3I"

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6          # 6 samples/chip at 10.23 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"              # over-the-wire; halves USB load

# Filter: BPSK-R(10) is a sinc² with nulls every 10.23 MHz. --sidelobes n keeps the main
# lobe + n sidelobes (±(n+1)·10.23 MHz). At 61.38 MHz (±30.69) n=2 is the whole signal.
B3_NULL_HZ = 10.23e6
MAX_SIDELOBES = 2
SIDELOBE_PRESETS = {
    "Main lobe only (±10.23 MHz)": 0,
    "Main + 1 sidelobe (±20.46 MHz)": 1,
    "Main + 2 sidelobes (±30.69 MHz, ≈ full)": 2,
}

FREQUENCIES = {"BeiDou B3I (1268.52 MHz)": B3_HZ / 1e6}   # presets in MHz

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

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_b3i_buffer(1)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=1, trans_hz=1.0e6)
    main = 10 * np.log10(band(filt, 0, B3_NULL_HZ) / band(base, 0, B3_NULL_HZ))
    cut = 10 * np.log10(band(filt, 24e6, 30e6) / max(band(base, 24e6, 30e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+1 sidelobe, {taps} taps): main lobe {main:+.3f} dB, far sidelobe "
          f"{cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b3i_buffer(prn: int):
    """Build a complex64 B3I baseband buffer over one 1 ms code period (loops
    seamlessly) at the fixed SAMP_RATE_HZ. Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(0.001 * sr))               # 1 ms — one code period
    bipolar = 1.0 - 2.0 * np.asarray(b3i_code(prn), dtype=np.int8)
    n = np.arange(n_samples, dtype=np.int64)
    chip = (n * CHIP_RATE_HZ // sr) % CODE_LEN
    return bipolar[chip].astype(np.complex64), n_samples


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
    """Circularly filter the looped B3I buffer to keep the main lobe + `sidelobes` sidelobes
    (±(n+1)·10.23 MHz). Circular convolution keeps the result exactly periodic (seam-free
    loop); unity passband gain leaves the kept lobes' power unchanged. Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * B3_NULL_HZ
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class B3ITx(gr.top_block):
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

    return B3ITx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (real BDS-SIS-ICD-B3I ranging code, BPSK-R(10), "
               "10.23 Mcps) — fixed 61.38 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=B3_HZ / 1e6,
                help="RF carrier in MHz (default B3 = 1268.52). Fixed per run.")
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
                 help="BeiDou SV / ranging-code number (1..63). Fixed per run.")
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

    base_iq, nsamp = build_b3i_buffer(args.prn)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b3i_", dir=shm)
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
    print(f"  SV / code num  : {args.prn}  (real B3I ranging code)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : BPSK-R(10) — 10.23 Mcps, ~20.46 MHz wide")
    print(f"  buffer         : {nsamp} samples (1 ms code period, {nsamp*8/1e6:.1f} MB)")
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

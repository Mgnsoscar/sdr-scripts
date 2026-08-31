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
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only). See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "bds_b2b"


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

B2B_HZ = 1207.14e6
CHIP_RATE_HZ = 10_230_000
CODE_LEN = 10230
RESET_CHIP = 8190
G1_TAPS = (1, 9, 10, 13)
G2_TAPS = (3, 4, 6, 9, 12, 13)
SIGNAL_NAME = "BeiDou B2b"

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6          # 6 samples/chip at 10.23 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"              # over-the-wire; halves USB load

# Filter: BPSK-R(10) is a sinc² with nulls every 10.23 MHz. --sidelobes n keeps the main
# lobe + n sidelobes (±(n+1)·10.23 MHz). At 61.38 MHz (±30.69) n=2 is the whole signal.
B2B_NULL_HZ = 10.23e6
MAX_SIDELOBES = 2
SIDELOBE_PRESETS = {
    "Main lobe only (±10.23 MHz)": 0,
    "Main + 1 sidelobe (±20.46 MHz)": 1,
    "Main + 2 sidelobes (±30.69 MHz, ≈ full)": 2,
}

FREQUENCIES = {"BeiDou B2b (1207.14 MHz)": B2B_HZ / 1e6}   # presets in MHz

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

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_b2b_buffer(6)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=1, trans_hz=1.0e6)
    main = 10 * np.log10(band(filt, 0, B2B_NULL_HZ) / band(base, 0, B2B_NULL_HZ))
    cut = 10 * np.log10(band(filt, 24e6, 30e6) / max(band(base, 24e6, 30e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+1 sidelobe, {taps} taps): main lobe {main:+.3f} dB, far sidelobe "
          f"{cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b2b_buffer(prn: int):
    """Build a complex64 B2b_I baseband buffer over one 1 ms code period (loops
    seamlessly) at the fixed SAMP_RATE_HZ. Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(0.001 * sr))
    bipolar = 1.0 - 2.0 * np.asarray(b2b_code(prn), dtype=np.int8)   # logic 1→−1, 0→+1
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
    """Circularly filter the looped B2b buffer to keep the main lobe + `sidelobes` sidelobes
    (±(n+1)·10.23 MHz). Circular convolution keeps the result exactly periodic (seam-free
    loop); unity passband gain leaves the kept lobes' power unchanged. Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * B2B_NULL_HZ
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class B2BTx(gr.top_block):
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

    return B2BTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} (I-component) transmitter (real BDS-SIS-ICD-B2b ranging code, "
               "BPSK-R(10), 10.23 Mcps) — fixed 61.38 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=B2B_HZ / 1e6,
                help="RF carrier in MHz (default B2b = 1207.14). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .integer("-PRN", "--prn", min=6, max=58, default=6, required=True,
                 help="BeiDou PRN / ranging-code number (6..58). Fixed per run.")
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

    base_iq, nsamp = build_b2b_buffer(args.prn)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b2b_", dir=shm)
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
    print(f"  PRN            : {args.prn}  (real B2b_I ranging code)")
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

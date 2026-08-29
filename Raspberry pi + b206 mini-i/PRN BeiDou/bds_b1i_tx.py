#!/usr/bin/env python3
"""
BeiDou B1I transmitter for GNU Radio + UHD (Ettus B200-mini family).

Generates a **bit-exact** BeiDou **B1I** signal (1561.098 MHz): BPSK-R(2) —
a 2.046 Mcps, 2046-chip ranging code, 1 ms period (~4 MHz wide).
Precomputed and replayed from a file (same recipe as gps_l5_tx.py).

Code fidelity — real BDS-SIS-ICD-B1I codes
──────────────────────────────────────────
A balanced Gold code (truncated by its last chip → 2046) from two 11-bit LFSRs
(ICD §4.3):
  G1(X) = 1+X+X^7+X^8+X^9+X^10+X^11
  G2(X) = 1+X+X^2+X^3+X^4+X^5+X^8+X^9+X^11
both initialised 01010101010. The per-SV code is G1(stage 11) ⊕ a XOR of selected
G2 stages (the ICD Table 4-1 "phase assignment", e.g. PRN 1 = 1⊕3, PRN 63 =
3⊕6⊕9). The generator + per-SV tap table match the ICD exactly and are byte-
identical to pmonta/GNSS-DSP-tools' beidou/b1i.py (which is used to acquire live
B1I); --self-test re-checks the codes against embedded reference values.

Scope: loops one 1 ms ranging-code period — spectrally correct and code-exact.
No navigation data / 1 kHz NH secondary code (those ride on the data).

⚠  RF SAFETY / LEGAL: B1I is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

Why it runs on a Pi + live tuning: see gps_l5_tx.py. sc8, 1:1 master clock, quiet;
level set in dBm (--power) with a live RF on/off (--rf). The default 20.46 MHz (= 10×2.046) gives 10 samples/chip.

CLI
───
    bds_b1i_tx.py --prn 6 --power -30
    bds_b1i_tx.py --self-test
    bds_b1i_tx.py --describe-params
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
CAL_SIGNAL_ID = "bds_b1i"


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

B1I_HZ = 1561.098e6
CHIP_RATE_HZ = 2_046_000
CODE_LEN = 2046
G_INIT = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)   # 01010101010, both registers
SIGNAL_NAME = "BeiDou B1I"

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6          # 30 samples/chip at 2.046 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"              # over-the-wire; halves USB load

# Filter: BPSK-R(2) is a sinc² with nulls every 2.046 MHz. --sidelobes n keeps the main
# lobe + n sidelobes (±(n+1)·2.046 MHz). At 61.38 MHz (±30.69) n=14 is the whole signal.
B1I_NULL_HZ = 2.046e6
MAX_SIDELOBES = 14
SIDELOBE_PRESETS = {
    "Main lobe only (±2.05 MHz)": 0,
    "Main + 1 sidelobe (±4.09 MHz)": 1,
    "Main + 2 sidelobes (±6.14 MHz)": 2,
    "Main + 4 sidelobes (±10.23 MHz)": 4,
    "Main + 9 sidelobes (±20.46 MHz)": 9,
}

FREQUENCIES = {"BeiDou B1I (1561.098 MHz)": B1I_HZ / 1e6}   # presets in MHz

# Per-SV G2 phase selection (XORed G2 stages, 1-indexed), BDS-SIS-ICD-B1I Table 4-1.
G2_TAPS = (
    (1,3),(1,4),(1,5),(1,6),(1,8),(1,9),(1,10),(1,11),(2,7),(3,4),
    (3,5),(3,6),(3,8),(3,9),(3,10),(3,11),(4,5),(4,6),(4,8),(4,9),
    (4,10),(4,11),(5,6),(5,8),(5,9),(5,10),(5,11),(6,8),(6,9),(6,10),
    (6,11),(8,9),(8,10),(8,11),(9,10),(9,11),(10,11),
    (1,2,7),(1,3,4),(1,3,6),(1,3,8),(1,3,10),(1,3,11),(1,4,5),(1,4,9),
    (1,5,6),(1,5,8),(1,5,10),(1,5,11),(1,6,9),(1,8,9),(1,9,10),(1,9,11),
    (2,3,7),(2,5,7),(2,7,9),(3,4,5),(3,4,9),(3,5,6),(3,5,8),(3,5,10),
    (3,5,11),(3,6,9),
)


# ── B1I ranging code (bit-exact, BDS-SIS-ICD-B1I §4.3) ─────────────────────────

def _g1_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[6] ^ r[7] ^ r[8] ^ r[9] ^ r[10]] + r[0:10]        # taps 1,7,8,9,10,11


def _g2_step(r: list[int]) -> list[int]:
    return [r[0] ^ r[1] ^ r[2] ^ r[3] ^ r[4] ^ r[7] ^ r[8] ^ r[10]] + r[0:10]  # 1,2,3,4,5,8,9,11


def b1i_code(prn: int) -> list[int]:
    """The 2046-chip B1I ranging code (0/1) for a PRN (1..63)."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    taps = G2_TAPS[prn - 1]
    g1 = list(G_INIT)
    g2 = list(G_INIT)
    out = [0] * CODE_LEN
    for i in range(CODE_LEN):
        g2_out = 0
        for t in taps:
            g2_out ^= g2[t - 1]
        out[i] = g1[10] ^ g2_out
        g1 = _g1_step(g1)
        g2 = _g2_step(g2)
    return out


# ── Self-test (period + code check values; no hardware) ────────────────────────

def _self_test() -> int:
    ok = True

    seen, r = {}, list(G_INIT)
    per = None
    for i in range(3000):
        t = tuple(r)
        if t in seen:
            per = i - seen[t]
            break
        seen[t] = i
        r = _g1_step(r)
    print(f"G1 period={per} (expect 2047) [{'OK' if per==2047 else 'FAIL'}]")
    ok = ok and per == 2047

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v
    checks = {1: 0o31333315, 2: 0o44461070, 6: 0o32442011, 38: 0o67733254, 63: 0o74366441}
    for prn, want in checks.items():
        c = b1i_code(prn)
        got = o24(c)
        good = got == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"B1I PRN{prn:2d}: first24={oct(got)} expect={oct(want)} len={len(c)} "
              f"[{'OK' if good else 'FAIL'}]")

    distinct = len({tuple(b1i_code(p)) for p in range(1, 11)}) == 10
    print(f"PRN 1..10 distinct: {distinct}")
    ok = ok and distinct

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_b1i_buffer(1)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=2, trans_hz=0.5e6)
    main = 10 * np.log10(band(filt, 0, B1I_NULL_HZ) / band(base, 0, B1I_NULL_HZ))
    kept = 10 * np.log10(band(filt, 2 * B1I_NULL_HZ, 3 * B1I_NULL_HZ)
                         / band(base, 2 * B1I_NULL_HZ, 3 * B1I_NULL_HZ))
    cut = 10 * np.log10(band(filt, 12e6, 20e6) / max(band(base, 12e6, 20e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+2 sidelobes, {taps} taps): main lobe {main:+.3f} dB, kept sidelobe "
          f"{kept:+.3f} dB, far sidelobe {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} "
          f"[{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (seamless 1 ms loop) ───────────────────────────────────────

def build_b1i_buffer(prn: int):
    """Build a complex64 B1I baseband buffer over one 1 ms code period (loops
    seamlessly) at the fixed SAMP_RATE_HZ. Real BPSK (Q=0). Returns (iq, n_samples)."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(0.001 * sr))               # 1 ms — one code period
    bipolar = 1.0 - 2.0 * np.asarray(b1i_code(prn), dtype=np.int8)
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
    """Circularly filter the looped B1I buffer to keep the main lobe + `sidelobes` sidelobes
    (±(n+1)·2.046 MHz). Circular convolution keeps the result exactly periodic (seam-free
    loop); unity passband gain leaves the kept lobes' power unchanged. Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * B1I_NULL_HZ
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class B1ITx(gr.top_block):
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

    return B1ITx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (real BDS-SIS-ICD-B1I ranging code, BPSK-R(2), "
               "2.046 Mcps) — fixed 61.38 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=B1I_HZ / 1e6,
                help="RF carrier in MHz (default B1I = 1561.098). Fixed per run.")
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
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, default=2,
                 presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of sidelobes KEPT beside the main lobe: "
                      "a ±(n+1)·2.046 MHz band. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.5,
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

    base_iq, nsamp = build_b1i_buffer(args.prn)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="bds_b1i_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "sidelobes": int(getattr(args, "sidelobes", 2) or 0),
             "trans_hz": float(getattr(args, "transition", 0.5) or 0.5) * 1e6}

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
    print(f"  SV / code num  : {args.prn}  (real B1I ranging code)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : BPSK-R(2) — 2.046 Mcps, ~4 MHz wide")
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

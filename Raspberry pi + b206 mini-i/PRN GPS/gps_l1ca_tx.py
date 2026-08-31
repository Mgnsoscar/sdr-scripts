#!/usr/bin/env python3
"""
GPS L1 C/A transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a BPSK GPS L1 C/A Gold code (1.023 Mcps) at the L1 carrier (1575.42 MHz),
prebuilt once and looped so a Raspberry Pi sustains the rate with no runtime IQ math.

⚠  RF SAFETY / LEGAL: L1 (1575.42 MHz) is a live GNSS band. Transmit ONLY into a
   shielded/conducted setup (cable + attenuators into a receiver or spectrum analyser)
   you are LICENSED / AUTHORISED to use. Radiating a PRN can jam or spoof real GNSS.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 60 samples/chip, exact), master clock pinned 1:1;
  • over-the-wire sc8 (constant-modulus BPSK loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid (the SDR
gain step and any active-component steps), and the banner reports the power actually
achieved on that grid. --gain instead commands the raw SDR gain (relative), overriding
--power. Uncalibrated, there is no dBm scale — use --gain. (See docs/calibration-v2.md.)

Digital passband filter (on the looped buffer — no runtime DSP)
──────────────────────────────────────────────────────────────
An optional steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power: if the main lobe measures
−2.5 dBm unfiltered it still reads −2.5 dBm filtered — the filter only removes what's outside
the passband (it lowers the main lobe only if the passband is narrow enough to cut into it).
  • --filter on/off             enable/disable (live);
  • --sidelobes <n>             passband keeps the main lobe + n C/A sidelobes, i.e. a
                                ±(n+1)·1.023 MHz band (live, presets by sidelobe count);
  • --transition <MHz>          skirt steepness — the transition width beyond the passband
                                edge (live).
All three are LIVE: changing one rebuilds the (circularly-)filtered loop in RAM and swaps it
in atomically — one brief seam at the swap, then it loops clean; the flowgraph never stops.
Disabling swaps back to the unfiltered loop. The loop is streamed from memory by an in-process
source block, NOT a file: a file_source returns -1 ("done") on any read hiccup, which silently
kills the source (the flowgraph keeps running but transmits nothing until the task is
restarted); an in-RAM loop can't hit a file/read error, so that failure mode is gone.

CLI
───
    gps_l1ca_tx.py --prn 5 --power -30                         # calibrated dBm, no filter
    gps_l1ca_tx.py --prn 5 --gain 60 --filter on --sidelobes 2 # relative gain, main+2 sidelobes
    gps_l1ca_tx.py --self-test        # verify the Gold-code generator (+ filter, if numpy)
    gps_l1ca_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# UHD/GNU Radio must be quiet BEFORE the libraries load (they read these at import). The
# heavy imports live inside main(), so setting them here takes effect.
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")   # no UHD console logging
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")  # no "UUUU" underflow spam
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")        # skip slow pref scan

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l1_ca"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 60 samples/chip at 1.023 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; BPSK is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L1 C/A) ───────────────────────────────────
CARRIER_HZ = 1575.42e6        # GPS L1 (the --freq default; retunable for bench testing)
CODE_RATE_HZ = 1.023e6        # C/A chip rate (~2 MHz null-to-null)
SIGNAL_NAME = "GPS L1 C/A"
CODE_LEN = 1023               # chips in a C/A Gold code period
CA_NULL_HZ = 1.023e6          # main-lobe null spacing == the chip rate; sidelobes step by this

FREQUENCIES = {"GPS L1 (1575.42 MHz)": CARRIER_HZ / 1e6}   # presets are in MHz

# Filter presets: {label: number of C/A sidelobes to KEEP}. The passband is the main lobe
# plus that many sidelobes, i.e. a ±(n+1)·1.023 MHz band. Max keeps the band inside ±Fs/2.
MAX_SIDELOBES = 28
SIDELOBE_PRESETS = {
    "Main lobe only (±1.02 MHz)": 0,
    "Main + 1 sidelobe (±2.05 MHz)": 1,
    "Main + 2 sidelobes (±3.07 MHz)": 2,
    "Main + 3 sidelobes (±4.09 MHz)": 3,
    "Main + 5 sidelobes (±6.14 MHz)": 5,
    "Main + 10 sidelobes (±11.25 MHz)": 10,
}

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present (SDR_CALIBRATION_FILE),
    else uncalibrated (relative gain only). Cached so build_script and main agree — and so
    --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# GPS ICD-200 Table 3-Ia: G2 code-phase tap pairs (1-indexed) selecting each PRN's C/A code.
G2_TAPS = {
    1: (2, 6),   2: (3, 7),   3: (4, 8),   4: (5, 9),   5: (1, 9),   6: (2, 10),
    7: (1, 8),   8: (2, 9),   9: (3, 10), 10: (2, 3),  11: (3, 4),  12: (5, 6),
    13: (6, 7), 14: (7, 8),  15: (8, 9),  16: (9, 10), 17: (1, 4),  18: (2, 5),
    19: (3, 6), 20: (4, 7),  21: (5, 8),  22: (6, 9),  23: (1, 3),  24: (4, 6),
    25: (5, 7), 26: (6, 8),  27: (7, 9),  28: (8, 10), 29: (1, 6),  30: (2, 7),
    31: (3, 8), 32: (4, 9),
}
# ICD reference: first 10 chips of each PRN's C/A code, octal — used only by --self-test.
_FIRST10_OCTAL = {
    1: 0o1440,  2: 0o1620,  3: 0o1710,  4: 0o1744,  5: 0o1133,  6: 0o1455,
    7: 0o1131,  8: 0o1454,  9: 0o1626, 10: 0o1504, 11: 0o1642, 12: 0o1750,
    13: 0o1764, 14: 0o1772, 15: 0o1775, 16: 0o1776, 17: 0o1156, 18: 0o1467,
    19: 0o1633, 20: 0o1715, 21: 0o1746, 22: 0o1763, 23: 0o1063, 24: 0o1706,
    25: 0o1743, 26: 0o1761, 27: 0o1770, 28: 0o1774, 29: 0o1127, 30: 0o1453,
    31: 0o1625, 32: 0o1712,
}


# ── C/A Gold-code generation (pure Python, no NumPy) ────────────────────────────────

def ca_code(prn: int) -> list[int]:
    """The 1023-chip GPS C/A Gold code for a PRN (1..32) as 0/1. Two 10-stage LFSRs
    (seeded all-ones): G1 = x^10+x^3+1, G2 = x^10+x^9+x^8+x^6+x^3+x^2+1; the chip is G1's
    output XOR two PRN-specific G2 taps."""
    if prn not in G2_TAPS:
        raise ValueError(f"PRN must be 1 to 32, got {prn}")
    g1 = [1] * 10
    g2 = [1] * 10
    ta, tb = G2_TAPS[prn]
    out: list[int] = []
    for _ in range(CODE_LEN):
        out.append(g1[9] ^ g2[ta - 1] ^ g2[tb - 1])
        fb1 = g1[2] ^ g1[9]
        fb2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [fb1] + g1[:9]
        g2 = [fb2] + g2[:9]
    return out


# ── Baseband buffer (one seamless-looping code period) ──────────────────────────────

def build_iq_buffer(prn: int):
    """The unit-magnitude complex64 C/A buffer at SAMP_RATE_HZ: a whole number of code
    periods that is an exact integer sample count, so it loops with no seam. BPSK: I = ±1,
    Q = 0 (amplitude is applied downstream). Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(SAMP_RATE_HZ))
    cr = int(round(CODE_RATE_HZ))
    spp = Fraction(sr * CODE_LEN, cr)              # samples per code period, exact
    n_periods = spp.denominator
    n_samples = spp.numerator

    code = np.asarray(ca_code(prn), dtype=np.float32)
    bipolar = 1.0 - 2.0 * code                     # 0 → +1, 1 → −1
    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr // sr) % CODE_LEN           # exact zero-order-hold chip mapping
    return bipolar[chip_idx].astype(np.complex64), n_samples, n_periods


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. `fc_hz` is the −6 dB
    cutoff; `trans_hz` sets the tap count (steeper skirt → more taps). Returns (h, n_taps)."""
    import numpy as np
    m = int(np.ceil(5.5 * SAMP_RATE_HZ / max(trans_hz, 1.0))) | 1     # odd
    m = min(m, (max_taps | 1))
    k = np.arange(m)
    c = (m - 1) / 2.0
    fcn = fc_hz / SAMP_RATE_HZ
    h = 2 * fcn * np.sinc(2 * fcn * (k - c))
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n1)
           + 0.14128 * np.cos(4 * np.pi * k / n1) - 0.01168 * np.cos(6 * np.pi * k / n1))
    h = h * win
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, sidelobes: int, trans_hz: float, base_fft=None):
    """Circularly filter the looped C/A buffer to keep the main lobe + `sidelobes` sidelobes.
    Circular convolution (multiply the buffer's DFT by the filter's) keeps the result exactly
    periodic, so the filtered loop has no seam; unity passband gain leaves the kept lobes'
    power unchanged. Pass `base_fft` (= np.fft.fft(base_iq)) to reuse it across live filter
    changes — the base loop is fixed per run, so its DFT need only be computed once, which
    cuts the per-change CPU spike (and the underflows it can cause). Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * CA_NULL_HZ          # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                         # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    if base_fft is None:
        base_fft = np.fft.fft(base_iq)
    filtered = np.fft.ifft(base_fft * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test (Gold code always; filter check when numpy is present) ────────────────

def _self_test() -> int:
    ok = True
    for prn in range(1, 33):
        code = ca_code(prn)
        first10 = 0
        for b in code[:10]:
            first10 = (first10 << 1) | b
        good = (len(code) == CODE_LEN and first10 == _FIRST10_OCTAL[prn] and sum(code) == 512)
        ok = ok and good
        print(f"PRN {prn:2d}: first10={first10:#06o} expect={_FIRST10_OCTAL[prn]:#06o} "
              f"ones={sum(code)} [{'OK' if good else 'FAIL'}]")
    print("Gold code: ALL PRN CHECKS PASSED" if ok else "Gold code: SELF-TEST FAILED")

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n, _ = build_iq_buffer(1)

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=2, trans_hz=0.5e6)
    main = 10 * np.log10(band(filt, 0, CA_NULL_HZ) / band(base, 0, CA_NULL_HZ))
    kept = 10 * np.log10(band(filt, 2 * CA_NULL_HZ, 3 * CA_NULL_HZ)
                         / band(base, 2 * CA_NULL_HZ, 3 * CA_NULL_HZ))
    cut = 10 * np.log10(band(filt, 10e6, 12e6) / band(base, 10e6, 12e6))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+2 sidelobes, {taps} taps): main lobe {main:+.3f} dB, kept sidelobe "
          f"{kept:+.3f} dB, far sidelobe {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} "
          f"[{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float, amplitude: float):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack.

    The looped baseband buffer is streamed from RAM by LoopSource, NOT a file_source:
    a file_source returns -1 ("done") on any read hiccup, which permanently kills the
    source (the flowgraph keeps running but transmits nothing until a restart). An
    in-RAM loop can't hit that, and a live filter change swaps the buffer atomically
    with no file I/O."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    class LoopSource(gr.sync_block):
        """Seamlessly repeats a complex64 buffer forever; supports an atomic swap to a
        new buffer (used by the live filter). No files → no fread error → the source
        can never silently die. work() copies with NumPy slices (C-speed memcpy) and
        allocates nothing in steady state, so it sustains the sample rate cleanly."""

        def __init__(self, iq):
            gr.sync_block.__init__(self, name="loop_source", in_sig=[],
                                   out_sig=[np.complex64])
            self._lock = threading.Lock()
            self._buf = np.ascontiguousarray(iq, dtype=np.complex64)
            self._pos = 0

        def swap(self, iq):
            buf = np.ascontiguousarray(iq, dtype=np.complex64)
            with self._lock:
                self._buf = buf
                if self._pos >= len(buf):
                    self._pos = 0

        def work(self, input_items, output_items):
            out = output_items[0]
            n = len(out)
            with self._lock:
                buf = self._buf
                length = len(buf)
                pos = self._pos
                filled = 0
                while filled < n:
                    take = min(n - filled, length - pos)
                    out[filled:filled + take] = buf[pos:pos + take]
                    filled += take
                    pos += take
                    if pos == length:
                        pos = 0
                self._pos = pos
            return n

    class PrnTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = LoopSource(initial_iq)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def swap(self, iq):
            self.src.swap(iq)                        # atomic in-RAM buffer swap

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return PrnTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} (C/A Gold code) transmitter — fixed 61.38 MHz / sc8, looped "
               "buffer, optional power-preserving digital passband filter. Level is set in "
               "dBm via the unit's calibration; uncalibrated it runs on a relative gain.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=CARRIER_HZ / 1e6,
                help="RF carrier in MHz (default L1 = 1575.42). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .integer("-PRN", "--prn", min=1, max=32, default=1, required=True,
                 help="GPS satellite PRN / Gold code index (1..32). Fixed per run.")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain, so "
                     "it preserves what it passes). Live.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, default=2,
                 presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of C/A sidelobes KEPT beside the main "
                      "lobe: a ±(n+1)·1.023 MHz band. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.5,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loop).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

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

    # Prebuild the unfiltered loop once (PRN is fixed per run); the filter derives from it.
    base_iq, nsamp, nper = build_iq_buffer(args.prn)

    # Filter "shape" (the regeneration-requiring params) — mutated by live changes.
    shape = {"on": getattr(args, "filter", "off") == "on",
             "sidelobes": int(getattr(args, "sidelobes", 2) or 0),
             "trans_hz": float(getattr(args, "transition", 0.5) or 0.5) * 1e6}

    base_fft = {"v": None}      # DFT of the fixed base loop — computed once, reused per change

    def make_current(report=False):
        """The buffer for the current shape: the base loop, or the circularly-filtered loop.
        Returns (iq, info) where info describes the filter for the banner/report."""
        if not shape["on"]:
            return base_iq, {"on": False}
        if base_fft["v"] is None:
            import numpy as np
            base_fft["v"] = np.fft.fft(base_iq)
        filtered, taps, fp = filter_buffer(base_iq, shape["sidelobes"], shape["trans_hz"],
                                           base_fft=base_fft["v"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}

    iq0, finfo = make_current()

    tb = _build_top_block(initial_iq=iq0, center_freq_hz=center_freq_hz,
                          gain_db=gain_db, amplitude=amplitude)

    def regenerate():
        """Rebuild the loop for the current filter shape and swap it in atomically (one
        seam, then it loops clean). Runs on the control thread; the flowgraph keeps
        streaming the old buffer until the swap. In-RAM — no file, so the source can
        never be left dead by a read error."""
        iq, info = make_current()
        tb.swap(iq)
        return info

    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    def _fmt_band(info):
        if not info.get("on"):
            return "off (full signal)"
        return (f"on — main + {info['sidelobes']} sidelobe(s) "
                f"(±{info['edge_hz']/1e6:.2f} MHz), {info['trans_hz']/1e6:g} MHz transition, "
                f"{info['taps']} taps")

    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  PRN            : {args.prn}")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  code rate      : 1.023 Mcps (~2.046 MHz null-to-null), loop {nsamp} samples")
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

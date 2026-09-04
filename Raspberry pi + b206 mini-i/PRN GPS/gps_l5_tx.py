#!/usr/bin/env python3
"""
GPS L5 transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a **bit-exact** GPS **L5** signal (1176.45 MHz): the QPSK of two 10.23 Mcps
BPSK channels — L5I ("data", 10230-chip primary × NH10, 10 ms) and L5Q (pilot,
primary × NH20, 20 ms). Prebuilt once and looped so a Raspberry Pi sustains the
40+ MS/s an L5 signal needs, with no runtime IQ math (same recipe as gps_l1ca_tx.py).

⚠  RF SAFETY / LEGAL: L5 (1176.45 MHz) is a live GNSS (aeronautical safety-of-life)
   band. Transmit ONLY into a shielded / conducted setup (cable + attenuators) you are
   LICENSED / AUTHORISED to use — never radiate over the air.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 6 samples/chip at 10.23 Mcps, exact), master clock 1:1;
  • over-the-wire sc8 (constant-modulus QPSK loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.
61.38 MHz (± 30.69 MHz) captures the ±10.23 MHz main lobe plus its first two sidelobes,
which the digital filter below can then shape.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid (the SDR
gain step and any active-component steps), and the banner reports the power actually
achieved. --gain instead commands the raw SDR gain (relative), overriding --power.
Uncalibrated, there is no dBm scale — use --gain. (See docs/calibration-v2.md.)

Digital passband filter (on the looped buffer — no runtime DSP)
──────────────────────────────────────────────────────────────
An optional steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power: if the main lobe measures
−2.5 dBm unfiltered it still reads −2.5 dBm filtered — the filter only removes what's outside
the passband (it lowers the main lobe only if the passband is narrow enough to cut into it).
L5's channels are BPSK-R(10) (sinc² nulls every 10.23 MHz), so the passband is set by how
many sidelobes to keep.
  • --filter on/off             enable/disable (live);
  • --sidelobes <n>             passband keeps the main lobe + n sidelobes, i.e. a
                                ±(n+1)·10.23 MHz band (live, presets by sidelobe count);
  • --transition <MHz>          skirt steepness — the transition width beyond the passband
                                edge (live).

CLI
───
    gps_l5_tx.py --prn 5 --power -30                          # calibrated dBm, no filter
    gps_l5_tx.py --prn 5 --channel Q --gain 60 --filter on    # pilot only, main+1 sidelobe
    gps_l5_tx.py --self-test
    gps_l5_tx.py --describe-params
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

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l5"

# Which parameter carries the transmit frequency, so a frequency-dependent calibration chain
# folds --power at the live carrier (see the C/A scripts / docs/calibration-v2.md).
CAL_FREQ_PARAM = "freq"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 6 samples/chip at 10.23 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; QPSK is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L5) ───────────────────────────────────────
L5_HZ = 1176.45e6                # GPS L5
CODE_RATE_HZ = 10_230_000        # 10.23 Mcps
L5_CODE_LEN = 10230              # chips in an L5 primary code (1 ms period)
SIGNAL_NAME = "GPS L5"
L5_NULL_HZ = 10.23e6             # main-lobe null spacing == the chip rate; sidelobes step by this

# Neuman-Hofman secondary codes (bit patterns), IS-GPS-705 §3.2.1.4.
NH10 = [0, 0, 0, 0, 1, 1, 0, 1, 0, 1]                                  # L5I, 10 ms
NH20 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0]    # L5Q, 20 ms

FREQUENCIES = {"GPS L5 (1176.45 MHz)": L5_HZ / 1e6}   # presets are in MHz now

# Filter presets: {label: number of sidelobes to KEEP}. The passband is the main lobe plus
# that many sidelobes, i.e. a ±(n+1)·10.23 MHz band. At 61.38 MHz (±30.69) n=2 is the whole
# signal, so meaningful filtering is main-only (n=0) or main+1 (n=1).
MAX_SIDELOBES = 2
SIDELOBE_PRESETS = {
    "Main lobe only (±10.23 MHz)": 0,
    "Main + 1 sidelobe (±20.46 MHz)": 1,
    "Main + 2 sidelobes (±30.69 MHz, ≈ full)": 2,
}

# Per-PRN XB code advance (chips), IS-GPS-705 — index = PRN-1, PRN 1..63.
# Cross-validated against pmonta/GNSS-DSP-tools and taroz/GNSS-SDRLIB (identical).
L5I_XB_ADVANCE = (
    266, 365, 804, 1138, 1509, 1559, 1756, 2084, 2170, 2303,
    2527, 2687, 2930, 3471, 3940, 4132, 4332, 4924, 5343, 5443,
    5641, 5816, 5898, 5918, 5955, 6243, 6345, 6477, 6518, 6875,
    7168, 7187, 7329, 7577, 7720, 7777, 8057,
    5358, 3550, 3412, 819, 4608, 3698, 962, 3001, 4441, 4937,
    3717, 4730, 7291, 2279, 7613, 5723, 7030, 1475, 2593, 2904,
    2056, 2757, 3756, 6205, 5053, 6437,
)
L5Q_XB_ADVANCE = (
    1701, 323, 5292, 2020, 5429, 7136, 1041, 5947, 4315, 148,
    535, 1939, 5206, 5910, 3595, 5135, 6082, 6990, 3546, 1523,
    4548, 4484, 1893, 3961, 7106, 5299, 4660, 276, 4389, 3783,
    1591, 1601, 749, 1387, 1661, 3210, 708,
    4226, 5604, 6375, 3056, 1772, 3662, 4401, 5218, 2838, 6913,
    1685, 1194, 6963, 5001, 6694, 991, 7489, 2441, 639, 2097,
    2498, 6470, 2399, 242, 3768, 1186,
)

# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# GPS L5 is a BPSK(10)-shaped signal (L5I + L5Q, both 10.23 Mcps): a sinc² power spectrum whose
# whole distribution is fixed by ONE measured number — the power spectral DENSITY at the main-lobe
# PEAK (the carrier), in dBm/Hz (per Hz, NOT per MHz). From that single number CAL_POWER_LAWS
# derives two absolute-power quantities the operator can pick for --power (in the calibration
# editor); the measured density itself (dBm/Hz) stays available as a third:
#
#   • Main-lobe integrated power (dBm)   = peak_dBm/Hz + 10·log10(Rc · I_ML)   ← ±10.23 MHz
#   • Carrier (total signal) power (dBm) = peak_dBm/Hz + 10·log10(Rc)          ← all frequency
#
# Rc = 10.23e6 Hz (chip rate); I_ML = 0.902823 is the sinc² power fraction inside the main lobe
# (±Rc). Both quantities are BANDWIDTH-INDEPENDENT constants, so --power is stable across the
# optional filter: narrowing the passband only lowers the EMITTED power, never above the carrier
# (total) reading — which makes the carrier power the safe amplifier-limiting quantity whatever
# the filter passes. (--self-test recomputes both constants from ∫sinc² and asserts these literals.)
I_ML = 0.902823                              # sinc² power fraction within the main lobe (±Rc)

CAL_POWER_LAWS = [
    {"id": "main_lobe_power", "name": "Main-lobe integrated power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 69.654784},    # 10·log10(Rc · I_ML), Rc = 10.23e6
    {"id": "carrier_power", "name": "Carrier (total signal) power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.098756},     # 10·log10(Rc), Rc = 10.23e6
]

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present (SDR_CALIBRATION_FILE),
    else uncalibrated (relative gain only). Cached so build_script and main agree — and so
    --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── L5 primary code (bit-exact XA ⊕ XB, IS-GPS-705) ────────────────────────────
# XA: x^13 + x^12 + x^10 + x^9 + 1, short-cycled to 8190 chips (reset one chip
# early). XB: x^13 + x^12 + x^8 + x^7 + x^6 + x^4 + x^3 + x + 1, maximal (8191).
# Each PRN's code is XA ⊕ (XB advanced by the per-PRN amount).

_XA: list | None = None
_XB: list | None = None


def _build_registers() -> None:
    global _XA, _XB
    if _XA is not None:
        return
    xa = [1] * 13
    ya = []
    for _ in range(L5_CODE_LEN):
        ya.append(xa[12])
        if xa == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]:   # short cycle → reset
            xa = [1] * 13
        else:
            xa = [xa[12] ^ xa[11] ^ xa[9] ^ xa[8]] + xa[0:12]
    xb = [1] * 13
    yb = []
    for _ in range(8191):
        yb.append(xb[12])
        xb = [xb[12] ^ xb[11] ^ xb[7] ^ xb[6] ^ xb[5] ^ xb[3] ^ xb[2] ^ xb[0]] + xb[0:12]
    _XA, _XB = ya, yb


def l5_primary(prn: int, channel: str) -> list[int]:
    """One 10230-chip L5 primary code (0/1) for a PRN (1..63) and channel
    ('I' or 'Q') — the real IS-GPS-705 code for that satellite."""
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    _build_registers()
    adv = (L5I_XB_ADVANCE if channel == "I" else L5Q_XB_ADVANCE)[prn - 1]
    return [_XA[i] ^ _XB[(adv + i) % 8191] for i in range(L5_CODE_LEN)]


# ── Baseband buffer (one seamless-looping 20 ms L5 period) ─────────────────────

def build_l5_buffer(prn: int, channel: str):
    """The complex64 L5 buffer at SAMP_RATE_HZ over one full 20 ms NH period (loops
    seamlessly). Constant-modulus QPSK; amplitude applied live downstream. Returns
    (iq, n_samples)."""
    import numpy as np

    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(0.020 * sr))          # 20 ms — full NH20 period

    bi = 1.0 - 2.0 * np.asarray(l5_primary(prn, "I"), dtype=np.float32)
    bq = 1.0 - 2.0 * np.asarray(l5_primary(prn, "Q"), dtype=np.float32)
    nh10 = 1.0 - 2.0 * np.asarray(NH10, dtype=np.float32)
    nh20 = 1.0 - 2.0 * np.asarray(NH20, dtype=np.float32)

    n = np.arange(n_samples, dtype=np.int64)
    gchip = n * CODE_RATE_HZ // sr               # 0 .. 204599
    ms_idx = gchip // L5_CODE_LEN                 # 0 .. 19
    chip = gchip % L5_CODE_LEN                    # 0 .. 10229

    i_val = bi[chip] * nh10[ms_idx % 10]
    q_val = bq[chip] * nh20[ms_idx % 20]

    if channel == "I":
        iq = i_val.astype(np.complex64)
    elif channel == "Q":
        iq = q_val.astype(np.complex64)
    else:  # IQ — equal-power QPSK, constant modulus
        iq = ((i_val + 1j * q_val) / np.sqrt(2.0)).astype(np.complex64)
    return iq, n_samples


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain. `fc_hz` is the −6 dB
    cutoff; `trans_hz` sets the tap count (steeper skirt → more taps). Returns (h, n_taps)."""
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


def filter_buffer(base_iq, sidelobes: int, trans_hz: float, base_fft=None):
    """Circularly filter the looped L5 buffer to keep the main lobe + `sidelobes` sidelobes.
    Circular convolution (multiply the buffer's DFT by the filter's) keeps the result exactly
    periodic, so the filtered loop has no seam; unity passband gain leaves the kept lobes'
    power unchanged. Pass `base_fft` (= np.fft.fft(base_iq)) to reuse it across live filter
    changes — the base loop is fixed per run, so its DFT need only be computed once, which
    cuts the per-change CPU spike (and the underflows it can cause). Returns
    (filtered_iq, n_taps, passband_edge_hz)."""
    import numpy as np
    fp = (int(sidelobes) + 1) * L5_NULL_HZ           # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                          # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    if base_fft is None:
        base_fft = np.fft.fft(base_iq)
    filtered = np.fft.ifft(base_fft * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Self-test (real-code validation + sizing + filter; pure Python + numpy) ─────

def _self_test() -> int:
    from fractions import Fraction
    # first-24-chips (octal) reference values — a transcription guard proving the
    # XB-advance tables produce the real IS-GPS-705 codes.
    check24 = {
        ("I", 1): 0o66124275, ("I", 2): 0o24763202,
        ("I", 10): 0o41006103, ("I", 32): 0o30576255,
        ("Q", 1): 0o63131310, ("Q", 2): 0o44165373,
        ("Q", 10): 0o47557674, ("Q", 32): 0o52731266,
    }
    ok = True

    _build_registers()
    xb_ones = sum(_XB)
    print(f"XB maximal m-sequence: len={len(_XB)} ones={xb_ones} "
          f"(expect 8191/4096) [{'OK' if len(_XB) == 8191 and xb_ones == 4096 else 'FAIL'}]")
    ok = ok and len(_XB) == 8191 and xb_ones == 4096

    for ch in ("I", "Q"):
        # Length exact, near-balanced (a 10230 code sits within ~±60 of 5115), distinct.
        bal = all(len(l5_primary(p, ch)) == L5_CODE_LEN
                  and abs(sum(l5_primary(p, ch)) - 5115) < 100 for p in range(1, 64))
        distinct = len({tuple(l5_primary(p, ch)) for p in range(1, 64)}) == 63
        print(f"{ch}: PRN 1..63 length/balance ok={bal}, distinct={distinct}")
        ok = ok and bal and distinct
        for (cch, prn), want in check24.items():
            if cch != ch:
                continue
            v = 0
            for b in l5_primary(prn, ch)[:24]:
                v = (v << 1) | b
            good = v == want
            ok = ok and good
            print(f"{ch} PRN{prn:2d}: first24={oct(v)} expect={oct(want)} "
                  f"[{'OK' if good else 'FAIL'}]")

    nh_ok = len(NH10) == 10 and len(NH20) == 20
    print(f"NH10 len={len(NH10)}, NH20 len={len(NH20)} [{'OK' if nh_ok else 'FAIL'}]")
    ok = ok and nh_ok

    sr = int(round(SAMP_RATE_HZ))
    n = 0.020 * sr
    chips = Fraction(int(round(n)) * CODE_RATE_HZ, sr)
    size_ok = n == int(n) and chips == 204600
    print(f"{SAMP_RATE_HZ/1e6:g} MHz → {n:.0f} samples/20ms (int={n == int(n)}), "
          f"chips={chips} (=204600), samples/chip={sr / CODE_RATE_HZ:.4f} [{'OK' if size_ok else 'FAIL'}]")
    ok = ok and size_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, _ = build_l5_buffer(1, "IQ")

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=1, trans_hz=1.0e6)
    main = 10 * np.log10(band(filt, 0, L5_NULL_HZ) / band(base, 0, L5_NULL_HZ))
    kept = 10 * np.log10(band(filt, L5_NULL_HZ, 2 * L5_NULL_HZ)
                         / band(base, L5_NULL_HZ, 2 * L5_NULL_HZ))
    cut = 10 * np.log10(band(filt, 24e6, 30e6) / max(band(base, 24e6, 30e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and abs(kept) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (main+1 sidelobe, {taps} taps): main lobe {main:+.3f} dB, kept sidelobe "
          f"{kept:+.3f} dB, far sidelobe {cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} "
          f"[{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    # Calibration law constants: recompute I_ML from ∫sinc² and assert the CAL_POWER_LAWS literals
    # (main-lobe / carrier) — so the baked constants can never silently drift from the spectrum.
    def _sinc2(x):
        if x == 0.0:
            return 1.0
        s = math.sin(math.pi * x) / (math.pi * x)
        return s * s
    step, acc, prev = 1e-3, 0.0, _sinc2(0.0)
    for i in range(1, int(round(1.0 / step)) + 1):
        cur = _sinc2(i * step); acc += 0.5 * (prev + cur) * step; prev = cur
    i_ml = 2.0 * acc                                    # sinc² power fraction within ±Rc
    Rc = CODE_RATE_HZ
    main_k = 10 * math.log10(Rc * i_ml)
    carrier_k = 10 * math.log10(Rc)
    laws = {l["id"]: l["k"] for l in CAL_POWER_LAWS}
    laws_ok = (abs(i_ml - I_ML) < 5e-4
               and abs(laws["main_lobe_power"] - main_k) < 5e-3
               and abs(laws["carrier_power"] - carrier_k) < 5e-3)
    print(f"calibration: I_ML={i_ml:.6f}, main-lobe k={main_k:.4f} (law {laws['main_lobe_power']}), "
          f"carrier k={carrier_k:.4f} (law {laws['carrier_power']}) "
          f"[{'OK' if laws_ok else 'FAIL'}]")
    ok = ok and laws_ok
    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(initial_iq, center_freq_hz: float, gain_db: float,
                     amplitude: float):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack.

    The looped baseband buffer is streamed from RAM by a C++ blocks.vector_source_c
    (repeat=True) — NOT a file_source, and NOT a Python source:
      • file_source streams GIL-free and holds the rate, but swapping it live (open()) races
        GNU Radio internally and THROWS "fread error", which kills the source and silences the
        radio. With no file that is impossible.
      • a Python gr.sync_block has no file either, but its work() runs through the Python
        gateway under the GIL and can't sustain 61.38 Msps on a Pi — it underflows even in
        steady state (no tuning).
    vector_source_c is C++ (GIL-free, no per-sample Python) AND has no file, so it streams as
    smoothly as file_source did, with none of the fread risk. Its one caveat: set_data() is not
    itself locked against work(), so the live filter swap runs it under top-block lock()/
    unlock(), which quiesces the flowgraph for the (brief) duration of the swap — the buffer is
    never freed under a running read. That pause happens only on a filter change; the steady
    stream is untouched."""
    import numpy as np
    from gnuradio import gr, blocks, uhd

    def _vec(iq):
        # vector_source_c wants a contiguous complex64 buffer; the filtered loop may not be.
        return np.ascontiguousarray(iq, dtype=np.complex64)

    class L5Tx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            # repeat=True loops the buffer in C++ (no per-wrap Python), vlen=1, no tags.
            self.src = blocks.vector_source_c(_vec(initial_iq), True, 1, [])
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def swap(self, iq):
            # set_data() reassigns (and frees) the source's buffer with no internal lock, so it
            # must not run while work() is reading. lock()/unlock() quiesces the flowgraph
            # around the swap — the only moment the stream pauses, and only on a filter change.
            data = _vec(iq)
            self.lock()
            try:
                self.src.set_data(data, [])
            finally:
                self.unlock()

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return L5Tx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (QPSK: L5I data + L5Q pilot, real IS-GPS-705 codes, "
               "10.23 Mcps, NH secondary codes) — fixed 61.38 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=L5_HZ / 1e6,
                help="RF carrier in MHz (default L5 = 1176.45). Fixed per run.")
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
                 help="GPS satellite PRN (1..63) — the real L5 code. Fixed per run.")
        .choice("-Channel", "--channel", options=["IQ", "I", "Q"], default="IQ",
                help="IQ = full L5 (QPSK); I = data channel only; Q = pilot only.")
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

    # Prebuild the unfiltered loop once (PRN/channel are fixed per run); the filter derives from it.
    base_iq, nsamp = build_l5_buffer(args.prn, args.channel)
    base_fft = {"v": None}      # DFT of the fixed base loop — computed once, reused per change

    # Filter "shape" (the regeneration-requiring params) — mutated by live changes.
    shape = {"on": getattr(args, "filter", "off") == "on",
             "sidelobes": int(getattr(args, "sidelobes", 1) or 0),
             "trans_hz": float(getattr(args, "transition", 1.0) or 1.0) * 1e6}

    def make_current():
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
        """Rebuild the loop for the current filter shape and swap it in atomically (one seam,
        then it loops clean). Runs on the control thread; the flowgraph keeps streaming the old
        buffer until the swap. In-RAM — no file, so the source can never be left dead by a
        read error."""
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
    print(f"  satellite PRN  : {args.prn}  (channel {args.channel}, real L5 code)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : QPSK-R(10) — 10.23 Mcps, ±10.23 MHz main lobe")
    print(f"  buffer         : {nsamp} samples (20 ms — full NH period, {nsamp*8/1e6:.1f} MB)")
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

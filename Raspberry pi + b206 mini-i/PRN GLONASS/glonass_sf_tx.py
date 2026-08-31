#!/usr/bin/env python3
"""
GLONASS L1SF / L2SF (FDMA high-accuracy "P-code") transmitter for GNU Radio+UHD.

Purpose
───────
Transmit the legacy GLONASS high-accuracy ranging signal — L1SF (~1602 MHz) or
L2SF (~1246 MHz) — in either of two modes:

  • CHANNEL: one satellite on its FDMA frequency channel k (−7…+6). BPSK of the
    P-code at 5.11 Mcps, carrier tuned to that channel.
  • BAND:    the whole FDMA band at once — all 14 channels summed at their
    frequency offsets around the band centre, as a wideband receiver sees them.

Is this reproducible? Yes — and here is the important distinction
────────────────────────────────────────────────────────────────
The GLONASS P-code is officially UNDOCUMENTED, but it is NOT encrypted. It was
publicly reverse-engineered in 1989 (Lennen, ION GPS-89) and civilian dual-
frequency survey receivers have tracked it on L2 ever since. So — unlike GPS
P(Y), whose W-code encryption forces a spectral surrogate — the GLONASS P-code
can be reproduced BIT-EXACT from its public definition. This script uses that
public definition; it contains no encrypted or classified content.

⚠  RF SAFETY / LEGAL: L1 (~1602 MHz) and L2 (~1246 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup (cable + attenuators into a
   receiver or spectrum analyser) that you are LICENSED / AUTHORISED to use.
   Radiating a GNSS code over the air can jam or spoof real receivers and is
   illegal in most places.

FDMA, shared code (same as L1OF/L2OF)
─────────────────────────────────────
Every satellite transmits the SAME P-code; satellites are told apart by carrier
frequency, so the per-satellite selector is the channel number k:

    L1SF:  f_k = 1602 MHz + k · 0.5625 MHz      (k = −7 … +6)
    L2SF:  f_k = 1246 MHz + k · 0.4375 MHz      (k = −7 … +6)

P-code
──────
A 25-stage LFSR, generating polynomial G(x) = 1 + x³ + x²⁵ (feedback from stages
3 and 25), output tapped at stage 25, register seeded all-ones. The full
m-sequence is 2²⁵−1 = 33 554 431 chips, but the P-code is TRUNCATED to
5 110 000 chips — exactly 1 s at 5.11 Mcps — and the register resets to all-ones
each second. --self-test verifies (via a GF(2) order test) that the polynomial
is primitive, so the sequence is a true maximal m-sequence, and that the 1 s
truncation is balanced (~50 %). No navigation data is modulated (clean BPSK
ranging spectrum, ±5.11 MHz main lobe).

BAND-mode note
──────────────
At 5.11 Mcps each channel's main lobe (±5.11 MHz) is far wider than the 0.5625
MHz channel spacing, so the 14 channels overlap almost completely and the
composite is a nearly-flat ~17 MHz-wide block spanning the band (you cannot pick
out individual channels — that is what the aggregate P-code band actually looks
like). Each channel is given a distinct code phase so the channels are not
mutually coherent.

Buffer size
───────────
The 1 s period makes this a large replay buffer. CHANNEL mode at the default
10.22 MHz (2 samp/chip) is 10.22 M samples ≈ 82 MB. BAND mode needs ≥ ~18 MHz to
hold the whole span, so its default is 20.44 MHz (4 samp/chip) ≈ 164 MB — still a
single 1 s buffer, not 14. Generation takes a few seconds at startup, once; then
the flowgraph only DMAs bytes.

Streaming levers (same as the other builders)
─────────────────────────────────────────────
PRECOMPUTE+LOOP · sc8 over the wire · silent after start() · master_clock_rate
pinned == sample rate (1:1). Live tuning (paramkit.live): gain, amplitude.

CLI
───
    glonass_sf_tx.py --band L1 --mode channel --channel 0
    glonass_sf_tx.py --band L1 --mode band
    glonass_sf_tx.py --self-test        # verify the P-code generator, no hardware
    glonass_sf_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# Stable calibration signal id (see the agent's docs/calibration.md). A task sets
# SDR_CAL_SIGNAL_ID to this and the agent injects this unit's resolved calibration
# (SDR_CALIBRATION_FILE); calkit reads it so --power maps through the unit's MEASURED
# curve at its real operating plane. Absent it, the baked constants below are used.
CAL_SIGNAL_ID = "glonass_sf"


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

HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling


_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script and
    main share one and --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) for a requested delivered power, through the active calibration."""
    return power_map().gain_for_power(float(delivered_dbm))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) an actual hardware gain produces, through the active map."""
    return power_map().power_for_gain(float(gain_db))


# ── Constants ─────────────────────────────────────────────────────────────────

CHIP_RATE_HZ = 5.11e6          # GLONASS P-code chip rate (10 × C/A)
CODE_LEN = 5_110_000           # truncated to 1 s (reset each second)
LFSR_DEG = 25                  # 25-stage register, G(x)=1+x^3+x^25
K_MIN, K_MAX = -7, 6

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
# GLONASS's 0.511 MHz FDMA base caps the rate at 61.32 MHz (= 12×5.11 = 120×0.511), the
# highest multiple inside the B200's range; it covers the P-code + FDMA band whole. NB the
# P-code period is 1 s, so at 61.32 MHz the loop is ~490 MB in RAM. Master clock 1:1.
SAMP_RATE_HZ = 61.32e6
OTW_FORMAT = "sc8"            # over-the-wire; halves USB load

BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6},
}

# Filter: a lowpass job either way — a single P-code channel (sinc², nulls every 5.11 MHz)
# or the FDMA comb of 14 carriers — so the passband is a direct half-bandwidth in MHz. The
# default (±8 MHz) keeps the whole FDMA band in band mode and the main lobe in channel mode.
MIN_PASSBAND_MHZ = 1.0
MAX_PASSBAND_MHZ = 30.6
PASSBAND_PRESETS = {
    "Channel main lobe (±5.11 MHz)": 5.11,
    "Full FDMA band (±8 MHz)": 8.0,
    "Wide (±12 MHz)": 12.0,
}


# ── P-code generation ──────────────────────────────────────────────────────────

def _pcode_into(buf) -> None:
    """Fill buf (length CODE_LEN) with the P-code 0/1 chips. 25-stage Fibonacci
    LFSR, taps 25 & 3, output stage 25, all-ones seed, truncated at CODE_LEN."""
    state = (1 << LFSR_DEG) - 1
    mask = (1 << LFSR_DEG) - 1
    for i in range(len(buf)):
        buf[i] = (state >> 24) & 1                  # stage 25 output
        fb = ((state >> 24) ^ (state >> 2)) & 1     # taps 25 and 3
        state = ((state << 1) | fb) & mask


def channel_freq(band: str, k: int) -> float:
    b = BANDS[band]
    return b["base"] + k * b["spacing"]


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    """Verify G(x)=1+x³+x²⁵ is primitive over GF(2) (so the P-code is a maximal
    m-sequence) via a fast order test — no 33-million-step walk — and that the
    1 s truncation is well balanced. Prints the FDMA plan."""
    POLY = (1 << 25) | (1 << 3) | (1 << 0)
    ORDER = (1 << 25) - 1                            # 33554431 = 31·601·1801
    FACTORS = (31, 601, 1801)

    def mulmod(a, b):
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> 25) & 1:
                a ^= POLY
        return r

    def powmod(base, e):
        r = 1
        while e:
            if e & 1:
                r = mulmod(r, base)
            base = mulmod(base, base)
            e >>= 1
        return r

    primitive = (powmod(2, ORDER) == 1
                 and 31 * 601 * 1801 == ORDER
                 and all(powmod(2, ORDER // p) != 1 for p in FACTORS))

    seg = bytearray(300_000)
    _pcode_into(seg)
    frac = sum(seg) / len(seg)
    bal_ok = 0.48 < frac < 0.52
    ok = primitive and bal_ok
    print(f"P-code poly 1+x^3+x^25 primitive: {primitive}  "
          f"(maximal m-seq, period {ORDER}) [{'OK' if primitive else 'FAIL'}]")
    print(f"balance (300k segment): {frac*100:.1f}% ones [{'OK' if bal_ok else 'FAIL'}]")
    print(f"period: {CODE_LEN} chips = {CODE_LEN/CHIP_RATE_HZ:.3f} s @ "
          f"{CHIP_RATE_HZ/1e6:g} Mcps")
    for band in ("L1", "L2"):
        lo, hi = channel_freq(band, K_MIN), channel_freq(band, K_MAX)
        print(f"{band}SF plan: k {K_MIN}..{K_MAX}  {lo/1e6:.4f}..{hi/1e6:.4f} MHz")

    sr_ok = int(round(SAMP_RATE_HZ)) % int(round(CHIP_RATE_HZ)) == 0
    print(f"fixed rate {SAMP_RATE_HZ/1e6:g} MHz = {SAMP_RATE_HZ/CHIP_RATE_HZ:.0f}×5.11 "
          f"[{'OK' if sr_ok else 'FAIL'}]")
    ok = ok and sr_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    # Validate the memory-bounded circular filter on a small synthetic loop (the real
    # 1 s buffer is ~61 M samples — too heavy for a self-test). A short BPSK-at-5.11 Mcps
    # sinc² proxy: 200 chips × 12 samp/chip, so nulls still sit at 5.11 MHz.
    spc = int(round(SAMP_RATE_HZ / CHIP_RATE_HZ))
    rng = np.random.default_rng(0)
    chips = 1 - 2 * rng.integers(0, 2, size=200).astype(np.float32)
    base = np.repeat(chips, spc).astype(np.complex64)

    def band_p(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    # Cross-check the overlap-add path against a direct circular DFT (must match).
    h, taps = _design_lowpass(8.5e6, 1.0e6, len(base) // 2)
    y_oa = _circular_convolve(base, h)
    y_fft = np.fft.ifft(np.fft.fft(base) * np.fft.fft(h, len(base))).astype(np.complex64)
    match = float(np.max(np.abs(y_oa - y_fft)))

    filt, ftaps, fp = filter_buffer(base, passband_hz=8.0e6, trans_hz=1.0e6)
    main = 10 * np.log10(band_p(filt, 0, CHIP_RATE_HZ) / band_p(base, 0, CHIP_RATE_HZ))
    peak = float(np.max(np.abs(filt)))
    f_ok = match < 1e-4 and abs(main) < 0.1 and peak * AMPLITUDE < 1.0
    print(f"filter (±8 MHz, {ftaps} taps): overlap-add vs DFT {match:.1e}, main lobe "
          f"{main:+.3f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    print("ALL GLONASS P-CODE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffers ───────────────────────────────────────────────────────────

def _validate_sr(samp_rate_hz: float) -> int:
    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    if sr % cr != 0:
        raise ValueError(f"sample rate must be an integer multiple of "
                         f"{CHIP_RATE_HZ/1e6:g} MHz; got {samp_rate_hz/1e6:g} MHz")
    return sr // cr


def build_channel_buffer():
    """One-channel real BPSK P-code buffer for one full 1 s period at the fixed
    SAMP_RATE_HZ (loops seamlessly on reset). Returns (iq, n_samples, samp_per_chip).
    Real → I, Q=0."""
    import numpy as np
    spc = _validate_sr(SAMP_RATE_HZ)
    chips = np.empty(CODE_LEN, dtype=np.int8)
    _pcode_into(chips)                               # ~2 s, once
    bipolar = (1 - 2 * chips).astype(np.float32)
    n_samples = CODE_LEN * spc
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.repeat(bipolar, spc)
    iq.imag = 0.0
    return iq, n_samples, spc


def build_band_buffer(band: str):
    """Full-band composite at the fixed SAMP_RATE_HZ: all 14 channels of the P-code summed
    at their frequency offsets (k·spacing) around the band centre, each with a distinct
    cyclic code phase. One 1 s buffer. Built in time-chunks to bound memory. Returns
    (iq, n_samples, samp_per_chip)."""
    import numpy as np
    spc = _validate_sr(SAMP_RATE_HZ)
    sr = int(round(SAMP_RATE_HZ))
    spacing = BANDS[band]["spacing"]

    chips = np.empty(CODE_LEN, dtype=np.int8)
    _pcode_into(chips)
    bipolar = (1 - 2 * chips).astype(np.float32)

    n_samples = CODE_LEN * spc                       # exactly 1 s
    ks = list(range(K_MIN, K_MAX + 1))
    shifts = [((k - K_MIN) * 401_887) % CODE_LEN for k in ks]  # distinct phases

    shift_of = dict(zip(ks, shifts))
    comp = np.empty(n_samples, dtype=np.complex64)
    CHUNK = 2_000_000
    for start in range(0, n_samples, CHUNK):
        end = min(start + CHUNK, n_samples)
        idx = np.arange(start, end, dtype=np.int64)
        chip = idx // spc                            # exact (sr = spc·chip_rate)
        # Channel tones are harmonics of the spacing, so build them by a running
        # phasor (one trig call per chunk) instead of a cos/sin per channel.
        ph = (2.0 * np.pi * spacing / sr) * idx
        base = (np.cos(ph) + 1j * np.sin(ph)).astype(np.complex64)  # k = +1 tone
        cbase = base.conj()                                         # k = −1 tone
        acc = bipolar[(chip + shift_of[0]) % CODE_LEN].astype(np.complex64)
        zp = base.copy()
        for k in range(1, K_MAX + 1):                # k = +1 … +6
            acc += bipolar[(chip + shift_of[k]) % CODE_LEN] * zp
            if k < K_MAX:
                zp *= base
        zn = cbase.copy()
        for k in range(-1, K_MIN - 1, -1):           # k = −1 … −7
            acc += bipolar[(chip + shift_of[k]) % CODE_LEN] * zn
            if k > K_MIN:
                zn *= cbase
        comp[start:end] = acc
    peak = float(np.max(np.abs(comp))) or 1.0
    comp /= peak
    return comp, n_samples, spc


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────
# The 1 s P-code loop is ~61 M samples at 61.32 MHz, so filtering uses a memory-bounded
# overlap-add circular convolution (a single monolithic DFT would need several GB).

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


def _circular_convolve(x, h):
    """Circular convolution of period len(x) between complex `x` and real FIR `h` (len ≤ len(x))
    by OVERLAP-ADD, aliasing the (M−1)-sample linear tail back to the head — which is what makes
    the result circular, so the filtered loop stays seam-free. Peak memory is one complex64 copy
    of the loop plus O(block), not a multi-GB monolithic DFT of the ~61 M-sample buffer."""
    import numpy as np
    n = len(x)
    m = len(h)
    if m >= n:
        return np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, n)).astype(np.complex64)
    nfft = 1
    while nfft < 4 * m:
        nfft <<= 1
    step = nfft - (m - 1)
    hf = np.fft.fft(h, nfft)
    y = np.zeros(n, dtype=np.complex64)
    for start in range(0, n, step):
        blk = x[start:start + step]
        yb = np.fft.ifft(np.fft.fft(blk, nfft) * hf)
        yb = yb[:len(blk) + m - 1]
        end = start + len(yb)
        if end <= n:
            y[start:end] += yb
        else:
            first = n - start
            y[start:n] += yb[:first]
            y[0:len(yb) - first] += yb[first:]
    return y


def filter_buffer(base_iq, passband_hz: float, trans_hz: float):
    """Circularly filter the looped GLONASS P-code buffer to a ±`passband_hz` band. The
    circular convolution keeps the result exactly periodic (seam-free loop); unity passband
    gain leaves the kept content's power unchanged. Returns (filtered_iq, n_taps,
    passband_edge_hz)."""
    fp = float(passband_hz)
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    return _circular_convolve(base_iq, h), m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class GlonassSfTx(gr.top_block):
        def __init__(self):
            super().__init__("GLONASS SF TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT,
                                      channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_path, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_amplitude(self, a): self.amp.set_k(a)
        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def swap_file(self, path): self.src.open(path, True)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return GlonassSfTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1SF/L2SF (FDMA high-accuracy P-code) transmitter — one channel or "
               "the whole band, real 5.11 Mcps P-code — fixed 61.32 MHz / sc8, looped 1 s "
               "buffer (~490 MB in RAM), optional power-preserving digital passband filter. "
               "Level is set in dBm via the unit's calibration; uncalibrated it runs on a "
               "relative gain. Authorised, shielded setups only.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1SF (~1602 MHz) or L2SF (~1246 MHz) — sets the carrier band. "
                     "Fixed per run.")
        .choice("-Mode", "--mode", options=["channel", "band"], default="channel",
                help="channel = one satellite on its FDMA carrier; band = all 14 channels "
                     "summed around the band centre. Fixed per run.")
        .integer("-Channel", "--channel", min=K_MIN, max=K_MAX, default=0,
                 help="FDMA channel number k (−7..+6) — sets the carrier in channel mode "
                      "(base + k·spacing); ignored in band mode. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain, so "
                     "it preserves what it passes; memory-bounded, but re-filtering the 1 s "
                     "loop takes a few seconds). Live.")
        .number("-Passband", "--passband", unit="MHz",
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=8.0,
                presets=PASSBAND_PRESETS, required=False, live=True,
                help="Passband half-bandwidth kept each side of the carrier (MHz). Default 8 "
                     "keeps the whole FDMA band (band mode) or the P-code main lobe (channel "
                     "mode). Live (rebuilds the filtered loop).")
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
    band, mode = args.band, args.mode
    center_freq_hz = BANDS[band]["base"] if mode == "band" else channel_freq(band, args.channel)

    pmap = power_map()
    amplitude = pmap.amplitude

    # Gain precedence: explicit --gain (raw) > calibrated --power (folded at the carrier) > refuse.
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

    print("[building] generating 1 s P-code buffer…", flush=True)
    if mode == "band":
        base_iq, nsamp, spc = build_band_buffer(band)
        desc = f"{band}SF band composite (k {K_MIN}..{K_MAX}, 14 channels)"
    else:
        base_iq, nsamp, spc = build_channel_buffer()
        desc = f"{band}SF channel k={args.channel}"

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="glonass_sf_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "passband_hz": float(getattr(args, "passband", 8.0) or 8.0) * 1e6,
             "trans_hz": float(getattr(args, "transition", 1.0) or 1.0) * 1e6}

    def make_current():
        if not shape["on"]:
            return base_iq, {"on": False}
        filtered, taps, fp = filter_buffer(base_iq, shape["passband_hz"], shape["trans_hz"])
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

    print("── GLONASS SF (P-code) TX ──────────────────────────────────")
    print(f"  signal         : {desc}")
    print(f"  carrier        : {center_freq_hz/1e6:.4f} MHz"
          + ("  (band centre)" if mode == "band" else f"  (channel {args.channel})"))
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  code           : 5.11 Mcps P-code, 1 s period (BPSK, ±5.11 MHz lobe)")
    print(f"  buffer         : {nsamp} samples ({spc} samp/chip, {nsamp*8/1e6:.0f} MB)")
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
        elif name in ("filter", "passband", "transition"):
            if name == "filter":
                shape["on"] = str(value).strip().lower() in ("on", "1", "true", "yes")
            elif name == "passband":
                shape["passband_hz"] = max(MIN_PASSBAND_MHZ, min(MAX_PASSBAND_MHZ,
                                                                 float(value))) * 1e6
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

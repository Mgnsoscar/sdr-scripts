#!/usr/bin/env python3
"""
GPS L2C transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a **bit-exact** GPS **L2C** signal (1227.60 MHz): the civil signal on L2,
the chip-by-chip time-multiplex of two 511.5 kcps codes interleaved to 1.023 Mcps
(BPSK-R(1), ~2 MHz wide) — L2 CM (10230 chips, 20 ms) and L2 CL (767250 chips,
1.5 s, dataless pilot). Prebuilt once and looped so a Raspberry Pi sustains the
rate with no runtime IQ math (same recipe as gps_l1ca_tx.py).

⚠  RF SAFETY / LEGAL: L2 (1227.60 MHz) is a live GNSS band. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never over the air.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (= 60 samples/combined-chip, exact), master clock 1:1;
  • over-the-wire sc8 (constant-modulus BPSK loses nothing at 8-bit; halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
None of these are parameters; they are fixed so the loop length and calibration stay exact.
L2C is only ~2 MHz wide, but the rate is pinned high (like C/A) so the full sinc²
sidelobe skirt is present for the digital filter below to shape.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered power (dBm). A task that sets SDR_CAL_SIGNAL_ID to
CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power then maps through it
(gain_for_power), the SDR gain is snapped to the calibration's achievable grid (the SDR
gain step and any active-component steps), and the banner reports the power actually
achieved. --gain instead commands the raw SDR gain (relative), overriding --power.
Uncalibrated, there is no dBm scale — use --gain. (See docs/calibration-v2.md.)

Loop length (--loop)
────────────────────
  full (default) : one whole CL period = 1.5 s (CM repeats 75×). Bit-exact CL phase and
                 complete spectrum. ~736 MB in /dev/shm at 61.38 MHz (it must live in RAM
                 to stream at rate), so the host needs the headroom. The live filter stays
                 responsive because it uses a memory-bounded overlap-add convolution (no
                 92-Msample monolithic FFT) and the flowgraph keeps looping the old buffer
                 until the new one is ready.
  cm             : one CM period = 20 ms (CL truncated to its first 10230 chips), ~9.8 MB.
                 The BPSK-R(1) envelope is identical; the CL line structure appears at 50 Hz
                 rather than its true 0.667 Hz (both unresolvable at practical RBW). Pick it
                 when RAM is tight or the true CL phase does not matter.

Digital passband filter (on the looped buffer — no runtime DSP)
──────────────────────────────────────────────────────────────
An optional steep FIR passband, applied to the PRECOMPUTED loop by CIRCULAR convolution, so
the filtered buffer still loops with no seam and there is no per-sample runtime cost. It has
UNITY passband gain, so whatever it passes is unchanged in power: if the main lobe measures
−2.5 dBm unfiltered it still reads −2.5 dBm filtered — the filter only removes what's outside
the passband (it lowers the main lobe only if the passband is narrow enough to cut into it).
L2C's spectrum is the same sinc² as C/A (nulls every 1.023 MHz), so the passband is set by
how many sidelobes to keep.
  • --filter on/off             enable/disable (live);
  • --sidelobes <n>             passband keeps the main lobe + n sidelobes, i.e. a
                                ±(n+1)·1.023 MHz band (live, presets by sidelobe count);
  • --transition <MHz>          skirt steepness — the transition width beyond the passband
                                edge (live).

CLI
───
    gps_l2c_tx.py --prn 5 --power -30                          # calibrated dBm, no filter
    gps_l2c_tx.py --prn 5 --gain 60 --filter on --sidelobes 2  # relative gain, main+2 sidelobes
    gps_l2c_tx.py --loop full --prn 5 --power -30              # bit-exact 1.5 s CL (big/slow)
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

# Stable calibration signal id. A task setting SDR_CAL_SIGNAL_ID to this value gets this
# unit's resolved calibration injected at $SDR_CALIBRATION_FILE; calkit maps --power through
# it at the unit's real operating plane (e.g. EIRP). Absent it, the script runs uncalibrated
# (relative gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "gps_l2c"

# ── Fixed radio setup (NOT parameters — see the module docstring) ───────────────────
SAMP_RATE_HZ = 61.38e6        # 60 samples/combined-chip at 1.023 Mcps (exact); master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; BPSK is constant-modulus, 8-bit is lossless here
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other PRN scripts) ─────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Signal constants (fixed — this IS GPS L2C) ──────────────────────────────────────
L2_HZ = 1227.60e6                # GPS L2
CM_LEN = 10230                   # L2 CM code length (20 ms at 511.5 kcps)
CL_LEN = 767250                  # L2 CL code length (1.5 s at 511.5 kcps)
CHANNEL_CHIP_RATE = 511_500      # each of CM / CL
COMBINED_CHIP_RATE = 1_023_000   # after chip-by-chip multiplexing (== sinc² null spacing)
SIGNAL_NAME = "GPS L2C"
L2C_NULL_HZ = 1.023e6            # main-lobe null spacing == the chip rate; sidelobes step by this
LFSR_MASK = 0o445112474          # IS-GPS-200 L2C feedback polynomial (Galois)

FREQUENCIES = {"GPS L2 (1227.60 MHz)": L2_HZ / 1e6}   # presets are in MHz now

# Filter presets: {label: number of sidelobes to KEEP}. The passband is the main lobe
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


# ── Baseband buffer (CM/CL time-multiplexed, seamless loop) ────────────────────

def build_l2c_buffer(prn: int, loop: str):
    """The complex64 L2C baseband buffer at SAMP_RATE_HZ (real BPSK, Q=0). loop='cm' →
    one 20 ms CM period (CL truncated); loop='full' → one 1.5 s CL period. Each is a whole
    number of combined chips that is an exact integer sample count, so it loops with no
    seam. Filled in chunks so the full 92-Msample loop needs only its own storage plus a
    small working set (not several int64 copies of it). Returns (iq, n_samples)."""
    import numpy as np

    n_cl = CL_LEN if loop == "full" else CM_LEN
    cm = np.asarray(cm_code(prn), dtype=np.int8)
    cl = np.asarray(cl_code(prn, n_cl), dtype=np.int8)

    period_s = n_cl / CHANNEL_CHIP_RATE          # 1.5 s (full) or 20 ms (cm)
    sr = int(round(SAMP_RATE_HZ))
    n_samples = int(round(period_s * sr))

    out = np.empty(n_samples, dtype=np.complex64)
    chunk = 1 << 22                              # ~4 M samples/pass → bounded working set
    for s in range(0, n_samples, chunk):
        e = min(s + chunk, n_samples)
        n = np.arange(s, e, dtype=np.int64)
        gchip = n * COMBINED_CHIP_RATE // sr     # 0 .. 2*n_cl-1
        half = gchip >> 1
        is_cl = (gchip & 1) == 1
        bit = np.where(is_cl, cl[half % n_cl], cm[half % CM_LEN])
        out[s:e] = (1.0 - 2.0 * bit).astype(np.complex64)
    return out, n_samples


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


def _circular_convolve(x, h):
    """Circular convolution of period len(x) between complex `x` and real FIR `h` (len ≤ len(x)).
    For a short filter on a huge loop (the 1.5 s CL buffer is ~92 M samples at 61.38 MHz) a single
    monolithic DFT would need several GB, so this uses OVERLAP-ADD with a small block FFT and then
    aliases the (M−1)-sample linear-convolution tail back to the head — which is exactly what makes
    the result circular, so the filtered loop still repeats with no seam. Peak memory is one
    complex64 copy of the loop plus O(block), not O(len·16 bytes)."""
    import numpy as np
    n = len(x)
    m = len(h)
    if m >= n:                                        # tiny loop — a direct DFT is fine
        return np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, n)).astype(np.complex64)
    nfft = 1
    while nfft < 4 * m:                               # comfortably larger than the filter
        nfft <<= 1
    step = nfft - (m - 1)                             # samples consumed per block
    hf = np.fft.fft(h, nfft)
    y = np.zeros(n, dtype=np.complex64)               # accumulator (bounded memory)
    for start in range(0, n, step):
        blk = x[start:start + step]
        yb = np.fft.ifft(np.fft.fft(blk, nfft) * hf)  # linear conv of this block (len blk + m − 1)
        yb = yb[:len(blk) + m - 1]
        end = start + len(yb)
        if end <= n:
            y[start:end] += yb
        else:                                         # wrap the tail → circular aliasing
            first = n - start
            y[start:n] += yb[:first]
            y[0:len(yb) - first] += yb[first:]
    return y


def filter_buffer(base_iq, sidelobes: int, trans_hz: float):
    """Circularly filter the looped L2C buffer to keep the main lobe + `sidelobes` sidelobes.
    Circular convolution keeps the result exactly periodic, so the filtered loop has no seam;
    unity passband gain leaves the kept lobes' power unchanged. Returns (filtered_iq, n_taps,
    passband_edge_hz)."""
    fp = (int(sidelobes) + 1) * L2C_NULL_HZ          # flat passband edge (kept up to here)
    fc = fp + trans_hz / 2.0                          # −6 dB cutoff = edge + half the transition
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    return _circular_convolve(base_iq, h), m, fp


# ── Self-test (generator vs official sheet + check values; filter when numpy present) ──

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

    cm1 = cm_code(1)
    len_ok = len(cm1) == CM_LEN
    print(f"CM len={len(cm1)} ones={sum(cm1)} (≈5115) [{'OK' if len_ok else 'FAIL'}]")
    ok = ok and len_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    base, n = build_l2c_buffer(1, "cm")

    def band(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, sidelobes=2, trans_hz=0.5e6)
    main = 10 * np.log10(band(filt, 0, L2C_NULL_HZ) / band(base, 0, L2C_NULL_HZ))
    kept = 10 * np.log10(band(filt, 2 * L2C_NULL_HZ, 3 * L2C_NULL_HZ)
                         / band(base, 2 * L2C_NULL_HZ, 3 * L2C_NULL_HZ))
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

def _build_top_block(initial_file: str, center_freq_hz: float, gain_db: float,
                     amplitude: float):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack."""
    from gnuradio import gr, blocks, uhd

    class L2CTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, initial_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def swap_file(self, path):
            self.src.open(path, True)                # switch at the next work boundary

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return L2CTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter (CM/CL time-multiplexed, real IS-GPS-200 codes, "
               "1.023 Mcps BPSK) — fixed 61.38 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS satellite PRN (1..63) — the real L2C code. Fixed per run.")
        .choice("-Loop", "--loop", options=["full", "cm"], default="full",
                help="full = whole 1.5 s CL period (bit-exact CL phase, complete spectrum; "
                     "~736 MB in RAM at 61.38 MHz); cm = 20 ms CM period (CL truncated; "
                     "~9.8 MB, envelope-correct) for tight RAM.")
        .number("-Center-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=L2_HZ / 1e6,
                help="RF carrier in MHz (default L2 = 1227.60). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Maps through the unit's "
                     "calibration and snaps to its achievable grid; ignored if --gain is "
                     "given. Live.")
        .choice("-Filter", "--filter", options=["off", "on"], default="off",
                required=False, live=True,
                help="Digital passband filter on the looped buffer (unity passband gain, so "
                     "it preserves what it passes). Live.")
        .integer("-Sidelobes", "--sidelobes", min=0, max=MAX_SIDELOBES, default=2,
                 presets=SIDELOBE_PRESETS, required=False, live=True,
                 help="Passband width, as the number of sidelobes KEPT beside the main lobe: "
                      "a ±(n+1)·1.023 MHz band. Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.5,
                required=False, live=True,
                help="Filter skirt transition width beyond the passband edge (MHz) — the "
                     "steepness knob. Live (rebuilds the filtered loop).")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────────

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

    # Prebuild the unfiltered loop once (PRN/loop are fixed per run); the filter derives from it.
    base_iq, nsamp = build_l2c_buffer(args.prn, args.loop)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gps_l2c_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    # Filter "shape" (the regeneration-requiring params) — mutated by live changes.
    shape = {"on": getattr(args, "filter", "off") == "on",
             "sidelobes": int(getattr(args, "sidelobes", 2) or 0),
             "trans_hz": float(getattr(args, "transition", 0.5) or 0.5) * 1e6}

    def make_current():
        """The buffer for the current shape: the base loop, or the circularly-filtered loop.
        Returns (iq, info) where info describes the filter for the banner/report."""
        if not shape["on"]:
            return base_iq, {"on": False}
        filtered, taps, fp = filter_buffer(base_iq, shape["sidelobes"], shape["trans_hz"])
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "sidelobes": shape["sidelobes"], "trans_hz": shape["trans_hz"]}

    iq0, finfo = make_current()
    box = {"file": write_buffer(iq0)}

    tb = _build_top_block(initial_file=box["file"], center_freq_hz=center_freq_hz,
                          gain_db=gain_db, amplitude=amplitude)

    def regenerate():
        """Rebuild the loop for the current filter shape and swap it in (one seam, then it
        loops clean). Runs on the control thread; the flowgraph keeps streaming until swap."""
        iq, info = make_current()
        new_file = write_buffer(iq)
        tb.swap_file(new_file)
        old, box["file"] = box["file"], new_file
        try:
            os.unlink(old)            # safe: file_source holds the old inode until it swaps
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
        return (f"on — main + {info['sidelobes']} sidelobe(s) "
                f"(±{info['edge_hz']/1e6:.2f} MHz), {info['trans_hz']/1e6:g} MHz transition, "
                f"{info['taps']} taps")

    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  satellite PRN  : {args.prn}  (real L2C code, CM+CL)")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  modulation     : BPSK-R(1) — 1.023 Mcps (CM/CL @ 511.5 kcps each)")
    print(f"  loop           : {args.loop} "
          f"({'1.5 s (bit-exact CL)' if args.loop=='full' else '20 ms (CL truncated)'}), "
          f"{nsamp} samples ({nsamp*8/1e6:.1f} MB)")
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

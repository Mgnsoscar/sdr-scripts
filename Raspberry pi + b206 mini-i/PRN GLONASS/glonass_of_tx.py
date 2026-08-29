#!/usr/bin/env python3
"""
GLONASS L1OF / L2OF (FDMA open signal) transmitter for GNU Radio + UHD.

Purpose
───────
Transmit the legacy open GLONASS signal — L1OF (~1602 MHz) or L2OF (~1246 MHz) —
in either of two modes:

  • CHANNEL: one satellite on its own FDMA frequency channel k (−7…+6). BPSK of
    the 511-chip C/A ranging code at 0.511 Mcps, carrier tuned to that channel.
  • BAND:    the whole FDMA band at once — all 14 channels summed at their
    frequency offsets around the band centre, as a wideband receiver sees them.

⚠  RF SAFETY / LEGAL: L1 (~1602 MHz) and L2 (~1246 MHz) are live GNSS bands.
   Transmit ONLY into a shielded / conducted setup (cable + attenuators into a
   receiver or spectrum analyser) that you are LICENSED / AUTHORISED to use.
   Radiating a GNSS code over the air can jam or spoof real receivers and is
   illegal in most places.

Why GLONASS is different: FDMA, not CDMA
────────────────────────────────────────
GPS / Galileo / BeiDou are CDMA — one carrier per band, a unique code per
satellite. GLONASS L1OF/L2OF are FDMA: EVERY satellite transmits the SAME
511-chip C/A code, and satellites are told apart by CARRIER FREQUENCY:

    L1OF:  f_k = 1602 MHz + k · 0.5625 MHz      (k = −7 … +6)
    L2OF:  f_k = 1246 MHz + k · 0.4375 MHz      (k = −7 … +6)

So the per-satellite selector is the channel number k (which sets the carrier),
not a code index. (GLONASS also runs on a 0.511 MHz time base, unrelated to the
1.023 MHz of the other systems, so the master clock is a multiple of 0.511 MHz.)

C/A ranging code
────────────────
A 9-stage LFSR, generating polynomial G(x) = 1 + x⁵ + x⁹, output tapped at the
7th stage, register seeded all-ones. This is a maximal m-sequence of period
511 chips (1 ms at 0.511 Mcps) — the same code for every satellite and both
bands. --self-test verifies it is maximal (period 511) and balanced (256 ones).
No navigation data or 100 Hz meander is modulated (both are ≤100 Hz and
negligible in the spectrum), giving a clean BPSK ranging spectrum with a
±0.511 MHz main lobe.

Modes and sample rate
─────────────────────
CHANNEL mode is a real BPSK signal on one carrier: default 10.22 MHz (20 samp/
chip) — a valid B2xx master clock that captures the main lobe plus sidelobes.
BAND mode is a complex sum of all 14 channels (each given a distinct code phase
so they are not mutually coherent) centred on the band; default 12.264 MHz
(24 samp/chip) covers the ~8 MHz FDMA span. Both rates must be an integer
multiple of 0.511 MHz; the buffer is one code period (CHANNEL, 1 ms) or two
(BAND, 2 ms — the shortest window in which every channel offset completes a
whole number of cycles), so it loops seamlessly from /dev/shm.

Streaming levers (same as the other builders)
─────────────────────────────────────────────
PRECOMPUTE+LOOP · sc8 over the wire · silent after start() · master_clock_rate
pinned == sample rate (1:1). Live tuning (paramkit.live): gain, amplitude.

CLI
───
    glonass_of_tx.py --band L1 --mode channel --channel 0
    glonass_of_tx.py --band L1 --mode band
    glonass_of_tx.py --self-test        # verify the C/A m-sequence, no hardware
    glonass_of_tx.py --describe-params  # paramkit JSON schema for the GUI
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
CAL_SIGNAL_ID = "glonass_of"


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

CHIP_RATE_HZ = 0.511e6         # GLONASS C/A chip rate
CODE_LEN = 511                 # C/A code length (chips) — 1 ms period
K_MIN, K_MAX = -7, 6           # FDMA channel numbers

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
# GLONASS is a 0.511 Mcps FDMA plan (not a 1.023 MHz multiple), so the ceiling is the
# highest 0.511 MHz multiple inside the B200's range: 61.32 MHz (= 120×0.511), which also
# covers the ~9 MHz FDMA band whole. 120 samp/chip; master clock 1:1.
SAMP_RATE_HZ = 61.32e6
OTW_FORMAT = "sc8"            # over-the-wire; halves USB load

# Per-band frequency plan: base carrier + channel spacing (ICD L1/L2, ed. 5.1).
BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6},
}

# Filter: GLONASS is a lowpass job either way — a single BPSK channel (sinc², nulls every
# 0.511 MHz) or the FDMA comb of 14 carriers — so the passband is a direct half-bandwidth in
# MHz. The default (±5 MHz) keeps the whole FDMA band in band mode and the main lobe plus
# several sidelobes in channel mode.
MIN_PASSBAND_MHZ = 0.5
MAX_PASSBAND_MHZ = 30.6
PASSBAND_PRESETS = {
    "Channel main lobe (±1.02 MHz)": 1.022,
    "Channel + sidelobes (±3 MHz)": 3.0,
    "Full FDMA band (±5 MHz)": 5.0,
}


# ── C/A ranging code (pure Python) ─────────────────────────────────────────────

def glonass_ca() -> list[int]:
    """Return the 511-chip GLONASS C/A ranging code as a list of 0/1.

    9-stage LFSR, G(x) = 1 + x⁵ + x⁹ (feedback from stages 5 and 9), output taken
    from stage 7, register initialised to all ones. Same code for every satellite
    and both bands."""
    reg = [1] * 9
    out = []
    for _ in range(CODE_LEN):
        out.append(reg[6])                 # stage 7 output
        fb = reg[4] ^ reg[8]               # taps at stages 5 and 9
        reg = [fb] + reg[:8]
    return out


def channel_freq(band: str, k: int) -> float:
    """FDMA carrier frequency for band ('L1'|'L2') and channel k."""
    b = BANDS[band]
    return b["base"] + k * b["spacing"]


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    """Verify the C/A code is a maximal m-sequence (period 511, balanced 256/255)
    and print the FDMA frequency plan. (The code is a public LFSR sequence, not a
    stored table — this checks the generator and the channel plan.)"""
    code = glonass_ca()
    ones = sum(code)
    len_ok = len(code) == CODE_LEN
    bal_ok = ones == 256                    # m-sequence: 2^(n-1) ones
    # maximality: the 9-bit state must not repeat before 511 steps
    reg, seen, steps = [1] * 9, set(), 0
    for _ in range(CODE_LEN + 5):
        s = tuple(reg)
        if s in seen:
            break
        seen.add(s)
        reg = [reg[4] ^ reg[8]] + reg[:8]
        steps += 1
    max_ok = steps == CODE_LEN
    ok = len_ok and bal_ok and max_ok
    print(f"C/A code: len={len(code)} ones={ones}/255 maximal(period={steps}) "
          f"[{'OK' if ok else 'FAIL'}]")
    for band in ("L1", "L2"):
        lo, hi = channel_freq(band, K_MIN), channel_freq(band, K_MAX)
        print(f"{band}OF plan: k {K_MIN}..{K_MAX}  {lo/1e6:.4f}..{hi/1e6:.4f} MHz  "
              f"(spacing {BANDS[band]['spacing']/1e6:g} MHz)")

    sr_ok = int(round(SAMP_RATE_HZ)) % int(round(CHIP_RATE_HZ)) == 0
    print(f"fixed rate {SAMP_RATE_HZ/1e6:g} MHz = {SAMP_RATE_HZ/CHIP_RATE_HZ:.0f}×0.511 "
          f"[{'OK' if sr_ok else 'FAIL'}]")
    ok = ok and sr_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    # Channel mode: single BPSK sinc², main lobe preserved / far skirt cut.
    base, n, _ = build_channel_buffer()

    def band_p(x, lo, hi):
        X = np.fft.fftshift(np.fft.fft(x))
        f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / SAMP_RATE_HZ))
        return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

    filt, taps, fp = filter_buffer(base, passband_hz=3.0e6, trans_hz=0.3e6)
    main = 10 * np.log10(band_p(filt, 0, CHIP_RATE_HZ) / band_p(base, 0, CHIP_RATE_HZ))
    cut = 10 * np.log10(band_p(filt, 8e6, 20e6) / max(band_p(base, 8e6, 20e6), 1e-30))
    peak = float(np.max(np.abs(filt)))
    f_ok = abs(main) < 0.1 and cut < -40 and peak * AMPLITUDE < 1.0
    print(f"filter (channel ±3 MHz, {taps} taps): main lobe {main:+.3f} dB, far skirt "
          f"{cut:.0f} dB, peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
    ok = ok and f_ok

    # Band mode: the whole FDMA band survives a ±5 MHz passband.
    bb, _, _ = build_band_buffer("L1")
    bfilt, btaps, bfp = filter_buffer(bb, passband_hz=5.0e6, trans_hz=0.3e6)
    bkept = 10 * np.log10(band_p(bfilt, 0, bfp) / band_p(bb, 0, bfp))
    bpeak = float(np.max(np.abs(bfilt)))
    b_ok = abs(bkept) < 0.1 and bpeak * AMPLITUDE < 1.0
    print(f"filter (band ±5 MHz, {btaps} taps): FDMA band {bkept:+.3f} dB, "
          f"peak×amp {bpeak*AMPLITUDE:.2f} [{'OK' if b_ok else 'FAIL'}]")
    ok = ok and b_ok

    print("ALL GLONASS CHECKS PASSED" if ok else "SELF-TEST FAILED")
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
    """One-channel real BPSK C/A buffer at the fixed SAMP_RATE_HZ (carrier already at
    f_k, so baseband is the code at DC). Returns (iq, n_samples, samp_per_chip). Real →
    I, Q = 0."""
    import numpy as np
    spc = _validate_sr(SAMP_RATE_HZ)
    bipolar = (1 - 2 * np.asarray(glonass_ca(), dtype=np.int8)).astype(np.float32)
    n_samples = CODE_LEN * spc                       # one 1 ms code period
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.repeat(bipolar, spc)                # zero-order hold, ±1
    iq.imag = 0.0
    return iq, n_samples, spc


def build_band_buffer(band: str):
    """Full-band composite at the fixed SAMP_RATE_HZ: all 14 channels summed at their
    frequency offsets around the band centre. Each channel carries the same C/A code with
    a distinct cyclic code phase (so the channels are not mutually coherent), frequency-
    shifted to k·spacing. Two code periods (2 ms) so every offset completes a whole number
    of cycles. Returns (iq, n_samples, samp_per_chip)."""
    import numpy as np
    spc = _validate_sr(SAMP_RATE_HZ)
    sr = int(round(SAMP_RATE_HZ))
    spacing = BANDS[band]["spacing"]
    bipolar = (1 - 2 * np.asarray(glonass_ca(), dtype=np.int8)).astype(np.float32)

    n_samples = 2 * CODE_LEN * spc                   # 2 ms window
    idx = np.arange(n_samples, dtype=np.int64)
    chip = (idx * int(round(CHIP_RATE_HZ))) // sr    # chip number over 2 ms
    t = idx / sr
    comp = np.zeros(n_samples, dtype=np.complex64)
    for k in range(K_MIN, K_MAX + 1):
        shift = ((k - K_MIN) * 37) % CODE_LEN        # distinct fixed code phase
        code_k = bipolar[(chip + shift) % CODE_LEN]
        comp += code_k * np.exp(1j * 2 * np.pi * k * spacing * t).astype(np.complex64)
    peak = float(np.max(np.abs(comp))) or 1.0
    return (comp / peak).astype(np.complex64), n_samples, spc


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


def filter_buffer(base_iq, passband_hz: float, trans_hz: float):
    """Circularly filter the looped GLONASS buffer to a ±`passband_hz` band. Circular
    convolution keeps the result exactly periodic (seam-free loop); unity passband gain
    leaves the kept content's power unchanged. Returns (filtered_iq, n_taps,
    passband_edge_hz)."""
    import numpy as np
    fp = float(passband_hz)
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class GlonassTx(gr.top_block):
        def __init__(self):
            super().__init__("GLONASS OF TX")
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

    return GlonassTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1OF/L2OF (FDMA) transmitter — one channel or the whole band, real "
               "511-chip C/A code — fixed 61.32 MHz / sc8, looped buffer, optional "
               "power-preserving digital passband filter. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain. Authorised, shielded "
               "setups only.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1OF (~1602 MHz) or L2OF (~1246 MHz) — sets the carrier band. "
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
                     "it preserves what it passes). Live.")
        .number("-Passband", "--passband", unit="MHz",
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=5.0,
                presets=PASSBAND_PRESETS, required=False, live=True,
                help="Passband half-bandwidth kept each side of the carrier (MHz). Default 5 "
                     "keeps the whole FDMA band (band mode) or the main lobe + sidelobes "
                     "(channel mode). Live (rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.05, max=5.0, default=0.3,
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

    if mode == "band":
        base_iq, nsamp, spc = build_band_buffer(band)
        desc = f"{band}OF band composite (k {K_MIN}..{K_MAX}, 14 channels)"
    else:
        base_iq, nsamp, spc = build_channel_buffer()
        desc = f"{band}OF channel k={args.channel}"

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="glonass_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "passband_hz": float(getattr(args, "passband", 5.0) or 5.0) * 1e6,
             "trans_hz": float(getattr(args, "transition", 0.3) or 0.3) * 1e6}

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

    print("── GLONASS OF TX ───────────────────────────────────────────")
    print(f"  signal         : {desc}")
    print(f"  carrier        : {center_freq_hz/1e6:.4f} MHz"
          + ("  (band centre)" if mode == "band" else f"  (channel {args.channel})"))
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  code           : 511-chip C/A @ 0.511 Mcps (BPSK)")
    print(f"  buffer         : {nsamp} samples ({spc} samp/chip, {nsamp*8/1e6:.1f} MB)")
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

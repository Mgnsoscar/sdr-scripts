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
from paramkit import Script


# ── Constants ─────────────────────────────────────────────────────────────────

CHIP_RATE_HZ = 5.11e6          # GLONASS P-code chip rate (10 × C/A)
CODE_LEN = 5_110_000           # truncated to 1 s (reset each second)
LFSR_DEG = 25                  # 25-stage register, G(x)=1+x^3+x^25
K_MIN, K_MAX = -7, 6

BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6},
}
CHAN_DEFAULT_SR = 10.22e6      # channel mode: 2 samp/chip (~82 MB)
BAND_DEFAULT_SR = 20.44e6      # band mode: 4 samp/chip, covers ~17 MHz (~164 MB)


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


def build_channel_buffer(samp_rate_hz: float):
    """One-channel real BPSK P-code buffer for one full 1 s period (loops
    seamlessly on reset). Returns (iq, n_samples, samp_per_chip). Real → I, Q=0."""
    import numpy as np
    spc = _validate_sr(samp_rate_hz)
    chips = np.empty(CODE_LEN, dtype=np.int8)
    _pcode_into(chips)                               # ~2 s, once
    bipolar = (1 - 2 * chips).astype(np.float32)
    n_samples = CODE_LEN * spc
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.repeat(bipolar, spc)
    iq.imag = 0.0
    return iq, n_samples, spc


def build_band_buffer(band: str, samp_rate_hz: float):
    """Full-band composite: all 14 channels of the P-code summed at their
    frequency offsets (k·spacing) around the band centre, each with a distinct
    cyclic code phase. One 1 s buffer. Built in time-chunks to bound memory.
    Returns (iq, n_samples, samp_per_chip)."""
    import numpy as np
    spc = _validate_sr(samp_rate_hz)
    sr = int(round(samp_rate_hz))
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


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class GlonassSfTx(gr.top_block):
        def __init__(self):
            super().__init__("GLONASS SF TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=otw_format,
                                      channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_path, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_amplitude(self, a): self.amp.set_k(a)
        def set_gain(self, g): self.usrp.set_gain(g, 0)
        def actual_gain(self): return self.usrp.get_gain(0)
        def actual_samp_rate(self): return self.usrp.get_samp_rate()

    return GlonassSfTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1SF/L2SF (FDMA high-accuracy P-code) transmitter — one "
               "channel or the whole band, real 5.11 Mcps P-code, file-replay. "
               "Transmit only into an authorised, shielded setup.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1SF (~1602 MHz) or L2SF (~1246 MHz). Fixed per run.")
        .choice("-Mode", "--mode", options=["channel", "band"], default="channel",
                help="channel = one satellite on its FDMA carrier; band = all 14 "
                     "channels summed around the band centre. Fixed per run.")
        .integer("-Channel", "--channel", min=K_MIN, max=K_MAX, default=0,
                 help="FDMA channel number k (−7..+6); sets the carrier in channel "
                      "mode; ignored in band mode. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=50,
                live=True, help="USRP TX gain.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.9,
                live=True, help="Baseband digital amplitude (0..1). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=10.22, max=61.44,
                default=0.0,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). "
                     "Leave 0 for the mode default (channel 10.22, band 20.44 MHz). "
                     "Must be a multiple of 5.11 MHz. Buffer is 1 s, so size scales "
                     "with the rate. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load; a single "
                     "P-code channel is constant-modulus so sc8 is ideal (the band "
                     "composite is multi-level, sc16 optional there).")
    )


def _apply_live_change(tb, ctrl, name, value):
    if name == "gain":
        tb.set_gain(value)
        ctrl.report("gain", tb.actual_gain())
    elif name == "amplitude":
        tb.set_amplitude(value)
        ctrl.report("amplitude", value)


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

    if args.samp_rate <= 0:
        samp_rate_hz = BAND_DEFAULT_SR if mode == "band" else CHAN_DEFAULT_SR
    else:
        samp_rate_hz = args.samp_rate * 1e6

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="glonass_sf_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    try:
        print("[building] generating 1 s P-code buffer…", flush=True)
        if mode == "band":
            iq, nsamp, spc = build_band_buffer(band, samp_rate_hz)
            center = BANDS[band]["base"]
            desc = f"{band}SF band composite (k {K_MIN}..{K_MAX}, 14 channels)"
        else:
            iq, nsamp, spc = build_channel_buffer(samp_rate_hz)
            center = channel_freq(band, args.channel)
            desc = f"{band}SF channel k={args.channel}"
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    iq_path = os.path.join(tmpdir, f"glonass_{band}sf_{mode}.fc32")
    iq.tofile(iq_path)
    print(f"[prebuilt] {desc} → {nsamp} samples ({spc} samp/chip, "
          f"{nsamp*8/1e6:.0f} MB) → {iq_path}")

    tb = _build_top_block(iq_path, center, samp_rate_hz, args.gain, args.amplitude,
                          args.otw, "")
    print("── GLONASS SF (P-code) TX ──────────────────────────────────")
    print(f"  signal         : {desc}")
    print(f"  carrier        : {center/1e6:.4f} MHz"
          + ("  (band centre)" if mode == "band" else f"  (channel {args.channel})"))
    print(f"  sample rate    : requested {samp_rate_hz/1e6:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  code           : 5.11 Mcps P-code, 1 s period (BPSK, ±5.11 MHz lobe)")
    print(f"  otw / gain     : {args.otw} / {args.gain:g} dB")
    print(f"  amplitude      : {args.amplitude:g}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                _apply_live_change(tb, ctrl, change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

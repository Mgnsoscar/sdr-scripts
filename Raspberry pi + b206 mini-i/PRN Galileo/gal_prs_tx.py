#!/usr/bin/env python3
"""
Galileo PRS (E1-A / E6-A) SPECTRAL-SURROGATE transmitter for GNU Radio + UHD.

What this is (and is not)
─────────────────────────
The Galileo Public Regulated Service (PRS) — signal components E1-A and E6-A —
is an access-controlled, government service whose spreading codes and navigation
data are ENCRYPTED and CLASSIFIED. They are not published in any open ICD and
are not reproduced here. This script therefore transmits a SPECTRAL SURROGATE:
the correct PRS *modulation* (cosine-phased BOC, chip rate, sub-carrier rate and
carrier) driven by a public pseudo-random m-sequence standing in for the secret
code. It reproduces the PRS spectral footprint for conducted receiver-front-end,
filter and band-occupancy testing. It is noise-like to a real PRS receiver: it
carries none of the PRS code, so it cannot correlate with, spoof, or replay the
real service — and could not, since the codes are secret.

This mirrors the GPS M-code surrogate (mcode_boc_tx.py): a spectrally-faithful
BOC test signal, not a bit-exact reproduction.

⚠  RF SAFETY / LEGAL: E1 (1575.42 MHz) and E6 (1278.75 MHz) are live GNSS bands;
   E1/E6 also overlie other services. Transmit ONLY into a shielded / conducted
   setup (cable + attenuators into a receiver or spectrum analyser) that you are
   LICENSED / AUTHORISED to use. Radiating in these bands over the air can jam
   real receivers and is illegal in most places.

PRS modulation (public parameters)
──────────────────────────────────
    component   modulation        sub-carrier    chip rate     carrier
    E1-A        BOC_cos(15, 2.5)  15.345 MHz     2.5575 Mcps   1575.42 MHz
    E6-A        BOC_cos(10, 5)    10.23  MHz     5.115  Mcps   1278.75 MHz

Each is a single BOC channel: s(t) = c(t)·sc_cos(t), a real signal placed on I
with Q = 0. The cosine-phased sub-carrier is the square wave sc_cos = sgn(cos
2π R_s t) (per BOC period the piecewise pattern is +,−,−,+), which gives PRS its
characteristic split spectrum with the two main lobes pushed out to the sub-
carrier edges. c(t) here is a maximal-length LFSR m-sequence (default degree 14,
16383 chips, exactly balanced) — a stand-in whose flat PSD reproduces the PRS
envelope; it is NOT the PRS ranging code.

Sample rate (cosine-BOC needs quarter-period alignment)
───────────────────────────────────────────────────────
sc_cos switches at ¼ and ¾ of each sub-carrier period, so the sample rate must
give a whole number of samples per sub-carrier period divisible by 4 (and an
integer number of samples per chip). Per band that means:
    E1-A → 61.38 MHz (4 samples/sub-carrier period, 24 samples/chip)
    E6-A → 40.92 MHz (4 samples/sub-carrier period,  8 samples/chip)
These are picked automatically; an explicit --samp_rate that doesn't fit the
band is rejected in favour of the band's native rate (with a note). One code
period is precomputed into /dev/shm and replayed with file_source(repeat=True),
so it loops seamlessly.

Streaming levers (same as the other builders)
─────────────────────────────────────────────
PRECOMPUTE+LOOP · sc8 over the wire · silent after start() · master_clock_rate
pinned == sample rate (1:1). Live tuning (paramkit.live): gain, amplitude.

CLI
───
    gal_prs_tx.py --band E1A --gain 55 --amplitude 0.9
    gal_prs_tx.py --band E6A
    gal_prs_tx.py --self-test        # verify m-sequence + BOC_cos, no hardware
    gal_prs_tx.py --describe-params  # paramkit JSON schema for the GUI
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

MHZ = 1.023e6                  # GNSS base unit (1.023 MHz)

# Public PRS modulation per band: carrier, sub-carrier rate, chip rate, native
# sample rate (4 samples per sub-carrier period → cosine-BOC quarter alignment).
BANDS = {
    "E1A": {
        "carrier": 1575.42e6, "sub_hz": 15 * MHZ, "chip_hz": 2.5 * MHZ,
        "native_sr": 61.38e6, "label": "BOC_cos(15, 2.5)",
    },
    "E6A": {
        "carrier": 1278.75e6, "sub_hz": 10 * MHZ, "chip_hz": 5 * MHZ,
        "native_sr": 40.92e6, "label": "BOC_cos(10, 5)",
    },
}

# Surrogate spreading code: maximal-length Fibonacci LFSR. Degree 14 → 16383-chip
# m-sequence (exactly balanced), dense enough that its line spectrum reads as the
# continuous PRS PSD envelope. taps verified maximal in --self-test. NOT the PRS code.
LFSR_DEGREE = 14
LFSR_TAPS = (14, 5, 3, 1)      # primitive polynomial x^14+x^5+x^3+x+1

DEFAULT_BAND = "E1A"


# ── Surrogate code + sub-carrier (pure Python) ─────────────────────────────────

def surrogate_code(degree: int = LFSR_DEGREE, taps=LFSR_TAPS) -> list[int]:
    """Maximal-length m-sequence of length 2^degree − 1 as a list of 0/1, from a
    Fibonacci LFSR (state seeded to 1). A public stand-in for the classified PRS
    ranging code — same flat PSD, none of the PRS code content."""
    state = 1
    length = (1 << degree) - 1
    out = []
    for _ in range(length):
        out.append(state & 1)
        fb = 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (fb << (degree - 1))
    return out


def boccos_chip_pattern(samples_per_subcarrier: int, subcarriers_per_chip: int):
    """One chip of the cosine-phased BOC sub-carrier as a list of ±1: the
    piecewise pattern of sgn(cos 2π R_s t) sampled over one sub-carrier period
    (switches at ¼ and ¾ → pattern +,−,−,+ for 4 samples/period), tiled for the
    whole chip."""
    import math
    per = []
    for i in range(samples_per_subcarrier):
        frac = (i + 0.5) / samples_per_subcarrier
        per.append(1 if math.cos(2 * math.pi * frac) >= 0 else -1)
    return per * subcarriers_per_chip


def _valid_rate(band: str, samp_rate_hz: float) -> bool:
    """True if the rate gives integer samples/chip and samples-per-sub-carrier-
    period divisible by 4 (exact cosine-BOC representation)."""
    b = BANDS[band]
    sr, chip, sub = int(round(samp_rate_hz)), int(round(b["chip_hz"])), int(round(b["sub_hz"]))
    if sr % chip != 0:
        return False
    if sr % sub != 0:
        return False
    return (sr // sub) % 4 == 0


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    """Verify the surrogate m-sequence is maximal and balanced, that the cosine-
    BOC pattern is correct and DC-free, and that the band native rates satisfy the
    cosine-BOC alignment. (There is no PRS code to check against — the codes are
    classified; this checks the surrogate and the modulation only.)"""
    ok = True
    code = surrogate_code()
    period_ok = len(code) == (1 << LFSR_DEGREE) - 1
    balance_ok = sum(code) == (1 << (LFSR_DEGREE - 1))       # m-sequence: 2^(n-1) ones
    # confirm maximality: sequence must not repeat before full length
    state, seen, steps = 1, set(), 0
    for _ in range((1 << LFSR_DEGREE)):
        if state in seen:
            break
        seen.add(state)
        fb = 0
        for t in LFSR_TAPS:
            fb ^= (state >> (t - 1)) & 1
        state = (state >> 1) | (fb << (LFSR_DEGREE - 1))
        steps += 1
    maximal = steps == (1 << LFSR_DEGREE) - 1
    ok = ok and period_ok and balance_ok and maximal
    print(f"surrogate m-seq: len={len(code)} ones={sum(code)} "
          f"maximal={maximal} [{'OK' if period_ok and balance_ok and maximal else 'FAIL'}]")

    pat = boccos_chip_pattern(4, 1)
    boc_ok = pat == [1, -1, -1, 1] and sum(pat) == 0
    ok = ok and boc_ok
    print(f"BOC_cos pattern (4/period): {pat} DC-free={sum(pat)==0} "
          f"[{'OK' if boc_ok else 'FAIL'}]")

    for band in ("E1A", "E6A"):
        b = BANDS[band]
        r = _valid_rate(band, b["native_sr"])
        spc = int(b["native_sr"] / b["chip_hz"])
        spp = int(b["native_sr"] / b["sub_hz"])
        ok = ok and r
        print(f"{band} {b['label']}: native {b['native_sr']/1e6:g} MHz → "
              f"{spc} samp/chip, {spp} samp/sub-carrier [{'OK' if r else 'FAIL'}]")
    print("ALL PRS SURROGATE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer ────────────────────────────────────────────────────────────

def build_iq_buffer(band: str, samp_rate_hz: float, degree: int = LFSR_DEGREE):
    """Build a complex64 buffer of one whole surrogate-code period (loops seam-
    lessly). s = c·sc_cos is real (single BOC channel) → placed on I, Q = 0,
    unit magnitude. Returns (iq, n_samples, samples_per_chip)."""
    import numpy as np

    b = BANDS[band]
    sr = int(round(samp_rate_hz))
    chip = int(round(b["chip_hz"]))
    sub = int(round(b["sub_hz"]))
    if not _valid_rate(band, sr):
        raise ValueError(
            f"{band} needs samples/chip integer and samples/sub-carrier-period "
            f"divisible by 4; {samp_rate_hz/1e6:g} MHz does not qualify "
            f"(native {b['native_sr']/1e6:g} MHz)")
    spc = sr // chip                       # samples per chip
    spp = sr // sub                        # samples per sub-carrier period
    sub_per_chip = chip and (sub // chip) if sub >= chip else 0
    sub_per_chip = sub // chip             # sub-carrier periods per chip (integer)

    code = np.asarray(surrogate_code(degree), dtype=np.int8)     # (L,) 0/1
    bip = (1 - 2 * code).astype(np.float32)                      # ±1
    perchip = np.asarray(boccos_chip_pattern(spp, sub_per_chip), dtype=np.float32)
    assert len(perchip) == spc, (len(perchip), spc)

    s = (bip[:, None] * perchip[None, :]).reshape(-1)            # L*spc real, ±1
    n_samples = s.size
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = s
    iq.imag = 0.0
    return iq, n_samples, spc


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class PrsTx(gr.top_block):
        def __init__(self):
            super().__init__("Galileo PRS surrogate TX")
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

    return PrsTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo PRS (E1-A / E6-A) SPECTRAL SURROGATE — correct cosine-BOC "
               "modulation with a public m-sequence stand-in (the PRS codes are "
               "classified). Transmit only into an authorised, shielded setup.")
        .choice("-Band", "--band", options=["E1A", "E6A"], default=DEFAULT_BAND,
                help="PRS component. E1A→BOC_cos(15,2.5)@1575.42 MHz, "
                     "E6A→BOC_cos(10,5)@1278.75 MHz. Sets carrier + modulation "
                     "+ native sample rate. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=89.75, default=50,
                live=True, help="USRP TX gain.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.9,
                live=True, help="Baseband digital amplitude (0..1). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=10.0, max=61.44,
                default=0.0,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). "
                     "Leave 0 to use the band's native rate (E1A 61.38, E6A 40.92 "
                     "MHz). An explicit rate that breaks cosine-BOC alignment is "
                     "rejected for the native rate. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load (needed at "
                     "61.38 MS/s on a Pi); sc16 for more dynamic range.")
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
    band = args.band
    b = BANDS[band]

    # Resolve sample rate: 0 (or a rate that breaks cosine-BOC alignment) → native.
    requested = args.samp_rate * 1e6
    if args.samp_rate <= 0 or not _valid_rate(band, requested):
        if args.samp_rate > 0:
            print(f"[note] {args.samp_rate:g} MHz doesn't fit {band}'s cosine-BOC "
                  f"grid; using native {b['native_sr']/1e6:g} MHz")
        samp_rate_hz = b["native_sr"]
    else:
        samp_rate_hz = requested

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gal_prs_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    iq, nsamp, spc = build_iq_buffer(band, samp_rate_hz)
    iq_path = os.path.join(tmpdir, f"prs_{band}.fc32")
    iq.tofile(iq_path)
    print(f"[prebuilt] PRS {band} surrogate → {nsamp} samples "
          f"({spc} samp/chip, {nsamp*8/1e6:.1f} MB) → {iq_path}")

    tb = _build_top_block(iq_path, b["carrier"], samp_rate_hz, args.gain,
                          args.amplitude, args.otw, "")
    print("── Galileo PRS surrogate TX ────────────────────────────────")
    print(f"  SURROGATE      : public m-sequence, NOT the classified PRS code")
    print(f"  band           : {band}  {b['label']}")
    print(f"  carrier        : {b['carrier']/1e6:.3f} MHz")
    print(f"  sample rate    : requested {samp_rate_hz/1e6:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  chip / sub-car : {b['chip_hz']/1e6:.4f} Mcps / {b['sub_hz']/1e6:.3f} MHz")
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

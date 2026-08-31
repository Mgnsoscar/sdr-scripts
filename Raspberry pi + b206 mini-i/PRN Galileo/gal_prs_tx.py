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
    gal_prs_tx.py --band E1A --power -30 --rf on
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
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only). See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "gal_prs"


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

# ── Fixed radio setup ───────────────────────────────────────────────────────────────
# The sample rate is per-band (cosine-BOC needs 4 samples/sub-carrier period): E1A at
# 61.38 MHz, E6A at 40.92 MHz — the band's native_sr, fixed once the band is chosen.
OTW_FORMAT = "sc8"            # over-the-wire; halves USB load

# Filter: PRS is a split (cosine-BOC) spectrum, so the passband is a direct half-bandwidth
# in MHz (a lowpass edge each side of the carrier), clamped to the band's Nyquist. The
# default keeps the main split lobes (E1A lobes ±15.345, E6A lobes ±10.23 MHz).
MIN_PASSBAND_MHZ = 5.0
MAX_PASSBAND_MHZ = 30.69
PASSBAND_PRESETS = {
    "Main split lobes (±18 MHz)": 18.0,
    "Tight (±12 MHz)": 12.0,
    "Wide (±25 MHz)": 25.0,
}


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

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the filter check)")
        return 0 if ok else 1

    for band in ("E1A", "E6A"):
        sr = BANDS[band]["native_sr"]
        base, n, _ = build_iq_buffer(band)

        def bandpow(x, lo, hi):
            X = np.fft.fftshift(np.fft.fft(x))
            f = np.fft.fftshift(np.fft.fftfreq(len(x), 1.0 / sr))
            return float(np.sum(np.abs(X[(np.abs(f) >= lo) & (np.abs(f) < hi)]) ** 2))

        pb = 18.0e6 if band == "E1A" else 15.0e6
        filt, taps, fp = filter_buffer(base, passband_hz=pb, trans_hz=1.5e6, sr_hz=sr)
        kept = 10 * np.log10(bandpow(filt, 0, fp) / bandpow(base, 0, fp))
        peak = float(np.max(np.abs(filt)))
        f_ok = abs(kept) < 0.1 and peak * AMPLITUDE < 1.0
        print(f"{band} filter (±{fp/1e6:.2f} MHz, {taps} taps): kept band {kept:+.3f} dB, "
              f"peak×amp {peak*AMPLITUDE:.2f} [{'OK' if f_ok else 'FAIL'}]")
        ok = ok and f_ok

    print("ALL PRS SURROGATE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer ────────────────────────────────────────────────────────────

def build_iq_buffer(band: str, degree: int = LFSR_DEGREE):
    """Build a complex64 buffer of one whole surrogate-code period (loops seam-
    lessly) at the band's native sample rate. s = c·sc_cos is real (single BOC
    channel) → placed on I, Q = 0, unit magnitude. Returns (iq, n_samples,
    samples_per_chip)."""
    import numpy as np

    b = BANDS[band]
    sr = int(round(b["native_sr"]))
    chip = int(round(b["chip_hz"]))
    sub = int(round(b["sub_hz"]))
    if not _valid_rate(band, sr):
        raise ValueError(f"{band} native rate {b['native_sr']/1e6:g} MHz fails "
                         "the cosine-BOC alignment check")
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


# ── Digital passband filter (unity gain, circular → loop-preserving) ────────────────
# The sample rate is a per-band argument here (unlike the fixed-rate signals).

def _design_lowpass(fc_hz: float, trans_hz: float, max_taps: int, sr_hz: float):
    """Blackman-Harris windowed-sinc lowpass, UNITY passband gain, at sample rate `sr_hz`."""
    import numpy as np
    m = int(np.ceil(5.5 * sr_hz / max(trans_hz, 1.0))) | 1     # odd
    m = min(m, (max_taps | 1))
    k = np.arange(m)
    c = (m - 1) / 2.0
    fcn = min(fc_hz / sr_hz, 0.499)                 # never above Nyquist
    h = 2 * fcn * np.sinc(2 * fcn * (k - c))
    n1 = m - 1
    win = (0.35875 - 0.48829 * np.cos(2 * np.pi * k / n1)
           + 0.14128 * np.cos(4 * np.pi * k / n1) - 0.01168 * np.cos(6 * np.pi * k / n1))
    h = h * win
    h = h / h.sum()                                 # unity DC (→ passband) gain
    return h.astype(np.float64), m


def filter_buffer(base_iq, passband_hz: float, trans_hz: float, sr_hz: float):
    """Circularly filter the looped PRS buffer to a ±`passband_hz` band at sample rate
    `sr_hz`. Circular convolution keeps the result exactly periodic (seam-free loop); unity
    passband gain leaves the kept lobes' power unchanged. Returns (filtered_iq, n_taps,
    passband_edge_hz)."""
    import numpy as np
    nyq = 0.499 * sr_hz
    fp = min(float(passband_hz), nyq)               # clamp the edge to the band's Nyquist
    fc = fp + trans_hz / 2.0
    n = len(base_iq)
    h, m = _design_lowpass(fc, trans_hz, n // 2, sr_hz)
    filtered = np.fft.ifft(np.fft.fft(base_iq) * np.fft.fft(h, n)).astype(np.complex64)
    return filtered, m, fp


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, samp_rate_hz, gain_db, amplitude):
    from gnuradio import gr, blocks, uhd

    class PrsTx(gr.top_block):
        def __init__(self):
            super().__init__("Galileo PRS surrogate TX")
            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT,
                                      channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
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

    return PrsTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo PRS (E1-A / E6-A) SPECTRAL SURROGATE — correct cosine-BOC modulation "
               "with a public m-sequence stand-in (the PRS codes are classified). Fixed "
               "per-band sample rate / sc8, looped buffer, optional power-preserving digital "
               "passband filter. Level is set in dBm via the unit's calibration; uncalibrated "
               "it runs on a relative gain. Authorised, shielded setups only.")
        .choice("-Band", "--band", options=["E1A", "E6A"], default=DEFAULT_BAND,
                help="PRS component — sets carrier, modulation AND sample rate. E1A → "
                     "BOC_cos(15,2.5) @ 1575.42 MHz (61.38 MHz), E6A → BOC_cos(10,5) @ "
                     "1278.75 MHz (40.92 MHz). Fixed per run.")
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
                min=MIN_PASSBAND_MHZ, max=MAX_PASSBAND_MHZ, default=18.0,
                presets=PASSBAND_PRESETS, required=False, live=True,
                help="Passband half-bandwidth kept each side of the carrier (MHz), clamped to "
                     "the band's Nyquist. The default keeps the main split lobes. Live "
                     "(rebuilds the filtered loop).")
        .number("-Transition", "--transition", unit="MHz", min=0.1, max=8.0, default=1.5,
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
    band = args.band
    b = BANDS[band]
    center_freq_hz = b["carrier"]
    samp_rate_hz = b["native_sr"]

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

    base_iq, nsamp, spc = build_iq_buffer(band)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gal_prs_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    shape = {"on": getattr(args, "filter", "off") == "on",
             "passband_hz": float(getattr(args, "passband", 18.0) or 18.0) * 1e6,
             "trans_hz": float(getattr(args, "transition", 1.5) or 1.5) * 1e6}

    def make_current():
        if not shape["on"]:
            return base_iq, {"on": False}
        filtered, taps, fp = filter_buffer(base_iq, shape["passband_hz"], shape["trans_hz"],
                                           samp_rate_hz)
        return filtered, {"on": True, "taps": taps, "edge_hz": fp,
                          "trans_hz": shape["trans_hz"]}

    iq0, finfo = make_current()
    box = {"file": write_buffer(iq0)}

    tb = _build_top_block(box["file"], center_freq_hz, samp_rate_hz, gain_db, amplitude)

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

    print("── Galileo PRS surrogate TX ────────────────────────────────")
    print(f"  SURROGATE      : public m-sequence, NOT the classified PRS code")
    print(f"  band           : {band}  {b['label']}")
    print(f"  carrier        : {center_freq_hz/1e6:.3f} MHz")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed for {band}, "
          f"1:1 master clock)")
    print(f"  chip / sub-car : {b['chip_hz']/1e6:.4f} Mcps / {b['sub_hz']/1e6:.3f} MHz")
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

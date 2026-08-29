#!/usr/bin/env python3
"""
Frequency-comb transmitter for GNU Radio + UHD (Ettus B200-mini family).

Transmit a static **comb** of N equal-power CW tones ("knives") — ALL present at the
same time (this is not the swept CW in cw_tx.py; nothing moves). You give the comb by
its edges and pitch: the frequency of the FIRST knife, the SPACING between knives, and
the frequency of the LAST knife. The knife count is derived from those (see below), so
the three numbers can never disagree.

Prebuilt once and looped from a /dev/shm file so a Raspberry Pi sustains the rate with
no runtime IQ math (same recipe as the PRN scripts).

⚠  RF SAFETY / LEGAL: many GNSS bands live in this range. Transmit ONLY into a shielded
   / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED to use — never
   radiate over the air.

Specifying the comb (first / spacing / last → count)
────────────────────────────────────────────────────
first, spacing and last over-determine the comb (a fourth number, the count, is implied),
so rather than ask for a count that could contradict them, SPACING is authoritative and
--last is a CEILING: the knives are first, first+spacing, first+2·spacing, … up to (and
including) the largest one that is ≤ last. So N = ⌊(last−first)/spacing⌋ + 1 and the true
top knife is first + (N−1)·spacing (≤ your --last). The banner reports the count and the
actual top knife. Want an exact top knife? make (last−first) a whole multiple of spacing.

Fixed radio setup
─────────────────
  • sample rate 61.38 MHz (the max), master clock pinned 1:1;
  • over-the-wire sc8 (halves USB load);
  • baseband amplitude 0.5 (the amplitude the calibration is measured at — not a knob).
The whole comb is carried at baseband around the LO, which sits at the comb centre, so it
must fit the usable band: every knife has to sit within ±25 MHz of the LO (≈ 0.4·Fs, inside
the analog reconstruction filter), i.e. the span (last−first) can be at most ~50 MHz. With an
odd knife count the centre knife lands on the LO (DC); it is emitted at the LO like any other.

Seamless loop + clean tones
───────────────────────────
The loop is 613800 samples (10 ms) — chosen so the DFT bin spacing is exactly 100 Hz, so
every knife lands on an exact bin and the buffer repeats with no seam. Knife frequencies
are therefore quantised to 100 Hz (negligible), but the SPACING is kept perfectly uniform
(an integer number of bins). The tones use Schroeder phases and the buffer is normalised to
unit RMS, so the crest factor stays low (no clipping at amplitude 0.5) and the TOTAL power
is independent of the knife count — which keeps the dBm calibration valid for any comb.

Level, from calibration (power / gain / achievable step)
────────────────────────────────────────────────────────
--power sets the ABSOLUTE delivered TOTAL power (dBm) of the whole comb. A task that sets
SDR_CAL_SIGNAL_ID to CAL_SIGNAL_ID gets this unit's MEASURED calibration injected; --power
maps through it (gain_for_power) folded at the LO, snaps to the achievable grid, and the
banner reports the power achieved and the per-knife power (total − 10·log₁₀N). --gain instead
commands the raw SDR gain (relative), overriding --power. Uncalibrated, use --gain.

CLI
───
    comb_tx.py --first 1560 --spacing 2 --last 1590 --power -30   # 16 knives, 2 MHz apart
    comb_tx.py --first 1575 --spacing 0.5 --last 1576 --gain 60   # 3 knives, relative gain
    comb_tx.py --self-test
    comb_tx.py --describe-params
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
# it at the unit's real operating plane. Absent it, the script runs uncalibrated (relative
# gain only). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "comb"

# ── Fixed radio setup (NOT parameters) ──────────────────────────────────────────────
SAMP_RATE_HZ = 61.38e6        # the max; master clock 1:1
OTW_FORMAT = "sc8"            # over-the-wire; halves USB load
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── RF chain limits (mirrors the other scripts) ─────────────────────────────────────
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script commands)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

# ── Comb / loop constants ───────────────────────────────────────────────────────────
FREQ_RES_HZ = 100.0                              # DFT bin spacing → knife-frequency quantum
N_SAMPLES = int(round(SAMP_RATE_HZ / FREQ_RES_HZ))   # 613800 → exactly 100 Hz bins, 10 ms loop
USABLE_HALF_BW_HZ = 25.0e6                        # each knife must sit within ±this of the LO
SIGNAL_NAME = "Frequency comb"

SPACING_PRESETS = {"0.1 MHz": 0.1, "0.5 MHz": 0.5, "1 MHz": 1.0, "2 MHz": 2.0,
                   "5 MHz": 5.0, "10 MHz": 10.0}

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration if present (SDR_CALIBRATION_FILE),
    else uncalibrated (relative gain only). Cached so build_script and main agree — and so
    --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Comb geometry (first / spacing / last → knives, LO, bins) ──────────────────────

def comb_plan(first_hz: float, spacing_hz: float, last_hz: float):
    """Resolve the comb. SPACING is authoritative and `last_hz` is a ceiling: knives are
    first, first+spacing, … up to the largest ≤ last. Returns a dict with the knife count,
    the (bin-quantised) knife frequencies, the LO (comb centre), and the per-knife DFT bins.

    The LO sits at the comb midpoint. Each knife is snapped to the nearest 100 Hz bin while
    keeping the spacing an exact integer number of bins (perfectly uniform); with an odd knife
    count the centre knife therefore lands on the LO (DC)."""
    if spacing_hz <= 0:
        raise ValueError("spacing must be > 0")
    if last_hz < first_hz:
        raise ValueError("last knife must be ≥ first knife")

    n = int(math.floor((last_hz - first_hz) / spacing_hz + 1e-9)) + 1
    top_hz = first_hz + (n - 1) * spacing_hz          # actual top knife (≤ last_hz)
    lo = 0.5 * (first_hz + top_hz)                     # LO at the comb centre

    # Snap to the 100 Hz bin grid, keeping the spacing an exact integer number of bins.
    bin_spacing = int(round(spacing_hz / FREQ_RES_HZ))
    if bin_spacing < 1:
        raise ValueError(f"spacing {spacing_hz/1e6:g} MHz is finer than the {FREQ_RES_HZ:g} Hz "
                         "bin grid")
    bin0 = int(round((first_hz - lo) / FREQ_RES_HZ))
    bins = [bin0 + k * bin_spacing for k in range(n)]
    knives = [lo + b * FREQ_RES_HZ for b in bins]     # actual (quantised) knife frequencies
    actual_spacing = bin_spacing * FREQ_RES_HZ

    max_off = max(abs(k - lo) for k in knives)
    return {"n": n, "lo_hz": lo, "knives_hz": knives, "bins": bins,
            "spacing_hz": actual_spacing, "max_off_hz": max_off,
            "span_hz": knives[-1] - knives[0]}


# ── Baseband buffer (one seamless-looping comb period) ─────────────────────────────

def build_comb_buffer(bins, n: int):
    """The complex64 comb buffer: N equal-amplitude tones at the given DFT `bins`, built in
    the frequency domain (so each tone is an exact bin → seamless loop). Schroeder phases
    keep the crest factor low; normalised to UNIT RMS so the total power is fixed regardless
    of N (the calibration stays valid). Amplitude is applied live downstream. Returns iq."""
    import numpy as np

    X = np.zeros(N_SAMPLES, dtype=np.complex128)
    for k in range(n):
        phi = -math.pi * (k * k) / n                  # Schroeder phasing → low PAPR
        X[bins[k] % N_SAMPLES] = np.exp(1j * phi)
    iq = np.fft.ifft(X)
    rms = math.sqrt(float(np.mean(np.abs(iq) ** 2)))
    iq = iq / (rms if rms > 0 else 1.0)               # unit RMS
    return iq.astype(np.complex64)


# ── Self-test (geometry + spectrum + crest, no hardware) ───────────────────────────

def _self_test() -> int:
    ok = True

    # Geometry: count derivation and last-knife snapping.
    p = comb_plan(1000e6, 5e6, 1024e6)                 # 1000,1005,1010,1015,1020 → 5 knives
    g_ok = (p["n"] == 5 and abs(p["knives_hz"][-1] - 1020e6) < FREQ_RES_HZ
            and abs(p["spacing_hz"] - 5e6) < 1e-6)
    print(f"geometry 1000/5/1024 MHz → {p['n']} knives, top {p['knives_hz'][-1]/1e6:.4f} MHz, "
          f"spacing {p['spacing_hz']/1e6:g} MHz [{'OK' if g_ok else 'FAIL'}]")
    ok = ok and g_ok

    p1 = comb_plan(1575.42e6, 1e6, 1575.42e6)          # single knife
    s_ok = p1["n"] == 1 and abs(p1["knives_hz"][0] - 1575.42e6) < FREQ_RES_HZ
    print(f"single knife 1575.42 MHz → {p1['n']} knife [{'OK' if s_ok else 'FAIL'}]")
    ok = ok and s_ok

    try:
        import numpy as np
    except ImportError:
        print("(numpy absent — skipping the spectrum check)")
        return 0 if ok else 1

    # Spectrum: exactly N bins carry power, at the expected positions, equal magnitude.
    iq = build_comb_buffer(p["bins"], p["n"])
    X = np.abs(np.fft.fft(iq))
    thresh = 0.5 * X.max()
    peaks = set(np.where(X > thresh)[0].tolist())
    want = {b % N_SAMPLES for b in p["bins"]}
    mags = np.array([X[b % N_SAMPLES] for b in p["bins"]])
    flat = float(mags.max() / mags.min())              # equal-power knives → ≈ 1.0
    spec_ok = peaks == want and flat < 1.001
    print(f"spectrum: {len(peaks)} tones at the expected bins, flatness {flat:.4f} "
          f"[{'OK' if spec_ok else 'FAIL'}]")
    ok = ok and spec_ok

    # Crest factor across a range of comb sizes → peak×amplitude must clear full scale.
    worst = 0.0
    for nk in (1, 2, 3, 5, 8, 16, 32, 64, 128):
        pk = comb_plan(1550e6, 0.5e6, 1550e6 + (nk - 1) * 0.5e6)
        b = build_comb_buffer(pk["bins"], pk["n"])
        crest = float(np.max(np.abs(b)))               # unit RMS → this is the crest factor
        worst = max(worst, crest * AMPLITUDE)
    crest_ok = worst < 1.0
    print(f"crest: worst peak×amp over N=1..128 is {worst:.3f} (< 1.0) [{'OK' if crest_ok else 'FAIL'}]")
    ok = ok and crest_ok

    print("SELF-TEST OK" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ───────────────────────────────────────────────────────────────────────

def _build_top_block(iq_file: str, center_freq_hz: float, gain_db: float, amplitude: float):
    """The GNU Radio top_block, imported lazily so the module loads without a radio stack."""
    from gnuradio import gr, blocks, uhd

    class CombTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            args = (f"master_clock_rate={SAMP_RATE_HZ:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            self.usrp = uhd.usrp_sink(
                args, uhd.stream_args(cpu_format="fc32", otw_format=OTW_FORMAT, channels=[0]))
            self.usrp.set_samp_rate(SAMP_RATE_HZ)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

        def actual_gain(self):
            return self.usrp.get_gain(0)

        def actual_samp_rate(self):
            return self.usrp.get_samp_rate()

    return CombTx()


# ── Parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} transmitter — a static comb of equal-power CW tones (all "
               "present at once), given by the first knife, the spacing, and the last knife "
               "(the count is derived). Fixed 61.38 MHz / sc8; level set in dBm via the "
               "unit's calibration, else a relative gain. Authorised, shielded setups only.")
        .number("-First-knife", "--first", unit="MHz", min=70.0, max=6000.0,
                default=1575.42, required=True,
                help="Frequency of the FIRST (lowest) knife, in MHz. Fixed per run.")
        .number("-Spacing", "--spacing", unit="MHz", min=0.01, max=50.0, default=1.0,
                presets=SPACING_PRESETS, required=True,
                help="Spacing between adjacent knives, in MHz. Authoritative — the knife "
                     "count follows from it and --last. Fixed per run.")
        .number("-Last-knife", "--last", unit="MHz", min=70.0, max=6000.0,
                default=1585.42, required=True,
                help="Ceiling for the LAST (highest) knife, in MHz: knives are placed up to "
                     "the largest one ≤ this. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE TOTAL power of the comb at the delivered plane (dBm). Maps "
                     "through the unit's calibration and snaps to its achievable grid; "
                     "ignored if --gain is given. Live.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: the SDR's raw TX gain (dB) directly, bypassing the dBm "
                     "calibration. When given, overrides --power. Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the gain AND baseband amplitude to 0; ON "
                     "restores them. Live.")
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

    # Resolve the comb geometry (spacing authoritative, --last a ceiling).
    try:
        plan = comb_plan(args.first * 1e6, args.spacing * 1e6, args.last * 1e6)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if plan["max_off_hz"] > USABLE_HALF_BW_HZ:
        print(f"error: the comb spans ±{plan['max_off_hz']/1e6:.2f} MHz around its LO, past the "
              f"±{USABLE_HALF_BW_HZ/1e6:g} MHz usable band at 61.38 MHz — narrow the range "
              f"(last−first ≤ ~{2*USABLE_HALF_BW_HZ/1e6:.0f} MHz).", file=sys.stderr)
        return 2

    lo_hz = plan["lo_hz"]
    pmap = power_map()
    amplitude = pmap.amplitude

    # Gain precedence: explicit --gain (raw) > calibrated --power (folded at the LO) > refuse.
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power, freq=lo_hz)
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    iq = build_comb_buffer(plan["bins"], plan["n"])

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="comb_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    iq_file = os.path.join(tmpdir, "comb.fc32")
    iq.tofile(iq_file)

    tb = _build_top_block(iq_file, lo_hz, gain_db, amplitude)

    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    per_knife = 10.0 * math.log10(plan["n"])          # total − this = per-knife power
    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  knives         : {plan['n']}  ({plan['knives_hz'][0]/1e6:.4f} … "
          f"{plan['knives_hz'][-1]/1e6:.4f} MHz)")
    print(f"  spacing        : {plan['spacing_hz']/1e6:g} MHz  (span {plan['span_hz']/1e6:g} MHz)")
    print(f"  LO centre      : {lo_hz/1e6:.4f} MHz  (knives at ±{plan['max_off_hz']/1e6:.2f} MHz)")
    print(f"  sample rate    : {tb.actual_samp_rate()/1e6:.6f} MHz (fixed, 1:1 master clock)")
    print(f"  buffer         : {N_SAMPLES} samples ({N_SAMPLES*8/1e6:.1f} MB, 10 ms loop)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm total  ({pmap.label})")
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=lo_hz):.2f} dBm total, "
              f"{pmap.power_for_gain(gain_db, freq=lo_hz) - per_knife:.2f} dBm/knife")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  otw            : {OTW_FORMAT}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "power" and pmap.has_absolute:
            state["gain"] = pmap.gain_for_power(float(value), freq=lo_hz)
            if state["rf_on"]:
                tb.set_gain(state["gain"])
            ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=lo_hz), 2))
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

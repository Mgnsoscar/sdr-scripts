#!/usr/bin/env python3
"""
FM-chirp transmitter for GNU Radio + UHD (Ettus B200-mini family).

A sine / triangle / sawtooth / square frequency sweep, precomputed and replayed
from a file so a Raspberry Pi can sustain a high sample rate (40–60 MS/s) — the
same recipe as gps_l1ca_tx.py, applied to the swept-tone chirp from TriangleChirp.

The instantaneous frequency is  f(t) = waveform(t) · (sweep_bw / 2), i.e. it
sweeps ±sweep_bw/2 around the carrier, and the modulating waveform repeats at
`sweep_rate`.

⚠  RF SAFETY / LEGAL: transmit ONLY into a shielded / conducted setup (cable +
   attenuators) on frequencies you are LICENSED / AUTHORISED to use.

Why this reaches 40–60 MS/s on a Pi (see gps_l1ca_tx.py for the full write-up)
────────────────────────────────────────────────────────────────────────────
  1. PRECOMPUTE + LOOP — one whole sweep is built once and replayed with
     blocks.file_source(repeat=True); no per-sample NumPy in a work() block
     (that per-sample synthesis is exactly what capped the old TriangleChirp).
  2. sc8 over the wire — halves USB payload.
  3. Quiet — UHD fastpath/console logging off, and the task runs with
     PYTHONUNBUFFERED=0 (configs/tasks.yaml) so nothing is written mid-stream.
  4. 1:1 master clock — master_clock_rate is pinned to the sample rate, so UHD
     runs the AD9361 with no FPGA resampling and no rate coercion.

Seamless looping of a SWEPT buffer
──────────────────────────────────
A frequency sweep only loops without a seam if the accumulated phase closes at
the wrap. Two things guarantee it here:
  • the buffer holds a whole number of sweep periods (integer samples/period), and
  • the swept frequency is forced to EXACT zero mean, so the phase integrated
    over the buffer returns to its start (verified to ~1e-14 rad; see --self-test).
The tiny mean removed is a fraction of a hertz for the symmetric waveforms, so
the carrier is unaffected.

Live tuning (retune while transmitting, via paramkit.live)
──────────────────────────────────────────────────────────
    freq       → UHD tune_request        (instant)
    lo_offset  → UHD tune_request        (instant)
    power      → dBm → set_gain          (instant — see the USER CALIBRATION block)
    rf         → on/off mute/unmute      (instant — gain AND amplitude to 0 / back)
    bw         → rebuild buffer + swap    ┐ shape changes: regenerate one sweep
    sweep_rate → rebuild buffer + swap    │ into a new /dev/shm file and
    waveform   → rebuild buffer + swap    ┘ file_source.open() it (one brief seam
                                            at the swap, then it loops clean)
Regeneration runs on the control thread; the flowgraph keeps streaming the old
buffer on GNU Radio's scheduler threads until the swap, so RF never stops.
Sample rate and otw are fixed per run (restart to change them).

CLI
───
    fm_chirp_tx.py --freq 1.57542e9 --bw 20 --rate 0.2 --waveform Sawtooth --power -30
    fm_chirp_tx.py --self-test        # verify seamless phase closure, no hardware
    fm_chirp_tx.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time

# Quiet UHD/GNU Radio BEFORE the libs load (imported lazily inside main()).
os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")
os.environ.setdefault("GR_DONT_LOAD_PREFS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the script runs uncalibrated (relative gain only)
# (unchanged behaviour). See the agent's docs/calibration.md.
CAL_SIGNAL_ID = "fm_chirp"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# (a cable/antenna whose loss varies with frequency) has a --power scale that MOVES with
# frequency, so the map is folded at THIS param's value — and it is live, so retuning the
# chirp's centre re-scales --power on the fly. The client folds the --power range shown in
# the Run / sequence form at the same frequency. See calkit / calibration-v2.
CAL_FREQ_PARAM = "freq"


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



# ── Power map: the unit's injected calibration curve if present, else the baked
#    constants above (identical to the old single-anchor slope-1 behaviour) ────────

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script and
    main share one — and so --power's schema bounds match the real operating range
    (calibrated → e.g. EIRP; else the baked SDR-port range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Constants ─────────────────────────────────────────────────────────────────

WAVEFORMS = ["Sine", "Triangle", "Sawtooth", "Square"]

# Tile the single sweep period up to at least this many samples so file_source
# wraps infrequently at high rate (whole periods only → still seamless).
MIN_BUFFER_SAMPS = 1 << 18   # 262144 samples ≈ 2 MB as fc32

# Named GNSS carriers (same list as the original TriangleChirp), in Hz.
FREQUENCIES = {
    "GPS L1": 1575.42e6, "GPS L2": 1227.60e6, "GPS L5": 1176.45e6,
    "Galileo E1": 1575.42e6, "Galileo E5a": 1176.45e6, "Galileo E5b": 1207.14e6,
    "Galileo E5": 1191.795e6, "Galileo E6": 1278.75e6,
    "BeiDou B1I": 1561.098e6, "BeiDou B1C": 1575.42e6, "BeiDou B2a": 1176.45e6,
    "BeiDou B2b": 1207.14e6, "BeiDou B2": 1191.795e6, "BeiDou B3": 1268.52e6,
    "GLONASS L1": 1602.0e6, "GLONASS L2": 1246.0e6, "GLONASS L3": 1202.025e6,
    "Iridium": 1621.25e6,
}


# ── Modulating waveform (NumPy, for the buffer builder) ────────────────────────

def _mod_waveform(kind: str, p):
    """Value in [-1, 1] for phase p in [0, 1). Zero-mean over a period.
    p is a NumPy array; the result modulates the instantaneous frequency."""
    import numpy as np
    if kind == "Sine":
        return np.sin(2 * np.pi * p)
    if kind == "Triangle":
        return 1.0 - 2.0 * np.abs(2.0 * p - 1.0)          # -1 → +1 → -1
    if kind == "Sawtooth":
        return 2.0 * p - 1.0                               # -1 → +1 (reset)
    if kind == "Square":
        return np.where(np.sin(2 * np.pi * p) >= 0, 1.0, -1.0)
    raise ValueError(f"unknown waveform {kind!r}")


def build_chirp_buffer(waveform: str, sweep_bw_hz: float, sweep_rate_hz: float,
                       samp_rate_hz: float):
    """Build a complex64 baseband buffer holding a whole number of sweep periods
    that loops with no seam. Unit magnitude (amplitude is applied live
    downstream). Returns (iq, samples_per_period, reps, actual_sweep_rate_hz)."""
    import numpy as np

    n_per = max(2, int(round(samp_rate_hz / sweep_rate_hz)))
    actual_sweep_rate = samp_rate_hz / n_per       # rate quantised to the grid

    p = np.arange(n_per, dtype=np.float64) / n_per
    freq = _mod_waveform(waveform, p) * (sweep_bw_hz / 2.0)
    freq = freq - freq.mean()                      # exact zero-mean → phase closes
    phase = (2.0 * np.pi / samp_rate_hz) * np.cumsum(freq)
    period = np.exp(1j * phase).astype(np.complex64)

    reps = max(1, -(-MIN_BUFFER_SAMPS // n_per))   # ceil division
    iq = np.tile(period, reps)
    return iq, n_per, reps, actual_sweep_rate


# ── Self-test: seamless phase closure, no NumPy / no hardware ──────────────────

def _self_test() -> int:
    import cmath
    import math

    def wave(kind, p):
        if kind == "Sine":     return math.sin(2 * math.pi * p)
        if kind == "Triangle": return 1.0 - 2.0 * abs(2.0 * p - 1.0)
        if kind == "Sawtooth": return 2.0 * p - 1.0
        if kind == "Square":   return 1.0 if math.sin(2 * math.pi * p) >= 0 else -1.0
        raise ValueError(kind)

    fs, bw, rate = 40e6, 20e6, 200e3
    ok = True
    for kind in WAVEFORMS:
        n = max(2, round(fs / rate))
        freq = [wave(kind, i / n) * (bw / 2.0) for i in range(n)]
        m = sum(freq) / n
        freq = [f - m for f in freq]
        acc, phase = 0.0, []
        for f in freq:
            acc += 2 * math.pi / fs * f
            phase.append(acc)
        iq = [cmath.exp(1j * ph) for ph in phase]
        expected = 2 * math.pi / fs * freq[0]
        measured = cmath.phase(iq[0] / iq[-1])
        seam_err = abs(((measured - expected + math.pi) % (2 * math.pi)) - math.pi)
        good = seam_err < 1e-9
        ok = ok and good
        print(f"{kind:9s}: n={n} seam_err={seam_err:.2e} rad [{'OK' if good else 'FAIL'}]")
    print("SEAMLESS — ALL WAVEFORMS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(initial_file: str, center_freq_hz: float, lo_offset_hz: float,
                     samp_rate_hz: float, gain_db: float, amplitude: float,
                     otw_format: str, extra_args: str):
    from gnuradio import gr, blocks, uhd

    class ChirpTx(gr.top_block):
        def __init__(self):
            super().__init__("FM chirp TX")
            self._freq_hz = center_freq_hz
            self._lo_hz = lo_offset_hz

            args = (f"master_clock_rate={samp_rate_hz:.0f},"
                    "num_send_frames=512,send_frame_size=16000")
            if extra_args:
                args += "," + extra_args

            self.usrp = uhd.usrp_sink(
                args,
                uhd.stream_args(cpu_format="fc32", otw_format=otw_format,
                                channels=[0]),
            )
            self.usrp.set_samp_rate(samp_rate_hz)
            self._retune()
            self.usrp.set_gain(gain_db, 0)

            self.src = blocks.file_source(gr.sizeof_gr_complex, initial_file,
                                          repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def _retune(self) -> None:
            self.usrp.set_center_freq(
                uhd.tune_request(self._freq_hz, self._lo_hz), 0)

        def set_center_frequency(self, hz: float) -> None:
            self._freq_hz = hz
            self._retune()

        def set_lo_offset(self, hz: float) -> None:
            self._lo_hz = hz
            self._retune()

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def swap_file(self, path: str) -> None:
            self.src.open(path, True)          # switch at next work boundary

        def actual_freq(self) -> float:
            return self.usrp.get_center_freq(0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

        def actual_samp_rate(self) -> float:
            return self.usrp.get_samp_rate()

    return ChirpTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("FM-chirp transmitter (sine/triangle/sawtooth/square sweep), "
               "file-replay at high sample rate. Transmit only into an "
               "authorised, shielded setup.")
        .number("-Center-frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1575.42e6, required=True, live=True,
                help="RF carrier. Live.")
        .number("-Sweep-BW", "--bw", unit="MHz", min=0.001, max=60.0, default=20.0,
                required=True, live=True,
                help="Peak-to-peak sweep width; f sweeps ±bw/2. Live (regenerates).")
        .number("-Sweep-rate", "--rate", unit="MHz", min=0.0001, max=5.0,
                default=0.2, required=True, live=True,
                help="How fast the sweep repeats. Live (regenerates).")
        .choice("-Waveform", "--waveform", options=WAVEFORMS, default="Sawtooth",
                required=True, live=True,
                help="Sweep shape. Live (regenerates).")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given (relative wins). Live.")
        .number("-LO-offset", "--lo_offset", unit="MHz", min=-30.0, max=30.0,
                default=0.0, live=True,
                help="LO offset to push LO leakage out of band. Live.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=1.0, max=61.44,
                default=40.0,
                help="Host/DAC sample rate; master clock pinned equal (1:1). "
                     "Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire format. sc8 halves USB load (needed for "
                     "40–60 MS/s on a Pi); sc16 for more dynamic range.")
        .choice("-RF", "--rf", options=["on", "off"], default="on",
                required=False, live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them. Change the power (or the "
                     "calibration gain) while OFF and it takes effect when you turn ON.")
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power).
        # This is also the calibration knob — set it while measuring output vs gain on
        # a spectrum analyser to fill in OUTPUT_POWER_DBM / GAIN_AT_MAX_DB above.
        .number("-Analog-bandwidth", "--bandwidth", unit="MHz", min=0.0, max=56.0,
                default=0.0, required=False,
                help="AD9361 analog TX filter bandwidth (MHz) — the real baseband LPF, set "
                     "independently of the sample rate. 0 = auto (UHD picks it from the "
                     "sample rate, the old behaviour); a positive value band-limits the "
                     "signal WITHOUT changing the master clock. Coerced to the radio's "
                     "range; the banner reports the actual.")
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. Live.")
    )
    return s


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()
    samp_rate_hz = args.sample_rate * 1e6
    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present,
    # so you can command a gain directly or measure output power at it.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif power_map().has_absolute:                  # calibrated: the authored absolute --power
        # Fold the calibration at the transmit frequency so a frequency-dependent chain maps
        # --power on the right scale (the chirp can retune live — see apply_change).
        gain_db = power_map().gain_for_power(args.power, freq=args.freq)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="fm_chirp_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    def write_buffer(iq) -> str:
        fd, path = tempfile.mkstemp(suffix=".fc32", dir=tmpdir)
        os.close(fd)
        iq.tofile(path)
        return path

    # Current "shape" (the regeneration-requiring params) — mutated by live changes.
    shape = {"waveform": args.waveform,
             "bw_hz": args.bw * 1e6,
             "rate_hz": args.rate * 1e6}

    iq, n_per, reps, actual_rate = build_chirp_buffer(
        shape["waveform"], shape["bw_hz"], shape["rate_hz"], samp_rate_hz)
    cur_file = write_buffer(iq)

    tb = _build_top_block(
        initial_file=cur_file, center_freq_hz=args.freq,
        lo_offset_hz=args.lo_offset * 1e6, samp_rate_hz=samp_rate_hz,
        gain_db=gain_db, amplitude=amplitude, otw_format=args.otw,
        extra_args="")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    # Track the live transmit frequency and (in absolute mode) the held target power, so a
    # live retune can re-map --power at the new frequency on a frequency-dependent chain.
    _target_power = args.power if (pmap.has_absolute and gain_cal is None) else None
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db,
             "freq": float(args.freq), "power": _target_power}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    box = {"file": cur_file}   # mutable holder so the closure can swap/clean up

    def regenerate() -> float:
        iq, _n, _r, actual = build_chirp_buffer(
            shape["waveform"], shape["bw_hz"], shape["rate_hz"], samp_rate_hz)
        new_file = write_buffer(iq)
        tb.swap_file(new_file)
        old, box["file"] = box["file"], new_file
        # Safe to unlink now: file_source keeps the old inode until it closes it
        # at the swap; the space frees then (Linux unlink-while-open semantics).
        try:
            os.unlink(old)
        except OSError:
            pass
        return actual

    print("── FM chirp TX ─────────────────────────────────────────────")
    print(f"  carrier        : {args.freq/1e6:.3f} MHz  (LO offset {args.lo_offset:g} MHz)")
    print(f"  sample rate    : requested {args.sample_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  waveform       : {args.waveform}")
    print(f"  sweep bw       : {args.bw:g} MHz (±{args.bw/2:g} MHz)")
    print(f"  sweep rate     : requested {args.rate*1e3:g} kHz, "
          f"got {actual_rate/1e3:.3f} kHz ({n_per} samples/period × {reps} reps)")
    if power_map().has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({power_map().label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:                       # e.g. calibration amplitude != this
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")   # script's fixed amplitude
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  otw            : {args.otw}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        if name == "freq":
            tb.set_center_frequency(value)
            state["freq"] = float(value)
            ctrl.report("freq", tb.actual_freq())
            # A frequency-dependent calibration re-scales --power with frequency, so re-map
            # the held target power at the new frequency — delivered power stays as requested.
            if state.get("power") is not None:
                state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
                if state["rf_on"]:
                    tb.set_gain(state["gain"])
                    ctrl.report("power",
                                round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"]), 2))
                else:
                    ctrl.report("power",
                                round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
        elif name == "lo_offset":
            tb.set_lo_offset(value * 1e6)
            ctrl.report("lo_offset", value)
        elif name == "power":
            # power/gain edits are staged into state["gain"] and only reach the radio
            # when RF is on; the --rf toggle mutes/restores gain AND amplitude. Folded at
            # the current transmit frequency (frequency-dependent chains).
            state["power"] = float(value)
            state["gain"] = pmap.gain_for_power(state["power"], freq=state["freq"])
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain(), freq=state["freq"]), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"], freq=state["freq"]), 2))
        elif name == "gain":
            # A raw gain overrides the dBm mapping — drop any held target power so a later
            # frequency change doesn't re-map back to it.
            state["power"] = None
            state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(value)))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("gain", round(tb.actual_gain(), 2))
            else:
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
        elif name in ("bw", "rate", "waveform"):
            if name == "bw":
                shape["bw_hz"] = value * 1e6
            elif name == "rate":
                shape["rate_hz"] = value * 1e6
            else:
                shape["waveform"] = value
            actual = regenerate()
            ctrl.report("rate" if name == "rate" else name,
                        actual / 1e6 if name == "rate" else value)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    _bw = float(getattr(args, "bandwidth", 0.0) or 0.0)
    if _bw > 0:                                   # AD9361 analog LPF, independent of master clock
        tb.usrp.set_bandwidth(_bw * 1e6, 0)
    try:
        print(f"  analog TX BW   : {tb.usrp.get_bandwidth(0)/1e6:.3f} MHz (AD9361 filter"
              f"{'' if _bw > 0 else ' — auto from Fs'})")
    except Exception:      # noqa: BLE001
        pass
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

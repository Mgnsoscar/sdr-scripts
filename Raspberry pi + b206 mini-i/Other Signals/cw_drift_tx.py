#!/usr/bin/env python3
"""
CW frequency-drift transmitter for GNU Radio + UHD (Ettus B200-mini family).

Emits a continuous-wave tone that DRIFTS linearly from a start frequency to an end
frequency over a duration — anything from a few kHz in minutes to HUNDREDS OF MHz over
many hours (up to 7 days). For a plain, non-drifting tone use the companion cw_tx.py.

How the drift works
───────────────────
The emitted frequency is the USRP analog LO plus a software baseband tone (a phase-
continuous NCO, analog.sig_source_c) mixed up by it. The baseband tone can only occupy
±(sample_rate/2), so two regimes are chosen automatically from the span |end−start|:

  • Narrow drift (the span fits one baseband window): the LO sits fixed at the drift
    centre and the NCO carries the whole sweep — perfectly continuous, no retunes.

  • Wide drift (e.g. 1600 → 1300 MHz): the sweep is split into WINDOWS of
    SWEEP_MARGIN × sample_rate. Within a window the analog LO is fixed and the NCO moves
    the tone; when the tone reaches a window edge the LO HOPS by one window and the NCO
    wraps to the opposite edge, so the emitted frequency continues exactly where it was.
    Each hop re-locks the synthesiser (a few ms), so the output is BLANKED around it
    (--hop_blank, 20 ms by default) to hide the settle. A 300 MHz drift at the default
    2 MHz sample rate is ~215 hops — over a three-hour drift that is one 20 ms blank
    every ~50 s. A higher sample rate widens the windows (10 MHz → ~43 hops) at the cost
    of USB/CPU load on the Pi.

Calibrated power TRACKS the drift: --power is re-folded through the unit's calibration
at the LIVE frequency (source flatness, cable tables, frequency-dependent limits) as the
tone moves — every REFOLD_STEP_HZ — so the delivered dBm stays as requested across the
whole sweep. Where a frequency-dependent ceiling sits below the request the gain is
clamped at the ceiling (safe), and the script says so at start (the stretches affected).
The drift shares the pure tone's calibration signal ("cw_tone"): at any instant it IS a
pure CW at one frequency, at the same amplitude, so the same measured curve applies —
the unit's frequency tables supply the rest. With a programmable attenuator in the chain
the SDR/attenuator SPLIT is chosen ONCE, at the drift's start frequency — the carrier the
agent positions the attenuator at — and PINNED for the whole drift: as the tone moves only
the SDR gain re-folds (PowerMap.gain_for_power(..., applied_db=…)), never the attenuation
the agent set. (Re-realizing at every new frequency would hop the assumed attenuation by
whole steps while the physical attenuator stayed put.) A live --power change re-picks the
split at the start frequency — exactly where the agent repositions the attenuator for it.

The drift runs on its own timeline from the moment the script starts — independent of
RF. --rf on/off is a pure mute/unmute and does NOT start, stop, or restart the sweep;
fire --restart to re-run the ramp from the start frequency.

    --drift once      : ramp start→end over --duration, then hold at end (default)
    --drift loop      : ramp start→end, jump back to start, repeat
    --drift pingpong  : ramp start→end→start→… (triangle)

⚠  RF SAFETY / LEGAL: many presets are live GNSS bands. Transmit ONLY into a
   shielded / conducted setup (cable + attenuators) you are LICENSED / AUTHORISED
   to use — never radiate over the air.

CLI
───
    cw_drift_tx.py --freq 1575.42 --freq_end 1575.43 --duration 20 --power -30 --rf on   # 10 kHz / 20 min (MHz, min)
    cw_drift_tx.py --freq 1600 --freq_end 1300 --duration 180 --power -30 --rf on       # 300 MHz / 3 h
    cw_drift_tx.py --freq 1227.6 --freq_end 1228.6 --drift pingpong --gain 60 --rf on
    cw_drift_tx.py --self-test        # drift + LO-hop planner math, no hardware
    cw_drift_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# Stable calibration signal id — SHARED with cw_tx.py: at any instant the drift is a pure CW
# tone at one frequency at the same baseband amplitude, so the same measured curve applies and
# one calibration serves both scripts; the unit's source-flatness / cable tables supply the
# frequency dependence the sweep crosses. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads it and
# --power maps through the unit's MEASURED curve. Absent it, the script runs uncalibrated.
CAL_SIGNAL_ID = "cw_tone"

# Which parameter carries the transmit frequency. A frequency-dependent calibration chain
# has a --power scale that MOVES with frequency, so the GUI folds the map at THIS param's value
# (the drift START); at runtime the script re-folds continuously at the live frequency.
CAL_FREQ_PARAM = "freq"


# ═══════════════════════════════════════════════════════════════════════════════
# RF chain limits — there is NO baked dBm power scale. Absolute --power (dBm) comes
# only from the unit's injected calibration; uncalibrated, the script runs on a
# relative gain (never invented power levels). GAIN_AT_MAX_DB is the safety ceiling.
# ═══════════════════════════════════════════════════════════════════════════════
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task parameter:
# the calibration is measured at THIS amplitude, so a unit calibrated at a different
# amplitude no longer matches. calkit detects that at load and runs UNCALIBRATED with a
# loud warning until it is re-calibrated here. Identical to cw_tx.py (shared calibration).
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum.
HW_MAX_GAIN_DB = 89.75


# ── Power map: the unit's injected calibration curve if present, else uncalibrated ──

_PMAP = None


def power_map() -> PowerMap:
    """The active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else uncalibrated (relative gain only). Cached, so build_script
    and main share one — and so --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ── Constants ─────────────────────────────────────────────────────────────────

# Named GNSS carriers in MHz — the unit --freq / --freq_end are declared in (the GUI shows/enters
# MHz, like the PRN scripts' -Center-frequency); main() scales to Hz once at the boundary and the
# planner math below is all Hz.
FREQUENCIES = {
    "GPS L1 / Galileo E1 / BeiDou B1C (1575.42 MHz)": 1575.42,
    "GPS L2 (1227.60 MHz)": 1227.60,
    "GPS L5 / Galileo E5a (1176.45 MHz)": 1176.45,
    "Galileo E5b / BeiDou B2b (1207.14 MHz)": 1207.14,
    "Galileo E5 centre (1191.795 MHz)": 1191.795,
    "Galileo E6 (1278.75 MHz)": 1278.75,
    "BeiDou B1I (1561.098 MHz)": 1561.098,
    "BeiDou B3I (1268.52 MHz)": 1268.52,
    "GLONASS L1 (1602.0 MHz)": 1602.0,
    "GLONASS L2 (1246.0 MHz)": 1246.0,
    "Iridium (1621.25 MHz)": 1621.25,
}
SAMPLE_RATES_MHZ = {"1 MHz (narrow drift)": 1.0, "2 MHz (default)": 2.0, "5 MHz": 5.0,
                    "10 MHz (wide drift, fewer hops)": 10.0, "20 MHz": 20.0}
MAX_DURATION_MIN = 7 * 24 * 60.0  # 7 days, in the MINUTES --duration is entered in — a "very long"
                                  # drift is the point of this script; main() scales to seconds once
DEFAULT_HOP_BLANK_MS = 20.0      # mute around each analog-LO hop (its synth relock settle)

# Fraction of the sample rate the baseband tone may span before the LO hops. Keeps the tone
# inside the flat part of the DAC / anti-alias response, clear of the band-edge rolloff.
SWEEP_MARGIN = 0.7

# Re-fold the calibrated gain once the tone has moved this far since the last fold. The
# unit's frequency tables (source flatness, cables) are sampled far coarser than this, so a
# finer cadence would only repeat the same gain.
REFOLD_STEP_HZ = 250e3

TICK_S = 0.1                     # main-loop period (the sweep is stepped this often)
PROGRESS_EVERY_S = 300.0         # a progress line in the task log every 5 min of a long drift


# ── Pure planner math (no hardware; exercised by --self-test and tests/test_cw_drift.py) ──

def drift_freq(elapsed: float, start: float, end: float, duration: float,
               mode: str) -> float:
    """The emitted frequency at `elapsed` seconds into the drift."""
    if duration <= 0 or end == start:
        return end
    u = elapsed / duration
    if mode == "loop":
        frac = u % 1.0
    elif mode == "pingpong":
        tri = u % 2.0
        frac = tri if tri <= 1.0 else 2.0 - tri
    else:  # once
        frac = min(u, 1.0)
    return start + (end - start) * frac


def plan_lo(f_hz: float, lo_hz: float, half_window_hz: float) -> float:
    """The analog LO to emit f_hz while keeping the baseband offset (f − LO) within
    ±half_window: steps the LO by whole windows (2·half_window) as the sweep crosses a
    window edge. Within a window the LO is fixed and the software NCO carries the offset,
    so only a whole-window hop ever moves the analog LO. (Ported from the X410 cw_channel.)"""
    step = 2.0 * half_window_hz
    while f_hz - lo_hz > half_window_hz:
        lo_hz += step
    while f_hz - lo_hz < -half_window_hz:
        lo_hz -= step
    return lo_hz


def half_window_hz(samp_rate_hz: float) -> float:
    """Half the baseband window the tone may sweep before the LO hops."""
    return SWEEP_MARGIN * samp_rate_hz / 2.0


def is_wide(start: float, end: float, samp_rate_hz: float) -> bool:
    """Whether a start→end drift needs LO hops at this sample rate (else the whole sweep
    fits one baseband window around the drift centre)."""
    return abs(end - start) > 2.0 * half_window_hz(samp_rate_hz)


def initial_lo(start: float, end: float, samp_rate_hz: float) -> float:
    """Where the analog LO starts: the drift CENTRE for a narrow sweep (the NCO carries
    everything), or — for a wide sweep — the window of a centre-based grid that contains the
    start frequency, so the first and last windows sit symmetrically about the sweep."""
    center = 0.5 * (start + end)
    if not is_wide(start, end, samp_rate_hz):
        return center
    return plan_lo(start, center, half_window_hz(samp_rate_hz))


def hop_count(start: float, end: float, samp_rate_hz: float) -> int:
    """How many analog-LO hops one start→end pass takes (0 for a narrow sweep)."""
    if end == start or not is_wide(start, end, samp_rate_hz):
        return 0
    half = half_window_hz(samp_rate_hz)
    lo0 = initial_lo(start, end, samp_rate_hz)
    lo1 = plan_lo(end, lo0, half)
    return int(round(abs(lo1 - lo0) / (2.0 * half)))


def needs_refold(f_hz: float, folded_at_hz: float, step_hz: float = REFOLD_STEP_HZ) -> bool:
    """Whether the calibrated gain should be re-folded for a tone now at f_hz, given the
    frequency it was last folded at."""
    return abs(float(f_hz) - float(folded_at_hz)) >= float(step_hz)


def coverage_gaps(pmap, power_dbm: float, start: float, end: float, samples: int = 41,
                  tol_db: float = 0.3, applied_db=None):
    """Stretches of the sweep where `power_dbm` can't be delivered — the calibration's
    ceiling (a frequency-dependent limit / the flatness) sits below the request, so the gain
    clamps there. Samples the sweep and returns [(f_lo_hz, f_hi_hz, worst_dbm), …] (empty when
    the whole sweep delivers the request within tol_db, or uncalibrated). `applied_db` is the
    pinned attenuator setting the drift runs with (see main), so the check folds the SDR gain
    exactly as the drift will."""
    if not getattr(pmap, "has_absolute", False) or samples < 2:
        return []
    lo, hi = (start, end) if start <= end else (end, start)
    gaps, cur = [], None
    for i in range(samples):
        f = lo + (hi - lo) * i / (samples - 1)
        try:
            g = pmap.gain_for_power(power_dbm, freq=f, applied_db=applied_db)
            got = pmap.power_for_gain(g, freq=f, applied_db=applied_db)
        except Exception:                                   # noqa: BLE001 — skip the sample
            continue
        if power_dbm - got > tol_db:
            if cur is None:
                cur = [f, f, got]
            else:
                cur[1] = f
                cur[2] = min(cur[2], got)
        elif cur is not None:
            gaps.append(tuple(cur))
            cur = None
    if cur is not None:
        gaps.append(tuple(cur))
    return gaps


def fmt_rate(rate_hz_per_s: float) -> str:
    """A readable drift rate: '−27.78 kHz/s (−100 MHz/h)'."""
    r = float(rate_hz_per_s)
    per_h = r * 3600.0
    if abs(r) >= 1e6:
        base = f"{r/1e6:.4g} MHz/s"
    elif abs(r) >= 1e3:
        base = f"{r/1e3:.4g} kHz/s"
    else:
        base = f"{r:.4g} Hz/s"
    if abs(per_h) >= 1e6:
        return f"{base} ({per_h/1e6:.4g} MHz/h)"
    return f"{base} ({per_h/1e3:.4g} kHz/h)"


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(center_freq_hz: float, samp_rate_hz: float, tone_hz: float,
                     gain_db: float, amplitude: float, extra_args: str):
    """A baseband NCO (sig_source_c) mixed up by the USRP LO. Imported lazily so
    the module loads without a radio stack for --describe-params."""
    from gnuradio import gr, analog, blocks, uhd

    class CwDriftTx(gr.top_block):
        def __init__(self):
            super().__init__("CW drift TX")
            self.usrp = uhd.usrp_sink(
                extra_args,
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)

            # Complex exponential at the baseband offset; phase-continuous when
            # set_frequency() is called mid-run (that's what makes the drift smooth).
            self.src = analog.sig_source_c(samp_rate_hz, analog.GR_COS_WAVE,
                                           tone_hz, 1.0, 0)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        # ── live setters (called from the main loop, device-safe) ──────────────
        def set_tone(self, hz: float) -> None:
            self.src.set_frequency(hz)          # continuous-phase frequency change

        def set_center_frequency(self, hz: float) -> None:
            self.usrp.set_center_freq(uhd.tune_request(hz), 0)   # an analog-LO hop

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

    return CwDriftTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    s = (
        Script("CW frequency-drift transmitter — a continuous-wave tone drifting linearly "
               "from a start to an end frequency over a duration: kHz over minutes or hundreds "
               "of MHz over hours/days (baseband NCO within a window, analog-LO hops between "
               "windows, blanked). Calibrated power is re-folded at the live frequency as the "
               "tone moves. For a plain non-drifting tone use cw_tx.py. Transmit only into an "
               "authorised, shielded setup.")
        .number("-Start-frequency", "--freq", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=1575.42, required=True,
                help="Drift START frequency in MHz. Presets are GNSS carriers; any value "
                     "allowed. The --power range shown is folded here; the script re-folds it "
                     "along the sweep.")
        .number("-End-frequency", "--freq_end", unit="MHz", min=70.0, max=6000.0,
                presets=FREQUENCIES, default=1575.43, required=True,
                help="Drift END frequency in MHz — any distance from the start, hundreds of "
                     "MHz included (a span wider than the baseband window is swept window by "
                     "window with blanked LO hops). Equal to --freq gives a static tone — "
                     "but use cw_tx.py for that.")
        .number("-Duration", "--duration", unit="min", min=0.1, max=MAX_DURATION_MIN,
                default=10.0,
                help="Minutes to drift start→end — from a few seconds (0.1 = 6 s) up to 7 days "
                     "(10080). The drift rate is (end − start) / duration; a 'once' drift then "
                     "holds at the end.")
        .choice("-Drift", "--drift", options=["once", "loop", "pingpong"],
                default="once",
                help="once = ramp then hold at the end; loop = repeat start→end; pingpong = "
                     "start→end→start…")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=0.2, max=61.44,
                presets=SAMPLE_RATES_MHZ, default=2.0, required=True,
                help="Host/DAC sample rate. A drift wider than 70 % of it is swept window by "
                     "window with analog-LO hops — a higher rate means fewer hops (300 MHz: "
                     "~215 at 2 MHz, ~43 at 10 MHz) at more USB/CPU load; above ~20 MHz a Pi "
                     "drops samples.")
        .number("-Hop-blank", "--hop_blank", unit="ms", min=0.0, max=500.0,
                default=DEFAULT_HOP_BLANK_MS, live=True,
                help="Wide drift only: mute for this long around each analog-LO hop to hide "
                     "the synthesiser's retune settle. 0 disables blanking. Live.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=True, live=True,
                help="ABSOLUTE power at the delivered plane (dBm), held across the whole sweep "
                     "(re-folded through the calibration at the live frequency). Bounds track "
                     "the unit's calibration when present, else the baked SDR-port scale. "
                     "Ignored if --gain is given (relative wins). Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="off",
                required=False, live=True, is_rf=True,
                help="RF output on/off. Starts OFF (muted pre-roll): set the power, then "
                     "switch ON to go on-air. The drift runs on its own timeline; OFF mutes "
                     "gain AND baseband amplitude; power edits made while OFF are staged and "
                     "applied when you switch ON.")
        .flag("-Restart", "--restart", live=True,
              help="Live trigger (tune-step): restart the drift from the start frequency. "
                   "Fire it to re-run the ramp from the beginning.")
        # RELATIVE power: the SDR's raw TX gain (dB), bypassing the dBm calibration.
        # No default, so its PRESENCE selects relative mode (it overrides --power).
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing the "
                     "dBm calibration (then NOT re-folded along the sweep). When given, "
                     "overrides --power. Live.")
    )
    return s


# ── Self-test: the drift + LO-hop planner math, no hardware ───────────────────────

def _self_test() -> int:
    failures = []

    def check(cond: bool, what: str) -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {what}")
        if not cond:
            failures.append(what)

    S, E, D = 1600e6, 1300e6, 10800.0                       # 300 MHz down over 3 h
    check(abs(drift_freq(0, S, E, D, "once") - S) < 1e-3, "drift starts at start")
    check(abs(drift_freq(D, S, E, D, "once") - E) < 1e-3, "drift 'once' reaches end")
    check(abs(drift_freq(2 * D, S, E, D, "once") - E) < 1e-3, "drift 'once' holds past end")
    check(abs(drift_freq(D / 2, S, E, D, "once") - 0.5 * (S + E)) < 1e-3, "drift is linear")
    check(abs(drift_freq(2 * D, S, E, D, "loop") - S) < 1e-3, "drift 'loop' wraps to start")
    check(abs(drift_freq(D, S, E, D, "pingpong") - E) < 1e-3, "pingpong turns at end")
    check(abs(drift_freq(1.5 * D, S, E, D, "pingpong") - 0.5 * (S + E)) < 1e-3,
          "pingpong comes back through the middle")
    check(abs(drift_freq(5, S, S, D, "once") - S) < 1e-3, "no --freq_end ⇒ pure CW")

    # The LO-hop planner over the full 300 MHz sweep: the baseband offset stays inside
    # ±half_window at every instant, and the hop count matches the closed form.
    for rate_mhz in (2.0, 10.0):
        rate = rate_mhz * 1e6
        half = half_window_hz(rate)
        check(is_wide(S, E, rate), f"300 MHz is a wide drift at {rate_mhz:g} MHz")
        lo = initial_lo(S, E, rate)
        hops, ok = 0, True
        for k in range(0, int(D) + 1, 5):
            f = drift_freq(k, S, E, D, "once")
            new_lo = plan_lo(f, lo, half)
            if new_lo != lo:
                hops += 1
                lo = new_lo
            ok = ok and abs(f - lo) <= half + 1e-6
        check(ok, f"NCO offset stays within ±{half/1e6:.3f} MHz across the sweep at {rate_mhz:g} MHz")
        check(hops == hop_count(S, E, rate),
              f"{hops} LO hops at {rate_mhz:g} MHz match hop_count() = {hop_count(S, E, rate)}")
    check(hop_count(1575.42e6, 1575.43e6, 2e6) == 0 and not is_wide(1575.42e6, 1575.43e6, 2e6),
          "a 10 kHz drift fits one window (no hops)")
    check(needs_refold(1500.3e6, 1500e6) and not needs_refold(1500.1e6, 1500e6),
          "the gain re-folds every 250 kHz of drift")
    check(coverage_gaps(power_map(), -30.0, S, E) == [] or power_map().has_absolute,
          "coverage check is silent when uncalibrated")
    print("SELF-TEST " + ("OK" if not failures else f"FAILED ({len(failures)})"))
    return 0 if not failures else 1


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()

    # --freq / --freq_end are MHz (the params' declared unit); the planner + radio work in Hz.
    start = float(args.freq) * 1e6
    end = float(args.freq_end) * 1e6 if args.freq_end and args.freq_end > 0 else start
    drifting = end != start
    duration_s = float(args.duration) * 60.0    # --duration is MINUTES; the drift law runs in seconds
    span = abs(end - start)
    samp_rate = float(args.sample_rate) * 1e6
    half = half_window_hz(samp_rate)
    wide = drifting and is_wide(start, end, samp_rate)
    lo0 = initial_lo(start, end, samp_rate) if drifting else start
    hops_per_pass = hop_count(start, end, samp_rate)

    # Power map: the unit's injected calibration curve if present (SDR_CALIBRATION_FILE),
    # else it runs uncalibrated — a relative gain only (no baked behaviour).
    pmap = power_map()
    amplitude = pmap.amplitude
    # A raw --gain (relative / calibration knob) overrides the dBm mapping when present.
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    applied = None                                  # the chain's pinned attenuator setting (dB)
    if gain_cal is not None:
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(gain_cal)))
    elif pmap.has_absolute:                         # calibrated: the authored absolute --power
        # Fold the calibration at the drift START — the carrier the agent realizes the chain's
        # active components (an attenuator) at. That SPLIT is pinned for the whole drift: as the
        # tone moves only the SDR gain re-folds (refold below), never the attenuation the agent
        # physically set. None (no active component) folds the SDR alone.
        applied = pmap.pinned_applied(args.power, freq=start)
        gain_db = pmap.gain_for_power(args.power, freq=start, applied_db=applied)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    tb = _build_top_block(lo0, samp_rate, start - lo0, gain_db, amplitude, extra_args="")

    # RF on/off state + the gain RF-on applies. Defaults to --rf off so it starts muted;
    # RF is a pure mute/unmute and does NOT touch the sweep. Power/gain edits made while
    # OFF are staged for the next ON. `power` is the held target (None in raw-gain mode);
    # `fold_f` is the frequency the gain was last folded at.
    state = {"rf_on": getattr(args, "rf", "off") == "on", "gain": gain_db,
             "power": args.power if (pmap.has_absolute and gain_cal is None) else None,
             "applied": applied,
             "freq": start, "lo": lo0, "fold_f": start,
             "blank_s": float(getattr(args, "hop_blank", DEFAULT_HOP_BLANK_MS)) / 1e3,
             "hops": 0}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    rate = (end - start) / duration_s if drifting else 0.0
    print("── CW drift TX ─────────────────────────────────────────────")
    if drifting:
        print(f"  drift          : {start/1e6:.6f} → {end/1e6:.6f} MHz over "
              f"{args.duration:g} min ({args.drift}) · {fmt_rate(rate)}")
        if wide:
            print(f"  sweep mode     : WIDE — {span/1e6:.3f} MHz in ±{half/1e6:.3f} MHz windows, "
                  f"{hops_per_pass} analog-LO hop{'s' if hops_per_pass != 1 else ''} per pass"
                  + (f", {args.hop_blank:g} ms blank each" if state["blank_s"] > 0 else ", unblanked"))
        else:
            print(f"  sweep mode     : narrow — LO fixed at {lo0/1e6:.6f} MHz, the whole "
                  f"±{span/2e6:g} MHz on the baseband NCO (fully continuous)")
    else:
        print(f"  tone           : {start/1e6:.6f} MHz (start == end — static; use cw_tx.py)")
    print(f"  sample rate    : {args.sample_rate:g} MHz")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})"
              + (" — re-folded along the sweep" if drifting and gain_cal is None else ""))
        print(f"  power (achieved on grid): "
              f"{pmap.power_for_gain(gain_db, freq=start, applied_db=applied):.2f} dBm "
              f"at {start/1e6:.3f} MHz"
              + (f" (attenuator pinned at {-applied:g} dB)" if applied else ""))
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.describe()}")
    if pmap.warning:                       # e.g. calibration amplitude != this
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")   # script's fixed amplitude
    if drifting and state["power"] is not None:
        for f_lo, f_hi, worst in coverage_gaps(pmap, state["power"], start, end,
                                               applied_db=state["applied"]):
            print(f"  ⚠ POWER        : {state['power']:g} dBm can't be delivered over "
                  f"{f_lo/1e6:.1f}–{f_hi/1e6:.1f} MHz (ceiling there ≈ {worst:.1f} dBm) — "
                  f"the tone is clamped to the ceiling in that stretch")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted — switch --rf on to unmute)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power (not re-folded)")
    if drifting:
        print("  drift runs on its own timeline from start; --rf is a pure mute, "
              "--restart re-runs the sweep.")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_gain(g: float) -> None:
        state["gain"] = g
        if state["rf_on"]:
            tb.set_gain(g)

    def refold(f: float, force: bool = False) -> None:
        """Re-fold the held --power at frequency f (a frequency-dependent chain moves the
        gain along the sweep); a no-op in raw-gain mode or until the tone has moved enough."""
        if state["power"] is None:
            return
        if not force and not needs_refold(f, state["fold_f"]):
            return
        state["fold_f"] = f
        g = pmap.gain_for_power(state["power"], freq=f, applied_db=state["applied"])
        if abs(g - state["gain"]) > 1e-9:
            apply_gain(g)

    def emit(f: float) -> None:
        """Point the emitted frequency at f. Within a window only the software NCO moves; at
        a window edge the analog LO hops (blanked while RF is on) and the NCO wraps."""
        new_lo = plan_lo(f, state["lo"], half) if wide else state["lo"]
        if new_lo == state["lo"] and f == state["freq"]:
            return                                  # a 'once' drift holding at its end
        if new_lo != state["lo"]:
            blank = state["rf_on"] and state["blank_s"] > 0
            if blank:
                tb.set_amplitude(0.0)
            tb.set_center_frequency(new_lo)
            tb.set_tone(f - new_lo)
            state["lo"] = new_lo
            state["hops"] += 1
            if blank:
                time.sleep(state["blank_s"])
                tb.set_amplitude(amplitude)
        else:
            tb.set_tone(f - state["lo"])
        state["freq"] = f
        refold(f)

    def apply_change(name, value):
        if name == "power":
            # A new level re-picks the SDR/attenuator split at the START frequency — exactly
            # where the agent repositions the attenuator for this tune — then folds the SDR
            # gain at the LIVE frequency with that split pinned. Staged, applied only when RF
            # is on.
            state["power"] = float(value)
            state["fold_f"] = state["freq"]
            state["applied"] = pmap.pinned_applied(state["power"], freq=start)
            apply_gain(pmap.gain_for_power(state["power"], freq=state["freq"],
                                           applied_db=state["applied"]))
            g = tb.actual_gain() if state["rf_on"] else state["gain"]
            ctrl.report("power", round(pmap.power_for_gain(g, freq=state["freq"],
                                                           applied_db=state["applied"]), 2))
        elif name == "gain":
            # Calibration knob: raw TX gain (dB), bypassing the dBm mapping (and the refold).
            state["power"] = None
            apply_gain(max(0.0, min(HW_MAX_GAIN_DB, float(value))))
            ctrl.report("gain", round(tb.actual_gain() if state["rf_on"] else state["gain"], 2))
        elif name == "rf":
            # Pure mute/unmute — does NOT start or restart the sweep (the drift keeps its
            # own timeline; use --restart to re-run it).
            on = str(value).strip().lower() in ("on", "1", "true", "yes")
            state["rf_on"] = on
            if on:
                tb.set_amplitude(amplitude)
                tb.set_gain(state["gain"])
            else:
                tb.set_gain(0.0)
                tb.set_amplitude(0.0)
            ctrl.report("rf", "on" if on else "off")
        elif name == "hop_blank":
            state["blank_s"] = max(0.0, min(0.5, float(value) / 1e3))
            ctrl.report("hop_blank", round(state["blank_s"] * 1e3, 1))
        elif name == "restart" and value:
            nonlocal_t0[0] = time.monotonic()          # re-run the drift from start
            emit(start)
            ctrl.report("restart", True)

    # The drift runs on its own timeline from start — independent of RF. RF on/off is
    # a pure mute; only --restart re-runs the sweep from the start frequency.
    nonlocal_t0 = [time.monotonic() if drifting else None]
    next_progress = time.monotonic() + PROGRESS_EVERY_S

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                apply_change(change.name, change.value)
            t0 = nonlocal_t0[0]
            if t0 is not None and drifting:
                now = time.monotonic()
                emit(drift_freq(now - t0, start, end, duration_s, args.drift))
                if now >= next_progress:
                    next_progress = now + PROGRESS_EVERY_S
                    pct = 100.0 * abs(state["freq"] - start) / span if span else 0.0
                    print(f"  drift @ {state['freq']/1e6:.3f} MHz ({pct:.0f} % of the span) · "
                          f"gain {state['gain']:.2f} dB · LO hops so far {state['hops']}",
                          flush=True)
            time.sleep(TICK_S)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

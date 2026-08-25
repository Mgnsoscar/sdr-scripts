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
# USER CALIBRATION — MEASURE THESE ONCE, THEN EDIT THE VALUES BELOW
# ═══════════════════════════════════════════════════════════════════════════════
# Set the transmit level in dBm. That works only if the script knows how the SDR's
# gain maps to output power, established once with a spectrum analyser: leave AMPLITUDE
# below, run --power at its max (commands GAIN_AT_MAX_DB), measure the port power, and
# put it in OUTPUT_POWER_DBM. CABLE_LOSS_DB / AMPLIFIER_GAIN_DB describe the RF chain
# after the port, so --power is the power delivered at the far end.
OUTPUT_POWER_DBM = -20.0    # max output (dBm) at GAIN_AT_MAX_DB and AMPLITUDE — MEASURE THIS
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands
CABLE_LOSS_DB = 0.0         # cabling insertion loss after the SDR port (positive dB)
AMPLIFIER_GAIN_DB = 0.0     # external amplifier gain after the SDR port (positive dB)

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED (baked levels) with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling

MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else the baked constants above. Cached, so build_script and
    main share one and --power's schema bounds match the real operating range."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
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

# Per-band frequency plan: base carrier + channel spacing (ICD L1/L2, ed. 5.1).
BANDS = {
    "L1": {"base": 1602.0e6, "spacing": 0.5625e6, "chan_default_sr": 10.22e6},
    "L2": {"base": 1246.0e6, "spacing": 0.4375e6, "chan_default_sr": 10.22e6},
}
BAND_DEFAULT_SR = 12.264e6     # BAND-mode default (24 samp/chip, covers ~8 MHz)


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


def build_channel_buffer(samp_rate_hz: float):
    """One-channel real BPSK C/A buffer (carrier already at f_k, so baseband is
    the code at DC). Returns (iq, n_samples, samp_per_chip). Real → I, Q = 0."""
    import numpy as np
    spc = _validate_sr(samp_rate_hz)
    bipolar = (1 - 2 * np.asarray(glonass_ca(), dtype=np.int8)).astype(np.float32)
    n_samples = CODE_LEN * spc                       # one 1 ms code period
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.repeat(bipolar, spc)                # zero-order hold, ±1
    iq.imag = 0.0
    return iq, n_samples, spc


def build_band_buffer(band: str, samp_rate_hz: float):
    """Full-band composite: all 14 channels summed at their frequency offsets
    around the band centre. Each channel carries the same C/A code with a distinct
    cyclic code phase (so the channels are not mutually coherent), frequency-
    shifted to k·spacing. Two code periods (2 ms) so every offset completes a whole
    number of cycles. Returns (iq, n_samples, samp_per_chip)."""
    import numpy as np
    spc = _validate_sr(samp_rate_hz)
    sr = int(round(samp_rate_hz))
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


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class GlonassTx(gr.top_block):
        def __init__(self):
            super().__init__("GLONASS OF TX")
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

    return GlonassTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GLONASS L1OF/L2OF (FDMA) transmitter — one channel or the whole "
               "band, real 511-chip C/A code, file-replay. Transmit only into an "
               "authorised, shielded setup.")
        .choice("-Band", "--band", options=["L1", "L2"], default="L1",
                help="L1OF (~1602 MHz) or L2OF (~1246 MHz). Fixed per run.")
        .choice("-Mode", "--mode", options=["channel", "band"], default="channel",
                help="channel = one satellite on its FDMA carrier; band = all 14 "
                     "channels summed around the band centre. Fixed per run.")
        .integer("-Channel", "--channel", min=K_MIN, max=K_MAX, default=0,
                 help="FDMA channel number k (−7..+6). Sets the carrier in channel "
                      "mode; ignored in band mode. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(power_map().min_power_dbm, 2),
                max=round(power_map().max_power_dbm, 2),
                default=round(power_map().max_power_dbm, 2), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the "
                     "unit's calibration when present (e.g. EIRP), else the baked "
                     "SDR-port scale. Ignored if --gain is given (relative wins). Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False,
                live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them. Change the power (or the "
                     "calibration gain) while OFF and it takes effect when you turn ON.")
        # RELATIVE power (also the calibration knob): raw TX gain (dB), bypassing the dBm
        # mapping. No default, so its PRESENCE selects relative mode and overrides --power.
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=5.11, max=61.44,
                default=0.0,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). "
                     "Leave 0 for the mode default (channel 10.22, band 12.264 MHz). "
                     "Must be a multiple of 0.511 MHz. Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load; the C/A "
                     "signal is constant-modulus so sc8 is ideal (band mode is "
                     "multi-level, sc16 optional there).")
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

    # Resolve sample rate: 0 → mode default.
    if args.samp_rate <= 0:
        samp_rate_hz = BAND_DEFAULT_SR if mode == "band" else BANDS[band]["chan_default_sr"]
    else:
        samp_rate_hz = args.samp_rate * 1e6

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="glonass_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    try:
        if mode == "band":
            iq, nsamp, spc = build_band_buffer(band, samp_rate_hz)
            center = BANDS[band]["base"]
            desc = f"{band}OF band composite (k {K_MIN}..{K_MAX}, 14 channels)"
        else:
            iq, nsamp, spc = build_channel_buffer(samp_rate_hz)
            center = channel_freq(band, args.channel)
            desc = f"{band}OF channel k={args.channel}"
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    iq_path = os.path.join(tmpdir, f"glonass_{band}_{mode}.fc32")
    iq.tofile(iq_path)
    print(f"[prebuilt] {desc} → {nsamp} samples ({spc} samp/chip, "
          f"{nsamp*8/1e6:.1f} MB) → {iq_path}")

    # Power map: the unit's injected calibration curve if present, else the baked
    # constants above. A raw --gain (relative) overrides the dBm mapping when present.
    pmap = power_map()
    amplitude = pmap.amplitude
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else pmap.gain_for_power(args.power)

    tb = _build_top_block(iq_path, center, samp_rate_hz, gain_db, amplitude,
                          args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the flow
    # muted; power/gain edits made while OFF are staged and reach the radio only on --rf on.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── GLONASS OF TX ───────────────────────────────────────────")
    print(f"  signal         : {desc}")
    print(f"  carrier        : {center/1e6:.4f} MHz"
          + ("  (band centre)" if mode == "band" else f"  (channel {args.channel})"))
    print(f"  sample rate    : requested {samp_rate_hz/1e6:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  code           : 511-chip C/A @ 0.511 Mcps (BPSK)")
    print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:                       # calibration measured at another amplitude
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  otw            : {args.otw}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)

    def apply_change(name, value):
        # power/gain edits stage into state["gain"] and reach the radio only when RF is on;
        # --rf mutes/restores gain AND amplitude.
        if name == "power":
            state["gain"] = pmap.gain_for_power(float(value))
            if state["rf_on"]:
                tb.set_gain(state["gain"])
                ctrl.report("power", round(pmap.power_for_gain(tb.actual_gain()), 2))
            else:
                ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
        elif name == "gain":
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

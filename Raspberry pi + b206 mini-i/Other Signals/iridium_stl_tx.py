#!/usr/bin/env python3
"""
Iridium/STL burst SPECTRAL-SURROGATE transmitter for GNU Radio + UHD (B200-mini).

Reproduces the RF/temporal footprint of an Iridium downlink burst in the STL
(Satellite Time & Location) band — DQPSK bursts at 25 ksps with realistic burst
length and idle gaps (TDMA duty cycle), RRC pulse-shaped — over a SURROGATE random
payload, precomputed and replayed from a file (same recipe as the GNSS scripts).

What this is (and is not)
─────────────────────────
STL (Satelles/Iridium) is a commercial PNT service whose burst PAYLOAD is
proprietary and encrypted; it is not published and is not reproduced here. STL is
also NOT a PRN/CDMA GNSS signal — it rides on Iridium's TDMA/FDMA L-band bursts.
This transmits a SPECTRAL SURROGATE: the correct Iridium physical-layer modulation
(BPSK preamble + differential-QPSK payload, 25 ksps, RRC shaping, bursted with
realistic idle gaps) over a PUBLIC pseudo-random payload standing in for the
encrypted STL content. It reproduces the STL-band spectral + burst footprint for
conducted front-end / filter / band-occupancy / RFI testing, and is noise-like to
a real STL receiver — it carries none of the STL data.

The Iridium PHY (25 ksps DQPSK bursts, ~41.67 kHz raster, 1616–1626.5 MHz) is the
public reverse-engineered layer (gr-iridium / Iridium Toolkit); STL rides on the
simplex/broadcast channels near 1626 MHz.

⚠  RF SAFETY / LEGAL: the Iridium band is licensed. Transmit ONLY into a shielded /
   conducted setup (cable + attenuators) you are LICENSED / AUTHORISED to use —
   never over the air.

Live tuning: gain and amplitude (instant). Frequency / burst parameters / sample
rate are fixed per run (restart to change them).

CLI
───
    iridium_stl_tx.py --freq 1626.270833e6 --gain 45
    iridium_stl_tx.py --self-test        # modulation + duty + occupancy, no hardware
    iridium_stl_tx.py --describe-params  # paramkit JSON schema for the GUI
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
CAL_SIGNAL_ID = "iridium_stl"


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

SYMBOL_RATE_HZ = 25_000        # Iridium burst symbol rate (25 ksps)
CHANNEL_SPACING_HZ = 41_667    # FDMA channel raster (reference only)

FREQUENCIES = {
    "Iridium Ring Alert (1626.270833 MHz)": 1626.270833e6,
    "Iridium/STL simplex band centre (1626.25 MHz)": 1626.25e6,
    "Iridium duplex band centre (1621.25 MHz)": 1621.25e6,
}
SAMPLE_RATES_MHZ = {"1 MHz (default)": 1.0, "2 MHz": 2.0, "0.5 MHz": 0.5}


# ── Burst synthesis (BPSK preamble + DQPSK payload, RRC-shaped) ────────────────

def _rrc(t, beta: float):
    """Root-raised-cosine impulse response, `t` in symbol periods (array)."""
    import numpy as np
    out = np.zeros_like(t, dtype=np.float64)
    z = np.isclose(t, 0.0)
    out[z] = 1.0 - beta + 4.0 * beta / np.pi
    if beta > 0:
        sing = np.isclose(np.abs(t), 1.0 / (4.0 * beta))
        out[sing] = (beta / np.sqrt(2.0)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    else:
        sing = np.zeros_like(t, dtype=bool)
    m = ~(z | sing)
    tm = t[m]
    num = np.sin(np.pi * tm * (1 - beta)) + 4 * beta * tm * np.cos(np.pi * tm * (1 + beta))
    den = np.pi * tm * (1 - (4 * beta * tm) ** 2)
    out[m] = num / den
    return out


def _burst_symbols(preamble_n: int, payload_n: int, rng):
    """One burst's complex unit symbols: BPSK alternating preamble (sync tone) then
    a differentially-encoded QPSK random payload."""
    import numpy as np
    pre = np.where(np.arange(preamble_n) % 2 == 0, 1.0, -1.0).astype(np.complex128)
    d = rng.integers(0, 4, size=payload_n)
    phase = np.cumsum(d) * (np.pi / 2.0)
    return np.concatenate([pre, np.exp(1j * phase)])


def build_iridium_buffer(samp_rate_hz: float, preamble_n: int, payload_n: int,
                         burst_period_ms: float, num_frames: int, rolloff: float,
                         seed: int):
    """Complex64 buffer of `num_frames` Iridium-style bursts (one per burst_period),
    RRC-shaped, separated by idle gaps. Returns (iq, n_samples, burst_ms, duty)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    spsym = samp_rate_hz / SYMBOL_RATE_HZ
    frame_samps = int(round(burst_period_ms * 1e-3 * samp_rate_hz))
    total = frame_samps * num_frames
    out = np.zeros(total, dtype=np.complex128)
    span = 8

    for fr in range(num_frames):
        syms = _burst_symbols(preamble_n, payload_n, rng)
        base = fr * frame_samps
        for k, s in enumerate(syms):
            center = base + k * spsym
            i0 = max(0, int(np.floor(center - span * spsym)))
            i1 = min(total, int(np.ceil(center + span * spsym)) + 1)
            idx = np.arange(i0, i1)
            out[idx] += s * _rrc((idx - center) / spsym, rolloff)

    peak = float(np.max(np.abs(out))) or 1.0
    iq = (out * (0.9 / peak)).astype(np.complex64)
    burst_ms = (preamble_n + payload_n) / SYMBOL_RATE_HZ * 1e3
    return iq, total, burst_ms, burst_ms / burst_period_ms


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    import numpy as np
    ok = True
    t = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
    r = _rrc(t, 0.4)
    rrc_ok = abs(r[0] - (1 - 0.4 + 4 * 0.4 / np.pi)) < 1e-9 and np.allclose(r[1], r[2])
    ok = ok and rrc_ok
    print(f"RRC: peak={r[0]:.4f} symmetric={bool(np.allclose(r[1],r[2]))} [{'OK' if rrc_ok else 'FAIL'}]")

    syms = _burst_symbols(64, 256, np.random.default_rng(1))
    mag_ok = np.allclose(np.abs(syms), 1.0)
    ok = ok and mag_ok
    print(f"symbols: {len(syms)} unit-magnitude={mag_ok} [{'OK' if mag_ok else 'FAIL'}]")

    iq, n, burst_ms, duty = build_iridium_buffer(1.0e6, 64, 256, 90.0, 4, 0.4, 7)
    active = float(np.mean(np.abs(iq) > 0.02))
    duty_ok = abs(active - duty) < 0.06
    ok = ok and duty_ok
    print(f"buffer: {n} samples, burst {burst_ms:.1f} ms, duty {duty*100:.1f}% "
          f"(measured {active*100:.1f}%) [{'OK' if duty_ok else 'FAIL'}]")

    one = iq[:int(round(1.0e6 * 90e-3))]
    S = np.abs(np.fft.fftshift(np.fft.fft(one))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(one), 1.0 / 1.0e6))
    inband = S[np.abs(f) <= 25e3].sum() / S.sum()
    bw_ok = inband > 0.98
    ok = ok and bw_ok
    print(f"occupancy: {inband*100:.1f}% within ±25 kHz (25 ksps RRC) [{'OK' if bw_ok else 'FAIL'}]")

    print("ALL IRIDIUM/STL SURROGATE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_file: str, center_freq_hz: float, samp_rate_hz: float,
                     gain_db: float, amplitude: float, extra_args: str):
    from gnuradio import gr, blocks, uhd

    class IridiumTx(gr.top_block):
        def __init__(self):
            super().__init__("Iridium/STL surrogate TX")
            self.usrp = uhd.usrp_sink(
                extra_args,
                uhd.stream_args(cpu_format="fc32", otw_format="sc16", channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            self.usrp.set_center_freq(uhd.tune_request(center_freq_hz), 0)
            self.usrp.set_gain(gain_db, 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, iq_file, repeat=True)
            self.amp = blocks.multiply_const_cc(amplitude)
            self.connect(self.src, self.amp, self.usrp)

        def set_amplitude(self, a: float) -> None:
            self.amp.set_k(a)

        def set_gain(self, g: float) -> None:
            self.usrp.set_gain(g, 0)

        def actual_gain(self) -> float:
            return self.usrp.get_gain(0)

    return IridiumTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Iridium/STL burst spectral-surrogate transmitter — realistic DQPSK "
               "bursts (25 ksps, RRC, TDMA duty) over a PUBLIC random payload (NOT "
               "the proprietary STL content). Transmit only into an authorised, "
               "shielded setup.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=1626.270833e6, required=True,
                help="Channel carrier in the Iridium/STL band.")
        .integer("-Payload-symbols", "--payload_symbols", min=16, max=4000, default=256,
                 help="DQPSK payload symbols per burst.")
        .integer("-Preamble-symbols", "--preamble_symbols", min=8, max=512, default=64,
                 help="BPSK preamble (sync tone) symbols per burst.")
        .number("-Burst-period", "--burst_period", unit="ms", min=1.0, max=1000.0,
                default=90.0, help="Time between bursts (Iridium frame ≈ 90 ms) — "
                     "sets the TDMA duty cycle.")
        .integer("-Frames", "--frames", min=1, max=64, default=8,
                 help="Independent random-payload frames in the looped file.")
        .number("-Rolloff", "--rolloff", min=0.05, max=1.0, default=0.4,
                help="RRC pulse-shaping roll-off.")
        .number("-Sample-rate", "--sample_rate", unit="MHz", min=0.2, max=61.44,
                presets=SAMPLE_RATES_MHZ, default=1.0, required=True,
                help="Host/DAC sample rate. A channel is ~35 kHz wide; ~1 MHz plenty.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE in-burst power at the delivered plane (dBm). Bounds track "
                     "the unit's calibration when present, else the baked SDR-port scale. "
                     "Ignored if --gain is given (relative wins). Live.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False,
                live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them (this replaces the old 'raise "
                     "amplitude on-air' control). Change the power while OFF and it "
                     "takes effect when you turn ON.")
        # RELATIVE power (also the calibration knob): raw TX gain (dB), bypassing the dBm
        # mapping. No default, so its PRESENCE selects relative mode and overrides --power.
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, "
                     "bypassing the dBm calibration. When given, overrides --power. Live.")
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
    samp_rate = args.sample_rate * 1e6

    iq, n, burst_ms, duty = build_iridium_buffer(
        samp_rate, args.preamble_symbols, args.payload_symbols, args.burst_period,
        args.frames, args.rolloff, seed=1)

    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="iridium_stl_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    iq_file = os.path.join(tmpdir, "burst.fc32")
    iq.tofile(iq_file)

    # Power map: the unit's injected calibration curve if present, else the baked
    # constants above. A raw --gain (relative) overrides the dBm mapping when present.
    pmap = power_map()
    amplitude = pmap.amplitude
    gain_cal = getattr(args, "gain", None)          # explicit --gain: a hard bench override
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif power_map().has_absolute:                  # calibrated: the authored absolute --power
        gain_db = power_map().gain_for_power(args.power)
    else:                                           # uncalibrated: a persisted fallback gain, or refuse
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    tb = _build_top_block(iq_file, args.freq, samp_rate, gain_db, amplitude,
                          extra_args="")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the flow
    # muted; power/gain edits made while OFF are staged and reach the radio only on --rf on.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── Iridium/STL surrogate TX ────────────────────────────────")
    print(f"  carrier        : {args.freq/1e6:.6f} MHz  (SURROGATE payload)")
    print(f"  burst          : {args.preamble_symbols}+{args.payload_symbols} sym "
          f"({burst_ms:.1f} ms) every {args.burst_period:g} ms → {duty*100:.0f}% duty")
    print(f"  sample rate    : {args.sample_rate:g} MHz  ({n} samples, {args.frames} frames)")
    if power_map().has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({power_map().label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), "
          f"amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:                       # calibration measured at another amplitude
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
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

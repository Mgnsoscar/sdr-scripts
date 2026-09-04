#!/usr/bin/env python3
"""
GPS L1 P-code transmitter — the PUBLIC (unencrypted) precision code, streamed in real
time and aligned to the current GPS time-of-week.

Why streaming (not the prebuild-and-loop the other PRN scripts use): the P-code period is
one week (6.187104e12 chips at 10.23 Mcps), so it can neither be prebuilt nor looped. The
"smart way" that fits a Pi + SDR:

  • the four component registers (X1A/X1B/X2A/X2B) short-cycle at 4092/4093 chips, so the
    two 1.5 s master sequences X1 and X2 are precomputed ONCE by tiling those ~4 kchip base
    patterns (with the ICD's 343/37/380-chip end-of-epoch holds); ~30 MB, built in ~1 s.
  • any window of the week is then `P_i[c] = X1[c mod L1] XOR X2[(c-i) mod L2]` — two
    modular gathers + an XOR, O(n), far faster than real time (no LFSR in the hot path).
  • a producer thread generates GPS-time-aligned IQ chunks and streams them into a named
    pipe (FIFO) that a GNU Radio `blocks.file_source` reads into `uhd.usrp_sink` — the same
    proven device path the C/A script uses, no custom block. (UHD's standalone Python
    bindings can't reliably enumerate a USB B2xx in this image; gnuradio's usrp_sink can.)
    The FIFO + the radio's own buffering ride out scheduler jitter.

The generator is validated bit-exact against IS-GPS-200N (§3.3.2.2, Tables 3-Ia / 3-VII):
`--self-test` checks the four register code vectors, maximal-length taps, the tiling/hold
seams, and the P-code first-12-chips octal per PRN. Run it before trusting a deployment.

Fidelity notes: start phase is set from the PC clock's GPS time-of-week — coarse (~ms,
NTP-limited), and the SDR's TCXO then free-runs, so this is a structurally-true P-code that
starts in the right region of the week, not a chip-true replica disciplined to GPS time.
The once-per-week end-of-week X2 hold (Table 3-VI) is not modelled — a brief glitch at the
week rollover only; irrelevant mid-week.

CLI
───
    gps_l1p_tx.py --prn 5 --power -30            # absolute dBm (calibrated), L1 P-code
    gps_l1p_tx.py --prn 5 --gain 60             # relative: raw SDR gain
    gps_l1p_tx.py --self-test                   # validate the generator (no hardware)
    gps_l1p_tx.py --dry-run --prn 5 --gain 60   # exercise the streaming path, no radio
    gps_l1p_tx.py --describe-params             # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time

os.environ.setdefault("UHD_LOG_CONSOLE_LEVEL", "off")
os.environ.setdefault("UHD_LOG_FASTPATH_DISABLE", "1")   # no "UUUU" underflow spam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script, PowerMap

try:
    import numpy as np                       # core to generation; kept optional so
except ImportError:                          # --describe-params works without it
    np = None

# ── calibration identity + RF chain limits (mirrors the other PRN scripts) ──────────
CAL_SIGNAL_ID = "gps_l1_p"
GAIN_AT_MAX_DB = 89.75       # operating gain ceiling (also the hard cap the script obeys)
HW_MAX_GAIN_DB = 89.75       # B200-mini physical TX-gain ceiling
AMPLITUDE = 0.5              # FIXED baseband amplitude the calibration is measured at

# ── signal constants (fixed — this IS GPS L1 P-code) ────────────────────────────────
SIGNAL_NAME = "GPS L1 P"
CARRIER_HZ = 1575.42e6                       # L1
CHIP_RATE_HZ = 10_230_000.0
# Sample-rate menu — integer samples/chip so upsampling is an exact repeat. {value: label}.
SAMPLE_RATE_CHOICES = {
    "20.46": "2 samples/chip (20.46 MHz)",
    "30.69": "3 samples/chip (30.69 MHz)",
}
DEFAULT_SAMPLE_RATE = "20.46"

# GPS-UTC offset (leap seconds). GPS has no leap seconds; update when a new one is added.
GPS_UTC_LEAP_SECONDS = 18
GPS_UNIX_EPOCH = 315_964_800                  # Unix time of 1980-01-06 00:00:00 UTC
SECONDS_PER_WEEK = 604_800

# ── Spectral-density calibration (docs/calibration-v2.md §13, sdr-agent) ─────────────
# The GPS L1 P(Y) code is a BPSK(10) sinc² spectrum, so its whole power distribution is fixed by
# ONE measured number — the power spectral DENSITY at the main-lobe PEAK (the carrier), in dBm/Hz
# (per Hz, NOT per MHz). From that single number CAL_POWER_LAWS derives two absolute-power
# quantities the operator can pick for --power (in the calibration editor); the measured density
# itself (dBm/Hz) stays available as a third:
#
#   • Main-lobe integrated power (dBm)   = peak_dBm/Hz + 10·log10(Rc · I_ML)   ← ±10.23 MHz
#   • Carrier (total signal) power (dBm) = peak_dBm/Hz + 10·log10(Rc)          ← all frequency
#
# Rc = 10.23e6 Hz (chip rate); I_ML = 0.902823 is the sinc² power fraction inside the main lobe
# (±Rc). This code streams (no passband filter), and at the fixed sample rate only ~the main lobe
# is representable, so the EMITTED power never exceeds the carrier (total) reading — making the
# carrier power the safe amplifier-limiting quantity. Both quantities are bandwidth-independent
# constants. (--self-test recomputes both from ∫sinc² and asserts these literals.)
I_ML = 0.902823                              # sinc² power fraction within the main lobe (±Rc)

CAL_POWER_LAWS = [
    {"id": "main_lobe_power", "name": "Main-lobe integrated power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 69.654784},    # 10·log10(Rc · I_ML), Rc = 10.23e6
    {"id": "carrier_power", "name": "Carrier (total signal) power", "unit": "dBm",
     "in": "density", "out": "abs", "k": 70.098756},     # 10·log10(Rc), Rc = 10.23e6
]

_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected L1 P-code calibration if present, else an
    uncalibrated (relative-gain-only) map. Cached so build_script and main agree."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.uncalibrated(0.0, GAIN_AT_MAX_DB, AMPLITUDE))
    return _PMAP


# ═══════════════════════════════════════════════════════════════════════════════════
# Public P-code generator — inlined & validated bit-exact against IS-GPS-200N §3.3.2.2.
# ═══════════════════════════════════════════════════════════════════════════════════
L1A, L1B = 4092, 4093
L2A, L2B = 4092, 4093
X1_PERIOD = 15_345_000                         # 3750 * 4092 = 1.5 s
X2_PERIOD = 15_345_037                        # X1_PERIOD + 37 (the weekly-slip mechanism)
WEEK_CHIPS = X1_PERIOD * 403_200              # 6.187104e12 chips = 1 week

X1A_TAPS = (6, 8, 11, 12)
X1B_TAPS = (1, 2, 5, 8, 9, 10, 11, 12)
X2A_TAPS = (1, 3, 4, 5, 7, 8, 9, 10, 11, 12)
X2B_TAPS = (2, 3, 4, 8, 9, 12)
# ICD code vectors: leftmost bit = stage 12 (current output) … stage 1. Reading a code
# vector left→right is the register's first 12 outputs.
X1A_INIT = "001001001000"
X1B_INIT = "010101010100"
X2A_INIT = "100100100101"
X2B_INIT = "010101010100"
# Table 3-Ia "First 12 Chips Octal — P": independent per-PRN reference for the self-test.
PCODE_FIRST12_OCTAL = {1: "4444", 2: "4000", 3: "4222", 4: "4333", 5: "4377",
                       6: "4355", 7: "4344", 8: "4340", 9: "4342", 10: "4343"}


def _lfsr(taps, init: str, n: int):
    """n output chips of a 12-stage Fibonacci LFSR (IS-GPS-200 §3.3.2.2): output = stage 12;
    feedback = XOR of tapped stages into stage 1; shift lower→higher. ``init`` is the ICD
    code vector (stage 12 … stage 1), reversed here into stage order."""
    reg = [int(b) for b in init[::-1]]
    out = np.empty(n, dtype=np.uint8)
    for k in range(n):
        out[k] = reg[11]
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:11]
    return out


def _period_seq(pat, full_cycles: int, total_len: int):
    """Tile a short base pattern, then HOLD its last output to pad to total_len (a stopped
    register holds its last output) — the ICD's end-of-epoch behaviour."""
    body = np.tile(pat, full_cycles)
    hold = np.full(total_len - body.size, pat[-1], dtype=np.uint8)
    return np.concatenate([body, hold])


class PCode:
    """Streaming public P-code source: build once, then pull arbitrary week-aligned chunks."""

    def __init__(self):
        if np is None:
            raise RuntimeError("numpy is required to generate the P-code")
        x1a = _period_seq(_lfsr(X1A_TAPS, X1A_INIT, L1A), 3750, X1_PERIOD)
        x1b = _period_seq(_lfsr(X1B_TAPS, X1B_INIT, L1B), 3749, X1_PERIOD)   # +343 hold
        x2a = _period_seq(_lfsr(X2A_TAPS, X2A_INIT, L2A), 3750, X2_PERIOD)   # +37 hold
        x2b = _period_seq(_lfsr(X2B_TAPS, X2B_INIT, L2B), 3749, X2_PERIOD)   # +380 hold
        self.x1 = x1a ^ x1b
        self.x2 = x2a ^ x2b

    def chunk(self, prn: int, start_chip: int, n: int):
        """`n` P-code chips for `prn` from week chip-index `start_chip`: two modular gathers
        + XOR. Returns uint8 {0,1}. P_i[c] = X1[c mod L1] ^ X2[(c-i) mod L2]."""
        c = start_chip + np.arange(n, dtype=np.int64)
        return self.x1[np.mod(c, X1_PERIOD)] ^ self.x2[np.mod(c - prn, X2_PERIOD)]


def gps_time_of_week(now_unix: float | None = None) -> tuple[int, float]:
    """(week, time_of_week_seconds) from the PC clock. GPS = UTC + leap seconds; coarse
    (NTP-limited)."""
    now_unix = time.time() if now_unix is None else now_unix
    gps = (now_unix - GPS_UNIX_EPOCH) + GPS_UTC_LEAP_SECONDS
    return int(gps // SECONDS_PER_WEEK), gps % SECONDS_PER_WEEK


def start_chip_for_now() -> int:
    _, tow = gps_time_of_week()
    return int(round(tow * CHIP_RATE_HZ)) % WEEK_CHIPS


# ── parameter schema ────────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script(f"{SIGNAL_NAME} (public precision code) transmitter — real-time streamed, "
               "aligned to the current GPS time-of-week. Level is set in dBm via the unit's "
               "calibration; uncalibrated it runs on a relative gain.")
        .integer("-PRN", "--prn", min=1, max=37, default=1, required=True,
                 help="GPS satellite PRN (1..37). Selects the X2 delay. Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                **power_map().power_field_kwargs(), required=False, live=True,
                help="ABSOLUTE power at the delivered plane (dBm). Bounds track the unit's "
                     "L1 P-code calibration when present. Ignored if --gain is given. Live.")
        .choice("-Sample-rate", "--samp_rate", options=SAMPLE_RATE_CHOICES,
                default=DEFAULT_SAMPLE_RATE,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). Both "
                     "options are integer samples/chip so the P-code main lobe (±10.23 MHz) "
                     "is represented exactly. Fixed per run.")
        .number("-Analog-bandwidth", "--bandwidth", unit="MHz", min=0.0, max=56.0,
                default=0.0, required=False,
                help="AD9361 analog TX filter bandwidth (MHz) — the real baseband LPF, set "
                     "independently of the sample rate. 0 = let UHD pick it from the sample "
                     "rate (the old behaviour). Set it to band-limit the transmitted signal "
                     "WITHOUT changing the master clock (e.g. 20.46 to pass just the P-code "
                     "main lobe). Coerced to the radio's range; the banner reports the actual.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load (helps at 20+ MS/s "
                     "on a Pi); sc16 for more dynamic range.")
        .choice("-RF", "--rf", options=["on", "off"], default="on", required=False, live=True,
                help="RF output on/off. OFF mutes the baseband amplitude to 0; ON restores "
                     "it. Live.")
        .number("-Duration", "--duration", unit="s", min=0.0, max=604800.0, default=0.0,
                required=False, help="Stop after this many seconds. 0 = run until stopped.")
        .number("-Gain", "--gain", unit="dB", min=0, max=HW_MAX_GAIN_DB,
                required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing the "
                     "dBm calibration. When given, overrides --power. Live.")
    )


# ── self-test: validate the generator against IS-GPS-200N (no hardware) ─────────────

def _self_test() -> int:
    if np is None:
        print("numpy required for --self-test", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    pc = PCode()
    print(f"masters built: X1 {pc.x1.size:,}, X2 {pc.x2.size:,} chips "
          f"({time.perf_counter()-t0:.2f}s)")
    assert pc.x1.size == X1_PERIOD and pc.x2.size == X2_PERIOD
    assert X1_PERIOD == 3750 * L1A and X2_PERIOD == X1_PERIOD + 37
    assert WEEK_CHIPS == 6_187_104_000_000
    for name, taps, cv, L in (("X1A", X1A_TAPS, X1A_INIT, L1A), ("X1B", X1B_TAPS, X1B_INIT, L1B),
                              ("X2A", X2A_TAPS, X2A_INIT, L2A), ("X2B", X2B_TAPS, X2B_INIT, L2B)):
        assert "".join(map(str, _lfsr(taps, cv, 12))) == cv, f"{name} code vector"
        s = _lfsr(taps, cv, 4095 * 2)
        assert np.array_equal(s[:4095], s[4095:]), f"{name} not maximal-length"
    print("register code vectors + maximal-length taps ✓")
    for prn, octal in PCODE_FIRST12_OCTAL.items():
        got = format(int("".join(map(str, pc.chunk(prn, 0, 12))), 2), "04o")
        assert got == octal, f"PRN {prn}: {got} != ICD {octal}"
    print(f"P-code first-12 chips match IS-GPS-200N Table 3-Ia (PRN 1..10) ✓")

    # Calibration law constants: recompute I_ML from ∫sinc² and assert the CAL_POWER_LAWS literals.
    def _sinc2(x):
        if x == 0.0:
            return 1.0
        s = math.sin(math.pi * x) / (math.pi * x)
        return s * s
    step, acc, prev = 1e-3, 0.0, _sinc2(0.0)
    for i in range(1, int(round(1.0 / step)) + 1):
        cur = _sinc2(i * step); acc += 0.5 * (prev + cur) * step; prev = cur
    i_ml = 2.0 * acc
    Rc = CHIP_RATE_HZ
    main_k, carrier_k = 10 * math.log10(Rc * i_ml), 10 * math.log10(Rc)
    laws = {l["id"]: l["k"] for l in CAL_POWER_LAWS}
    assert abs(i_ml - I_ML) < 5e-4
    assert abs(laws["main_lobe_power"] - main_k) < 5e-3
    assert abs(laws["carrier_power"] - carrier_k) < 5e-3
    print(f"calibration: I_ML={i_ml:.6f}, main-lobe k={main_k:.4f} (law {laws['main_lobe_power']}), "
          f"carrier k={carrier_k:.4f} (law {laws['carrier_power']}) ✓")
    print("SELF-TEST OK")
    return 0


# ── IQ generation ──────────────────────────────────────────────────────────────────

def _samples_per_chip(samp_rate_hz: float) -> int:
    spc = round(samp_rate_hz / CHIP_RATE_HZ)
    if abs(spc * CHIP_RATE_HZ - samp_rate_hz) > 1.0:
        raise ValueError(f"sample rate {samp_rate_hz} is not an integer multiple of the "
                         f"10.23 MHz chip rate")
    return spc


def _make_iq(pc: "PCode", prn: int, start_chip: int, n_chips: int, spc: int, amp: float):
    """BPSK the P-code chips and upsample to `spc` samples/chip → complex64 baseband.
    chip 0 → +amp, chip 1 → -amp (sign convention is arbitrary but consistent)."""
    bits = pc.chunk(prn, start_chip, n_chips)
    sym = (1.0 - 2.0 * bits.astype(np.float32)) * amp
    return np.repeat(sym, spc).astype(np.complex64)


# ── entry point ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    dry_run = "--dry-run" in sys.argv[1:]
    if dry_run:                                   # not a paramkit param — strip before parse
        sys.argv = [a for a in sys.argv if a != "--dry-run"]

    script = build_script()
    args = script.parse()

    if np is None:
        print("numpy is required to run", file=sys.stderr)
        return 2

    samp_rate_hz = float(args.samp_rate) * 1e6
    spc = _samples_per_chip(samp_rate_hz)
    pmap = power_map()
    amplitude = pmap.amplitude

    # gain precedence: explicit --gain > calibrated --power > persisted fallback > refuse.
    gain_cal = getattr(args, "gain", None)
    if gain_cal is not None:
        gain_db = float(gain_cal)
    elif pmap.has_absolute:
        gain_db = pmap.gain_for_power(args.power)
    else:
        _fb = os.environ.get("SDR_CAL_FALLBACK_GAIN")
        if _fb is None:
            print("error: this signal is not calibrated on this unit — absolute --power (dBm) "
                  "has no meaning here; set a relative gain (the client does this for you).",
                  file=sys.stderr)
            return 2
        gain_db = max(0.0, min(HW_MAX_GAIN_DB, float(_fb)))

    prn = int(args.prn)
    week, tow = gps_time_of_week()
    start_chip = int(round(tow * CHIP_RATE_HZ)) % WEEK_CHIPS

    print(f"── {SIGNAL_NAME} TX ─────────────────────────────────────────")
    print(f"  PRN            : {prn}   (X2 delay {prn} chips)")
    print(f"  carrier        : {CARRIER_HZ/1e6:.3f} MHz (L1)")
    print(f"  chip rate      : 10.230 Mcps   sample rate {samp_rate_hz/1e6:g} MHz "
          f"({spc} samp/chip)")
    print(f"  GPS start      : week {week}, TOW {tow:.3f}s → chip {start_chip:,} of the week")
    print(f"                   (coarse PC-clock alignment; SDR TCXO free-runs after start)")
    if pmap.has_absolute:
        print(f"  power (target) : {args.power:g} dBm  ({pmap.label})")
    print(f"  → gain         : {gain_db:.2f} dB (max {pmap.max_gain_db:g}), amplitude {amplitude:g}")
    print(f"  calibration    : {pmap.source}")
    if pmap.warning:
        print(f"  ⚠ CALIBRATION  : {pmap.warning}")
    if gain_cal is not None:
        print(f"  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")

    t0 = time.perf_counter()
    pc = PCode()
    print(f"  masters built  : {pc.x1.size:,} + {pc.x2.size:,} chips in "
          f"{time.perf_counter()-t0:.2f}s")
    print("────────────────────────────────────────────────────────────")

    # shared state (live-tunable). amplitude is applied by the flowgraph, so the producer
    # always generates FULL-SCALE ±1 chips — --rf/level changes never re-generate buffers.
    state = {"rf_on": getattr(args, "rf", "on") == "on",
             "gain": gain_db, "stop": False}
    CHUNK_CHIPS = 102_300               # 0.01 s of chips per buffer

    stop_evt = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_evt.set())
    signal.signal(signal.SIGINT, lambda *_: stop_evt.set())

    ctrl = script.live_control(args)

    def apply_live():
        """Drain live edits into `state` (gain/rf). Returns True if anything changed."""
        changed = False
        for ch in ctrl.drain():
            if ch.name == "power" and pmap.has_absolute:
                state["gain"] = pmap.gain_for_power(float(ch.value)); changed = True
                ctrl.report("power", round(pmap.power_for_gain(state["gain"]), 2))
            elif ch.name == "gain":
                state["gain"] = max(0.0, min(HW_MAX_GAIN_DB, float(ch.value))); changed = True
                ctrl.report("gain", round(state["gain"], 2))
            elif ch.name == "rf":
                state["rf_on"] = str(ch.value).strip().lower() in ("on", "1", "true", "yes")
                changed = True
                ctrl.report("rf", "on" if state["rf_on"] else "off")
        return changed

    duration = float(getattr(args, "duration", 0.0) or 0.0)
    deadline = (time.monotonic() + duration) if duration > 0 else None

    if dry_run:
        import queue
        q: "queue.Queue" = queue.Queue(maxsize=20)

        def producer_q():
            c = start_chip
            while not state["stop"]:
                try:
                    q.put(_make_iq(pc, prn, c, CHUNK_CHIPS, spc, 1.0), timeout=1.0)
                    c += CHUNK_CHIPS
                except queue.Full:
                    if state["stop"]:
                        break
        prod = threading.Thread(target=producer_q, daemon=True); prod.start()
        return _dry_run_consumer(q, state, stop_evt, apply_live, deadline,
                                 samp_rate_hz, spc, prod)

    # ── real hardware: GNU Radio usrp_sink (the fleet's proven device path) fed by a FIFO ──
    # UHD's standalone Python bindings can't always enumerate a USB B2xx in this image, but
    # gnuradio's usrp_sink opens it exactly as the C/A script does. The Python producer streams
    # generated IQ into a named pipe that blocks.file_source reads — no custom block, no loop.
    import fcntl
    import tempfile
    from gnuradio import gr, blocks, uhd

    tmpdir = tempfile.mkdtemp(prefix="gps_l2p_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    fifo_path = os.path.join(tmpdir, "iq.fifo")
    os.mkfifo(fifo_path)

    def producer_fifo():
        c = start_chip
        try:
            fd = os.open(fifo_path, os.O_WRONLY)          # blocks until file_source opens read end
        except OSError:
            return
        try:
            fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, 1 << 20)  # ~1 MB pipe buffer (best effort)
        except (OSError, AttributeError):
            pass
        try:
            while not stop_evt.is_set() and not state["stop"]:
                iq = _make_iq(pc, prn, c, CHUNK_CHIPS, spc, 1.0)   # full-scale ±1
                c += CHUNK_CHIPS
                os.write(fd, iq.tobytes())
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    prod = threading.Thread(target=producer_fifo, daemon=True)
    prod.start()                                          # opens the write end (blocks until read end)

    class _PTx(gr.top_block):
        def __init__(self):
            super().__init__(f"{SIGNAL_NAME} TX")
            dev = (f"master_clock_rate={samp_rate_hz:.0f},"
                   "num_send_frames=512,send_frame_size=16000")
            extra = os.environ.get("SDR_UHD_ARGS", "")
            if extra:
                dev += "," + extra
            self.usrp = uhd.usrp_sink(
                dev, uhd.stream_args(cpu_format="fc32", otw_format=args.otw, channels=[0]))
            self.usrp.set_samp_rate(samp_rate_hz)
            _bw = float(getattr(args, "bandwidth", 0.0) or 0.0)
            if _bw > 0:                                   # AD9361 analog LPF, independent of Fs
                self.usrp.set_bandwidth(_bw * 1e6, 0)
            self.usrp.set_center_freq(uhd.tune_request(CARRIER_HZ), 0)
            self.usrp.set_gain(state["gain"], 0)
            self.src = blocks.file_source(gr.sizeof_gr_complex, fifo_path, repeat=False)
            self.amp = blocks.multiply_const_cc(amplitude if state["rf_on"] else 0.0)
            self.connect(self.src, self.amp, self.usrp)

        def set_gain(self, g):
            self.usrp.set_gain(g, 0)

        def set_amplitude(self, a):
            self.amp.set_k(a)

    tb = _PTx()                                           # file_source opens the read end → producer unblocks
    tb.start()
    try:
        _auto = " — auto from Fs" if not float(getattr(args, "bandwidth", 0.0) or 0.0) else ""
        print(f"  analog TX BW   : {tb.usrp.get_bandwidth(0)/1e6:.3f} MHz (AD9361 filter{_auto})")
    except Exception:      # noqa: BLE001
        pass

    last = (state["gain"], state["rf_on"])
    try:
        while not stop_evt.is_set():
            if apply_live() and (state["gain"], state["rf_on"]) != last:
                tb.set_gain(state["gain"])
                tb.set_amplitude(amplitude if state["rf_on"] else 0.0)
                last = (state["gain"], state["rf_on"])
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    finally:
        state["stop"] = True
        stop_evt.set()
        try:
            tb.stop(); tb.wait()
        except Exception:      # noqa: BLE001
            pass
        prod.join(timeout=1.0)
        ctrl.close()
        try:
            os.remove(fifo_path); os.rmdir(tmpdir)
        except OSError:
            pass
    print(f"{SIGNAL_NAME} stopped.")
    return 0


def _dry_run_consumer(q, state, stop_evt, apply_live, deadline, samp_rate_hz, spc, prod) -> int:
    """No radio: drain the producer, measure sustained throughput and prebuffer occupancy —
    exercises the generation + threading path so it can be validated off-hardware."""
    print("[dry-run] no UHD; draining the producer and measuring throughput …")
    t0 = time.perf_counter(); samples = 0; nbuf = 0; min_fill = 1 << 30
    warmup = 5
    while not stop_evt.is_set():
        apply_live()
        buf = q.get()
        nbuf += 1
        if nbuf > warmup:
            samples += buf.size
        min_fill = min(min_fill, q.qsize())
        if deadline is not None and time.monotonic() >= deadline:
            break
        if nbuf >= 500:                      # ~5 s worth at 0.01 s/buf
            break
    dt = time.perf_counter() - t0
    state["stop"] = True
    prod.join(timeout=2.0)
    if samples:
        msps = samples / (dt) / 1e6
        print(f"[dry-run] {nbuf} buffers, sustained ~{msps:.1f} Msps "
              f"({msps/(samp_rate_hz/1e6):.1f}x real-time), min prebuffer fill {min_fill}")
    print("[dry-run] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

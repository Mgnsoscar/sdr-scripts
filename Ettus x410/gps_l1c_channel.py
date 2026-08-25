#!/usr/bin/env python3
"""
gps_l1c_channel — GPS L1C channel-task for the X410 engine (composite mode).

Plays a spectrally-correct GPS **L1C** signal (1575.42 MHz) on one engine channel:

    L1Cp (pilot, 75% power) : Weil code × TMBOC(6,1,4/33) subcarrier × overlay
    L1Cd (data,  25% power) : Weil code × BOC(1,1) subcarrier   (bare code here)

This is a *channel-task*: the persistent x410_engine owns UHD; this builds the IQ
and drives one channel. See gps_l1ca_channel.py for the lifecycle; L1C differs in
being a two-component signal with an 18 s pilot overlay, which maps onto the
engine's **composite** mode.

Why composite (and why it's exact)
──────────────────────────────────
L1C is (pilot × overlay) + data — the overlay (±1 per 10 ms period) multiplies
only the pilot. Over one 10 ms period the signal is therefore one of exactly two
blocks:

    B0 = data + pilot     (overlay +1)
    B1 = data − pilot      (overlay −1)

so the full 18 s (1800-period) signal is those two 10 ms blocks played in the
order the 1800-symbol overlay dictates. We hand the engine [B0, B1] plus that
selector sequence; it streams blocks[selectors[k]] and loops — byte-identical to
a single fully-baked 18 s buffer, from ~2 blocks instead of gigabytes.
`--secondary off` drops the overlay (10 ms primary loop only; spectrally identical,
no secondary sync) → a single block.

Code fidelity — real IS-GPS-800 Weil codes
──────────────────────────────────────────
Both 10230-chip primaries are the real IS-GPS-800 codes: Legendre (mod 10223) →
Weil W[k]=L[k]⊕L[k+w] → 7-chip insertion [0110100] at index p. The pilot overlay
is the IS-GPS-800 11-bit LFSR. All validated in --self-test against the standard.

⚠  RF SAFETY / LEGAL: L1 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gps_l1c_channel.py --channel 0 --prn 5 --gain 55 --amplitude 0
    gps_l1c_channel.py --channel 1 --prn 5 --component pilot --secondary off
    gps_l1c_channel.py --self-test        # code + overlay + fidelity, no engine
    gps_l1c_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from engine_client import EngineClient, EngineError


# ── Constants ─────────────────────────────────────────────────────────────────

L1_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10
LEG_N = 10223
INSERT = (0, 1, 1, 0, 1, 0, 0)
TMBOC = (1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
SEC_LEN = 1800
A_PILOT = math.sqrt(0.75)
A_DATA = math.sqrt(0.25)

FREQUENCIES = {"GPS L1 (1575.42 MHz)": L1_HZ}

# Target sample rates (negotiated to the nearest engine-clock divisor). L1C needs
# to span the TMBOC BOC(6,1) energy at ±6 MHz, so keep it wide.
SAMPLE_RATES_MHZ = {
    "24.576 MHz (min — main lobe + BOC(1,1))": 24.576,
    "49.152 MHz (default — full TMBOC)":        49.152,
}

# Per-PRN (Weil index w, insertion index p), IS-GPS-800 (validated vs the sheet).
L1CP_WP = (
    (5111, 412), (5109, 161), (5108, 1), (5106, 303), (5103, 207), (5101, 4971),
    (5100, 4496), (5098, 5), (5095, 4557), (5094, 485), (5093, 253), (5091, 4676),
    (5090, 1), (5081, 66), (5080, 4485), (5069, 282), (5068, 193), (5054, 5211),
    (5044, 729), (5027, 4848), (5026, 982), (5014, 5955), (5004, 9805), (4980, 670),
    (4915, 464), (4909, 29), (4893, 429), (4885, 394), (4832, 616), (4824, 9457),
    (4591, 4429), (3706, 4771), (5092, 365), (4986, 9705), (4965, 9489), (4920, 4193),
    (4917, 9947), (4858, 824), (4847, 864), (4790, 347), (4770, 677), (4318, 6544),
    (4126, 6312), (3961, 9804), (3790, 278), (4911, 9461), (4881, 444), (4827, 4839),
    (4795, 4144), (4789, 9875), (4725, 197), (4675, 1156), (4539, 4674), (4535, 10035),
    (4458, 4504), (4197, 5), (4096, 9937), (3484, 430), (3481, 5), (3393, 355),
    (3175, 909), (2360, 1622), (1852, 6284),
)
L1CD_WP = (
    (5097, 181), (5110, 359), (5079, 72), (4403, 1110), (4121, 1480), (5043, 5034),
    (5042, 4622), (5104, 1), (4940, 4547), (5035, 826), (4372, 6284), (5064, 4195),
    (5084, 368), (5048, 1), (4950, 4796), (5019, 523), (5076, 151), (3736, 713),
    (4993, 9850), (5060, 5734), (5061, 34), (5096, 6142), (4983, 190), (4783, 644),
    (4991, 467), (4815, 5384), (4443, 801), (4769, 594), (4879, 4450), (4894, 9437),
    (4985, 4307), (5056, 5906), (4921, 378), (5036, 9448), (4812, 9432), (4838, 5849),
    (4855, 5547), (4904, 9546), (4753, 9132), (4483, 403), (4942, 3766), (4813, 3),
    (4957, 684), (4618, 9711), (4669, 333), (4969, 6124), (5031, 10216), (5038, 4251),
    (4740, 9893), (4073, 9884), (4843, 4627), (4979, 4449), (4867, 9798), (4964, 985),
    (5025, 4272), (4579, 126), (4390, 10024), (4763, 434), (4612, 1029), (4784, 561),
    (3716, 289), (4703, 638), (4851, 4353),
)
# Pilot overlay: (S1 polynomial octal, S1 initial state octal), IS-GPS-800.
L1CO_PARAMS = (
    (0o5111, 0o3266), (0o5421, 0o2040), (0o5501, 0o1527), (0o5403, 0o3307), (0o6417, 0o3756), (0o6141, 0o3026),
    (0o6351, 0o0562), (0o6501, 0o0420), (0o6205, 0o3415), (0o6235, 0o0337), (0o7751, 0o0265), (0o6623, 0o1230),
    (0o6733, 0o2204), (0o7627, 0o1440), (0o5667, 0o2412), (0o5051, 0o3516), (0o7665, 0o2761), (0o6325, 0o3750),
    (0o4365, 0o2701), (0o4745, 0o1206), (0o7633, 0o1544), (0o6747, 0o1774), (0o4475, 0o0546), (0o4225, 0o2213),
    (0o7063, 0o3707), (0o4423, 0o2051), (0o6651, 0o3650), (0o4161, 0o1777), (0o7237, 0o3203), (0o4473, 0o1762),
    (0o5477, 0o2100), (0o6163, 0o0571), (0o7223, 0o3710), (0o6323, 0o3535), (0o7125, 0o3110), (0o7035, 0o1426),
    (0o4341, 0o0255), (0o4353, 0o0321), (0o4107, 0o3124), (0o5735, 0o0572), (0o6741, 0o1736), (0o7071, 0o3306),
    (0o4563, 0o1307), (0o5755, 0o3763), (0o6127, 0o1604), (0o4671, 0o1021), (0o4511, 0o2624), (0o4533, 0o0406),
    (0o5357, 0o0114), (0o5607, 0o0077), (0o6673, 0o3477), (0o6153, 0o1000), (0o7565, 0o3460), (0o7107, 0o2607),
    (0o6211, 0o2057), (0o4321, 0o3467), (0o7201, 0o0706), (0o4451, 0o2032), (0o5411, 0o1464), (0o5141, 0o0520),
    (0o7041, 0o1766), (0o6637, 0o3270), (0o4577, 0o0341),
)


# ── L1C codes (bit-exact primaries + overlay, IS-GPS-800) ──────────────────────

_LEG = None


def _legendre():
    global _LEG
    if _LEG is None:
        qr = {(x * x) % LEG_N for x in range(1, LEG_N)}
        _LEG = [0] + [1 if k in qr else 0 for k in range(1, LEG_N)]
    return _LEG


def _primary(prn: int, component: str):
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    w, p = (L1CP_WP if component == "pilot" else L1CD_WP)[prn - 1]
    L = _legendre()
    W = [L[k] ^ L[(k + w) % LEG_N] for k in range(LEG_N)]
    return W[0:p - 1] + list(INSERT) + W[p - 1:LEG_N]


def _overlay(prn: int):
    """The 1800-symbol pilot overlay (0/1), IS-GPS-800 single 11-bit LFSR."""
    poly, init = L1CO_PARAMS[prn - 1]
    p = [((poly // 2) >> i) & 1 for i in range(11)]
    x = [(init >> i) & 1 for i in range(11)]
    c = [0] * SEC_LEN
    for i in range(SEC_LEN):
        c[i] = x[10]
        fb = 0
        for a, b in zip(x, p):
            fb ^= a & b
        x = [fb] + x[:-1]
    return c


# ── Baseband: raw pilot/data components, one 10 ms primary period ──────────────

def _components_raw(prn: int, samp_rate_hz: float):
    """The unnormalised in-phase pilot and data component buffers (complex128, one
    10 ms primary period), and the sample count. Normalisation and the ±pilot
    overlay blocks are formed by the caller."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(PRIMARY_MS * 1e-3 * sr))

    pilot = 1 - 2 * np.asarray(_primary(prn, "pilot"), dtype=np.int8)
    data = 1 - 2 * np.asarray(_primary(prn, "data"), dtype=np.int8)
    tmboc = np.asarray(TMBOC, dtype=np.int8)

    n = np.arange(n_samples, dtype=np.int64)
    num = n * CHIP_RATE_HZ
    chip = num // sr
    rem = num - chip * sr
    boc11 = np.where(rem * 2 < sr, 1.0, -1.0)
    boc61 = np.where(((rem * 12) // sr) % 2 == 0, 1.0, -1.0)
    cidx = chip % CODE_LEN
    use61 = tmboc[chip % 33] == 1

    data_s = (A_DATA * data[cidx] * boc11).astype(np.complex128)
    pilot_s = (A_PILOT * pilot[cidx] * np.where(use61, boc61, boc11)).astype(np.complex128)
    return data_s, pilot_s, n_samples


def build_l1c_blocks(prn: int, samp_rate_hz: float, component: str, secondary: str):
    """Build the composite playlist for L1C at a sample rate:

      • blocks : [B0]                 when there's no active overlay, or
                 [B0, B1] = [data+pilot, data−pilot]  with the pilot overlay,
      • selectors : [0]               or the 1800-symbol overlay as 0/1 indices.

    Peak-normalised over both blocks (so amplitude 0..1 never clips). Returns
    (blocks: list[np.ndarray complex64], selectors: list[int], n_samples)."""
    import numpy as np

    data_s, pilot_s, n = _components_raw(prn, samp_rate_hz)
    inc_p = component in ("both", "pilot")
    inc_d = component in ("both", "data")
    dp = data_s if inc_d else np.zeros_like(data_s)
    pp = pilot_s if inc_p else np.zeros_like(pilot_s)

    plus = dp + pp
    minus = dp - pp
    norm = max(np.max(np.abs(plus)), np.max(np.abs(minus))) or 1.0

    if inc_p and secondary == "full":
        b0 = (plus / norm).astype(np.complex64)
        b1 = (minus / norm).astype(np.complex64)
        signs = _overlay(prn)                     # 0/1 → +1/−1 applied to pilot
        selectors = [0 if s == 0 else 1 for s in signs]   # 0→+overlay(B0), 1→−overlay(B1)
        return [b0, b1], selectors, n
    return [(plus / norm).astype(np.complex64)], [0], n


# ── Self-test (codes + overlay + negotiation fidelity, no engine/hardware) ─────

def _self_test() -> int:
    ok = True

    def o24(bits):
        v = 0
        for b in bits[:24]:
            v = (v << 1) | b
        return v

    leg = _legendre()
    lok = sum(leg) == (LEG_N - 1) // 2
    print(f"Legendre({LEG_N}) ones={sum(leg)} (expect {(LEG_N-1)//2}) [{'OK' if lok else 'FAIL'}]")
    tok = sum(TMBOC) == 4 and [i for i, x in enumerate(TMBOC) if x] == [0, 4, 6, 29]
    print(f"TMBOC BOC(6,1) chips {[i for i,x in enumerate(TMBOC) if x]} [{'OK' if tok else 'FAIL'}]")
    ok = ok and lok and tok

    prim = {("P", 1): 0o5752067, ("P", 2): 0o70146401, ("P", 63): 0o56350460,
            ("D", 1): 0o77001425, ("D", 2): 0o23342754, ("D", 63): 0o34665654}
    for (kind, prn), want in prim.items():
        c = _primary(prn, "pilot" if kind == "P" else "data")
        good = o24(c) == want and len(c) == CODE_LEN and sum(c) == 5115
        ok = ok and good
        print(f"{'L1Cp' if kind=='P' else 'L1Cd'} PRN{prn:2d}: first24={oct(o24(c))} "
              f"expect={oct(want)} [{'OK' if good else 'FAIL'}]")

    ovl = {1: 0o65550354, 63: 0o7034020}
    for prn, want in ovl.items():
        c = _overlay(prn)
        good = o24(c) == want and len(c) == SEC_LEN and sum(c) == 900
        ok = ok and good
        print(f"overlay PRN{prn:2d}: first24={oct(o24(c))} expect={oct(want)} "
              f"len={len(c)} ones={sum(c)} [{'OK' if good else 'FAIL'}]")

    # Composite structure + negotiation fidelity (needs NumPy; skip if absent).
    try:
        import numpy as np
        blocks, sel, n = build_l1c_blocks(5, 49.152e6, "both", "full")
        # B0/B1 must be exactly data±pilot at the shared peak-normalisation.
        d_raw, p_raw, _ = _components_raw(5, 49.152e6)
        norm = max(np.max(np.abs(d_raw + p_raw)), np.max(np.abs(d_raw - p_raw)))
        exp0 = ((d_raw + p_raw) / norm).astype(np.complex64)
        exp1 = ((d_raw - p_raw) / norm).astype(np.complex64)
        good = (len(blocks) == 2 and len(sel) == SEC_LEN
                and blocks[0].size == n and blocks[1].size == n
                and np.max(np.abs(blocks[0])) <= 1.0 + 1e-6
                and np.array_equal(blocks[0], exp0) and np.array_equal(blocks[1], exp1)
                and not np.array_equal(blocks[0], blocks[1]))
        ok = ok and good
        print(f"composite: {len(blocks)} blocks × {n} samples, {len(sel)} selectors, "
              f"B0=data+pilot B1=data−pilot [{'OK' if good else 'FAIL'}]")

        # secondary off → a single block
        b_off, s_off, _ = build_l1c_blocks(5, 49.152e6, "both", "off")
        good_off = len(b_off) == 1 and s_off == [0]
        ok = ok and good_off
        print(f"secondary off → 1 block [{'OK' if good_off else 'FAIL'}]")

        # Negotiation fidelity: the primary-only (B0) code must acquire as well at
        # the negotiated 49.152 MHz as at the integer-samples/chip ideal 49.104.
        from gnss_acq import check_negotiation_fidelity
        ok = check_negotiation_fidelity(
            lambda r: build_l1c_blocks(5, r, "both", "off")[0][0],
            chip_rate_hz=CHIP_RATE_HZ, ideal_rate_hz=49.104e6,
            negotiated_rate_hz=49.152e6, label="L1C", min_db=18.0) and ok
    except ImportError:
        print("composite/fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("GPS L1C channel-task — plays the modernized L1C signal (pilot+data, "
               "TMBOC/BOC, 18 s overlay) on one X410 engine channel via composite mode.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=1, required=True,
                 help="GPS L1C PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=L1_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .choice("-Component", "--component", options=["both", "pilot", "data"],
                default="both", help="Which L1C components to transmit. Fixed per run.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="Pilot overlay: 'full' (18 s secondary via composite) or 'off' "
                     "(10 ms primary loop, spectrally identical). Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=10.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=49.152, required=True,
                help="Target channel sample rate; the engine negotiates the nearest "
                     "supported rate. Fixed per run.")
        .number("-Gain", "--gain", unit="dB", min=0, max=65, default=55,
                required=True, live=True, help="Channel TX gain. Live.")
        .number("-Amplitude", "--amplitude", min=0.0, max=1.0, default=0.0,
                required=True, live=True,
                help="Digital amplitude 0..1. Start at 0 for a pre-roll and raise "
                     "on-air via a tune-step; or set >0 to transmit on load. Live.")
        .text("-Engine-socket", "--engine_socket", default="/tmp/x410_engine.sock",
              help="Unix socket of the running x410_engine.")
        .text("-Owner", "--owner", default="",
              help="Channel ownership tag (default: auto from channel + PID).")
    )


# ── Entry point ─────────────────────────────────────────────────────────────────

def _write_shm(iq, idx: int) -> str:
    import tempfile
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, path = tempfile.mkstemp(prefix=f"gps_l1c_b{idx}_", suffix=".fc32", dir=shm)
    os.close(fd)
    iq.tofile(path)
    return path


def _connect_engine(socket_path: str, attempts: int = 20) -> EngineClient:
    last = None
    for _ in range(attempts):
        try:
            return EngineClient(socket_path).connect()
        except OSError as exc:
            last = exc
            time.sleep(0.25)
    raise SystemExit(f"could not reach engine at {socket_path}: {last}")


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    script = build_script()
    args = script.parse()
    ch = args.channel
    owner = args.owner or f"ch{ch}-{os.getpid()}"

    eng = _connect_engine(args.engine_socket)
    try:
        eng.acquire(ch, owner)
        actual_rate = eng.configure(ch, owner, args.samp_rate * 1e6)

        blocks, selectors, n_samples = build_l1c_blocks(
            args.prn, actual_rate, args.component, args.secondary)
        block_files = [_write_shm(b, i) for i, b in enumerate(blocks)]

        print("── GPS L1C channel-task ────────────────────────────────────")
        print(f"  engine channel : {ch}   owner {owner}")
        print(f"  PRN            : {args.prn}   component {args.component}")
        print(f"  carrier        : {args.freq/1e6:.3f} MHz")
        print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
              f"engine gave {actual_rate/1e6:.6f} MHz")
        print(f"  secondary      : {args.secondary}  "
              f"({len(blocks)} block(s) × {n_samples} samples, {len(selectors)} selectors)")
        print(f"  gain / amp     : {args.gain:g} dB / {args.amplitude:g} "
              f"({'MUTED — raise on-air' if args.amplitude == 0 else 'live on load'})")
        print("────────────────────────────────────────────────────────────")
        sys.stdout.flush()

        try:
            eng.load(ch, owner, {
                "mode": "composite", "freq_hz": args.freq, "gain_db": args.gain,
                "amplitude": args.amplitude, "block_files": block_files,
                "selectors": selectors, "label": f"gps_l1c prn{args.prn}"})
        finally:
            for f in block_files:      # engine copied them into RAM at load
                try:
                    os.unlink(f)
                except OSError:
                    pass

        ctrl = script.live_control(args)

        def apply_change(name, value):
            if name == "amplitude":
                eng.set(ch, owner, amplitude=value); ctrl.report("amplitude", value)
            elif name == "gain":
                eng.set(ch, owner, gain_db=value); ctrl.report("gain", value)
            elif name == "freq":
                eng.set(ch, owner, freq_hz=value); ctrl.report("freq", value)

        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())

        while not stop.is_set():
            for change in ctrl.drain():
                try:
                    apply_change(change.name, change.value)
                except EngineError as exc:
                    print(f"[warn] live {change.name}={change.value} rejected: {exc}",
                          flush=True)
            time.sleep(0.1)
        ctrl.close()
    finally:
        # Best-effort: tolerate a dropped connection so cleanup never raises a
        # secondary BrokenPipeError over the original traceback.
        try:
            eng.release(ch, owner)
        except Exception:
            pass
        try:
            eng.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

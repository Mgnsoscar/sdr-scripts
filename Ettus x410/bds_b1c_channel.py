#!/usr/bin/env python3
"""
bds_b1c_channel — BeiDou B1C channel-task for the X410 engine (composite mode).

Spectrally-correct BeiDou **B1C** signal (1575.42 MHz) — BeiDou's modernized
civil L1 signal — as the QMBOC composite of its data and pilot components.

Modulation — QMBOC(6,1,4/33), BDS-SIS-ICD-B1C §4.2 (data:pilot power 1:3); the
BOC(6,1) pilot sits in quadrature to BOC(1,1):
    I = ½·D·C_data·BOC(1,1)  −  √(1/11)·C_pilot·BOC(6,1)
    Q =                          √(29/44)·C_pilot·BOC(1,1)

Codes (BDS-SIS-ICD-B1C §5.2): data + pilot primaries are truncated Weil (Legendre
mod 10243), 10230 chips; the pilot secondary is a truncated Weil (mod 3607), 1800
chips = 18 s period. Validated in --self-test against the ICD first-24/last-24
check values.

Composite mapping (same as GPS L1C)
───────────────────────────────────
The secondary (±1 per 10 ms period) multiplies only the pilot, so over one period
the signal is one of two blocks: B0 = data+pilot, B1 = data−pilot. We hand the
engine [B0, B1] + the 1800-symbol secondary as a selector sequence; it streams the
full 18 s signal from two 10 ms blocks — byte-identical to a fully-baked buffer.
`--secondary off` drops it (10 ms primary loop only; spectrally identical).

Sample rate default 49.152 MHz (negotiated) to span the BOC(6,1) energy at ±6 MHz.
See gps_l1c_channel.py for the same composite pattern.

⚠  RF SAFETY / LEGAL: B1/L1 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    bds_b1c_channel.py --channel 0 --prn 19 --gain 55 --amplitude 0
    bds_b1c_channel.py --channel 1 --prn 19 --component pilot --secondary off
    bds_b1c_channel.py --self-test        # codes vs ICD + fidelity, no engine
    bds_b1c_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

B1C_HZ = 1575.42e6
CHIP_RATE_HZ = 1_023_000
CODE_LEN = 10230
PRIMARY_MS = 10                  # primary period (ms) = one secondary chip
WEIL_N = 10243                   # primary Weil Legendre prime
SEC_N = 3607                     # secondary Weil Legendre prime
SEC_LEN = 1800                   # pilot secondary length (chips) → 18 s period

A_DATA = 0.5
A_PILOT_A = math.sqrt(29 / 44)   # pilot BOC(1,1), on Q
A_PILOT_B = math.sqrt(1 / 11)    # pilot BOC(6,1), on I (quadrature)

FREQUENCIES = {"BeiDou B1C (1575.42 MHz)": B1C_HZ}
SAMPLE_RATES_MHZ = {"24.576 MHz (min — main lobe + BOC(1,1))": 24.576,
                    "49.152 MHz (default — full QMBOC)": 49.152}

# Per-PRN (Weil phase w, truncation point p), BDS-SIS-ICD-B1C Tables 5-2/5-3/5-4.
B1C_DATA_WP = (
    (2678, 699), (4802, 694), (958, 7318), (859, 2127), (3843, 715), (2232, 6682),
    (124, 7850), (4352, 5495), (1816, 1162), (1126, 7682), (1860, 6792), (4800, 9973),
    (2267, 6596), (424, 2092), (4192, 19), (4333, 10151), (2656, 6297), (4148, 5766),
    (243, 2359), (1330, 7136), (1593, 1706), (1470, 2128), (882, 6827), (3202, 693),
    (5095, 9729), (2546, 1620), (1733, 6805), (4795, 534), (4577, 712), (1627, 1929),
    (3638, 5355), (2553, 6139), (3646, 6339), (1087, 1470), (1843, 6867), (216, 7851),
    (2245, 1162), (726, 7659), (1966, 1156), (670, 2672), (4130, 6043), (53, 2862),
    (4830, 180), (182, 2663), (2181, 6940), (2006, 1645), (1080, 1582), (2288, 951),
    (2027, 6878), (271, 7701), (915, 1823), (497, 2391), (139, 2606), (3693, 822),
    (2054, 6403), (4342, 239), (3342, 442), (2592, 6769), (1007, 2560), (310, 2502),
    (4203, 5072), (455, 7268), (4318, 341),
)
B1C_PILOT_WP = (
    (796, 7575), (156, 2369), (4198, 5688), (3941, 539), (1374, 2270), (1338, 7306),
    (1833, 6457), (2521, 6254), (3175, 5644), (168, 7119), (2715, 1402), (4408, 5557),
    (3160, 5764), (2796, 1073), (459, 7001), (3594, 5910), (4813, 10060), (586, 2710),
    (1428, 1546), (2371, 6887), (2285, 1883), (3377, 5613), (4965, 5062), (3779, 1038),
    (4547, 10170), (1646, 6484), (1430, 1718), (607, 2535), (2118, 1158), (4709, 526),
    (1149, 7331), (3283, 5844), (2473, 6423), (1006, 6968), (3670, 1280), (1817, 1838),
    (771, 1989), (2173, 6468), (740, 2091), (1433, 1581), (2458, 1453), (3459, 6252),
    (2155, 7122), (1205, 7711), (413, 7216), (874, 2113), (2463, 1095), (1106, 1628),
    (1590, 1713), (3873, 6102), (4026, 6123), (4272, 6070), (3556, 1115), (128, 8047),
    (1200, 6795), (130, 2575), (4494, 53), (1871, 1729), (3073, 6388), (4386, 682),
    (4098, 5565), (1923, 7160), (1176, 2277),
)
B1C_SEC_WP = (
    (269, 1889), (1448, 1268), (1028, 1593), (1324, 1186), (822, 1239), (5, 1930),
    (155, 176), (458, 1696), (310, 26), (959, 1344), (1238, 1271), (1180, 1182),
    (1288, 1381), (334, 1604), (885, 1333), (1362, 1185), (181, 31), (1648, 704),
    (838, 1190), (313, 1646), (750, 1385), (225, 113), (1477, 860), (309, 1656),
    (108, 1921), (1457, 1173), (149, 1928), (322, 57), (271, 150), (576, 1214),
    (1103, 1148), (450, 1458), (399, 1519), (241, 1635), (1045, 1257), (164, 1687),
    (513, 1382), (687, 1514), (422, 1), (303, 1583), (324, 1806), (495, 1664),
    (725, 1338), (780, 1111), (367, 1706), (882, 1543), (631, 1813), (37, 228),
    (647, 2871), (1043, 2884), (24, 1823), (120, 75), (134, 11), (136, 63),
    (158, 1937), (214, 22), (335, 1768), (340, 1526), (661, 1402), (889, 1445),
    (929, 1680), (1002, 1290), (1149, 1245),
)


# ── B1C Weil codes (bit-exact, BDS-SIS-ICD-B1C §5.2) ───────────────────────────

_LEG = {}


def _legendre(N: int):
    if N not in _LEG:
        qr = {(x * x) % N for x in range(1, N)}
        _LEG[N] = [0] + [1 if k in qr else 0 for k in range(1, N)]
    return _LEG[N]


def _weil(w: int, p: int, N: int, length: int):
    L = _legendre(N)
    W = [L[k] ^ L[(k + w) % N] for k in range(N)]
    return [W[(n + p - 1) % N] for n in range(length)]


def _primary(prn: int, component: str):
    if not 1 <= prn <= 63:
        raise ValueError(f"PRN must be 1..63, got {prn}")
    w, p = (B1C_DATA_WP if component == "data" else B1C_PILOT_WP)[prn - 1]
    return _weil(w, p, WEIL_N, CODE_LEN)


def _secondary(prn: int):
    """The 1800-chip pilot secondary code (0/1)."""
    w, p = B1C_SEC_WP[prn - 1]
    return _weil(w, p, SEC_N, SEC_LEN)


# ── Baseband components (data + pilot, one 10 ms primary period) ───────────────

def _components_raw(prn: int, samp_rate_hz: float):
    """Unnormalised data and pilot QMBOC component buffers (complex128, one 10 ms
    primary period), and the sample count."""
    import numpy as np

    sr = int(round(samp_rate_hz))
    n_samples = int(round(PRIMARY_MS * 1e-3 * sr))

    cd = 1.0 - 2.0 * np.asarray(_primary(prn, "data"), dtype=np.int8)
    cp = 1.0 - 2.0 * np.asarray(_primary(prn, "pilot"), dtype=np.int8)

    n = np.arange(n_samples, dtype=np.int64)
    num = n * CHIP_RATE_HZ
    chip = num // sr
    rem = num - chip * sr
    boc11 = np.where(rem * 2 < sr, 1.0, -1.0)
    boc61 = np.where(((rem * 12) // sr) % 2 == 0, 1.0, -1.0)
    cidx = chip % CODE_LEN
    cdv, cpv = cd[cidx], cp[cidx]

    data = (A_DATA * cdv * boc11).astype(np.complex128)               # on I
    pilot = (-A_PILOT_B * cpv * boc61) + 1j * (A_PILOT_A * cpv * boc11)
    return data, pilot.astype(np.complex128), n_samples


def build_b1c_blocks(prn: int, samp_rate_hz: float, component: str, secondary: str):
    """Build the composite playlist for B1C: [B0] (no overlay) or [B0, B1] =
    [data+pilot, data−pilot] with the pilot secondary as selectors. Peak-normalised
    over both blocks. Returns (blocks: list[np.ndarray complex64], selectors, n)."""
    import numpy as np

    data, pilot, n = _components_raw(prn, samp_rate_hz)
    inc_d = component in ("both", "data")
    inc_p = component in ("both", "pilot")
    dp = data if inc_d else np.zeros_like(data)
    pp = pilot if inc_p else np.zeros_like(pilot)

    plus = dp + pp
    minus = dp - pp
    norm = max(np.max(np.abs(plus)), np.max(np.abs(minus))) or 1.0

    if inc_p and secondary == "full":
        b0 = (plus / norm).astype(np.complex64)
        b1 = (minus / norm).astype(np.complex64)
        signs = _secondary(prn)                     # 0/1 → +/−
        selectors = [0 if s == 0 else 1 for s in signs]
        return [b0, b1], selectors, n
    return [(plus / norm).astype(np.complex64)], [0], n


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    ok = True

    def octs(bits):
        f = l = 0
        for b in bits[:24]:
            f = (f << 1) | b
        for b in bits[-24:]:
            l = (l << 1) | b
        return "%08o" % f, "%08o" % l

    chk = {
        ("data", 1): ("53773116", "42711657"), ("data", 63): ("27571255", "47160627"),
        ("pilot", 1): ("71676756", "13053205"), ("pilot", 63): ("03210227", "56250500"),
    }
    for (comp, prn), want in chk.items():
        c = _primary(prn, comp)
        good = octs(c) == want and len(c) == CODE_LEN
        ok = ok and good
        print(f"{comp} primary PRN{prn:2d}: {octs(c)} expect {want} [{'OK' if good else 'FAIL'}]")

    for prn, want in {1: ("27516364", "67377026")}.items():
        s = _secondary(prn)
        good = octs(s) == want and len(s) == SEC_LEN
        ok = ok and good
        print(f"pilot secondary PRN{prn}: {octs(s)} expect {want} [{'OK' if good else 'FAIL'}]")

    for N, exp in ((WEIL_N, 5121), (SEC_N, 1803)):
        got = sum(_legendre(N))
        ok = ok and got == exp
        print(f"Legendre({N}) ones={got} (expect {exp}) [{'OK' if got==exp else 'FAIL'}]")

    try:
        import numpy as np
        blocks, sel, n = build_b1c_blocks(19, 49.152e6, "both", "full")
        d, p, _ = _components_raw(19, 49.152e6)
        norm = max(np.max(np.abs(d + p)), np.max(np.abs(d - p)))
        good = (len(blocks) == 2 and len(sel) == SEC_LEN
                and np.array_equal(blocks[0], ((d + p) / norm).astype(np.complex64))
                and np.array_equal(blocks[1], ((d - p) / norm).astype(np.complex64)))
        ok = ok and good
        print(f"composite: {len(blocks)} blocks × {n} samples, {len(sel)} selectors, "
              f"B0=data+pilot B1=data−pilot [{'OK' if good else 'FAIL'}]")

        from gnss_acq import check_negotiation_fidelity
        ok = check_negotiation_fidelity(
            lambda r: build_b1c_blocks(19, r, "both", "off")[0][0],
            chip_rate_hz=CHIP_RATE_HZ, ideal_rate_hz=49.104e6, negotiated_rate_hz=49.152e6,
            label="B1C", min_db=18.0) and ok
    except ImportError:
        print("composite/fidelity: skipped (no NumPy here)")

    print("ALL CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("BeiDou B1C channel-task — QMBOC(6,1,4/33) civil L1 signal (data+"
               "pilot, 18 s pilot overlay, real BDS-SIS-ICD-B1C Weil codes) on one "
               "X410 engine channel via composite mode.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .integer("-PRN", "--prn", min=1, max=63, default=19, required=True,
                 help="BeiDou B1C PRN (1..63). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=B1C_HZ, required=True, live=True,
                help="RF carrier. Live (retunes the channel).")
        .choice("-Component", "--component", options=["both", "pilot", "data"],
                default="both", help="Which B1C components to transmit. Fixed per run.")
        .choice("-Secondary", "--secondary", options=["full", "off"], default="full",
                help="Pilot overlay: 'full' (18 s secondary via composite) or 'off' "
                     "(10 ms primary loop). Fixed per run.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=10.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=49.152, required=True,
                help="Target channel sample rate (negotiated). Fixed per run.")
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


def build(args, rate_hz):
    blocks, selectors, n_samples = build_b1c_blocks(
        args.prn, rate_hz, args.component, args.secondary)
    files = [write_shm(b, f"bds_b1c_b{i}") for i, b in enumerate(blocks)]
    spec = {"mode": "composite", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "block_files": files, "selectors": selectors,
            "label": f"bds_b1c prn{args.prn}"}
    info = [f"PRN            : {args.prn}   component {args.component}   secondary {args.secondary}",
            f"composite      : {len(blocks)} block(s) × {n_samples} samples, {len(selectors)} selectors"]
    return spec, files, info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    return run_channel(script, args, build, title="BeiDou B1C channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

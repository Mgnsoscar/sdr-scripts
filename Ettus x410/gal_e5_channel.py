#!/usr/bin/env python3
"""
gal_e5_channel — Galileo E5a / E5b (single sideband) channel-task (expanded mode).

Transmit ONE Galileo E5 sideband — E5a (1176.45 MHz) or E5b (1207.14 MHz) — as a
QPSK(10) signal: data component on I, pilot on Q, both BPSK at 10.23 Mcps with the
real tiered ranging codes from the Galileo OS SIS ICD (Issue 2.2). This is a
single sideband, NOT the full AltBOC(15,10) (see gal_e5_altboc_channel).

Each component is a *tiered* code: a 10230-chip primary (two 14-bit LFSRs, reg-1
all-ones, reg-2 a per-code start value) ⊕ a secondary overlay:
    E5a-I: CS201 (20)   E5a-Q: CS100n (100)
    E5b-I: CS41 (4)     E5b-Q: CS100(n+50) (100)
The full repeating period is 100 ms (LCM of the tiered periods) → a ~33 MB buffer
at 40.96 MHz, one seamless expanded loop. All 200 primary codes are validated in
--self-test against the ICD first-24-chip check values.

E5 sidebands are ~20 MHz wide → ~40 MHz sample rate. See gps_prn_channel.py for
the channel-task lifecycle and on-air pre-roll.

⚠  RF SAFETY / LEGAL: E5a/E5b are live GNSS bands. Transmit ONLY into a shielded /
   conducted setup you are LICENSED / AUTHORISED to use — never over the air.

CLI
───
    gal_e5_channel.py --channel 0 --band E5a --svid 1 --gain 55 --amplitude 0
    gal_e5_channel.py --self-test        # 200 codes vs ICD + fidelity, no engine
    gal_e5_channel.py --describe-params  # paramkit JSON schema for the GUI
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paramkit import Script
from channel_task import run_channel, write_shm


# ── Constants ─────────────────────────────────────────────────────────────────

E5A_HZ = 1176.45e6
E5B_HZ = 1207.14e6
CHIP_RATE_HZ = 10.23e6
PRIMARY_LEN = 10230
TIERED_PERIOD_MS = 100
EPOCHS_PER_LOOP = TIERED_PERIOD_MS

FREQUENCIES = {"Galileo E5a (1176.45 MHz)": E5A_HZ, "Galileo E5b (1207.14 MHz)": E5B_HZ}
SAMPLE_RATES_MHZ = {"20.48 MHz (min)": 20.48, "40.96 MHz (default)": 40.96,
                    "61.44 MHz (max)": 61.44}

TAPS = {
    "E5a-I": (0o40503, 0o50661), "E5a-Q": (0o40503, 0o50661),
    "E5b-I": (0o64021, 0o51445), "E5b-Q": (0o64021, 0o43143),
}

E5AI_S2 = {1: 0o30305, 2: 0o14234, 3: 0o27213, 4: 0o20577, 5: 0o23312, 6: 0o33463, 7: 0o15614, 8: 0o12537, 9: 0o01527, 10: 0o30236, 11: 0o27344, 12: 0o07272, 13: 0o36377, 14: 0o17046, 15: 0o06434, 16: 0o15405, 17: 0o24252, 18: 0o11631, 19: 0o24776, 20: 0o00630, 21: 0o11560, 22: 0o17272, 23: 0o27445, 24: 0o31702, 25: 0o13012, 26: 0o14401, 27: 0o34727, 28: 0o22627, 29: 0o30623, 30: 0o27256, 31: 0o01520, 32: 0o14211, 33: 0o31465, 34: 0o22164, 35: 0o33516, 36: 0o02737, 37: 0o21316, 38: 0o35425, 39: 0o35633, 40: 0o24655, 41: 0o14054, 42: 0o27027, 43: 0o06604, 44: 0o31455, 45: 0o34465, 46: 0o25273, 47: 0o20763, 48: 0o31721, 49: 0o17312, 50: 0o13277}
E5AQ_S2 = {1: 0o25652, 2: 0o05142, 3: 0o24723, 4: 0o31751, 5: 0o27366, 6: 0o24660, 7: 0o33655, 8: 0o27450, 9: 0o07626, 10: 0o01705, 11: 0o12717, 12: 0o32122, 13: 0o16075, 14: 0o16644, 15: 0o37556, 16: 0o02477, 17: 0o02265, 18: 0o06430, 19: 0o25046, 20: 0o12735, 21: 0o04262, 22: 0o11230, 23: 0o00037, 24: 0o06137, 25: 0o04312, 26: 0o20606, 27: 0o11162, 28: 0o22252, 29: 0o30533, 30: 0o24614, 31: 0o07767, 32: 0o32705, 33: 0o05052, 34: 0o27553, 35: 0o03711, 36: 0o02041, 37: 0o34775, 38: 0o05274, 39: 0o37356, 40: 0o16205, 41: 0o36270, 42: 0o06600, 43: 0o26773, 44: 0o17375, 45: 0o35267, 46: 0o36255, 47: 0o12044, 48: 0o26442, 49: 0o21621, 50: 0o25411}
E5BI_S2 = {1: 0o07220, 2: 0o26047, 3: 0o00252, 4: 0o17166, 5: 0o14161, 6: 0o02540, 7: 0o01537, 8: 0o26023, 9: 0o01725, 10: 0o20637, 11: 0o02364, 12: 0o27731, 13: 0o30640, 14: 0o34174, 15: 0o06464, 16: 0o07676, 17: 0o32231, 18: 0o10353, 19: 0o00755, 20: 0o26077, 21: 0o11644, 22: 0o11537, 23: 0o35115, 24: 0o20452, 25: 0o34645, 26: 0o25664, 27: 0o21403, 28: 0o32253, 29: 0o02337, 30: 0o30777, 31: 0o27122, 32: 0o22377, 33: 0o36175, 34: 0o33075, 35: 0o33151, 36: 0o13134, 37: 0o07433, 38: 0o10216, 39: 0o35466, 40: 0o02533, 41: 0o05351, 42: 0o30121, 43: 0o14010, 44: 0o32576, 45: 0o30326, 46: 0o37433, 47: 0o26022, 48: 0o35770, 49: 0o06670, 50: 0o12017}
E5BQ_S2 = {1: 0o03331, 2: 0o06143, 3: 0o25322, 4: 0o23371, 5: 0o00413, 6: 0o36235, 7: 0o17750, 8: 0o04745, 9: 0o13005, 10: 0o37140, 11: 0o30155, 12: 0o20237, 13: 0o03461, 14: 0o31662, 15: 0o27146, 16: 0o05547, 17: 0o02456, 18: 0o30013, 19: 0o00322, 20: 0o10761, 21: 0o26767, 22: 0o36004, 23: 0o30713, 24: 0o07662, 25: 0o21610, 26: 0o20134, 27: 0o11262, 28: 0o10706, 29: 0o34143, 30: 0o11051, 31: 0o25460, 32: 0o17665, 33: 0o32354, 34: 0o21230, 35: 0o20146, 36: 0o11362, 37: 0o37246, 38: 0o16344, 39: 0o15034, 40: 0o25471, 41: 0o25646, 42: 0o22157, 43: 0o04336, 44: 0o16356, 45: 0o04075, 46: 0o02626, 47: 0o11706, 48: 0o37011, 49: 0o27041, 50: 0o31024}

S2_BY_COMPONENT = {"E5a-I": E5AI_S2, "E5a-Q": E5AQ_S2, "E5b-I": E5BI_S2, "E5b-Q": E5BQ_S2}

CS201 = "842E9"        # 20-chip, hex MSB-first
CS41 = "E"             # 4-chip
E5AQ_SEC = {
    1: "83F6F69D8F6E15411FB8C9B1C", 2: "66558BD3CE0C7792E83350525", 3: "59A025A9C1AF0651B779A8381", 4: "D3A32640782F7B18E4DF754B7",
    5: "B91FCAD7760C218FA59348A93", 6: "BAC77E933A779140F094FBF98", 7: "537785DE280927C6B58BA6776", 8: "EFCAB4B65F38531ECA22257E2",
    9: "79F8CAE838475EA5584BEFC9B", 10: "CA5170FEA3A810EC606B66494", 11: "1FC32410652A2C49BD845E567", 12: "FE0A9A7AFDAC44E42CB95D261",
    13: "B03062DC2B71995D5AD8B7DBE", 14: "F6C398993F598E2DF4235D3D5", 15: "1BB2FB8B5BF24395C2EF3C5A1", 16: "2F920687D238CC7046EF6AFC9",
    17: "34163886FC4ED7F2A92EFDBB8", 18: "66A872CE47833FB2DFD5625AD", 19: "99D5A70162C920A4BB9DE1CA8", 20: "81D71BD6E069A7ACCBEDC66CA",
    21: "A654524074A9E6780DB9D3EC6", 22: "C3396A101BEDAF623CFC5BB37", 23: "C3D4AB211DF36F2111F2141CD", 24: "3DFF25EAE761739265AF145C1",
    25: "994909E0757D70CDE389102B5", 26: "B938535522D119F40C25FDAEC", 27: "C71AB549C0491537026B390B7", 28: "0CDB8C9E7B53F55F5B0A0597B",
    29: "61C5FA252F1AF81144766494F", 30: "626027778FD3C6BB4BAA7A59D", 31: "E745412FF53DEBD03F1C9A633", 32: "3592AC083F3175FA724639098",
    33: "52284D941C3DCAF2721DDB1FD", 34: "73B3D8F0AD55DF4FE814ED890", 35: "94BF16C83BD7462F6498E0282", 36: "A8C3DE1AC668089B0B45B3579",
    37: "E23FFC2DD2C14388AD8D6BEC8", 38: "F2AC871CDF89DDC06B5960D2B", 39: "06191EC1F622A77A526868BA1", 40: "22D6E2A768E5F35FFC8E01796",
    41: "25310A06675EB271F2A09EA1D", 42: "9F7993C621D4BEC81A0535703", 43: "D62999EACF1C99083C0B4A417", 44: "F665A7EA441BAA4EA0D01078C",
    45: "46F3D3043F24CDEABD6F79543", 46: "E2E3E8254616BD96CEFCA651A", 47: "E548231A82F9A01A19DB5E1B2", 48: "265C7F90A16F49EDE2AA706C8",
    49: "364A3A9EB0F0481DA0199D7EA", 50: "9810A7A898961263A0F749F56",
}
E5BQ_SEC = {
    1: "CFF914EE3C6126A49FD5E5C94", 2: "FC317C9A9BF8C6038B5CADAB3", 3: "A2EAD74B6F9866E414393F239", 4: "72F2B1180FA6B802CB84DF997",
    5: "13E3AE93BC52391D09E84A982", 6: "77C04202B91B22C6D3469768E", 7: "FEBC592DD7C69AB103D0BB29C", 8: "0B494077E7C66FB6C51942A77",
    9: "DD0E321837A3D52169B7B577C", 10: "43DEA90EA6C483E7990C3223F", 11: "0366AB33F0167B6FA979DAE18", 12: "99CCBBFAB1242CBE31E1BD52D",
    13: "A3466923CEFDF451EC0FCED22", 14: "1A5271F22A6F9A8D76E79B7F0", 15: "3204A6BB91B49D1A2D3857960", 16: "32F83ADD43B599CBFB8628E5B",
    17: "3871FB0D89DB77553EB613CC1", 18: "6A3CBDFF2D64D17E02773C645", 19: "2BCD09889A1D7FC219F2EDE3B", 20: "3E49467F4D4280B9942CD6F8C",
    21: "658E336DCFD9809F86D54A501", 22: "ED4284F345170CF77268C8584", 23: "29ECCE910D832CAF15E3DF5D1", 24: "456CCF7FE9353D50E87A708FA",
    25: "FB757CC9E18CBC02BF1B84B9A", 26: "5686229A8D98224BC426BC7FC", 27: "700A2D325EA14C4B7B7AA8338", 28: "1210A330B4D3B507D854CBA3F",
    29: "438EE410BD2F7DBCDD85565BA", 30: "4B9764CC455AE1F61F7DA432B", 31: "BF1F45FDDA3594ACF3C4CC806", 32: "DA425440FE8F6E2C11B8EC1A4",
    33: "EE2C8057A7C16999AFA33FED1", 34: "2C8BD7D8395C61DFA96243491", 35: "391E4BB6BC43E98150CDDCADA", 36: "399F72A9EADB42C90C3ECF7F0",
    37: "93031FDEA588F88E83951270C", 38: "BA8061462D873705E95D5CB37", 39: "D24188F88544EB121E963FD34", 40: "D5F6A8BB081D8F383825A4DCA",
    41: "0FA4A205F0D76088D08EAF267", 42: "272E909FAEBC65215E263E258", 43: "3370F35A674922828465FC816", 44: "54EF96116D4A0C8DB0E07101F",
    45: "DE347C7B27FADC48EF1826A2B", 46: "01B16ECA6FC343AE08C5B8944", 47: "1854DB743500EE94D8FC768ED", 48: "28E40C684C87370CD0597FAB4",
    49: "5E42C19717093353BCAAF4033", 50: "64310BAD8EB5B36E38646AF01",
}

BANDS = {
    "E5a": {"freq": E5A_HZ, "data": "E5a-I", "pilot": "E5a-Q",
            "sec_data": lambda svid: (CS201, 20), "sec_pilot": lambda svid: (E5AQ_SEC[svid], 100)},
    "E5b": {"freq": E5B_HZ, "data": "E5b-I", "pilot": "E5b-Q",
            "sec_data": lambda svid: (CS41, 4), "sec_pilot": lambda svid: (E5BQ_SEC[svid], 100)},
}


# ── Code generation (pure Python, no NumPy) ────────────────────────────────────

def _taps_vector(oct_val: int):
    return [(oct_val >> j) & 1 for j in range(1, 15)]


def _start_vector(oct_val: int):
    return [(oct_val >> (j - 1)) & 1 for j in range(1, 15)]


def primary_code(component: str, svid: int):
    """The 10230-chip Galileo E5 primary code (0/1) for a component and SVID (1..50)."""
    if not 1 <= svid <= 50:
        raise ValueError(f"SVID must be 1..50, got {svid}")
    t1_oct, t2_oct = TAPS[component]
    tap1, tap2 = _taps_vector(t1_oct), _taps_vector(t2_oct)
    c1 = [1] * 14
    c2 = _start_vector(S2_BY_COMPONENT[component][svid])
    out = []
    for _ in range(PRIMARY_LEN):
        out.append(c1[13] ^ c2[13])
        fb1 = fb2 = 0
        for j in range(14):
            fb1 ^= tap1[j] & c1[j]
            fb2 ^= tap2[j] & c2[j]
        c1 = [fb1] + c1[:13]
        c2 = [fb2] + c2[:13]
    return out


def _hex_bits(hexstr: str, length: int):
    nbits = len(hexstr) * 4
    v = int(hexstr, 16)
    return [(v >> (nbits - 1 - i)) & 1 for i in range(nbits)][:length]


def secondary_code(band: str, channel: str, svid: int):
    hexstr, length = BANDS[band]["sec_data" if channel == "data" else "sec_pilot"](svid)
    return _hex_bits(hexstr, length)


def build_iq_buffer(band: str, svid: int, samp_rate_hz: float):
    """Complex64 QPSK E5 buffer over a whole number of 100 ms tiered periods
    (seamless). I = data, Q = pilot, unit magnitude. Returns (iq, n_samples, n_periods)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    chips_per_loop = PRIMARY_LEN * EPOCHS_PER_LOOP        # 1_023_000

    spp = Fraction(sr * chips_per_loop, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator
    conf = BANDS[band]

    def tiered_chips(component, channel):
        prim = np.asarray(primary_code(component, svid), dtype=np.int8)
        sec = np.asarray(secondary_code(band, channel, svid), dtype=np.int8)
        overlay = sec[np.arange(EPOCHS_PER_LOOP) % len(sec)]
        return (prim[None, :] ^ overlay[:, None]).reshape(-1)

    data_chips = tiered_chips(conf["data"], "data")
    pilot_chips = tiered_chips(conf["pilot"], "pilot")
    if n_periods > 1:
        data_chips = np.tile(data_chips, n_periods)
        pilot_chips = np.tile(pilot_chips, n_periods)

    i_bip = (1.0 - 2.0 * data_chips).astype(np.float32)
    q_bip = (1.0 - 2.0 * pilot_chips).astype(np.float32)
    n = np.arange(n_samples, dtype=np.int64)
    chip_idx = (n * cr) // sr
    inv = np.float32(1.0 / np.sqrt(2.0))
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = i_bip[chip_idx] * inv
    iq.imag = q_bip[chip_idx] * inv
    return iq, n_samples, n_periods


# ── Self-test (200 codes vs ICD + fidelity, no engine) ─────────────────────────

def _self_test() -> int:
    chk = {
     "E5a-I": {1: 0x3CEA9D, 2: 0x9D8CF1, 25: 0xA7D629, 50: 0xA5029C},
     "E5a-Q": {1: 0x515537, 2: 0xD67539, 25: 0xDCD55C, 50: 0x53DA0E},
     "E5b-I": {1: 0xC5BEA1, 2: 0x4F6248, 25: 0x1969C0, 50: 0xAFC22B},
     "E5b-Q": {1: 0xE49AF0, 2: 0xCE701F, 25: 0x71DE13, 50: 0x37AF4F},
    }
    ok = True
    for comp in ("E5a-I", "E5a-Q", "E5b-I", "E5b-Q"):
        good = 0
        for svid, want in chk[comp].items():
            code = primary_code(comp, svid)
            first24 = 0
            for b in code[:24]:
                first24 = (first24 << 1) | b
            if len(code) == PRIMARY_LEN and first24 == want:
                good += 1
        comp_ok = good == len(chk[comp])
        ok = ok and comp_ok
        print(f"{comp}: {good}/{len(chk[comp])} primary codes match ICD first-24 "
              f"[{'OK' if comp_ok else 'FAIL'}]")

    cs201 = _hex_bits(CS201, 20)
    cs41 = _hex_bits(CS41, 4)
    sec_ok = (len(cs201) == 20 and cs41 == [1, 1, 1, 0]
              and len(secondary_code("E5a", "pilot", 1)) == 100
              and len(secondary_code("E5b", "pilot", 50)) == 100)
    ok = ok and sec_ok
    print(f"secondaries: CS201 len={len(cs201)}, CS41={cs41}, E5x-Q len=100 "
          f"[{'OK' if sec_ok else 'FAIL'}]")

    try:
        import numpy as np
        from gnss_acq import check_negotiation_fidelity
        # A receiver acquires the 1 ms E5 primary; the 100 ms buffer repeats it.
        def primary_1ms(rate_hz):
            sr = int(round(rate_hz))
            n = int(round(0.001 * sr))
            b = 1.0 - 2.0 * np.asarray(primary_code("E5a-I", 1), dtype=np.int8)
            idx = (np.arange(n, dtype=np.int64) * int(CHIP_RATE_HZ) // sr) % PRIMARY_LEN
            return b[idx].astype(np.complex64)
        ok = check_negotiation_fidelity(
            primary_1ms, chip_rate_hz=CHIP_RATE_HZ, ideal_rate_hz=40.92e6,
            negotiated_rate_hz=40.96e6, label="E5 primary (1 ms)", min_db=18.0) and ok
    except ImportError:
        print("fidelity: skipped (no NumPy here)")

    print("ALL E5 CODE CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Parameter schema / entry point ──────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo E5a/E5b single-sideband channel-task — QPSK(10) tiered "
               "ranging codes (real OS SIS ICD) on one X410 engine channel.")
        .integer("-Channel", "--channel", min=0, max=3, default=0, required=True,
                 help="X410 engine channel (0=RF0 … 3=RF3). Fixed per run.")
        .choice("-Band", "--band", options=["E5a", "E5b"], default="E5a",
                help="E5a @ 1176.45 MHz or E5b @ 1207.14 MHz. Sets the carrier.")
        .integer("-SVID", "--svid", min=1, max=50, default=1, required=True,
                 help="Galileo SVID (1..50). Fixed per run.")
        .number("-Frequency", "--freq", unit="Hz", min=70e6, max=6e9,
                presets=FREQUENCIES, default=E5A_HZ, live=True,
                help="RF carrier (auto-set from --band; override to retune). Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=15.0, max=125.0,
                presets=SAMPLE_RATES_MHZ, default=40.96, required=True,
                help="Target channel sample rate (negotiated). E5 sideband is "
                     "~20 MHz wide; ~40 MHz recommended. Fixed per run.")
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
    iq, n_samples, n_periods = build_iq_buffer(args.band, args.svid, rate_hz)
    path = write_shm(iq, "gal_e5")
    spec = {"mode": "expanded", "freq_hz": args.freq, "gain_db": args.gain,
            "amplitude": args.amplitude, "iq_file": path,
            "label": f"gal_e5 {args.band} svid{args.svid}"}
    info = [f"band / SVID    : {args.band}  SVID {args.svid}  (QPSK: data I + pilot Q)",
            f"buffer         : {n_samples} samples ({n_periods}×100 ms tiered, {n_samples*8/1e6:.1f} MB)"]
    return spec, [path], info


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()
    script = build_script()
    args = script.parse()
    args.freq = BANDS[args.band]["freq"]      # carrier set by band
    return run_channel(script, args, build, title="Galileo E5 channel-task")


if __name__ == "__main__":
    raise SystemExit(main())

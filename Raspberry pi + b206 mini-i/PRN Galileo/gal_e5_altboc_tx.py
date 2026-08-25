#!/usr/bin/env python3
"""
Galileo E5 AltBOC(15,10) transmitter for GNU Radio + UHD (B200-mini family).

Purpose
───────
Transmit the FULL wideband Galileo E5 signal — both sidebands at once — as a
single constant-envelope AltBOC(15,10) waveform centred on the E5 carrier
(1191.795 MHz). E5a (data E5a-I + pilot E5a-Q) forms the lower sideband
(−15.345 MHz → 1176.45 MHz) and E5b (E5b-I + E5b-Q) the upper sideband
(+15.345 MHz → 1207.14 MHz), exactly as a real Galileo satellite radiates them,
using the real 10.23 Mcps tiered ranging codes from the OS SIS ICD (v2.2).

This supersedes the per-sideband gal_e5_tx.py: instead of one QPSK half, it
emits the true composite that a wideband E5 receiver sees, and the two halves
can still be received independently as QPSK at 1176.45 / 1207.14 MHz.

⚠  RF SAFETY / LEGAL: E5 is a live GNSS band. Transmit ONLY into a shielded /
   conducted setup (cable + attenuators into a receiver or spectrum analyser)
   that you are LICENSED / AUTHORISED to use. Radiating a GNSS code over the air
   can jam or spoof real receivers and is illegal in most places.

AltBOC(15,10) modulation (ICD §2.3.1)
─────────────────────────────────────
The composite is a constant-envelope 8-PSK signal (ICD Eq. 6):

        s_E5(t) = exp( j·π/4·k(t) ),   k ∈ {1..8}

The phase index k is a pure look-up (ICD Table 7) from two things:
  • the input quadruple (e_E5a-I, e_E5b-I, e_E5a-Q, e_E5b-Q) — each ±1, the four
    tiered code chips at time t (logic→signal per ICD Table 12);
  • the sub-period index iTs ∈ {0..7} — which eighth of the 15.345 MHz
    sub-carrier period Ts,E5 the sample falls in: iTs = ⌊8·(t mod Ts)/Ts⌋.
This look-up table is the exact constant-envelope AltBOC generator that real
satellites use; --self-test proves it reproduces the ICD's independent direct
sub-carrier formula (Eq. 3–4 with the Table 6 AS/AP coefficients) to 1e-12 over
all 8×16 = 128 cases, so both the table and the sideband/sign conventions are
verified, and the embedded E5 primary codes are checked against the ICD's
first-24-chip "Initial Sequence" values (200/200) as in gal_e5_tx.py.

    component      primary   secondary            tiered period
    E5a-I (data)   10230     CS201  (20 chip)      20 ms
    E5a-Q (pilot)  10230     CS100n (100 chip)     100 ms
    E5b-I (data)   10230     CS41   (4 chip)       4 ms
    E5b-Q (pilot)  10230     CS100(n+50) (100)     100 ms
The composite repeats every 100 ms. No navigation data is modulated (data
symbols held constant) for a clean, deterministic ranging spectrum.

Sample rate
───────────
AltBOC needs both ±15.345 MHz sidebands, so it must run near the B200's ceiling.
The default 61.38 MHz (= 60×1.023) gives 6 samples/chip and exactly 4 samples
per 15.345 MHz sub-carrier period, so every code-chip and sub-carrier edge lands
on the sample grid and the ±15.345 MHz sidebands sit well inside the ±30.69 MHz
Nyquist band. At 61.38 MS/s one 100 ms period is exactly 6 138 000 samples
(~49 MB), precomputed once into /dev/shm and replayed with
blocks.file_source(repeat=True) — bit-exact and seam-free. (Exact reproduction
of all 8 AltBOC sub-periods would need 122.76 MS/s, which the B200 cannot clock;
at 61.38 the in-band sideband structure is reproduced faithfully and the signal
stays constant-envelope.)

Streaming levers (same as the other builders)
─────────────────────────────────────────────
PRECOMPUTE+LOOP · sc8 over the wire · silent after start() · master_clock_rate
pinned == sample rate (1:1, no FPGA resampling), so samples/chip and the loop
length stay exact.

Live tuning (paramkit.live):  gain → set_gain,  amplitude → multiply_const_cc.
Carrier is fixed at E5 centre; SVID / sample rate / otw are fixed per run.

CLI
───
    gal_e5_altboc_tx.py --svid 1 --power -30 --rf on
    gal_e5_altboc_tx.py --self-test        # verify codes + AltBOC table, no hw
    gal_e5_altboc_tx.py --describe-params  # paramkit JSON schema for the GUI
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

# Stable calibration signal id. When a task sets SDR_CAL_SIGNAL_ID to this value the
# agent injects this unit's resolved calibration (SDR_CALIBRATION_FILE); calkit reads
# it and --power maps through the unit's MEASURED curve at its real operating plane
# (e.g. EIRP). Absent it, the baked USER CALIBRATION constants below are used. See the
# agent's docs/calibration.md.
CAL_SIGNAL_ID = "gal_e5_altboc"


# ═══════════════════════════════════════════════════════════════════════════════
# USER CALIBRATION — MEASURE THESE ONCE, THEN EDIT THE VALUES BELOW
# ═══════════════════════════════════════════════════════════════════════════════
# You set the transmit level in dBm. That only works if the script knows how the
# SDR's gain maps to real output power, which you establish once with a spectrum
# analyser: leave AMPLITUDE at the value below, run with --power at its maximum
# (that commands GAIN_AT_MAX_DB), measure the actual output power at the SDR RF
# port, and put that number in OUTPUT_POWER_DBM. From that anchor the script maps
# any requested power to a gain (1 dB gain ≈ 1 dB power, across the B200's linear
# range). CABLE_LOSS_DB / AMPLIFIER_GAIN_DB describe the RF chain AFTER the port,
# so the number you dial in is the power delivered at the far end.

OUTPUT_POWER_DBM = -20.0    # max output (dBm) at GAIN_AT_MAX_DB and AMPLITUDE — MEASURE THIS
GAIN_AT_MAX_DB = 89.75      # the gain that produced it; also the HARD ceiling the script commands
CABLE_LOSS_DB = 0.0         # cabling insertion loss after the SDR port (positive dB)
AMPLIFIER_GAIN_DB = 0.0     # external amplifier gain after the SDR port (positive dB)

# Fixed baseband digital amplitude (0..1). NOT a user control and never a task
# parameter: the calibration is measured at THIS amplitude, so a unit calibrated at a
# different amplitude no longer matches. calkit detects that at load and runs
# UNCALIBRATED (baked levels) with a loud warning until it is re-calibrated here.
AMPLITUDE = 0.5

# Hardware TX-gain ceiling of the B200-mini (dB) — the physical maximum, distinct
# from GAIN_AT_MAX_DB. The (normally-commented) calibration gain knob uses it.
HW_MAX_GAIN_DB = 89.75

# Derived delivered-power limits (computed — do not edit).
MAX_DELIVERED_DBM = OUTPUT_POWER_DBM - CABLE_LOSS_DB + AMPLIFIER_GAIN_DB
MIN_DELIVERED_DBM = MAX_DELIVERED_DBM - GAIN_AT_MAX_DB


_PMAP = None


def power_map() -> PowerMap:
    """Active power map: the unit's injected calibration curve if present
    (SDR_CALIBRATION_FILE), else the baked constants above (identical to the old single-anchor
    slope-1 behaviour). Cached, so build_script and main share one — and so --power's schema
    bounds match the real operating range (calibrated → e.g. EIRP; else the baked SDR-port
    range)."""
    global _PMAP
    if _PMAP is None:
        _PMAP = PowerMap.load(PowerMap.from_linear(
            0.0, GAIN_AT_MAX_DB, MIN_DELIVERED_DBM, MAX_DELIVERED_DBM, AMPLITUDE))
    return _PMAP


def gain_for_power(delivered_dbm: float) -> float:
    """TX gain (dB) for a requested delivered power, through the active calibration (the
    unit's measured curve when present, else the baked anchor). Every caller keeps working."""
    return power_map().gain_for_power(float(delivered_dbm))


def power_for_gain(gain_db: float) -> float:
    """Delivered power (dBm) an actual hardware gain produces, through the active map."""
    return power_map().power_for_gain(float(gain_db))


# ── Constants ─────────────────────────────────────────────────────────────────

E5_HZ = 1191.795e6              # E5 centre carrier (between E5a and E5b)
E5A_HZ = 1176.45e6
E5B_HZ = 1207.14e6

CHIP_RATE_HZ = 10.23e6         # E5 component chip rate
SUBCARRIER_HZ = 15.345e6       # AltBOC side-band sub-carrier rate (15×1.023)
PRIMARY_LEN = 10230
EPOCHS_PER_LOOP = 100          # 100 primary epochs (1 ms each) = 100 ms period
DEFAULT_SR_HZ = 61.38e6        # 6 samp/chip, 4 samp/sub-carrier period

# Feedback taps per component (ICD Table 16); register-1 start = all-ones.
TAPS = {
    "E5a-I": (0o40503, 0o50661),
    "E5a-Q": (0o40503, 0o50661),
    "E5b-I": (0o64021, 0o51445),
    "E5b-Q": (0o64021, 0o43143),
}

E5AI_S2 = {1:0o30305, 2:0o14234, 3:0o27213, 4:0o20577, 5:0o23312, 6:0o33463, 7:0o15614, 8:0o12537, 9:0o01527, 10:0o30236, 11:0o27344, 12:0o07272, 13:0o36377, 14:0o17046, 15:0o06434, 16:0o15405, 17:0o24252, 18:0o11631, 19:0o24776, 20:0o00630, 21:0o11560, 22:0o17272, 23:0o27445, 24:0o31702, 25:0o13012, 26:0o14401, 27:0o34727, 28:0o22627, 29:0o30623, 30:0o27256, 31:0o01520, 32:0o14211, 33:0o31465, 34:0o22164, 35:0o33516, 36:0o02737, 37:0o21316, 38:0o35425, 39:0o35633, 40:0o24655, 41:0o14054, 42:0o27027, 43:0o06604, 44:0o31455, 45:0o34465, 46:0o25273, 47:0o20763, 48:0o31721, 49:0o17312, 50:0o13277}
E5AQ_S2 = {1:0o25652, 2:0o05142, 3:0o24723, 4:0o31751, 5:0o27366, 6:0o24660, 7:0o33655, 8:0o27450, 9:0o07626, 10:0o01705, 11:0o12717, 12:0o32122, 13:0o16075, 14:0o16644, 15:0o37556, 16:0o02477, 17:0o02265, 18:0o06430, 19:0o25046, 20:0o12735, 21:0o04262, 22:0o11230, 23:0o00037, 24:0o06137, 25:0o04312, 26:0o20606, 27:0o11162, 28:0o22252, 29:0o30533, 30:0o24614, 31:0o07767, 32:0o32705, 33:0o05052, 34:0o27553, 35:0o03711, 36:0o02041, 37:0o34775, 38:0o05274, 39:0o37356, 40:0o16205, 41:0o36270, 42:0o06600, 43:0o26773, 44:0o17375, 45:0o35267, 46:0o36255, 47:0o12044, 48:0o26442, 49:0o21621, 50:0o25411}
E5BI_S2 = {1:0o07220, 2:0o26047, 3:0o00252, 4:0o17166, 5:0o14161, 6:0o02540, 7:0o01537, 8:0o26023, 9:0o01725, 10:0o20637, 11:0o02364, 12:0o27731, 13:0o30640, 14:0o34174, 15:0o06464, 16:0o07676, 17:0o32231, 18:0o10353, 19:0o00755, 20:0o26077, 21:0o11644, 22:0o11537, 23:0o35115, 24:0o20452, 25:0o34645, 26:0o25664, 27:0o21403, 28:0o32253, 29:0o02337, 30:0o30777, 31:0o27122, 32:0o22377, 33:0o36175, 34:0o33075, 35:0o33151, 36:0o13134, 37:0o07433, 38:0o10216, 39:0o35466, 40:0o02533, 41:0o05351, 42:0o30121, 43:0o14010, 44:0o32576, 45:0o30326, 46:0o37433, 47:0o26022, 48:0o35770, 49:0o06670, 50:0o12017}
E5BQ_S2 = {1:0o03331, 2:0o06143, 3:0o25322, 4:0o23371, 5:0o00413, 6:0o36235, 7:0o17750, 8:0o04745, 9:0o13005, 10:0o37140, 11:0o30155, 12:0o20237, 13:0o03461, 14:0o31662, 15:0o27146, 16:0o05547, 17:0o02456, 18:0o30013, 19:0o00322, 20:0o10761, 21:0o26767, 22:0o36004, 23:0o30713, 24:0o07662, 25:0o21610, 26:0o20134, 27:0o11262, 28:0o10706, 29:0o34143, 30:0o11051, 31:0o25460, 32:0o17665, 33:0o32354, 34:0o21230, 35:0o20146, 36:0o11362, 37:0o37246, 38:0o16344, 39:0o15034, 40:0o25471, 41:0o25646, 42:0o22157, 43:0o04336, 44:0o16356, 45:0o04075, 46:0o02626, 47:0o11706, 48:0o37011, 49:0o27041, 50:0o31024}


CS201 = "842E9"   # E5a-I 20-chip secondary (all SVIDs)
CS41  = "E"          # E5b-I  4-chip secondary (all SVIDs)
E5AQ_SEC = {
     1: "83F6F69D8F6E15411FB8C9B1C",
     2: "66558BD3CE0C7792E83350525",
     3: "59A025A9C1AF0651B779A8381",
     4: "D3A32640782F7B18E4DF754B7",
     5: "B91FCAD7760C218FA59348A93",
     6: "BAC77E933A779140F094FBF98",
     7: "537785DE280927C6B58BA6776",
     8: "EFCAB4B65F38531ECA22257E2",
     9: "79F8CAE838475EA5584BEFC9B",
    10: "CA5170FEA3A810EC606B66494",
    11: "1FC32410652A2C49BD845E567",
    12: "FE0A9A7AFDAC44E42CB95D261",
    13: "B03062DC2B71995D5AD8B7DBE",
    14: "F6C398993F598E2DF4235D3D5",
    15: "1BB2FB8B5BF24395C2EF3C5A1",
    16: "2F920687D238CC7046EF6AFC9",
    17: "34163886FC4ED7F2A92EFDBB8",
    18: "66A872CE47833FB2DFD5625AD",
    19: "99D5A70162C920A4BB9DE1CA8",
    20: "81D71BD6E069A7ACCBEDC66CA",
    21: "A654524074A9E6780DB9D3EC6",
    22: "C3396A101BEDAF623CFC5BB37",
    23: "C3D4AB211DF36F2111F2141CD",
    24: "3DFF25EAE761739265AF145C1",
    25: "994909E0757D70CDE389102B5",
    26: "B938535522D119F40C25FDAEC",
    27: "C71AB549C0491537026B390B7",
    28: "0CDB8C9E7B53F55F5B0A0597B",
    29: "61C5FA252F1AF81144766494F",
    30: "626027778FD3C6BB4BAA7A59D",
    31: "E745412FF53DEBD03F1C9A633",
    32: "3592AC083F3175FA724639098",
    33: "52284D941C3DCAF2721DDB1FD",
    34: "73B3D8F0AD55DF4FE814ED890",
    35: "94BF16C83BD7462F6498E0282",
    36: "A8C3DE1AC668089B0B45B3579",
    37: "E23FFC2DD2C14388AD8D6BEC8",
    38: "F2AC871CDF89DDC06B5960D2B",
    39: "06191EC1F622A77A526868BA1",
    40: "22D6E2A768E5F35FFC8E01796",
    41: "25310A06675EB271F2A09EA1D",
    42: "9F7993C621D4BEC81A0535703",
    43: "D62999EACF1C99083C0B4A417",
    44: "F665A7EA441BAA4EA0D01078C",
    45: "46F3D3043F24CDEABD6F79543",
    46: "E2E3E8254616BD96CEFCA651A",
    47: "E548231A82F9A01A19DB5E1B2",
    48: "265C7F90A16F49EDE2AA706C8",
    49: "364A3A9EB0F0481DA0199D7EA",
    50: "9810A7A898961263A0F749F56",
}
E5BQ_SEC = {
     1: "CFF914EE3C6126A49FD5E5C94",
     2: "FC317C9A9BF8C6038B5CADAB3",
     3: "A2EAD74B6F9866E414393F239",
     4: "72F2B1180FA6B802CB84DF997",
     5: "13E3AE93BC52391D09E84A982",
     6: "77C04202B91B22C6D3469768E",
     7: "FEBC592DD7C69AB103D0BB29C",
     8: "0B494077E7C66FB6C51942A77",
     9: "DD0E321837A3D52169B7B577C",
    10: "43DEA90EA6C483E7990C3223F",
    11: "0366AB33F0167B6FA979DAE18",
    12: "99CCBBFAB1242CBE31E1BD52D",
    13: "A3466923CEFDF451EC0FCED22",
    14: "1A5271F22A6F9A8D76E79B7F0",
    15: "3204A6BB91B49D1A2D3857960",
    16: "32F83ADD43B599CBFB8628E5B",
    17: "3871FB0D89DB77553EB613CC1",
    18: "6A3CBDFF2D64D17E02773C645",
    19: "2BCD09889A1D7FC219F2EDE3B",
    20: "3E49467F4D4280B9942CD6F8C",
    21: "658E336DCFD9809F86D54A501",
    22: "ED4284F345170CF77268C8584",
    23: "29ECCE910D832CAF15E3DF5D1",
    24: "456CCF7FE9353D50E87A708FA",
    25: "FB757CC9E18CBC02BF1B84B9A",
    26: "5686229A8D98224BC426BC7FC",
    27: "700A2D325EA14C4B7B7AA8338",
    28: "1210A330B4D3B507D854CBA3F",
    29: "438EE410BD2F7DBCDD85565BA",
    30: "4B9764CC455AE1F61F7DA432B",
    31: "BF1F45FDDA3594ACF3C4CC806",
    32: "DA425440FE8F6E2C11B8EC1A4",
    33: "EE2C8057A7C16999AFA33FED1",
    34: "2C8BD7D8395C61DFA96243491",
    35: "391E4BB6BC43E98150CDDCADA",
    36: "399F72A9EADB42C90C3ECF7F0",
    37: "93031FDEA588F88E83951270C",
    38: "BA8061462D873705E95D5CB37",
    39: "D24188F88544EB121E963FD34",
    40: "D5F6A8BB081D8F383825A4DCA",
    41: "0FA4A205F0D76088D08EAF267",
    42: "272E909FAEBC65215E263E258",
    43: "3370F35A674922828465FC816",
    44: "54EF96116D4A0C8DB0E07101F",
    45: "DE347C7B27FADC48EF1826A2B",
    46: "01B16ECA6FC343AE08C5B8944",
    47: "1854DB743500EE94D8FC768ED",
    48: "28E40C684C87370CD0597FAB4",
    49: "5E42C19717093353BCAAF4033",
    50: "64310BAD8EB5B36E38646AF01",
}

S2_BY_COMPONENT = {
    "E5a-I": E5AI_S2, "E5a-Q": E5AQ_S2,
    "E5b-I": E5BI_S2, "E5b-Q": E5BQ_S2,
}

# AltBOC 8-PSK phase-state look-up table (ICD Table 7). Rows = sub-period index
# iTs 0..7; columns = input quadruple encoded as
# col = 8·(e_aI>0) + 4·(e_bI>0) + 2·(e_aQ>0) + 1·(e_bQ>0); value = phase index k
# in 1..8 giving s = exp(j·π/4·k).
ALTBOC_K = [
    [5, 4, 4, 3, 6, 3, 1, 2, 6, 5, 7, 2, 7, 8, 8, 1],
    [5, 4, 8, 3, 2, 3, 1, 2, 6, 5, 7, 6, 7, 4, 8, 1],
    [1, 4, 8, 7, 2, 3, 1, 2, 6, 5, 7, 6, 3, 4, 8, 5],
    [1, 8, 8, 7, 2, 3, 1, 6, 2, 5, 7, 6, 3, 4, 4, 5],
    [1, 8, 8, 7, 2, 7, 5, 6, 2, 1, 3, 6, 3, 4, 4, 5],
    [1, 8, 4, 7, 6, 7, 5, 6, 2, 1, 3, 2, 3, 8, 4, 5],
    [5, 8, 4, 3, 6, 7, 5, 6, 2, 1, 3, 2, 7, 8, 4, 1],
    [5, 4, 4, 3, 6, 7, 5, 2, 6, 1, 3, 2, 7, 8, 8, 1],
]


# ── Code generation (pure Python) ──────────────────────────────────────────────

def _taps_vector(o):
    return [(o >> j) & 1 for j in range(1, 15)]


def _start_vector(o):
    return [(o >> (j - 1)) & 1 for j in range(1, 15)]


def primary_code(component: str, svid: int) -> list[int]:
    """10230-chip Galileo E5 primary code (ICD §3.4.1) for a component and SVID
    (1..50), as 0/1. reg1 all-ones, reg2 = per-code start value; out = reg1⊕reg2."""
    if not 1 <= svid <= 50:
        raise ValueError(f"SVID must be 1..50, got {svid}")
    t1, t2 = _taps_vector(TAPS[component][0]), _taps_vector(TAPS[component][1])
    c1 = [1] * 14
    c2 = _start_vector(S2_BY_COMPONENT[component][svid])
    out = []
    for _ in range(PRIMARY_LEN):
        out.append(c1[13] ^ c2[13])
        f1 = f2 = 0
        for j in range(14):
            f1 ^= t1[j] & c1[j]
            f2 ^= t2[j] & c2[j]
        c1 = [f1] + c1[:13]
        c2 = [f2] + c2[:13]
    return out


def _hex_bits(hexstr: str, length: int) -> list[int]:
    nbits = len(hexstr) * 4
    v = int(hexstr, 16)
    return [(v >> (nbits - 1 - i)) & 1 for i in range(nbits)][:length]


def secondary_code(band: str, channel: str, svid: int) -> list[int]:
    """Secondary overlay for band 'E5a'|'E5b', channel 'data'|'pilot', SVID."""
    if band == "E5a":
        return _hex_bits(CS201, 20) if channel == "data" else _hex_bits(E5AQ_SEC[svid], 100)
    return _hex_bits(CS41, 4) if channel == "data" else _hex_bits(E5BQ_SEC[svid], 100)


# ── Self-test ──────────────────────────────────────────────────────────────────

def _self_test() -> int:
    """Verify (a) the embedded E5 primary codes against the ICD first-24-chip
    'Initial Sequence' values (Tables 17–20, 200 codes) and (b) the AltBOC phase
    look-up table against the ICD's independent direct sub-carrier formula
    (Eq. 3–4, Table 6) over all 128 (iTs × quadruple) cases. Returns 0 on OK."""
    import cmath
    chk = {
     "E5a-I": {1:0x3CEA9D, 2:0x9D8CF1, 3:0x45D1C8, 4:0x7A0133, 5:0x64D423, 6:0x23300D, 7:0x91CEF2, 8:0xAA82DC, 9:0xF2A17D, 10:0x3D84AE, 11:0x446D38, 12:0xC514F2, 13:0x0C0184, 14:0x8767E0, 15:0xCB8EFF, 16:0x93EBCD, 17:0x5D55CE, 18:0xB19B7C, 19:0x5805FC, 20:0xF99EA1, 21:0xB23CE5, 22:0x8515E8, 23:0x436822, 24:0x30F77B, 25:0xA7D629, 26:0x9BFAC7, 27:0x18A25B, 28:0x69A39F, 29:0x39B27D, 30:0x454598, 31:0xF2BC62, 32:0x9DDBC6, 33:0x332827, 34:0x6E2FCA, 35:0x22C6D5, 36:0xE881D9, 37:0x74C4DB, 38:0x13AB03, 39:0x119323, 40:0x594886, 41:0x9F4D89, 42:0x47A3C0, 43:0xC9ED53, 44:0x334994, 45:0x1B2A30, 46:0x5513F3, 47:0x7831C1, 48:0x30B93A, 49:0x84D5B4, 50:0xA5029C},
     "E5a-Q": {1:0x515537, 2:0xD67539, 3:0x58B2E5, 4:0x305914, 5:0x442710, 6:0x593CF8, 7:0x214AD7, 8:0x435EA6, 9:0xC1A7D5, 10:0xF0E94A, 11:0xA8C239, 12:0x2EB63B, 13:0x8F0A46, 14:0x896DD4, 15:0x0245F1, 16:0xEB0160, 17:0xED28B3, 18:0xCB9F5B, 19:0x576592, 20:0xA88811, 21:0xDD3649, 22:0xB59F42, 23:0xFF81F6, 24:0xCE8128, 25:0xDCD55C, 26:0x79E450, 27:0xB63460, 28:0x6D562B, 29:0x3A9010, 30:0x59CD72, 31:0xC0211A, 32:0x28EB96, 33:0xD7554B, 34:0x425126, 35:0xE0DAFB, 36:0xEF79F2, 37:0x18085D, 38:0xD50CD8, 39:0x0447B9, 40:0x8DE877, 41:0x0D1FA0, 42:0xC9FCF7, 43:0x48116D, 44:0x840BCC, 45:0x152004, 46:0x0D4897, 47:0xAF6D25, 48:0x4B7593, 49:0x71BB1B, 50:0x53DA0E},
     "E5b-I": {1:0xC5BEA1, 2:0x4F6248, 3:0xFD5488, 4:0x86277B, 5:0x9E39D5, 6:0xEA7EDE, 7:0xF28321, 8:0x4FB0C9, 9:0xF0AB64, 10:0x79833B, 11:0xEC2D91, 12:0x409B11, 13:0x397E16, 14:0x1E0FCD, 15:0xCB2F5A, 16:0xC1079A, 17:0x2D9BC6, 18:0xBC5146, 19:0xF848B0, 20:0x4F01E8, 21:0xB16C9B, 22:0xB2827D, 23:0x16C809, 24:0x7B570F, 25:0x1969C0, 26:0x512FA9, 27:0x73F36B, 28:0x2D5317, 29:0xEC8390, 30:0x380374, 31:0x46B4DE, 32:0x6C01D9, 33:0x0E0BB6, 34:0x2708C7, 35:0x265B55, 36:0xA68E1C, 37:0xC3916E, 38:0xBDC595, 39:0x1327D0, 40:0xEA921F, 41:0xD45869, 42:0x3EB98A, 43:0x9FDE16, 44:0x2A04CA, 45:0x3CA56F, 46:0x03928A, 47:0x4FB5B9, 48:0x101EC7, 49:0xC91D4F, 50:0xAFC22B},
     "E5b-Q": {1:0xE49AF0, 2:0xCE701F, 3:0x54B709, 4:0x641AB1, 5:0xFBD0AE, 6:0x0D8BC9, 7:0x805FA5, 8:0xD86BA0, 9:0xA7E921, 10:0x067E55, 11:0x3E4B58, 12:0x7D82FB, 13:0xE33BC2, 14:0x31372C, 15:0x46676F, 16:0xD2613E, 17:0xEB443C, 18:0x3FD0B1, 19:0xFCB7CF, 20:0xB83815, 21:0x48224A, 22:0x0FEE25, 23:0x38D33B, 24:0xC135B9, 25:0x71DE13, 26:0x7E8CFB, 27:0xB536C3, 28:0xB8E68C, 29:0x1E7272, 30:0xB75B69, 31:0x533F65, 32:0x812B41, 33:0x2C4DE1, 34:0x759E2C, 35:0x7E6434, 36:0xB43640, 37:0x05671B, 38:0x8C6FE0, 39:0x978D4E, 40:0x5319BF, 41:0x516499, 42:0x6E4292, 43:0xDC86A3, 44:0x8C46BE, 45:0xDF0B03, 46:0xE9A5B2, 47:0xB0E553, 48:0x07DBAC, 49:0x4778E4, 50:0x37AF4F},
    }
    ok = True
    for comp in ("E5a-I", "E5a-Q", "E5b-I", "E5b-Q"):
        good = 0
        for svid in range(1, 51):
            code = primary_code(comp, svid)
            first24 = 0
            for b in code[:24]:
                first24 = (first24 << 1) | b
            if len(code) == PRIMARY_LEN and first24 == chk[comp][svid]:
                good += 1
        comp_ok = good == 50
        ok = ok and comp_ok
        print(f"{comp}: {good}/50 primary codes match ICD [{'OK' if comp_ok else 'FAIL'}]")

    # AltBOC table vs direct formula (ICD Eq. 3–4 + Table 6 AS/AP coefficients).
    r2 = math.sqrt(2.0)
    AS = [(r2 + 1) / 2, 0.5, -0.5, (-r2 - 1) / 2, (-r2 - 1) / 2, -0.5, 0.5, (r2 + 1) / 2]
    AP = [(-r2 + 1) / 2, 0.5, -0.5, (r2 - 1) / 2, (r2 - 1) / 2, -0.5, 0.5, (-r2 + 1) / 2]
    maxerr = 0.0
    for iTs in range(8):
        for c in range(16):
            eaI = 1 if (c >> 3) & 1 else -1
            ebI = 1 if (c >> 2) & 1 else -1
            eaQ = 1 if (c >> 1) & 1 else -1
            ebQ = 1 if (c >> 0) & 1 else -1
            edaI, edaQ = eaQ * ebI * ebQ, eaI * ebI * ebQ   # ICD Eq. 4 products
            edbI, edbQ = ebQ * eaI * eaQ, ebI * eaI * eaQ
            scS, scSd = AS[iTs], AS[(iTs - 2) % 8]
            scP, scPd = AP[iTs], AP[(iTs - 2) % 8]
            s = ((eaI + 1j * eaQ) * (scS - 1j * scSd)
                 + (ebI + 1j * ebQ) * (scS + 1j * scSd)
                 + (edaI + 1j * edaQ) * (scP - 1j * scPd)
                 + (edbI + 1j * edbQ) * (scP + 1j * scPd)) / (2 * r2)
            k = ALTBOC_K[iTs][c]
            maxerr = max(maxerr, abs(s - cmath.exp(1j * k * math.pi / 4)))
    table_ok = maxerr < 1e-9
    ok = ok and table_ok
    print(f"AltBOC table vs direct formula: max|Δ|={maxerr:.2e} over 128 cases "
          f"[{'OK' if table_ok else 'FAIL'}]")
    print("ALL E5 AltBOC CHECKS PASSED" if ok else "SELF-TEST FAILED")
    return 0 if ok else 1


# ── Baseband buffer (one seamless-looping 100 ms AltBOC period) ────────────────

def build_iq_buffer(svid: int, samp_rate_hz: float):
    """Build a complex64 constant-envelope AltBOC buffer for one whole number of
    100 ms E5 periods that is also an exact integer number of samples (seamless).
    Each sample is exp(j·π/4·k) from the ICD Table-7 look-up on the tiered code
    quadruple and the sub-period index. Returns (iq, n_samples, n_periods, spc)."""
    import numpy as np
    from fractions import Fraction

    sr = int(round(samp_rate_hz))
    cr = int(round(CHIP_RATE_HZ))
    sc = int(round(SUBCARRIER_HZ))
    if sr % cr != 0:
        raise ValueError(f"sample rate must be an integer multiple of "
                         f"{CHIP_RATE_HZ/1e6:g} MHz; got {samp_rate_hz/1e6:g} MHz")
    if (sr * 8) % sc != 0:
        raise ValueError(f"sample rate must place sub-carrier sub-periods on the "
                         f"grid (sr·8 divisible by {SUBCARRIER_HZ/1e6:g} MHz); "
                         f"got {samp_rate_hz/1e6:g} MHz")

    chips_per_loop = PRIMARY_LEN * EPOCHS_PER_LOOP        # 10230*100 = 1_023_000
    spp = Fraction(sr * chips_per_loop, cr)
    n_periods = spp.denominator
    n_samples = spp.numerator
    spc = sr // cr

    # Tiered ±1 signal for one component over the full 100 ms (logic→signal).
    def tiered(component: str, channel: str, band: str) -> "np.ndarray":
        prim = np.asarray(primary_code(component, svid), dtype=np.int8)   # (10230,)
        sec = np.asarray(secondary_code(band, channel, svid), dtype=np.int8)
        overlay = sec[np.arange(EPOCHS_PER_LOOP) % len(sec)]              # (100,)
        chips = (prim[None, :] ^ overlay[:, None]).reshape(-1)           # (1_023_000,)
        return (1 - 2 * chips).astype(np.int8)                          # ±1

    e_aI = tiered("E5a-I", "data", "E5a")
    e_aQ = tiered("E5a-Q", "pilot", "E5a")
    e_bI = tiered("E5b-I", "data", "E5b")
    e_bQ = tiered("E5b-Q", "pilot", "E5b")

    # Map each output sample to its chip and sub-period, then look up the phase.
    idx = np.arange(n_samples, dtype=np.int64)
    chip_of = (idx * cr) // sr                              # 0..chips_per_loop-1
    col = (((e_aI[chip_of] > 0).astype(np.int64) << 3)
           | ((e_bI[chip_of] > 0).astype(np.int64) << 2)
           | ((e_aQ[chip_of] > 0).astype(np.int64) << 1)
           | ((e_bQ[chip_of] > 0).astype(np.int64)))
    iTs = ((idx * (8 * sc)) // sr) % 8
    ktab = np.asarray(ALTBOC_K, dtype=np.int64)             # (8,16)
    k = ktab[iTs, col]
    phase = (math.pi / 4.0) * k
    iq = np.empty(n_samples, dtype=np.complex64)
    iq.real = np.cos(phase).astype(np.float32)
    iq.imag = np.sin(phase).astype(np.float32)              # |iq| = 1 (const env)
    return iq, n_samples, n_periods, spc


# ── Flowgraph ──────────────────────────────────────────────────────────────────

def _build_top_block(iq_path, center_freq_hz, samp_rate_hz, gain_db, amplitude,
                     otw_format, extra_args):
    from gnuradio import gr, blocks, uhd

    class E5AltBocTx(gr.top_block):
        def __init__(self):
            super().__init__("Galileo E5 AltBOC TX")
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

    return E5AltBocTx()


# ── Parameter schema ────────────────────────────────────────────────────────────

def build_script() -> Script:
    return (
        Script("Galileo E5 AltBOC(15,10) transmitter — full wideband E5 (E5a+E5b, "
               "constant-envelope 8-PSK, real tiered codes), file-replay. "
               "Transmit only into an authorised, shielded setup.")
        .integer("-SVID", "--svid", min=1, max=50, default=1,
                 help="Galileo SVID / primary-code number (1..50; operational "
                      "1..36). Fixed per run.")
        .number("-Power", "--power", unit="dBm",
                min=round(power_map().min_power_dbm, 2),
                max=round(power_map().max_power_dbm, 2),
                default=round(power_map().max_power_dbm, 2), required=True, live=True,
                help="Target output power at the delivered plane (after cable loss + "
                     "amplifier gain). Max = what the SDR produces at its calibrated "
                     "max gain; raise it by editing the calibration constants.")
        .choice("-RF", "--rf", options=["on", "off"], default="on",
                required=False, live=True,
                help="RF output on/off. OFF mutes the signal (gain AND baseband "
                     "amplitude to 0); ON restores them. Change the power (or the "
                     "calibration gain) while OFF and it takes effect when you turn ON.")
        # RELATIVE power (also the calibration knob): the SDR's raw TX gain (dB), bypassing
        # the dBm mapping. No default, so its PRESENCE selects relative mode and OVERRIDES
        # --power. Set it while measuring output vs gain to fill in OUTPUT_POWER_DBM /
        # GAIN_AT_MAX_DB above.
        .number("-Gain", "--gain", unit="dB",
                min=0, max=HW_MAX_GAIN_DB, required=False, live=True,
                help="RELATIVE power: set the SDR's raw TX gain (dB) directly, bypassing "
                     "the dBm calibration. When given, overrides --power. Live.")
        .number("-Sample-rate", "--samp_rate", unit="MHz", min=30.69, max=61.44,
                default=DEFAULT_SR_HZ / 1e6,
                help="Host/DAC sample rate; master clock pinned equal to it (1:1). "
                     "Must be a multiple of 10.23 MHz with sub-carrier sub-periods "
                     "on-grid. Default/recommended 61.38 MHz (both ±15.345 MHz "
                     "sidebands inside Nyquist). Fixed per run.")
        .choice("-OTW-format", "--otw", options=["sc8", "sc16"], default="sc8",
                help="Over-the-wire sample format. sc8 halves USB load (needed at "
                     "61.38 MS/s on a Pi); sc16 for more dynamic range.")
    )


def _apply_live_change(tb, ctrl, state, name, value):
    # power/gain edits are staged into state["gain"] and only reach the radio when
    # RF is on; the --rf toggle mutes/restores gain AND baseband amplitude.
    if name == "power":
        state["gain"] = gain_for_power(float(value))
        if state["rf_on"]:
            tb.set_gain(state["gain"])
            ctrl.report("power", round(power_for_gain(tb.actual_gain()), 2))
        else:
            ctrl.report("power", round(power_for_gain(state["gain"]), 2))
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
            tb.set_amplitude(AMPLITUDE)
            tb.set_gain(state["gain"])
        else:
            tb.set_gain(0.0)
            tb.set_amplitude(0.0)
        ctrl.report("rf", "on" if on else "off")


# ── Entry point ─────────────────────────────────────────────────────────────────

def main() -> int:
    if "--self-test" in sys.argv[1:]:
        return _self_test()

    import atexit
    import shutil
    import tempfile

    script = build_script()
    args = script.parse()
    if args.svid not in E5AI_S2:
        print(f"SVID {args.svid} out of range 1..50", file=sys.stderr)
        return 2

    samp_rate_hz = args.samp_rate * 1e6
    # A raw calibration gain (the normally-commented --gain knob) overrides the dBm
    # mapping when present, so you can measure output power at a chosen gain.
    gain_cal = getattr(args, "gain", None)
    gain_db = float(gain_cal) if gain_cal is not None else gain_for_power(args.power)
    shm = "/dev/shm" if os.path.isdir("/dev/shm") else None
    tmpdir = tempfile.mkdtemp(prefix="gal_e5ab_", dir=shm)
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

    try:
        iq, nsamp, nper, spc = build_iq_buffer(args.svid, samp_rate_hz)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    iq_path = os.path.join(tmpdir, f"e5altboc_svid{args.svid}.fc32")
    iq.tofile(iq_path)
    print(f"[prebuilt] E5 AltBOC SVID {args.svid} → {nsamp} samples "
          f"({nper}×100 ms period, {spc} samp/chip, {nsamp*8/1e6:.1f} MB) → {iq_path}")

    tb = _build_top_block(iq_path, E5_HZ, samp_rate_hz, gain_db, AMPLITUDE,
                          args.otw, "")

    # RF on/off state + the gain RF-on applies. Starting with --rf off builds the
    # flow muted; power/gain edits made while OFF are staged and reach the radio
    # only when RF is switched ON.
    state = {"rf_on": getattr(args, "rf", "on") == "on", "gain": gain_db}
    if not state["rf_on"]:
        tb.set_gain(0.0)
        tb.set_amplitude(0.0)

    print("── Galileo E5 AltBOC(15,10) TX ─────────────────────────────")
    print(f"  signal         : full E5 (E5a lower + E5b upper), constant-envelope")
    print(f"  SVID           : {args.svid}")
    print(f"  carrier        : {E5_HZ/1e6:.3f} MHz  (E5a {E5A_HZ/1e6:.2f} / "
          f"E5b {E5B_HZ/1e6:.2f})")
    print(f"  sample rate    : requested {args.samp_rate:g} MHz, "
          f"got {tb.actual_samp_rate()/1e6:.6f} MHz (1:1 master clock)")
    print(f"  sub-carrier    : 15.345 MHz AltBOC, chip 10.23 Mcps")
    print(f"  power (target) : {args.power:g} dBm delivered "
          f"(cable −{CABLE_LOSS_DB:g} dB, amp +{AMPLIFIER_GAIN_DB:g} dB)")
    print(f"  → gain         : {gain_db:.2f} dB (max {power_map().max_gain_db:g}), "
          f"amplitude {AMPLITUDE:g}")
    print(f"  calibration    : {power_map().source}")
    if power_map().warning:                # calibration measured at another amplitude
        print(f"  ⚠ CALIBRATION  : {power_map().warning}")
    print(f"  RF             : {'ON' if state['rf_on'] else 'OFF (muted)'}")
    if gain_cal is not None:
        print("  ⚠ CALIBRATION  : raw --gain knob active — overrides --power")
    print(f"  otw            : {args.otw}")
    print("────────────────────────────────────────────────────────────")
    sys.stdout.flush()

    ctrl = script.live_control(args)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    tb.start()
    try:
        while not stop.is_set():
            for change in ctrl.drain():
                _apply_live_change(tb, ctrl, state, change.name, change.value)
            time.sleep(0.1)
    finally:
        ctrl.close()
        tb.stop()
        tb.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

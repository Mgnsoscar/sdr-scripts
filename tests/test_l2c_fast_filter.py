"""gps_l2c_tx.py — the fast real-FFT filter path (a warm-up speed-up for the 1.5 s CL loop).

The L2C base is real BPSK (Q=0), so build_l2c_buffer now returns a REAL float32 loop and
_circular_convolve filters it with a real-FFT (rfft/irfft) overlap-add that accumulates straight
into the real slots of the complex64 output — ~2x faster to build+filter the 92-Msample loop with
NO change to the transmitted samples. These tests pin that contract without a radio:

  • the base is real float32; the filtered loop is complex64 with an EXACTLY-zero imaginary part;
  • the real fast path reproduces the original complex path bit-for-bit (to float rounding);
  • a filter at least as long as the loop (m >= n) is REFUSED, never silently truncated (review
    fix #22), and filter_buffer's tap budget (max_taps = n // 2) keeps every shipped configuration
    (loop 'cm' / 'full', every sidelobe count) clear of that branch.
"""
import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

_L2C = (Path(__file__).resolve().parents[1]
        / "Raspberry pi + b206 mini-i" / "PRN GPS" / "gps_l2c_tx.py")


def _load():
    # paramkit lives in the sibling sdr-agent; put it on the path so the module imports.
    import sys
    for p in Path(__file__).resolve().parents:
        agent = p / "sdr-agent"
        if (agent / "paramkit").is_dir():
            sys.path.insert(0, str(agent))
            break
    spec = importlib.util.spec_from_file_location("gps_l2c_tx", _L2C)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


l2c = _load()


def test_base_is_real_float32_and_filtered_is_zero_imag_complex64():
    base, n = l2c.build_l2c_buffer(1, "cm")           # 20 ms loop — small + fast
    assert base.dtype == np.float32
    assert np.isrealobj(base)
    assert set(np.unique(base).tolist()) <= {-1.0, 1.0}   # BPSK ±1
    filt, taps, fp = l2c.filter_buffer(base, sidelobes=2, trans_hz=l2c.TRANS_HZ)
    assert filt.dtype == np.complex64
    assert np.max(np.abs(filt.imag)) == 0.0           # a real signal through a real filter


def test_real_fast_path_matches_the_complex_path():
    """The real overlap-add must reproduce the original complex path (what the radio used to get)."""
    base, n = l2c.build_l2c_buffer(3, "cm")
    h, m = l2c._design_lowpass((2 + 1) * l2c.L2C_NULL_HZ + l2c.TRANS_HZ / 2.0,
                               l2c.TRANS_HZ, n // 2)
    real_out = l2c._circular_convolve(base, h)                     # fast real path
    cplx_out = l2c._circular_convolve(base.astype(np.complex64), h)  # original complex path
    assert real_out.dtype == np.complex64 and cplx_out.dtype == np.complex64
    assert np.max(np.abs(real_out - cplx_out)) < 1e-5             # float32 rounding only


def test_a_filter_at_least_as_long_as_the_loop_is_refused_not_truncated():
    """m >= n: `np.fft.fft(h, n)` would silently TRUNCATE the FIR to n taps — not a circular
    convolution with the full filter (that would alias h modulo n). The branch now refuses
    loudly, for a real AND a complex input; one tap short of the loop is still a valid (proper,
    overlap-add) circular convolution and matches the monolithic reference."""
    rng = np.random.default_rng(0)
    x = (1.0 - 2.0 * rng.integers(0, 2, size=97)).astype(np.float32)   # real ±1
    h = l2c._design_lowpass(3e6, l2c.TRANS_HZ, 4096)[0]                # m >> n
    assert len(h) >= len(x)
    with pytest.raises(ValueError, match="filter longer than the loop"):
        l2c._circular_convolve(x, h)
    with pytest.raises(ValueError, match="filter longer than the loop"):
        l2c._circular_convolve(x.astype(np.complex64), h)
    # the boundary: exactly m == n is refused too …
    h_eq = h[:len(x)] / h[:len(x)].sum()
    with pytest.raises(ValueError, match="m >= n"):
        l2c._circular_convolve(x, h_eq)
    with pytest.raises(ValueError, match="m >= n"):
        l2c._circular_convolve(x.astype(np.complex64), h_eq)
    # … while m == n − 1 goes through overlap-add and IS the full circular convolution.
    h_ok = h[:len(x) - 1] / h[:len(x) - 1].sum()
    ref = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h_ok, len(x))).astype(np.complex64)
    real_out = l2c._circular_convolve(x, h_ok)
    cplx_out = l2c._circular_convolve(x.astype(np.complex64), h_ok)
    assert real_out.dtype == np.complex64 and cplx_out.dtype == np.complex64
    assert np.max(np.abs(real_out - ref)) < 1e-4
    assert np.max(np.abs(cplx_out - ref)) < 1e-4
    assert np.max(np.abs(real_out.imag)) == 0.0


def test_filter_buffer_taps_are_always_shorter_than_the_loop():
    """The refusing branch must stay UNREACHABLE from the transmit path: for both shipped loops
    ('cm' = one 20 ms CM period, 'full' = one 1.5 s CL period) and EVERY sidelobe count the
    schema admits, the designed tap count is < n. filter_buffer caps the design at
    max_taps = n // 2, so this holds by construction; the shipped skirt asks for far fewer taps
    than either budget anyway (the full loop is exercised through the design, not built — 92 M
    samples — its length follows from the CM loop: CL is exactly CL_LEN/CM_LEN CM periods)."""
    base_cm, n_cm = l2c.build_l2c_buffer(1, "cm")
    assert len(base_cm) == n_cm
    assert l2c.CL_LEN % l2c.CM_LEN == 0
    n_full = n_cm * (l2c.CL_LEN // l2c.CM_LEN)
    assert n_full == int(round(l2c.CL_LEN / l2c.CHANNEL_CHIP_RATE * round(l2c.SAMP_RATE_HZ)))
    wanted = int(np.ceil(5.5 * l2c.SAMP_RATE_HZ / l2c.TRANS_HZ)) | 1    # the skirt's own ask
    for n in (n_cm, n_full):
        for sidelobes in range(0, l2c.MAX_SIDELOBES + 1):
            fp = (sidelobes + 1) * l2c.L2C_NULL_HZ
            fc = fp + l2c.TRANS_HZ / 2.0
            h, m = l2c._design_lowpass(fc, l2c.TRANS_HZ, n // 2)     # exactly filter_buffer's design
            assert len(h) == m
            assert m <= min(wanted, (n // 2) | 1)
            assert m < n, (n, sidelobes, m)
    # and the real call path on the CM loop, at the extremes of the sidelobe range
    for sidelobes in (0, l2c.DEFAULT_SIDELOBES, l2c.MAX_SIDELOBES):
        filt, taps, fp = l2c.filter_buffer(base_cm, sidelobes=sidelobes, trans_hz=l2c.TRANS_HZ)
        assert taps < n_cm and len(filt) == n_cm


def test_filter_is_periodic_no_seam():
    """Circular convolution keeps the filtered loop periodic — the wrap sample equals the start."""
    base, n = l2c.build_l2c_buffer(5, "cm")
    filt, _, _ = l2c.filter_buffer(base, sidelobes=0, trans_hz=l2c.TRANS_HZ)
    # a short block-FFT overlap-add with tail-aliasing is exactly circular, so filtering the
    # DOUBLED loop and taking one period reproduces the single-period filter (no seam energy).
    dbl = np.concatenate([base, base])
    filt2, _, _ = l2c.filter_buffer(dbl, sidelobes=0, trans_hz=l2c.TRANS_HZ)
    assert np.max(np.abs(filt2[:n] - filt)) < 1e-4

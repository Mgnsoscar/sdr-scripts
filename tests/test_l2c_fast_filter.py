"""gps_l2c_tx.py — the fast real-FFT filter path (a warm-up speed-up for the 1.5 s CL loop).

The L2C base is real BPSK (Q=0), so build_l2c_buffer now returns a REAL float32 loop and
_circular_convolve filters it with a real-FFT (rfft/irfft) overlap-add that accumulates straight
into the real slots of the complex64 output — ~2x faster to build+filter the 92-Msample loop with
NO change to the transmitted samples. These tests pin that contract without a radio:

  • the base is real float32; the filtered loop is complex64 with an EXACTLY-zero imaginary part;
  • the real fast path reproduces the original complex path bit-for-bit (to float rounding);
  • the tiny-loop (m >= n) branch and a direct monolithic circular convolution agree too.
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


def test_tiny_loop_branch_real_equals_complex_and_reference():
    """m >= n (filter longer than the loop) takes the direct-DFT branch; real == complex == ref."""
    rng = np.random.default_rng(0)
    x = (1.0 - 2.0 * rng.integers(0, 2, size=97)).astype(np.float32)   # real ±1
    h = l2c._design_lowpass(3e6, l2c.TRANS_HZ, 4096)[0]                # m >> n → tiny branch
    assert len(h) >= len(x)
    real_out = l2c._circular_convolve(x, h)
    cplx_out = l2c._circular_convolve(x.astype(np.complex64), h)
    ref = np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, len(x)))          # monolithic circular conv
    assert real_out.dtype == np.complex64
    assert np.max(np.abs(real_out - cplx_out)) < 1e-5
    assert np.max(np.abs(real_out - ref.astype(np.complex64))) < 1e-4
    assert np.max(np.abs(real_out.imag)) == 0.0


def test_filter_is_periodic_no_seam():
    """Circular convolution keeps the filtered loop periodic — the wrap sample equals the start."""
    base, n = l2c.build_l2c_buffer(5, "cm")
    filt, _, _ = l2c.filter_buffer(base, sidelobes=0, trans_hz=l2c.TRANS_HZ)
    # a short block-FFT overlap-add with tail-aliasing is exactly circular, so filtering the
    # DOUBLED loop and taking one period reproduces the single-period filter (no seam energy).
    dbl = np.concatenate([base, base])
    filt2, _, _ = l2c.filter_buffer(dbl, sidelobes=0, trans_hz=l2c.TRANS_HZ)
    assert np.max(np.abs(filt2[:n] - filt)) < 1e-4

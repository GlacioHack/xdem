# test_regular_grid_log_pairs.py

import numpy as np
import pytest

try:
    import dask.array as da
except Exception:  # pragma: no cover
    da = None

from xdem._metricspace import RegularGridLogProbabilisticPairs


def _make_random_nan_grid(ny=120, nx=160, nan_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(ny, nx)).astype(np.float32)
    mask = rng.random(size=(ny, nx)) < nan_frac
    arr[mask] = np.nan
    return arr


def _finite_pair_count(v1, v2):
    return int(np.count_nonzero(np.isfinite(v1) & np.isfinite(v2)))


def _log_uniform_sanity(r, n_bins=10):
    lr = np.log(r)
    hist, _ = np.histogram(lr, bins=n_bins)
    # Avoid super strict requirements; just ensure no bin is totally starved
    # and distribution isn't insanely skewed.
    return hist.min(), hist.max()


def test_numpy_runs_and_nan_oversampling_reasonable():
    arr = _make_random_nan_grid(ny=120, nx=160, nan_frac=0.2, seed=42)

    sampler = RegularGridLogProbabilisticPairs(
        array=arr,
        dx=30.0,
        dy=30.0,
        samples=5000,
        seed=123,
        max_oversample=10.0,
    )

    v1, v2, r = sampler.sample_values()

    # NumPy path returns NumPy arrays eagerly
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert isinstance(r, np.ndarray)

    n_valid_pairs = _finite_pair_count(v1, v2)

    # With nan_frac=0.2, f_valid=0.8, expected valid pairs ~ samples (by design).
    # Allow some randomness tolerance.
    assert n_valid_pairs >= 0.8 * sampler.samples
    assert n_valid_pairs <= 1.2 * sampler.samples

    mn, mx = _log_uniform_sanity(r, n_bins=12)
    assert mn > 0  # shouldn't completely starve any log-distance bin
    assert mx / max(mn, 1) < 10  # very loose skew bound


@pytest.mark.skipif(da is None, reason="dask is not installed")
def test_dask_is_lazy_and_nan_oversampling_reasonable():
    arr_np = _make_random_nan_grid(ny=120, nx=160, nan_frac=0.2, seed=43)

    # Make a Dask array with chunking (simulates out-of-core)
    arr = da.from_array(arr_np, chunks=(40, 50))

    sampler = RegularGridLogProbabilisticPairs(
        array=arr,
        dx=30.0,
        dy=30.0,
        samples=5000,
        seed=456,
        max_oversample=10.0,
    )

    v1, v2, r = sampler.sample_values()

    # Dask path should return lazy arrays for values
    assert isinstance(v1, da.Array)
    assert isinstance(v2, da.Array)
    assert isinstance(r, np.ndarray)

    # Compute the sampled values now
    v1c, v2c = da.compute(v1, v2)
    n_valid_pairs = _finite_pair_count(v1c, v2c)

    assert n_valid_pairs >= 0.8 * sampler.samples
    assert n_valid_pairs <= 1.2 * sampler.samples

    mn, mx = _log_uniform_sanity(r, n_bins=12)
    assert mn > 0
    assert mx / max(mn, 1) < 10

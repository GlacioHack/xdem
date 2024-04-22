"""Tests for dask functions (temporarily in xDEM, might move to GeoUtils)."""
import os

import numpy as np
import pytest
import xarray as xr

from xdem.examples import _EXAMPLES_DIRECTORY
from xdem.coreg.delayed import delayed_subsample, delayed_interp_points, _random_state_from_user_input

class TestDelayed:

    # Create test file (replace with data in time?)
    fn_tmp = os.path.join(_EXAMPLES_DIRECTORY, "test.nc")
    if not os.path.exists(fn_tmp):
        data = np.random.normal(size=400000000).reshape(20000, 20000)
        da = xr.DataArray(data=data, dims=["x", "y"])
        ds = xr.Dataset(data_vars={"test": da})
        encoding_kwargs = {"test": {"chunksizes": (100, 100)}}
        ds.to_netcdf(fn_tmp, encoding=encoding_kwargs)
        del ds, da, data

    # Chunk size in memory
    chunksize = 500

    @pytest.mark.parametrize("subsample_size", [2, 100, 100000])
    def test_delayed_subsample(self, subsample_size: int):
        """Checks for delayed subsampling function."""

        # Open dataset with chunks
        ds = xr.open_dataset(self.fn_tmp, chunks={"x": self.chunksize, "y": self.chunksize})
        darr = ds["test"].data

        # Derive subsample from delayed function
        sub = delayed_subsample(darr, subsample=subsample_size, random_state=42)

        # The subsample should have exactly the prescribed length, with only valid values
        assert len(sub) == subsample_size
        assert all(np.isfinite(sub))

        # To verify the sampling works correctly, we can get its subsample indices with the argument return_indices
        # And compare to the same subsample with vindex (now that we know the coordinates of valid values sampled)
        indices = delayed_subsample(darr, subsample=subsample_size, random_state=42, return_indices=True)
        sub2 = np.array(darr.vindex[indices[0], indices[1]])
        assert np.array_equal(sub, sub2)

    @pytest.mark.parametrize("ninterp", [2, 100, 100000])
    def test_delayed_interp_points(self, ninterp: int):
        """Checks for delayed interpolate points function."""

        # Open dataset with chunks
        ds = xr.open_dataset(self.fn_tmp, chunks={"x": self.chunksize, "y": self.chunksize})
        darr = ds["test"].data

        # Create random point coordinates to interpolate
        rng = np.random.default_rng(seed=42)
        interp_x = rng.choice(ds.x.size, ninterp) + rng.random(ninterp)
        interp_y = rng.choice(ds.y.size, ninterp) + rng.random(ninterp)

        # Interpolate with delayed function
        interp1 = delayed_interp_points(darr=darr, points=(interp_x, interp_y), resolution=(1, 1))

        # Interpolate directly with Xarray and compare results are the same
        xx = xr.DataArray(interp_x, dims='z', name='x')
        yy = xr.DataArray(interp_y, dims='z', name='y')
        interp2 = ds.test.interp(x=xx, y=yy)
        interp2.compute()
        interp2 = np.array(interp2.values)

        assert np.array_equal(interp1, interp2, equal_nan=True)



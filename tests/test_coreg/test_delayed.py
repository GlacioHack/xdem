"""Tests for dask functions (temporarily in xDEM, might move to GeoUtils)."""
import os

import numpy as np
import pytest
import xarray as xr
import dask.array as da
from pyproj import CRS
import rasterio as rio
import dask

from xdem.examples import _EXAMPLES_DIRECTORY
from xdem.coreg.delayed import delayed_subsample, delayed_interp_points, delayed_reproject

class TestDelayed:

    # Create test file (replace with data in time?)
    fn_tmp = os.path.join(_EXAMPLES_DIRECTORY, "test.nc")
    if not os.path.exists(fn_tmp):
        data = np.random.normal(size=400000000).reshape(20000, 20000)
        data_arr = xr.DataArray(data=data, dims=["x", "y"])
        ds = xr.Dataset(data_vars={"test": data_arr})
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

    # Let's check a lot of different scenarios
    def test_delayed_reproject(self):

        # Open dataset with chunks
        ds = xr.open_dataset(self.fn_tmp, chunks={"x": self.chunksize, "y": self.chunksize})
        darr = ds["test"].data

        dask.config.set(scheduler='single-threaded')

        src_shape = darr.shape

        # src_shape = (150, 100)
        # src_chunksizes = (25, 10)
        # rng = da.random.default_rng(seed=42)
        # darr = rng.normal(size=src_shape, chunks=src_chunksizes)
        # darr = da.ones(src_shape, chunks=src_chunksizes)
        src_crs = CRS(4326)
        src_transform = rio.transform.from_bounds(0, 0, 5, 5, src_shape[0], src_shape[1])

        dst_shape = (77, 50)
        dst_crs = CRS(32631)
        dst_chunksizes = (7, 5)

        # Build an intersecting dst_transform that is not aligned
        src_res = (src_transform[0], abs(src_transform[4]))
        bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(src_shape[0], src_shape[1], src_transform))
        # First, an aligned transform in the new CRS that allows to get
        # temporary new bounds and resolution in the units of the new CRS
        tmp_transform = rio.warp.calculate_default_transform(
            src_crs,
            dst_crs,
            src_shape[1],
            src_shape[0],
            left=bounds.left,
            right=bounds.right,
            top=bounds.top,
            bottom=bounds.bottom,
            dst_width=dst_shape[1],
            dst_height=dst_shape[0],
        )[0]
        tmp_res = (tmp_transform[0], abs(tmp_transform[4]))
        tmp_bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(dst_shape[0], dst_shape[1], tmp_transform))
        # Now we modify the destination grid by changing bounds by a bit + the resolution
        dst_transform = rio.transform.from_origin(tmp_bounds.left + 100*tmp_res[0], tmp_bounds.top + 150*tmp_res[0],
                                                  tmp_res[0]*2.5, tmp_res[1]*0.7)

        # Other arguments
        src_nodata = -9999
        dst_nodata = 99999
        resampling = rio.enums.Resampling.bilinear

        # Run delayed reproject
        reproj_arr = delayed_reproject(darr, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform,
                                       dst_crs=dst_crs, dst_shape=dst_shape, src_nodata=src_nodata, dst_nodata=dst_nodata,
                                       resampling=resampling, dst_chunksizes=dst_chunksizes)

        # Save file out-of-memory
        # TODO: Would need to wrap the georef data in the netCDF, but not needed to test this
        fn_tmp_out = os.path.join(_EXAMPLES_DIRECTORY, "test_reproj.nc")
        data_arr = xr.DataArray(data=reproj_arr, dims=["x", "y"])
        ds_out = xr.Dataset(data_vars={"test_reproj": data_arr})
        write_delayed = ds_out.to_netcdf(fn_tmp_out, compute=False)
        write_delayed.compute()

        # Load in-memory and compare with a direct reproject
        dst_arr = np.zeros(dst_shape)
        _ = rio.warp.reproject(
            np.array(darr),
            dst_arr,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=dst_nodata,
        )

        assert np.allclose(reproj_arr.compute(), dst_arr, atol=0.02)

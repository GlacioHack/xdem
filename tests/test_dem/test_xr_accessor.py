"""
Test module for 'dem' Xarray accessor mirroring DEM API.
Most function tests are actually located in "test_base", to check consistently for equality, loading and lazy behaviour
across the entire API.
"""

import numpy as np
import pytest

from xdem import examples, open_dem


class TestAccessor:
    """
    Test for Xarray accessor subclass.

    Note: This test class only tests functionalities that are specific to the DEMAccessor subclass. Overridden
    abstract methods, loading behaviour and Dask laziness are tested in test_base directly to mirror DEM tests.

    This class thus tests:
    - The open_dem function,
    - The instantiation __init__ through ds.dem,
    - The to_geoutils() method.
    """

    longyearbyen_path = examples.get_path_test("longyearbyen_ref_dem")

    def test_open_raster(self) -> None:
        pass

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])
    def test_copy(self, path_dem: str) -> None:

        ds = open_dem(path_dem)
        ds_copy = ds.rst.copy()

        assert np.array_equal(ds.data, ds_copy.data, equal_nan=True)
        assert ds.rst.transform == ds_copy.rst.transform
        assert ds.rst.crs == ds_copy.rst.crs
        assert ds.rst.nodata == ds_copy.rst.nodata

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])
    def test_open__loaded(self, path_dem: str) -> None:
        """
        Test that a DataArray opened using "open_raster" maintains implicit loading logic.

        Tests checking loading for all attributes and methods are done in TestBase.

        Note: this is different from using lazy Dask arrays: for any array type, Xarray only loads metadata, and
        implicitly loads data in memory when .data or .load() is called.
        """

        # Open raster with/without chunks, should not load in memory yet
        ds = open_dem(path_dem)
        assert not ds._in_memory

        # The array should be NumPy
        assert isinstance(ds.data, np.ndarray)
        ds.load()
        assert ds._in_memory

    @pytest.mark.parametrize("path_dem", [longyearbyen_path])
    def test_open__dask(self, path_dem: str) -> None:
        """
        Check that a DataArray opened with chunks using "open_raster" maintains Dask laziness.

        Note: this is different from loading mechanism of Xarray (triggers when calling .data).
        """
        pytest.importorskip("dask")
        import dask.array as da

        # Open raster lazily with chunks
        ds = open_dem(path_dem, chunks={"band": 1, "x": 10, "y": 10})

        # Array should be a Dask array (chunks exist)
        ds_arr = ds.data
        assert not ds._in_memory
        assert isinstance(ds_arr, da.Array)
        assert ds_arr.chunks is not None

        # After compute, it should be a NumPy array
        ds_comp = ds.compute()
        assert isinstance(ds_comp.data, np.ndarray)
        assert ds_comp._in_memory

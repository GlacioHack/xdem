"""
Test module for 'dem' Xarray accessor mirroring DEM API.
Most function tests are actually located in "test_base", to check consistently for equality, loading and lazy behaviour
across the entire API.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from geoutils import Raster
from pandas.testing import assert_frame_equal
from pyproj import CRS

from xdem import DEM, EPC, examples, open_dem


class TestAccessor:
    """
    Test for Xarray accessor subclass.

    Note: This test class only tests functionalities that are specific to the DEMAccessor subclass. Overridden
    abstract methods, loading behaviour and Dask laziness are tested in test_base directly to mirror DEM tests.

    This class thus tests:
    - The open_dem function,
    - The instantiation __init__ through ds.dem and its single-band constraint,
    - Copying and conversion between DataArray, DEM and GeoUtils Raster objects,
    - Preservation of loading, Dask laziness, VCRS and raster metadata through those conversions,
    - File and point-cloud outputs specific to the DEM accessor.

    DEMBase methods are compared automatically between DEM and this accessor in test_base. Chunked terrain and VCRS
    algorithms have their exhaustive backend cases in the terrain and VCRS test modules.
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

    @pytest.mark.parametrize("method", ["to_xdem", "to_geoutils"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_methods__conversion_loading_laziness(self, accessor_dem_path: Path, method: str, lazy: bool) -> None:
        """Checks that native conversions load eager sources and preserve Dask graphs, values and metadata."""

        # 1/ Open the file without loading its values and save the Dask data object when used
        if lazy:
            pytest.importorskip("dask.array")
        source = open_dem(accessor_dem_path, vcrs=5703, chunks=13 if lazy else None)
        graph = source.data if lazy else None
        assert not source._in_memory

        # 2/ Convert to DEM or Raster and check that a Dask source remains unchanged and lazy
        result = getattr(source.dem, method)()
        assert type(result) is (DEM if method == "to_xdem" else Raster)
        assert result.is_loaded
        assert source._in_memory is not lazy
        if lazy:
            assert source.data is graph

        # 3/ Compare exact values, masks and metadata with an independent native reader
        expected = DEM(accessor_dem_path, vcrs=5703)
        assert expected.raster_equal(result, strict_masked=False, warn_failure_reason=True)
        assert result.tags == source.dem.tags
        assert source._in_memory is not lazy

    @pytest.mark.parametrize("loaded", [False, True])
    def test_to_xarray__loading(self, accessor_dem_path: Path, loaded: bool) -> None:
        """Checks that native Xarray export loads the source and roundtrips exact DEM values and metadata."""

        # 1/ Export native DEMs that start either loaded or unloaded
        source = DEM(accessor_dem_path, vcrs=5703, load_data=loaded)
        assert source.is_loaded is loaded
        result = source.to_xarray(name="elevation")
        assert isinstance(result, xr.DataArray)
        assert source.is_loaded
        assert result.name == "elevation"

        # 2/ Convert back through the DEM accessor and retain vertical and raster metadata
        converted = result.dem.to_xdem()
        assert isinstance(converted, DEM)
        assert source.raster_equal(converted, strict_masked=False, warn_failure_reason=True)
        for name, value in source.tags.items():
            assert converted.tags[name] == value

    @pytest.mark.parametrize(
        "file_vcrs, user_vcrs, expected_vcrs", [(None, None, 5773), (None, 6360, 6360), (5703, None, 5703)]
    )
    def test_open_dem__product_vertical_reference(
        self, accessor_dem_path: Path, file_vcrs: Any, user_vcrs: Any, expected_vcrs: int
    ) -> None:
        """Checks that product metadata supplies a datum only when the file and user do not define a vertical CRS."""

        # 1/ Write AW3D30 product metadata with or without an explicit file VCRS
        source = DEM(accessor_dem_path, vcrs=file_vcrs)
        source.tags["product"] = "AW3D30"
        path = accessor_dem_path.with_name("product.tif")
        source.to_file(path)

        # 2/ Compare native and accessor selection without loading elevation values
        native = DEM(path, vcrs=user_vcrs)
        ds = open_dem(path, vcrs=user_vcrs)
        assert native.vcrs == ds.dem.vcrs == CRS.from_epsg(expected_vcrs)
        assert not native.is_loaded and not ds._in_memory

    def test_open_dem__missing_horizontal_reference(self, accessor_dem_path: Path) -> None:
        """Checks that a DEM without a CRS opens normally and rejects vertical operations until its CRS is defined."""

        # 1/ Keep a spatial transform but deliberately omit the horizontal CRS
        source = DEM(accessor_dem_path)
        source.set_crs(None)
        path = accessor_dem_path.with_name("no_crs.tif")
        source.to_file(path)
        native = DEM(path)
        ds = open_dem(path)

        # 2/ Inspect metadata and reject vertical operations before loading elevations
        for accessor in [native, ds.dem]:
            assert accessor.vcrs is None
            with pytest.raises(ValueError, match="horizontal CRS"):
                accessor.set_vcrs(5703)
            with pytest.raises(ValueError, match="horizontal CRS"):
                accessor.to_vcrs(6360, force_source_vcrs=5703)
        assert not native.is_loaded and not ds._in_memory

    @pytest.mark.parametrize("vcrs", [None, "Ellipsoid", 5703, CRS.from_epsg(5703)])
    def test_open_dem__vertical_reference_and_loading(self, accessor_dem_path: Path, vcrs: Any) -> None:
        """Checks that opening a DEM attaches the requested vertical CRS without loading its elevation array."""

        # 1/ Compare both file-opening interfaces before asking either for data
        ds = open_dem(accessor_dem_path, vcrs=vcrs)
        dem = DEM(accessor_dem_path, vcrs=vcrs)
        assert not ds._in_memory and not dem.is_loaded
        assert ds.dem.crs == dem.crs
        assert ds.dem.vcrs == dem.vcrs
        assert ds.dem.name == str(accessor_dem_path)

        # 2/ Compare values, nodata and spatial metadata after explicit loading
        assert dem.raster_equal(ds, warn_failure_reason=True)
        assert ds.dem.tags["survey"] == "synthetic"

    def test_open_dem__invalid_band_count(self, tmp_path: Path) -> None:
        """Checks that native DEMs and the DEM accessor reject multi-band rasters before reading values."""

        # 1/ Write a valid two-band raster, which cannot represent one elevation surface
        raster = Raster.from_array(np.ones((2, 5, 7), dtype=np.float32), Affine(20, 0, 500000, 0, -20, 8600000), 32633)
        path = tmp_path / "two_bands.tif"
        raster.to_file(path)

        # 2/ Enforce the same elevation constraint through both opening routes
        with pytest.raises(ValueError, match="one band only"):
            DEM(path)
        with pytest.raises(ValueError, match="one band only"):
            open_dem(path)

    @pytest.mark.parametrize("with_band", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_slope__single_band_dataarray(self, accessor_dem_path: Path, with_band: bool, lazy: bool) -> None:
        """Checks that 2D and singleton-band DataArrays retain their shape and values through terrain methods."""

        # 1/ Build loaded or Dask data with and without a band dimension
        ds = open_dem(accessor_dem_path)
        if with_band:
            ds = ds.expand_dims(band=[1])
        if lazy:
            pytest.importorskip("dask.array")
            ds = ds.chunk({"y": 13, "x": 17})

        # 2/ Match the native calculation exactly while preserving input and output dimensions
        result = ds.dem.slope()
        expected = DEM(accessor_dem_path).slope().get_nanarray()
        assert result.dims == ds.dims
        np.testing.assert_array_equal(result.compute().data.squeeze(), expected)
        if lazy:
            assert not ds._in_memory and not result._in_memory

    @pytest.mark.parametrize("lazy", [False, True])
    def test_copy__vertical_metadata_is_independent(self, accessor_dem_path: Path, lazy: bool) -> None:
        """Checks that DEM accessor copies retain data and VCRS without sharing mutable vertical metadata."""

        # 1/ Copy a loaded or Dask DataArray without loading the Dask source
        if lazy:
            pytest.importorskip("dask.array")
        ds = open_dem(accessor_dem_path, vcrs=5703, **({"chunks": 13} if lazy else {}))
        copied = ds.dem.copy()
        assert copied is not ds
        assert copied.dem.crs == ds.dem.crs
        if lazy:
            assert not ds._in_memory and not copied._in_memory

        # 2/ Change only the copy's VCRS and preserve the original source metadata
        copied.dem.set_vcrs("Ellipsoid")
        assert copied.dem.vcrs == "Ellipsoid"
        assert ds.dem.vcrs == CRS.from_epsg(5703)

    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_file__roundtrip(self, accessor_dem_path: Path, tmp_path: Path, lazy: bool) -> None:
        """Checks that DEM accessor file output preserves VCRS, nodata, masks and elevations."""

        # 1/ Write the same loaded or Dask DEM through the accessor
        if lazy:
            pytest.importorskip("dask.array")
        ds = open_dem(accessor_dem_path, vcrs=5703, **({"chunks": 13} if lazy else {}))
        path = tmp_path / "roundtrip.tif"
        ds.dem.to_file(path)

        # 2/ Read the file back, compare its data and metadata, and check that the Dask source remains unloaded
        reread = open_dem(path)
        assert reread.dem.crs == ds.dem.crs
        assert reread.dem.nodata == ds.dem.nodata
        np.testing.assert_array_equal(reread.data, ds.compute().data)
        if lazy:
            assert not ds._in_memory

    @pytest.mark.parametrize("as_array", [False, True])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_pointcloud__equality_and_laziness(self, accessor_dem_path: Path, as_array: bool, lazy: bool) -> None:
        """Checks that DEM point conversion preserves elevations, coordinates and the expected output interface."""

        # 1/ Open equivalent native and accessor sources with an explicit elevation-column name
        if lazy:
            pytest.importorskip("dask_geopandas")
        ds = open_dem(accessor_dem_path, vcrs=5703, **({"chunks": 13} if lazy else {}))
        native = DEM(accessor_dem_path, vcrs=5703)
        options = {"data_column_name": "height", "as_array": as_array, "force_pixel_offset": "center"}

        # 2/ Compare arrays directly; otherwise check that DEM returns EPC and Xarray returns a GeoDataFrame
        expected = native.to_pointcloud(**options)
        result = ds.dem.to_pointcloud(**options)
        if lazy:
            assert not ds._in_memory
        if as_array:
            np.testing.assert_array_equal(expected, result)
        else:
            assert isinstance(expected, EPC)
            assert isinstance(result, gpd.GeoDataFrame)
            assert expected.pointcloud_equal(result)
            assert_frame_equal(expected.ds, result, check_exact=True)
            assert result.epc.vcrs == native.vcrs
            assert result.epc.data_column == "height"

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

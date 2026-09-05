"""Tests for the base shared EPC behavior, checking exact backend equality and point cloud loading behaviour."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from geopandas.testing import assert_geodataframe_equal
from geoutils import PointCloud
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.base import PointCloudBase
from geoutils.pointcloud.pd_accessor import PointCloudAccessor
from pandas.testing import assert_frame_equal
from pyproj import Transformer

from xdem import DEM, EPC, coreg, open_epc
from xdem.dem.xr_accessor import DEMAccessor
from xdem.epc.base import EPCBase
from xdem.epc.pd_accessor import EPCAccessor, _register_dask_epc_accessor
from xdem.vcrs import _VerticalReference


class NeedsTestError(ValueError):
    """Error to remember to add a test when a new EPC elevation method is added."""


class TestEPCInheritance:
    """
    Test the elevation API shared by EPC and its Pandas accessor on top of GeoUtils point clouds.

    The tests first check that concrete interfaces inherit shared implementations without overriding them, then compare
    public metadata and a representative set of inherited PointCloudBase methods. Vertical transformations and
    coregistration require dedicated inputs and are covered by the two following classes.
    """

    elevation_bases = (EPCBase, _VerticalReference)
    elevation_properties = [
        name
        for base in elevation_bases
        for name, member in vars(base).items()
        if not name.startswith("_")
        and isinstance(member, property)
        and not getattr(member, "__isabstractmethod__", False)
    ]
    elevation_methods = [
        name
        for base in elevation_bases
        for name, member in vars(base).items()
        if not name.startswith("_")
        and not isinstance(member, property)
        and callable(member)
        and not getattr(member, "__isabstractmethod__", False)
    ]
    properties_tested = ["vcrs"]
    methods_tested_separately = ["set_vcrs", "to_vcrs", "coregister_3d"]

    def test_shared_method_ownership(self) -> None:
        """
        Checks that EPC and its accessor share all elevation methods and inherit existing point cloud implementations.
        """

        # 1/ Check that EPC and its accessor inherit shared methods while defining their own CRS storage
        assert EPCBase.__abstractmethods__
        for parent in (EPCBase, _VerticalReference):
            for name in vars(parent):
                if not name.startswith("_") and not getattr(vars(parent)[name], "__isabstractmethod__", False):
                    assert getattr(EPC, name) is getattr(EPCAccessor, name)

        # 2/ Check that copying, construction and LAS writing use the existing GeoUtils methods
        assert EPC.copy is EPCAccessor.copy is PointCloudBase.copy
        assert EPCAccessor.__init__ is PointCloudAccessor.__init__
        assert EPCAccessor.to_las is PointCloudAccessor.to_las
        assert EPC.to_las is PointCloud.to_las

    @pytest.mark.parametrize(
        "property_name",
        ["vcrs", "crs", "bounds", "point_count", "columns", "data_column"],
    )
    def test_properties__equality(self, epc_frame: gpd.GeoDataFrame, property_name: str) -> None:
        """Checks that native and Pandas EPC metadata describe exactly the same point cloud."""

        native = EPC(epc_frame, data_column="height")
        accessor = epc_frame.epc
        expected = getattr(native, property_name)
        actual = getattr(accessor, property_name)

        # Column names are indexed metadata; all other properties are scalar spatial metadata
        if property_name == "columns":
            assert expected.equals(actual)
        else:
            assert expected == actual

    @pytest.mark.parametrize(
        "method, kwargs",
        [
            ("copy", {}),
            ("to_array", {}),
            ("to_xyz", {}),
            ("get_stats", {"stats_name": ["mean", "median", "valid_count"]}),
            ("crop", {"bbox": [500010, 8599920, 500110, 8600000]}),
            ("reproject", {"crs": 32632}),
        ],
    )
    def test_methods__equality(self, epc_frame: gpd.GeoDataFrame, method: str, kwargs: dict[str, Any]) -> None:
        """Checks that inherited point cloud methods produce the same native and accessor results."""

        # 1/ Call inherited methods that return values or another point cloud
        native = EPC(epc_frame, data_column="height")
        expected = getattr(native, method)(**kwargs)
        actual = getattr(epc_frame.epc, method)(**kwargs)

        # 2/ Check that point cloud results are EPC objects for native calls and GeoDataFrames for accessor calls
        if method in ("copy", "crop", "reproject"):
            assert isinstance(expected, EPC) and isinstance(actual, gpd.GeoDataFrame)
            assert_geodataframe_equal(expected.ds, actual)
            assert_frame_equal(expected.ds, actual, check_exact=True)
            assert actual.epc.data_column == expected.data_column
        elif isinstance(expected, dict):
            assert expected == actual
        else:
            np.testing.assert_array_equal(expected, actual)

    def test_methods__test_coverage(self) -> None:
        """Checks that every public EPC elevation property and method remains assigned to an explicit test."""

        # Compare the automatically discovered API with the focused tests in this module
        missing_properties = [name for name in self.elevation_properties if name not in self.properties_tested]
        missing_methods = [name for name in self.elevation_methods if name not in self.methods_tested_separately]
        if missing_properties or missing_methods:
            raise NeedsTestError(
                f"EPC elevation API not covered by tests: properties={missing_properties}, methods={missing_methods}"
            )


class TestEPCVerticalTransform:
    """
    Test EPC vertical reference metadata and elevation transformations through both concrete interfaces.

    The tests cover missing CRS errors, elevation columns and 3D geometry, eager, Dask and multiprocessing execution,
    exact PyProj results, metadata changes before loading and no-op or incompatible backend cases. Generic raster VCRS
    behavior is covered in test_vcrs.py.
    """

    def test_vcrs__missing_horizontal_reference(self, epc_frame: gpd.GeoDataFrame) -> None:
        """Checks that points without a horizontal CRS retain their values and reject vertical operations clearly."""

        # 1/ Coordinates alone cannot locate a geoid transformation without a horizontal reference
        frame = epc_frame.set_crs(None, allow_override=True)
        native = EPC(frame, data_column="height")
        for accessor in [native, frame.epc]:
            assert accessor.vcrs is None
            with pytest.raises(ValueError, match="horizontal CRS"):
                accessor.set_vcrs(5703)
            with pytest.raises(ValueError, match="horizontal CRS"):
                accessor.to_vcrs(6360, force_source_vcrs=5703)

        # 2/ Failed metadata changes must leave the original elevations and auxiliary data available
        np.testing.assert_array_equal(native.data, epc_frame.height)
        np.testing.assert_array_equal(frame.intensity, epc_frame.intensity)

    @pytest.mark.parametrize("use_z", [False, True])
    def test_to_vcrs__eager_geometry_and_columns(self, epc_frame: gpd.GeoDataFrame, use_z: bool) -> None:
        """
        Checks that vertical conversion preserves auxiliary columns and transforms either elevation columns or 3D
        geometry.
        """

        # 1/ Store elevations either in a height column or directly in the 3D point geometry
        frame = epc_frame.copy()
        if use_z:
            frame.geometry = gpd.points_from_xy(frame.geometry.x, frame.geometry.y, z=frame.height, crs=frame.crs)
            frame = frame.drop(columns="height")
            frame.attrs["data_column"] = None
        column = None if use_z else "height"
        native = EPC(frame, data_column=column)
        original = frame.copy()

        # 2/ Convert NAVD88 metres to feet and check that only elevations change
        expected = native.to_vcrs(6360)
        actual = frame.epc.to_vcrs(6360)
        parallel = native.to_vcrs(6360, mp_config=MultiprocConfig(chunks=3))
        assert isinstance(expected, EPC) and isinstance(actual, gpd.GeoDataFrame)
        assert isinstance(parallel, EPC)
        assert_frame_equal(expected.ds, parallel.ds, check_exact=True)
        assert_geodataframe_equal(expected.ds, actual)
        # Use Pandas as well because it checks column types and 3D coordinates exactly
        assert_frame_equal(expected.ds, actual, check_exact=True)
        np.testing.assert_allclose(expected.data, native.data * (3937 / 1200), rtol=1e-15)
        np.testing.assert_array_equal(actual.intensity, original.intensity)
        assert_geodataframe_equal(frame, original)
        assert_frame_equal(frame, original, check_exact=True)

    @pytest.mark.parametrize("partition_size", [7, 11])
    @pytest.mark.parametrize(
        "source_vcrs, destination_vcrs", [(5703, 6360), ("Ellipsoid", "EGM96"), ("EGM96", "Ellipsoid")]
    )
    def test_to_vcrs__dask_and_multiprocessing(
        self, epc_las_path: Path, partition_size: int, source_vcrs: str | int, destination_vcrs: str | int
    ) -> None:
        """
        Checks that LAS/LAZ unit and geoid transforms match PyProj exactly in every backend without loading sources.
        """

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # 1/ Open test files
        # Open the same file as an EPC and a Dask table, using different chunk sizes for each calculation
        source = EPC(epc_las_path)
        source.set_vcrs(source_vcrs)
        lazy = open_epc(epc_las_path, columns="all", chunks=partition_size)
        lazy.epc.set_vcrs(source_vcrs)
        original_graph = lazy.expr
        original_crs = lazy.epc.crs
        tasks: list[Any] = []
        config = MultiprocConfig(chunks=3)

        # 2/ Create the Dask and multiprocessing results without loading their source objects
        # The callback records every executed Dask task, so the list must remain empty until compute is called
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            output = lazy.epc.to_vcrs(destination_vcrs)
        assert tasks == []
        assert isinstance(output, dgpd.GeoDataFrame)
        parallel = source.to_vcrs(destination_vcrs, mp_config=config)
        assert not source.is_loaded and not lazy.epc.is_loaded

        # 3/ Compare Dask with a loaded table, and multiprocessing with a loaded EPC
        # The table comparison includes every LAS column, while EPC loads its main elevation column
        eager_frame = open_epc(epc_las_path, columns="all")
        eager_frame.epc.set_vcrs(source_vcrs)
        expected = eager_frame.epc.to_vcrs(destination_vcrs)
        computed = output.compute()
        assert_geodataframe_equal(computed, expected)
        assert_frame_equal(computed, expected, check_exact=True)
        native = EPC(epc_las_path)
        native.set_vcrs(source_vcrs)
        expected_native = native.to_vcrs(destination_vcrs)
        assert isinstance(parallel, EPC) and isinstance(expected_native, EPC)
        assert isinstance(expected, gpd.GeoDataFrame) and isinstance(computed, gpd.GeoDataFrame)
        assert parallel.pointcloud_equal(expected_native)
        assert_frame_equal(parallel.ds, expected_native.ds, check_exact=True)

        # 4/ Compare with the same CRS change calculated directly by PyProj
        transformer = Transformer.from_crs(eager_frame.crs, expected.crs, always_xy=True)
        elevations = transformer.transform(eager_frame.geometry.x, eager_frame.geometry.y, eager_frame.epc.data)[2]
        np.testing.assert_array_equal(computed.epc.data, elevations)

        # Check that computing the result did not change or load the Dask source
        assert lazy.expr is original_graph and not lazy.epc.is_loaded
        assert lazy.epc.crs == lazy.pc.crs == original_crs
        assert not source.is_loaded

    @pytest.mark.parametrize("backend", ["native", "eager", "dask"])
    def test_set_vcrs__before_loading(self, epc_las_path: Path, backend: str) -> None:
        """
        Checks that assigning vertical metadata survives later loading and is visible through both dataframe accessors.
        """

        # 1/ Open the same source as an EPC, a Pandas table or a Dask table
        if backend == "dask":
            pytest.importorskip("dask_geopandas")
        source = (
            EPC(epc_las_path)
            if backend == "native"
            else open_epc(epc_las_path, chunks=7 if backend == "dask" else None)
        )
        accessor = source if backend == "native" else source.epc
        initially_loaded = accessor.is_loaded

        # 2/ Change only the vertical CRS and check that the loading state stays the same
        accessor.set_vcrs("Ellipsoid")
        assert accessor.vcrs == "Ellipsoid"
        assert accessor.is_loaded == initially_loaded
        # 3/ Load the values when needed and check that both point accessors report the assigned vertical CRS
        if backend == "native":
            source.load()
            assert source.vcrs == "Ellipsoid"
        else:
            assert source.pc.crs == source.epc.crs
            computed = source.epc.load() if backend == "dask" else source
            assert computed.epc.vcrs == "Ellipsoid"
            assert source.epc.is_loaded == initially_loaded

    def test_to_vcrs__noop_and_errors(self, epc_las_path: Path) -> None:
        """
        Checks that identical vertical references preserve unloaded files and mixed backends fail before reading data.
        """

        # 1/ Transforming to the existing VCRS returns an independent object without reading its points
        source = EPC(epc_las_path)
        with pytest.warns(UserWarning, match="same, skipping"):
            copied = source.to_vcrs(5703)
        assert isinstance(copied, EPC) and copied is not source
        assert not copied.is_loaded and not source.is_loaded

        # 2/ Reject a call that requests both Dask and multiprocessing
        pytest.importorskip("dask_geopandas")
        lazy = open_epc(epc_las_path, chunks=7)
        with pytest.raises(ValueError, match="simultaneously"):
            lazy.epc.to_vcrs(6360, mp_config=MultiprocConfig(chunks=3))
        assert not lazy.epc.is_loaded


class TestEPCCoregistration:
    """
    Test eager EPC coregistration through the native class and Pandas accessor.

    The tests compare exact vertical-shift results for elevation columns and 3D geometry against raster references in
    both native and accessor form. They also define the current Dask boundary and verify that rejecting it leaves the
    source graph untouched.
    """

    @pytest.mark.parametrize("use_z", [False, True])
    @pytest.mark.parametrize("reference_accessor", [False, True])
    def test_coregister_3d__vertical_shift(
        self, epc_frame: gpd.GeoDataFrame, reference_accessor: bool, use_z: bool
    ) -> None:
        """
        Checks that EPC coregistration uses the selected elevation column and matches the native point cloud result.
        """

        # 1/ Shift every point by exactly 8 metres so the correct result is known in advance
        values = epc_frame.height.to_numpy().reshape((5, 7))
        reference = DEM.from_array(values, Affine(20, 0, 499990, 0, -20, 8600010), epc_frame.crs, nodata=-9999)
        points = reference.to_pointcloud(data_column_name="height", force_pixel_offset="ul")
        shifted = points.copy(new_array=points.data + 8)
        frame = shifted.ds.copy()
        frame["intensity"] = epc_frame.intensity.to_numpy()
        frame.index = np.arange(len(frame)) * 3
        frame.attrs["data_column"] = "height"

        # Store elevations either in a height column or in the 3D point geometry
        if use_z:
            frame.geometry = gpd.points_from_xy(frame.geometry.x, frame.geometry.y, z=frame.height, crs=frame.crs)
            frame = frame.drop(columns="height")
            frame.attrs["data_column"] = None
        shifted = EPC(frame, data_column=None if use_z else "height")
        reference_ds = DEMAccessor.from_array(reference.data, reference.transform, reference.crs, nodata=-9999)
        ref = reference_ds.dem if reference_accessor else reference

        # 2/ Run EPC and Pandas calls with the same reference and random seed
        expected = shifted.coregister_3d(ref, coreg.VerticalShift(), random_state=42)
        actual = frame.epc.coregister_3d(ref, coreg.VerticalShift(), random_state=42)
        assert isinstance(expected, EPC) and isinstance(actual, gpd.GeoDataFrame)
        assert_geodataframe_equal(expected.ds, actual)
        assert_frame_equal(expected.ds, actual, check_exact=True)
        np.testing.assert_array_equal(actual.epc.data, points.data)
        np.testing.assert_array_equal(actual.intensity, frame.intensity)
        np.testing.assert_array_equal(actual.index, frame.index)
        np.testing.assert_array_equal(frame.epc.data, shifted.data)

    def test_coregister_3d__dask_rejected(self, epc_frame: gpd.GeoDataFrame) -> None:
        """Checks that Dask EPC coregistration fails explicitly without executing the point graph."""

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # 1/ Create a Dask point cloud and save its work plan for the final check
        _register_dask_epc_accessor()
        lazy = dgpd.from_geopandas(epc_frame, npartitions=3)
        graph = lazy.expr
        tasks: list[Any] = []

        # 2/ Reject coregistration before Dask runs and leave the source unchanged
        # The callback records every executed Dask task, so the list must remain empty
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            with pytest.raises(NotImplementedError, match="Dask coregistration"):
                lazy.epc.coregister_3d(EPC(epc_frame, data_column="height"), coreg.VerticalShift())
        assert tasks == []
        assert lazy.expr is graph and not lazy.epc.is_loaded

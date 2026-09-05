"""Tests specific to Pandas 'epc' accessor, optional file formats and inherited lazy point cloud operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from geoutils import PointCloud
from geoutils.multiproc import ClusterGenerator, MultiprocConfig
from pandas.testing import assert_frame_equal
from shapely.geometry import Polygon

from xdem import EPC, open_epc
from xdem.epc.pd_accessor import EPCAccessor, _register_dask_epc_accessor


class TestEPCAccessor:
    """
    Test construction and eager behavior specific to the EPC Pandas accessor.

    The tests cover accessor validation, shared point constructors, independent copies and native EPC conversion, then
    verify vector-file opening with elevation-column, auxiliary-column and vertical-reference metadata intact. Shared
    elevation methods are compared with EPC in test_base.py.
    """

    def test_init__validation(self, epc_frame: gpd.GeoDataFrame) -> None:
        """Checks that the EPC accessor accepts point GeoDataFrames and rejects missing or non-point geometry."""

        # 1/ Accept a point GeoDataFrame and read its selected elevation column
        assert isinstance(epc_frame.epc, EPCAccessor)
        assert epc_frame.epc.data_column == "height"
        np.testing.assert_array_equal(epc_frame.epc.data, epc_frame.height)

        # 2/ Reject a table without geometry and a GeoDataFrame containing polygons
        with pytest.raises(AttributeError, match="point-cloud"):
            pd.DataFrame({"height": [1]}).epc
        polygons = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (0, 1)])], crs=32633)
        with pytest.raises(AttributeError, match="point geometries"):
            polygons.epc

    @pytest.mark.parametrize("constructor", ["from_xyz", "from_array", "from_tuples"])
    @pytest.mark.parametrize("use_z", [False, True])
    def test_methods__constructors(self, epc_frame: gpd.GeoDataFrame, constructor: str, use_z: bool) -> None:
        """Checks that inherited EPC constructors produce equivalent elevation objects and GeoDataFrames."""

        # 1/ Build the input expected by each constructor from the same X, Y and Z values
        xx, yy, zz = epc_frame.geometry.x.to_numpy(), epc_frame.geometry.y.to_numpy(), epc_frame.height.to_numpy()
        data: Any
        if constructor == "from_xyz":
            arguments = {"x": xx, "y": yy, "z": zz}
        elif constructor == "from_array":
            data = np.column_stack((xx, yy, zz))
            arguments = {"data": data}
        else:
            data = list(zip(xx, yy, zz))
            arguments = {"tuples_xyz": data}
        options = {"crs": epc_frame.crs, "data_column": "height", "use_z": use_z}

        # 2/ Check that EPC and its accessor build the same points and vertical CRS
        expected = getattr(EPC, constructor)(**arguments, **options)
        actual = getattr(EPCAccessor, constructor)(**arguments, **options)
        assert isinstance(expected, EPC)
        assert_geodataframe_equal(expected.ds, actual)
        assert_frame_equal(expected.ds, actual, check_exact=True)
        assert actual.epc.vcrs == expected.vcrs

    def test_copy__data_column_and_conversion(self, epc_frame: gpd.GeoDataFrame) -> None:
        """
        Checks that copying, selecting an elevation column and converting to EPC preserve data and independent state.
        """

        # 1/ Copy the frame, change its heights and then select intensity as the elevation column
        copied = epc_frame.epc.copy(new_array=epc_frame.height.to_numpy() + 2)
        np.testing.assert_array_equal(copied.height, epc_frame.height + 2)
        np.testing.assert_array_equal(copied.intensity, epc_frame.intensity)
        copied.epc.set_data_column("intensity")

        # 2/ Convert the copy to EPC and check that the original still selects height
        converted = copied.epc.to_xdem()
        assert isinstance(converted, EPC)
        assert converted.data_column == "intensity"
        assert epc_frame.epc.data_column == "height"
        np.testing.assert_array_equal(converted.data, epc_frame.intensity)
        with pytest.raises(ValueError, match="not found"):
            copied.epc.set_data_column("missing")

    def test_open_epc__vector_file(self, epc_frame: gpd.GeoDataFrame, tmp_path: Path) -> None:
        """Checks that EPC opening uses the vector reader and preserves the selected elevation column and CRS."""

        # 1/ Write an EPC to GeoPackage and reopen it with open_epc
        path = tmp_path / "elevation.gpkg"
        EPC(epc_frame, data_column="height").to_file(path)
        result = open_epc(path, data_column="height")

        # 2/ Compare all values and check that height remains the selected elevation column
        # GeoPackage stores unsigned LAS intensity as a wider integer; both readers must recover identical values
        expected = EPC(path, data_column="height")
        assert_geodataframe_equal(result, expected.ds)
        assert_frame_equal(result, expected.ds, check_exact=True)
        np.testing.assert_array_equal(result.intensity, epc_frame.intensity)
        assert result.epc.data_column == "height"
        assert result.epc.vcrs == epc_frame.epc.vcrs


class TestEPCLas:
    """
    Test optional LAS and LAZ support through native, Pandas, Dask and multiprocessing interfaces.

    The tests cover empty files, partitioned reading and writing, exact packed dimensions, CRS and elevation metadata,
    source loading state and missing optional dependencies. LASPy supplies an independent file-level comparison where
    possible, while GeoUtils owns the underlying format implementation.
    """

    @pytest.mark.parametrize("lazy", [False, True])
    def test_open_epc__empty_las(self, epc_las_path: Path, tmp_path: Path, lazy: bool) -> None:
        """Checks that empty LAS/LAZ files retain CRS and column metadata through opening and vertical conversion."""

        if lazy:
            pytest.importorskip("dask_geopandas")

        # 1/ Write an empty LAS or LAZ file with the same columns and CRS as the fixture
        frame = EPC(epc_las_path).ds.iloc[:0].copy()
        path = tmp_path / ("empty" + epc_las_path.suffix)
        EPC(frame, data_column="Z").to_las(
            path, version="1.4", point_format=6, scales=(0.001, 0.001, 0.001), offsets=(500000, 8600000, 0)
        )
        native = EPC(path)
        assert native.point_count == 0 and not native.is_loaded

        # 2/ Open the empty file as a Pandas or Dask table and change its vertical CRS
        source = open_epc(path, columns="all", chunks=7 if lazy else None)
        graph = source.expr if lazy else None
        assert source.epc.crs == native.crs
        result = source.epc.to_vcrs(6360)
        computed = result.compute() if lazy else result

        # 3/ Empty results still carry a usable elevation column and destination reference
        assert len(computed) == 0
        assert computed.epc.data_column == "Z"
        assert computed.epc.vcrs.to_epsg() == 6360
        assert computed.columns.equals(source.columns)
        assert source.epc.crs == native.crs
        if lazy:
            assert source.expr is graph and not source.epc.is_loaded

    def test_methods__process_workers(self, epc_las_path: Path, tmp_path: Path) -> None:
        """Checks that separate workers can load, transform and write LAS/LAZ elevation partitions exactly."""

        # 1/ Use two processes to transform and write seven points at a time
        source = EPC(epc_las_path)
        path = tmp_path / ("workers" + epc_las_path.suffix)
        with ClusterGenerator("multiprocessing", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=7, cluster=cluster)
            transformed = source.to_vcrs(6360, mp_config=config)
            transformed.to_las(
                path,
                version="1.4",
                point_format=6,
                scales=(0.001, 0.001, 0.001),
                offsets=(500000, 8600000, 0),
                mp_config=config,
            )
        assert not source.is_loaded

        # 2/ Compare values before writing, then compare files written with the same LAS precision
        expected = EPC(epc_las_path).to_vcrs(6360)
        assert transformed.pointcloud_equal(expected)
        assert_frame_equal(transformed.ds, expected.ds, check_exact=True)
        eager_path = tmp_path / ("eager" + epc_las_path.suffix)
        expected.to_las(
            eager_path, version="1.4", point_format=6, scales=(0.001, 0.001, 0.001), offsets=(500000, 8600000, 0)
        )
        written = EPC(path)
        expected_written = EPC(eager_path)
        assert written.pointcloud_equal(expected_written)
        assert_frame_equal(written.ds, expected_written.ds, check_exact=True)
        assert not source.is_loaded

    @pytest.mark.parametrize("columns", ["main", "all", ["Z", "intensity"]])
    def test_open_epc__las_backends(self, epc_las_path: Path, columns: Any) -> None:
        """Checks that optional LAS/LAZ loading gives exact coordinates, values and columns in every backend."""

        # 1/ Compare an EPC loaded from its header with a Pandas table loaded in one call
        native = EPC(epc_las_path)
        assert not native.is_loaded
        eager = open_epc(epc_las_path, columns=columns)
        native.load(columns=columns)
        assert_geodataframe_equal(native.ds, eager)
        assert_frame_equal(native.ds, eager, check_exact=True)

        # 2/ Load three points per process and compare their order and values with the Pandas result
        parallel = EPC(epc_las_path)
        parallel.load(columns=columns, mp_config=MultiprocConfig(chunks=3))
        assert_geodataframe_equal(parallel.ds, eager)
        assert_frame_equal(parallel.ds, eager, check_exact=True)

        # 3/ Compute the Dask table, compare it and check that the source remains lazy
        pytest.importorskip("dask_geopandas")
        lazy = open_epc(epc_las_path, columns=columns, chunks=7)
        graph = lazy.expr
        assert not lazy.epc.is_loaded
        computed = lazy.compute()
        assert_geodataframe_equal(computed, eager)
        assert_frame_equal(computed, eager, check_exact=True)
        assert lazy.expr is graph and not lazy.epc.is_loaded

    @pytest.mark.parametrize("backend", ["eager", "multiprocessing", "dask"])
    def test_to_las__roundtrip(self, epc_las_path: Path, tmp_path: Path, backend: str) -> None:
        """Checks that EPC accessor LAS/LAZ output preserves the native dimensions and vertical CRS across writers."""

        # 1/ Open every LAS column and use the source precision when writing the new file
        laspy = pytest.importorskip("laspy")
        if backend == "dask":
            pytest.importorskip("dask_geopandas")
        source = open_epc(epc_las_path, columns="all", chunks=7 if backend == "dask" else None)
        path = tmp_path / ("written" + epc_las_path.suffix)
        options: dict[str, Any] = {
            "version": "1.4",
            "point_format": 6,
            "scales": (0.001, 0.001, 0.001),
            "offsets": (500000, 8600000, 0),
        }
        if backend == "multiprocessing":
            options.update({"chunks": 7, "mp_config": MultiprocConfig(chunks=3)})

        # 2/ Write the file and read it with LasPy and EPC for independent comparisons
        source.epc.to_las(path, **options)
        expected = laspy.read(epc_las_path)
        actual = laspy.read(path)
        np.testing.assert_array_equal(actual.xyz, expected.xyz)
        np.testing.assert_array_equal(actual.intensity, expected.intensity)
        # LasPy's packed records check every stored LAS field, including fields not exposed as XYZ
        np.testing.assert_array_equal(actual.points.array, expected.points.array)
        assert actual.header.parse_crs() == expected.header.parse_crs()
        assert EPC(path).pointcloud_equal(EPC(epc_las_path))
        # 3/ Check that writing a Dask point cloud leaves its source table lazy
        if backend == "dask":
            assert not source.epc.is_loaded

    def test_open_epc__missing_optional_dependency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that EPC LAS readers report a missing optional LasPy dependency through both public interfaces."""

        import geoutils.pointcloud.las as las_module

        # Make the LasPy import fail to check the error shown by both readers
        def missing_laspy(name: str, **kwargs: Any) -> Any:
            """Report an unavailable LasPy backend for the public reader checks."""
            raise ImportError("Optional dependency 'laspy' required for LAS support.")

        monkeypatch.setattr(las_module, "import_optional", missing_laspy)
        readers: list[Callable[..., Any]] = [EPC, open_epc]
        for reader in readers:
            with pytest.raises(ImportError, match="Optional dependency 'laspy'"):
                reader("missing.las")


class TestEPCLazyMethods:
    """
    Test inherited point cloud methods through eager Pandas and Dask EPC accessors.

    The tests compare conversion types and loading behavior, native dataframe export, lazy copying and explicit loading,
    then exact statistics and gridding across point partitions. Every Dask case retains the original source graph after
    the output is evaluated.
    """

    @pytest.mark.parametrize("method", ["to_xdem", "to_geoutils"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_methods__conversion_loading_laziness(self, epc_frame: gpd.GeoDataFrame, method: str, lazy: bool) -> None:
        """Checks that native point conversions preserve exact data and the original eager or Dask loading state."""

        # 1/ Select height explicitly in both Pandas and Dask because the frame has two numeric columns
        source = epc_frame.copy()
        if lazy:
            dgpd = pytest.importorskip("dask_geopandas")
            _register_dask_epc_accessor()
            source = dgpd.from_geopandas(source, npartitions=3)
            source.epc.set_data_column("height")
        graph = source.expr if lazy else None
        assert source.epc.is_loaded is not lazy

        # 2/ Convert to EPC or PointCloud and check that a Dask source remains unchanged and lazy
        result = getattr(source.epc, method)()
        assert type(result) is (EPC if method == "to_xdem" else PointCloud)
        assert result.is_loaded
        assert result.data_column == "height"
        assert source.epc.is_loaded is not lazy
        if lazy:
            assert source.expr is graph

        # 3/ Preserve elevation, auxiliary columns, index, geometry and compound CRS exactly
        assert_geodataframe_equal(result.ds, epc_frame)
        assert_frame_equal(result.ds, epc_frame, check_exact=True)
        assert source.epc.is_loaded is not lazy

    @pytest.mark.parametrize("loaded", [False, True])
    def test_ds__loading(self, epc_las_path: Path, loaded: bool) -> None:
        """Checks that dataframe access loads LAS/LAZ sources and preserves already loaded auxiliary dimensions."""

        # 1/ Access the EPC table and check which LAS columns are loaded from disk
        source = EPC(epc_las_path)
        if loaded:
            source.load(columns="all")
        assert source.is_loaded is loaded
        result = source.ds
        assert isinstance(result, gpd.GeoDataFrame)
        assert source.is_loaded
        assert result.epc.crs == source.crs
        assert result.epc.data_column == source.data_column

        # 2/ An independent GeoUtils reader supplies the same requested LAS dimensions for exact comparison
        expected = PointCloud(epc_las_path)
        expected.load(columns="all" if loaded else "main")
        assert_geodataframe_equal(result, expected.ds)
        assert_frame_equal(result, expected.ds, check_exact=True)

    @pytest.mark.parametrize("partition_count", [3, 5])
    def test_copy__load_and_conversion(self, epc_frame: gpd.GeoDataFrame, partition_count: int) -> None:
        """
        Checks that inherited copying stays lazy while explicit loading and EPC conversion leave their source untouched.
        """

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # 1/ Create a Dask table with the requested number of parts and save its work plan
        _register_dask_epc_accessor()
        source = dgpd.from_geopandas(epc_frame, npartitions=partition_count)
        # Dask-GeoPandas construction discards arbitrary attrs, so explicitly choose the elevation column
        source.epc.set_data_column("height")
        graph = source.expr
        tasks: list[Any] = []

        # 2/ Copy the table and add two to height without running any Dask tasks
        # The callback records every executed Dask task, so the list must remain empty
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            copied = source.epc.copy(new_array=source.height + 2)
        assert tasks == []
        assert isinstance(copied, dgpd.GeoDataFrame) and not copied.epc.is_loaded
        # 3/ Compute the copy and conversion, compare their values and check that the source remains lazy
        expected = epc_frame.copy()
        expected["height"] += 2
        loaded = copied.epc.load()
        converted = source.epc.to_xdem().ds
        assert_geodataframe_equal(loaded, expected)
        assert_frame_equal(loaded, expected, check_exact=True)
        assert_geodataframe_equal(converted, epc_frame)
        assert_frame_equal(converted, epc_frame, check_exact=True)
        assert source.expr is graph and not source.epc.is_loaded

    def test_methods__stats_and_grid(self, epc_las_path: Path) -> None:
        """
        Checks that inherited compact statistics and lazy gridding match eager EPC results without replacing the source.
        """

        # 1/ Open the same LAS file as a Dask table and an EPC, then save the Dask work plan
        pytest.importorskip("dask_geopandas")
        da = pytest.importorskip("dask.array")
        source = open_epc(epc_las_path, chunks=7)
        graph = source.expr
        expected = EPC(epc_las_path)

        # 2/ Compare statistics whose results must not depend on how the points are split
        stats = ["mean", "median", "valid_count"]
        assert source.epc.get_stats(stats) == expected.get_stats(stats)
        assert source.expr is graph and not source.epc.is_loaded

        # 3/ Grid with nearest neighbours, using grid chunks that differ from the point chunks
        options = {"shape": (5, 7), "bounds": source.epc.bounds, "resampling": "nearest", "dist_nodata_pixel": 100}
        eager_grid = expected.grid(**options)
        lazy_grid = source.epc.grid(**options, chunksizes=(3, 2))
        assert isinstance(lazy_grid.data, da.Array)
        assert eager_grid.raster_equal(lazy_grid.compute(), strict_masked=False, warn_failure_reason=True)
        assert source.expr is graph and not source.epc.is_loaded

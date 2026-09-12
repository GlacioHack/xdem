"""Tests for vertical CRS transformation tools."""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Any, Literal

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from geoutils.multiproc import MultiprocConfig
from pandas.testing import assert_frame_equal
from pyproj import CRS, Transformer

import xdem
import xdem.vcrs
from xdem import examples


class TestVCRS:
    def test_parse_vcrs_name_from_product(self) -> None:
        """Test parsing of vertical CRS name from DEM product name."""

        # Check that the value for the key is returned by the function
        for product in xdem.vcrs.vcrs_dem_products.keys():
            assert xdem.vcrs._parse_vcrs_name_from_product(product) == xdem.vcrs.vcrs_dem_products[product]

        # And that, otherwise, it's a None
        assert xdem.vcrs._parse_vcrs_name_from_product("BESTDEM") is None

    # Expect outputs for the inputs
    @pytest.mark.parametrize(
        "input_output",
        [
            (CRS("EPSG:4326"), None),
            (CRS("EPSG:4979"), "Ellipsoid"),
            (CRS("EPSG:4326+5773"), CRS("EPSG:5773")),
            (CRS("EPSG:32610"), None),
            (CRS("EPSG:32610").to_3d(), "Ellipsoid"),
        ],
    )
    def test_vcrs_from_crs(self, input_output: tuple[CRS, CRS]) -> None:
        """Test the extraction of a vertical CRS from a CRS."""

        input = input_output[0]
        output = input_output[1]

        # Extract vertical CRS from CRS
        vcrs = xdem.vcrs._vcrs_from_crs(crs=input)

        # Check that the result is as expected
        if isinstance(output, CRS):
            assert isinstance(vcrs, CRS)
            assert vcrs.equals(output)
        elif isinstance(output, str):
            assert vcrs == "Ellipsoid"
        else:
            assert vcrs is None

    @pytest.mark.parametrize(
        "crs, expected",
        [
            # Compound CRS with vertical meters
            (CRS("EPSG:4326+5773"), "m"),  # WGS84 + EGM96 height
            # Vertical CRS alone
            (CRS("EPSG:5773"), "m"),  # EGM96 height
            # Compound CRS projected + vertical
            (CRS("EPSG:32633+5773"), "m"),  # UTM 33N + EGM96 height
            # Vertical CRS in feet (NAVD88)
            (CRS("EPSG:6360"), "ftUS"),
            # Pure 2D CRS (no vertical axis)
            (CRS("EPSG:4326"), None),
            (CRS("EPSG:32633"), None),
        ],
    )
    def test_vertical_unit_symbol(self, crs: CRS, expected: str | None) -> None:
        """Test extraction of vertical unit symbols from CRS."""

        assert xdem.vcrs.vertical_unit_symbol(crs) == expected

    @pytest.mark.parametrize(
        "vcrs_input",
        [
            "EGM08",
            "EGM96",
            "us_noaa_geoid06_ak.tif",
            pathlib.Path("is_lmi_Icegeoid_ISN93.tif"),
            3855,
            CRS.from_epsg(5773),
        ],
    )
    def test_vcrs_from_user_input(self, vcrs_input: str | pathlib.Path | int | CRS) -> None:
        """Tests the function _vcrs_from_user_input for varying user inputs, for which it will return a CRS."""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid .*")

        # Get user input
        vcrs = xdem.vcrs._vcrs_from_user_input(vcrs_input)

        # Check output type
        assert isinstance(vcrs, CRS)
        assert vcrs.is_vertical

    @pytest.mark.parametrize(
        "vcrs_input", ["Ellipsoid", "ellipsoid", "wgs84", 4326, 4979, CRS.from_epsg(4326), CRS.from_epsg(4979)]
    )
    def test_vcrs_from_user_input__ellipsoid(self, vcrs_input: str | int) -> None:
        """Tests the function _vcrs_from_user_input for inputs where it returns "Ellipsoid"."""

        # Get user input
        vcrs = xdem.vcrs._vcrs_from_user_input(vcrs_input)

        # Check output type
        assert vcrs == "Ellipsoid"

    def test_vcrs_from_user_input__errors(self) -> None:
        """Tests errors of vcrs_from_user_input."""

        # Check that an error is raised when the type is wrong
        with pytest.raises(TypeError, match="New vertical CRS must be a string, path or VerticalCRS, received.*"):
            xdem.vcrs._vcrs_from_user_input(np.zeros(1))  # type: ignore

        # Check that an error is raised if the CRS is not vertical
        with pytest.raises(
            ValueError,
            match=re.escape(
                "New vertical CRS must have a vertical axis, 'WGS 84 / UTM "
                "zone 1N' does not (check with `CRS.is_vertical`)."
            ),
        ):
            xdem.vcrs._vcrs_from_user_input(32601)

        # Check that a warning is raised if the CRS has other dimensions than vertical
        with pytest.warns(
            UserWarning,
            match="New vertical CRS has a vertical dimension but also other components, "
            "extracting the vertical reference only.",
        ):
            xdem.vcrs._vcrs_from_user_input(CRS("EPSG:4326+5773"))

        # Check that an error is raised for impossible strings (not in key strings and doesn't end with .tif/.json/pol)
        with pytest.raises(ValueError, match="String vcrs input 'EGM2008' is not recognized.*"):
            xdem.vcrs._vcrs_from_user_input("EGM2008")

    @pytest.mark.parametrize(
        "grid", ["us_noaa_geoid06_ak.tif", "is_lmi_Icegeoid_ISN93.tif", "us_nga_egm08_25.tif", "us_nga_egm96_15.tif"]
    )
    def test_build_vcrs_from_grid(self, grid: str) -> None:
        """Test that vertical CRS are correctly built from grid"""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid .*")

        # Build vertical CRS
        vcrs = xdem.vcrs._build_vcrs_from_grid(grid=grid)
        assert vcrs.is_vertical

        # Check that the explicit construction yields the same CRS as "the old init way" (see function description)
        vcrs_oldway = xdem.vcrs._build_vcrs_from_grid(grid=grid, old_way=True)
        assert vcrs.equals(vcrs_oldway)

    def test_build_vcrs_from_grid__errors(self) -> None:
        """Check errors for non-existing grids."""

        with pytest.warns(UserWarning, match="Grid 'not_a_grid.tif' not found in .*"):
            with pytest.raises(ValueError, match="The provided grid 'not_a_grid.tif' does not exist.*"):
                xdem.vcrs._build_vcrs_from_grid(grid="not_a_grid.tif")

    # Test for WGS84 in 2D and 3D, UTM, CompoundCRS, everything should work
    @pytest.mark.parametrize("crs", [CRS("EPSG:4326"), CRS("EPSG:4979"), CRS("32610"), CRS("EPSG:4326+5773")])
    @pytest.mark.parametrize("vcrs_input", [CRS("EPSG:5773"), "is_lmi_Icegeoid_ISN93.tif", "EGM96", "Ellipsoid"])
    def test_combine_crs_and_vcrs(self, crs: CRS, vcrs_input: CRS | str) -> None:
        """Test combining a horizontal CRS with a vertical CRS."""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid .*")

        # Get the vertical CRS from user input
        vcrs = xdem.vcrs._vcrs_from_user_input(vcrs_input=vcrs_input)

        # Build the complete 3D CRS

        # For a 3D horizontal CRS, a condition based on pyproj version is needed
        if len(crs.axis_info) > 2:
            import pyproj
            from packaging.version import Version

            # If the version is higher than 3.5.0, it should pass
            if Version(pyproj.__version__) > Version("3.5.0"):
                combined_crs = xdem.vcrs._combine_crs_and_vcrs(crs=crs, vcrs=vcrs)
            # Otherwise, it should raise an error
            else:
                with pytest.raises(
                    NotImplementedError,
                    match="pyproj >= 3.5.1 is required to demote a 3D CRS to 2D and be able to compound "
                    "with a new vertical CRS. Update your dependencies or pass the 2D source CRS "
                    "manually.",
                ):
                    xdem.vcrs._combine_crs_and_vcrs(crs=crs, vcrs=vcrs)
                return None
        # If the CRS is 2D, it should pass
        else:
            combined_crs = xdem.vcrs._combine_crs_and_vcrs(crs=crs, vcrs=vcrs)

        assert isinstance(combined_crs, CRS)
        is_3d = len(combined_crs.axis_info) == 3
        assert is_3d

    def test_combine_crs_and_vcrs__errors(self) -> None:
        """Test errors are correctly raised when combining incompatible CRS inputs."""

        with pytest.raises(
            ValueError, match="Invalid vcrs given. Must be a vertical " "CRS or the literal string 'Ellipsoid'."
        ):
            xdem.vcrs._combine_crs_and_vcrs(crs=CRS("EPSG:4326"), vcrs="NotAVerticalCRS")  # type: ignore

    # Compare to manually-extracted shifts at specific coordinates for the geoid grids
    egm96_chile = {"grid": "us_nga_egm96_15.tif", "lon": -68, "lat": -20, "shift": 42}
    egm08_chile = {"grid": "us_nga_egm08_25.tif", "lon": -68, "lat": -20, "shift": 42}
    geoid96_alaska = {"grid": "us_noaa_geoid06_ak.tif", "lon": -145, "lat": 62, "shift": 15}
    isn93_iceland = {"grid": "is_lmi_Icegeoid_ISN93.tif", "lon": -18, "lat": 65, "shift": 68}

    @pytest.mark.parametrize("grid_shifts", [egm08_chile, egm08_chile, geoid96_alaska, isn93_iceland])
    def test_transform_zz(self, grid_shifts: dict[str, Any]) -> None:
        """Tests grids to convert vertical CRS."""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid .*")

        # Using an arbitrary elevation of 100 m (no influence on the transformation)
        zz = 100
        xx = grid_shifts["lon"]
        yy = grid_shifts["lat"]
        crs_from = CRS.from_epsg(4326)
        source_crs = xdem.vcrs._combine_crs_and_vcrs(crs=crs_from, vcrs="Ellipsoid")

        # Build the compound CRS
        vcrs_to = xdem.vcrs._vcrs_from_user_input(vcrs_input=grid_shifts["grid"])
        destination_crs = xdem.vcrs._combine_crs_and_vcrs(crs=crs_from, vcrs=vcrs_to)
        transformer = xdem.vcrs._build_vertical_transformer(crs_from=source_crs, crs_to=destination_crs)
        # Apply the transformation
        zz_trans = xdem.vcrs._transform_zz(transformer=transformer, xx=xx, yy=yy, zz=zz)

        # Compare the elevation difference
        z_diff = 100 - zz_trans

        # Check the shift is the one expect within 10%
        assert z_diff == pytest.approx(grid_shifts["shift"], rel=0.1)


class TestToVCRSChunked:
    """
    Test vertical transformations across eager, Dask and multiprocessing backends.

    This class tests:
    - ``to_vcrs`` for DEM and Xarray inputs with ``Area`` and ``Point`` coordinates,
    - Fractional raster coordinates, optional band dimensions and exact backend equality,
    - ``to_vcrs`` for empty and partitioned point clouds,
    - Output metadata, source laziness and independent PyProj results.
    """

    @pytest.mark.parametrize("area_or_point", ["Area", "Point"])
    @pytest.mark.parametrize(
        "force_source_vcrs, dst_vcrs",
        [
            ("EGM96", "Ellipsoid"),
            ("Ellipsoid", "EGM96"),
        ],
        ids=["egm96_to_ellipsoid", "ellipsoid_to_egm96"],
    )
    def test_to_vcrs_chunked_backends_equal(
        self,
        force_source_vcrs: str,
        dst_vcrs: str,
        area_or_point: str,
    ) -> None:
        """
        Checks that to_vcrs gives the same result in memory, with Dask and with multiprocessing.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # 1/ Open test files
        # Load the example as both a DEM and an Xarray so their results can be compared directly
        path_dem = examples.get_path_test("longyearbyen_ref_dem")
        dem_base = xdem.DEM(path_dem)
        dem_base.load()
        xr_base = gu.open_raster(path_dem)
        xr_base.load()

        # Split the DEM at different rows and columns in multiprocessing and Dask
        # This checks that each method uses the original DEM coordinates rather than positions within a chunk
        dem_mp = xdem.DEM(path_dem)
        mp_config = MultiprocConfig(chunks=10)
        ds = gu.open_raster(path_dem, chunks={"x": 13, "y": 17})

        # Save the Dask data object so it can be checked for replacement after the result is calculated
        source_array = ds.data
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None

        # Area values describe pixels, while Point values describe samples at coordinates
        # Apply the selected meaning to all four inputs before comparing them
        for accessor in [dem_base, xr_base.dem, dem_mp, ds.dem]:
            accessor.set_area_or_point(area_or_point)

        # 2/ Compute transforms and check output laziness
        # Transform the two loaded inputs through the DEM and Xarray interfaces
        base_dem = dem_base.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(base_dem, xdem.DEM)

        base_xr = xr_base.dem.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(base_xr, xr.DataArray)

        # Run the multiprocessing version and check that its input and result remain unloaded
        mp_dem = dem_mp.to_vcrs(
            vcrs=dst_vcrs,
            force_source_vcrs=force_source_vcrs,
            mp_config=mp_config,
        )
        assert isinstance(mp_dem, xdem.DEM)
        assert not mp_dem.is_loaded

        # Create the Dask result and check that the result still contains delayed work
        dask_dem = ds.dem.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(dask_dem, xr.DataArray)
        assert isinstance(dask_dem.data, da.Array)

        # Check that the multiprocessing and Dask calls did not load their inputs
        assert not dem_mp.is_loaded
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)

        # 3/ Compare outputs
        dask_dem = dask_dem.compute()
        assert base_dem.raster_equal(dask_dem, warn_failure_reason=True, strict_masked=False)
        assert base_dem.raster_equal(mp_dem, warn_failure_reason=True, strict_masked=False)
        assert base_dem.raster_equal(base_xr, warn_failure_reason=True, strict_masked=False)

        # Check that computing the result did not load or replace either source opened lazily
        assert ds.data is source_array and not ds._in_memory
        assert not dem_mp.is_loaded

        # 4/ Compare with an independent PyProj transformation
        reference_crs = xdem.DEM(dem_base, vcrs=force_source_vcrs).crs
        transformer = Transformer.from_crs(reference_crs, base_dem.crs, always_xy=True)
        xx, yy = dem_base.coords(grid=True)
        expected = transformer.transform(xx, yy, dem_base.data)[2].astype(dem_base.dtype)
        np.testing.assert_array_equal(base_dem.data, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("with_band", [False, True])
    @pytest.mark.parametrize("area_or_point", ["Area", "Point"])
    @pytest.mark.parametrize("chunks", [(7, 11), (10, 6)])
    def test_to_vcrs__chunked_coordinates_and_dimensions(
        self,
        tmp_path: pathlib.Path,
        dtype: Any,
        with_band: bool,
        area_or_point: Literal["Area", "Point"],
        chunks: tuple[int, int],
    ) -> None:
        """Checks that every raster input uses the same fractional pixel coordinates for a geoid transformation."""

        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # 1/ Open test files with fractional pixel sizes and large projected coordinates
        # Use fractional pixel sizes and origins so the test would fail if chunks used slightly different coordinates
        rows, cols = np.indices((23, 29))
        values = (900 + rows * 1.3 + cols * 1.7).astype(dtype)

        # Add missing values inside the DEM and at its edge to check that they remain missing
        values[6:10, 9:14] = np.nan
        values[0, 0] = np.nan
        reference = xdem.DEM.from_array(
            values,
            Affine(20.1, 0, 500000.3, 0, -19.7, 8600000.7),
            32633,
            nodata=-9999,
            area_or_point=area_or_point,
            tags={"survey": "fractional"},
        )
        reference.set_vcrs("EGM96")
        path = tmp_path / "fractional.tif"
        reference.to_file(path)

        # Open the same file as an unloaded DEM, a loaded Xarray and a Dask Xarray
        native = xdem.DEM(path)
        eager = xdem.open_dem(path)
        lazy = xdem.open_dem(path, chunks={"y": chunks[0], "x": chunks[1]})

        # Optionally add a band dimension to check both supported Xarray shapes
        if with_band:
            eager = eager.expand_dims(band=[1])
            lazy = lazy.expand_dims(band=[1])

        # Save the Dask data object so the final check can confirm that the source was not replaced
        graph = lazy.data

        # Xarray receives the chunk sizes as Y/X above, while multiprocessing expects X/Y
        config = MultiprocConfig(chunks=(chunks[1], chunks[0]), outfile=str(tmp_path / "vertical.tif"))

        # 2/ Compute transforms and check output laziness
        # Record executed Dask tasks while creating the result; the list must remain empty until compute is called
        tasks: list[Any] = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            lazy_result = lazy.dem.to_vcrs("Ellipsoid")
        assert tasks == []
        assert isinstance(lazy_result.data, da.Array) and not lazy_result._in_memory

        # Write the multiprocessing result to disk without loading the input or returned DEM into memory
        parallel = native.to_vcrs("Ellipsoid", mp_config=config)
        assert isinstance(parallel, xdem.DEM) and not parallel.is_loaded
        assert not native.is_loaded and not lazy._in_memory

        # Transform the loaded DEM and Xarray to obtain expected values without splitting the data into chunks
        expected = reference.to_vcrs("Ellipsoid")
        eager_result = eager.dem.to_vcrs("Ellipsoid")
        assert isinstance(expected, xdem.DEM)

        # 3/ Compare exact elevations, missing values, dimensions and vertical metadata
        computed = lazy_result.compute()

        # Check the Xarray shape first, then remove the optional band only for the value comparison
        for result in [eager_result, computed]:
            assert isinstance(result, xr.DataArray) and result.dims == eager.dims
            assert result.dtype == expected.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(result.data.squeeze(), expected.get_nanarray())
            assert result.dem.crs == expected.crs
            assert result.dem.area_or_point == area_or_point
            assert result.dem.tags["survey"] == "fractional"

        # Compare the multiprocessing result and confirm that its output file has the expected CRS
        assert expected.raster_equal(parallel, strict_masked=False, warn_failure_reason=True)
        assert xdem.DEM(config.outfile).crs == expected.crs

        # 4/ Compare with PyProj evaluated at the full raster's original coordinates
        # Transform the full coordinate grid with PyProj, without calling xDEM's transformation method
        xx, yy = reference.coords(grid=True)
        transformer = Transformer.from_crs(reference.crs, expected.crs, always_xy=True)
        transformed = transformer.transform(xx, yy, reference.get_nanarray())[2].astype(dtype)
        np.testing.assert_array_equal(expected.get_nanarray(), transformed)

        # Check that calculating the Dask result did not load the source or replace its data object
        assert lazy.data is graph and not lazy._in_memory
        assert not native.is_loaded

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("point_count", [0, 1, 17])
    @pytest.mark.parametrize("partition_size", [3, 7])
    def test_to_vcrs__point_partitions(
        self,
        dtype: Any,
        point_count: int,
        partition_size: int,
    ) -> None:
        """
        Checks that regular, Dask and multiprocessing point clouds give the same result, including when empty.
        """

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        from xdem.epc.pd_accessor import _register_dask_epc_accessor

        # 1/ Prepare points whose elevations, row order and extra values can all be compared exactly
        coordinates = np.arange(point_count)
        heights = (100 + coordinates * 0.25).astype(dtype)

        # Add a missing elevation to the larger point cloud
        if point_count > 1:
            heights[5] = np.nan

        # Use spaced index values to check row order and an integer intensity column to check its type is kept
        frame = gpd.GeoDataFrame(
            {"height": heights, "intensity": coordinates.astype(np.uint16)},
            geometry=gpd.points_from_xy(500000 + coordinates * 20, 8600000 - coordinates * 20),
            crs="EPSG:32633+5703",
            index=coordinates * 3 + 7,
        )
        frame.attrs["data_column"] = "height"
        original = frame.copy()

        # Create EPC and Dask versions; the original GeoDataFrame is used for the Pandas accessor call below
        native = xdem.EPC(frame, data_column="height")
        _register_dask_epc_accessor()
        source = dgpd.from_geopandas(frame, chunksize=partition_size)
        source.epc.set_data_column("height")

        # Save Dask's work plan so the final check can confirm that the source was not changed
        graph = source.expr

        # Give multiprocessing a different chunk size so the two methods split the points differently
        config = MultiprocConfig(chunks=partition_size + 1)

        # 2/ Transform with Pandas, Dask, EPC and multiprocessing
        # Record executed Dask tasks; creating the result must not read any points
        tasks: list[Any] = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            lazy = source.epc.to_vcrs(6360)
        assert tasks == [] and isinstance(lazy, dgpd.GeoDataFrame)

        # Run the same transformation with EPC, Pandas and multiprocessing
        expected = native.to_vcrs(6360)
        parallel = native.to_vcrs(6360, mp_config=config)
        eager = frame.epc.to_vcrs(6360)

        # 3/ Compare every column, index and geometry, including empty results
        assert isinstance(expected, xdem.EPC) and isinstance(parallel, xdem.EPC)

        # Compare complete tables so changed point order, index values, column types or geometry would fail
        for result in [parallel.ds, eager, lazy.compute()]:
            assert_frame_equal(result, expected.ds, check_exact=True)
            assert result.crs == expected.crs
            assert result.height.dtype == np.dtype(dtype)

        # Call PyProj directly to check the transformed elevations without using xDEM
        transformer = Transformer.from_crs(frame.crs, expected.crs, always_xy=True)
        elevations = transformer.transform(
            frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy(), frame.height.to_numpy()
        )[2]
        np.testing.assert_array_equal(expected.data, np.asarray(elevations).astype(dtype))

        # 4/ Check that computing the results did not change the original table or the Dask source
        assert_frame_equal(frame, original, check_exact=True)
        assert source.expr is graph and not source.epc.is_loaded
        assert source.epc.crs == original.crs

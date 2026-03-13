"""Tests for vertical CRS transformation tools."""

from __future__ import annotations

import pathlib
import re
import warnings
from typing import Any

import numpy as np
import xarray as xr
import pytest
from pyproj import CRS
import geoutils as gu
from geoutils.multiproc import MultiprocConfig

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
            assert vcrs.equals(input_output[1])
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
    def test_build_ccrs_from_crs_and_vcrs(self, crs: CRS, vcrs_input: CRS | str) -> None:
        """Test the function build_ccrs_from_crs_and_vcrs."""

        # Most grids aren't going to be downloaded, so this warning can be raised
        warnings.filterwarnings("ignore", category=UserWarning, message="Grid .*")

        # Get the vertical CRS from user input
        vcrs = xdem.vcrs._vcrs_from_user_input(vcrs_input=vcrs_input)

        # Build the compound CRS

        # For a 3D horizontal CRS, a condition based on pyproj version is needed
        if len(crs.axis_info) > 2:
            import pyproj
            from packaging.version import Version

            # If the version is higher than 3.5.0, it should pass
            if Version(pyproj.__version__) > Version("3.5.0"):
                ccrs = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs, vcrs=vcrs)
            # Otherwise, it should raise an error
            else:
                with pytest.raises(
                    NotImplementedError,
                    match="pyproj >= 3.5.1 is required to demote a 3D CRS to 2D and be able to compound "
                    "with a new vertical CRS. Update your dependencies or pass the 2D source CRS "
                    "manually.",
                ):
                    xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs, vcrs=vcrs)
                return None
        # If the CRS is 2D, it should pass
        else:
            ccrs = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs, vcrs=vcrs)

        assert isinstance(ccrs, CRS)
        is_3d = len(ccrs.axis_info) == 3
        assert is_3d

    def test_build_ccrs_from_crs_and_vcrs__errors(self) -> None:
        """Test errors are correctly raised from the build_ccrs function."""

        with pytest.raises(
            ValueError, match="Invalid vcrs given. Must be a vertical " "CRS or the literal string 'Ellipsoid'."
        ):
            xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=CRS("EPSG:4326"), vcrs="NotAVerticalCRS")  # type: ignore

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
        ccrs_from = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs_from, vcrs="Ellipsoid")

        # Build the compound CRS
        vcrs_to = xdem.vcrs._vcrs_from_user_input(vcrs_input=grid_shifts["grid"])
        ccrs_to = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs_from, vcrs=vcrs_to)
        transformer = xdem.vcrs._build_vertical_transformer(crs_from=ccrs_from, crs_to=ccrs_to)
        # Apply the transformation
        zz_trans = xdem.vcrs._transform_zz(transformer=transformer, xx=xx, yy=yy, zz=zz)

        # Compare the elevation difference
        z_diff = 100 - zz_trans

        # Check the shift is the one expect within 10%
        assert z_diff == pytest.approx(grid_shifts["shift"], rel=0.1)


class TestToVCRSChunked:

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
    ) -> None:
        """
        Test that to_vcrs yields identical or nearly identical output for base (in-memory),
        chunked Dask, and Multiprocessing backends.

        Notes:
          - Vertical transforms are pointwise, so outputs should generally match exactly.
          - We still use a small tolerance to remain robust to backend-specific casting/order.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Get DEM path
        path_dem = examples.get_path_test("longyearbyen_ref_dem")

        # 1/ Open test files
        # DEM base input (in-memory)
        dem_base = xdem.DEM(path_dem)
        dem_base.load()

        # Xarray base input (in-memory data array)
        xr_base = gu.open_raster(path_dem)
        xr_base.load()

        # Multiprocessing input (keep lazy)
        dem_mp = xdem.DEM(path_dem)
        mp_config = MultiprocConfig(chunk_size=10)

        # Dask input (lazy)
        ds = gu.open_raster(path_dem, chunks={"x": 10, "y": 10})
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None

        # 2/ Compute transforms and check output laziness
        # DEM base
        base_dem = dem_base.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(base_dem, xdem.DEM)

        # Xarray base
        base_xr = xr_base.dem.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(base_xr, xr.DataArray)

        # Multiprocessing
        mp_dem = dem_mp.to_vcrs(
            vcrs=dst_vcrs,
            force_source_vcrs=force_source_vcrs,
            mp_config=mp_config,
        )
        assert isinstance(mp_dem, xdem.DEM)
        assert not mp_dem.is_loaded

        # Dask
        dask_dem = ds.dem.to_vcrs(vcrs=dst_vcrs, force_source_vcrs=force_source_vcrs)
        assert isinstance(dask_dem, xr.DataArray)
        assert isinstance(dask_dem.data, da.Array)

        # Inputs also stay lazy where expected
        assert not dem_mp.is_loaded
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)

        # 3/ Compare outputs
        # Vertical CRS transform is pointwise, so differences should be tiny
        dask_dem = dask_dem.compute()
        assert base_dem.raster_equal(dask_dem, warn_failure_reason=True, strict_masked=False)
        assert base_dem.raster_equal(mp_dem, warn_failure_reason=True, strict_masked=False)
        assert base_dem.raster_equal(base_xr, warn_failure_reason=True, strict_masked=False)
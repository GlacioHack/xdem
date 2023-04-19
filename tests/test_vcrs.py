"""Tests for vertical CRS transformation tools."""

import os
import re
import pathlib
from typing import Any

from pyproj.crs import CRS, VerticalCRS, BoundCRS, CompoundCRS
from pyproj.transformer import Transformer
import pytest
import rasterio as rio
import numpy as np

import xdem
import xdem.vcrs
from xdem import DEM


class TestVCRS:

    def test_parse_vcrs_name_from_product(self) -> None:
        """Test parsing of vertical CRS name from DEM product name."""

        # Check that the value for the key is returned by the function
        for product in xdem.vcrs.vcrs_dem_products.keys():
            assert xdem.vcrs._parse_vcrs_name_from_product(product) == xdem.vcrs.vcrs_dem_products[product]

        # And that, otherwise, it's a None
        assert xdem.vcrs._parse_vcrs_name_from_product("BESTDEM") is None

    @pytest.mark.parametrize("vcrs_input", ["EGM08",
                                            "EGM96",
                                            "us_noaa_geoid06_ak.tif",
                                            pathlib.Path("is_lmi_Icegeoid_ISN93.tif"),
                                            3855,
                                            CRS.from_epsg(5773)])  # type: ignore
    def test_vcrs_from_user_input(self, vcrs_input: str | pathlib.Path | int | CRS) -> None:
        """Tests the function _vcrs_from_user_input for varying user inputs, for which it will return a CRS."""

        # Get user input
        vcrs = xdem.dem._vcrs_from_user_input(vcrs_input)

        # Check output type
        assert isinstance(vcrs, CRS)
        assert vcrs.is_vertical

    @pytest.mark.parametrize("vcrs_input", [
        "Ellipsoid",
        "ellipsoid",
        "wgs84",
        4326,
        4979,
        CRS.from_epsg(4326),
        CRS.from_epsg(4979)])  # type: ignore
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
        with pytest.raises(ValueError, match=re.escape("New vertical CRS must have a vertical axis, 'WGS 84 / UTM zone 1N' does not (check with `CRS.is_vertical`).")):
            xdem.vcrs._vcrs_from_user_input(32601)

        # Check that a warning is raised if the CRS has other dimensions than vertical
        with pytest.warns(UserWarning, match="New vertical CRS has a vertical dimension but also other components, "
                                             "extracting the first vertical reference only."):
            xdem.vcrs._vcrs_from_user_input(CRS("EPSG:4326+5773"))

    @pytest.mark.parametrize("grid", ["us_noaa_geoid06_ak.tif", "is_lmi_Icegeoid_ISN93.tif",
                                            "us_nga_egm08_25.tif", "us_nga_egm96_15.tif"])  # type: ignore
    def test_build_vcrs_from_grid(self, grid: str) -> None:
        """Test that vertical CRS are correctly built from grid"""

        # Build vertical CRS
        vcrs = xdem.vcrs._build_vcrs_from_grid(grid=grid)
        assert vcrs.is_vertical

        # Check that the explicit construction yields the same CRS as "the old init way" (see function description)
        vcrs_oldway = xdem.vcrs._build_vcrs_from_grid(grid=grid, old_way=True)
        assert vcrs.equals(vcrs_oldway)

    # Test for WGS84 in 2D and 3D, UTM, CompoundCRS, everything should work
    @pytest.mark.parametrize("crs", [CRS("EPSG:4326"), CRS("EPSG:4979"), CRS("32610"), CRS("EPSG:4326+5773")])  # type: ignore
    @pytest.mark.parametrize("vcrs_input", [CRS("EPSG:5773"), "is_lmi_Icegeoid_ISN93.tif", "EGM96"])  # type: ignore
    def test_build_ccrs_from_crs_and_vcrs(self, crs: CRS, vcrs_input: CRS | str) -> None:
        """Test the function build_ccrs_from_crs_and_vcrs."""

        # Get the vertical CRS from user input
        vcrs = xdem.vcrs._vcrs_from_user_input(vcrs_input=vcrs_input)

        # Build the compound CRS
        ccrs = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs, vcrs=vcrs)

        assert isinstance(ccrs, CRS)
        assert ccrs.is_vertical


    # Compare to manually-extracted shifts at specific coordinates for the geoid grids
    egm96_chile = {"grid": "us_nga_egm96_15.tif", "lon": -68, "lat": -20, "shift": 42}
    egm08_chile = {"grid": "us_nga_egm08_25.tif", "lon": -68, "lat": -20, "shift": 42}
    geoid96_alaska = {"grid": "us_noaa_geoid06_ak.tif", "lon": -145, "lat": 62, "shift": 15}
    isn93_iceland = {"grid": "is_lmi_Icegeoid_ISN93.tif", "lon": -18, "lat": 65, "shift": 68}

    @pytest.mark.parametrize("grid_shifts", [egm08_chile, egm08_chile, geoid96_alaska, isn93_iceland])
    def test_transform_zz(self, grid_shifts: dict[str, Any]) -> None:
        """Tests grids to convert vertical CRS."""

        # Using an arbitrary elevation of 100 m (no influence on the transformation)
        zz = 100
        xx = grid_shifts["lon"]
        yy = grid_shifts["lat"]
        crs_from = CRS.from_epsg(4326)
        ccrs_from = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs_from, vcrs="Ellipsoid")

        # Build the compound CRS
        vcrs_to = xdem.vcrs._vcrs_from_user_input(vcrs_input=grid_shifts["grid"])
        ccrs_to = xdem.vcrs._build_ccrs_from_crs_and_vcrs(crs=crs_from, vcrs=vcrs_to)

        # Apply the transformation
        zz_trans = xdem.vcrs._transform_zz(crs_from=ccrs_from, crs_to=ccrs_to, xx=xx, yy=yy, zz=zz)

        # Compare the elevation difference
        z_diff = 100 - zz_trans

        # Check the shift is the one expect within 10%
        assert z_diff == pytest.approx(grid_shifts["shift"], rel=0.1)

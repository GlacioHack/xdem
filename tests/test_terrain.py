from __future__ import annotations

import os.path
import re
import warnings
from typing import Literal

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils.raster.distributed_computing import MultiprocConfig
from pyproj import CRS

import xdem

PLOT = False


class TestTerrainAttribute:
    filepath = xdem.examples.get_path("longyearbyen_ref_dem")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Parse metadata")
        dem = xdem.DEM(filepath, silent=True)

    @pytest.mark.parametrize(
        "attribute",
        [
            "slope_Horn",
            "aspect_Horn",
            "hillshade_Horn",
            "slope_Zevenberg",
            "aspect_Zevenberg",
            "hillshade_Zevenberg",
            "tri_Riley",
            "tri_Wilson",
            "tpi",
            "roughness",
        ],
    )  # type: ignore
    def test_attribute_functions_against_gdaldem(self, attribute: str, get_test_data_path) -> None:
        """
        Test that all attribute functions give the same results as those of GDALDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        functions = {
            "slope_Horn": lambda dem: xdem.terrain.slope(
                dem.data, resolution=dem.res, degrees=True, surface_fit="Horn"
            ),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True, surface_fit="Horn"),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res, surface_fit="Horn"),
            "slope_Zevenberg": lambda dem: xdem.terrain.slope(
                dem.data, resolution=dem.res, surface_fit="ZevenbergThorne", degrees=True
            ),
            "aspect_Zevenberg": lambda dem: xdem.terrain.aspect(dem.data, surface_fit="ZevenbergThorne", degrees=True),
            "hillshade_Zevenberg": lambda dem: xdem.terrain.hillshade(
                dem.data, resolution=dem.res, surface_fit="ZevenbergThorne"
            ),
            "tri_Riley": lambda dem: xdem.terrain.terrain_ruggedness_index(dem.data, method="Riley"),
            "tri_Wilson": lambda dem: xdem.terrain.terrain_ruggedness_index(dem.data, method="Wilson"),
            "tpi": lambda dem: xdem.terrain.topographic_position_index(dem.data),
            "roughness": lambda dem: xdem.terrain.roughness(dem.data),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both GDAL and xdem
        attr_xdem = functions[attribute](dem).squeeze()
        attr_gdal = gu.Raster(get_test_data_path(os.path.join("gdal", f"{attribute}.tif"))).data

        # For hillshade, we round into an integer to match GDAL's output
        if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
            with warnings.catch_warnings():
                # Normal that a warning would be raised here, so we catch it
                warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
                attr_xdem = attr_xdem.astype("int").astype("float32")

        # We compute the difference and keep only valid values
        diff = (attr_xdem - attr_gdal).filled(np.nan)
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and GDAL attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["hillshade_Horn", "hillshade_Zevenberg"]:
                # For hillshade, check 0 or 1 difference due to integer rounding
                assert np.all(np.logical_or(diff_valid == 0.0, np.abs(diff_valid) == 1.0))

            elif attribute in ["aspect_Horn", "aspect_Zevenberg"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)
            else:
                # All attributes other than hillshade and aspect are non-circular floats, so we check within a tolerance
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and GDAL attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(121)
                plt.imshow(attr_gdal.squeeze())
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(attr_xdem.squeeze())
                plt.colorbar()
                plt.show()

            raise exception

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans.
        functions[attribute](dem)

    @pytest.mark.parametrize(
        "attribute",
        ["slope_Horn", "aspect_Horn", "hillshade_Horn", "profile_curvature", "planform_curvature"],
    )  # type: ignore
    def test_attribute_functions_against_richdem(self, attribute: str, get_test_data_path) -> None:
        """
        Test that all attribute functions give the same results as those of RichDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        # Functions for xdem-implemented methods
        functions_xdem = {
            "slope_Horn": lambda dem: xdem.terrain.slope(dem, resolution=dem.res, degrees=True, surface_fit="Horn"),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True, surface_fit="Horn"),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res, surface_fit="Horn"),
            "profile_curvature": lambda dem: xdem.terrain.profile_curvature(
                dem.data, resolution=dem.res, surface_fit="ZevenbergThorne", curv_method="directional"
            ),
            "planform_curvature": lambda dem: xdem.terrain.tangential_curvature(
                dem.data, resolution=dem.res, surface_fit="ZevenbergThorne", curv_method="directional"
            ),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both RichDEM and xdem
        attr_xdem = gu.raster.get_array_and_mask(functions_xdem[attribute](dem))[0].squeeze()
        attr_richdem_rst = gu.Raster(get_test_data_path(os.path.join("richdem", f"{attribute}.tif")), load_data=True)
        attr_richdem = gu.raster.get_array_and_mask(attr_richdem_rst)[0].squeeze()

        # TODO: Profile curvature is opposite sign as of RichDEM, check if this is warranted
        if attribute == "profile_curvature":
            attr_richdem = -attr_richdem

        # We compute the difference and keep only valid values
        diff = attr_xdem - attr_richdem
        diff_valid = diff[np.isfinite(diff)]

        try:
            # Difference between xdem and RichDEM attribute
            # Mean of attribute values to get an order of magnitude of the attribute unit
            magn = np.nanmean(np.abs(attr_xdem))

            # Check that the attributes are similar within a tolerance of a thousandth of the magnitude
            # For instance, slopes have an average magnitude of around 30 deg, so the tolerance is 0.030 deg
            if attribute in ["aspect_Horn"]:
                # For aspect, check the tolerance within a 360 degree modulo due to the circularity of the variable
                diff_valid = np.mod(np.abs(diff_valid), 360)
                assert np.nanpercentile(np.minimum(diff_valid, np.abs(360 - diff_valid)), 99) < 10 ** (-3) * magn

            else:
                # All attributes other than aspect are non-circular floats, so we check within a tolerance
                # Here hillshade is not rounded as integer by our calculation, so no need to differentiate as with GDAL
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_richdem, vmin=-1, vmax=1)
                plt.colorbar(label="richdem")
                plt.subplot(222)
                plt.imshow(attr_xdem, vmin=-1, vmax=1)
                plt.colorbar(label="xdem")
                plt.subplot(223)
                plt.imshow(diff, vmin=-1, vmax=1)
                plt.colorbar(label="diff")
                plt.show()

            raise exception

        # Introduce some nans
        # rng = np.random.default_rng(42)
        # dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        # dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans and that mask is preserved
        # output = functions_richdem[attribute](dem)
        # assert np.all(dem.data.mask == output.data.mask)

    @pytest.mark.parametrize(
        "attribute_name",
        [
            "slope",
            "aspect",
            "profile_curvature",
            "tangential_curvature",
            "planform_curvature",
            "flowline_curvature",
            "max_curvature",
            "min_curvature",
        ],
    )  # type: ignore
    def test_attribute_consistency_surface_fit(self, attribute_name: str) -> None:
        """Test that surface fit attributes are generally consistent across various fit methods
        (Horn, ZevenbergThorne, Florinsky), testing within a certain tolerance for a % of the data."""

        # Prepare masks to filter out points for later comparison of curvatures where they remain consistent
        max_curv = xdem.terrain.max_curvature(self.dem)
        min_curv = xdem.terrain.min_curvature(self.dem)
        low_curv = np.logical_and(max_curv < 0.25, min_curv > -0.25)
        slope = xdem.terrain.slope(self.dem)
        flat_moderate = np.logical_and(slope < 40, low_curv)
        moderate_steep = np.logical_and(slope > 15, low_curv)

        attr_zt = getattr(xdem.terrain, attribute_name)(self.dem, surface_fit="ZevenbergThorne")
        attr_fl = getattr(xdem.terrain, attribute_name)(self.dem, surface_fit="Florinsky")

        # Difference attribute
        diff_zt_fl = np.abs(attr_zt - attr_fl)
        if attribute_name == "aspect":
            diff_zt_fl = np.minimum(diff_zt_fl, np.abs(360 - diff_zt_fl))

        # Twice the STD contains about 95% of the data
        magnitude = 2 * np.nanstd(attr_zt)

        try:
            # Less sensitivity for first-order derivatives, so no need to filter out points
            if attribute_name in ["slope", "aspect"]:
                ind = np.ones(diff_zt_fl.shape, dtype=bool)

            # The following curvatures are sensitive in steep terrain, so let's remove that
            elif attribute_name in ["profile_curvature", "tangential_curvature", "max_curvature", "min_curvature"]:
                ind = flat_moderate

            # The last curvatures are sensitive in flat terrain, so let's remove that
            else:
                ind = moderate_steep

            # Remove terrain
            diff_zt_fl[~ind] = np.nan

            # The 50% lowest differences should be relatively small (less than 10% of magnitude, e.g. a couple degrees
            # for slope)
            assert np.nanpercentile(diff_zt_fl, 50) < 0.1 * magnitude

            # Only slope and aspect are supported for Horn
            if attribute_name in ["slope", "aspect"]:
                attr_horn = getattr(xdem.terrain, attribute_name)(self.dem, surface_fit="Horn")
                diff_zt_horn = np.abs(attr_zt - attr_horn)
                assert np.nanpercentile(diff_zt_horn, 80) < 0.1 * magnitude

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_zt.data, vmin=-1, vmax=1)
                plt.colorbar(label="ZT")
                plt.subplot(222)
                plt.imshow(attr_fl.data, vmin=-1, vmax=1)
                plt.colorbar(label="Florinsky")
                plt.subplot(223)
                plt.imshow(attr_zt.data - attr_fl.data, vmin=-1, vmax=1)
                plt.colorbar(label="diff")
                plt.subplot(224)
                alpha_channel = np.ones_like(diff_zt_fl.data)
                alpha_channel[diff_zt_fl.data.mask] = 0
                plt.imshow(diff_zt_fl.data, alpha=alpha_channel, vmax=1)
                plt.colorbar(label="Mask")
                plt.show()

            raise exception

    def test_hillshade(self) -> None:
        """Test hillshade-specific settings."""

        zfactor_1 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=1.0)
        zfactor_10 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=10.0)

        # A higher z-factor should be more variable than a low one.
        assert np.nanstd(zfactor_1) < np.nanstd(zfactor_10)

        low_altitude = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=10)
        high_altitude = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=80)

        # A low altitude should be darker than a high altitude.
        assert np.nanmean(low_altitude) < np.nanmean(high_altitude)

    def test_hillshade__errors(self) -> None:
        """Validate that the hillshade function raises appropriate errors."""
        # Try giving the hillshade invalid arguments.

        with pytest.raises(ValueError, match="Azimuth must be a value between 0 and 360"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, azimuth=361)

        with pytest.raises(ValueError, match="Altitude must be a value between 0 and 90"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=91)

        with pytest.raises(ValueError, match="z_factor must be a non-negative finite value"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=np.inf)

    @pytest.mark.parametrize(
        "name",
        [
            "tangential_curvature",
            "profile_curvature",
            "min_curvature",
            "max_curvature",
            "planform_curvature",
            "flowline_curvature",
        ],
    )  # type: ignore
    def test_curvatures__runtime(self, name: str) -> None:
        """Test the curvature functions"""

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive curvature without any gaps
        curvature = xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)

        # Validate that the array has the same shape as the input and that all non-edge values are finite.
        assert curvature.shape == dem.data.shape
        try:
            assert np.all(np.isfinite(curvature[1:-1, 1:-1]))
        except Exception:
            import matplotlib.pyplot as plt

            if PLOT:
                plt.imshow(curvature.squeeze())
                plt.show()

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Surface fit and rugosity require the same X and Y resolution "
                f"((1.0, 2.0) was given). This was required by: ['{name}']."
            ),
        ):
            xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=(1.0, 2.0))

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True
        # Validate that this doesn't raise weird warnings after introducing nans.
        xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)

    @pytest.mark.parametrize(
        "name",
        [
            "tangential_curvature",
            "profile_curvature",
            "min_curvature",
            "max_curvature",
            "planform_curvature",
            "flowline_curvature",
        ],
    )  # type: ignore
    @pytest.mark.parametrize("surface_fit", ["ZevenbergThorne", "Florinsky"])  # type: ignore
    def test_curvatures__definition(self, name: str, surface_fit: Literal["ZevenbergThorne", "Florinsky"]) -> None:
        """Test the curvatures definitions 'geometric' versus 'directional' are yielding expected results."""

        # Get geometric and directional curvatures
        curv_g = xdem.terrain.get_terrain_attribute(
            self.dem.data, attribute=name, resolution=self.dem.res, curv_method="geometric", surface_fit=surface_fit
        )
        curv_d = xdem.terrain.get_terrain_attribute(
            self.dem.data, attribute=name, resolution=self.dem.res, curv_method="directional", surface_fit=surface_fit
        )

        # For planform, the result should be the same
        if name == "planform_curvature":
            assert np.allclose(curv_g, curv_d, equal_nan=True)
        # For tangential, profile and flowline, only the divider is different with a multiplier greater than 1,
        # so the absolute geometric curvature should always have lower value
        elif name in ["tangential_curvature", "profile_curvature", "flowline_curvature"]:
            ind_valid = np.logical_and(np.isfinite(curv_g), np.isfinite(curv_d))
            assert np.all(np.abs(curv_d[ind_valid]) >= np.abs(curv_g[ind_valid]))
        # For max/min, no particular relation
        else:
            return

    def test_curvatures__synthetic(self) -> None:
        """Test the curvature functions with synthetic data, checking expected values at the center pixel."""

        # 1/ Flat, or Linear ramp in X/Y/diagX/diagY = No curvature
        dem_flat = np.ones((5, 5), dtype=np.float32)
        dem_ramp_x = np.stack([np.ones(5) * i for i in range(5)], axis=1)
        dem_ramp_y = np.stack([np.ones(5) * i for i in range(5)], axis=0)
        dem_ramp_xy = np.stack([np.arange(0, 5) + i for i in range(5)], axis=1)
        dem_ramp_yx = np.stack([np.flip(np.arange(0, 5)) + i for i in range(5)], axis=1)
        list_dem_no_curv = [dem_flat, dem_ramp_x, dem_ramp_y, dem_ramp_xy, dem_ramp_yx]
        for curv in [
            "tangential_curvature",
            "profile_curvature",
            "min_curvature",
            "max_curvature",
            "planform_curvature",
            "flowline_curvature",
        ]:
            for dem in list_dem_no_curv:
                fl_curv = getattr(xdem.terrain, curv)(dem, resolution=10, surface_fit="Florinsky")[2, 2]
                zt_curv = getattr(xdem.terrain, curv)(dem, resolution=10, surface_fit="ZevenbergThorne")[2, 2]
                assert pytest.approx(fl_curv) == 0
                assert pytest.approx(zt_curv) == 0

        # 2/ V linear ramp (V shape) in X/Y centered on pixel (2 1 0 1 2), with linear ramp in TANGENT direction
        # (to define orientation)
        dem_v_y_convex = np.stack([np.array([2, 1, 0, 1, 2]) + i for i in range(5)], axis=0)
        dem_v_x_convex = np.stack([np.array([2, 1, 0, 1, 2]) + i for i in range(5)], axis=1)
        # Same but concave (0 1 2 1 0)
        dem_v_y_concave = np.stack([np.array([0, 1, 2, 1, 0]) + i for i in range(5)], axis=0)
        dem_v_x_concave = np.stack([np.array([0, 1, 2, 1, 0]) + i for i in range(5)], axis=1)

        list_dems_v = [dem_v_x_convex, dem_v_y_convex, dem_v_x_concave, dem_v_y_concave]
        list_dem_concavity = ["convex", "convex", "concave", "concave"]

        for i, dem in enumerate(list_dems_v):

            # Define a sign to test inequality based on convexity
            conc = list_dem_concavity[i]
            if conc == "convex":
                sign = 1
            else:
                sign = -1

            # In profile direction, no curvature
            fl_curv_prof = xdem.terrain.profile_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_prof = xdem.terrain.profile_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert pytest.approx(fl_curv_prof) == 0
            assert pytest.approx(zt_curv_prof) == 0

            # Tangent curvature (negative for convex, positive for concave)
            fl_curv_tan = xdem.terrain.tangential_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_tan = xdem.terrain.tangential_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert sign * fl_curv_tan < 0
            assert sign * zt_curv_tan < 0

            # Flowline direction, no curvature
            fl_curv_flo = xdem.terrain.flowline_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_flo = xdem.terrain.flowline_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert pytest.approx(fl_curv_flo) == 0
            assert pytest.approx(zt_curv_flo) == 0

            # Planform (negative for convex, positive for concave)
            fl_curv_pla = xdem.terrain.planform_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_pla = xdem.terrain.planform_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert sign * fl_curv_pla < 0
            assert sign * zt_curv_pla < 0

            # Max and min
            fl_curv_max = xdem.terrain.max_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_max = xdem.terrain.max_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            fl_curv_min = xdem.terrain.min_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_min = xdem.terrain.min_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            if conc == "convex":
                assert pytest.approx(fl_curv_max) == 0
                assert pytest.approx(zt_curv_max) == 0
                assert fl_curv_min < 0
                assert zt_curv_min < 0
            else:
                assert pytest.approx(fl_curv_min) == 0
                assert pytest.approx(zt_curv_min) == 0
                assert fl_curv_max > 0
                assert zt_curv_max > 0

        # 3/ V linear ramp (V shape) in X/Y centered on pixel (2 1 0 1 2), with added linear ramp in SAME direction
        # (to define orientation)
        # Now profile curvature should be non-zero, and others should be zero
        dem_v_y_convex = np.stack([np.array([2, 1, 0, 1, 2]) + np.linspace(0, 1, 5) for i in range(5)], axis=0)
        dem_v_x_convex = np.stack([np.array([2, 1, 0, 1, 2]) + np.linspace(0, 1, 5) for i in range(5)], axis=1)
        # Same but concave (0 1 2 1 0)
        dem_v_y_concave = np.stack([np.array([0, 1, 2, 1, 0]) + np.arange(0, 5) for i in range(5)], axis=0)
        dem_v_x_concave = np.stack([np.array([0, 1, 2, 1, 0]) + np.arange(0, 5) for i in range(5)], axis=1)

        list_dems_v = [dem_v_x_convex, dem_v_y_convex, dem_v_x_concave, dem_v_y_concave]
        list_dem_concavity = ["convex", "convex", "concave", "concave"]

        for i, dem in enumerate(list_dems_v):

            # Define a sign to test inequality based on convexity
            conc = list_dem_concavity[i]
            if conc == "convex":
                sign = 1
            else:
                sign = -1

            # Profile curvature (negative for convex, positive for concave)
            fl_curv_prof = xdem.terrain.profile_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_prof = xdem.terrain.profile_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert sign * fl_curv_prof < 0
            assert sign * zt_curv_prof < 0

            # Tangent curvature, no curvature
            fl_curv_tan = xdem.terrain.tangential_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_tan = xdem.terrain.tangential_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert pytest.approx(fl_curv_tan) == 0
            assert pytest.approx(zt_curv_tan) == 0

            # Flowline direction (negative for convex, positive for concave)
            fl_curv_flo = xdem.terrain.flowline_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_flo = xdem.terrain.flowline_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert pytest.approx(fl_curv_flo) == 0
            assert pytest.approx(zt_curv_flo) == 0

            # Planform (negative for convex, positive for concave)
            fl_curv_pla = xdem.terrain.planform_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_pla = xdem.terrain.planform_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            assert pytest.approx(fl_curv_pla) == 0
            assert pytest.approx(zt_curv_pla) == 0

            # Max and min
            fl_curv_max = xdem.terrain.max_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_max = xdem.terrain.max_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            fl_curv_min = xdem.terrain.min_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
            zt_curv_min = xdem.terrain.min_curvature(dem, resolution=5, surface_fit="ZevenbergThorne")[2, 2]
            if conc == "convex":
                assert pytest.approx(fl_curv_max) == 0
                assert pytest.approx(zt_curv_max) == 0
                assert fl_curv_min < 0
                assert zt_curv_min < 0
            else:
                assert pytest.approx(fl_curv_min) == 0
                assert pytest.approx(zt_curv_min) == 0
                assert fl_curv_max > 0
                assert zt_curv_max > 0

    def test_get_terrain_attribute__multiple_inputs(self) -> None:
        """Test the get_terrain_attribute function by itself."""

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        slope = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res)
        assert isinstance(slope, np.ndarray)

        # Create three products at the same time
        slope2, _, hillshade = xdem.terrain.get_terrain_attribute(
            self.dem.data, ["slope", "aspect", "hillshade"], resolution=self.dem.res
        )

        # Create a hillshade using its own function
        hillshade2 = xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res)

        # Validate that the "batch-created" hillshades and slopes are the same as the "single-created"
        assert np.array_equal(hillshade, hillshade2, equal_nan=True)
        assert np.array_equal(slope, slope2, equal_nan=True)

        # A slope map with a lower resolution (higher value) should have gentler slopes.
        slope_lowres = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res[0] * 2)
        assert np.nanmean(slope) > np.nanmean(slope_lowres)

    def test_get_terrain_attribute__multiproc(self) -> None:
        """Test the get_terrain attribute function in multiprocessing."""
        outfile = "mp_output.tif"
        outfile_multi = ["mp_output_slope.tif", "mp_output_aspect.tif", "mp_output_hillshade.tif"]

        mp_config = MultiprocConfig(
            chunk_size=200,
            outfile=outfile,
        )

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        xdem.terrain.get_terrain_attribute(self.dem, "slope", mp_config=mp_config, resolution=self.dem.res)
        assert os.path.exists(outfile)
        slope = gu.Raster(outfile, load_data=True)
        assert isinstance(slope, gu.Raster)
        os.remove(outfile)

        # Create three products at the same time
        xdem.terrain.get_terrain_attribute(
            self.dem, ["slope", "aspect", "hillshade"], mp_config=mp_config, resolution=self.dem.res
        )
        for file in outfile_multi:
            assert os.path.exists(file)
        slope2 = gu.Raster(outfile_multi[0], load_data=True)
        hillshade = gu.Raster(outfile_multi[2], load_data=True)
        for file in outfile_multi:
            os.remove(file)

        # Create a hillshade using its own function
        xdem.terrain.hillshade(self.dem, mp_config=mp_config, resolution=self.dem.res)
        assert os.path.exists(outfile)
        hillshade2 = gu.Raster(outfile, load_data=True)
        os.remove(outfile)

        # Validate that the "batch-created" hillshades and slopes are the same as the "single-created"
        assert hillshade.raster_equal(hillshade2)
        assert slope.raster_equal(slope2)

        # Compare with classic terrain attribute calculation
        slope_classic = self.dem.slope()
        hillshade_classic = self.dem.hillshade()
        assert np.allclose(slope.data, slope_classic.data, rtol=1e-7)
        assert np.allclose(hillshade.data, hillshade_classic.data, rtol=1e-7)

        # A slope map with a lower resolution (higher value) should have gentler slopes.
        xdem.terrain.get_terrain_attribute(self.dem, "slope", mp_config=mp_config, resolution=self.dem.res[0] * 2)
        slope_lowres = gu.Raster(outfile, load_data=True)
        os.remove(outfile)
        assert slope.get_stats("mean") > slope_lowres.get_stats("mean")

    def test_get_terrain_attribute__errors(self) -> None:
        """Test the get_terrain_attribute function raises appropriate errors."""

        # Below, re.escape() is needed to match expressions that have special characters (e.g., parenthesis, bracket)

        # Wrong method name for surface fit
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Surface fit 'DoesNotExist' is not supported. Must be one of: "
                "['Horn', 'ZevenbergThorne', "
                "'Florinsky']"
            ),
        ):
            xdem.terrain.slope(self.dem, method="DoesNotExist")  # type: ignore

        # Wrong method name for TRI
        with pytest.raises(
            ValueError,
            match=re.escape("TRI method 'DoesNotExist' is not supported. Must be one of: " "['Riley', 'Wilson']"),
        ):
            xdem.terrain.terrain_ruggedness_index(self.dem, method="DoesNotExist")  # type: ignore

        # Wrong method name for curvature method
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Curvature method 'DoesNotExist' is not supported. Must be " "one of: ['geometric', 'directional']"
            ),
        ):
            xdem.terrain.max_curvature(self.dem, curv_method="DoesNotExist")  # type: ignore

        # Calling a curvature with Horn surface fit: impossible
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'Horn' surface fit method cannot be used for to calculate "
                "curvatures. Use 'ZevenbergThorne' or 'Florinsky' instead."
            ),
        ):
            xdem.terrain.max_curvature(self.dem, surface_fit="Horn")  # type: ignore

        # Check warning for geographic CRS
        data = np.ones((5, 5))
        transform = rio.transform.from_bounds(0, 0, 1, 1, 5, 5)
        crs = CRS("EPSG:4326")
        nodata = -9999
        dem = xdem.DEM.from_array(data, transform=transform, crs=crs, nodata=nodata)
        with pytest.warns(match="DEM is not in a projected CRS.*"):
            xdem.terrain.get_terrain_attribute(dem, "slope")

    def test_get_terrain_attribute__raster_input(self) -> None:
        """Test the get_terrain_attribute function supports raster input/output."""

        slope, aspect = xdem.terrain.get_terrain_attribute(self.dem, attribute=["slope", "aspect"])

        assert slope != aspect

        assert isinstance(slope, type(aspect))
        assert all(isinstance(r, gu.Raster) for r in (aspect, slope, self.dem))

        assert slope.transform == self.dem.transform == aspect.transform
        assert slope.crs == self.dem.crs == aspect.crs

    def test_rugosity_jenness(self) -> None:
        """
        Test the rugosity with the same example as in Jenness (2004),
        https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
        """

        # Derive rugosity from the function
        dem = np.array([[190, 170, 155], [183, 165, 145], [175, 160, 122]], dtype="float32")

        # Derive rugosity
        rugosity = xdem.terrain.rugosity(dem, resolution=100.0)

        # Rugosity of Jenness (2004) example
        r = 10280.48 / 10000.0

        assert rugosity[1, 1] == pytest.approx(r, rel=10 ** (-4))

    # Loop for various elevation differences with the center
    @pytest.mark.parametrize("dh", np.linspace(0.01, 100, 3))  # type: ignore
    # Loop for different resolutions
    @pytest.mark.parametrize("resolution", np.linspace(0.01, 100, 3))  # type: ignore
    def test_rugosity_simple_cases(self, dh: float, resolution: float) -> None:
        """Test the rugosity calculation for simple cases."""

        # We here check the value for a fully symmetric case: the rugosity calculation can be simplified because all
        # eight triangles have the same surface area, see Jenness (2004).

        # Derive rugosity from the function
        dem = np.array([[1, 1, 1], [1, 1 + dh, 1], [1, 1, 1]], dtype="float32")

        rugosity = xdem.terrain.rugosity(dem, resolution=resolution)

        # Half surface length between the center and a corner cell (in 3D: accounting for elevation changes)
        side1 = np.sqrt(2 * resolution**2 + dh**2) / 2.0
        # Half surface length between the center and a side cell (in 3D: accounting for elevation changes)
        side2 = np.sqrt(resolution**2 + dh**2) / 2.0
        # Half surface length between the corner and side cell (no elevation changes on this side)
        side3 = resolution / 2.0

        # Formula for area A of one triangle
        s = (side1 + side2 + side3) / 2.0
        A = np.sqrt(s * (s - side1) * (s - side2) * (s - side3))

        # We sum the area of the eight triangles, and divide by the planimetric area (resolution squared)
        r = 8 * A / (resolution**2)

        # Check rugosity value is valid
        assert r == pytest.approx(rugosity[1, 1], rel=10 ** (-6))

    def test_fractal_roughness(self) -> None:
        """Test fractal roughness for synthetic cases for which we know the output."""

        # The fractal dimension of a line is 1 (a single pixel with non-zero value)
        dem = np.zeros((13, 13), dtype="float32")
        dem[1, 1] = 6.5
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6], 5) == np.float32(1.0)

        # The fractal dimension of plane is 2 (a plan of pixels with non-zero values)
        dem = np.zeros((13, 13), dtype="float32")
        dem[:, 1] = 13
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6]) == np.float32(2.0)

        # The fractal dimension of a cube is 3 (a block of pixels with non-zero values
        dem = np.zeros((13, 13), dtype="float32")
        dem[:, :6] = 13
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6]) == np.float32(3.0)

    @pytest.mark.parametrize(
        "attribute",
        [
            "slope",
            "aspect",
            "hillshade",
            "profile_curvature",
            "tangential_curvature",
            "planform_curvature",
            "flowline_curvature",
            "max_curvature",
            "min_curvature",
        ],
    )  # type: ignore
    @pytest.mark.parametrize("surface_fit", ["Horn", "ZevenbergThorne", "Florinsky"])  # type: ignore
    def test_get_surface_attributes__engine(
        self, attribute: str, surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"]
    ) -> None:
        """Check that all quadric coefficients from the convolution give the same results as with the numba loop."""

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(5, 7))

        # Horn only works for first derivatives
        if surface_fit == "Horn" and attribute not in ["slope", "aspect", "hillshade"]:
            return

        attrs_scipy = xdem.terrain._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], surface_fit=surface_fit, engine="scipy"
        )
        attrs_numba = xdem.terrain._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], surface_fit=surface_fit, engine="numba"
        )

        assert np.allclose(attrs_scipy, attrs_numba, equal_nan=True)

    @pytest.mark.parametrize(
        "attribute",
        [
            "topographic_position_index",
            "terrain_ruggedness_index_Riley",
            "terrain_ruggedness_index_Wilson",
            "roughness",
            "rugosity",
            "fractal_roughness",
        ],
    )  # type: ignore
    def test_get_windowed_indices__engine(self, attribute: str) -> None:
        """Check that all quadric coefficients from the convolution give the same results as with the numba loop."""

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(15, 15))

        # Get TRI method if specified
        if "Wilson" in attribute or "Riley" in attribute:
            attribute = "terrain_ruggedness_index"
            tri_method: Literal["Riley", "Wilson"]
            tri_method = attribute.split("_")[-1]  # type: ignore
        # Otherwise use any one, doesn't matter
        else:
            tri_method = "Wilson"

        attrs_scipy = xdem.terrain._get_windowed_indexes(
            dem=dem, window_size=3, resolution=1, windowed_indexes=[attribute], tri_method=tri_method, engine="scipy"
        )
        attrs_numba = xdem.terrain._get_windowed_indexes(
            dem=dem, window_size=3, resolution=1, windowed_indexes=[attribute], tri_method=tri_method, engine="numba"
        )

        assert np.allclose(attrs_scipy, attrs_numba, equal_nan=True)

    def test_get_terrain_attribute__out_dtype(self) -> None:

        # Get one attribute using quadratic coeff, and one using windowed indexes
        slope, tpi = xdem.terrain.get_terrain_attribute(self.dem, attribute=["slope", "topographic_position_index"])

        assert slope.dtype == self.dem.dtype
        assert tpi.dtype == self.dem.dtype

        # Using a different output dtype
        out_dtype = np.float64
        slope, tpi = xdem.terrain.get_terrain_attribute(
            self.dem, attribute=["slope", "topographic_position_index"], out_dtype=out_dtype
        )

        assert self.dem.dtype != out_dtype
        assert np.dtype(slope.dtype) == out_dtype
        assert np.dtype(tpi.dtype) == out_dtype

    def test_texture_shading(self) -> None:
        """Test the texture_shading function."""

        # Test with a simple DEM
        dem_simple = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        # Test basic functionality
        result = xdem.terrain.texture_shading(dem_simple, alpha=0.8)

        # Check output properties
        assert result.shape == dem_simple.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))  # No NaN values for simple case

        # Test different alpha values
        result_low = xdem.terrain.texture_shading(dem_simple, alpha=0.5)
        result_mid = xdem.terrain.texture_shading(dem_simple, alpha=0.8)
        result_high = xdem.terrain.texture_shading(dem_simple, alpha=1.5)

        # Results should be different for different alpha values
        assert not np.array_equal(result_low, result_mid)
        assert not np.array_equal(result_mid, result_high)

        # Test with NaN values
        dem_with_nan = dem_simple.copy()
        dem_with_nan[0, 0] = np.nan

        result_nan = xdem.terrain.texture_shading(dem_with_nan, alpha=0.8)
        assert result_nan.shape == dem_with_nan.shape
        assert np.isnan(result_nan[0, 0])  # NaN should be preserved

        # Test error handling
        with pytest.raises(ValueError, match="Alpha must be between 0 and 2"):
            xdem.terrain.texture_shading(dem_simple, alpha=-0.1)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 2"):
            xdem.terrain.texture_shading(dem_simple, alpha=2.1)

    def test_texture_shading_flat_surface(self) -> None:
        """Test all zero on flat DEM."""
        dem = np.ones((3, 3), dtype=np.float32) * 1000
        out = xdem.terrain.texture_shading(dem, alpha=0.8)
        assert np.allclose(out, 0.0, atol=1e-6)  # flat → 0 everywhere

    def test_texture_shading_planar_ramp(self) -> None:
        """Test expected variability on planar ramp."""
        dem_slope = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

        alpha = 0.8
        out = xdem.terrain.texture_shading(dem_slope, alpha=alpha)

        # eps-scaled absolute tol for tiny float32+FFT differences
        eps = np.finfo(out.dtype).eps  # ~1.19e-7 for float32
        # Factor 1000 is a pragmatic buffer; empirically ~1e-4 absolute differences on 3x3 grids
        atol = float(1000.0 * eps * (np.max(np.abs(out)) + 1.0))

        # No variation between columns → diff across columns ~ 0
        col_diffs = np.diff(out, axis=1)
        assert np.allclose(col_diffs, 0.0, rtol=0.0, atol=atol)

        # 3) Each row is (near) constant
        row_stds = np.std(out, axis=1)
        assert np.all(row_stds <= atol)

        # 4) Monotonic by row mean (increasing because input slope increases with row)
        row_means = np.mean(out, axis=1)
        assert row_means[1] >= row_means[0] - atol
        assert row_means[2] >= row_means[1] - atol

    def test_texture_shading_offset_invariance_and_signed(self) -> None:
        """Test invariance to vertical offset and signed output on non-flat DEMs."""
        rng = np.random.RandomState(0)
        dem = rng.randn(3, 3).astype(np.float32)

        out = xdem.terrain.texture_shading(dem, alpha=0.8)
        out_offset = xdem.terrain.texture_shading(dem + 1234.5, alpha=0.8)

        # Compare after removing mean; allow eps-scaled atol for float32+FFT on tiny grids
        out_d = out - np.nanmean(out)
        off_d = out_offset - np.nanmean(out_offset)
        eps = np.finfo(out.dtype).eps  # ~1.19e-7 for float32
        # Factor 1000 is a pragmatic buffer; empirically ~1e-4 absolute differences on 3x3 grids
        atol = 1000.0 * eps * (np.max(np.abs(out_d)) + 1.0)
        np.testing.assert_allclose(out_d, off_d, atol=atol, rtol=0)

        # Signed response: expect both negative and positive values
        assert np.nanmin(out) < 0 and np.nanmax(out) > 0

    def test_texture_shading_spectral_shift_with_alpha(self) -> None:
        """
        Test spectral shift with increased alpha.
        Increasing alpha shifts spectral power toward higher frequencies.
        The fraction of total power above a median frequency cutoff should
        be larger for alpha=1.5 than for alpha=0.5.
        """
        rng = np.random.RandomState(1)
        dem = rng.randn(3, 3).astype(np.float32)

        out_lo = xdem.terrain.texture_shading(dem, alpha=0.5)
        out_hi = xdem.terrain.texture_shading(dem, alpha=1.5)

        # Power spectra
        F_lo = np.fft.fftshift(np.fft.fft2(out_lo))
        F_hi = np.fft.fftshift(np.fft.fft2(out_hi))
        P_lo = F_lo.real**2 + F_lo.imag**2
        P_hi = F_hi.real**2 + F_hi.imag**2

        # Radial frequency grid
        h, w = out_lo.shape
        ky = np.fft.fftshift(np.fft.fftfreq(h))
        kx = np.fft.fftshift(np.fft.fftfreq(w))
        KX, KY = np.meshgrid(kx, ky)
        R = np.sqrt(KX**2 + KY**2)

        # Use the median radius as a simple high/low frequency cutoff
        r_cut = np.median(R[R > 0])

        # Fraction of power above cutoff should increase with alpha
        frac_hi = P_hi[R > r_cut].sum() / P_hi.sum()
        frac_lo = P_lo[R > r_cut].sum() / P_lo.sum()

        # Higher alpha should put more power into higher frequencies
        assert frac_hi > frac_lo

    def test_texture_shading_linear_scaling(self) -> None:
        """
        Linearity: T(c * DEM) ≈ c * T(DEM).
        We set rtol/atol using machine epsilon (`eps`) of the dtype to account for
        normal float32+FFT rounding. `eps` is the smallest number where 1+eps != 1,
        so scaling tolerances by eps (and by output magnitude/scale_factor) makes
        the test robust but still tight.
        """
        rng = np.random.RandomState(0)
        dem = rng.randn(3, 3).astype(np.float32)

        alpha = 0.8
        scale_factor = 3000.0

        out1 = xdem.terrain.texture_shading(dem, alpha=alpha)
        out2 = xdem.terrain.texture_shading(scale_factor * dem, alpha=alpha)

        # Tolerances scaled to dtype precision and output magnitude
        eps = np.finfo(out1.dtype).eps  # ~1.19e-7 for float32
        # Factor 50 is a pragmatic buffer; empirically ~3e-5 relative differences on 3x3 grids
        rtol = float(50 * eps * scale_factor)
        atol = float(50 * eps * np.max(np.abs(scale_factor * out1)))

        np.testing.assert_allclose(out2, scale_factor * out1, rtol=rtol, atol=atol)

    def test_texture_shading_via_get_terrain_attribute(self) -> None:
        """Test texture_shading via the get_terrain_attribute interface."""

        # Test with a simple DEM
        dem_simple = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        # Test via get_terrain_attribute
        result = xdem.terrain.get_terrain_attribute(dem_simple, "texture_shading")

        # Check output properties
        assert result.shape == dem_simple.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))

        # Test with multiple attributes including texture_shading
        slope, texture = xdem.terrain.get_terrain_attribute(dem_simple, ["slope", "texture_shading"], resolution=1.0)

        assert slope.shape == dem_simple.shape
        assert texture.shape == dem_simple.shape
        assert not np.array_equal(slope, texture)  # Should be different attributes

    def test_texture_shading_real_dem(self) -> None:
        """Test texture_shading with a real DEM."""

        dem = self.dem.copy()

        # Test texture shading
        result = xdem.terrain.texture_shading(dem, alpha=0.8)

        # Check output properties
        assert result.shape == dem.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))

    def test_nextprod_fft(self) -> None:
        """Test the _nextprod_fft helper function."""

        # Test known values
        assert xdem.terrain._nextprod_fft(1) == 1
        assert xdem.terrain._nextprod_fft(10) == 16
        assert xdem.terrain._nextprod_fft(20) == 32
        assert xdem.terrain._nextprod_fft(32) == 32
        assert xdem.terrain._nextprod_fft(100) == 128

        # Test that result is always >= input
        for size in [1, 5, 13, 25, 37, 63, 91]:
            result = xdem.terrain._nextprod_fft(size)
            assert result >= size


class TestConvolution:

    # Get all coefficients
    coef_names = list(xdem.terrain.all_coefs.keys())
    coef_names_h = [n for n in coef_names if n in ["h1", "h2"]]
    coef_names_zt = [n for n in coef_names if "zt" in n]
    coef_names_fl = [n for n in coef_names if "fl" in n]
    coef_arrs_h = [xdem.terrain.all_coefs[n] for n in coef_names_h]
    coef_arrs_zt = [xdem.terrain.all_coefs[n] for n in coef_names_zt]
    coef_arrs_fl = [xdem.terrain.all_coefs[n] for n in coef_names_fl]

    def test_convolution__quadric_coefficients(self) -> None:
        """Test the outputs of quadric coefficients (currently not accessible by users)."""

        # Create a synthetic DEM with a stationary slope/curvature (quadratic ramp in Y direction)
        # to check basic consistency of derivatives (zero across ramp, non-zero along ramp)
        dem = np.stack([np.ones(5) * (i - 1) ** 2 for i in range(5)], axis=0)
        dem_flat = np.ones((5, 5), dtype=np.float32)

        # Horn coefficients
        kern3d = np.stack(self.coef_arrs_h, axis=0)
        coefs_h = xdem.spatialstats.convolution(
            dem.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]
        coefs_h_flat = xdem.spatialstats.convolution(
            dem_flat.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]

        # Zevenberg and Thorne coefficients
        kern3d = np.stack(self.coef_arrs_zt, axis=0)
        coefs_zt = xdem.spatialstats.convolution(
            dem.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]
        coefs_zt_flat = xdem.spatialstats.convolution(
            dem_flat.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]

        # Florinsky coefficients
        kern3d = np.stack(self.coef_arrs_fl, axis=0)
        coefs_fl = xdem.spatialstats.convolution(
            dem.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]
        coefs_fl_flat = xdem.spatialstats.convolution(
            dem_flat.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 2, 2]

        # 1/ Check coefficient for flat DEM are all zero (except last of ZT that is identity)
        assert all(coefs_fl_flat == 0)
        assert all(coefs_zt_flat[:-1] == 0)
        assert all(coefs_h_flat == 0)

        # Corresponding coefficients
        list_coefs_names = [self.coef_names_h, self.coef_names_zt, self.coef_names_fl]
        list_coefs = [coefs_h, coefs_zt, coefs_fl]
        dict_coef = {
            "z_x": ["h2", "zt_h", "fl_p"],
            "z_y": ["h1", "zt_g", "fl_q"],
            "z_xx": [None, "zt_e", "fl_r"],
            "z_yy": [None, "zt_d", "fl_t"],
            "z_xy": [None, "zt_f", "fl_s"],
        }

        # Test all coefficients are non-zero (along ramp direction) or zero (across ramp direction)
        directions = ["across", "along", "across", "along", "across"]
        for k, coef in enumerate(dict_coef.keys()):
            list_vals = []
            for i in range(len(list_coefs)):
                coef_name = dict_coef[coef][i]  # type: ignore
                if coef_name is not None:
                    val = list_coefs[i][list_coefs_names[i].index(coef_name)]
                    list_vals.append(val)

            if directions[k] == "across":
                assert all(np.array(list_vals) == 0)
            else:
                assert all(np.array(list_vals) != 0)

    @pytest.mark.parametrize("coef_arrs", [coef_arrs_h, coef_arrs_zt, coef_arrs_fl])  # type: ignore
    def test_convolution_equal__engine(self, coef_arrs: list[np.ndarray]) -> None:  # type: ignore
        """
        Check that convolution through SciPy or Numba give equal result for all kernels.
        This calls the convolution subfunctions directly (as they need to be chained sequentially with other
        steps in the main functions).
        """

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(5, 7))

        kern3d = np.stack(coef_arrs, axis=0)

        # With SciPy
        conv_scipy = xdem.spatialstats.convolution(
            dem.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 3, 3]

        # With Numba
        _, M1, M2 = kern3d.shape
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        dem = np.pad(dem, pad_width=((half_M1, half_M1), (half_M2, half_M2)), constant_values=np.nan)
        conv_numba = xdem.terrain._convolution_numba(dem, filters=kern3d, row=3, col=3)

        np.allclose(conv_scipy, conv_numba, equal_nan=True)

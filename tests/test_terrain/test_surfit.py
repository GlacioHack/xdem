from __future__ import annotations

import re
import warnings
from typing import Literal

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

import xdem
from xdem.terrain.surfit import all_coefs

PLOT = False


class TestTerrainAttribute:
    filepath = xdem.examples.get_path("longyearbyen_ref_dem")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Parse metadata")
        dem = xdem.DEM(filepath, silent=True)

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
        # For max/min, no particular relation between the geometric and direction definitions
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

        # 4/ Saddle-shaped DEM (hyperbolic paraboloid): Check min/max
        n = 5
        x = np.linspace(-1, 1, n)
        y = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, y)
        dem = X**2 - Y**2

        # Min and max should be exactly opposite
        fl_curv_max = xdem.terrain.max_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_min = xdem.terrain.min_curvature(dem, resolution=5, surface_fit="Florinsky")[2, 2]

        assert pytest.approx(fl_curv_max) == -fl_curv_min

        # 5/ Linear slope with Gaussian ridge/trough: Check planform/tangential
        a = 0.6
        b = 1.0
        sigma = 1
        # ridge
        dem_ridge = a * X + b * np.exp(-((Y / sigma) ** 2))
        # trough
        dem_trough = a * X - b * np.exp(-((Y / sigma) ** 2))

        # Planform/tangential should be exactly opposite, respectively
        fl_curv_pla_ridge = xdem.terrain.planform_curvature(dem_ridge, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_tan_ridge = xdem.terrain.tangential_curvature(dem_ridge, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_pla_trough = xdem.terrain.planform_curvature(dem_trough, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_tan_trough = xdem.terrain.tangential_curvature(dem_trough, resolution=5, surface_fit="Florinsky")[2, 2]
        assert pytest.approx(fl_curv_tan_ridge) == -fl_curv_tan_trough
        assert pytest.approx(fl_curv_pla_ridge) == -fl_curv_pla_trough

        # Flowline and profile should be zero
        fl_curv_prof_ridge = xdem.terrain.profile_curvature(dem_ridge, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_flo_ridge = xdem.terrain.flowline_curvature(dem_ridge, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_prof_trough = xdem.terrain.profile_curvature(dem_trough, resolution=5, surface_fit="Florinsky")[2, 2]
        fl_curv_flo_trough = xdem.terrain.flowline_curvature(dem_trough, resolution=5, surface_fit="Florinsky")[2, 2]
        assert pytest.approx(fl_curv_prof_ridge) == 0
        assert pytest.approx(fl_curv_flo_ridge) == 0
        assert pytest.approx(fl_curv_prof_trough) == 0
        assert pytest.approx(fl_curv_flo_trough) == 0

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

        attrs_scipy = xdem.terrain.surfit._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], surface_fit=surface_fit, engine="scipy"
        )
        attrs_numba = xdem.terrain.surfit._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], surface_fit=surface_fit, engine="numba"
        )

        assert np.allclose(attrs_scipy, attrs_numba, equal_nan=True)

    @pytest.mark.parametrize(
        "attribute",
        [
            attr
            for attr in xdem.terrain.available_attributes
            if attr in ["aspect", "slope", "hillshade"] or "curvature" in attr
        ],
    )  # type: ignore
    @pytest.mark.parametrize("surface_fit", ["Horn", "ZevenbergThorne", "Florinsky"])  # type: ignore
    def test_surface_fit_attribute__nan_propag(
        self, attribute: str, surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"]
    ) -> None:
        """
        Check that NaN propagation behaves as intended for surface fit attributes, in short: NaN are propagated
        from the edges and from NaNs based on window size associated with the surface fit (3x3 or 5x5).
        """

        if surface_fit == "Horn" and attribute not in ["aspect", "slope", "hillshade"]:
            return

        # Generate DEM
        rng = np.random.default_rng(42)
        dem = rng.normal(size=(20, 20))
        # Introduce NaNs
        dem[4, 4:6] = np.nan
        dem[17, 16] = np.nan
        mask_nan_dem = ~np.isfinite(dem)

        # Generate attribute
        attr = xdem.terrain.get_terrain_attribute(dem, resolution=1, attribute=attribute, surface_fit=surface_fit)
        mask_nan_attr = ~np.isfinite(attr)

        # We dilate the initial mask by a structuring element matching the window size of the surface fit
        if surface_fit == "Florinsky":
            struct = np.ones((5, 5), dtype=bool)
            hw = 2
        else:
            struct = np.ones((3, 3), dtype=bool)
            hw = 1
        eroded_mask_dem = binary_dilation(mask_nan_dem.astype(int), structure=struct, iterations=1)
        # On edges, NaN should be expanded by the half-width rounded down of the window
        eroded_mask_dem[:hw, :] = True
        eroded_mask_dem[-hw:, :] = True
        eroded_mask_dem[:, :hw] = True
        eroded_mask_dem[:, -hw:] = True
        # We check the two masks are indeed the same
        assert np.array_equal(eroded_mask_dem, mask_nan_attr)


class TestConvolution:

    # Get all coefficients
    coef_names = list(all_coefs.keys())
    coef_names_h = [n for n in coef_names if n in ["h1", "h2"]]
    coef_names_zt = [n for n in coef_names if "zt" in n]
    coef_names_fl = [n for n in coef_names if "fl" in n]
    coef_arrs_h = [all_coefs[n] for n in coef_names_h]
    coef_arrs_zt = [all_coefs[n] for n in coef_names_zt]
    coef_arrs_fl = [all_coefs[n] for n in coef_names_fl]

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
        conv_numba = xdem.terrain.surfit._convolution_numba(dem, filters=kern3d, row=3, col=3)

        np.allclose(conv_scipy, conv_numba, equal_nan=True)

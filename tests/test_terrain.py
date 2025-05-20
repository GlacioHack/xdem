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
            "slope_Horn": lambda dem: xdem.terrain.slope(dem.data, resolution=dem.res, degrees=True),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res),
            "slope_Zevenberg": lambda dem: xdem.terrain.slope(
                dem.data, resolution=dem.res, method="ZevenbergThorne", degrees=True
            ),
            "aspect_Zevenberg": lambda dem: xdem.terrain.aspect(dem.data, method="ZevenbergThorne", degrees=True),
            "hillshade_Zevenberg": lambda dem: xdem.terrain.hillshade(
                dem.data, resolution=dem.res, method="ZevenbergThorne"
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
        ["slope_Horn", "aspect_Horn", "hillshade_Horn", "curvature", "profile_curvature", "planform_curvature"],
    )  # type: ignore
    def test_attribute_functions_against_richdem(self, attribute: str, get_test_data_path) -> None:
        """
        Test that all attribute functions give the same results as those of RichDEM within a small tolerance.

        :param attribute: The attribute to test (e.g. 'slope')
        """

        # Functions for xdem-implemented methods
        functions_xdem = {
            "slope_Horn": lambda dem: xdem.terrain.slope(dem, resolution=dem.res, degrees=True),
            "aspect_Horn": lambda dem: xdem.terrain.aspect(dem.data, degrees=True),
            "hillshade_Horn": lambda dem: xdem.terrain.hillshade(dem.data, resolution=dem.res),
            "curvature": lambda dem: xdem.terrain.curvature(dem.data, resolution=dem.res),
            "profile_curvature": lambda dem: xdem.terrain.profile_curvature(dem.data, resolution=dem.res),
            "planform_curvature": lambda dem: xdem.terrain.planform_curvature(dem.data, resolution=dem.res),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both RichDEM and xdem
        attr_xdem = gu.raster.get_array_and_mask(functions_xdem[attribute](dem))[0].squeeze()
        attr_richdem_rst = gu.Raster(get_test_data_path(os.path.join("richdem", f"{attribute}.tif")), load_data=True)
        attr_richdem = gu.raster.get_array_and_mask(attr_richdem_rst)[0].squeeze()

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
                assert np.all(np.minimum(diff_valid, np.abs(360 - diff_valid)) < 10 ** (-3) * magn)

            else:
                # All attributes other than aspect are non-circular floats, so we check within a tolerance
                # Here hillshade is not rounded as integer by our calculation, so no need to differentiate as with GDAL
                assert np.all(np.abs(diff_valid < 10 ** (-3) * magn))

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_richdem)
                plt.colorbar()
                plt.subplot(222)
                plt.imshow(attr_xdem)
                plt.colorbar()
                plt.subplot(223)
                plt.imshow(diff)
                plt.colorbar()
                plt.show()

            raise exception

        # Introduce some nans
        # rng = np.random.default_rng(42)
        # dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        # dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans and that mask is preserved
        # output = functions_richdem[attribute](dem)
        # assert np.all(dem.data.mask == output.data.mask)

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
        "name", ["curvature", "planform_curvature", "profile_curvature", "maximum_curvature"]
    )  # type: ignore
    def test_curvatures(self, name: str) -> None:
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
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Slope method 'DoesNotExist' is not supported. Must be one of: " "['Horn', 'ZevenbergThorne']"
            ),
        ):
            xdem.terrain.slope(self.dem.data, resolution=self.dem.res, method="DoesNotExist")  # type: ignore

        with pytest.raises(
            ValueError,
            match=re.escape("TRI method 'DoesNotExist' is not supported. Must be one of: " "['Riley', 'Wilson']"),
        ):
            xdem.terrain.terrain_ruggedness_index(self.dem.data, method="DoesNotExist")  # type: ignore

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

    def test_convolution__quadric_coefficients(self) -> None:
        """Test the outputs of quadric coefficients (not accessible by users)."""

        dem = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        # Get all coefficients and convolve middle mixel
        coef_arrs = list(xdem.terrain.all_coefs.values())
        coef_names = list(xdem.terrain.all_coefs.keys())
        kern3d = np.stack(coef_arrs, axis=0)
        coefs = xdem.spatialstats.convolution(
            dem.reshape((1, dem.shape[0], dem.shape[1])), filters=kern3d, method="scipy"
        ).squeeze()[:, 1, 1]

        # The 4th to last coefficient is identity, so the dem itself
        assert np.array_equal(coefs[coef_names.index("zt_i")], dem[1, 1])

        # The third should be concave in the x-direction
        assert coefs[coef_names.index("zt_d")] < 0

        # The fourth should be concave in the y-direction
        assert coefs[coef_names.index("zt_e")] < 0

    def test_convolution_equal__engine(self) -> None:
        """
        Check that convolution through SciPy or Numba give equal result for all kernels.
        This calls the convolution subfunctions directly (as they need to be chained sequentially with other
        steps in the main functions).
        """

        # Stack to convolve all coefs at once
        coef_arrs = list(xdem.terrain.all_coefs.values())
        kern3d = np.stack(coef_arrs, axis=0)

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(5, 7))

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

    @pytest.mark.parametrize(
        "attribute",
        ["slope", "aspect", "hillshade", "curvature", "profile_curvature", "planform_curvature", "maximum_curvature"],
    )  # type: ignore
    @pytest.mark.parametrize("slope_method", ["Horn", "ZevenbergThorne"])  # type: ignore
    def test_get_surface_attributes__engine(
        self, attribute: str, slope_method: Literal["Horn", "ZevenbergThorne"]
    ) -> None:
        """Check that all quadric coefficients from the convolution give the same results as with the numba loop."""

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(5, 7))

        attrs_scipy = xdem.terrain._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], slope_method=slope_method, engine="scipy"
        )
        attrs_numba = xdem.terrain._get_surface_attributes(
            dem=dem, resolution=2, surface_attributes=[attribute], slope_method=slope_method, engine="numba"
        )

        assert np.allclose(attrs_scipy, attrs_numba, equal_nan=True)
        # assert np.allclose(coefs_numba, coefs_numba_cv, equal_nan=True)

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

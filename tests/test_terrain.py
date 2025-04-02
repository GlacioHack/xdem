from __future__ import annotations

import os.path
import re
import warnings

import geoutils as gu
import numpy as np
import pytest
from geoutils.raster.distributed_computing import MultiprocConfig

import xdem

xdem.examples.download_longyearbyen_examples()

PLOT = True


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

    def test_hillshade_errors(self) -> None:
        """Validate that the hillshade function raises appropriate errors."""
        # Try giving the hillshade invalid arguments.

        with pytest.raises(ValueError, match="Azimuth must be a value between 0 and 360"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, azimuth=361)

        with pytest.raises(ValueError, match="Altitude must be a value between 0 and 90"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, altitude=91)

        with pytest.raises(ValueError, match="z_factor must be a non-negative finite value"):
            xdem.terrain.hillshade(self.dem.data, resolution=self.dem.res, z_factor=np.inf)

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

    @pytest.mark.parametrize(
        "name", ["curvature", "planform_curvature", "profile_curvature", "maximum_curvature"]
    )  # type: ignore
    def test_curvatures(self, name: str) -> None:
        """Test the curvature functions"""

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive curvature without any gaps
        curvature = xdem.terrain.get_terrain_attribute(
            dem.data, attribute=name, resolution=dem.res, edge_method="nearest"
        )

        # Validate that the array has the same shape as the input and that all values are finite.
        assert curvature.shape == dem.data.shape
        try:
            assert np.all(np.isfinite(curvature))
        except Exception:
            import matplotlib.pyplot as plt

            plt.imshow(curvature.squeeze())
            plt.show()

        with pytest.raises(ValueError, match="Quadric surface fit requires the same X and Y resolution."):
            xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=(1.0, 2.0))

        # Introduce some nans
        rng = np.random.default_rng(42)
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True
        # Validate that this doesn't raise weird warnings after introducing nans.
        xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)

    def test_get_terrain_attribute(self) -> None:
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

    def test_get_terrain_attribute_multiproc(self) -> None:
        """Test the get_terrain attribute function in multiprocessing."""
        outfile = "mp_output.tif"
        outfile_multi = ["mp_output_slope.tif", "mp_output_aspect.tif", "mp_output_hillshade.tif"]

        mp_config = MultiprocConfig(
            chunk_size=200,
            outfile=outfile,
        )

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        xdem.terrain.get_terrain_attribute(self.dem, "slope", multiproc_config=mp_config, resolution=self.dem.res)
        assert os.path.exists(outfile)
        slope = gu.Raster(outfile, load_data=True)
        assert isinstance(slope, gu.Raster)
        os.remove(outfile)

        # Create three products at the same time
        xdem.terrain.get_terrain_attribute(
            self.dem, ["slope", "aspect", "hillshade"], multiproc_config=mp_config, resolution=self.dem.res
        )
        for file in outfile_multi:
            assert os.path.exists(file)
        slope2 = gu.Raster(outfile_multi[0], load_data=True)
        hillshade = gu.Raster(outfile_multi[2], load_data=True)
        for file in outfile_multi:
            os.remove(file)

        # Create a hillshade using its own function
        xdem.terrain.hillshade(self.dem, multiproc_config=mp_config, resolution=self.dem.res)
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
        xdem.terrain.get_terrain_attribute(
            self.dem, "slope", multiproc_config=mp_config, resolution=self.dem.res[0] * 2
        )
        slope_lowres = gu.Raster(outfile, load_data=True)
        os.remove(outfile)
        assert slope.get_stats("mean") > slope_lowres.get_stats("mean")

    def test_get_terrain_attribute_errors(self) -> None:
        """Test the get_terrain_attribute function raises appropriate errors."""

        # Below, re.escape() is needed to match expressions that have special characters (e.g., parenthesis, bracket)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Slope method 'DoesNotExist' is not supported. Must be one of: " "['Horn', 'ZevenbergThorne']"
            ),
        ):
            xdem.terrain.slope(self.dem.data, method="DoesNotExist")

        with pytest.raises(
            ValueError,
            match=re.escape("TRI method 'DoesNotExist' is not supported. Must be one of: " "['Riley', 'Wilson']"),
        ):
            xdem.terrain.terrain_ruggedness_index(self.dem.data, method="DoesNotExist")

    def test_raster_argument(self) -> None:

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
    @pytest.mark.parametrize("dh", np.linspace(0.01, 100, 10))  # type: ignore
    # Loop for different resolutions
    @pytest.mark.parametrize("resolution", np.linspace(0.01, 100, 10))  # type: ignore
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

    def test_get_quadric_coefficients(self) -> None:
        """Test the outputs and exceptions of the get_quadric_coefficients() function."""

        dem = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        coefficients = xdem.terrain.get_quadric_coefficients(
            dem, resolution=1.0, edge_method="nearest", make_rugosity=True
        )

        # Check all coefficients are finite with an edge method
        assert np.all(np.isfinite(coefficients))

        # The 4th to last coefficient is the dem itself (could maybe be removed in the future as it is duplication..)
        assert np.array_equal(coefficients[-4, :, :], dem)

        # The middle pixel (index 1, 1) should be concave in the x-direction
        assert coefficients[3, 1, 1] < 0

        # The middle pixel (index 1, 1) should be concave in the y-direction
        assert coefficients[4, 1, 1] < 0

        with pytest.raises(ValueError, match="Invalid input array shape"):
            xdem.terrain.get_quadric_coefficients(dem.reshape((1, 1, -1)), 1.0)

        # Validate that when using the edge_method="none", only the one non-edge value is kept.
        coefs = xdem.terrain.get_quadric_coefficients(dem, resolution=1.0, edge_method="none")
        assert np.count_nonzero(np.isfinite(coefs[0, :, :])) == 1
        # When using edge wrapping, all coefficients should be finite.
        coefs = xdem.terrain.get_quadric_coefficients(dem, resolution=1.0, edge_method="wrap")
        assert np.count_nonzero(np.isfinite(coefs[0, :, :])) == 9

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

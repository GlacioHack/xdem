from __future__ import annotations

import os.path
import re
import warnings
from typing import Any, Callable

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils.raster.distributed_computing import MultiprocConfig
from pyproj import CRS

import xdem

PLOT = False


class TestTerrainAttribute:
    filepath = xdem.examples.get_path_test("longyearbyen_ref_dem")
    dem = xdem.DEM(filepath)

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
    )
    def test_attribute_functions_against_gdaldem(
        self, attribute: str, get_test_data_path: Callable[[str], str]
    ) -> None:
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
                assert np.all(np.logical_or(np.allclose(diff_valid, 0), np.allclose(np.abs(diff_valid), 1.0)))

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
        dem.data.mask.ravel()[rng.choice(dem.data.size, 25, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans.
        functions[attribute](dem)

    @pytest.mark.parametrize(
        "attribute",
        ["slope_Horn", "aspect_Horn", "hillshade_Horn", "profile_curvature", "planform_curvature"],
    )
    def test_attribute_functions_against_richdem(
        self, attribute: str, get_test_data_path: Callable[[str], str]
    ) -> None:
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

        slope = gu.raster.get_array_and_mask(functions_xdem["slope_Horn"](dem))[0].squeeze()
        # Derive the attribute using both RichDEM and xdem
        attr_xdem = gu.raster.get_array_and_mask(functions_xdem[attribute](dem))[0].squeeze()
        attr_richdem_rst = gu.Raster(get_test_data_path(os.path.join("richdem", f"{attribute}.tif")), load_data=True)
        attr_richdem = gu.raster.get_array_and_mask(attr_richdem_rst)[0].squeeze()

        # RichDEM has the opposite sign for profile curvature compared to Minar et al. (2020)
        if attribute == "profile_curvature":
            attr_richdem = -attr_richdem

        # Remove nearly flat terrain where aspect is extremely sensitive to numerical errors
        if attribute == "aspect_Horn":
            attr_xdem[slope < 3] = np.nan

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
                # We use a 99% percentile to remove potential outliers/edge effects
                assert np.nanpercentile(np.abs(diff_valid), 99) < 10 ** (-3) * magn

        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                # Plotting the xdem and RichDEM attributes for comparison (plotting "diff" can also help debug)
                plt.subplot(221)
                plt.imshow(attr_richdem)
                plt.colorbar(label="richdem")
                plt.subplot(222)
                plt.imshow(attr_xdem)
                plt.colorbar(label="xdem")
                plt.subplot(223)
                plt.imshow(diff)
                plt.colorbar(label="diff")
                plt.subplot(224)
                plt.imshow(dem.data)
                plt.colorbar(label="dem")
                plt.show()

            raise exception

        # Introduce some nans
        # rng = np.random.default_rng(42)
        # dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        # dem.data.mask.ravel()[rng.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans and that mask is preserved
        # output = functions_richdem[attribute](dem)
        # assert np.all(dem.data.mask == output.data.mask)

    @pytest.mark.parametrize("attribute", xdem.terrain.available_attributes)
    def test_attributes_default_call(self, attribute: str) -> None:
        from_str_to_fun = {
            "slope": lambda: self.dem.slope(),
            "aspect": lambda: self.dem.aspect(),
            "hillshade": lambda: self.dem.hillshade(),
            "profile_curvature": lambda: self.dem.profile_curvature(),
            "tangential_curvature": lambda: self.dem.tangential_curvature(),
            "planform_curvature": lambda: self.dem.planform_curvature(),
            "flowline_curvature": lambda: self.dem.flowline_curvature(),
            "max_curvature": lambda: self.dem.max_curvature(),
            "min_curvature": lambda: self.dem.min_curvature(),
            "topographic_position_index": lambda: self.dem.topographic_position_index(),
            "terrain_ruggedness_index": lambda: self.dem.terrain_ruggedness_index(),
            "roughness": lambda: self.dem.roughness(),
            "rugosity": lambda: self.dem.rugosity(),
            "texture_shading": lambda: self.dem.texture_shading(),
            "fractal_roughness": lambda: self.dem.fractal_roughness(),
        }

        res_gta = xdem.terrain.get_terrain_attribute(self.dem, attribute=attribute)
        res_fun = from_str_to_fun[attribute]()
        assert res_gta == res_fun

    @pytest.mark.parametrize("surfit_windowsizes", [("Florinsky", 3, 5), ("ZevenbergThorne", 7, 13)])
    def test_get_terrain_attribute__multiple_inputs(self, surfit_windowsizes: tuple[str, int, int]) -> None:
        """Test the get_terrain_attribute function by itself."""

        # Fractal roughness with tested window sizes of less than 13 will expectedly raise a warning
        warnings.filterwarnings("ignore", category=UserWarning, message="Fractal roughness results.*")

        # Unpack argument of surface fit/window size
        surface_fit, window_size, window_size_fractal = surfit_windowsizes

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        slope_u = xdem.terrain.get_terrain_attribute(
            self.dem.data, "slope", resolution=self.dem.res, window_size=window_size, surface_fit=surface_fit
        )  # type: ignore
        assert isinstance(slope_u, np.ndarray)

        # Create four products at the same time
        # slope/hillshade in list_requiring_surface_fit, roughness in list_requiring_windowed_index
        # and fractal_roughness in list_requiring_windowed_fractal_index
        slope_m, roughness_m, hillshade_m, fractal_roughness_m = xdem.terrain.get_terrain_attribute(
            self.dem.data,
            ["slope", "roughness", "hillshade", "fractal_roughness"],
            resolution=self.dem.res,
            window_size=window_size,
            window_size_fractal=window_size_fractal,
            surface_fit=surface_fit,
        )  # type: ignore

        # Create attributes using its own function
        hillshade_u = xdem.terrain.hillshade(
            self.dem.data, resolution=self.dem.res, surface_fit=surface_fit
        )  # type: ignore
        fractal_roughness_u = xdem.terrain.fractal_roughness(self.dem.data, window_size_fractal=window_size_fractal)
        roughness_u = xdem.terrain.roughness(self.dem.data, window_size=window_size)

        # Validate that the "batch-created" attributes are the same as the "single-created"
        assert np.array_equal(hillshade_u, hillshade_m, equal_nan=True)
        assert np.array_equal(slope_u, slope_m, equal_nan=True)
        assert np.array_equal(fractal_roughness_u, fractal_roughness_m, equal_nan=True)
        assert np.array_equal(roughness_u, roughness_m, equal_nan=True)

        # A slope map with a lower resolution (higher value) should have gentler slopes.
        slope_lowres = xdem.terrain.get_terrain_attribute(
            self.dem.data, "slope", resolution=self.dem.res[0] * 2, window_size=window_size
        )
        assert np.nanmean(slope_u) > np.nanmean(slope_lowres)

    @pytest.mark.parametrize("surfit_windowsizes", [("Florinsky", 3, 5), ("ZevenbergThorne", 7, 13)])
    @pytest.mark.parametrize("attribute", xdem.terrain.available_attributes)
    def test_attributes__multiproc(self, attribute: str, surfit_windowsizes: tuple[str, int, int]) -> None:
        """
        Test that terrain attributes are exactly equal in multiprocessing or in normal processing, and for varying
        window sizes/surface fit methods, to verify that the depth (overlap) of the map_overlap is properly defined."""

        # Fractal roughness with tested window sizes of less than 13 will expectedly raise a warning
        warnings.filterwarnings("ignore", category=UserWarning, message="Fractal roughness results.*")

        # Attributes based on frequency will not match exactly
        if attribute == "texture_shading":
            return

        # Define multiproc config
        outfile = "tmp_mp_output.tif"
        mp_config = MultiprocConfig(
            chunk_size=50,
            outfile=outfile,
        )

        # Unpack argument of surface fit/window size
        surface_fit, window_size, window_size_fractal = surfit_windowsizes
        kwargs: dict[str, Any]

        if attribute in xdem.terrain.list_requiring_surface_fit:
            kwargs = {"surface_fit": surface_fit}
        elif attribute in xdem.terrain.list_requiring_windowed_index and attribute != "rugosity":
            kwargs = {"window_size": window_size}
        elif attribute in xdem.terrain.list_requiring_windowed_fractal_index:
            kwargs = {"window_size_fractal": window_size_fractal}

        # Rugosity is an exception: window size is not variable
        else:
            kwargs = {}

        # Derive with "DEM.attribute()" function, with and without multiproc
        attr_mp = getattr(self.dem, attribute)(mp_config=mp_config, **kwargs)
        attr_nomp = getattr(self.dem, attribute)(**kwargs)

        # Check equality
        assert attr_mp.georeferenced_grid_equal(attr_nomp)
        assert np.allclose(attr_mp.data.filled(), attr_nomp.data.filled())
        assert np.array_equal(attr_mp.data.mask, attr_nomp.data.mask)

        # Clean up outfile
        os.remove(outfile)

    @pytest.mark.parametrize("surfit_windowsizes", [("Florinsky", 3, 5), ("ZevenbergThorne", 7, 13)])
    def test_get_terrain_attribute__multiproc_inputs(self, surfit_windowsizes: tuple[str, int, int]) -> None:
        """Test the get_terrain attribute function in multiprocessing returns the right input number/type."""

        # Fractal roughness with tested window sizes of less than 13 will expectedly raise a warning
        warnings.filterwarnings("ignore", category=UserWarning, message="Fractal roughness results.*")

        outfile = "mp_output.tif"
        outfile_multi = [
            "mp_output_slope.tif",
            "mp_output_roughness.tif",
            "mp_output_hillshade.tif",
            "mp_output_fractal_roughness.tif",
        ]

        mp_config = MultiprocConfig(
            chunk_size=200,
            outfile=outfile,
        )

        # Unpack argument of surface fit/window size
        surface_fit, window_size, window_size_fractal = surfit_windowsizes

        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        xdem.terrain.get_terrain_attribute(
            self.dem, "slope", mp_config=mp_config, resolution=self.dem.res, surface_fit=surface_fit
        )  # type: ignore
        assert os.path.exists(outfile)
        slope_u = gu.Raster(outfile, load_data=True)
        assert isinstance(slope_u, gu.Raster)
        os.remove(outfile)

        # Create four products at the same time
        xdem.terrain.get_terrain_attribute(
            self.dem,
            ["slope", "roughness", "hillshade", "fractal_roughness"],
            mp_config=mp_config,
            resolution=self.dem.res,
            window_size=window_size,
            window_size_fractal=window_size_fractal,
            surface_fit=surface_fit,
        )  # type: ignore
        for file in outfile_multi:
            assert os.path.exists(file)
        slope_m = gu.Raster(outfile_multi[0], load_data=True)
        roughness_m = gu.Raster(outfile_multi[1], load_data=True)
        hillshade_m = gu.Raster(outfile_multi[2], load_data=True)
        fractal_roughness_m = gu.Raster(outfile_multi[3], load_data=True)
        for file in outfile_multi:
            os.remove(file)

        # Create a hillshade using its own function
        xdem.terrain.hillshade(self.dem, mp_config=mp_config, surface_fit=surface_fit)  # type: ignore
        assert os.path.exists(outfile)
        hillshade_u = gu.Raster(outfile, load_data=True)
        os.remove(outfile)

        # Create a roughness using its own function
        xdem.terrain.roughness(self.dem, mp_config=mp_config, window_size=window_size)
        assert os.path.exists(outfile)
        roughness_u = gu.Raster(outfile, load_data=True)
        os.remove(outfile)

        # Create a fractal roughness using its own function
        xdem.terrain.fractal_roughness(self.dem, mp_config=mp_config, window_size_fractal=window_size_fractal)
        assert os.path.exists(outfile)
        fractal_roughness_u = gu.Raster(outfile, load_data=True)
        os.remove(outfile)

        # Validate that the "batch-created" attributes are the same as the "single-created"
        assert hillshade_u.raster_equal(hillshade_m)
        assert slope_u.raster_equal(slope_m)
        assert roughness_u.raster_equal(roughness_m)
        assert fractal_roughness_u.raster_equal(fractal_roughness_m)

        # Compare with classic terrain attribute calculation
        slope_classic = self.dem.slope(surface_fit=surface_fit)
        hillshade_classic = self.dem.hillshade(surface_fit=surface_fit)
        roughness_classic = self.dem.roughness(window_size=window_size)
        fractal_roughness_classic = self.dem.fractal_roughness(window_size_fractal=window_size_fractal)
        assert np.allclose(slope_u.data, slope_classic.data, rtol=1e-7)
        assert np.allclose(hillshade_u.data, hillshade_classic.data, rtol=1e-7)
        assert np.allclose(roughness_u.data, roughness_classic.data, rtol=1e-7)
        assert np.allclose(fractal_roughness_u.data, fractal_roughness_classic.data, rtol=1e-7)

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
            xdem.terrain.slope(self.dem, surface_fit="DoesNotExist")  # type: ignore

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

        # Check warnings if window_size_fractal < 13
        with pytest.raises(
            UserWarning,
            match=re.escape("Fractal roughness can only be computed on window sizes larger or equal to 5."),
        ):
            xdem.terrain.fractal_roughness(self.dem, window_size_fractal=3)  # type: ignore

        with pytest.raises(
            UserWarning,
            match=re.escape("Fractal roughness results with window size of less than 13 can be inaccurate."),
        ):
            xdem.terrain.fractal_roughness(self.dem, window_size_fractal=10)  # type: ignore

    def test_get_terrain_attribute__raster_input(self) -> None:
        """Test the get_terrain_attribute function supports raster input/output."""

        slope, aspect = xdem.terrain.get_terrain_attribute(self.dem, attribute=["slope", "aspect"])

        assert slope != aspect

        assert isinstance(slope, type(aspect))
        assert all(isinstance(r, gu.Raster) for r in (aspect, slope, self.dem))

        assert slope.transform == self.dem.transform == aspect.transform
        assert slope.crs == self.dem.crs == aspect.crs

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


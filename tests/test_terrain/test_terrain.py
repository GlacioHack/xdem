from __future__ import annotations

import os.path
import re
import warnings

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils.raster.chunked import MultiprocConfig
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

        # RichDEM has the opposite sign for profile curvature compared to Minar et al. (2020)
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
                # We use a 99% percentile to remove potential outliers/edge effects
                assert np.nanpercentile(np.abs(diff_valid), 99) < 10 ** (-3) * magn

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

    @pytest.mark.parametrize("surfit_windowsize", [("Florinsky", 3), ("ZevenbergThorne", 7)])  # type: ignore
    @pytest.mark.parametrize("attribute", xdem.terrain.available_attributes)  # type: ignore
    def test_attributes__multiproc(self, attribute, surfit_windowsize) -> None:
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
        surface_fit, window_size = surfit_windowsize
        if attribute in xdem.terrain.list_requiring_surface_fit:
            kwargs = {"surface_fit": surface_fit}
        elif attribute in xdem.terrain.list_requiring_windowed_index and attribute != "rugosity":
            kwargs = {"window_size": window_size}
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

    def test_get_terrain_attribute__multiproc_inputs(self) -> None:
        """Test the get_terrain attribute function in multiprocessing returns the right input number/type."""
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

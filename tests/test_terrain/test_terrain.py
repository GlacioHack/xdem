from __future__ import annotations

import os.path
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from affine import Affine
from geoutils.multiproc import MultiprocConfig
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

        # Define multiproc config
        outfile = "tmp_mp_output.tif"
        mp_config = MultiprocConfig(
            chunks=50,
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
        np.testing.assert_array_equal(attr_mp.get_nanarray(), attr_nomp.get_nanarray())
        np.testing.assert_array_equal(np.ma.getmaskarray(attr_mp.data), np.ma.getmaskarray(attr_nomp.data))

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
            chunks=200,
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


class TestTerrainAttributeChunked:
    """
    Test terrain attributes across eager, Dask and multiprocessing backends.

    This class tests:
    - ``get_terrain_attribute`` for exact backend equality with uneven and empty chunks,
    - Window overlap when calculation windows are larger than chunks,
    - Requested attribute order, dimensions, chunk sizes and output dtypes,
    - Errors for incompatible Dask and multiprocessing inputs.
    """

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    @pytest.mark.parametrize("chunks", [(4, 5), (11, 13)])
    @pytest.mark.parametrize(
        "surface_fit, window_size, window_size_fractal", [("Florinsky", 7, 13), ("ZevenbergThorne", 9, 15)]
    )
    def test_get_terrain_attribute__overlap_and_order(
        self,
        tmp_path: Path,
        engine: Literal["scipy", "numba"],
        chunks: tuple[int, int],
        surface_fit: Literal["Florinsky", "ZevenbergThorne"],
        window_size: int,
        window_size_fractal: int,
    ) -> None:
        """
        Checks that every terrain family matches eager results at chunk edges, including windows larger than tiles.
        """

        da = pytest.importorskip("dask.array")
        if engine == "numba":
            pytest.importorskip("numba")
        from dask.callbacks import Callback

        # 1/ Create a surface with missing data, then open it with Dask and multiprocessing
        # The wide missing area makes some chunks fully empty; the corner gap checks calculations at the raster edge
        rows, cols = np.indices((31, 37), dtype=np.float32)
        values = 900 + rows**2 / 8 + 2 * cols + 3 * np.sin(cols / 3)
        values[6:23, 8:27] = np.nan
        values[:2, :3] = np.nan
        reference = gu.Raster.from_array(values, Affine(20, 0, 500000, 0, -20, 8600000), 32633, nodata=-9999)
        path = tmp_path / "terrain.tif"
        reference.to_file(path)
        source = gu.open_raster(path, chunks={"y": chunks[0], "x": chunks[1]})
        source_array = source.data
        native = gu.Raster(path)
        config = MultiprocConfig(chunks=(chunks[1], chunks[0]), outfile=str(tmp_path / "attributes.tif"))
        original_chunks, original_outfile = config.chunks, config.outfile

        # Request every attribute in mixed order, with repeats, to check that outputs follow the requested order
        attributes = ["texture_shading", "roughness"] + xdem.terrain.available_attributes + ["curvature", "slope"]
        options: dict[str, Any] = {
            "surface_fit": surface_fit,
            "window_size": window_size,
            "window_size_fractal": window_size_fractal,
            "engine": engine,
            "texture_alpha": 1.2,
            "degrees": False,
            "tri_method": "Wilson",
        }

        # 2/ Create Dask, multiprocessing and in-memory results without loading either file-backed source
        # The callback records every executed Dask task, so the list must remain empty until compute is called
        tasks: list[Any] = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            lazy_outputs = xdem.terrain.get_terrain_attribute(source, attributes, **options)
        assert tasks == []
        parallel_outputs = xdem.terrain.get_terrain_attribute(native, attributes, mp_config=config, **options)
        expected_outputs = xdem.terrain.get_terrain_attribute(reference, attributes, **options)
        assert not source._in_memory and not native.is_loaded
        assert (config.chunks, config.outfile) == (original_chunks, original_outfile)

        # 3/ Compare every pixel, including chunk boundaries, missing areas and raster edges
        assert len(lazy_outputs) == len(parallel_outputs) == len(expected_outputs) == len(attributes)
        for name, lazy, parallel, expected in zip(attributes, lazy_outputs, parallel_outputs, expected_outputs):
            assert isinstance(lazy, xr.DataArray) and isinstance(lazy.data, da.Array)
            assert type(parallel) is gu.Raster and not parallel.is_loaded
            assert lazy.dtype == parallel.dtype == expected.dtype
            assert expected.raster_equal(lazy.compute(), strict_masked=False, warn_failure_reason=True)
            assert expected.raster_equal(parallel, strict_masked=False, warn_failure_reason=True)
            # Texture shading uses the full image at once, but its Dask result must still use the source chunk sizes
            if name == "texture_shading":
                assert lazy.data.chunks == source_array.chunks

        # 4/ Check that computing the results did not load or replace either file-backed source
        assert source.data is source_array and not source._in_memory
        assert not native.is_loaded

    @pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
    @pytest.mark.parametrize("out_dtype", [None, np.float32, np.float64])
    @pytest.mark.parametrize("with_band", [False, True])
    def test_get_terrain_attribute__chunked_dimensions_and_dtype(
        self,
        dtype: Any,
        out_dtype: Any,
        with_band: bool,
    ) -> None:
        """Checks that chunked terrain arrays preserve spatial dimensions and the requested floating precision."""

        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # 1/ Create the same whole-number surface as NumPy and Dask arrays, with or without a band dimension
        rows, cols = np.indices((19, 23))
        values = (300 + rows**2 + cols**2).astype(dtype)
        if with_band:
            values = values[None, ...]
        source = da.from_array(values, chunks=(1, 4, 5) if with_band else (4, 5))
        graph = source.__dask_graph__()
        attributes = ["texture_shading", "slope", "curvature", "texture_shading"]
        options: dict[str, Any] = {"resolution": 20, "texture_alpha": 1.2, "out_dtype": out_dtype}

        # 2/ Create the Dask outputs without running any calculations
        # The callback records every executed Dask task, including work needed by texture shading
        tasks: list[Any] = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            outputs = xdem.terrain.get_terrain_attribute(source, attributes, **options)
        assert tasks == []
        expected = xdem.terrain.get_terrain_attribute(values, attributes, **options)
        # Integer inputs default to float32 results; out_dtype selects a different result type when provided
        default_dtype = np.float32 if dtype is np.int16 else dtype
        expected_dtype = np.dtype(default_dtype if out_dtype is None else out_dtype)

        # 3/ Compare values, shapes and types, then check the chunk sizes used by texture shading
        for name, output, reference in zip(attributes, outputs, expected):
            assert isinstance(output, da.Array)
            assert output.shape == reference.shape == values.shape
            assert output.dtype == reference.dtype == expected_dtype
            np.testing.assert_array_equal(output.compute(), reference)
            if name == "texture_shading":
                assert output.chunks == source.chunks
        assert source.__dask_graph__() is graph

    def test_get_terrain_attribute__chunked_backend_errors(self) -> None:
        """Checks that incompatible chunked backends fail before executing Dask tasks."""

        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # 1/ Prepare matching NumPy and Dask arrays plus multiprocessing options that cannot be used with them
        values = np.arange(99, dtype=float).reshape(9, 11)
        source = da.from_array(values, chunks=(4, 5))
        graph = source.__dask_graph__()
        config = MultiprocConfig(chunks=3)
        tasks: list[Any] = []

        # 2/ Reject Dask with multiprocessing, and reject multiprocessing for a plain NumPy array
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            with pytest.raises(ValueError, match="simultaneously"):
                xdem.terrain.get_terrain_attribute(source, "roughness", mp_config=config)
        with pytest.raises(TypeError, match="must be a Raster"):
            xdem.terrain.get_terrain_attribute(values, "roughness", mp_config=config)

        # 3/ Check that validation did not run or replace the Dask source
        assert tasks == []
        assert source.__dask_graph__() is graph

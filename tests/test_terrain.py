import os
import tempfile
import warnings

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio

import xdem

xdem.examples.download_longyearbyen_examples()

PLOT = True


def run_gdaldem(filepath: str, processing: str) -> np.ma.masked_array:
    """Run GDAL's DEMProcessing and return the read numpy array."""
    # rasterio strongly recommends against importing gdal along rio, so this is done here instead.
    from osgeo import gdal

    temp_dir = tempfile.TemporaryDirectory()
    temp_path = os.path.join(temp_dir.name, "output.tif")
    gdal.DEMProcessing(
        destName=temp_path,
        srcDS=filepath,
        processing=processing,
        options=gdal.DEMProcessingOptions(azimuth=315, altitude=45),
    )

    data = gu.Raster(temp_path).data
    temp_dir.cleanup()
    return data


class TestTerrainAttribute:
    filepath = xdem.examples.get_path("longyearbyen_ref_dem")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Parse metadata")
        dem = xdem.DEM(filepath, silent=True)

    @pytest.mark.parametrize("attribute", ["slope", "aspect", "hillshade"])
    def test_attribute_functions(self, attribute: str) -> None:
        """
        Test that all attribute functions (e.g. xdem.terrain.slope) behave appropriately.

        :param attribute: The attribute to test (e.g. 'slope')
        """
        warnings.simplefilter("error")

        functions = {
            "slope": lambda dem: xdem.terrain.slope(dem.data, dem.res, degrees=True),
            "aspect": lambda dem: xdem.terrain.aspect(dem.data, degrees=True),
            "hillshade": lambda dem: xdem.terrain.hillshade(dem.data, dem.res),
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both GDAL and xdem
        attr_xdem = functions[attribute](dem).squeeze()
        attr_gdal = run_gdaldem(self.filepath, attribute)

        # Check that the xdem and gdal hillshades are relatively similar.
        diff = (attr_xdem - attr_gdal).filled(np.nan)
        try:
            assert np.nanmean(diff) < 5
            assert xdem.spatialstats.nmad(diff) < 5
        except Exception as exception:

            if PLOT:
                import matplotlib.pyplot as plt

                plt.subplot(121)
                plt.imshow(attr_gdal.squeeze())
                plt.subplot(122)
                plt.imshow(attr_xdem.squeeze())
                plt.show()

            
            raise exception


        # Introduce some nans
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[np.random.choice(dem.data.size, 50000, replace=False)] = True

        # Validate that this doesn't raise weird warnings after introducing nans.
        functions[attribute](dem)

    def test_hillshade_errors(self) -> None:
        """Validate that the hillshade function raises appropriate errors."""
        # Try giving the hillshade invalid arguments.
        warnings.simplefilter("error")

        with pytest.raises(ValueError, match="Azimuth must be a value between 0 and 360"):
            xdem.terrain.hillshade(self.dem.data, self.dem.res, azimuth=361)

        with pytest.raises(ValueError, match="Altitude must be a value between 0 and 90"):
            xdem.terrain.hillshade(self.dem.data, self.dem.res, altitude=91)

        with pytest.raises(ValueError, match="z_factor must be a non-negative finite value"):
            xdem.terrain.hillshade(self.dem.data, self.dem.res, z_factor=np.inf)

    def test_hillshade(self) -> None:
        """Test hillshade-specific settings."""
        warnings.simplefilter("error")
        zfactor_1 = xdem.terrain.hillshade(self.dem.data, self.dem.res, z_factor=1.0)
        zfactor_10 = xdem.terrain.hillshade(self.dem.data, self.dem.res, z_factor=10.0)

        # A higher z-factor should be more variable than a low one.
        assert np.std(zfactor_1) < np.std(zfactor_10)

        low_altitude = xdem.terrain.hillshade(self.dem.data, self.dem.res, altitude=10)
        high_altitude = xdem.terrain.hillshade(self.dem.data, self.dem.res, altitude=80)

        # A low altitude should be darker than a high altitude.
        assert np.mean(low_altitude) < np.mean(high_altitude)

    @pytest.mark.parametrize("name", ["curvature", "planform_curvature", "profile_curvature"])
    def test_curvatures(self, name: str) -> None:
        """Test the curvature function (which has no GDAL equivalent)"""
        warnings.simplefilter("error")

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        curvature = xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)

        # Validate that the array has the same shape as the input and that all values are finite.
        assert curvature.shape == dem.data.shape
        try:
            assert np.all(np.isfinite(curvature))
        except:
            import matplotlib.pyplot as plt

            plt.imshow(curvature.squeeze())
            plt.show()

        with pytest.raises(ValueError, match="Quadric surface fit requires the same X and Y resolution."):
            xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=(1., 2.))

        # Introduce some nans
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[np.random.choice(dem.data.size, 50000, replace=False)] = True
        # Validate that this doesn't raise weird warnings after introducing nans.
        xdem.terrain.get_terrain_attribute(dem.data, attribute=name, resolution=dem.res)


    def test_get_terrain_attribute(self) -> None:
        """Test the get_terrain_attribute function by itself."""
        warnings.simplefilter("error")
        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        slope = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res)
        assert isinstance(slope, np.ndarray)

        # Create three products at the same time
        slope2, _, hillshade = xdem.terrain.get_terrain_attribute(
            self.dem.data, ["slope", "aspect", "hillshade"], resolution=self.dem.res
        )

        # Create a hillshade using its own function
        hillshade2 = xdem.terrain.hillshade(self.dem.data, self.dem.res)

        # Validate that the "batch-created" hillshades and slopes are the same as the "single-created"
        assert np.array_equal(hillshade, hillshade2)
        assert np.array_equal(slope, slope2)

        # A slope map with a lower resolution (higher value) should have gentler slopes.
        slope_lowres = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res[0] * 2)
        assert np.mean(slope) > np.mean(slope_lowres)

    def test_raster_argument(self):

        slope, aspect = xdem.terrain.get_terrain_attribute(self.dem, attribute=["slope", "aspect"])

        assert slope != aspect

        assert type(slope) == type(aspect)
        assert all(isinstance(r, gu.Raster) for r in (aspect, slope, self.dem))

        assert slope.transform == self.dem.transform == aspect.transform
        assert slope.crs == self.dem.crs == aspect.crs


def test_get_quadric_coefficients() -> None:
    """Test the outputs and exceptions of the get_quadric_coefficients() function."""
    warnings.simplefilter("error")

    dem = np.array([[1, 1, 1],
                    [1, 2, 1],
                    [1, 1, 1]], dtype="float32")

    coefficients = xdem.terrain.get_quadric_coefficients(dem, resolution=1.0)

    assert np.all(np.isfinite(coefficients))

    # The last coefficient is the dem itself (could maybe be removed in the future as it is duplication..)
    assert np.array_equal(coefficients[-1, :, :], dem)

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




import warnings
import xdem
import tempfile
import os
import numpy as np
import geoutils as gu
import pytest
import rasterio as rio

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
        options=gdal.DEMProcessingOptions(azimuth=315, altitude=45)
    )

    data = gu.Raster(temp_path).data
    temp_dir.cleanup()
    return data

class TestTerrainAttribute:
    filepath = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]

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
            "hillshade": lambda dem: xdem.terrain.hillshade(dem.data, dem.res)
        }

        # Copy the DEM to ensure that the inter-test state is unchanged, and because the mask will be modified.
        dem = self.dem.copy()

        # Derive the attribute using both GDAL and xdem
        attr_xdem = functions[attribute](dem).squeeze()
        attr_gdal = run_gdaldem(self.filepath, attribute)


        # Check that the xdem and gdal hillshades are relatively similar.
        diff = (attr_xdem - attr_gdal).filled(np.nan)
        assert np.nanmean(diff) < 5
        assert xdem.spatial_tools.nmad(diff) < 5

        # Introduce some nans
        dem.data.mask = np.zeros_like(dem.data, dtype=bool)
        dem.data.mask.ravel()[np.random.choice(
            dem.data.size, 50000, replace=False)] = True

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


    def test_get_terrain_attribute(self) -> None:
        """Test the get_terrain_attribute function by itself."""
        warnings.simplefilter("error")
        # Validate that giving only one terrain attribute only returns that, and not a list of len() == 1
        slope = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res)
        assert isinstance(slope, np.ndarray)

        # Create three products at the same time
        slope2, _, hillshade = xdem.terrain.get_terrain_attribute(self.dem.data, ["slope", "aspect", "hillshade"], resolution=self.dem.res)

        # Create a hillshade using its own function
        hillshade2 = xdem.terrain.hillshade(self.dem.data, self.dem.res)

        # Validate that the "batch-created" hillshades and slopes are the same as the "single-created"
        assert np.array_equal(hillshade, hillshade2)
        assert np.array_equal(slope, slope2)


        # A slope map with a lower resolution (higher value) should have gentler slopes.
        slope_lowres = xdem.terrain.get_terrain_attribute(self.dem.data, "slope", resolution=self.dem.res[0] * 2)
        assert np.mean(slope) > np.mean(slope_lowres)


"""Functions to test the coregistration approaches.

Author(s):
    Erik S. Holmlund

"""
import geoutils as gu
import pytest

from DemUtils import coreg
EXAMPLE_PATHS = {
    "dem1": "examples/Longyearbyen/data/DEM_2009_ref.tif",
    "dem2": "examples/Longyearbyen/data/DEM_1995.tif",
    "glacier_mask": "examples/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
}


class TestCoreg:

    def test_icp(self):
        """Test the ICP coregistration method."""
        reference_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem1"])
        to_be_aligned_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem2"])
        glacier_mask = gu.geovector.Vector(EXAMPLE_PATHS["glacier_mask"])

        _, error = coreg.coregister(
            reference_raster, to_be_aligned_raster, method="icp", mask=glacier_mask)

        assert error < 10

    def test_amaury(self):
        """Test the Amaury/ Nuth & Kääb method."""
        reference_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem1"])
        to_be_aligned_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem2"])
        glacier_mask = gu.geovector.Vector(EXAMPLE_PATHS["glacier_mask"])

        _, error = coreg.coregister(
            reference_raster, to_be_aligned_raster, method="amaury", mask=glacier_mask)

        assert error < 10

    def test_only_paths(self):
        """Test that raster paths can be specified instead of Raster objects."""
        reference_raster = EXAMPLE_PATHS["dem1"]
        to_be_aligned_raster = EXAMPLE_PATHS["dem2"]

        coreg.coregister(reference_raster, to_be_aligned_raster)

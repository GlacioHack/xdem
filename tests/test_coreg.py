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


def load_examples() -> tuple[gu.georaster.Raster, gu.georaster.Raster, gu.geovector.Vector]:
    reference_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem1"])
    to_be_aligned_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem2"])
    glacier_mask = gu.geovector.Vector(EXAMPLE_PATHS["glacier_mask"])
    return reference_raster, to_be_aligned_raster, glacier_mask


def test_coreg_method_enum():
    """Test that the CoregMethod enum works as it should."""
    # Try to generate an enum from a string
    icp = coreg.CoregMethod.from_str("icp")
    # Make sure the enum points to the right function
    assert icp == coreg.icp_coregistration

    # Make sure the following madness ends up in a ValueError
    try:
        coreg.CoregMethod.from_str("klawld")

        raise AssertionError
    except ValueError:
        pass


class TestCoreg:
    """Test different types of coregistration methods."""

    ref, tba, mask = load_examples()  # Load example reference, to-be-aligned and mask.

    def test_amaury(self):
        """Test the Amaury/ Nuth & Kääb method."""
        _, error = coreg.coregister(self.ref, self.tba, method="amaury", mask=self.mask)

        assert error < 10

    def test_amaury_high_degree(self):
        """Test the Amaury / Nuth & Kääb method with nonlinear deramping."""
        _, error = coreg.coregister(self.ref, self.tba, mask=self.mask, method="icp", deramping_degree=3)

        assert error < 10

    def test_deramping(self):
        """Test the deramping coregistration method."""
        _, error = coreg.coregister(self.ref, self.tba, method="deramp", mask=self.mask)

        assert error < 10

    def test_icp(self):
        """Test the ICP coregistration method."""
        _, error = coreg.coregister(self.ref, self.tba, method="icp", mask=self.mask)

        assert error < 10


def test_only_paths():
    """Test that raster paths can be specified instead of Raster objects."""
    reference_raster = EXAMPLE_PATHS["dem1"]
    to_be_aligned_raster = EXAMPLE_PATHS["dem2"]

    coreg.coregister(reference_raster, to_be_aligned_raster)

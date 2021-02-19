"""Functions to test the coregistration approaches.

Author(s):
    Erik S. Holmlund

"""
import os
import tempfile

import geoutils as gu
import pytest

from DemUtils import coreg

EXAMPLE_PATHS = {
    "dem1": "examples/Longyearbyen/data/DEM_2009_ref.tif",
    "dem2": "examples/Longyearbyen/data/DEM_1995.tif",
    "glacier_mask": "examples/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
}


def load_examples() -> tuple[gu.georaster.Raster, gu.georaster.Raster, gu.geovector.Vector]:
    """Load example files to try coregistration methods with."""
    reference_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem1"])
    to_be_aligned_raster = gu.georaster.Raster(EXAMPLE_PATHS["dem2"])
    glacier_mask = gu.geovector.Vector(EXAMPLE_PATHS["glacier_mask"])
    return reference_raster, to_be_aligned_raster, glacier_mask


def test_coreg_method_enum():
    """Test that the CoregMethod enum works as it should."""
    # Try to generate an enum from a string
    icp = coreg.CoregMethod.from_str("icp")
    # Make sure the enum points to the right function
    assert icp == coreg.icp_coregistration   # pylint: disable=comparison-with-callable

    # Make sure the following madness ends up in a ValueError
    try:
        coreg.CoregMethod.from_str("klawld")

        raise AssertionError
    except ValueError:
        pass


class TestCoreg:
    """Test different types of coregistration methods."""

    ref, tba, mask = load_examples()  # Load example reference, to-be-aligned and mask.

    def test_deramping(self):
        """Test the deramping coregistration method."""
        _, error = coreg.coregister(self.ref, self.tba, method="deramp", mask=self.mask)

        assert error < 10

    def test_raster_mask(self):
        """Test different ways of providing the mask as a raster instead of vector."""
        # Create a mask Raster.
        raster_mask = gu.georaster.Raster.from_array(
            data=self.mask.create_mask(self.ref),
            transform=self.ref.transform,  # pylint: disable=no-member
            crs=self.ref.crs  # pylint: disable=no-member
        )

        # Try to use it with the coregistration.
        # Run deramping as it's the fastest
        coreg.coregister(self.ref, self.tba, method="deramp", mask=raster_mask)

        # Save the raster to a temporary file
        temp_dir = tempfile.TemporaryDirectory()
        temp_raster_path = os.path.join(temp_dir.name, "raster_mask.tif")
        raster_mask.save(temp_raster_path)

        # Try to use the filepath to the temporary mask Raster instead.
        coreg.coregister(self.ref, self.tba, method="deramp", mask=temp_raster_path)

        # Make sure that the correct exception occurs when an invalid filepath is given.
        try:
            coreg.coregister(self.ref, self.tba, method="deramp", mask="jaajwdkjldkal.ddd")
        except ValueError as exception:
            if "Mask path not in a supported Raster or Vector format" in str(exception):
                pass
            else:
                raise exception

    def test_amaury(self):
        """Test the Amaury/ Nuth & Kääb method."""
        _, error = coreg.coregister(self.ref, self.tba, method="amaury", mask=self.mask)

        assert error < 10

    def test_amaury_high_degree(self):
        """Test the Amaury / Nuth & Kääb method with nonlinear deramping."""
        _, error = coreg.coregister(self.ref, self.tba, mask=self.mask, method="icp", deramping_degree=3)

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

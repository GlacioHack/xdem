"""Functions to test the coregistration approaches.

Author(s):
    Erik S. Holmlund

"""
from __future__ import annotations

import os
import tempfile
from typing import Any

import geoutils as gu

from xdem import coreg, examples


def load_examples() -> tuple[gu.georaster.Raster, gu.georaster.Raster, gu.geovector.Vector]:
    """Load example files to try coregistration methods with."""
    examples.download_longyearbyen_examples(overwrite=False)

    reference_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_ref_dem"])
    to_be_aligned_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_tba_dem"])
    glacier_mask = gu.geovector.Vector(examples.FILEPATHS["longyearbyen_glacier_outlines"])
    return reference_raster, to_be_aligned_raster, glacier_mask


def test_coreg_method_enum():
    """Test that the CoregMethod enum works as it should."""
    # Try to generate an enum from a string
    icp = coreg.CoregMethod.from_str("icp_pdal")
    # Make sure the enum points to the right function
    assert icp == coreg.icp_coregistration_pdal   # pylint: disable=comparison-with-callable

    # Make sure the following madness ends up in a ValueError
    try:
        coreg.CoregMethod.from_str("klawld")

        raise AssertionError
    except ValueError:
        pass


class TestCoreg:
    """Test different types of coregistration methods."""

    ref, tba, mask = load_examples()  # Load example reference, to-be-aligned and mask.

    def test_icp_opencv(self):
        """Test the opencv ICP coregistration method."""
        metadata: dict[str, Any] = {}
        _, error = coreg.coregister(self.ref, self.tba, method="icp_opencv", mask=self.mask, metadata=metadata)
        print(metadata)

        assert abs(metadata["icp_opencv"]["nmad"] - error) < 0.01

        assert error < 10

    def test_icp_pdal(self):
        """Test the ICP coregistration method."""
        metadata: dict[str, Any] = {}
        _, error = coreg.coregister(self.ref, self.tba, method="icp_pdal", mask=self.mask, metadata=metadata)

        assert abs(metadata["icp_pdal"]["nmad"] - error) < 0.01

        assert error < 10

    def test_deramping(self):
        """Test the deramping coregistration method."""
        metadata = {}
        _, error = coreg.coregister(self.ref, self.tba, method="deramp", mask=self.mask, metadata=metadata)

        assert round(metadata["deramp"]["nmad"], 1) == round(error, 1)

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
        metadata = {}
        _, error = coreg.coregister(self.ref, self.tba, method="amaury", mask=self.mask, metadata=metadata)

        assert metadata["nuth_kaab"]["nmad"] == error

        assert error < 10

    def test_amaury_high_degree(self):
        """Test the Amaury / Nuth & Kääb method with nonlinear deramping."""
        _, error = coreg.coregister(self.ref, self.tba, mask=self.mask, method="icp", deramping_degree=3)

        assert error < 10


def test_only_paths():
    """Test that raster paths can be specified instead of Raster objects."""
    reference_raster = examples.FILEPATHS["longyearbyen_ref_dem"]
    to_be_aligned_raster = examples.FILEPATHS["longyearbyen_tba_dem"]

    coreg.coregister(reference_raster, to_be_aligned_raster)

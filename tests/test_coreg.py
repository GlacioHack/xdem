"""Functions to test the coregistration approaches.

Author(s):
    Erik S. Holmlund

"""
from __future__ import annotations

import copy
import os
import tempfile
import warnings
from typing import Any

import geoutils as gu
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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


class TestCoregClass:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    mask = outlines.create_mask(ref) == 255

    def test_bias(self):
        warnings.simplefilter("error")

        # Create a bias correction instance
        biascorr = coreg.BiasCorr()
        # Fit the bias model to the data
        biascorr.fit(reference_dem=self.ref.data, dem_to_be_aligned=self.tba.data, mask=self.mask)

        # Check that a bias was found.
        assert biascorr._meta.get("bias") is not None
        assert biascorr._meta["bias"] != 0.0

        # Copy the bias to see if it changes in the test (it shouldn't)
        bias = copy.copy(biascorr._meta["bias"])

        # Check that the to_matrix function works as it should
        matrix = biascorr.to_matrix()
        assert matrix[2, 3] == bias, matrix

        # Create some 3D coordinates with Z coordinates being 0
        points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
        # Cehck that the first z coordinate is now the bias
        assert biascorr.apply_pts(points)[0, 2] == biascorr._meta["bias"]

        # Apply the model to correct the DEM
        tba_unbiased = biascorr.apply(self.tba.data, None)

        # Create a new bias correction model
        biascorr2 = coreg.BiasCorr()
        # Check that this is indeed a new object
        assert biascorr is not biascorr2
        # Fit the corrected DEM to see if the bias will be close to or at zero
        biascorr2.fit(reference_dem=self.ref.data, dem_to_be_aligned=tba_unbiased, mask=self.mask)
        # Test the bias
        assert abs(biascorr2._meta.get("bias")) < 0.01

        # Check that the original model's bias has not changed (the _meta dicts are two different objects)
        assert biascorr._meta["bias"] == bias

    def test_icp_opencv(self):
        warnings.simplefilter("error")

        icp = coreg.ICP(max_iterations=3)

        icp.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        aligned_dem = icp.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline(self):
        warnings.simplefilter("error")

        pipeline = coreg.CoregPipeline([coreg.BiasCorr(), coreg.ICP(max_iterations=3)])

        pipeline.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        aligned_dem = pipeline.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

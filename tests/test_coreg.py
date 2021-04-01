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

    fit_params = dict(
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        mask=~mask,
        transform=ref.transform
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_bias(self):
        warnings.simplefilter("error")

        # Create a bias correction instance
        biascorr = coreg.BiasCorr()
        # Fit the bias model to the data
        biascorr.fit(**self.fit_params)

        # Check that a bias was found.
        assert biascorr._meta.get("bias") is not None
        assert biascorr._meta["bias"] != 0.0

        # Copy the bias to see if it changes in the test (it shouldn't)
        bias = copy.copy(biascorr._meta["bias"])

        # Check that the to_matrix function works as it should
        matrix = biascorr.to_matrix()
        assert matrix[2, 3] == bias, matrix

        # Cehck that the first z coordinate is now the bias
        assert biascorr.apply_pts(self.points)[0, 2] == biascorr._meta["bias"]

        # Apply the model to correct the DEM
        tba_unbiased = biascorr.apply(self.tba.data, None)

        # Create a new bias correction model
        biascorr2 = coreg.BiasCorr()
        # Check that this is indeed a new object
        assert biascorr is not biascorr2
        # Fit the corrected DEM to see if the bias will be close to or at zero
        biascorr2.fit(reference_dem=self.ref.data, dem_to_be_aligned=tba_unbiased, mask=~self.mask)
        # Test the bias
        assert abs(biascorr2._meta.get("bias")) < 0.01

        # Check that the original model's bias has not changed (that the _meta dicts are two different objects)
        assert biascorr._meta["bias"] == bias

    def test_nuth_kaab(self):
        warnings.simplefilter("error")

        nuth_kaab = coreg.NuthKaab(max_iterations=10)

        # Synthesize a shifted and vertically offset DEM
        pixel_shift = 2
        bias = 5
        shifted_dem = self.ref.data.squeeze().copy()
        shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
        shifted_dem[:, :pixel_shift] = np.nan
        shifted_dem += bias

        nuth_kaab.fit(self.ref.data.squeeze(), shifted_dem, transform=self.ref.transform)

        assert abs(nuth_kaab._meta["offset_east_px"] - pixel_shift) < 0.03
        assert abs(nuth_kaab._meta["offset_north_px"]) < 0.03
        assert abs(nuth_kaab._meta["bias"] + bias) < 0.03

        unshifted_dem = nuth_kaab.apply(shifted_dem, transform=self.ref.transform)
        diff = np.asarray(self.ref.data.squeeze() - unshifted_dem)

        assert np.abs(np.nanmedian(diff)) < 0.01
        assert np.sqrt(np.nanmean(np.square(diff))) < 1

        transformed_points = nuth_kaab.apply_pts(self.points)

        assert abs((transformed_points[0, 0] - self.points[0, 0]) + pixel_shift * self.ref.res[0]) < 0.1
        assert abs((transformed_points[0, 2] - self.points[0, 2]) + bias) < 0.1

    def test_deramping(self):
        warnings.simplefilter("error")

        deramp = coreg.Deramp(degree=1)

        deramp.fit(**self.fit_params)

        deramped_dem = deramp.apply(self.tba.data, self.ref.transform)

        periglacial_offset = (self.ref.data.squeeze() - deramped_dem)[~self.mask.squeeze()]
        pre_offset = (self.ref.data - self.tba.data).squeeze()[~self.mask]

        # Check that the error improved
        assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

        # Check that the mean periglacial offset is low
        assert -1 < np.mean(periglacial_offset) < 1

        deramp0 = coreg.Deramp(degree=0)

        deramp0.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        assert len(deramp0._meta["coefficients"]) == 1
        bias = deramp0._meta["coefficients"][0]

        # Make sure to_matrix does not throw an error.
        deramp0.to_matrix()

        # Create some 3D coordinates with Z coordinates being 0
        points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

        assert deramp0.apply_pts(points)[0, 2] == bias

    def test_icp_opencv(self):
        warnings.simplefilter("error")

        icp = coreg.ICP(max_iterations=3)

        icp.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        aligned_dem = icp.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline(self):
        warnings.simplefilter("error")

        pipeline = coreg.CoregPipeline([coreg.BiasCorr(), coreg.ICP(max_iterations=3)])

        pipeline.fit(**self.fit_params)

        aligned_dem = pipeline.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

        pipeline2 = coreg.CoregPipeline([coreg.BiasCorr(), coreg.BiasCorr()])
        pipeline2.pipeline[0]._meta["bias"] = 1
        pipeline2.pipeline[1]._meta["bias"] = 1

        pipeline2.to_matrix()[2, 3] == 2.0

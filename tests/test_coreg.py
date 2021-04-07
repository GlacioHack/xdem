"""Functions to test the coregistration approaches.

Author(s):
    Erik S. Holmlund

"""
from __future__ import annotations

import copy
import os
import tempfile
import time
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

        # Check that the first z coordinate is now the bias
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

        # Fit the synthesized shifted DEM to the original
        nuth_kaab.fit(self.ref.data.squeeze(), shifted_dem, transform=self.ref.transform)

        # Make sure that the estimated offsets are similar to what was synthesized.
        assert abs(nuth_kaab._meta["offset_east_px"] - pixel_shift) < 0.03
        assert abs(nuth_kaab._meta["offset_north_px"]) < 0.03
        assert abs(nuth_kaab._meta["bias"] + bias) < 0.03

        # Apply the estimated shift to "revert the DEM" to its original state.
        unshifted_dem = nuth_kaab.apply(shifted_dem, transform=self.ref.transform)
        # Measure the difference (should be more or less zero)
        diff = np.asarray(self.ref.data.squeeze() - unshifted_dem)

        # Check that the median is very close to zero
        assert np.abs(np.nanmedian(diff)) < 0.01
        # Check that the RMSE is low
        assert np.sqrt(np.nanmean(np.square(diff))) < 1

        # Transform some arbitrary points.
        transformed_points = nuth_kaab.apply_pts(self.points)

        # Check that the x shift is close to the pixel_shift * image resolution
        assert abs((transformed_points[0, 0] - self.points[0, 0]) + pixel_shift * self.ref.res[0]) < 0.1
        # Check that the z shift is close to the original bias.
        assert abs((transformed_points[0, 2] - self.points[0, 2]) + bias) < 0.1

    def test_deramping(self):
        warnings.simplefilter("error")

        # Try a 1st degree deramping.
        deramp = coreg.Deramp(degree=1)

        # Fit the data
        deramp.fit(**self.fit_params)

        # Apply the deramping to a DEm
        deramped_dem = deramp.apply(self.tba.data, self.ref.transform)

        # Get the periglacial offset after deramping
        periglacial_offset = (self.ref.data.squeeze() - deramped_dem)[~self.mask.squeeze()]
        # Get the periglacial offset before deramping
        pre_offset = (self.ref.data - self.tba.data).squeeze()[~self.mask]

        # Check that the error improved
        assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

        # Check that the mean periglacial offset is low
        assert np.abs(np.mean(periglacial_offset)) < 1

        # Try a 0 degree deramp (basically bias correction)
        deramp0 = coreg.Deramp(degree=0)
        deramp0.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        # Check that only one coefficient exists (y = x + a => coefficients=["a"])
        assert len(deramp0._meta["coefficients"]) == 1
        # Extract said bias
        bias = deramp0._meta["coefficients"][0]

        # Make sure to_matrix does not throw an error. It will for higher degree deramps
        deramp0.to_matrix()

        # Check that the apply_pts would apply a z shift equal to the bias
        assert deramp0.apply_pts(self.points)[0, 2] == bias

    def test_icp_opencv(self):
        warnings.simplefilter("error")

        # Do a fast an dirty 3 iteration ICP just to make sure it doesn't error out.
        icp = coreg.ICP(max_iterations=3)
        icp.fit(self.ref.data, self.tba.data, ~self.mask, transform=self.ref.transform)

        aligned_dem = icp.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline(self):
        warnings.simplefilter("error")

        # Create a pipeline from two coreg methods.
        pipeline = coreg.CoregPipeline([coreg.BiasCorr(), coreg.ICP(max_iterations=3)])
        pipeline.fit(**self.fit_params)

        aligned_dem = pipeline.apply(self.tba.data, self.ref.transform)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

        # Make a new pipeline with two bias correction approaches.
        pipeline2 = coreg.CoregPipeline([coreg.BiasCorr(), coreg.BiasCorr()])
        # Set both "estimated" biases to be 1
        pipeline2.pipeline[0]._meta["bias"] = 1
        pipeline2.pipeline[1]._meta["bias"] = 1

        # Assert that the combined bias is 2
        pipeline2.to_matrix()[2, 3] == 2.0

    def test_coreg_add(self):
        warnings.simplefilter("error")
        # Test with a bias of 4
        bias = 4

        bias1 = coreg.BiasCorr()
        bias2 = coreg.BiasCorr()

        # Set the bias attribute
        for bias_corr in (bias1, bias2):
            bias_corr._meta["bias"] = bias

        # Add the two coregs and check that the resulting bias is 2* bias
        bias3 = bias1 + bias2
        assert bias3.to_matrix()[2, 3] == bias * 2

        # Make sure the correct exception is raised on incorrect additions
        try:
            bias1 + 1
        except ValueError as exception:
            if "Incompatible add type" not in str(exception):
                raise exception

        # Try to add a Coreg step to an already existing CoregPipeline
        bias4 = bias3 + bias1
        assert bias4.to_matrix()[2, 3] == bias * 3

        # Try to add two CoregPipelines
        bias5 = bias3 + bias3
        assert bias5.to_matrix()[2, 3] == bias * 4

    def test_subsample(self):
        warnings.simplefilter("error")

        # Test subsampled bias correction
        bias_sub = coreg.BiasCorr()

        # Fit the bias using 50% of the unmasked data using a fraction
        bias_sub.fit(**self.fit_params, subsample=0.5)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        bias_sub.fit(**self.fit_params, subsample=self.tba.data.size // 2)

        # Do full bias corr to compare
        bias_full = coreg.BiasCorr()
        bias_full.fit(**self.fit_params)

        # Check that the estimated biases are similar
        assert abs(bias_sub._meta["bias"] - bias_full._meta["bias"]) < 0.1

        # Test ICP with subsampling
        icp_full = coreg.ICP(max_iterations=20)
        icp_sub = coreg.ICP(max_iterations=20)

        # Measure the start and stop time to get the duration
        start_time = time.time()
        icp_full.fit(**self.fit_params)
        icp_full_duration = time.time() - start_time

        # Do the same with 50% subsampling
        start_time = time.time()
        icp_sub.fit(**self.fit_params, subsample=0.5)
        icp_sub_duration = time.time() - start_time

        # Make sure that the subsampling increased performance
        assert icp_full_duration > icp_sub_duration

        # Calculate the difference in the full vs. subsampled ICP matrices
        matrix_diff = np.abs(icp_full.to_matrix() - icp_sub.to_matrix())
        # Check that the x/y/z differences do not exceed 30cm
        assert np.count_nonzero(matrix_diff > 0.3) == 0

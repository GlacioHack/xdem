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

import cv2
import geoutils as gu
import numpy as np
import pytransform3d.transformations

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem import coreg, examples, spatial_tools


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
            data=self.mask.create_mask(self.ref).astype('uint8'),
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
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        verbose=False,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_from_classmethods(self):
        warnings.simplefilter("error")

        # Check that the from_matrix function works as expected.
        bias = 5
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[2, 3] = bias
        coreg_obj = coreg.Coreg.from_matrix(matrix)
        transformed_points = coreg_obj.apply_pts(self.points)
        assert transformed_points[0, 2] == bias

        # Check that the from_translation function works as expected.
        x_offset = 5
        coreg_obj2 = coreg.Coreg.from_translation(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply_pts(self.points)
        assert np.array_equal(self.points[:, 0] + x_offset, transformed_points2[:, 0])

        # Try to make a Coreg object from a nan translation (should fail).
        try:
            coreg.Coreg.from_translation(np.nan)
        except ValueError as exception:
            if "non-finite values" not in str(exception):
                raise exception

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
        tba_unbiased = biascorr.apply(self.tba.data, self.ref.transform)

        # Create a new bias correction model
        biascorr2 = coreg.BiasCorr()
        # Check that this is indeed a new object
        assert biascorr is not biascorr2
        # Fit the corrected DEM to see if the bias will be close to or at zero
        biascorr2.fit(reference_dem=self.ref.data, dem_to_be_aligned=tba_unbiased, inlier_mask=self.inlier_mask)
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
        nuth_kaab.fit(self.ref.data.squeeze(), shifted_dem,
                      transform=self.ref.transform, verbose=self.fit_params["verbose"])

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
        assert abs((transformed_points[0, 0] - self.points[0, 0]) - pixel_shift * self.ref.res[0]) < 0.1
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
        periglacial_offset = (self.ref.data.squeeze() - deramped_dem)[self.inlier_mask.squeeze()]
        # Get the periglacial offset before deramping
        pre_offset = ((self.ref.data - self.tba.data)[self.inlier_mask]).squeeze()

        # Check that the error improved
        assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

        # Check that the mean periglacial offset is low
        assert np.abs(np.mean(periglacial_offset)) < 1

        # Try a 0 degree deramp (basically bias correction)
        deramp0 = coreg.Deramp(degree=0)
        deramp0.fit(**self.fit_params)

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
        icp.fit(**self.fit_params)

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

    def test_apply_matrix(self):
        warnings.simplefilter("error")
        # This should maybe be its own function, but would just repeat the data loading procedure..

        # Test only bias (it should just apply the bias and not make anything else)
        bias = 5
        matrix = np.diag(np.ones(4, float))
        matrix[2, 3] = bias
        transformed_dem = coreg.apply_matrix(self.ref.data.squeeze(), self.ref.transform, matrix)
        reverted_dem = transformed_dem - bias

        # Check that the revered DEM has the exact same values as the initial one
        # (resampling is not an exact science, so this will only apply for bias corrections)
        assert np.nanmedian(reverted_dem) == np.nanmedian(np.asarray(self.ref.data))

        # Synthesize a shifted and vertically offset DEM
        pixel_shift = 11
        bias = 5
        shifted_dem = self.ref.data.squeeze().copy()
        shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
        shifted_dem[:, :pixel_shift] = np.nan
        shifted_dem += bias

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] = pixel_shift * self.tba.res[0]
        matrix[2, 3] = -bias

        transformed_dem = coreg.apply_matrix(shifted_dem.data.squeeze(), self.ref.transform, matrix)

        diff = np.asarray(self.ref.data.squeeze() - transformed_dem)

        # Check that the median is very close to zero
        assert np.abs(np.nanmedian(diff)) < 0.05
        # Check that the NMAD is low
        assert spatial_tools.nmad(diff) < 3

        def rotation_matrix(rotation=30):
            rotation = np.deg2rad(rotation)
            matrix = np.array([
                [1, 0, 0, 0],
                [0, np.cos(rotation), -np.sin(rotation), 0],
                [0, np.sin(rotation), np.cos(rotation), 0],
                [0, 0, 0, 1]
            ])
            return matrix

        rotation = 4
        rotated_dem = coreg.apply_matrix(
            self.ref.data.squeeze(),
            self.ref.transform,
            rotation_matrix(rotation),
        )
        # Make sure that the rotated DEM is way off.
        assert np.abs(np.nanmedian(rotated_dem - self.ref.data.data)) > 400

        # Apply a rotation in the opposite direction
        unrotated_dem = coreg.apply_matrix(
            rotated_dem,
            self.ref.transform,
            rotation_matrix(-rotation * 0.989)  # This is not exactly -rotation, probably due to displaced pixels.
        )

        diff = np.asarray(self.ref.data.squeeze() - unrotated_dem)

        if False:
            import matplotlib.pyplot as plt

            vmin = 0
            vmax = 1500
            extent = (self.ref.bounds.left, self.ref.bounds.right, self.ref.bounds.bottom, self.ref.bounds.top)
            plot_params = dict(
                extent=extent,
                vmin=vmin,
                vmax=vmax
            )
            plt.figure(figsize=(22, 4), dpi=100)
            plt.subplot(151)
            plt.title("Original")
            plt.imshow(self.ref.data.squeeze(), **plot_params)
            plt.xlim(*extent[:2])
            plt.ylim(*extent[2:])
            plt.subplot(152)
            plt.title(f"Rotated {rotation} degrees")
            plt.imshow(rotated_dem, **plot_params)
            plt.xlim(*extent[:2])
            plt.ylim(*extent[2:])
            plt.subplot(153)
            plt.title(f"De-rotated {-rotation} degrees")
            plt.imshow(unrotated_dem, **plot_params)
            plt.xlim(*extent[:2])
            plt.ylim(*extent[2:])
            plt.subplot(154)
            plt.title("Original vs. de-rotated")
            plt.imshow(diff, extent=extent, vmin=-10, vmax=10, cmap="coolwarm_r")
            plt.colorbar()
            plt.xlim(*extent[:2])
            plt.ylim(*extent[2:])
            plt.subplot(155)
            plt.title("Original vs. de-rotated")
            plt.hist(diff[np.isfinite(diff)], bins=np.linspace(-10, 10, 100))
            plt.tight_layout(w_pad=0.05)
            plt.show()

        # Check that the median is very close to zero
        assert np.abs(np.nanmedian(diff)) < 0.5
        # Check that the NMAD is low
        assert spatial_tools.nmad(diff) < 5
        print(np.nanmedian(diff), spatial_tools.nmad(diff))

    def test_z_scale_corr(self):
        warnings.simplefilter("error")

        # Instantiate a Z scale correction object
        zcorr = coreg.ZScaleCorr()

        # This is the z-scale to multiply the DEM with.
        factor = 1.2
        scaled_dem = self.ref.data * factor

        # Fit the correction
        zcorr.fit(self.ref.data, scaled_dem)

        # Apply the correction
        unscaled_dem = zcorr.apply(scaled_dem, None)

        # Make sure the difference is now minimal
        diff = (self.ref.data - unscaled_dem).filled(np.nan)
        assert np.abs(np.nanmedian(diff)) < 0.01

        # Create a spatially correlated error field to mess with the algorithm a bit.
        corr_size = int(self.ref.data.shape[2] / 100)
        error_field = cv2.resize(
            cv2.GaussianBlur(
                np.repeat(np.repeat(
                    np.random.randint(0, 255, (self.ref.data.shape[1]//corr_size,
                                               self.ref.data.shape[2]//corr_size), dtype='uint8'),
                    corr_size, axis=0), corr_size, axis=1),
                ksize=(2*corr_size + 1, 2*corr_size + 1),
                sigmaX=corr_size) / 255,
            dsize=(self.ref.data.shape[2], self.ref.data.shape[1])
        )

        # Create 50000 random nans
        dem_with_nans = self.ref.data.copy()
        dem_with_nans.mask = np.zeros_like(dem_with_nans, dtype=bool)
        dem_with_nans.mask.ravel()[np.random.choice(dem_with_nans.data.size, 50000, replace=False)] = True

        # Add spatially correlated errors in the order of +- 5 m
        dem_with_nans += error_field * 3

        # Try the fit now with the messed up DEM as reference.
        zcorr.fit(dem_with_nans, scaled_dem)
        unscaled_dem = zcorr.apply(scaled_dem, None)
        diff = (dem_with_nans - unscaled_dem).filled(np.nan)
        assert np.abs(np.nanmedian(diff)) < 0.05

        # Try a second-degree scaling
        scaled_dem = 1e-4 * self.ref.data ** 2 + 300 + self.ref.data * factor

        # Try to correct using a nonlinear correction.
        zcorr_nonlinear = coreg.ZScaleCorr(degree=2)
        zcorr_nonlinear.fit(dem_with_nans, scaled_dem)

        # Make sure the difference is minimal
        unscaled_dem = zcorr_nonlinear.apply(scaled_dem, None)
        diff = (dem_with_nans - unscaled_dem).filled(np.nan)
        assert np.abs(np.nanmedian(diff)) < 0.05

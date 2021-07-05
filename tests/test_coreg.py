"""Functions to test the coregistration tools."""
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
import pytest
import rasterio as rio
import pytransform3d.transformations

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem import coreg, examples, spatial_tools, misc
    import xdem


def load_examples() -> tuple[gu.georaster.Raster, gu.georaster.Raster, gu.geovector.Vector]:
    """Load example files to try coregistration methods with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_raster = gu.georaster.Raster(examples.get_path("longyearbyen_ref_dem"))
        to_be_aligned_raster = gu.georaster.Raster(examples.get_path("longyearbyen_tba_dem"))
        glacier_mask = gu.geovector.Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


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

    @pytest.mark.parametrize("coreg_class", [coreg.BiasCorr, coreg.ICP, coreg.NuthKaab])
    def test_copy(self, coreg_class: coreg.Coreg):
        """Test that copying work expectedly (that no attributes still share references)."""
        warnings.simplefilter("error")

        # Create a coreg instance and copy it.
        corr = coreg_class()
        corr_copy = corr.copy()

        # Assign some attributes and metadata after copying
        corr.foo = "bar"
        corr._meta["hello"] = "there"
        # Make sure these don't appear in the copy
        assert corr_copy._meta != corr._meta
        assert not hasattr(corr_copy, "foo")

        # Create a pipeline, add some metadata, and copy it
        pipeline = coreg_class() + coreg_class()
        pipeline.pipeline[0]._meta["shouldexist"] = True

        pipeline_copy = pipeline.copy()

        # Add some more metadata after copying (this should not be transferred)
        pipeline._meta["hello"] = "there"
        pipeline_copy.pipeline[0]._meta["foo"] = "bar"

        assert pipeline._meta != pipeline_copy._meta
        assert pipeline.pipeline[0]._meta != pipeline_copy.pipeline[0]._meta
        assert pipeline_copy.pipeline[0]._meta["shouldexist"]


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
        biascorr2.fit(reference_dem=self.ref.data, dem_to_be_aligned=tba_unbiased, transform=self.ref.transform, inlier_mask=self.inlier_mask)
        # Test the bias
        assert abs(biascorr2._meta.get("bias")) < 0.01

        # Check that the original model's bias has not changed (that the _meta dicts are two different objects)
        assert biascorr._meta["bias"] == bias

    def test_all_nans(self):
        """Check that the coregistration approaches fail gracefully when given only nans."""
        dem1 = np.ones((50, 50), dtype=float)
        dem2 = dem1.copy() + np.nan
        affine = rio.transform.from_origin(0, 0, 1, 1)

        biascorr = coreg.BiasCorr()
        icp = coreg.ICP()
        
        pytest.raises(ValueError, biascorr.fit, dem1, dem2, transform=affine)
        pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)

        dem2[[3, 20, 40], [2, 21, 41]] = 1.2

        biascorr.fit(dem1, dem2, transform=affine)

        pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)
    

    def test_error_method(self):
        """Test different error measures."""
        dem1 = np.ones((50, 50), dtype=float)
        # Create a biased dem
        dem2 = dem1 + 2
        affine = rio.transform.from_origin(0, 0, 1, 1)

        biascorr = coreg.BiasCorr()
        # Fit the bias
        biascorr.fit(dem1, dem2, transform=affine)

        # Check that the bias after coregistration is zero
        assert biascorr.error(dem1, dem2, transform=affine, error_type="median") == 0

        # Remove the bias fit and see what happens.
        biascorr._meta["bias"] = 0
        # Now it should be equal to dem1 - dem2
        assert biascorr.error(dem1, dem2, transform=affine, error_type="median") == -2

        # Create random noise and see if the standard deviation is equal (it should)
        dem3 = dem1 + np.random.random(size=dem1.size).reshape(dem1.shape)
        assert abs(biascorr.error(dem1, dem3, transform=affine, error_type="std") - np.std(dem3)) < 1e-6


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
        pipeline = coreg.CoregPipeline([coreg.BiasCorr(), coreg.NuthKaab()])
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

        # Test NuthKaab with subsampling
        nuthkaab_full = coreg.NuthKaab()
        nuthkaab_sub = coreg.NuthKaab()

        # Measure the start and stop time to get the duration
        start_time = time.time()
        nuthkaab_full.fit(**self.fit_params)
        icp_full_duration = time.time() - start_time

        # Do the same with 50% subsampling
        start_time = time.time()
        nuthkaab_sub.fit(**self.fit_params, subsample=0.5)
        icp_sub_duration = time.time() - start_time

        # Make sure that the subsampling increased performance
        # Temporarily add a fallback assertion that if it's slower, it shouldn't be much slower (2021-05-17).
        # This doesn't work with GitHub's CI, but it works locally. I'm disabling this for now (2021-05-20).
        #assert icp_full_duration > icp_sub_duration or (abs(icp_full_duration - icp_sub_duration) < 1)

        # Calculate the difference in the full vs. subsampled matrices
        matrix_diff = np.abs(nuthkaab_full.to_matrix() - nuthkaab_sub.to_matrix())
        # Check that the x/y/z differences do not exceed 30cm
        assert np.count_nonzero(matrix_diff > 0.3) == 0


    def test_z_scale_corr(self):
        warnings.simplefilter("error")

        # Instantiate a Z scale correction object
        zcorr = coreg.ZScaleCorr()

        # This is the z-scale to multiply the DEM with.
        factor = 1.2
        scaled_dem = self.ref.data * factor

        # Fit the correction
        zcorr.fit(self.ref.data, scaled_dem, transform=self.ref.transform)

        # Apply the correction
        unscaled_dem = zcorr.apply(scaled_dem, self.ref.transform)

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
        zcorr.fit(dem_with_nans, scaled_dem, transform=self.ref.transform)
        unscaled_dem = zcorr.apply(scaled_dem, self.ref.transform)
        diff = (dem_with_nans - unscaled_dem).filled(np.nan)
        assert np.abs(np.nanmedian(diff)) < 0.05

        # Try a second-degree scaling
        scaled_dem = 1e-4 * self.ref.data ** 2 + 300 + self.ref.data * factor

        # Try to correct using a nonlinear correction.
        zcorr_nonlinear = coreg.ZScaleCorr(degree=2)
        zcorr_nonlinear.fit(dem_with_nans, scaled_dem, transform=self.ref.transform)

        # Make sure the difference is minimal
        unscaled_dem = zcorr_nonlinear.apply(scaled_dem, self.ref.transform)
        diff = (dem_with_nans - unscaled_dem).filled(np.nan)
        assert np.abs(np.nanmedian(diff)) < 0.05

    @pytest.mark.parametrize(
        "pipeline",
        [
            coreg.BiasCorr(),
            coreg.BiasCorr() + coreg.NuthKaab()
        ]
    )
    @pytest.mark.parametrize(
        "subdivision",
        [
            4,
            10,
        ]
    )
    def test_blockwise_coreg(self, pipeline, subdivision):
        warnings.simplefilter("error")

        blockwise = coreg.BlockwiseCoreg(coreg=pipeline, subdivision=subdivision)

        # Results can not yet be extracted (since fit has not been called) and should raise an error
        with pytest.raises(AssertionError, match="No coreg results exist.*"):
            blockwise.to_points()
    
        blockwise.fit(**self.fit_params)
        points = blockwise.to_points()

        # Validate that the number of points is equal to the amount of subdivisions.
        assert points.shape[0] == subdivision

        # Validate that the points do not represent only the same location.
        assert np.sum(np.linalg.norm(points[:, :, 0] - points[:, :, 1], axis=1)) != 0.0

        z_diff = points[:, 2, 1] - points[:, 2, 0]

        # Validate that all values are different
        assert np.unique(z_diff).size == z_diff.size, "Each coreg cell should have different results."

        # Validate that the BlockwiseCoreg doesn't accept uninstantiated Coreg classes
        with pytest.raises(ValueError, match="instantiated Coreg subclass"):
            coreg.BlockwiseCoreg(coreg=coreg.BiasCorr, subdivision=1)  # type: ignore

        # Metadata copying has been an issue. Validate that all chunks have unique ids
        chunk_numbers = [m["i"] for m in blockwise._meta["coreg_meta"]]
        assert np.unique(chunk_numbers).shape[0] == len(chunk_numbers)

        transformed_dem = blockwise.apply(self.tba.data, self.tba.transform)

        ddem_pre = (self.ref.data - self.tba.data)[~self.inlier_mask].squeeze().filled(np.nan)
        ddem_post = (self.ref.data.squeeze() - transformed_dem)[~self.inlier_mask.squeeze()].filled(np.nan)

        # Check that the periglacial difference is lower after coregistration.
        assert abs(np.nanmedian(ddem_post)) < abs(np.nanmedian(ddem_pre))

        stats = blockwise.stats()

        # Check that nans don't exist (if they do, something has gone very wrong)
        assert np.all(np.isfinite(stats["nmad"]))
        # Check that offsets were actually calculated.
        assert np.sum(np.abs(np.linalg.norm(stats[["x_off", "y_off", "z_off"]], axis=0))) > 0

    def test_coreg_raster_and_ndarray_args(_) -> None:

        # Create a small sample-DEM
        dem1 = xdem.DEM.from_array(
            np.arange(25, dtype="int32").reshape(5, 5),
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999
        )
        # Assign a funny value to one particular pixel. This is to validate that reprojection works perfectly.
        dem1.data[0, 1, 1] = 100

        # Translate the DEM 1 "meter" right and add a bias
        dem2 = dem1.reproject(dst_bounds=rio.coords.BoundingBox(1, 0, 6, 5))
        dem2 += 1

        # Create a biascorr for Rasters ("_r") and for arrays ("_a")
        biascorr_r = coreg.BiasCorr()
        biascorr_a = biascorr_r.copy()

        # Fit the data
        biascorr_r.fit(
            reference_dem=dem1,
            dem_to_be_aligned=dem2
        )
        biascorr_a.fit(
            reference_dem=dem1.data,
            dem_to_be_aligned=dem2.reproject(dem1).data,
            transform=dem1.transform
        )

        # Validate that they ended up giving the same result.
        assert biascorr_r._meta["bias"] == biascorr_a._meta["bias"]

        # De-shift dem2
        dem2_r = biascorr_r.apply(dem2)
        dem2_a = biascorr_a.apply(dem2.data, dem2.transform)

        # Validate that the return formats were the expected ones, and that they are equal.
        assert isinstance(dem2_r, xdem.DEM)
        assert isinstance(dem2_a, np.ma.masked_array)
        assert np.array_equal(dem2_r, dem2_r)

        # If apply on a masked_array was given without a transform, it should fail.
        with pytest.raises(ValueError, match="'transform' must be given"):
            biascorr_a.apply(dem2.data)

        with pytest.warns(UserWarning, match="DEM .* overrides the given 'transform'"):
            biascorr_a.apply(dem2, transform=dem2.transform)


    @pytest.mark.parametrize("combination", [
        ("dem1", "dem2", "None", "fit", "passes", ""),
        ("dem1", "dem2", "None", "apply", "passes", ""),
        ("dem1.data", "dem2.data", "dem1.transform", "fit", "passes", ""),
        ("dem1.data", "dem2.data", "dem1.transform", "apply", "passes", ""),
        ("dem1", "dem2.data", "dem1.transform", "fit", "warns", "'reference_dem' .* overrides the given 'transform'"),
        ("dem1.data", "dem2", "dem1.transform", "fit", "warns", "'dem_to_be_aligned' .* overrides .*"),
        ("dem1.data", "dem2.data", "None", "fit", "error", "'transform' must be given if both DEMs are array-like."),
        ("dem1", "dem2.data", "None", "apply", "error", "'transform' must be given if DEM is array-like."),
        ("dem1", "dem2", "dem2.transform", "apply", "warns", "DEM .* overrides the given 'transform'"),
        ("None", "None", "None", "fit", "error", "Both DEMs need to be array-like"),
        ("dem1 + np.nan", "dem2", "None", "fit", "error", "'reference_dem' had only NaNs"),
        ("dem1", "dem2 + np.nan", "None", "fit", "error", "'dem_to_be_aligned' had only NaNs"),
    ])
    def test_coreg_raises(_, combination: tuple[str, str, str, str, str, str]) -> None:
        """
        Assert that the expected warnings/errors are triggered under different circumstances.

        The 'combination' param contains this in order:
            1. The reference_dem (will be eval'd)
            2. The dem to be aligned (will be eval'd)
            3. The transform to use (will be eval'd)
            4. Which coreg method to assess
            5. The expected outcome of the test.
            6. The error/warning message (if applicable)
        """
        warnings.simplefilter("error")

        ref_dem, tba_dem, transform, testing_step, result, text = combination
        # Create a small sample-DEM
        dem1 = xdem.DEM.from_array(
            np.arange(25, dtype="int32").reshape(5, 5),
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999
        )
        dem2 = dem1.copy()

        # Evaluate the parametrization (e.g. 'dem2.transform')
        ref_dem, tba_dem, transform = map(eval, (ref_dem, tba_dem, transform))
        
        # Use BiasCorr as a representative example.
        biascorr = xdem.coreg.BiasCorr()

        fit_func = lambda: biascorr.fit(ref_dem, tba_dem, transform=transform)
        apply_func = lambda: biascorr.apply(tba_dem, transform=transform)

        # Try running the methods in order and validate the result.
        for method, method_call in [("fit", fit_func), ("apply", apply_func)]:
            with warnings.catch_warnings():
                if method != testing_step:  # E.g. skip warnings for 'fit' if 'apply' is being tested.
                    warnings.simplefilter("ignore")

                if result == "warns" and testing_step == method:
                    with pytest.warns(UserWarning, match=text):
                        method_call()
                elif result == "error" and testing_step == method:
                    with pytest.raises(ValueError, match=text):
                        method_call()
                else:
                    method_call()

                if testing_step == "fit":   # If we're testing 'fit', 'apply' does not have to be run.
                    return


    def test_coreg_oneliner(_) -> None:
        """Test that a DEM can be coregistered in one line by chaining calls."""
        dem_arr = np.ones((5, 5), dtype="int32")
        dem_arr2 = dem_arr + 1
        transform = rio.transform.from_origin(0, 5, 1, 1)

        dem_arr2_fixed = coreg.BiasCorr().fit(dem_arr, dem_arr2, transform=transform).apply(dem_arr2, transform=transform)

        assert np.array_equal(dem_arr, dem_arr2_fixed)



def test_apply_matrix():
    warnings.simplefilter("error")
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.

    # Test only bias (it should just apply the bias and not make anything else)
    bias = 5
    matrix = np.diag(np.ones(4, float))
    matrix[2, 3] = bias
    transformed_dem = coreg.apply_matrix(ref.data.squeeze(), ref.transform, matrix)
    reverted_dem = transformed_dem - bias

    # Check that the reverted DEM has the exact same values as the initial one
    # (resampling is not an exact science, so this will only apply for bias corrections)
    assert np.nanmedian(reverted_dem) == np.nanmedian(np.asarray(ref.data))

    # Synthesize a shifted and vertically offset DEM
    pixel_shift = 11
    bias = 5
    shifted_dem = ref.data.squeeze().copy()
    shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
    shifted_dem[:, :pixel_shift] = np.nan
    shifted_dem += bias

    matrix = np.diag(np.ones(4, dtype=float))
    matrix[0, 3] = pixel_shift * tba.res[0]
    matrix[2, 3] = -bias

    transformed_dem = coreg.apply_matrix(shifted_dem.data.squeeze(),
                                         ref.transform, matrix, resampling="bilinear")

    # Dilate the mask a bit to ensure that edge pixels are removed.
    transformed_dem_dilated = coreg.apply_matrix(
        shifted_dem.data.squeeze(),
        ref.transform, matrix, resampling="bilinear", dilate_mask=True)
    # Validate that some pixels were removed.
    assert np.count_nonzero(np.isfinite(transformed_dem)) > np.count_nonzero(np.isfinite(transformed_dem_dilated))

    diff = np.asarray(ref.data.squeeze() - transformed_dem)

    # Check that the median is very close to zero
    assert np.abs(np.nanmedian(diff)) < 0.01
    # Check that the NMAD is low
    assert spatial_tools.nmad(diff) < 0.01

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
    centroid = [np.mean([ref.bounds.left, ref.bounds.right]), np.mean(
        [ref.bounds.top, ref.bounds.bottom]), ref.data.mean()]
    rotated_dem = coreg.apply_matrix(
        ref.data.squeeze(),
        ref.transform,
        rotation_matrix(rotation),
        centroid=centroid
    )
    # Make sure that the rotated DEM is way off, but is centered around the same approximate point.
    assert np.abs(np.nanmedian(rotated_dem - ref.data.data)) < 1
    assert spatial_tools.nmad(rotated_dem - ref.data.data) > 500

    # Apply a rotation in the opposite direction
    unrotated_dem = coreg.apply_matrix(
        rotated_dem,
        ref.transform,
        rotation_matrix(-rotation * 0.99),
        centroid=centroid
    ) + 4.0  # TODO: Check why the 0.99 rotation and +4 biases were introduced.

    diff = np.asarray(ref.data.squeeze() - unrotated_dem)

    if False:
        import matplotlib.pyplot as plt

        vmin = 0
        vmax = 1500
        extent = (ref.bounds.left, ref.bounds.right, ref.bounds.bottom, ref.bounds.top)
        plot_params = dict(
            extent=extent,
            vmin=vmin,
            vmax=vmax
        )
        plt.figure(figsize=(22, 4), dpi=100)
        plt.subplot(151)
        plt.title("Original")
        plt.imshow(ref.data.squeeze(), **plot_params)
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


def test_warp_dem():
    """Test that the warp_dem function works expectedly."""
    warnings.simplefilter("error")

    small_dem = np.zeros((5, 10), dtype="float32")
    small_transform = rio.transform.from_origin(0, 5, 1, 1)

    source_coords = np.array(
        [
            [0, 0, 0],
            [0, 5, 0],
            [10, 0, 0],
            [10, 5, 0]
        ]
    ).astype(small_dem.dtype)

    dest_coords = source_coords.copy()
    dest_coords[0, 0] = -1e-5

    warped_dem = coreg.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
        trim_border=False
    )
    assert np.nansum(np.abs(warped_dem - small_dem)) < 1e-6

    elev_shift = 5.0
    dest_coords[1, 2] = elev_shift
    warped_dem = coreg.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
    )

    # The warped DEM should have the value 'elev_shift' in the upper left corner.
    assert warped_dem[0, 0] == elev_shift
    # The corner should be zero, so the corner pixel (represents the corner minus resolution / 2) should be close.
    assert warped_dem[-1, -1] < 1.0

    # Synthesise some X/Y/Z coordinates on the DEM.
    source_coords = np.array(
        [
            [0, 0, 200],
            [480, 20, 200],
            [460, 480, 200],
            [10, 460, 200],
            [250, 250, 200],
        ]
    )

    # Copy the source coordinates and apply some shifts
    dest_coords = source_coords.copy()
    # Apply in the X direction
    dest_coords[0, 0] += 20
    dest_coords[1, 0] += 7
    dest_coords[2, 0] += 10
    dest_coords[3, 0] += 5

    # Apply in the Y direction
    dest_coords[4, 1] += 5

    # Apply in the Z direction
    dest_coords[3, 2] += 5
    test_shift = 6   # This shift will be validated below
    dest_coords[4, 2] += test_shift

    # Generate a semi-random DEM
    transform = rio.transform.from_origin(0, 500, 1, 1)
    shape = (500, 550)
    dem = misc.generate_random_field(shape, 100) * 200 + misc.generate_random_field(shape, 10) * 50

    # Warp the DEM using the source-destination coordinates.
    transformed_dem = coreg.warp_dem(
        dem=dem,
        transform=transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear"
    )

    # Try to undo the warp by reversing the source-destination coordinates.
    untransformed_dem = coreg.warp_dem(
        dem=transformed_dem,
        transform=transform,
        source_coords=dest_coords,
        destination_coords=source_coords,
        resampling="linear"
    )
    # Validate that the DEM is now more or less the same as the original.
    # Due to the randomness, the threshold is quite high, but would be something like 10+ if it was incorrect.
    assert spatial_tools.nmad(dem - untransformed_dem) < 0.5

    if False:
        import matplotlib.pyplot as plt

        plt.figure(dpi=200)
        plt.subplot(141)

        plt.imshow(dem, vmin=0, vmax=300)
        plt.subplot(142)
        plt.imshow(transformed_dem, vmin=0, vmax=300)
        plt.subplot(143)
        plt.imshow(untransformed_dem, vmin=0, vmax=300)

        plt.subplot(144)
        plt.imshow(dem - untransformed_dem, cmap="coolwarm_r", vmin=-10, vmax=10)
        plt.show()



"""Functions to test the coregistration tools."""
from __future__ import annotations

import copy
import os
import tempfile
import warnings
from typing import Any, Callable

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xdem
    from xdem import coreg, examples, misc, spatialstats
    from xdem._typing import NDArrayf
    from xdem.coreg.affine import AffineCoreg
    from xdem.coreg.base import CoregDict, apply_matrix


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
        to_be_aligned_raster = Raster(examples.get_path("longyearbyen_tba_dem"))
        glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


class TestCoregClass:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
        verbose=False,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_from_classmethods(self) -> None:
        warnings.simplefilter("error")

        # Check that the from_matrix function works as expected.
        vshift = 5
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[2, 3] = vshift
        coreg_obj = AffineCoreg.from_matrix(matrix)
        transformed_points = coreg_obj.apply_pts(self.points)
        assert transformed_points[0, 2] == vshift

        # Check that the from_translation function works as expected.
        x_offset = 5
        coreg_obj2 = AffineCoreg.from_translation(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply_pts(self.points)
        assert np.array_equal(self.points[:, 0] + x_offset, transformed_points2[:, 0])

        # Try to make a Coreg object from a nan translation (should fail).
        try:
            AffineCoreg.from_translation(np.nan)
        except ValueError as exception:
            if "non-finite values" not in str(exception):
                raise exception

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], AffineCoreg]) -> None:
        """Test that copying work expectedly (that no attributes still share references)."""
        warnings.simplefilter("error")

        # Create a coreg instance and copy it.
        corr = coreg_class()
        corr_copy = corr.copy()

        # Assign some attributes and metadata after copying, respecting the CoregDict type class
        corr.vshift = 1
        corr._meta["resolution"] = 30
        # Make sure these don't appear in the copy
        assert corr_copy._meta != corr._meta
        assert not hasattr(corr_copy, "vshift")

        # Create a pipeline, add some metadata, and copy it
        pipeline = coreg_class() + coreg_class()
        pipeline.pipeline[0]._meta["vshift"] = 1

        pipeline_copy = pipeline.copy()

        # Add some more metadata after copying (this should not be transferred)
        pipeline._meta["resolution"] = 30
        pipeline_copy.pipeline[0]._meta["offset_north_px"] = 0.5

        assert pipeline._meta != pipeline_copy._meta
        assert pipeline.pipeline[0]._meta != pipeline_copy.pipeline[0]._meta
        assert pipeline_copy.pipeline[0]._meta["vshift"]

    def test_vertical_shift(self) -> None:
        warnings.simplefilter("error")

        # Create a vertical shift correction instance
        vshiftcorr = coreg.VerticalShift()
        # Fit the vertical shift model to the data
        vshiftcorr.fit(**self.fit_params)

        # Check that a vertical shift was found.
        assert vshiftcorr._meta.get("vshift") is not None
        assert vshiftcorr._meta["vshift"] != 0.0

        # Copy the vertical shift to see if it changes in the test (it shouldn't)
        vshift = copy.copy(vshiftcorr._meta["vshift"])

        # Check that the to_matrix function works as it should
        matrix = vshiftcorr.to_matrix()
        assert matrix[2, 3] == vshift, matrix

        # Check that the first z coordinate is now the vertical shift
        assert vshiftcorr.apply_pts(self.points)[0, 2] == vshiftcorr._meta["vshift"]

        # Apply the model to correct the DEM
        tba_unshifted, _ = vshiftcorr.apply(self.tba.data, self.ref.transform, self.ref.crs)

        # Create a new vertical shift correction model
        vshiftcorr2 = coreg.VerticalShift()
        # Check that this is indeed a new object
        assert vshiftcorr is not vshiftcorr2
        # Fit the corrected DEM to see if the vertical shift will be close to or at zero
        vshiftcorr2.fit(
            reference_dem=self.ref.data,
            dem_to_be_aligned=tba_unshifted,
            transform=self.ref.transform,
            crs=self.ref.crs,
            inlier_mask=self.inlier_mask,
        )
        # Test the vertical shift
        newmeta: CoregDict = vshiftcorr2._meta
        new_vshift = newmeta["vshift"]
        assert np.abs(new_vshift) < 0.01

        # Check that the original model's vertical shift has not changed
        # (that the _meta dicts are two different objects)
        assert vshiftcorr._meta["vshift"] == vshift

    def test_all_nans(self) -> None:
        """Check that the coregistration approaches fail gracefully when given only nans."""
        dem1 = np.ones((50, 50), dtype=float)
        dem2 = dem1.copy() + np.nan
        affine = rio.transform.from_origin(0, 0, 1, 1)
        crs = rio.crs.CRS.from_epsg(4326)

        vshiftcorr = coreg.VerticalShift()
        icp = coreg.ICP()

        pytest.raises(ValueError, vshiftcorr.fit, dem1, dem2, transform=affine)
        pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)

        dem2[[3, 20, 40], [2, 21, 41]] = 1.2

        vshiftcorr.fit(dem1, dem2, transform=affine, crs=crs)

        pytest.raises(ValueError, icp.fit, dem1, dem2, transform=affine)

    def test_error_method(self) -> None:
        """Test different error measures."""
        dem1: NDArrayf = np.ones((50, 50)).astype(np.float32)
        # Create a vertically shifted dem
        dem2 = dem1.copy() + 2.0
        affine = rio.transform.from_origin(0, 0, 1, 1)
        crs = rio.crs.CRS.from_epsg(4326)

        vshiftcorr = coreg.VerticalShift()
        # Fit the vertical shift
        vshiftcorr.fit(dem1, dem2, transform=affine, crs=crs)

        # Check that the vertical shift after coregistration is zero
        assert vshiftcorr.error(dem1, dem2, transform=affine, crs=crs, error_type="median") == 0

        # Remove the vertical shift fit and see what happens.
        vshiftcorr._meta["vshift"] = 0
        # Now it should be equal to dem1 - dem2
        assert vshiftcorr.error(dem1, dem2, transform=affine, crs=crs, error_type="median") == -2

        # Create random noise and see if the standard deviation is equal (it should)
        dem3 = dem1.copy() + np.random.random(size=dem1.size).reshape(dem1.shape)
        assert abs(vshiftcorr.error(dem1, dem3, transform=affine, crs=crs, error_type="std") - np.std(dem3)) < 1e-6

    def test_ij_xy(self, i: int = 10, j: int = 20) -> None:
        """
        Test the reversibility of ij2xy and xy2ij, which is important for point co-registration.
        """
        x, y = self.ref.ij2xy(i, j, offset="ul")
        i, j = self.ref.xy2ij(x, y, shift_area_or_point=False)
        assert i == pytest.approx(10)
        assert j == pytest.approx(20)

    def test_coreg_example(self, verbose: bool = False) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.
        """

        # Run co-registration
        nuth_kaab = xdem.coreg.NuthKaab()
        nuth_kaab.fit(self.ref, self.tba, inlier_mask=self.inlier_mask, verbose=verbose)

        # Check the output metadata is always the same
        assert nuth_kaab._meta["offset_east_px"] == pytest.approx(-0.46255704521968716)
        assert nuth_kaab._meta["offset_north_px"] == pytest.approx(-0.13618536563846081)
        assert nuth_kaab._meta["vshift"] == pytest.approx(-1.9815309753424906)

    def test_coreg_example_gradiendescending(
        self, downsampling: int = 10000, samples: int = 20000, inlier_mask: bool = True, verbose: bool = False
    ) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.
        """
        if inlier_mask:
            inlier_mask = self.inlier_mask

        # Run co-registration
        gds = xdem.coreg.GradientDescending(downsampling=downsampling)
        gds.fit_pts(self.ref, self.tba, inlier_mask=inlier_mask, verbose=verbose, samples=samples)
        assert gds._meta["offset_east_px"] == pytest.approx(-0.496000, rel=1e-1, abs=0.1)
        assert gds._meta["offset_north_px"] == pytest.approx(-0.1875, rel=1e-1, abs=0.1)
        assert gds._meta["vshift"] == pytest.approx(-1.8730, rel=1e-1)

    def test_coreg_example_shift_test(
        self,
        shift_px: tuple[float, float] = (1, 1),
        verbose: bool = False,
        coregs: tuple[str, ...] = ("NuthKaab", "GradientDescending", "NuthKaab_pts"),
        downsampling: int = 10000,
    ) -> None:
        """
        For comparison of coreg algorithms:
        Shift a ref_dem on purpose, e.g. shift_px = (1,1), and then apply coreg to shift it back.
        """
        warnings.simplefilter("error")

        res = self.ref.res[0]

        # Shift DEM by shift_px
        shifted_ref = self.ref.copy()
        shifted_ref.shift(shift_px[0] * res, shift_px[1] * res)

        for cor in coregs:
            # Do coreg on shifted DEM
            if cor == "NuthKaab":
                print("\n(1) NuthKaab")
                nuth_kaab = xdem.coreg.NuthKaab()
                nuth_kaab.fit(shifted_ref, self.ref, inlier_mask=self.inlier_mask, verbose=verbose)
                assert nuth_kaab._meta["offset_east_px"] == pytest.approx(-shift_px[0], rel=1e-2)
                assert nuth_kaab._meta["offset_north_px"] == pytest.approx(-shift_px[1], rel=1e-2)

            if cor == "GradientDescending":
                print("\n(2) GradientDescending")
                gds = xdem.coreg.GradientDescending(downsampling=downsampling)
                gds.fit_pts(shifted_ref, self.ref, inlier_mask=self.inlier_mask, verbose=verbose)
                assert gds._meta["offset_east_px"] == pytest.approx(-shift_px[0], rel=1e-2)
                assert gds._meta["offset_north_px"] == pytest.approx(-shift_px[1], rel=1e-2)

            if cor == "NuthKaab_pts":
                print("\n(3) NuthKaab running on pts_fit")
                nuth_kaab = xdem.coreg.NuthKaab()
                nuth_kaab.fit_pts(shifted_ref, self.ref, inlier_mask=self.inlier_mask, verbose=verbose)
                assert nuth_kaab._meta["offset_east_px"] == pytest.approx(-shift_px[0], rel=1e-2)
                assert nuth_kaab._meta["offset_north_px"] == pytest.approx(-shift_px[1], rel=1e-2)

    def test_nuth_kaab(self) -> None:
        warnings.simplefilter("error")

        nuth_kaab = coreg.NuthKaab(max_iterations=10)

        # Synthesize a shifted and vertically offset DEM
        pixel_shift = 2
        vshift = 5
        shifted_dem = self.ref.data.squeeze().copy()
        shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
        shifted_dem[:, :pixel_shift] = np.nan
        shifted_dem += vshift

        # Fit the synthesized shifted DEM to the original
        nuth_kaab.fit(
            self.ref.data.squeeze(),
            shifted_dem,
            transform=self.ref.transform,
            crs=self.ref.crs,
            verbose=self.fit_params["verbose"],
        )

        # Make sure that the estimated offsets are similar to what was synthesized.
        assert nuth_kaab._meta["offset_east_px"] == pytest.approx(pixel_shift, abs=0.03)
        assert nuth_kaab._meta["offset_north_px"] == pytest.approx(0, abs=0.03)
        assert nuth_kaab._meta["vshift"] == pytest.approx(-vshift, 0.03)

        # Apply the estimated shift to "revert the DEM" to its original state.
        unshifted_dem, _ = nuth_kaab.apply(shifted_dem, transform=self.ref.transform, crs=self.ref.crs)
        # Measure the difference (should be more or less zero)
        diff = self.ref.data.squeeze() - unshifted_dem
        diff = diff.compressed()  # turn into a 1D array with only unmasked values

        # Check that the median is very close to zero
        assert np.abs(np.median(diff)) < 0.01
        # Check that the RMSE is low
        assert np.sqrt(np.mean(np.square(diff))) < 1

        # Transform some arbitrary points.
        transformed_points = nuth_kaab.apply_pts(self.points)

        # Check that the x shift is close to the pixel_shift * image resolution
        assert abs((transformed_points[0, 0] - self.points[0, 0]) - pixel_shift * self.ref.res[0]) < 0.1
        # Check that the z shift is close to the original vertical shift.
        assert abs((transformed_points[0, 2] - self.points[0, 2]) + vshift) < 0.1

    def test_deramping(self) -> None:
        warnings.simplefilter("error")

        # Try a 1st degree deramping.
        deramp = coreg.Tilt()

        # Fit the data
        deramp.fit(**self.fit_params)

        # Apply the deramping to a DEM
        deramped_dem = deramp.apply(self.tba)

        # Get the periglacial offset after deramping
        periglacial_offset = (self.ref - deramped_dem)[self.inlier_mask]
        # Get the periglacial offset before deramping
        pre_offset = (self.ref - self.tba)[self.inlier_mask]

        # Check that the error improved
        assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

        # Check that the mean periglacial offset is low
        assert np.abs(np.mean(periglacial_offset)) < 1

    def test_icp_opencv(self) -> None:
        warnings.simplefilter("error")

        # Do a fast and dirty 3 iteration ICP just to make sure it doesn't error out.
        icp = coreg.ICP(max_iterations=3)
        icp.fit(**self.fit_params)

        aligned_dem, _ = icp.apply(self.tba.data, self.ref.transform, self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline(self) -> None:
        warnings.simplefilter("error")

        # Create a pipeline from two coreg methods.
        pipeline = coreg.CoregPipeline([coreg.VerticalShift(), coreg.NuthKaab()])
        pipeline.fit(**self.fit_params)

        aligned_dem, _ = pipeline.apply(self.tba.data, self.ref.transform, self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

        # Make a new pipeline with two vertical shift correction approaches.
        pipeline2 = coreg.CoregPipeline([coreg.VerticalShift(), coreg.VerticalShift()])
        # Set both "estimated" vertical shifts to be 1
        pipeline2.pipeline[0]._meta["vshift"] = 1
        pipeline2.pipeline[1]._meta["vshift"] = 1

        # Assert that the combined vertical shift is 2
        assert pipeline2.to_matrix()[2, 3] == 2.0

    def test_coreg_add(self) -> None:
        warnings.simplefilter("error")
        # Test with a vertical shift of 4
        vshift = 4

        vshift1 = coreg.VerticalShift()
        vshift2 = coreg.VerticalShift()

        # Set the vertical shift attribute
        for vshift_corr in (vshift1, vshift2):
            vshift_corr._meta["vshift"] = vshift

        # Add the two coregs and check that the resulting vertical shift is 2* vertical shift
        vshift3 = vshift1 + vshift2
        assert vshift3.to_matrix()[2, 3] == vshift * 2

        # Make sure the correct exception is raised on incorrect additions
        with pytest.raises(ValueError, match="Incompatible add type"):
            vshift1 + 1  # type: ignore

        # Try to add a Coreg step to an already existing CoregPipeline
        vshift4 = vshift3 + vshift1
        assert vshift4.to_matrix()[2, 3] == vshift * 3

        # Try to add two CoregPipelines
        vshift5 = vshift3 + vshift3
        assert vshift5.to_matrix()[2, 3] == vshift * 4

    def test_subsample(self) -> None:
        warnings.simplefilter("error")

        # Test subsampled vertical shift correction
        vshift_sub = coreg.VerticalShift()

        # Fit the vertical shift using 50% of the unmasked data using a fraction
        vshift_sub.fit(**self.fit_params, subsample=0.5)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        vshift_sub.fit(**self.fit_params, subsample=self.tba.data.size // 2)

        # Do full vertical shift corr to compare
        vshift_full = coreg.VerticalShift()
        vshift_full.fit(**self.fit_params)

        # Check that the estimated vertical shifts are similar
        assert abs(vshift_sub._meta["vshift"] - vshift_full._meta["vshift"]) < 0.1

        # Test NuthKaab with subsampling
        nuthkaab_full = coreg.NuthKaab()
        nuthkaab_sub = coreg.NuthKaab()

        # Measure the start and stop time to get the duration
        # start_time = time.time()
        nuthkaab_full.fit(**self.fit_params)
        # icp_full_duration = time.time() - start_time

        # Do the same with 50% subsampling
        # start_time = time.time()
        nuthkaab_sub.fit(**self.fit_params, subsample=0.5)
        # icp_sub_duration = time.time() - start_time

        # Make sure that the subsampling increased performance
        # Temporarily add a fallback assertion that if it's slower, it shouldn't be much slower (2021-05-17).
        # This doesn't work with GitHub's CI, but it works locally. I'm disabling this for now (2021-05-20).
        # assert icp_full_duration > icp_sub_duration or (abs(icp_full_duration - icp_sub_duration) < 1)

        # Calculate the difference in the full vs. subsampled matrices
        matrix_diff = np.abs(nuthkaab_full.to_matrix() - nuthkaab_sub.to_matrix())
        # Check that the x/y/z differences do not exceed 30cm
        assert np.count_nonzero(matrix_diff > 0.5) == 0

        # Test subsampled deramping
        deramp_sub = coreg.Tilt()

        # Fit the bias using 50% of the unmasked data using a fraction
        deramp_sub.fit(**self.fit_params, subsample=0.5)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        deramp_sub.fit(**self.fit_params, subsample=self.tba.data.size // 2)

        # Do full bias corr to compare
        deramp_full = coreg.Tilt()
        deramp_full.fit(**self.fit_params)

        # Check that the estimated biases are similar
        assert deramp_sub._meta["coefficients"] == pytest.approx(deramp_full._meta["coefficients"], rel=1e-1)

    @pytest.mark.parametrize(
        "pipeline", [coreg.VerticalShift(), coreg.VerticalShift() + coreg.NuthKaab()]
    )  # type: ignore
    @pytest.mark.parametrize("subdivision", [4, 10])  # type: ignore
    def test_blockwise_coreg(self, pipeline: AffineCoreg, subdivision: int) -> None:
        warnings.simplefilter("error")

        blockwise = coreg.BlockwiseCoreg(step=pipeline, subdivision=subdivision)

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
            coreg.BlockwiseCoreg(step=coreg.VerticalShift, subdivision=1)  # type: ignore

        # Metadata copying has been an issue. Validate that all chunks have unique ids
        chunk_numbers = [m["i"] for m in blockwise._meta["step_meta"]]
        assert np.unique(chunk_numbers).shape[0] == len(chunk_numbers)

        transformed_dem = blockwise.apply(self.tba)

        ddem_pre = (self.ref - self.tba)[~self.inlier_mask]
        ddem_post = (self.ref - transformed_dem)[~self.inlier_mask]

        # Check that the periglacial difference is lower after coregistration.
        assert abs(np.ma.median(ddem_post)) < abs(np.ma.median(ddem_pre))

        stats = blockwise.stats()

        # Check that nans don't exist (if they do, something has gone very wrong)
        assert np.all(np.isfinite(stats["nmad"]))
        # Check that offsets were actually calculated.
        assert np.sum(np.abs(np.linalg.norm(stats[["x_off", "y_off", "z_off"]], axis=0))) > 0

    def test_blockwise_coreg_large_gaps(self) -> None:
        """Test BlockwiseCoreg when large gaps are encountered, e.g. around the frame of a rotated DEM."""
        warnings.simplefilter("error")
        reference_dem = self.ref.reproject(dst_crs="EPSG:3413", dst_res=self.ref.res, resampling="bilinear")
        dem_to_be_aligned = self.tba.reproject(dst_ref=reference_dem, resampling="bilinear")

        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 64, warn_failures=False)

        # This should not fail or trigger warnings as warn_failures is False
        blockwise.fit(reference_dem, dem_to_be_aligned)

        stats = blockwise.stats()

        # We expect holes in the blockwise coregistration, so there should not be 64 "successful" blocks.
        assert stats.shape[0] < 64

        # Statistics are only calculated on finite values, so all of these should be finite as well.
        assert np.all(np.isfinite(stats))

        # Copy the TBA DEM and set a square portion to nodata
        tba = self.tba.copy()
        mask = np.zeros(np.shape(tba.data), dtype=bool)
        mask[450:500, 450:500] = True
        tba.set_mask(mask=mask)

        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), 8, warn_failures=False)

        # Align the DEM and apply the blockwise to a zero-array (to get the zshift)
        aligned = blockwise.fit(self.ref, tba).apply(tba)
        zshift, _ = blockwise.apply(np.zeros_like(tba.data), transform=tba.transform, crs=tba.crs)

        # Validate that the zshift is not something crazy high and that no negative values exist in the data.
        assert np.nanmax(np.abs(zshift)) < 50
        assert np.count_nonzero(aligned.data.compressed() < -50) == 0

        # Check that coregistration improved the alignment
        ddem_post = (aligned - self.ref).data.compressed()
        ddem_pre = (tba - self.ref).data.compressed()
        assert abs(np.nanmedian(ddem_pre)) > abs(np.nanmedian(ddem_post))
        assert np.nanstd(ddem_pre) > np.nanstd(ddem_post)

    def test_coreg_raster_and_ndarray_args(self) -> None:

        # Create a small sample-DEM
        dem1 = xdem.DEM.from_array(
            np.arange(25, dtype="int32").reshape(5, 5),
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        # Assign a funny value to one particular pixel. This is to validate that reprojection works perfectly.
        dem1.data[1, 1] = 100

        # Translate the DEM 1 "meter" right and add a vertical shift
        dem2 = dem1.reproject(dst_bounds=rio.coords.BoundingBox(1, 0, 6, 5), silent=True)
        dem2 += 1

        # Create a vertical shift correction for Rasters ("_r") and for arrays ("_a")
        vshiftcorr_r = coreg.VerticalShift()
        vshiftcorr_a = vshiftcorr_r.copy()

        # Fit the data
        vshiftcorr_r.fit(reference_dem=dem1, dem_to_be_aligned=dem2)
        vshiftcorr_a.fit(
            reference_dem=dem1.data,
            dem_to_be_aligned=dem2.reproject(dem1, silent=True).data,
            transform=dem1.transform,
            crs=dem1.crs,
        )

        # Validate that they ended up giving the same result.
        assert vshiftcorr_r._meta["vshift"] == vshiftcorr_a._meta["vshift"]

        # De-shift dem2
        dem2_r = vshiftcorr_r.apply(dem2)
        dem2_a, _ = vshiftcorr_a.apply(dem2.data, dem2.transform, dem2.crs)

        # Validate that the return formats were the expected ones, and that they are equal.
        # Issue - dem2_a does not have the same shape, the first dimension is being squeezed
        # TODO - Fix coreg.apply?
        assert isinstance(dem2_r, xdem.DEM)
        assert isinstance(dem2_a, np.ma.masked_array)
        assert np.ma.allequal(dem2_r.data.squeeze(), dem2_a)

        # If apply on a masked_array was given without a transform, it should fail.
        with pytest.raises(ValueError, match="'transform' must be given"):
            vshiftcorr_a.apply(dem2.data, crs=dem2.crs)

        # If apply on a masked_array was given without a crs, it should fail.
        with pytest.raises(ValueError, match="'crs' must be given"):
            vshiftcorr_a.apply(dem2.data, transform=dem2.transform)

        # If transform provided with input Raster, should raise a warning
        with pytest.warns(UserWarning, match="DEM .* overrides the given 'transform'"):
            vshiftcorr_a.apply(dem2, transform=dem2.transform)

        # If crs provided with input Raster, should raise a warning
        with pytest.warns(UserWarning, match="DEM .* overrides the given 'crs'"):
            vshiftcorr_a.apply(dem2, crs=dem2.crs)

    # Inputs contain: coregistration method, is implemented, comparison is "strict" or "approx"
    @pytest.mark.parametrize(
        "inputs",
        [
            [xdem.coreg.VerticalShift(), True, "strict"],
            [xdem.coreg.Tilt(), True, "strict"],
            [xdem.coreg.NuthKaab(), True, "approx"],
            [xdem.coreg.NuthKaab() + xdem.coreg.Tilt(), True, "approx"],
            [xdem.coreg.BlockwiseCoreg(step=xdem.coreg.NuthKaab(), subdivision=16), False, ""],
            [xdem.coreg.ICP(), False, ""],
        ],
    )  # type: ignore
    def test_apply_resample(self, inputs: list[Any]) -> None:
        """
        Test that the option resample of coreg.apply works as expected.
        For vertical correction only (VerticalShift, Deramp...), option True or False should yield same results.
        For horizontal shifts (NuthKaab etc), georef should differ, but DEMs should be the same after resampling.
        For others, the method is not implemented.
        """
        # Get test inputs
        coreg_method, is_implemented, comp = inputs
        ref_dem, tba_dem, outlines = load_examples()  # Load example reference, to-be-aligned and mask.

        # Prepare coreg
        inlier_mask = ~outlines.create_mask(ref_dem)
        coreg_method.fit(tba_dem, ref_dem, inlier_mask=inlier_mask)

        # If not implemented, should raise an error
        if not is_implemented:
            with pytest.raises(NotImplementedError, match="Option `resample=False` not implemented for coreg method *"):
                dem_coreg_noresample = coreg_method.apply(tba_dem, resample=False)
            return
        else:
            dem_coreg_resample = coreg_method.apply(tba_dem)
            dem_coreg_noresample = coreg_method.apply(tba_dem, resample=False)

        if comp == "strict":
            # Both methods should yield the exact same output
            assert dem_coreg_resample == dem_coreg_noresample
        elif comp == "approx":
            # The georef should be different
            assert dem_coreg_noresample.transform != dem_coreg_resample.transform

            # After resampling, both results should be almost equal
            dem_final = dem_coreg_noresample.reproject(dem_coreg_resample)
            diff = dem_final - dem_coreg_resample
            assert np.all(np.abs(diff.data) == pytest.approx(0, abs=1e-2))
            # assert np.count_nonzero(diff.data) == 0

        # Test it works with different resampling algorithms
        dem_coreg_resample = coreg_method.apply(tba_dem, resample=True, resampling=rio.warp.Resampling.nearest)
        dem_coreg_resample = coreg_method.apply(tba_dem, resample=True, resampling=rio.warp.Resampling.cubic)
        with pytest.raises(ValueError, match="`resampling` must be a rio.warp.Resampling algorithm"):
            dem_coreg_resample = coreg_method.apply(tba_dem, resample=True, resampling=None)

    @pytest.mark.parametrize(
        "combination",
        [
            ("dem1", "dem2", "None", "None", "fit", "passes", ""),
            ("dem1", "dem2", "None", "None", "apply", "passes", ""),
            ("dem1.data", "dem2.data", "dem1.transform", "dem1.crs", "fit", "passes", ""),
            ("dem1.data", "dem2.data", "dem1.transform", "dem1.crs", "apply", "passes", ""),
            (
                "dem1",
                "dem2.data",
                "dem1.transform",
                "dem1.crs",
                "fit",
                "warns",
                "'reference_dem' .* overrides the given 'transform'",
            ),
            ("dem1.data", "dem2", "dem1.transform", "None", "fit", "warns", "'dem_to_be_aligned' .* overrides .*"),
            (
                "dem1.data",
                "dem2.data",
                "None",
                "dem1.crs",
                "fit",
                "error",
                "'transform' must be given if both DEMs are array-like.",
            ),
            (
                "dem1.data",
                "dem2.data",
                "dem1.transform",
                "None",
                "fit",
                "error",
                "'crs' must be given if both DEMs are array-like.",
            ),
            (
                "dem1",
                "dem2.data",
                "None",
                "dem1.crs",
                "apply",
                "error",
                "'transform' must be given if DEM is array-like.",
            ),
            (
                "dem1",
                "dem2.data",
                "dem1.transform",
                "None",
                "apply",
                "error",
                "'crs' must be given if DEM is array-like.",
            ),
            ("dem1", "dem2", "dem2.transform", "None", "apply", "warns", "DEM .* overrides the given 'transform'"),
            ("None", "None", "None", "None", "fit", "error", "Both DEMs need to be array-like"),
            ("dem1 + np.nan", "dem2", "None", "None", "fit", "error", "'reference_dem' had only NaNs"),
            ("dem1", "dem2 + np.nan", "None", "None", "fit", "error", "'dem_to_be_aligned' had only NaNs"),
        ],
    )  # type: ignore
    def test_coreg_raises(self, combination: tuple[str, str, str, str, str, str, str]) -> None:
        """
        Assert that the expected warnings/errors are triggered under different circumstances.

        The 'combination' param contains this in order:
            1. The reference_dem (will be eval'd)
            2. The dem to be aligned (will be eval'd)
            3. The transform to use (will be eval'd)
            4. The CRS to use (will be eval'd)
            5. Which coreg method to assess
            6. The expected outcome of the test.
            7. The error/warning message (if applicable)
        """
        warnings.simplefilter("error")

        ref_dem, tba_dem, transform, crs, testing_step, result, text = combination

        # Create a small sample-DEM
        dem1 = xdem.DEM.from_array(
            np.arange(25, dtype="float64").reshape(5, 5),
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        dem2 = dem1.copy()  # noqa

        # Evaluate the parametrization (e.g. 'dem2.transform')
        ref_dem, tba_dem, transform, crs = map(eval, (ref_dem, tba_dem, transform, crs))

        # Use VerticalShift as a representative example.
        vshiftcorr = xdem.coreg.VerticalShift()

        def fit_func() -> AffineCoreg:
            return vshiftcorr.fit(ref_dem, tba_dem, transform=transform, crs=crs)

        def apply_func() -> NDArrayf:
            return vshiftcorr.apply(tba_dem, transform=transform, crs=crs)

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

                if testing_step == "fit":  # If we're testing 'fit', 'apply' does not have to be run.
                    return

    def test_coreg_oneliner(self) -> None:
        """Test that a DEM can be coregistered in one line by chaining calls."""
        dem_arr = np.ones((5, 5), dtype="int32")
        dem_arr2 = dem_arr + 1
        transform = rio.transform.from_origin(0, 5, 1, 1)
        crs = rio.crs.CRS.from_epsg(4326)

        dem_arr2_fixed, _ = (
            coreg.VerticalShift()
            .fit(dem_arr, dem_arr2, transform=transform, crs=crs)
            .apply(dem_arr2, transform=transform, crs=crs)
        )

        assert np.array_equal(dem_arr, dem_arr2_fixed)


def test_apply_matrix() -> None:
    warnings.simplefilter("error")
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    ref_arr = gu.raster.get_array_and_mask(ref)[0]

    # Test only vertical shift (it should just apply the vertical shift and not make anything else)
    vshift = 5
    matrix = np.diag(np.ones(4, float))
    matrix[2, 3] = vshift
    transformed_dem = apply_matrix(ref_arr, ref.transform, matrix)
    reverted_dem = transformed_dem - vshift

    # Check that the reverted DEM has the exact same values as the initial one
    # (resampling is not an exact science, so this will only apply for vertical shift corrections)
    assert np.nanmedian(reverted_dem) == np.nanmedian(np.asarray(ref.data))

    # Synthesize a shifted and vertically offset DEM
    pixel_shift = 11
    vshift = 5
    shifted_dem = ref_arr.copy()
    shifted_dem[:, pixel_shift:] = shifted_dem[:, :-pixel_shift]
    shifted_dem[:, :pixel_shift] = np.nan
    shifted_dem += vshift

    matrix = np.diag(np.ones(4, dtype=float))
    matrix[0, 3] = pixel_shift * tba.res[0]
    matrix[2, 3] = -vshift

    transformed_dem = apply_matrix(shifted_dem, ref.transform, matrix, resampling="bilinear")
    diff = np.asarray(ref_arr - transformed_dem)

    # Check that the median is very close to zero
    assert np.abs(np.nanmedian(diff)) < 0.01
    # Check that the NMAD is low
    assert spatialstats.nmad(diff) < 0.01

    def rotation_matrix(rotation: float = 30) -> NDArrayf:
        rotation = np.deg2rad(rotation)
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(rotation), -np.sin(rotation), 0],
                [0, np.sin(rotation), np.cos(rotation), 0],
                [0, 0, 0, 1],
            ]
        )
        return matrix

    rotation = 4
    centroid = (
        np.mean([ref.bounds.left, ref.bounds.right]),
        np.mean([ref.bounds.top, ref.bounds.bottom]),
        ref.data.mean(),
    )
    rotated_dem = apply_matrix(ref.data.squeeze(), ref.transform, rotation_matrix(rotation), centroid=centroid)
    # Make sure that the rotated DEM is way off, but is centered around the same approximate point.
    assert np.abs(np.nanmedian(rotated_dem - ref.data.data)) < 1
    assert spatialstats.nmad(rotated_dem - ref.data.data) > 500

    # Apply a rotation in the opposite direction
    unrotated_dem = (
        apply_matrix(rotated_dem, ref.transform, rotation_matrix(-rotation * 0.99), centroid=centroid) + 4.0
    )  # TODO: Check why the 0.99 rotation and +4 vertical shift were introduced.

    diff = np.asarray(ref.data.squeeze() - unrotated_dem)

    # if False:
    #     import matplotlib.pyplot as plt
    #
    #     vmin = 0
    #     vmax = 1500
    #     extent = (ref.bounds.left, ref.bounds.right, ref.bounds.bottom, ref.bounds.top)
    #     plot_params = dict(
    #         extent=extent,
    #         vmin=vmin,
    #         vmax=vmax
    #     )
    #     plt.figure(figsize=(22, 4), dpi=100)
    #     plt.subplot(151)
    #     plt.title("Original")
    #     plt.imshow(ref.data.squeeze(), **plot_params)
    #     plt.xlim(*extent[:2])
    #     plt.ylim(*extent[2:])
    #     plt.subplot(152)
    #     plt.title(f"Rotated {rotation} degrees")
    #     plt.imshow(rotated_dem, **plot_params)
    #     plt.xlim(*extent[:2])
    #     plt.ylim(*extent[2:])
    #     plt.subplot(153)
    #     plt.title(f"De-rotated {-rotation} degrees")
    #     plt.imshow(unrotated_dem, **plot_params)
    #     plt.xlim(*extent[:2])
    #     plt.ylim(*extent[2:])
    #     plt.subplot(154)
    #     plt.title("Original vs. de-rotated")
    #     plt.imshow(diff, extent=extent, vmin=-10, vmax=10, cmap="coolwarm_r")
    #     plt.colorbar()
    #     plt.xlim(*extent[:2])
    #     plt.ylim(*extent[2:])
    #     plt.subplot(155)
    #     plt.title("Original vs. de-rotated")
    #     plt.hist(diff[np.isfinite(diff)], bins=np.linspace(-10, 10, 100))
    #     plt.tight_layout(w_pad=0.05)
    #     plt.show()

    # Check that the median is very close to zero
    assert np.abs(np.nanmedian(diff)) < 0.5
    # Check that the NMAD is low
    assert spatialstats.nmad(diff) < 5
    print(np.nanmedian(diff), spatialstats.nmad(diff))


def test_warp_dem() -> None:
    """Test that the warp_dem function works expectedly."""
    warnings.simplefilter("error")

    small_dem = np.zeros((5, 10), dtype="float32")
    small_transform = rio.transform.from_origin(0, 5, 1, 1)

    source_coords = np.array([[0, 0, 0], [0, 5, 0], [10, 0, 0], [10, 5, 0]]).astype(small_dem.dtype)

    dest_coords = source_coords.copy()
    dest_coords[0, 0] = -1e-5

    warped_dem = coreg.base.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
        trim_border=False,
    )
    assert np.nansum(np.abs(warped_dem - small_dem)) < 1e-6

    elev_shift = 5.0
    dest_coords[1, 2] = elev_shift
    warped_dem = coreg.base.warp_dem(
        dem=small_dem,
        transform=small_transform,
        source_coords=source_coords,
        destination_coords=dest_coords,
        resampling="linear",
    )

    # The warped DEM should have the value 'elev_shift' in the upper left corner.
    assert warped_dem[0, 0] == elev_shift
    # The corner should be zero, so the corner pixel (represents the corner minus resolution / 2) should be close.
    # We select the pixel before the corner (-2 in X-axis) to avoid the NaN propagation on the bottom row.
    assert warped_dem[-2, -1] < 1

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
    test_shift = 6  # This shift will be validated below
    dest_coords[4, 2] += test_shift

    # Generate a semi-random DEM
    transform = rio.transform.from_origin(0, 500, 1, 1)
    shape = (500, 550)
    dem = misc.generate_random_field(shape, 100) * 200 + misc.generate_random_field(shape, 10) * 50

    # Warp the DEM using the source-destination coordinates.
    transformed_dem = coreg.base.warp_dem(
        dem=dem, transform=transform, source_coords=source_coords, destination_coords=dest_coords, resampling="linear"
    )

    # Try to undo the warp by reversing the source-destination coordinates.
    untransformed_dem = coreg.base.warp_dem(
        dem=transformed_dem,
        transform=transform,
        source_coords=dest_coords,
        destination_coords=source_coords,
        resampling="linear",
    )
    # Validate that the DEM is now more or less the same as the original.
    # Due to the randomness, the threshold is quite high, but would be something like 10+ if it was incorrect.
    assert spatialstats.nmad(dem - untransformed_dem) < 0.5

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


def test_create_inlier_mask() -> None:
    """Test that the create_inlier_mask function works expectedly."""
    warnings.simplefilter("error")

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and outlines

    # - Assert that without filtering create_inlier_mask behaves as if calling Vector.create_mask - #
    # Masking inside - using Vector
    inlier_mask_comp = ~outlines.create_mask(ref, as_array=True)
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba,
        ref,
        [
            outlines,
        ],
        filtering=False,
    )
    assert np.all(inlier_mask_comp == inlier_mask)

    # Masking inside - using string
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba,
        ref,
        [
            outlines.name,
        ],
        filtering=False,
    )
    assert np.all(inlier_mask_comp == inlier_mask)

    # Masking outside - using Vector
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba,
        ref,
        [
            outlines,
        ],
        inout=[
            -1,
        ],
        filtering=False,
    )
    assert np.all(~inlier_mask_comp == inlier_mask)

    # Masking outside - using string
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba,
        ref,
        [
            outlines.name,
        ],
        inout=[-1],
        filtering=False,
    )
    assert np.all(~inlier_mask_comp == inlier_mask)

    # - Test filtering options only - #
    # Test the slope filter only
    slope = xdem.terrain.slope(ref)
    slope_lim = [1, 50]
    inlier_mask_comp2 = np.ones(tba.data.shape, dtype=bool)
    inlier_mask_comp2[slope.data < slope_lim[0]] = False
    inlier_mask_comp2[slope.data > slope_lim[1]] = False
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba, ref, filtering=True, slope_lim=slope_lim, nmad_factor=np.inf
    )
    assert np.all(inlier_mask == inlier_mask_comp2)

    # Test the nmad_factor filter only
    nmad_factor = 3
    ddem = tba - ref
    inlier_mask_comp3 = (np.abs(ddem.data - np.median(ddem)) < nmad_factor * xdem.spatialstats.nmad(ddem)).filled(False)
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba, ref, filtering=True, slope_lim=[0, 90], nmad_factor=nmad_factor
    )
    assert np.all(inlier_mask == inlier_mask_comp3)

    # Test the sum of both
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba, ref, shp_list=[], inout=[], filtering=True, slope_lim=slope_lim, nmad_factor=nmad_factor
    )
    inlier_mask_all = inlier_mask_comp2 & inlier_mask_comp3
    assert np.all(inlier_mask == inlier_mask_all)

    # Test the dh_max filter only
    dh_max = 200
    inlier_mask_comp4 = (np.abs(ddem.data) < dh_max).filled(False)
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba, ref, filtering=True, slope_lim=[0, 90], nmad_factor=np.inf, dh_max=dh_max
    )
    assert np.all(inlier_mask == inlier_mask_comp4)

    # - Test the sum of outlines + dh_max + slope - #
    # nmad_factor will have a different behavior because it calculates nmad from the inliers of previous filters
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
        tba,
        ref,
        shp_list=[
            outlines,
        ],
        inout=[
            -1,
        ],
        filtering=True,
        slope_lim=slope_lim,
        nmad_factor=np.inf,
        dh_max=dh_max,
    )
    inlier_mask_all = ~inlier_mask_comp & inlier_mask_comp2 & inlier_mask_comp4
    assert np.all(inlier_mask == inlier_mask_all)

    # - Test that proper errors are raised for wrong inputs - #
    with pytest.raises(ValueError, match="`shp_list` must be a list/tuple"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, shp_list=outlines)

    with pytest.raises(ValueError, match="`shp_list` must be a list/tuple of strings or geoutils.Vector instance"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, shp_list=[1])

    with pytest.raises(ValueError, match="`inout` must be a list/tuple"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
            tba,
            ref,
            shp_list=[
                outlines,
            ],
            inout=1,  # type: ignore
        )

    with pytest.raises(ValueError, match="`inout` must contain only 1 and -1"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
            tba,
            ref,
            shp_list=[
                outlines,
            ],
            inout=[
                0,
            ],
        )

    with pytest.raises(ValueError, match="`inout` must be of same length as shp"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(
            tba,
            ref,
            shp_list=[
                outlines,
            ],
            inout=[1, 1],
        )

    with pytest.raises(ValueError, match="`slope_lim` must be a list/tuple"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, filtering=True, slope_lim=1)  # type: ignore

    with pytest.raises(ValueError, match="`slope_lim` must contain 2 elements"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, filtering=True, slope_lim=[30])

    with pytest.raises(ValueError, match=r"`slope_lim` must be a tuple/list of 2 elements in the range \[0-90\]"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, filtering=True, slope_lim=[-1, 40])

    with pytest.raises(ValueError, match=r"`slope_lim` must be a tuple/list of 2 elements in the range \[0-90\]"):
        inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba, ref, filtering=True, slope_lim=[1, 120])


def test_dem_coregistration() -> None:
    """
    Test that the dem_coregistration function works expectedly.
    Tests the features that are specific to dem_coregistration.
    For example, many features are tested in create_inlier_mask, so not tested again here.
    TODO: Add DEMs with different projection/grid to test that regridding works as expected.
    """
    # Load example reference, to-be-aligned and outlines
    ref_dem, tba_dem, outlines = load_examples()

    # - Check that it works with default parameters - #
    dem_coreg, coreg_method, coreg_stats, inlier_mask = xdem.coreg.dem_coregistration(tba_dem, ref_dem)

    # Assert that outputs have expected format
    assert isinstance(dem_coreg, xdem.DEM)
    assert isinstance(coreg_method, xdem.coreg.Coreg)
    assert isinstance(coreg_stats, pd.DataFrame)

    # Assert that default coreg_method is as expected
    assert hasattr(coreg_method, "pipeline")
    assert isinstance(coreg_method.pipeline[0], xdem.coreg.NuthKaab)
    assert isinstance(coreg_method.pipeline[1], xdem.coreg.VerticalShift)

    # The result should be similar to applying the same coreg by hand with:
    # - DEMs converted to Float32
    # - default inlier_mask
    # - no resampling
    coreg_method_ref = xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift()
    inlier_mask = xdem.coreg.pipelines.create_inlier_mask(tba_dem, ref_dem)
    coreg_method_ref.fit(ref_dem.astype("float32"), tba_dem.astype("float32"), inlier_mask=inlier_mask)
    dem_coreg_ref = coreg_method_ref.apply(tba_dem, resample=False)
    assert dem_coreg == dem_coreg_ref

    # Assert that coregistration improved the residuals
    assert abs(coreg_stats["med_orig"].values) > abs(coreg_stats["med_coreg"].values)
    assert coreg_stats["nmad_orig"].values > coreg_stats["nmad_coreg"].values

    # - Check some alternative arguments - #
    # Test with filename instead of DEMs
    dem_coreg2, _, _, _ = xdem.coreg.dem_coregistration(tba_dem.filename, ref_dem.filename)
    assert dem_coreg2 == dem_coreg

    # Test saving to file (mode = "w" is necessary to work on Windows)
    outfile = tempfile.NamedTemporaryFile(suffix=".tif", mode="w", delete=False)
    xdem.coreg.dem_coregistration(tba_dem, ref_dem, out_dem_path=outfile.name)
    dem_coreg2 = xdem.DEM(outfile.name)
    assert dem_coreg2 == dem_coreg
    outfile.close()

    # Test that shapefile is properly taken into account - inlier_mask should be False inside outlines
    # Need to use resample=True, to ensure that dem_coreg has same georef as inlier_mask
    dem_coreg, coreg_method, coreg_stats, inlier_mask = xdem.coreg.dem_coregistration(
        tba_dem,
        ref_dem,
        shp_list=[
            outlines,
        ],
        resample=True,
    )
    gl_mask = outlines.create_mask(dem_coreg, as_array=True)
    assert np.all(~inlier_mask[gl_mask])

    # Testing with plot
    out_fig = tempfile.NamedTemporaryFile(suffix=".png", mode="w", delete=False)
    assert os.path.getsize(out_fig.name) == 0
    xdem.coreg.dem_coregistration(tba_dem, ref_dem, plot=True, out_fig=out_fig.name)
    assert os.path.getsize(out_fig.name) > 0
    out_fig.close()

    # Testing different coreg method
    dem_coreg, coreg_method, coreg_stats, inlier_mask = xdem.coreg.dem_coregistration(
        tba_dem, ref_dem, coreg_method=xdem.coreg.Tilt()
    )
    assert isinstance(coreg_method, xdem.coreg.Tilt)
    assert abs(coreg_stats["med_orig"].values) > abs(coreg_stats["med_coreg"].values)
    assert coreg_stats["nmad_orig"].values > coreg_stats["nmad_coreg"].values

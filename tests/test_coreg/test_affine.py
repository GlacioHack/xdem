"""Functions to test the affine coregistrations."""

from __future__ import annotations

import os.path
import re
import warnings

import geopandas as gpd
import geoutils
import numpy as np
import pytest
import rasterio as rio
import scipy.optimize
from geoutils import Raster, Vector
from geoutils.raster import RasterType
from geoutils.raster.geotransformations import _translate
from scipy.ndimage import binary_dilation

from xdem import coreg, examples
from xdem.coreg.affine import (
    AffineCoreg,
    _reproject_horizontal_shift_samecrs,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)


def load_examples(crop: bool = True) -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    if crop:
        # Crop to smaller extents for test speed
        res = reference_dem.res
        crop_geom = (
            reference_dem.bounds.left,
            reference_dem.bounds.bottom,
            reference_dem.bounds.left + res[0] * 300,
            reference_dem.bounds.bottom + res[1] * 300,
        )
        reference_dem = reference_dem.crop(crop_geom)
        to_be_aligned_dem = to_be_aligned_dem.crop(crop_geom)

    return reference_dem, to_be_aligned_dem, glacier_mask


class TestAffineCoreg:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    # Check all point-raster possibilities supported
    # Use the reference DEM for both, it will be artificially misaligned during tests
    # Raster-Raster
    fit_args_rst_rst = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # Convert DEMs to points with a bit of subsampling for speed-up
    ref_pts = ref.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
    tba_pts = ref.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds

    # Raster-Point
    fit_args_rst_pts = dict(reference_elev=ref, to_be_aligned_elev=tba_pts, inlier_mask=inlier_mask)

    # Point-Raster
    fit_args_pts_rst = dict(reference_elev=ref_pts, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    all_fit_args = [fit_args_rst_rst, fit_args_rst_pts, fit_args_pts_rst]

    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    @pytest.mark.parametrize(
        "xoff_yoff",
        [(ref.res[0], ref.res[1]), (10 * ref.res[0], 10 * ref.res[1]), (-1.2 * ref.res[0], -1.2 * ref.res[1])],
    )  # type: ignore
    def test_reproject_horizontal_shift_samecrs__gdal(self, xoff_yoff: tuple[float, float], get_test_data_path) -> None:
        """Check that the same-CRS reprojection based on SciPy (replacing Rasterio due to subpixel errors)
        is accurate by comparing to GDAL."""

        ref = load_examples(crop=False)[0]

        # Reproject with SciPy
        xoff, yoff = xoff_yoff
        dst_transform = _translate(transform=ref.transform, xoff=xoff, yoff=yoff, distance_unit="georeferenced")
        output = _reproject_horizontal_shift_samecrs(
            raster_arr=ref.data, src_transform=ref.transform, dst_transform=dst_transform
        )

        # Reproject with GDAL
        path_output2 = get_test_data_path(os.path.join("gdal", f"shifted_reprojected_xoff{xoff}_yoff{yoff}.tif"))
        output2 = Raster(path_output2).data.data

        # Reproject and NaN propagation is exactly the same for shifts that are a multiple of pixel resolution
        if xoff % ref.res[0] == 0 and yoff % ref.res[1] == 0:
            assert np.array_equal(output, output2, equal_nan=True)

        # For sub-pixel shifts, NaN propagation differs slightly (within 1 pixel) but the resampled values are the same
        else:
            # Verify all close values
            valids = np.logical_and(np.isfinite(output), np.isfinite(output2))
            # Max relative tolerance that is reached just for a small % of points
            assert np.allclose(output[valids], output2[valids], rtol=10e-2)
            # Median precision is much higher
            # (here absolute, equivalent to around 10e-7 relative as raster values are in the 1000s)
            assert np.nanmedian(np.abs(output[valids] - output2[valids])) < 0.0001

            # NaNs differ by 1 pixel max, i.e. the mask dilated by one includes the other
            mask_nans = ~np.isfinite(output)
            mask_dilated_plus_one = binary_dilation(mask_nans, iterations=1).astype(bool)
            assert np.array_equal(np.logical_or(mask_dilated_plus_one, ~np.isfinite(output2)), mask_dilated_plus_one)

    def test_from_classmethods(self) -> None:

        # Check that the from_matrix function works as expected.
        vshift = 5
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[2, 3] = vshift
        coreg_obj = AffineCoreg.from_matrix(matrix)
        transformed_points = coreg_obj.apply(self.points)
        assert all(transformed_points["z"].values == vshift)

        # Check that the from_translation function works as expected.
        x_offset = 5
        coreg_obj2 = AffineCoreg.from_translations(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply(self.points)
        assert np.array_equal(self.points.geometry.x.values + x_offset, transformed_points2.geometry.x.values)

        # Try to make a Coreg object from a nan translation (should fail).
        try:
            AffineCoreg.from_translations(np.nan)
        except ValueError as exception:
            if "non-finite values" not in str(exception):
                raise exception

    def test_raise_all_nans(self) -> None:
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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("shifts", [(20, 5, 2), (-50, 100, 2)])  # type: ignore
    @pytest.mark.parametrize("coreg_method", [coreg.NuthKaab, coreg.DhMinimize, coreg.ICP, coreg.LZD])  # type: ignore
    def test_coreg_translations__synthetic(self, fit_args, shifts, coreg_method) -> None:
        """
        Test the horizontal/vertical shift coregistrations with synthetic shifted data. These tests include NuthKaab,
        ICP and DhMinimize.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.

        We verify that the shifts found by the coregistration are within 1% of the synthetic shifts with opposite sign
        of the ones introduced, and that applying the coregistration to the shifted elevations corrects more than
        99% of the variance from the initial elevation differences (hence, that the direction of coregistration has
        to be the right one; and that there is no other errors introduced in the process).
        """

        warnings.filterwarnings("ignore", message="Covariance of the parameters*")

        horizontal_coreg = coreg_method()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        # Create synthetic translation from the reference DEM
        ref = self.ref
        ref_shifted = ref.translate(shifts[0], shifts[1]) + shifts[2]
        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_shifted = ref_shifted.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted

        # Run coregistration
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=subsample_size, random_state=42)

        # Check all fit parameters are the opposite of those used above, within a relative 1% (10% for ICP)
        fit_shifts = [-horizontal_coreg.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]

        # ICP can be less precise than other methods
        rtol = 10e-2 if coreg_method == coreg.NuthKaab else 10e-1
        assert np.allclose(fit_shifts, shifts, rtol=rtol)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_shifted.geometry.x.values, ref_shifted.geometry.y.values)) - ref_shifted["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_shifted.reproject(ref)
            dh = ref - coreg_elev.reproject(ref)

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Check applying the coregistration removes 99% of the variance (95% for ICP)
        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        tol = 0.01 if coreg_method == coreg.NuthKaab else 0.05
        assert np.nanvar(dh / np.nanstd(init_dh)) < tol

    @pytest.mark.parametrize(
        "coreg_method__shift",
        [
            (coreg.NuthKaab, (9.202739, 2.735573, -1.97733)),
            (coreg.DhMinimize, (10.0850892, 2.898172, -1.943001)),
            (coreg.LZD, (9.969819, 2.140150, -1.9257709)),
            (coreg.ICP, (5.417970, 1.1282436, -2.0662609)),
        ],
    )  # type: ignore
    def test_coreg_translations__example(
        self, coreg_method__shift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the translation co-registration outputs are always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_shifts = coreg_method__shift

        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations match the exact values
        shifts = [c.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]  # type: ignore
        assert shifts == pytest.approx(expected_shifts)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("vshift", [0.2, 10.0, 1000.0])  # type: ignore
    def test_coreg_vertical_translation__synthetic(self, fit_args, vshift) -> None:
        """
        Test the vertical shift coregistration with synthetic shifted data. These tests include VerticalShift.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.
        """

        # Create a vertical shift correction instance
        vshiftcorr = coreg.VerticalShift()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        # Create synthetic vertical shift from the reference DEM
        ref = self.ref
        ref_vshifted = ref + vshift

        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_vshifted = ref_vshifted.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
        elev_fit_args["to_be_aligned_elev"] = ref_vshifted

        # Fit the vertical shift model to the data
        coreg_elev = vshiftcorr.fit_and_apply(**elev_fit_args, subsample=50000, random_state=42)

        # Check that the right vertical shift was found
        assert vshiftcorr.meta["outputs"]["affine"]["shift_z"] == pytest.approx(-vshift, rel=10e-2)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_vshifted.geometry.x.values, ref_vshifted.geometry.y.values)) - ref_vshifted["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_vshifted
            dh = ref - coreg_elev

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Check that the median difference is zero, and that no additional variance
        # was introduced, so that the variance is also close to zero (no variance for a constant vertical shift)
        assert np.nanmedian(dh) == pytest.approx(0, abs=10e-6)
        assert np.nanvar(dh) == pytest.approx(0, abs=10e-6)

    @pytest.mark.parametrize("coreg_method__vshift", [(coreg.VerticalShift, -2.305015)])  # type: ignore
    def test_coreg_vertical_translation__example(
        self, coreg_method__vshift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the vertical translation co-registration output is always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_vshift = coreg_method__vshift

        # Run co-registration
        c = coreg_method(subsample=50000)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations match the exact values
        vshift = c.meta["outputs"]["affine"]["shift_z"]
        assert vshift == pytest.approx(expected_vshift)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize(
        "shifts_rotations", [(20, 5, 0.1, 0.1, 0.05, 0.01), (-50, 100, 0.1, 1, 0.5, 0.01)]
    )  # type: ignore
    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.LZD, coreg.CPD])  # type: ignore
    def test_coreg_rigid__synthetic(self, fit_args, shifts_rotations, coreg_method) -> None:
        """
        Test the rigid coregistrations with synthetic misaligned (shifted and rotated) data.

        We test all combinaison of inputs: raster-raster, point-raster and raster-point.

        We verify that the matrix found by the coregistration is within 1% of the synthetic matrix, and inverted from
        the one introduced, and that applying the coregistration to the misaligned elevations corrects more than
        95% of the variance from the initial elevation differences (hence, that the direction of coregistration has
        to be the right one; and that there is no other errors introduced in the process).
        """

        # Initiate coregistration
        horizontal_coreg = coreg_method()

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        ref = self.ref

        # Create synthetic rigid transformation (translation and rotation) from the reference DEM
        matrix = matrix_from_translations_rotations(*shifts_rotations)

        # Pass a centroid
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_shifted_rotated = ref_shifted_rotated.to_pointcloud(
                data_column_name="z", subsample=50000, random_state=42
            ).ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted_rotated

        # Run coregistration
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=subsample_size, random_state=42)

        # Check that fit matrix is the invert of those used above, within a relative % for rotations
        fit_matrix = horizontal_coreg.meta["outputs"]["affine"]["matrix"]
        invert_fit_matrix = invert_matrix(fit_matrix)
        invert_fit_shifts_rotations = translations_rotations_from_matrix(invert_fit_matrix)

        # Check that shifts are not unreasonable within 100%, except for CPD that has trouble
        if coreg_method != coreg.CPD:
            assert np.allclose(shifts_rotations[0:3], invert_fit_shifts_rotations[:3], rtol=1)

        # Specify rotation precision: LZD is usually more precise than ICP
        atol = 10e-3 if coreg_method == coreg.LZD else 2 * 10e-2
        assert np.allclose(shifts_rotations[3:], invert_fit_shifts_rotations[3:], atol=atol)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_shifted_rotated.geometry.x.values, ref_shifted_rotated.geometry.y.values))
                - ref_shifted_rotated["z"]
            )
            dh = ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values)) - coreg_elev["z"]
        else:
            init_dh = ref - ref_shifted_rotated
            dh = ref - coreg_elev

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            init_dh.plot()
            plt.show()
            dh.plot()
            plt.show()

        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        # Checking for 95% of variance as ICP cannot always resolve the small shifts
        # And only 30% of variance for CPD that can't resolve shifts at all
        fac_reduc_var = 0.05 if coreg_method != coreg.CPD else 0.7
        assert np.nanvar(dh / np.nanstd(init_dh)) < fac_reduc_var

    @pytest.mark.parametrize(
        "coreg_method__shifts_rotations",
        [
            (coreg.ICP, (5.417970, 1.128243, -2.066260, 0.0071103, -0.007524, -0.0047392)),
            (coreg.LZD, (9.969819, 2.140150, -1.925771, 0.0070245, -0.00766, -0.008174)),
            (coreg.CPD, (0.005405, 0.005163, -2.047066, 0.0070245, -0.00755, -0.0000405)),
        ],
    )  # type: ignore
    def test_coreg_rigid__example(
        self, coreg_method__shifts_rotations: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the rigid co-registration outputs is always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba = load_examples(crop=False)[0:2]
        inlier_mask = ~self.outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_shifts_rots = coreg_method__shifts_rotations

        # Run co-registration
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations and rotations match the exact values
        fit_matrix = c.meta["outputs"]["affine"]["matrix"]
        fit_shifts_rotations = translations_rotations_from_matrix(fit_matrix)
        assert fit_shifts_rotations == pytest.approx(expected_shifts_rots, abs=10e-6)

    @pytest.mark.parametrize(
        "rigid_coreg",
        [
            coreg.ICP(method="point-to-point", max_iterations=20),
            coreg.ICP(method="point-to-plane"),
            coreg.ICP(fit_minimizer="lsq_approx"),
            coreg.ICP(fit_minimizer=scipy.optimize.least_squares),
            coreg.ICP(picky=True),
            coreg.ICP(picky=False),
            coreg.CPD(weight=0.5),
        ],
    )  # type: ignore
    def test_coreg_rigid__specific_args(self, rigid_coreg) -> None:
        """
        Check that all specific arguments (non-fitting and binning, subsampling, iterative) of rigid coregistrations
        run correctly and yield with a reasonable output by comparing back after a synthetic transformation.
        """

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        shifts_rotations = (300, 150, 75, 1, 0.5, 0.2)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Coregister
        subsample_size = 50000 if rigid_coreg.__class__.__name__ != "CPD" else 500
        rigid_coreg.fit(ref, ref_shifted_rotated, random_state=42, subsample=subsample_size)

        # Check that fit matrix is the invert of those used above, within a relative % for rotations
        fit_matrix = rigid_coreg.meta["outputs"]["affine"]["matrix"]
        invert_fit_matrix = invert_matrix(fit_matrix)
        invert_fit_shifts_rotations = translations_rotations_from_matrix(invert_fit_matrix)

        # Not so precise for shifts
        if rigid_coreg.__class__.__name__ != "CPD":
            assert np.allclose(invert_fit_shifts_rotations[:3], shifts_rotations[:3], rtol=1)
        # Precise for rotations
        assert np.allclose(invert_fit_shifts_rotations[3:], shifts_rotations[3:], rtol=10e-1, atol=2 * 10e-2)

    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.CPD, coreg.LZD])  # type: ignore
    def test_coreg_rigid__only_translation(self, coreg_method) -> None:

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (300, 150, 75, 0.01, 0.01, 0.01)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Run co-registration
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size, only_translation=True)
        c.fit(ref, ref_shifted_rotated, random_state=42)

        # Get invert of resulting matrix
        fit_matrix = c.meta["outputs"]["affine"]["matrix"]
        invert_fit_matrix = invert_matrix(fit_matrix)
        invert_fit_shifts_translations = translations_rotations_from_matrix(invert_fit_matrix)

        # Check that rotations are not solved for
        assert np.allclose(invert_fit_shifts_translations[3:], 0)

        # Check that translations are not far from expected values
        if coreg_method != coreg.CPD:
            assert np.allclose(invert_fit_shifts_translations[:3], shifts_rotations[:3], rtol=10e-1)

    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.CPD])  # type: ignore
    def test_coreg_rigid__standardize(self, coreg_method) -> None:

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (300, 150, 75, 1, 0.5, 0.2)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # 1/ Run co-registration with standardization
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        c_std = coreg_method(subsample=subsample_size, standardize=True)
        c_std.fit(ref, ref_shifted_rotated, random_state=42)

        # Get invert of resulting matrix
        fit_matrix_std = c_std.meta["outputs"]["affine"]["matrix"]
        invert_fit_shifts_translations_std = translations_rotations_from_matrix(invert_matrix(fit_matrix_std))

        # Check that standardized result are OK
        if coreg_method != coreg.CPD:
            assert np.allclose(invert_fit_shifts_translations_std[:3], shifts_rotations[:3], rtol=1)
        assert np.allclose(invert_fit_shifts_translations_std[3:], shifts_rotations[3:], rtol=10e-1, atol=2 * 10e-2)

        # 2/ Run coregistration without standardization

        c_nonstd = coreg_method(subsample=subsample_size, standardize=False)

        # For CPD, without standardization, the numerics fail
        if coreg_method == coreg.CPD:
            with pytest.raises(
                ValueError,
                match=re.escape("CPD coregistration numerics during np.linalg.svd(), " "try setting standardize=True."),
            ):
                c_nonstd.fit(ref, ref_shifted_rotated, random_state=42)
            return
        # For ICP, the numerics pass
        else:
            c_nonstd.fit(ref, ref_shifted_rotated, random_state=42)

        fit_matrix_nonstd = c_nonstd.meta["outputs"]["affine"]["matrix"]
        invert_fit_shifts_translations_nonstd = translations_rotations_from_matrix(invert_matrix(fit_matrix_nonstd))

        # Check results are worse for non-standardized
        assert np.allclose(invert_fit_shifts_translations_nonstd[:3], shifts_rotations[:3], rtol=1)
        assert np.allclose(invert_fit_shifts_translations_nonstd[3:], shifts_rotations[3:], rtol=10e-1, atol=2 * 10e-2)

    def test_nuthkaab_no_vertical_shift(self) -> None:
        ref, tba = load_examples(crop=False)[0:2]

        # Compare Nuth and Kaab method with and without applying vertical shift
        coreg_method1 = coreg.NuthKaab(vertical_shift=True)
        coreg_method2 = coreg.NuthKaab(vertical_shift=False)

        coreg_method1.fit(ref, tba, random_state=42)
        coreg_method2.fit(ref, tba, random_state=42)

        # Recover the shifts computed by coregistration in matrix form
        matrix1 = coreg_method1.to_matrix()
        matrix2 = coreg_method2.to_matrix()

        # Assert vertical shift is 0 for the 2nd coreg method
        assert matrix2[2, 3] == 0

        # Assert horizontal shifts are the same
        matrix2[2, 3] = matrix1[2, 3]
        assert np.array_equal(matrix1, matrix2)

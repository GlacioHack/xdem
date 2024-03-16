"""Functions to test the affine coregistrations."""
from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType

import xdem
from xdem import coreg, examples
from xdem.coreg.affine import AffineCoreg, CoregDict


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_raster = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


class TestAffineCoreg:

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

    def test_vertical_shift(self) -> None:

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

    def test_coreg_example(self, verbose: bool = False) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.
        """

        # Run co-registration
        nuth_kaab = xdem.coreg.NuthKaab()
        nuth_kaab.fit(self.ref, self.tba, inlier_mask=self.inlier_mask, verbose=verbose, random_state=42)

        # Check the output metadata is always the same
        shifts = (nuth_kaab._meta["offset_east_px"], nuth_kaab._meta["offset_north_px"], nuth_kaab._meta["vshift"])
        assert shifts == pytest.approx((-0.463, -0.133, -1.9876264671765433))

    def test_gradientdescending(self, subsample: int = 10000, inlier_mask: bool = True, verbose: bool = False) -> None:
        """
        Test the co-registration outputs performed on the example are always the same. This overlaps with the test in
        test_examples.py, but helps identify from where differences arise.

        It also implicitly tests the z_name kwarg and whether a geometry column can be provided instead of E/N cols.
        """
        if inlier_mask:
            inlier_mask = self.inlier_mask

        # Run co-registration
        gds = xdem.coreg.GradientDescending(subsample=subsample)
        gds.fit_pts(
            self.ref.to_points().ds,
            self.tba,
            inlier_mask=inlier_mask,
            verbose=verbose,
            subsample=subsample,
            z_name="b1",
        )
        assert gds._meta["offset_east_px"] == pytest.approx(-0.496000, rel=1e-1, abs=0.1)
        assert gds._meta["offset_north_px"] == pytest.approx(-0.1875, rel=1e-1, abs=0.1)
        assert gds._meta["vshift"] == pytest.approx(-1.8730, rel=1e-1)

    @pytest.mark.parametrize("shift_px", [(1, 1), (2, 2)])  # type: ignore
    @pytest.mark.parametrize("coreg_class", [coreg.NuthKaab, coreg.GradientDescending, coreg.ICP])  # type: ignore
    @pytest.mark.parametrize("points_or_raster", ["raster", "points"])
    def test_coreg_example_shift(self, shift_px, coreg_class, points_or_raster, verbose=False, subsample=5000):
        """
        For comparison of coreg algorithms:
        Shift a ref_dem on purpose, e.g. shift_px = (1,1), and then applying coreg to shift it back.
        """
        res = self.ref.res[0]

        # shift DEM by shift_px
        shifted_ref = self.ref.copy()
        shifted_ref.shift(shift_px[0] * res, shift_px[1] * res, inplace=True)

        shifted_ref_points = shifted_ref.to_points(as_array=False, subsample=subsample, pixel_offset="center").ds
        shifted_ref_points["E"] = shifted_ref_points.geometry.x
        shifted_ref_points["N"] = shifted_ref_points.geometry.y
        shifted_ref_points.rename(columns={"b1": "z"}, inplace=True)

        kwargs = {} if coreg_class.__name__ != "GradientDescending" else {"subsample": subsample}

        coreg_obj = coreg_class(**kwargs)

        best_east_diff = 1e5
        best_north_diff = 1e5
        if points_or_raster == "raster":
            coreg_obj.fit(shifted_ref, self.ref, verbose=verbose, random_state=42)
        elif points_or_raster == "points":
            coreg_obj.fit_pts(shifted_ref_points, self.ref, verbose=verbose, random_state=42)

        if coreg_class.__name__ == "ICP":
            matrix = coreg_obj.to_matrix()
            # The ICP fit only creates a matrix and doesn't normally show the alignment in pixels
            # Since the test is formed to validate pixel shifts, these calls extract the approximate pixel shift
            # from the matrix (it's not perfect since rotation/scale can change it).
            coreg_obj._meta["offset_east_px"] = -matrix[0][3] / res
            coreg_obj._meta["offset_north_px"] = -matrix[1][3] / res

        # ICP can never be expected to be much better than 1px on structured data, as its implementation often finds a
        # minimum between two grid points. This is clearly warned for in the documentation.
        precision = 1e-2 if coreg_class.__name__ != "ICP" else 1

        if coreg_obj._meta["offset_east_px"] == pytest.approx(-shift_px[0], rel=precision) and coreg_obj._meta[
            "offset_north_px"
        ] == pytest.approx(-shift_px[0], rel=precision):
            return
        best_east_diff = coreg_obj._meta["offset_east_px"] - shift_px[0]
        best_north_diff = coreg_obj._meta["offset_north_px"] - shift_px[1]

        raise AssertionError(f"Diffs are too big. east: {best_east_diff:.2f} px, north: {best_north_diff:.2f} px")

    def test_nuth_kaab(self) -> None:

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

    def test_tilt(self) -> None:

        # Try a 1st degree deramping.
        tilt = coreg.Tilt()

        # Fit the data
        tilt.fit(**self.fit_params, random_state=42)

        # Apply the deramping to a DEM
        tilted_dem = tilt.apply(self.tba)

        # Get the periglacial offset after deramping
        periglacial_offset = (self.ref - tilted_dem)[self.inlier_mask]
        # Get the periglacial offset before deramping
        pre_offset = (self.ref - self.tba)[self.inlier_mask]

        # Check that the error improved
        assert np.abs(np.mean(periglacial_offset)) < np.abs(np.mean(pre_offset))

        # Check that the mean periglacial offset is low
        assert np.abs(np.mean(periglacial_offset)) < 0.02

    def test_icp_opencv(self) -> None:

        # Do a fast and dirty 3 iteration ICP just to make sure it doesn't error out.
        icp = coreg.ICP(max_iterations=3)
        icp.fit(**self.fit_params)

        aligned_dem, _ = icp.apply(self.tba.data, self.ref.transform, self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

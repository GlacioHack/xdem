"""Functions to test the affine coregistrations."""

from __future__ import annotations

import logging
import os.path
import warnings
from functools import partial
from typing import Any, Callable, Literal

import geopandas as gpd
import geoutils
import numpy as np
import pytest
import rasterio as rio
import scipy.optimize
from geoutils import Raster, Vector
from geoutils._typing import Number
from geoutils.raster.transformation import _translate
from geoutils.stats import nmad
from scipy.ndimage import binary_dilation

from xdem import DEM, coreg, examples
from xdem._typing import NDArrayf
from xdem.coreg.affine import (
    AffineCoreg,
    _design_matrix_nuth_kaab,
    _iterate_method,
    _lzd_fit_error_propag,
    _nuth_kaab_fit_func,
    _reproject_horizontal_shift_samecrs,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)
from xdem.coreg.pipeline import CoregPipeline


def load_examples() -> tuple[Raster, Raster, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path_test("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


def load_examples_fullsize() -> tuple[Raster, Raster, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


@pytest.fixture
def curved_dem() -> Raster:
    """Create projected terrain with variation in both gradients and all rigid parameters."""

    rows, columns = np.mgrid[:40, :44]
    values = 5 * np.sin(columns / 7) + 3 * np.cos(rows / 6) + 2 * np.sin((columns + rows) / 11)
    transform = rio.Affine(2, 0, 500_000, 0, -2, 7_000_000)
    return Raster.from_array(values, transform, 32632, nodata=np.nan, area_or_point="Area")


def _point_samples(dem: Raster) -> geoutils.PointCloud:
    """We sample raster values at the coordinates defined by its pixel interpretation."""

    # Draw the same pixels for either input while locating them independently of raster-to-point defaults
    positions = np.random.default_rng(3).choice(dem.data.size, 900, replace=False)
    rows, columns = np.unravel_index(positions, dem.shape)
    x, y = dem.ij2xy(rows, columns)
    frame = gpd.GeoDataFrame({"z": dem.data[rows, columns]}, geometry=gpd.points_from_xy(x, y), crs=dem.crs)
    return geoutils.PointCloud(frame, data_name="z")


class TestAffineCoreg:
    """Test module for affine coregistrations."""

    ref, tba, outlines = load_examples()
    inlier_mask = ~outlines.create_mask(ref)

    # Check all point-raster possibilities supported
    # Use the reference DEM for both, it will be artificially misaligned during tests
    # Raster-Raster
    fit_args_rst_rst = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # Convert DEMs to points with a bit of subsampling for speed-up
    ref_pts = ref.to_pointcloud(data_column_name="z", subsample=3000, random_state=42).ds
    tba_pts = ref.to_pointcloud(data_column_name="z", subsample=3000, random_state=42).ds

    # Raster-Point
    fit_args_rst_pts = dict(reference_elev=ref, to_be_aligned_elev=tba_pts, inlier_mask=inlier_mask)

    # Point-Raster
    fit_args_pts_rst = dict(reference_elev=ref_pts, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # TODO: Find out why fit_args_pts_rst can yield significantly different accuracy...
    all_fit_args = [fit_args_rst_rst, fit_args_rst_pts]

    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("area_or_point", [None, "Area", "Point"])
    @pytest.mark.parametrize("shift_interpretation", [False, True])
    def test_fit__identity(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        area_or_point: Literal["Area", "Point"] | None,
        shift_interpretation: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that fitting a DEM to itself gives zero shifts, for every pixel interpretation and coreg method."""

        # Set the interpretation
        reference = curved_dem.copy()
        reference.set_area_or_point(area_or_point, shift_area_or_point=False)
        monkeypatch.setitem(geoutils.config, "shift_area_or_point", shift_interpretation)

        # Fit the same finite surface at a fixed sample of its locations
        fitted = method(subsample=600).fit(reference, reference, random_state=42)

        # Identity alignment must not introduce a half pixel translation or a rotation
        np.testing.assert_allclose(fitted.to_matrix(), np.eye(4), atol=1e-7)
        assert reference.transform == curved_dem.transform

    @pytest.mark.parametrize(
        "xoff_yoff",
        [(ref.res[0], ref.res[1]), (10 * ref.res[0], 10 * ref.res[1]), (-1.2 * ref.res[0], -1.2 * ref.res[1])],
    )
    def test_reproject_horizontal_shift_samecrs__gdal(
        self, xoff_yoff: tuple[float, float], get_test_data_path: Callable[[str], str]
    ) -> None:
        """Check that the same-CRS reprojection based on SciPy (replacing Rasterio due to subpixel errors)
        is accurate by comparing to GDAL."""

        ref = load_examples()[0]

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
        transformed_points = coreg_obj.apply(self.points, z_name="z")
        assert all(transformed_points["z"].values == vshift)

        # Check that the from_translation function works as expected.
        x_offset = 5
        coreg_obj2 = AffineCoreg.from_translations(x_off=x_offset)
        transformed_points2 = coreg_obj2.apply(self.points, z_name="z")
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

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("shifts", [(20, 5, 2), (-20, 50, 2)])
    @pytest.mark.parametrize("coreg_method", [coreg.NuthKaab, coreg.ICP, coreg.LZD])
    def test_coreg_translations__synthetic(
        self, fit_args: Any, shifts: tuple[float, float, float], coreg_method: coreg.Coreg
    ) -> None:
        """
        Test the horizontal/vertical shift coregistrations with synthetic shifted data. These tests include NuthKaab,
        ICP and DhMinimize.

        We test all combination of inputs: raster-raster, point-raster and raster-point.

        We verify that the shifts found by the coregistration are within 1% of the synthetic shifts with opposite sign
        of the ones introduced, and that applying the coregistration to the shifted elevations corrects more than
        99% of the variance from the initial elevation differences (hence, that the direction of coregistration has
        to be the right one; and that there is no other errors introduced in the process).
        """

        warnings.filterwarnings("ignore", message="Covariance of the parameters*")

        if coreg_method == coreg.NuthKaab:
            init_args = {"bin_before_fit": False}
        else:
            init_args = {}
        horizontal_coreg = coreg_method(**init_args)

        # Copy dictionary and remove inlier mask
        elev_fit_args = fit_args.copy()
        elev_fit_args.pop("inlier_mask")

        # Create synthetic translation from the reference DEM
        ref = self.ref
        ref_shifted = ref.translate(shifts[0], shifts[1]) + shifts[2]
        # Convert to point cloud if input was point cloud
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            ref_shifted = ref_shifted.to_pointcloud(data_column_name="z", random_state=42).ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted

        # Run coregistration
        subsample_size = 1 if coreg_method != coreg.CPD else 500
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=subsample_size, random_state=42)

        # Check all fit parameters are the opposite of those used above, within a relative 1% (10% for ICP)
        fit_shifts = [-horizontal_coreg.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]

        # Allow one pixel for ICP, half a pixel for NuthKaab, and a tenth of a pixel for LZD
        if coreg_method == coreg.ICP:
            atol = ref.res[0]
        elif coreg_method == coreg.NuthKaab:
            atol = 0.5 * ref.res[0]
        else:
            atol = 0.1 * ref.res[0]
        assert np.allclose(fit_shifts, shifts, atol=atol)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_shifted.geometry.x.values, ref_shifted.geometry.y.values), as_array=True)
                - ref_shifted["z"]
            )
            dh = (
                ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values), as_array=True)
                - coreg_elev["z"]
            )
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

        # Check applying the coregistration reduces the variance by 90%
        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        tol = 0.15 if coreg_method == coreg.NuthKaab else 0.1
        assert np.nanvar(dh / np.nanstd(init_dh)) < tol

    @pytest.mark.parametrize(
        "method",
        [
            coreg.NuthKaab,
            coreg.LZD,
            coreg.ICP,
            pytest.param(partial(coreg.LZD, linearized=False), id="LZD-nonlinear"),
        ],
    )
    @pytest.mark.parametrize("pair", ["raster-raster", "raster-point", "point-raster"])
    @pytest.mark.parametrize("area_or_point", ["Area", "Point"])
    def test_fit__translated_raster_and_points(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        pair: str,
        area_or_point: Literal["Area", "Point"],
    ) -> None:
        """Checks that raster and point fits recover an artificial translation with either input as reference."""

        # Translate the same terrain in map coordinates and height without resampling its elevations
        reference = curved_dem.copy()
        reference.set_area_or_point(area_or_point, shift_area_or_point=False)
        shifted = reference.translate(1.2, -0.8) + 2.5
        reference_input: Any = reference
        shifted_input: Any = shifted
        if pair == "raster-point":
            shifted_input = _point_samples(shifted)
        elif pair == "point-raster":
            reference_input = _point_samples(reference)

        # Fit every observation to check pixel coordinates and input order independently of sampling
        options = {} if method == coreg.NuthKaab else {"only_translation": True}
        fitted = method(subsample=1, **options).fit(reference_input, shifted_input, random_state=42)

        # Should all be close to it within tiny tolerance
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), atol=0.06)
        np.testing.assert_allclose(fitted.to_rotations(), 0, atol=0.02)
        assert reference.transform == curved_dem.transform

    @pytest.mark.parametrize("pair", ["raster-raster", "raster-point", "point-raster"])
    @pytest.mark.parametrize("loss", [nmad, np.nanstd])
    def test_lzd__custom_nelder_mead(self, curved_dem: Raster, pair: str, loss: Callable[..., Any]) -> None:
        """Checks that a custom Nelder-Mead wrapper works to mimic old DhMinimize behaviour."""

        # We add a large Z offset to verify that residual spread and median height are fitted separately
        reference: Any = curved_dem
        shifted: Any = curved_dem.translate(1.2, -0.8) + 500
        if pair == "raster-point":
            shifted = _point_samples(shifted)
        elif pair == "point-raster":
            reference = _point_samples(reference)

        # Adapt LZD residual vector to a scalar spread objective and fit the unobservable Z shift separately
        calls = []
        optimizer_options = {"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000}

        def fit_minimizer(
            residuals: Callable[[NDArrayf], NDArrayf], x0: NDArrayf, *, loss: Callable[..., Any], **kwargs: Any
        ) -> Any:
            """Custom minimize with Nelder-Mead and different horizontal/vertical logic."""

            calls.append((x0.copy(), loss, kwargs))

            def horizontal_residuals(xy: NDArrayf) -> NDArrayf:
                """Evaluate residuals with a fixed vertical shift during the horizontal solve."""

                return residuals(np.array([xy[0], xy[1], x0[2]]))

            result = scipy.optimize.minimize(
                lambda xy: loss(horizontal_residuals(xy)), np.ones(2), method="Nelder-Mead", **kwargs
            )
            z_shift = x0[2] + np.median(horizontal_residuals(result.x))
            result.x = np.array([result.x[0], result.x[1], z_shift])
            return result

        # Run the old method and its nonlinear LZD replacement with the same sample and loss
        with pytest.warns(DeprecationWarning, match="DhMinimize is deprecated"):
            legacy = coreg.DhMinimize(subsample=1, fit_loss_func=loss)
        legacy.fit(reference, shifted, random_state=42)
        replacement = coreg.LZD(
            linearized=False,
            only_translation=True,
            fit_minimizer=fit_minimizer,
            fit_loss_func=loss,
            subsample=1,
        ).fit(reference, shifted, random_state=42, options=optimizer_options)

        # Both sampling directions can differ slightly after interpolation, within 3% of a 2 m pixel
        np.testing.assert_allclose(legacy.to_translations(), (-1.2, 0.8, -500), atol=0.06)
        np.testing.assert_allclose(replacement.to_translations(), (-1.2, 0.8, -500), atol=0.06)
        np.testing.assert_array_equal(replacement.to_matrix()[:3, :3], np.eye(3))

        # Check that every iteration received the selected loss and unchanged optimizer options
        assert calls
        for initial, received_loss, received_kwargs in calls:
            np.testing.assert_array_equal(initial, np.zeros(3))
            assert received_loss is loss
            assert received_kwargs == {"options": optimizer_options}

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("resolution", [(0.5, 0.5), (4, 4), (2, 3), (3, 2)])
    def test_fit__pixel_sizes(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        resolution: tuple[float, float],
    ) -> None:
        """Checks that square and rectangular pixel sizes scale the recovered map translation correctly."""

        # Assign different pixel sizes to the same indexed terrain and plant a fixed fraction of each pixel
        res_x, res_y = resolution
        reference = curved_dem.copy()
        reference.set_transform(rio.Affine(res_x, 0, 500_000, 0, -res_y, 7_000_000))
        shift_x, shift_y = 0.6 * res_x, -0.4 * res_y
        shifted = reference.translate(shift_x, shift_y) + 2.5

        # Fit translations while deriving gradients with the actual X/Y pixel sizes
        options = {} if method == coreg.NuthKaab else {"only_translation": True}
        fitted = method(subsample=1, **options).fit(reference, shifted, random_state=42)

        # Express horizontal errors in pixels; use the same interpolation allowance for every resolution
        error = np.asarray(fitted.to_translations()) + (shift_x, shift_y, 2.5)
        np.testing.assert_array_less(np.abs(error) / (res_x, res_y, 1), 0.03)

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("resample", [False, True])
    def test_apply__pixel_interpretation(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], resample: bool
    ) -> None:
        """Checks that applying a fitted transform preserves a raster pixel interpretation."""

        # We use Point interpretation, as losing its metadata would change future sample coordinates
        reference = curved_dem.copy()
        reference.set_area_or_point("Point", shift_area_or_point=False)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Fit and apply, with/without resampling onto the input grid
        options = {} if method == coreg.NuthKaab else {"only_translation": True}
        fitted = method(subsample=1, **options).fit(reference, shifted, random_state=42)
        aligned = fitted.apply(shifted, resample=resample)

        # Further raster/point comparisons must use the same interpretation as the original data
        assert aligned.area_or_point == "Point"
        assert shifted.area_or_point == "Point"

    @pytest.mark.parametrize("method", [coreg.VerticalShift, coreg.NuthKaab, coreg.LZD])
    @pytest.mark.parametrize("shape", [(40, 44), (33, 37)])
    def test_fit__different_grids(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], shape: tuple[int, int]
    ) -> None:
        """
        Checks that fitting corrects only the added height shift when rasters sample the same terrain with
        different X/Y grid origins.
        """

        # Create a bilinear reference DEM
        ref_rows, ref_columns = np.indices(curved_dem.shape)
        reference_values = 100 + 0.02 * ref_columns * ref_rows + 0.3 * ref_columns - 0.2 * ref_rows
        reference = Raster.from_array(
            reference_values,
            curved_dem.transform,
            curved_dem.crs,
            nodata=np.nan,
            area_or_point="Area",
        )

        # Sample the same surface on an X/Y offset grid and add a Z shift
        rows, columns = np.indices(shape)
        source_columns = columns + 0.3
        source_rows = rows + 0.4
        values = 100 + 0.02 * source_columns * source_rows + 0.3 * source_columns - 0.2 * source_rows
        transform = curved_dem.transform * rio.Affine.translation(0.3, 0.4)
        shifted = Raster.from_array(values + 0.35, transform, curved_dem.crs, nodata=np.nan, area_or_point="Area")

        # Exclude edges where interpolation clamps coordinates
        valid = np.zeros(reference.shape, dtype=bool)
        valid[2 : shape[0] - 2, 2 : shape[1] - 2] = True

        # Fit the two grids
        options = {"only_translation": True} if method == coreg.LZD else {}
        fitted = method(subsample=1, **options).fit(reference, shifted, inlier_mask=valid, random_state=42)

        # Check that only the height shift is corrected
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.35), atol=1e-5)

    @pytest.mark.parametrize(
        "method",
        [
            coreg.VerticalShift,
            coreg.NuthKaab,
            pytest.param(partial(coreg.LZD, only_translation=True), id="LZD"),
        ],
    )
    @pytest.mark.parametrize("mask_type", ["array", "raster"])
    def test_fit__cropped_elevations_consistency(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], mask_type: str
    ) -> None:
        """
        Checks that cropping elevations or masking the same area gives the same fitted transform.

        This also crops both DEMs, extending the mask-only comparison in test_fit_and_apply__cropped_mask().
        All samples are used so cropping cannot change a random sample.
        """

        # Create non-linear DEM (enough aspect variation to converge)
        rows, columns = np.indices(curved_dem.shape)
        values = 5 * np.sin(columns / 3) + 3 * np.cos(rows / 3) + 2 * np.sin((columns + rows) / 5)
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan, area_or_point="Area")

        # Translate  and add errors outside crop (so ignoring the full-size mask changes the fit)
        shifted = reference.translate(1.2, -0.8) + 2.5
        crop_bounds = (6, 5, 38, 35)
        outside_crop = np.ones(reference.shape, dtype=bool)
        outside_crop[5:35, 6:38] = False
        shifted.data[outside_crop] += 20
        reference_crop = reference.icrop(crop_bounds)
        shifted_crop = shifted.icrop(crop_bounds)

        # Leave border around selected area (so gradients and interpolation use the same neighbours)
        full_mask = np.zeros(reference.shape, dtype=bool)
        full_mask[8:32, 9:35] = True
        cropped_mask = Raster.from_array(full_mask, reference.transform, reference.crs).icrop(crop_bounds)
        mask_input = cropped_mask.data if mask_type == "array" else cropped_mask

        # Fit on full/crop grids
        full_fit = method(subsample=1).fit(reference, shifted, inlier_mask=full_mask)
        cropped_fit = method(subsample=1).fit(reference_crop, shifted_crop, inlier_mask=mask_input)

        # Should get the same transformations
        np.testing.assert_allclose(full_fit.to_matrix(), cropped_fit.to_matrix(), rtol=0, atol=1e-8)
        assert full_fit.meta["outputs"]["random"] == cropped_fit.meta["outputs"]["random"]

    @pytest.mark.parametrize("method", [coreg.VerticalShift, coreg.NuthKaab, coreg.LZD])
    def test_fit__native_mask_overlap(self, curved_dem: Raster, method: Callable[..., AffineCoreg]) -> None:
        """
        Checks that fitting corrects only the added height shift when valid data overlap, despite
        different validity masks.
        """

        # Create overlapping rasters with valid data in opposite array halves
        reference = curved_dem.copy()
        reference.set_mask(np.indices(reference.shape)[1] >= 22)
        shifted_values = np.full(curved_dem.shape, np.nan)
        shifted_values[:, 22:] = curved_dem.data[:, :22] + 0.35
        transform = curved_dem.transform * rio.Affine.translation(-22, 0)
        shifted = Raster.from_array(shifted_values, transform, curved_dem.crs, nodata=np.nan, area_or_point="Area")

        # Fit the overlapping halves
        fitted = method(subsample=1).fit(reference, shifted, random_state=42)

        # Check the height correction
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.35), atol=1e-5)

    @pytest.mark.parametrize(
        "coreg_method, shift",
        [
            (coreg.NuthKaab, (9.492646, 2.841536, -2.058883)),
            (coreg.DhMinimize, (10.164111, 2.600748, -1.995853)),
            (coreg.LZD, (10.281134, 2.300993, -1.817791)),
            (coreg.ICP, (9.892070, 2.108643, -1.837282)),
        ],
    )
    def test_coreg_translations__example(
        self, coreg_method: type[coreg.Coreg], shift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Checks that translation coregistration stays close to reference fits on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba, outlines = load_examples_fullsize()
        inlier_mask = ~outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        expected_shifts = shift

        subsample_size = 5000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check fitted translations against reference values within their interpolation precision
        shifts = [c.meta["outputs"]["affine"][k] for k in ["shift_x", "shift_y", "shift_z"]]  # type: ignore
        tolerance = 0.5 if coreg_method == coreg.DhMinimize else 0.05
        assert shifts == pytest.approx(expected_shifts, abs=tolerance)

    @pytest.mark.parametrize("subsample, expected", [(1, 1760), (0.5, 880), (37, 37)])
    def test_vertical_shift__sample_count(self, curved_dem: Raster, subsample: float, expected: int) -> None:
        """Checks that VerticalShift reports the number of sampled observations rather than coordinate axes."""

        # Add a constant height offset without changing valid locations
        shifted = curved_dem + 2.5

        # Fit with every observation, half of them, or a fixed count
        fitted = coreg.VerticalShift(subsample=subsample).fit(curved_dem, shifted, random_state=42)

        # The sample metadata describe observations and the height correction remains exact
        assert fitted.meta["outputs"]["random"]["subsample_final"] == expected
        assert fitted.to_translations() == pytest.approx((0, 0, -2.5))

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    def test_vertical_shift__planar_height(self, curved_dem: Raster, surface: str) -> None:
        """Checks that a height-only fit remains observable on flat or planar terrain."""

        # Replace the terrain with a flat surface or ramp and add only a height offset
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Fit the single parameter supported by these surfaces
        fitted = coreg.VerticalShift().fit(reference, reference + 2.5)

        # Planarity removes horizontal information but does not prevent measuring a height difference
        assert fitted.to_translations() == pytest.approx((0, 0, -2.5))

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("vshift", [0.2, 10.0, 1000.0])
    def test_coreg_vertical_translation__synthetic(self, fit_args: Any, vshift: float) -> None:
        """
        Test the vertical shift coregistration with synthetic shifted data. These tests include VerticalShift.

        We test all combination of inputs: raster-raster, point-raster and raster-point.
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
            ref_vshifted = ref_vshifted.to_pointcloud(data_column_name="z").ds
        elev_fit_args["to_be_aligned_elev"] = ref_vshifted

        # Fit the vertical shift model to the data
        coreg_elev = vshiftcorr.fit_and_apply(**elev_fit_args)

        # Check that the right vertical shift was found
        assert vshiftcorr.meta["outputs"]["affine"]["shift_z"] == pytest.approx(-vshift, rel=10e-2)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points((ref_vshifted.geometry.x.values, ref_vshifted.geometry.y.values), as_array=True)
                - ref_vshifted["z"]
            )
            dh = (
                ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values), as_array=True)
                - coreg_elev["z"]
            )
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

    @pytest.mark.parametrize("coreg_method__vshift", [(coreg.VerticalShift, -2.305015)])
    def test_coreg_vertical_translation__example(
        self, coreg_method__vshift: tuple[type[AffineCoreg], tuple[float, float, float]]
    ) -> None:
        """
        Test that the vertical translation co-registration output is always exactly the same on the real example data.
        """

        # Use entire DEMs here (to compare to original values from older package versions)
        ref, tba, outlines = load_examples_fullsize()
        inlier_mask = ~outlines.create_mask(ref)

        # Get the coregistration method and expected shifts from the inputs
        coreg_method, expected_vshift = coreg_method__vshift

        # Run co-registration
        c = coreg_method(subsample=50000)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations match the exact values
        vshift = c.meta["outputs"]["affine"]["shift_z"]
        assert vshift == pytest.approx(expected_vshift)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    def test_icp__planar_p2point_identity(self, curved_dem: Raster, surface: str) -> None:
        """Checks that point-to-point ICP can exactly match a planar DEM."""

        # We create flat/ramp DEM (both planar)
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Fit with point-to-point
        fitted = coreg.ICP(method="point-to-point", subsample=1).fit(reference, reference, random_state=42)

        # Exact correspondences determine the identity even though all surface normals are parallel
        np.testing.assert_allclose(fitted.to_matrix(), np.eye(4), atol=1e-7)

    @pytest.mark.parametrize(
        "method",
        [
            coreg.LZD,
            pytest.param(partial(coreg.LZD, linearized=False), id="LZD-nonlinear"),
            coreg.ICP,
        ],
    )
    @pytest.mark.parametrize("area_or_point", ["Area", "Point"])
    def test_fit__rigid_points(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        area_or_point: Literal["Area", "Point"],
    ) -> None:
        """
        Checks that a known small rigid point transformation is inverted on the original DEM.

        This test is not redundant with others because it uses exact point transforms
        (to avoid raster resampling errors), allowing precise X/Y/Z checks for both pixel interpretations,
        including rotations with nonlinear LZD.
        """

        # Transform DEM exactly
        reference = curved_dem.copy()
        reference.set_area_or_point(area_or_point, shift_area_or_point=False)
        points = _point_samples(reference).gdf
        centroid = (float(points.geometry.x.mean()), float(points.geometry.y.mean()), float(points.z.mean()))
        matrix = matrix_from_translations_rotations(0.6, -0.4, 0.8, 0.01, -0.015, 0.02)
        shifted = coreg.apply_matrix(points, matrix, centroid=centroid)

        # Fit and apply inverse to the same transformed point cloud
        fitted = method(subsample=1, tolerance_translation=1e-5, tolerance_rotation=1e-5).fit(
            reference,
            shifted,
            random_state=42,
            z_name="z",
        )
        corrected = fitted.apply(shifted, z_name="z")

        # Compare directly transformed coordinates (avoiding ambiguity between different rotation centroids)
        np.testing.assert_allclose(corrected.geometry.x, points.geometry.x, atol=0.02)
        np.testing.assert_allclose(corrected.geometry.y, points.geometry.y, atol=0.02)
        np.testing.assert_allclose(corrected.z, points.z, atol=0.02)

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("shifts_rotations", [(20, 5, 10, 0, 0, 0), (0, 0, 0, 1, 0.5, 0.01)])
    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.LZD, coreg.CPD])
    def test_coreg_rigid__synthetic(
        self,
        fit_args: Any,
        shifts_rotations: tuple[float, float, float, float, float, float],
        coreg_method: coreg.Coreg,
    ) -> None:
        """
        Test the rigid coregistrations with synthetic misaligned (shifted and rotated) data.

        We test all combination of inputs: raster-raster, point-raster and raster-point.

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
            ref_shifted_rotated = ref_shifted_rotated.to_pointcloud(data_column_name="z").ds
        elev_fit_args["to_be_aligned_elev"] = ref_shifted_rotated

        # Run coregistration
        subsample_size = 1 if coreg_method != coreg.CPD else 500
        coreg_elev = horizontal_coreg.fit_and_apply(**elev_fit_args, subsample=subsample_size, random_state=42)

        # Check that fit matrix is the invert of those used above, within a relative % for rotations
        fit_matrix = horizontal_coreg.meta["outputs"]["affine"]["matrix"]
        invert_fit_matrix = invert_matrix(fit_matrix)
        invert_fit_shifts_rotations = translations_rotations_from_matrix(invert_fit_matrix)

        # Check that shifts are not unreasonable within 4 pixels for ICP and 0.5 pixel for LZD, skip for CPD
        # (more difficult to constrain when mixed with rotation)
        if coreg_method != coreg.CPD:
            atol_translation = 5 * ref.res[0] if coreg_method == coreg.ICP else 0.5 * ref.res[0]
            assert np.allclose(shifts_rotations[0:3], invert_fit_shifts_rotations[:3], atol=atol_translation)

        # Specify rotation precision: LZD is usually more precise than ICP/CPD
        # XY rotation is the most precise
        atol_xy_rotation = 10e-4 if coreg_method == coreg.LZD else 3 * 10e-2
        assert np.allclose(shifts_rotations[3:5], invert_fit_shifts_rotations[3:5], atol=atol_xy_rotation)
        # Z rotation is a bit harder, especially on a tiny DEM
        atol_z_rotation = 10e-3 if coreg_method == coreg.LZD else 10e-1
        assert np.allclose(shifts_rotations[5], invert_fit_shifts_rotations[5], atol=atol_z_rotation)

        # For a point cloud output, need to interpolate with the other DEM to get dh
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            init_dh = (
                ref.interp_points(
                    (ref_shifted_rotated.geometry.x.values, ref_shifted_rotated.geometry.y.values), as_array=True
                )
                - ref_shifted_rotated["z"]
            )
            dh = (
                ref.interp_points((coreg_elev.geometry.x.values, coreg_elev.geometry.y.values), as_array=True)
                - coreg_elev["z"]
            )
        else:
            init_dh = ref - ref_shifted_rotated
            dh = ref - coreg_elev

        # Plots for debugging
        PLOT = False
        if PLOT and isinstance(dh, geoutils.Raster):
            import matplotlib.pyplot as plt

            plt.subplot(121)
            init_dh.plot()
            plt.subplot(221)
            dh.plot()
            plt.show()

        # Need to standardize by the elevation difference spread to avoid huge/small values close to infinity
        # Checking 80% of variance is corrected
        # CPD can't resolve shifts at all, so just check it doesn't make it worse for non-zero shifts
        fac_reduc_var = 1 if (coreg_method == coreg.CPD and np.linalg.norm(shifts_rotations[0:3]) != 0) else 0.2
        assert np.nanvar(dh / np.nanstd(init_dh)) < fac_reduc_var

    @pytest.mark.parametrize(
        "coreg_method, shifts_rotations",
        [
            (coreg.ICP, (9.892070, 2.108643, -1.837282, 0.0078704, -0.0079582, -0.0065497)),
            (coreg.LZD, (10.281134, 2.300993, -1.817791, 0.0079446, -0.0081343, -0.0057847)),
            (coreg.CPD, (9.550005, 3.360676, -1.958521, 0.0052223, -0.0084495, -0.0387999)),
        ],
    )
    def test_coreg_rigid__example(
        self, coreg_method: type[AffineCoreg], shifts_rotations: tuple[float, float, float]
    ) -> None:
        """
        Test that the rigid co-registration outputs is always exactly the same on the real example data.
        """

        # Open the full-size example and use a fixed sample for a stable, fast regression check
        ref, tba, outlines = load_examples_fullsize()
        inlier_mask = ~outlines.create_mask(ref)

        # Run co-registration
        subsample_size = 5000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size)
        c.fit(ref, tba, inlier_mask=inlier_mask, random_state=42)

        # Check the output translations and rotations match the exact values
        fit_matrix = c.meta["outputs"]["affine"]["matrix"]
        fit_shifts_rotations = translations_rotations_from_matrix(fit_matrix)
        assert fit_shifts_rotations[:3] == pytest.approx(shifts_rotations[:3], abs=0.02)
        assert fit_shifts_rotations[3:] == pytest.approx(shifts_rotations[3:], abs=0.001)

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("max_iterations, converged", [(1, False), (20, True)])
    def test_fit__convergence_status(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        max_iterations: int,
        converged: bool,
    ) -> None:
        """Checks that iteration metadata distinguish converged and reached max iterations."""

        # Artificial translation
        shifted = curved_dem.translate(1.2, -0.8) + 2.5

        # We run with too few/many iterations to ensure convergence or not
        fitted = method(subsample=1, max_iterations=max_iterations).fit(curved_dem, shifted, random_state=42)

        # Check convergence is properly reported
        assert fitted.meta["outputs"]["iterative"]["converged"] is converged

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, "0.000"),
            (1.23456789, "1.235"),
            (-1.23456789, "-1.235"),
            (0.001, "0.001"),
            (100_000.0, "100000.000"),
            (0.0000123456789, "1.235e-05"),
            (-1_234_567.89, "-1.235e+06"),
        ],
    )
    def test_iterate_method__display_precision(
        self, value: float, expected: str, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Checks that iteration output rounds values properly for display."""

        def iteration_step(matrix: NDArrayf) -> tuple[NDArrayf, dict[str, float], None]:
            return matrix, {"translation": value, "rotation": 0.0}, None

        # We disable one tolerance to also check the final message
        with caplog.at_level(logging.INFO):
            _, statistics, output, _ = _iterate_method(
                iteration_step,
                np.eye(4),
                (),
                tolerances={"translation": None, "rotation": 0.123456789},
                max_iterations=1,
            )
        printed = capsys.readouterr().out

        # Check display rules are respected
        assert f"Last translation offset: {expected}" in printed
        assert "Last rotation offset: 0.000" in printed
        assert "translation: None; rotation: 0.123" in printed
        assert statistics["translation"] == value
        assert output["iteration_stats"].loc[1, "translation"] == value

    def test_iterate_method__quiet_logging(
        self, curved_dem: Raster, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Checks that coregistration prints no iteration progress when the logging level is above INFO."""

        shifted = curved_dem + 2.5

        # Capture the log
        with caplog.at_level(logging.WARNING):
            fitted = coreg.LZD(only_translation=True, subsample=1).fit(curved_dem, shifted, random_state=42)
        captured = capsys.readouterr()

        # Check silence on both
        assert captured.out == ""
        assert captured.err == ""
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -2.5), atol=1e-6)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize(
        "options, expected",
        [
            ({}, "ols"),
            ({"fit_minimizer": "ols"}, "ols"),
            ({"linearized": False}, scipy.optimize.least_squares),
            ({"fit_loss_func": "soft_l1"}, scipy.optimize.least_squares),
            ({"fit_minimizer": scipy.optimize.least_squares}, scipy.optimize.least_squares),
        ],
    )
    def test_fit__linear_minimizer_selection(
        self, method: Callable[..., AffineCoreg], options: dict[str, Any], expected: str | Callable[..., Any]
    ) -> None:
        """Checks that linearized fits select OLS while nonlinear models and robust losses use least squares."""

        # Select a linear model, a nonlinear model, a robust loss, or an explicit minimizer
        fitted = method(**options)

        # Check that metadata and info() describe the minimizer that will run
        minimizer = fitted.meta["inputs"]["fitorbin"]["fit_minimizer"]
        assert minimizer == expected
        expected_name = expected if isinstance(expected, str) else expected.__name__
        assert expected_name in fitted.info(as_str=True)

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param(coreg.LZD, id="LZD-linearized"),
            pytest.param(partial(coreg.LZD, linearized=False), id="LZD-nonlinear"),
            pytest.param(coreg.ICP, id="ICP-linearized"),
            pytest.param(partial(coreg.ICP, linearized=False), id="ICP-nonlinear"),
            pytest.param(partial(coreg.CPD, lsg=False), id="CPD-classic"),
            pytest.param(coreg.CPD, id="CPD-LSG"),
        ],
    )
    @pytest.mark.parametrize("only_translation", [False, True])
    def test_fit__custom_minimizer(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], only_translation: bool
    ) -> None:
        """Checks that custom minimizers receive the loss and recover the same step as the default solver."""

        # We translate the DEM and define custom minimize options
        shifted = curved_dem.translate(1.2, -0.8) + 2.5
        options = {
            "only_translation": only_translation,
            "subsample": 100,
            "max_iterations": 1,
            "fit_loss_func": "soft_l1",
        }
        calls = []

        def fit_minimizer(residuals: Callable[[NDArrayf], NDArrayf], x0: NDArrayf, *, loss: str) -> Any:
            calls.append((x0.copy(), loss))
            return scipy.optimize.least_squares(residuals, x0, loss=loss)

        # Fit with the built-in and custom minimizer
        expected = method(**options).fit(curved_dem, shifted, random_state=42)
        fitted = method(fit_minimizer=fit_minimizer, **options).fit(curved_dem, shifted, random_state=42)

        # Check that the callable runs with an identity step and returns the usual fitted transform
        assert calls
        for initial, loss in calls:
            np.testing.assert_array_equal(initial, np.zeros(3 if only_translation else 6))
            assert loss == "soft_l1"
        np.testing.assert_allclose(fitted.to_matrix(), expected.to_matrix(), rtol=0, atol=1e-10)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("only_translation", [False, True])
    @pytest.mark.parametrize("trim_residuals", [False, True])
    @pytest.mark.parametrize("pair", ["raster-raster", "raster-point"])
    def test_fit__linear_minimizer_consistency(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        only_translation: bool,
        trim_residuals: bool,
        pair: str,
    ) -> None:
        """Checks that direct OLS and iterative least squares give the same linearized step transform."""

        # Translate DEM
        shifted: Any = curved_dem.translate(1.2, -0.8) + 2.5
        if pair == "raster-point":
            shifted = _point_samples(shifted)
        options = {
            "subsample": 600,
            "only_translation": only_translation,
            "trim_residuals": trim_residuals,
            "max_iterations": 1,
        }

        # Compare solvers (difference should only be speed)
        direct = method(fit_minimizer="ols", **options).fit(curved_dem, shifted, random_state=42)
        iterative = method(fit_minimizer=scipy.optimize.least_squares, **options).fit(
            curved_dem, shifted, random_state=42
        )

        # Check match
        np.testing.assert_allclose(direct.to_matrix(), iterative.to_matrix(), rtol=0, atol=1e-6)
        assert direct.meta["outputs"]["random"] == iterative.meta["outputs"]["random"]
        assert direct.centroid() == iterative.centroid()

    @pytest.mark.parametrize(
        "rigid_coreg",
        [
            coreg.ICP(method="point-to-point", max_iterations=20),
            coreg.ICP(method="point-to-plane"),
            coreg.ICP(linearized=False),
            coreg.ICP(fit_minimizer=scipy.optimize.least_squares),
            coreg.ICP(picky=True),
            coreg.ICP(picky=False),
            coreg.CPD(weight=0.5),
            coreg.CPD(lsg=False),
            coreg.CPD(lsg=True),
        ],
    )
    def test_coreg_rigid__specific_args(self, rigid_coreg: coreg.Coreg) -> None:
        """
        Check that all specific arguments (non-fitting and binning, subsampling, iterative) of rigid coregistrations
        run correctly.
        """

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        shifts_rotations = (20, 5, 0.1, 0.1, 0.05, 0.01)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Coregister
        subsample_size = 1 if rigid_coreg.__class__.__name__ != "CPD" else 500
        rigid_coreg.fit(ref, ref_shifted_rotated, random_state=42, subsample=subsample_size)

    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.CPD, coreg.LZD])
    def test_coreg_rigid__only_translation(self, coreg_method: coreg.Coreg) -> None:

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (300, 150, 75, 0.01, 0.01, 0.01)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Run co-registration
        subsample_size = 1 if coreg_method != coreg.CPD else 300
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
    def test_coreg_rigid__sampling_strategy(self, coreg_method: Callable[..., AffineCoreg]) -> None:
        """Checks that point-to-point methods apply independent and same-location sampling strategies."""

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (10, 10, 10, 0.5, 0.5, 0.5)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Run co-registration
        subsample_size = 50000 if coreg_method != coreg.CPD else 500
        c = coreg_method(subsample=subsample_size, sampling_strategy="independent")
        c.fit(ref, ref_shifted_rotated, random_state=42)

        c2 = coreg_method(subsample=subsample_size, sampling_strategy="same_xy")
        c2.fit(ref, ref_shifted_rotated, random_state=42)

        c3 = coreg_method(subsample=subsample_size, sampling_strategy="iterative_same_xy")
        c3.fit(ref, ref_shifted_rotated, random_state=42)

        # Get invert of resulting matrices
        invert_fit_matrix = invert_matrix(c.meta["outputs"]["affine"]["matrix"])
        invert_fit_shifts_translations = translations_rotations_from_matrix(invert_fit_matrix)
        invert_fit_matrix2 = invert_matrix(c2.meta["outputs"]["affine"]["matrix"])
        invert_fit_shifts_translations2 = translations_rotations_from_matrix(invert_fit_matrix2)
        invert_fit_matrix3 = invert_matrix(c3.meta["outputs"]["affine"]["matrix"])
        invert_fit_shifts_translations3 = translations_rotations_from_matrix(invert_fit_matrix3)

        # TODO: Add checks depending on magnitude of translation/rotation
        #  ("independent" should perform better for large magnitude, "same_xy" better for small,
        #  and "iterative_same_xy" best for all)

        # Check that rotations are not far from expected values, skipping Z axis which is harder to get
        assert np.allclose(invert_fit_shifts_translations[3:5], shifts_rotations[3:5], atol=0.25)
        assert np.allclose(invert_fit_shifts_translations2[3:5], shifts_rotations[3:5], atol=0.25)
        assert np.allclose(invert_fit_shifts_translations3[3:5], shifts_rotations[3:5], atol=0.25)

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("physical_holes", [False, True])
    def test_fit__perforated_mask(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], physical_holes: bool
    ) -> None:
        """Checks that a perforated mask still permits recovery of a known translation."""

        # Select patches of the curved surface and optionally replace the excluded elevations with nodata
        rows, columns = np.indices(curved_dem.shape)
        valid = np.sin(columns / 4) + np.cos(rows / 5) > 0.3
        reference = curved_dem.copy()
        if physical_holes:
            reference.set_mask(~valid)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Fit on the reference mask with enough samples to represent the remaining gradients
        fitted = method(subsample=1).fit(reference, shifted, inlier_mask=valid, random_state=42)

        # Excluding terrain affects support, while the planted map translation is unchanged
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), atol=0.06)

    @pytest.mark.parametrize("trim", [False, True])
    def test_nuth_kaab__gross_height_outliers(self, curved_dem: Raster, trim: bool) -> None:
        """Checks that sparse gross height outliers do not bias the binned NuthKaab translation."""

        # Contaminate sparse elevations after planting a known map and height translation
        shifted = curved_dem.translate(1.2, -0.8) + 2.5
        shifted.data[::7, ::7] += 1000

        # Fit with the median vertical estimate, with and without residual trimming
        fitted = coreg.NuthKaab(subsample=1, trim_residuals=trim).fit(curved_dem, shifted, random_state=42)

        # Median aspect bins limit the effect of gross errors; interpolation limits subpixel recovery precision
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), atol=0.06)

    @pytest.mark.parametrize("trim_central_statistic", [np.nanmean, np.nanmedian])  # type: ignore
    @pytest.mark.parametrize("trim_spread_statistic", [np.nanstd, nmad])  # type: ignore
    @pytest.mark.parametrize("trim_spread_coverage", [3, 5])  # type: ignore
    @pytest.mark.parametrize("trim_iterative", [True, False])  # type: ignore
    @pytest.mark.parametrize("coreg_method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])  # type: ignore
    def test_coreg_rigid__trimming(
        self,
        coreg_method: Callable[..., AffineCoreg],
        trim_central_statistic: Callable[[NDArrayf], np.floating[Any]],
        trim_spread_statistic: Callable[[NDArrayf], np.floating[Any]],
        trim_spread_coverage: float,
        trim_iterative: bool,
    ) -> None:
        """Checks that rigid methods fit translated data with every supported trimming scheme."""

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (50, 50, 50, 0, 0, 0)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # Due to small sample sizes, no binning
        if coreg_method == coreg.NuthKaab:
            kwargs = {"bin_before_fit": False}
        else:
            kwargs = {}

        # Run co-registration
        subsample_size = 50000
        c = coreg_method(
            subsample=subsample_size,
            trim_residuals=True,
            trim_central_statistic=trim_central_statistic,
            trim_spread_statistic=trim_spread_statistic,
            trim_spread_coverage=trim_spread_coverage,
            trim_iterative=trim_iterative,
            **kwargs,
        )
        c.fit(ref, ref_shifted_rotated, random_state=42)

        # Get invert of resulting matrices
        invert_fit_matrix = invert_matrix(c.to_matrix())
        invert_fit_shifts_translations = translations_rotations_from_matrix(invert_fit_matrix)
        # Check translations and rotations, tolerance 0.1 pixel for NK/LZD and 1 pixel for ICP
        if coreg_method in [coreg.NuthKaab, coreg.LZD]:
            atol_px = 0.1
        else:
            atol_px = 1
        assert np.allclose(invert_fit_shifts_translations[0:3], shifts_rotations[0:3], atol=atol_px * ref.res[0])

    @pytest.mark.parametrize("coreg_method", [coreg.ICP, coreg.CPD])
    def test_coreg_rigid__standardize(self, coreg_method: type[coreg.Coreg]) -> None:

        # Get reference elevation
        ref = self.ref

        # Add artificial shift and rotations
        # (Define small rotations on purpose, so that the "translation only" coregistration is not affected)
        shifts_rotations = (300, 150, 75, 1, 0.5, 0.2)
        matrix = matrix_from_translations_rotations(*shifts_rotations)
        centroid = (ref.bounds.left, ref.bounds.bottom, np.nanmean(ref))
        ref_shifted_rotated = coreg.apply_matrix(ref, matrix=matrix, centroid=centroid)

        # 1/ Run co-registration with standardization
        subsample_size = 1 if coreg_method != coreg.CPD else 300
        c_std = coreg_method(subsample=subsample_size, standardize=True)
        c_std.fit(ref, ref_shifted_rotated, random_state=42)

        # Get invert of resulting matrix
        fit_matrix_std = c_std.meta["outputs"]["affine"]["matrix"]
        invert_fit_shifts_translations_std = translations_rotations_from_matrix(invert_matrix(fit_matrix_std))

        # Check that standardized result are OK
        if coreg_method != coreg.CPD:
            assert np.allclose(invert_fit_shifts_translations_std[:3], shifts_rotations[:3], atol=1 * ref.res[0])
            assert np.allclose(invert_fit_shifts_translations_std[3:5], shifts_rotations[3:5], atol=2 * 10e-2)

        # 2/ Run coregistration without standardization
        c_nonstd = coreg_method(subsample=subsample_size, standardize=False)
        c_nonstd.fit(ref, ref_shifted_rotated, random_state=42)

        fit_matrix_nonstd = c_nonstd.meta["outputs"]["affine"]["matrix"]
        invert_fit_shifts_translations_nonstd = translations_rotations_from_matrix(invert_matrix(fit_matrix_nonstd))

        # Check results are worse for non-standardized
        if coreg_method != coreg.CPD:
            assert np.allclose(invert_fit_shifts_translations_nonstd[:3], shifts_rotations[:3], rtol=1 * ref.res[0])
            assert np.allclose(invert_fit_shifts_translations_nonstd[3:5], shifts_rotations[3:5], atol=2 * 10e-2)

    @pytest.mark.parametrize(
        "coreg_method",
        [
            pytest.param(coreg.VerticalShift(subsample=1), id="VerticalShift"),
            pytest.param(coreg.ICP(method="point-to-plane", subsample=1), id="ICP-p2plane"),
            pytest.param(coreg.ICP(method="point-to-point", subsample=1), id="ICP-p2point"),
            pytest.param(coreg.LZD(subsample=1), id="LZD"),
            pytest.param(coreg.NuthKaab(subsample=500), id="NuthKaab"),
            pytest.param(coreg.CPD(lsg=False, subsample=500), id="CPD"),
            pytest.param(coreg.CPD(lsg=True, subsample=500), id="CPD-LSG"),
        ],
    )
    def test_coreg__symmetry(self, coreg_method: coreg.Coreg) -> None:
        """
        Checks that coregistration is symmetric between point cloud-raster inputs holding the same data points,
        and with ref/tba input (only if the method is symmetric; i.e. not use a gradient preferentialy derived on the
        gridded input).

        Fast methods use all points so raster and point cloud inputs contain the same locations. CPD uses the same
        seeded topk sample to keep the test fast.

        Coregistration output should be:
        - Exactly equal for these 3 inputs: DEM1/DEM2, DEM1/EPC2 and opposite of EPC2/DEM1.
        - Additionally, for methods that do not rely on a gradient (e.g., ICP point-to-point) and are thus fully
            symmetric, the other 3 combinations should also be exactly equal: DEM2/DEM1, EPC1/DEM2 and DEM2/EPC1.
        """

        # Open DEMs and convert them to point clouds
        fn_ref = examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_ref = dem_ref.to_pointcloud()
        epc_tba = dem_tba.to_pointcloud()

        def is_method_symmetric(c: coreg.Coreg) -> bool:
            """Return if coregistration is symmetric."""
            icp_or_cpd = c.__class__.__name__ in ["ICP", "CPD"]

            icp_p2point = (
                "specific" in c.meta["inputs"]
                and "icp_method" in c.meta["inputs"]["specific"]
                and c.meta["inputs"]["specific"]["icp_method"] == "point-to-point"
            )

            cpd_nolsg = (
                "specific" in c.meta["inputs"]
                and "cpd_lsg" in c.meta["inputs"]["specific"]
                and not c.meta["inputs"]["specific"]["cpd_lsg"]
            )

            return isinstance(c, coreg.VerticalShift) or (icp_or_cpd and (icp_p2point or cpd_nolsg))

        input_pairs = [
            # Those three output should always be equal
            ("DEM1/DEM2", dem_ref, dem_tba, 1),
            ("DEM1/EPC2", dem_ref, epc_tba, 1),
            ("EPC2/DEM1", epc_tba, dem_ref, -1),  # Opposite
            # Those are only equal for symmetric methods
            ("DEM2/DEM1", dem_tba, dem_ref, -1),  # Opposite
            ("EPC1/DEM2", epc_ref, dem_tba, 1),
            ("DEM2/EPC1", dem_tba, epc_ref, -1),  # Opposite
        ]
        baseline = None
        for pair in input_pairs:
            # Gradient-based methods are not expected to agree when the reference and aligned inputs are reversed
            if pair[0] in ["EPC2/DEM1", "DEM2/DEM1", "EPC1/DEM2", "DEM2/EPC1"] and not is_method_symmetric(
                coreg_method
            ):
                continue

            # A bounded CPD sample selects different reference points after converting the reference to a point cloud
            if pair[0] == "EPC1/DEM2" and coreg_method.__class__.__name__ == "CPD":
                continue

            # Copy coreg method and fit pair
            m = coreg_method.copy()
            m.fit(pair[1], pair[2], random_state=42)
            # Extract translations and rotations
            matrix = m.to_matrix()
            # We invert the result if we compare to the opposite
            if pair[3] == -1:
                matrix = invert_matrix(matrix)
            tr = coreg.translations_rotations_from_matrix(matrix)

            print(pair[0])
            print(np.asarray(tr))

            if baseline is None:
                baseline = np.asarray(tr)
            else:
                # Vertical shifts agree across representations; point-to-point ICP also agrees exactly
                # when the reference grid and rotation centroid are unchanged
                if isinstance(coreg_method, coreg.VerticalShift) or (
                    coreg_method.__class__.__name__ == "ICP"
                    and is_method_symmetric(coreg_method)
                    and pair[0] == "DEM1/EPC2"
                ):
                    np.testing.assert_allclose(baseline, np.asarray(tr), rtol=0, atol=1e-10)
                    continue

                # Reversing the moving point cloud changes the optimization path slightly even for symmetric methods
                if pair[0] in ["EPC2/DEM1", "DEM2/DEM1", "DEM2/EPC1"] and is_method_symmetric(coreg_method):
                    atol_trans = max(0.25, float(0.3 * np.linalg.norm(baseline[0:3])))
                    atol_rot = max(0.01, float(0.5 * np.linalg.norm(baseline[3:6])))
                else:
                    atol_trans = 0.25
                    atol_rot = 0.01

                # Use absolute differences because some expected translations and rotations are close to zero
                assert np.allclose(baseline[0:3], np.asarray(tr)[0:3], atol=atol_trans)
                assert np.allclose(baseline[3:6], np.asarray(tr)[3:6], atol=atol_rot)

    def test_nuthkaab__no_vertical_shift(self) -> None:
        """Checks that disabling the Nuth and Kääb vertical shift preserves its fitted horizontal shifts."""

        ref, tba = load_examples()[0:2]

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

    @pytest.mark.parametrize(
        "coreg_method", [coreg.VerticalShift, coreg.ICP, coreg.CPD, coreg.NuthKaab, coreg.LZD, coreg.DhMinimize]
    )
    @pytest.mark.parametrize("initial_shift", [(0, 0, 0), (8, 4, 0)])
    def test_coreg_initial_shift(self, coreg_method: coreg.Coreg, initial_shift: tuple[float, float, float]) -> None:
        """
        Test that the initial_shift does not impact fit_and_apply process.
        """

        ref = load_examples()[0]
        shift = (4, 2, 10)
        ref_shifted = ref.translate(shift[0], shift[1]) + shift[2]

        subsample_size = 1000 if coreg_method != coreg.CPD else 500
        c = coreg_method(initial_shift=initial_shift, subsample=subsample_size)
        dem_aligned_is = c.fit_and_apply(ref, ref_shifted, random_state=42)
        assert c.meta["inputs"]["affine"]["initial_shift"] == initial_shift
        output = c.meta["outputs"]["affine"]
        shifts_out_is = [output[k] for k in ["shift_x", "shift_y", "shift_z"] if k in output]  # type: ignore

        c = coreg_method(subsample=subsample_size)
        dem_aligned = c.fit_and_apply(ref, ref_shifted, random_state=42)
        assert "initial_shift" not in c.meta["inputs"]["affine"]
        output = c.meta["outputs"]["affine"]
        shifts_out = [output[k] for k in ["shift_x", "shift_y", "shift_z"] if k in output]  # type: ignore

        # Test shifts (skip CPD: more difficult to constrain)
        if not (initial_shift != (0, 0, 0) and coreg_method == coreg.CPD):
            assert shifts_out_is == pytest.approx(shifts_out, 10 - 6)

        assert dem_aligned_is.transform == dem_aligned.transform
        assert dem_aligned_is.crs == dem_aligned.crs

    def test_pipeline_nested_coregpipeline(self) -> None:
        """Test nested CoregPipeline"""

        nk1 = coreg.NuthKaab()
        nk2 = coreg.NuthKaab()
        nk3 = coreg.NuthKaab()
        nk4 = coreg.NuthKaab()
        pipeline = coreg.CoregPipeline([nk1, coreg.CoregPipeline([nk2, nk3])])
        assert len(pipeline.pipeline) == 3
        for n, nk in enumerate([nk1, nk2, nk3]):
            assert pipeline.pipeline[n] == nk

        pipeline = coreg.CoregPipeline([coreg.CoregPipeline([nk1, nk2]), coreg.CoregPipeline([nk3, nk4])])
        assert len(pipeline.pipeline) == 4
        for n, nk in enumerate([nk1, nk2, nk3, nk4]):
            assert pipeline.pipeline[n] == nk

    @pytest.mark.parametrize("initial_shift", [None, (8, 4, 0)])
    @pytest.mark.parametrize("array", [True, False])
    def test_pipeline_initial_shift(self, initial_shift: tuple[Number, Number, Number] | None, array: bool) -> None:
        """
        Test that the initial_shift in the first coreg of a CoregPipeline works well
        """
        ref = load_examples()[0]
        shift = (10, 2, 0)
        ref_shifted = ref.translate(shift[0], shift[1]) + shift[2]
        shifts = ["shift_x", "shift_y", "shift_z"]
        warnings.filterwarnings("ignore", category=UserWarning)

        if array:
            transform = ref.transform
            crs = ref.crs
            ref = ref.data
        else:
            transform = None
            crs = None

        # Handmade NuthKaab pipeline
        nk_1 = coreg.NuthKaab(initial_shift=initial_shift)
        nk_1.fit(reference_elev=ref, transform=transform, crs=crs, to_be_aligned_elev=ref_shifted, random_state=42)
        shifts_out_nk1 = [nk_1.meta["outputs"]["affine"][k] for k in shifts]  # type: ignore
        output_tmp = nk_1.apply(elev=ref_shifted)
        nk_2 = coreg.NuthKaab(initial_shift=None)
        nk_2.fit(reference_elev=ref, transform=transform, crs=crs, to_be_aligned_elev=output_tmp, random_state=42)
        shifts_out_nk2 = [nk_2.meta["outputs"]["affine"][k] for k in shifts]  # type: ignore

        # Automatic pipeline
        pipeline = coreg.NuthKaab(initial_shift=initial_shift) + coreg.NuthKaab(initial_shift=None)
        if initial_shift is not None:
            assert pipeline.pipeline[0].meta["inputs"]["affine"]["initial_shift"] == initial_shift
        else:
            assert "initial_shift" not in pipeline.pipeline[0].meta["inputs"]["affine"]
        assert "initial_shift" not in pipeline.pipeline[1].meta["inputs"]["affine"]
        pipeline.fit(
            reference_elev=ref,
            to_be_aligned_elev=ref_shifted,
            transform=transform,
            crs=crs,
            random_state=42,
        )
        assert [pipeline.pipeline[0].meta["outputs"]["affine"][k] for k in shifts] == shifts_out_nk1  # type: ignore
        assert [pipeline.pipeline[1].meta["outputs"]["affine"][k] for k in shifts] == shifts_out_nk2  # type: ignore

    @pytest.mark.parametrize(
        "initial_shifts",
        [[None, (1, 1, 0), None], [None, None, (2, 2, 0)], [(3, 3, 0), (4, 4, 0), None]],
    )
    def test_pipeline_initial_shift_errors(self, initial_shifts: list[tuple[int, int, int] | None]) -> None:
        """
        Test that coreg initial_shift management in function on its place in the pipeline
        """

        is1, is2, is3 = initial_shifts

        def test_results(pipeline: CoregPipeline, is1: tuple[int, int, int] | None) -> None:
            if is1 is None:
                assert "initial_shift" not in pipeline.pipeline[0].meta["inputs"]["affine"]
            else:
                assert pipeline.pipeline[0].meta["inputs"]["affine"]["initial_shift"] == is1
            assert "initial_shift" not in pipeline.pipeline[1].meta["inputs"]["affine"]
            assert "initial_shift" not in pipeline.pipeline[2].meta["inputs"]["affine"]

        # Test N&K series
        with pytest.warns(UserWarning, match="No initial shift can be"):
            pipeline = (
                coreg.NuthKaab(initial_shift=is1)
                + coreg.NuthKaab(initial_shift=is2)
                + coreg.NuthKaab(initial_shift=is3)
            )
        test_results(pipeline, is1)

        # CoregPipeline with a list of N&K
        with pytest.warns(UserWarning, match="No initial shift can be"):
            pipeline = coreg.CoregPipeline(
                [
                    coreg.NuthKaab(initial_shift=is1),
                    coreg.NuthKaab(initial_shift=is2),
                    coreg.NuthKaab(initial_shift=is3),
                ]
            )
        test_results(pipeline, is1)

        # Test N&K series with VerticalShift before
        with pytest.warns(UserWarning, match="No initial shift can be"):
            pipeline = (
                coreg.VerticalShift()
                + coreg.NuthKaab(initial_shift=is1)
                + coreg.NuthKaab(initial_shift=is2)
                + coreg.NuthKaab(initial_shift=is3)
            )
        test_results(pipeline, is1=None)

        # Test nested CoregPipeline
        with pytest.warns(UserWarning, match="No initial shift can be"):
            pipeline = coreg.CoregPipeline(
                [
                    coreg.NuthKaab(initial_shift=is1),
                    coreg.CoregPipeline([coreg.NuthKaab(initial_shift=is2), coreg.NuthKaab(initial_shift=is3)]),
                ]
            )
        test_results(pipeline, is1)

        # Test nested CoregPipeline
        with pytest.warns(UserWarning, match="No initial shift can be"):
            pipeline = coreg.CoregPipeline(
                [
                    coreg.CoregPipeline([coreg.NuthKaab(initial_shift=is1), coreg.NuthKaab(initial_shift=is3)]),
                    coreg.CoregPipeline([coreg.NuthKaab(initial_shift=is2), coreg.NuthKaab(initial_shift=is3)]),
                ]
            )
        test_results(pipeline, is1)


class TestLZDErrorAware:
    """Test module for error-aware LZD."""

    @staticmethod
    def _dense_gls(
        xcoord: NDArrayf,
        ycoord: NDArrayf,
        X: NDArrayf,
        Y: NDArrayf,
        sig_Y: NDArrayf,
        lengthscale: float,
    ) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
        """Solve GLS with a dense Gaussian covariance, as a reference against the optional GPyTorch backend."""

        # Construct the same Gaussian kernel and jitter used by the lazy covariance solver
        distances_squared = (xcoord[:, None] - xcoord) ** 2 + (ycoord[:, None] - ycoord) ** 2
        correlation = np.exp(-distances_squared / (2 * lengthscale**2)) + 1e-3 * np.eye(len(Y))
        covariance = correlation * np.outer(sig_Y, sig_Y)

        # Solve the normal equations and propagate the known observation covariance to the coefficients
        weighted_design = np.linalg.solve(covariance, X)
        parameter_covariance = np.linalg.inv(X.T @ weighted_design)
        beta = parameter_covariance @ weighted_design.T @ Y
        return beta, np.sqrt(np.diag(parameter_covariance)), parameter_covariance

    @pytest.mark.parametrize("optimizer", ["gls", "tls"])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_lzd__error_aware_parameters(
        self, monkeypatch: pytest.MonkeyPatch, optimizer: Literal["gls", "tls"], only_translation: bool
    ) -> None:
        """Checks that GLS and TLS recover known LZD parameters and return uncertainties for each fitted parameter."""

        # Create independent coordinates and gradients so all six rigid parameters can be determined
        rng = np.random.default_rng(42)
        x, y, z, gx, gy = rng.normal(size=(5, 80))
        design = np.column_stack((-gx, -gy, np.ones_like(x)))
        expected = np.array([0.7, -0.4, 2.5])
        if not only_translation:
            rotations = np.column_stack((y + gy * z, -x - gx * z, gx * y - gy * x))
            design = np.column_stack((design, rotations))
            expected = np.concatenate((expected, [0.02, -0.01, 0.03]))
        dh = design @ expected
        sigma = rng.uniform(0.1, 0.3, len(x))

        # Use a dense GLS reference so the parameter selection is tested without optional GPyTorch dependencies
        monkeypatch.setattr("xdem.coreg.affine._gls_lazy_gpytorch", self._dense_gls)
        beta, standard_errors, covariance = _lzd_fit_error_propag(
            x=x,
            y=y,
            z=z,
            dh=dh,
            gx=gx,
            gy=gy,
            pixel_size=2,
            sig_h_other=sigma,
            corr_h_other=(lambda distance: np.exp(-((distance / 2) ** 2)), 2),
            sig_h_grid=sigma / 2 if optimizer == "tls" else None,
            force_opti=optimizer,
            only_translation=only_translation,
        )

        # Check the parameter order and covariance dimensions for translations and rigid transforms
        np.testing.assert_allclose(beta, expected, rtol=0, atol=1e-7)
        assert standard_errors.shape == expected.shape
        assert covariance.shape == (len(expected), len(expected))
        assert np.all(np.isfinite(standard_errors))
        assert np.all(standard_errors >= 0)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)

        # Compare GLS covariance in tx, ty, tz order with an independent normal-equation calculation
        if optimizer == "gls":
            distances_squared = (x[:, None] - x) ** 2 + (y[:, None] - y) ** 2
            correlation = np.exp(-distances_squared) + 1e-3 * np.eye(len(x))
            observation_covariance = correlation * np.outer(sigma, sigma)
            expected_covariance = np.linalg.inv(design.T @ np.linalg.solve(observation_covariance, design))
            np.testing.assert_allclose(covariance, expected_covariance, rtol=1e-10)
            np.testing.assert_allclose(standard_errors, np.sqrt(np.diag(expected_covariance)), rtol=1e-10)

    @pytest.mark.parametrize(
        "optimizer, error_source",
        [("gls", "other"), ("gls", "grid"), ("tls", "both"), ("tls", "other"), (None, "other"), (None, "grid")],
    )
    @pytest.mark.parametrize("trim_residuals", [False, True])
    def test_fit__error_aware_translation(
        self,
        curved_dem: Raster,
        monkeypatch: pytest.MonkeyPatch,
        optimizer: str | None,
        error_source: str,
        trim_residuals: bool,
    ) -> None:
        """Checks that error-aware LZD corrects translations without adding rotations, including after trimming."""

        # Translate curved terrain and give the two inputs different, spatially varying elevation errors
        shifted = curved_dem.translate(1.2, -0.8) + 2.5
        rows, columns = np.indices(curved_dem.shape)
        sigma_grid = 0.02 + rows / 2000
        sigma_other = 0.05 + columns / 2000
        if trim_residuals:
            shifted.data[:4, :] += 20
        correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)
        errors: dict[str, Any] = {"force_opti": optimizer}
        if error_source in ("grid", "both"):
            errors.update(sig_ref=sigma_grid, corr_ref=correlation)
        if error_source in ("other", "both"):
            errors.update(sig_tba=sigma_other, corr_tba=correlation)

        # Fit translations using explicit or automatic GLS/TLS selection and the dense GLS reference
        monkeypatch.setattr("xdem.coreg.affine._gls_lazy_gpytorch", self._dense_gls)
        fitted = coreg.LZD(only_translation=True, subsample=200, trim_residuals=trim_residuals).fit(
            curved_dem, shifted, random_state=42, **errors
        )

        # Allow interpolation error in the recovered shifts and require exactly zero rotation
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), rtol=0, atol=0.03)
        np.testing.assert_array_equal(fitted.to_matrix()[:3, :3], np.eye(3))
        standard_errors = fitted.meta["outputs"]["fitorbin"]["fit_perr"]
        assert standard_errors.shape == (3,)
        assert np.all(np.isfinite(standard_errors))
        assert np.all(standard_errors >= 0)


class TestNuthKaabDesignMatrix:
    """Test the Nuth and Kääb design matrix, optimizer selection and parameter conversions."""

    def test_design_matrix__shape(self) -> None:
        """Checks that the design matrix has one row per aspect and three coefficient columns."""
        aspect = np.linspace(0, 2 * np.pi, 200)
        X = _design_matrix_nuth_kaab(aspect)
        assert X.shape == (200, 3)

    def test_design_matrix__columns(self) -> None:
        """Checks that the design matrix contains cosine, sine and intercept columns."""
        aspect = np.linspace(0, 2 * np.pi, 300)
        X = _design_matrix_nuth_kaab(aspect)
        assert np.allclose(X[:, 0], np.cos(aspect))
        assert np.allclose(X[:, 1], np.sin(aspect))
        assert np.all(X[:, 2] == 1.0)

    def test_design_matrix__roundtrip_with_fit_func(self) -> None:
        """Checks that OLS recovers the coefficients used by the original trigonometric model."""
        rng = np.random.default_rng(42)
        aspect = rng.uniform(0, 2 * np.pi, 500)
        a, b, c = 3.5, 1.2, -0.8
        y = _nuth_kaab_fit_func(aspect, a, b, c)  # type: ignore[arg-type]

        X = _design_matrix_nuth_kaab(aspect)
        A, B, fitted_c = np.linalg.lstsq(X, y, rcond=None)[0]

        # Easting = a*sin(b) = B, northing = a*cos(b) = A
        assert A == pytest.approx(a * np.cos(b), abs=1e-6)
        assert B == pytest.approx(a * np.sin(b), abs=1e-6)
        assert fitted_c == pytest.approx(c, abs=1e-6)

    def test_design_matrix__prediction_matches_fit_func(self) -> None:
        """Checks that matrix predictions reproduce the original trigonometric model."""
        rng = np.random.default_rng(0)
        aspect = rng.uniform(0, 2 * np.pi, 1000)
        a, b, c = 2.1, 0.7, 1.3

        y_expected = _nuth_kaab_fit_func(aspect, a, b, c)  # type: ignore[arg-type]
        A, B = a * np.cos(b), a * np.sin(b)
        X = _design_matrix_nuth_kaab(aspect)
        y_pred = X @ np.array([A, B, c])

        assert np.allclose(y_pred, y_expected)

    def test_nuth_kaab__fit_optimizer_selection(self) -> None:
        """Checks that NuthKaab records OLS by default and preserves a user optimizer."""

        # Instantiate both optimizer paths without fitting
        coreg_ols = coreg.NuthKaab()
        coreg_curve_fit = coreg.NuthKaab(fit_optimizer=scipy.optimize.curve_fit)

        # Check that metadata and info expose the optimizer that each instance will run
        assert coreg_ols.meta["inputs"]["fitorbin"]["fit_optimizer"] == "ols"
        assert "ols" in coreg_ols.info(as_str=True)
        assert coreg_curve_fit.meta["inputs"]["fitorbin"]["fit_optimizer"] is scipy.optimize.curve_fit
        assert "curve_fit" in coreg_curve_fit.info(as_str=True)

    def test_nuth_kaab__fit_optimizer_output_consistency(self) -> None:
        """Checks that OLS and curve_fit produce consistent X, Y and Z shifts."""

        ref, tba = load_examples()[0:2]
        coreg_ols = coreg.NuthKaab(fit_optimizer="ols", max_iterations=1).fit(ref, tba, random_state=42)
        coreg_curve_fit = coreg.NuthKaab(fit_optimizer=scipy.optimize.curve_fit, max_iterations=1).fit(
            ref, tba, random_state=42
        )

        shift_keys = ["shift_x", "shift_y", "shift_z"]
        shifts_ols = [coreg_ols.meta["outputs"]["affine"][key] for key in shift_keys]  # type: ignore
        shifts_curve_fit = [coreg_curve_fit.meta["outputs"]["affine"][key] for key in shift_keys]  # type: ignore
        assert shifts_ols == pytest.approx(shifts_curve_fit)


class TestAffineErrors:
    """Test module for errors/warnings of affine coregistrations."""

    def test_dh_minimize__deprecation_notice(self) -> None:
        """Checks DhMinimize deprecation."""

        with pytest.warns(DeprecationWarning, match="LZD with linearized=False, only_translation=True") as records:
            coreg.DhMinimize()
        assert "custom fit_minimizer wrapping Nelder-Mead" in str(records[0].message)
        assert "fit_loss_func=nmad" in str(records[0].message)
        assert "Z shift separately from the median residual" in str(records[0].message)
        assert records[0].filename == __file__

    @pytest.mark.parametrize(
        "method, options",
        [
            (coreg.LZD, {"linearized": False}),
            (coreg.ICP, {"linearized": False}),
            (coreg.ICP, {"method": "point-to-point"}),
        ],
    )
    def test_fit__error_ols_nonlinear(self, method: Callable[..., AffineCoreg], options: dict[str, Any]) -> None:
        """Checks an error is raised when OLS is requested for a nonlinear coreg."""

        with pytest.raises(ValueError, match="requires a linearized LZD or point-to-plane ICP model"):
            method(fit_minimizer="ols", **options)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    def test_fit__error_ols_loss(self, method: Callable[..., AffineCoreg]) -> None:
        """Checks an error is raised for OLS with a different loss than linear."""

        with pytest.raises(ValueError, match="requires fit_loss_func='linear'"):
            method(fit_minimizer="ols", fit_loss_func="soft_l1")

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    @pytest.mark.parametrize("only_translation", [False, True])
    @pytest.mark.parametrize("linearized", [False, True])
    @pytest.mark.parametrize("trim", [False, True])
    def test_lzd__error_rank_deficient_design_matrix(
        self,
        curved_dem: Raster,
        surface: str,
        only_translation: bool,
        linearized: bool,
        trim: bool,
    ) -> None:
        """Checks an error is raised for LZD parameters that flat or planar terrain cannot determine."""

        # We define a constant surface or an exact plane
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Check failure
        method = coreg.LZD(subsample=1, only_translation=only_translation, linearized=linearized, trim_residuals=trim)
        with pytest.raises(ValueError, match="LZD cannot determine .*the surface may be too flat or planar"):
            method.fit(reference, shifted, random_state=42)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    @pytest.mark.parametrize("bin_before_fit", [False, True])
    def test_nuth_kaab__error_constant_aspect(self, curved_dem: Raster, surface: str, bin_before_fit: bool) -> None:
        """Checks an error is raised when NuthKaab has no slopes or no variation in aspect."""

        # Build a flat surface or a ramp (gradient has exactly one aspect)
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Check failure
        method = coreg.NuthKaab(subsample=1, bin_before_fit=bin_before_fit)
        with pytest.raises(ValueError, match="no more valid values|the surface may be too flat or planar"):
            method.fit(reference, shifted, random_state=42)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    @pytest.mark.parametrize("only_translation", [False, True])
    def test_icp__error_planar_p2plane(self, curved_dem: Raster, surface: str, only_translation: bool) -> None:
        """Checks an error is raised when planar normals cannot constrain the requested ICP transform."""

        # Create equal surface normals across a flat DEM or a ramp
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Check failure
        method = coreg.ICP(subsample=1, only_translation=only_translation)
        with pytest.raises(ValueError, match="ICP cannot determine .*the surface may be too flat or planar"):
            method.fit(reference, reference, random_state=42)

    @pytest.mark.parametrize("pair", ["raster-raster", "raster-point", "point-raster"])
    def test_fit__error_nonoverlap_mask(self, curved_dem: Raster, pair: str) -> None:
        """Checks an error is raised for an inlier mask that not overlap with elevation inputs."""

        # Create a non-overlaping mask
        mask = Raster.from_array(np.ones(curved_dem.shape, dtype=bool), curved_dem.transform, curved_dem.crs)
        mask = mask.translate(10_000, 0)
        reference: Any = curved_dem
        shifted: Any = curved_dem + 2.5
        if pair == "raster-point":
            shifted = shifted.to_pointcloud(data_name="z")
        elif pair == "point-raster":
            reference = curved_dem.to_pointcloud(data_name="z")

        # Check error
        with pytest.raises(ValueError, match="inlier_mask.*no inliers"):
            coreg.VerticalShift().fit(reference, shifted, inlier_mask=mask)

"""Functions to test the affine coregistrations."""

from __future__ import annotations

import logging
import os.path
import sys
import warnings
from functools import partial
from typing import Any, Callable, Literal, cast
from unittest.mock import patch

import geopandas as gpd
import geoutils
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import scipy.ndimage
import scipy.optimize
import scipy.spatial
from geoutils import Raster, Vector
from geoutils._typing import Number
from geoutils.raster.transformation import _translate
from geoutils.stats import nmad
from geoutils.stats.variography import VariogramModel
from scipy.ndimage import binary_dilation
from scipy.stats import chi2

from xdem import DEM, EPC, coreg, examples
from xdem._typing import NDArrayf
from xdem.coreg.affine import (
    AffineCoreg,
    _cpd_estep_classic,
    _cpd_fit,
    _cpd_iteration_step,
    _cpd_mstep_lsg_least_squares,
    _cpd_precompute,
    _design_matrix_nuth_kaab,
    _fit_gls,
    _get_centroid_scale,
    _gls_approximate,
    _gls_inducing_factor,
    _gls_lazy_gpytorch,
    _gls_neighbors,
    _icp_fit,
    _icp_fit_func,
    _icp_iteration_step,
    _iterate_method,
    _lsg_build_whitened_terms,
    _lzd_fit_gls_tls,
    _lzd_tls_observation_jacobians,
    _lzd_tls_spatial_covariance,
    _nuth_kaab_fit_func,
    _reproject_horizontal_shift_samecrs,
    _SampledErrors,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)
from xdem.coreg.base import InFitOrBinDict
from xdem.coreg.pipeline import CoregPipeline

# We list and ID methods to reuse consistently in tests
_RIGID_METHODS = [
    coreg.LZD,
    pytest.param(partial(coreg.LZD, linearized=False), id="LZD-nonlinear"),
    coreg.ICP,
    pytest.param(partial(coreg.ICP, linearized=False), id="ICP-nonlinear"),
    pytest.param(partial(coreg.ICP, method="point-to-point"), id="ICP-point"),
    pytest.param(partial(coreg.CPD, lsg=False), id="CPD"),
    pytest.param(coreg.CPD, id="CPD-LSG"),
]


_GLS_TLS_OPTIONS = [
    pytest.param({"force_opti": "gls", "gls_method": "approximate"}, id="gls-approximate"),
    pytest.param({"force_opti": "gls", "gls_method": "exact"}, id="gls-exact"),
    pytest.param({"force_opti": "gls", "gls_method": "inducing"}, id="gls-inducing"),
    pytest.param({"force_opti": "tls", "tls_method": "independent"}, id="tls-independent"),
    pytest.param({"force_opti": "tls", "tls_method": "spatial"}, id="tls-spatial"),
]


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


@pytest.fixture
def small_dem(request: pytest.FixtureRequest) -> Raster:
    """Create small DEM with slope/aspect variations in X/Y for tests."""

    # Create waves or smooth terrain at native pixel coordinates in a projected CRS
    rows, columns = np.indices((16, 18))
    x = (columns - 8.5) * 2
    y = (7.5 - rows) * 3
    if request.param == "smooth":
        values = 40 + 0.1 * x + 0.05 * y + 0.002 * x * y + 0.5 * np.sin(x / 12) + 0.3 * np.cos(y / 14)
    else:
        values = 40 + 5 * np.sin(x / 6) + 3 * np.cos(y / 8) + 2 * np.sin((x + y) / 7)
    transform = rio.Affine(2, 0, 500_000, 0, -3, 5_100_000)
    return Raster.from_array(values, transform, 32632, nodata=np.nan, area_or_point="Point")


def _point_samples(dem: Raster, count: int | None = 900) -> geoutils.PointCloud:
    """Convert a DEM to points at its pixel centers."""

    return dem.to_pointcloud(data_name="z", subsample=1 if count is None else count, random_state=3)


def _dense_gls(
    xcoord: NDArrayf,
    ycoord: NDArrayf,
    X: NDArrayf,
    Y: NDArrayf,
    errors: _SampledErrors,
) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
    """Solve GLS with a dense covariance, as a reference against the optional GPyTorch backend."""

    # Combine independent input covariances using the input correlation functions
    distances = np.sqrt((xcoord[:, None] - xcoord) ** 2 + (ycoord[:, None] - ycoord) ** 2)
    covariance = np.zeros((len(Y), len(Y)))
    for sigma, correlation in ((errors[0], errors[1]), (errors[2], errors[3])):
        if sigma is None:
            continue
        if correlation is None:
            covariance += np.diag(sigma**2)
        else:
            correlations = correlation[0](distances) + 1e-3 * np.eye(len(Y))
            covariance += correlations * np.outer(sigma, sigma)

    # Solve the normal equations and propagate the known observation covariance to the coefficients
    weighted_design = np.linalg.solve(covariance, X)
    parameter_covariance = np.linalg.inv(X.T @ weighted_design)
    beta = parameter_covariance @ weighted_design.T @ Y
    return beta, np.sqrt(np.diag(parameter_covariance)), parameter_covariance


def _tls_observation_case(only_translation: bool, spatial: bool, residual: float = 0) -> dict[str, Any]:
    """Create sampled TLS observations with known parameters and separate error models."""

    # Create curved observations with independently varying gradients for a full-rank fit
    rows, columns = np.indices((6, 8))
    x, y = (columns.ravel() - 3.5) * 2, (2.5 - rows.ravel()) * 2
    coordinates = np.column_stack((x, y))
    z = 8 * np.sin(x / 7) + 5 * np.cos(y / 9) + 3 * np.sin((x + y) / 11)
    gx = 8 / 7 * np.cos(x / 7) + 3 / 11 * np.cos((x + y) / 11)
    gy = -5 / 9 * np.sin(y / 9) + 3 / 11 * np.cos((x + y) / 11)
    expected = np.array([0.3, -0.2, 0.5, 0.003, -0.002, 0.004])
    if only_translation:
        expected = expected[:3]
    design = np.column_stack((-gx, -gy, np.ones(len(x))))
    if not only_translation:
        design = np.column_stack((design, y + gy * z, -x - gx * z, gx * y - gy * x))
    dh = design @ expected + residual * np.sin(np.arange(len(x)) / 5)

    # Define the model on observations, including the locations used by centered gradient differences
    from geoutils.stats.variography import VariogramModel

    correlation = VariogramModel("gaussian", effective_range=6, partial_sill=1) if spatial else None
    other_model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.008, correlation)])
    grid_model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.005, correlation)])
    corr = (grid_model.predict_correlation, 0.0) if spatial else None
    errors = (np.full(len(x), 0.008), corr, np.full(len(x), 0.005), corr)
    offsets = np.array([[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]])
    field_coordinates, field_indexes = np.unique(
        np.concatenate([coordinates + offset for offset in offsets]), axis=0, return_inverse=True
    )

    def observations(
        grid_error: NDArrayf, other_error: NDArrayf
    ) -> tuple[NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf]:
        """Perturb sampled heights and their centered gradient differences from the same field."""

        field = grid_error[field_indexes].reshape(5, len(x))
        return (
            x,
            y,
            z + other_error,
            dh + field[0] - other_error,
            gx + (field[1] - field[2]) / 4,
            gy + (field[3] - field[4]) / 4,
        )

    def fit(grid_error: NDArrayf, other_error: NDArrayf) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
        """Fit the perturbed 1D observations with the existing TLS estimator."""

        return _lzd_fit_gls_tls(
            *observations(grid_error, other_error), 2, *errors, only_translation=only_translation, force_opti="tls"
        )

    zero_grid, zero_other = np.zeros(len(field_coordinates)), np.zeros(len(x))
    covariance_inputs = dict(
        beta=fit(zero_grid, zero_other)[0],
        inputs=observations(zero_grid, zero_other),
        errors=errors,
        pixel_size=2,
        error_structure=(other_model, grid_model),
        error_predictors=({}, {}),
    )
    return dict(
        covariance_inputs=covariance_inputs,
        fit=fit,
        expected=expected,
        field_coordinates=field_coordinates,
        field_indexes=field_indexes,
        coordinates=coordinates,
        grid_size=len(zero_grid),
        other_size=len(x),
    )


class TestAffineCoreg:
    """Test module for shared affine fitting, application, sampling and pipeline behavior."""

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

    @pytest.mark.parametrize("shape", [(5, 7), (6, 8), (5, 8)])
    @pytest.mark.parametrize("angle", [0, 10, 90])
    @pytest.mark.parametrize("missing", ["finite", "nan", "masked"])
    def test_get_centroid_scale__raster(self, shape: tuple[int, int], angle: float, missing: str) -> None:
        """Checks that grid centroids match coordinate medians and scale is unchanged by rotation."""

        # Create odd/even grids and remove an edge value to check missing heights
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        if missing == "nan":
            values[0, 0] = np.nan
        elif missing == "masked":
            values = np.ma.array(values, mask=values == 0)
        transform = rio.Affine.translation(500_000, 7_000_000) * rio.Affine.rotation(angle) * rio.Affine.scale(2, -3)

        # Derive centroid and scale before any random sampling
        centroid, scale = _get_centroid_scale(values, transform)

        # Compare the centroid to explicit coordinates on these small grids
        rows, columns = np.indices(shape)
        x, y = transform * (columns, rows)
        heights = np.ma.masked_invalid(values).compressed()
        expected_centroid = (np.median(x), np.median(y), np.median(heights))
        np.testing.assert_allclose(centroid, expected_centroid, rtol=0, atol=1e-8)

        # Compare scale to the original unrotated coordinates with rectangular pixels
        expected_scale = np.mean([nmad(2 * columns), nmad(-3 * rows), nmad(heights)])
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-12)

    @pytest.mark.parametrize("method", [coreg.NuthKaab, coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("area_or_point", [None, "Area", "Point"])
    def test_fit__identity(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        area_or_point: Literal["Area", "Point"] | None,
    ) -> None:
        """Checks that fitting a DEM to itself gives zero shifts, for every pixel interpretation and coreg method."""

        # Set the interpretation
        reference = curved_dem.copy()
        reference.set_area_or_point(area_or_point)

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
        reference.set_area_or_point(area_or_point)
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

        # Use Point interpretation to check that fit() and apply() preserve its metadata
        reference = curved_dem.copy()
        reference.set_area_or_point("Point")
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Fit and apply, with/without resampling onto the input grid
        options = {} if method == coreg.NuthKaab else {"only_translation": True}
        fitted = method(subsample=1, **options).fit(reference, shifted, random_state=42)
        aligned = fitted.apply(shifted, resample=resample)

        # Check the pixel interpretation of the input and applied rasters
        assert aligned.area_or_point == "Point"
        assert shifted.area_or_point == "Point"

    @pytest.mark.parametrize("method", [coreg.VerticalShift, coreg.NuthKaab, coreg.LZD])
    @pytest.mark.parametrize("shape", [(40, 44), (33, 37)])
    def test_fit__different_grids_z_shift(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], shape: tuple[int, int]
    ) -> None:
        """
        Checks that fit() estimates the inverse Z shift and zero X/Y shifts when DEMs sample
        the same terrain on grids with different origins.
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

        # Check the inverse Z shift and zero X/Y shifts
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
    def test_fit__cropped_mask_consistency(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], mask_type: str
    ) -> None:
        """
        Checks that cropping elevations or masking the same area (that is cropped) results in the same fitting.
        (No subsample is used for exact equality)
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
    def test_fit__partial_dem_overlap_z_shift(self, curved_dem: Raster, method: Callable[..., AffineCoreg]) -> None:
        """
        Checks that fit() estimates only a Z shift for DEMs with partially overlapping map extents.
        """

        # Mask reference DEM right half
        reference = curved_dem.copy()
        reference.set_mask(np.indices(reference.shape)[1] >= 22)

        # Copy its valid elevations to the second DEM right half and add a Z shift
        shifted_values = np.full(curved_dem.shape, np.nan)
        shifted_values[:, 22:] = curved_dem.data[:, :22] + 0.35

        # Move the second grid left so both valid halves cover the same map locations
        transform = curved_dem.transform * rio.Affine.translation(-22, 0)
        shifted = Raster.from_array(shifted_values, transform, curved_dem.crs, nodata=np.nan, area_or_point="Area")

        # Fit the DEMs over their common valid area
        fitted = method(subsample=1).fit(reference, shifted, random_state=42)

        # Check the inverse Z shift and zero X/Y shifts
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.35), atol=1e-5)

    @pytest.mark.parametrize(
        "coreg_method, shift",
        [
            (coreg.NuthKaab, (9.492646, 2.841536, -2.058883)),
            (coreg.DhMinimize, (10.164111, 2.600748, -1.995853)),
            (coreg.LZD, (10.281134, 2.300993, -1.817791)),
            (coreg.ICP, (9.142172, 2.879442, -1.519637)),
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

    @pytest.mark.parametrize("small_dem", ["waves", "smooth"], indirect=True)
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize(
        "method, options, fit_kwargs",
        [
            (coreg.LZD, {}, {}),
            (coreg.LZD, {"linearized": False}, {}),
            (
                coreg.LZD,
                {"initial_shift": (-0.3, 0.2)},
                {"error_structure": geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01)])},
            ),
            (coreg.LZD, {"fit_loss_func": "soft_l1"}, {}),
            (coreg.LZD, {"trim_residuals": True}, {}),
            (coreg.LZD, {"linearized": False, "trim_residuals": True}, {}),
            (coreg.ICP, {}, {}),
            (coreg.ICP, {"linearized": False}, {}),
            (coreg.ICP, {"picky": False}, {}),
            (coreg.ICP, {"anisotropic": None}, {}),
            (coreg.ICP, {"anisotropic": "per_axis"}, {}),
            (coreg.ICP, {"fit_minimizer": scipy.optimize.least_squares}, {}),
            (coreg.ICP, {"fit_loss_func": "soft_l1"}, {}),
            (coreg.ICP, {"trim_residuals": True}, {}),
            (coreg.ICP, {"method": "point-to-point", "max_iterations": 20}, {}),
            (coreg.ICP, {"method": "point-to-point", "trim_residuals": True}, {}),
            (coreg.ICP, {"standardize": False}, {}),
            (
                coreg.ICP,
                {"initial_shift": (-0.3, 0.2)},
                {"error_structure": geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01)])},
            ),
            (coreg.ICP, {"sampling_strategy": "independent"}, {}),
            (coreg.ICP, {"sampling_strategy": "iterative_same_xy"}, {}),
            (coreg.CPD, {"lsg": False, "estep_knearest": None}, {}),
            (coreg.CPD, {"lsg": False, "estep_knearest": 64}, {}),
            (coreg.CPD, {"lsg": True, "estep_knearest": None}, {}),
            (coreg.CPD, {"lsg": True, "estep_knearest": 64}, {}),
            (coreg.CPD, {"initial_shift": (-0.3, 0.2)}, {}),
            (coreg.CPD, {"sampling_strategy": "independent"}, {}),
            (coreg.CPD, {"sampling_strategy": "iterative_same_xy"}, {}),
        ],
    )
    def test_fit_and_apply__dem_epc_rigid_transform(
        self,
        small_dem: Raster,
        method: Callable[..., AffineCoreg],
        options: dict[str, Any],
        fit_kwargs: dict[str, Any],
        only_translation: bool,
    ) -> None:
        """Checks that all affine coregistration with main options recover a translation+rotation on DEM/EPC."""

        # Cnovert DEM to points (to avoid any interpolation and loss of info), and transform
        points = _point_samples(small_dem, count=None)
        angles = (0, 0, 0) if only_translation else (0.4, -0.3, 0.5)
        matrix = matrix_from_translations_rotations(0.6, -0.4, 0.8, *angles)
        centroid = (points.gdf.geometry.x.mean(), points.gdf.geometry.y.mean(), points.data.mean())
        moving = EPC(coreg.apply_matrix(points.gdf, matrix=matrix, centroid=centroid), data_name="z")

        # Fit and apply
        method_options = dict(
            only_translation=only_translation,
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_rotation=1e-6,
        )
        method_options.update(options)
        fitted = method(**method_options)
        with patch("xdem.coreg.affine._fit_gls", wraps=_fit_gls) as fit_gls:
            aligned = fitted.fit_and_apply(small_dem, moving, fit_kwargs=fit_kwargs, random_state=41)

        # Check equality
        original = np.column_stack((points.gdf.geometry.x, points.gdf.geometry.y, points.data))
        restored = np.column_stack((aligned.gdf.geometry.x, aligned.gdf.geometry.y, aligned.data))
        np.testing.assert_allclose(restored, original, rtol=0, atol=1e-6)
        assert fitted.meta["outputs"]["iterative"]["converged"]

        # Check that fit_and_apply() forwards the error model to the GLS solver
        if fit_kwargs:
            assert fit_gls.called

    @pytest.mark.parametrize("method", _RIGID_METHODS)
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize("pair", ["point-raster", "geodataframe", "geometry-z"])
    @pytest.mark.parametrize("initial_shift", [None, (-0.3, 0.2)])
    def test_fit_and_apply__dem_epc_input_formats(
        self,
        small_dem: Raster,
        method: Callable[..., AffineCoreg],
        pair: str,
        initial_shift: tuple[float, float] | None,
    ) -> None:
        """Checks that all point input formats (Z axis or data_name) and initial shifts recover the same translation."""

        # Translate points and modify to specific point input format
        points = _point_samples(small_dem, count=None)
        matrix = matrix_from_translations_rotations(0.6, -0.4, 0.8)
        moving = EPC(coreg.apply_matrix(points.gdf, matrix=matrix), data_name="z")
        reference_input: Any = moving
        moving_input: Any = small_dem
        expected = np.column_stack((moving.gdf.geometry.x, moving.gdf.geometry.y, moving.data))
        if pair == "geodataframe":
            reference_input = moving.gdf.rename(columns={"z": "height"})
        elif pair == "geometry-z":
            moving_input = EPC.from_xyz(*expected.T, crs=small_dem.crs, use_z=True)
            reference_input = small_dem
            expected = np.column_stack((points.gdf.geometry.x, points.gdf.geometry.y, points.data))

        # Fit and apply
        fitted = method(
            only_translation=True,
            initial_shift=initial_shift,
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_rotation=1e-6,
        )
        aligned = fitted.fit_and_apply(
            reference_input,
            moving_input,
            z_name="height" if pair == "geodataframe" else "z",
            resample=False,
            random_state=41,
        )

        # Convert raster output to points, account for its float32 elevation values
        if isinstance(aligned, Raster):
            aligned = _point_samples(aligned, count=None)
            expected[:, 2] = expected[:, 2].astype(np.float32)

        # Check equality, and that Z heights are preserved
        actual = np.column_stack((aligned.gdf.geometry.x, aligned.gdf.geometry.y, aligned.data))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
        if pair == "geometry-z":
            assert aligned.data_name is None
            assert aligned.gdf.has_z.all()

    @pytest.mark.parametrize("method", [partial(coreg.ICP, method="point-to-point"), partial(coreg.CPD, lsg=False)])
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize("height_columns", [("z", "z"), ("elevation", "height"), (None, None)])
    @pytest.mark.parametrize("sampling_strategy", ["independent", "same_xy", "iterative_same_xy"])
    def test_fit_and_apply__epc_epc_rigid_transform(
        self,
        small_dem: Raster,
        method: Callable[..., AffineCoreg],
        height_columns: tuple[str | None, str | None],
        sampling_strategy: str,
    ) -> None:
        """Checks that two EPC inputs can recover an artificial rigid transform with different height column setup."""

        # Store same coordinates in a given height column (Z or data_name)
        points = _point_samples(small_dem, count=None)
        matrix = matrix_from_translations_rotations(0.6, -0.4, 0.8, 0.4, -0.3, 0.5)
        centroid = (points.gdf.geometry.x.mean(), points.gdf.geometry.y.mean(), points.data.mean())
        moving = EPC(coreg.apply_matrix(points.gdf, matrix=matrix, centroid=centroid), data_name="z")
        original = np.column_stack((points.gdf.geometry.x, points.gdf.geometry.y, points.data))
        transformed = np.column_stack((moving.gdf.geometry.x, moving.gdf.geometry.y, moving.data))
        reference = EPC.from_xyz(
            *original.T, crs=small_dem.crs, data_name=height_columns[0], use_z=height_columns[0] is None
        )
        moving = EPC.from_xyz(
            *transformed.T, crs=small_dem.crs, data_name=height_columns[1], use_z=height_columns[1] is None
        )

        # Fit and apply
        fitted = method(
            sampling_strategy=sampling_strategy,
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_rotation=1e-6,
        )
        aligned = fitted.fit_and_apply(reference, moving, random_state=41)

        # Check equality
        actual = np.column_stack((aligned.gdf.geometry.x, aligned.gdf.geometry.y, aligned.data))
        np.testing.assert_allclose(actual, original, rtol=0, atol=1e-6)
        assert aligned.data_name == height_columns[1]

    @pytest.mark.parametrize("method", _RIGID_METHODS)
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize("geometry", ["rotated", "gap"])
    def test_fit__raster_geometry(self, small_dem: Raster, method: Callable[..., AffineCoreg], geometry: str) -> None:
        """Checks that rotated raster grids or NaNs still estimate an added rigid transform."""

        # Rotate grid or set a patch of NaNs
        values = np.asarray(small_dem.data).copy()
        transform = small_dem.transform
        if geometry == "rotated":
            transform = transform * rio.Affine.rotation(10)
        else:
            values[6:9, 8:11] = np.nan
        reference = Raster.from_array(values, transform, small_dem.crs, nodata=np.nan, area_or_point="Point")
        points = _point_samples(reference, count=None)
        matrix = matrix_from_translations_rotations(0.6, -0.4, 0.8, 0.4, -0.3, 0.5)
        centroid = (points.gdf.geometry.x.mean(), points.gdf.geometry.y.mean(), points.data.mean())
        moving = EPC(coreg.apply_matrix(points.gdf, matrix=matrix, centroid=centroid), data_name="z")

        # Fit and apply
        fitted = method(
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_rotation=1e-6,
        ).fit(reference, moving, random_state=41)
        aligned = fitted.apply(moving)

        # Check equality
        original = np.column_stack((points.gdf.geometry.x, points.gdf.geometry.y, points.data))
        actual = np.column_stack((aligned.gdf.geometry.x, aligned.gdf.geometry.y, aligned.data))
        np.testing.assert_allclose(actual, original, rtol=0, atol=1e-6)

    @pytest.mark.parametrize(
        "method, step",
        [
            (coreg.ICP, _icp_iteration_step),
            (partial(coreg.CPD, lsg=False), _cpd_iteration_step),
            (coreg.CPD, _cpd_iteration_step),
        ],
    )
    def test_fit__iterative_sampling_budget(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        step: Callable[..., Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that iterative sampling stops at max_iterations."""

        # Translate a DEM
        moving = curved_dem.translate(2.2, -1.6) + 1
        calls = []

        # Count optimization steps
        def count_step(*args: Any, **kwargs: Any) -> Any:
            calls.append(None)
            return step(*args, **kwargs)

        monkeypatch.setattr("xdem.coreg.affine." + step.__name__, count_step)

        # Run two iterations with resampling after each step
        fitted = method(
            sampling_strategy="iterative_same_xy",
            subsample=100,
            max_iterations=2,
            tolerance_translation=1e-14,
            tolerance_rotation=1e-14,
        ).fit(curved_dem, moving, random_state=41)

        # Both optimization work and reported rows follow the same total budget
        assert len(calls) == 2
        assert fitted.meta["outputs"]["iterative"]["last_iteration"] == 2
        assert len(fitted.meta["outputs"]["iterative"]["iteration_stats"]) == 2
        assert not fitted.meta["outputs"]["iterative"]["converged"]

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
            (coreg.ICP, (9.1421721, 2.8794423, -1.5196373, 0.0100435, -0.0073668, -0.0087533)),
            (coreg.LZD, (10.281134, 2.300993, -1.817791, 0.0079446, -0.0081343, -0.0057847)),
            (coreg.CPD, (8.243680, 3.946520, -1.703947, 0.0086867, -0.0094364, -0.0014735)),
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

    @pytest.mark.parametrize("use_tqdm", [False, True])
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
        self,
        value: float,
        expected: str,
        use_tqdm: bool,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Checks that iteration output rounds values properly for display."""

        # Select tqdm output or logging fallback
        if use_tqdm:
            pytest.importorskip("tqdm")
        else:
            monkeypatch.setitem(sys.modules, "tqdm", None)
            monkeypatch.setitem(sys.modules, "tqdm.auto", None)

        def iteration_step(matrix: NDArrayf) -> tuple[NDArrayf, dict[str, float], None]:
            """Return the input offsets without changing the matrix."""

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
        printed = capsys.readouterr().out if use_tqdm else caplog.text

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

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize("side", ["ref", "tba", "both"])
    @pytest.mark.parametrize("optimizer", [None, "gls", "tls"])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_fit__gls_tls(
        self,
        small_dem: Raster,
        monkeypatch: pytest.MonkeyPatch,
        method: Callable[..., AffineCoreg],
        side: str,
        optimizer: str | None,
        only_translation: bool,
    ) -> None:
        """Checks that LZD and ICP runs properly with GLS/TLS optimizers, applied to ref/tba, and only translation."""

        from xdem.coreg import affine

        # Define error structure, and input ("ref"/"tba") to which it is applied
        model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01)])
        models = {name: model for name in ("ref", "tba") if side in (name, "both")}
        calls = []

        # We record GLS/TLS solves
        gls, tls = affine._fit_gls, affine._fit_tls

        def fit_gls(*args: Any, **kwargs: Any) -> Any:
            calls.append("gls")
            return gls(*args, **kwargs)

        def fit_tls(*args: Any, **kwargs: Any) -> Any:
            calls.append("tls")
            return tls(*args, **kwargs)

        # Fit a simple Z translation, with automatic or forced GLS/TLS selection
        monkeypatch.setattr(affine, "_fit_gls", fit_gls)
        monkeypatch.setattr(affine, "_fit_tls", fit_tls)
        fitted = method(only_translation=only_translation, subsample=40).fit(
            small_dem, small_dem + 0.2, error_structure=models, force_opti=optimizer, random_state=12
        )

        # Check fit, and the solver calls
        expected = optimizer or ("tls" if "ref" in models else "gls")
        assert calls and set(calls) == {expected}
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.2), rtol=0, atol=1e-7)
        np.testing.assert_allclose(fitted.to_rotations(), 0, rtol=0, atol=1e-7)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    def test_fit__tls_methods(self, small_dem: Raster, method: Callable[..., AffineCoreg]) -> None:
        """Checks that TLS methods fit the same transformation and report different spatial uncertainty."""

        pytest.importorskip("skgstat")

        # Create correlated errors in the DEM used to derive gradients
        correlation = VariogramModel("gaussian", effective_range=6, partial_sill=1)
        errors = {"ref": geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01, correlation)])}
        coreg_method = method(only_translation=True, subsample=40)

        # Fit both covariance methods using the same observations
        spatial = coreg_method.copy().fit(small_dem, small_dem + 0.2, error_structure=errors, random_state=12)
        independent = coreg_method.copy().fit(
            small_dem, small_dem + 0.2, error_structure=errors, tls_method="independent", random_state=12
        )

        # Check an unchanged point estimate and the additional covariance between observations
        np.testing.assert_array_equal(spatial.to_matrix(), independent.to_matrix())
        spatial_std = spatial.meta["outputs"]["fitorbin"]["fit_perr"][:3]
        independent_std = independent.meta["outputs"]["fitorbin"]["fit_perr"][:3]
        assert np.all(spatial_std > independent_std)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize("active_dims", [None, (0,), (1,)])
    @pytest.mark.parametrize("backend", ["approximate", "exact", "inducing"])
    def test_fit__gls_components(
        self, small_dem: Raster, method: Callable[..., AffineCoreg], active_dims: tuple[int, ...] | None, backend: str
    ) -> None:
        """Checks that every GLS backend sums the separate spatial components from both inputs."""

        pytest.importorskip("skgstat")

        pytest.importorskip("linear_operator")
        if backend == "inducing":
            pytest.importorskip("gpytorch")

        # Create different correlation scales in both elevation inputs
        gaussian = VariogramModel("gaussian", effective_range=6, partial_sill=1, active_dims=active_dims)
        exponential = VariogramModel("exponential", effective_range=3, partial_sill=1, active_dims=active_dims)
        models = {
            "ref": geoutils.ErrorStructure(
                [
                    geoutils.ErrorComponent("short", 0.01, exponential),
                    geoutils.ErrorComponent("long", 0.02, gaussian),
                ]
            ),
            "tba": geoutils.ErrorStructure(
                [
                    geoutils.ErrorComponent("measurement", 0.03),
                    geoutils.ErrorComponent("long", 0.01, gaussian),
                ]
            ),
        }

        # Fit the mapped models with GLS, including reference errors in the elevation differences
        fitted = method(only_translation=True, subsample=1).fit(
            small_dem, small_dem + 0.2, error_structure=models, force_opti="gls", gls_method=backend, random_state=12
        )

        # Compare the small exact normal equations, including the solvers' covariance regularization
        x, y = small_dem.coords(grid=True)
        coordinates = np.column_stack((x.ravel(), y.ravel()))
        covariance = sum(model.to_covariance_matrix(coordinates) for model in models.values())
        covariance += 1e-3 * (0.01**2 + 0.02**2 + 0.01**2) * np.eye(small_dem.data.size)
        gy, gx = np.gradient(np.asarray(small_dem.data), -small_dem.res[1], small_dem.res[0])
        design = np.column_stack((-gx.ravel(), -gy.ravel(), np.ones(small_dem.data.size)))
        expected = np.linalg.inv(design.T @ np.linalg.solve(covariance, design))
        np.testing.assert_allclose(fitted.meta["outputs"]["fitorbin"]["fit_pcov"][:3, :3], expected, rtol=1e-7)
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.2), rtol=0, atol=1e-8)

    @pytest.mark.parametrize("small_dem", ["waves"], indirect=True)
    @pytest.mark.parametrize(
        "solver, options",
        [
            (coreg.LZD, {}),
            (coreg.ICP, {"sampling_strategy": "same_xy"}),
            (coreg.ICP, {"sampling_strategy": "independent"}),
        ],
    )
    @pytest.mark.parametrize("predictor_kind", ["array", "raster"])
    @pytest.mark.parametrize("point_input", [False, True])
    def test_fit__cosampled_predictors(
        self,
        small_dem: Raster,
        monkeypatch: pytest.MonkeyPatch,
        solver: Callable[..., AffineCoreg],
        options: dict[str, Any],
        point_input: bool,
        predictor_kind: str,
    ) -> None:
        """Checks that both inputs' predictors follow the same sampled rows as their elevations."""

        from xdem.coreg import affine

        # Create different predictors with the same name and missing values on each input
        reference = small_dem
        moving = reference + 0.2
        quality = np.linspace(0, 1, reference.data.size).reshape(reference.shape)
        first_quality, second_quality = quality.copy(), 1 - quality
        first_quality[3, 3] = np.nan
        second_quality[4, 4] = np.nan
        if point_input:
            x, y = moving.coords(grid=True)
            moving = geoutils.PointCloud.from_xyz(x.ravel(), y.ravel(), moving.data.ravel(), crs=reference.crs)
            second_quality = second_quality.ravel()
        statistics = pd.DataFrame(
            {"std": [0.01, 0.02], "count": [100, 100]}, index=pd.Index([0.0, 1.0], name="quality")
        )
        magnitude = geoutils.ErrorMagnitude.variable_from_grouped_stats(statistics)
        model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", magnitude)])
        predictors: dict[str, dict[str, Any]] = {"ref": {"quality": first_quality}, "tba": {"quality": second_quality}}
        if predictor_kind == "raster":
            for values in predictors.values():
                values["quality"] = reference.copy(new_array=values["quality"].reshape(reference.shape))
        mask = np.zeros(reference.shape, dtype=bool)
        mask[2:-2, 2:-2] = True

        # Record the existing shared preparation without changing its observation selection
        sampled = []
        sample = affine._subsample_rst_pts
        derive = affine._dem_normals_curvature
        normals = []

        def record_samples(*args: Any, **kwargs: Any) -> Any:
            """Record the jointly sampled observations and auxiliaries."""

            result = sample(*args, **kwargs)
            sampled.append(result)
            return result

        def record_normals(*args: Any, **kwargs: Any) -> Any:
            """Record each derivation of reference normals during the fit."""

            normals.append(True)
            return derive(*args, **kwargs)

        monkeypatch.setattr(affine, "_subsample_rst_pts", record_samples)
        monkeypatch.setattr(affine, "_dem_normals_curvature", record_normals)
        fitted = solver(only_translation=True, subsample=1, **options).fit(
            reference,
            moving,
            inlier_mask=mask,
            error_structure={"ref": model, "tba": model},
            error_predictors=predictors,
            random_state=12,
        )

        # Check one preparation call and predictor values derived independently from sampled coordinates
        assert len(sampled) == 1
        assert len(normals) == (1 if solver is coreg.ICP else 0)
        reference_sample, moving_sample, auxiliary = sampled[0]
        assert auxiliary is not None
        x0, y0 = reference.ij2xy(0, 0)
        for side, observations in (("ref", reference_sample), ("tba", moving_sample)):
            column = (observations[0] - x0) / reference.res[0]
            row = (y0 - observations[1]) / reference.res[1]
            expected = (row * reference.shape[1] + column) / (reference.data.size - 1)
            expected = expected if side == "ref" else 1 - expected
            actual = auxiliary[f"error_{side}_quality"]
            assert actual.shape == (observations.shape[1],)
            assert np.all(np.isfinite(actual))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -0.2), rtol=0, atol=1e-7)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize(
        "backend, options",
        [
            pytest.param(None, {}, id="default"),
            ("exact", {}),
            ("inducing", {}),
            pytest.param("inducing", {"gls_rank": 32, "gls_neighbors": 7}, id="inducing-options"),
        ],
    )
    def test_fit__gls_backends(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        monkeypatch: pytest.MonkeyPatch,
        only_translation: bool,
        backend: str | None,
        options: dict[str, Any],
    ) -> None:
        """Checks that GLS backends fit all observations with bounded covariance and the requested settings."""

        pytest.importorskip("linear_operator")
        if backend == "inducing":
            pytest.importorskip("gpytorch")

        # 1/ Create a vertical offset
        source = curved_dem + 1
        rows, columns = np.indices(curved_dem.shape)
        count = curved_dem.data.size

        # Record covariance allocations and approximation settings
        evaluated_shapes, ranks, requested_neighbors = [], [], []

        def correlation(distance: NDArrayf) -> NDArrayf:
            assert distance.shape != (count, count)
            evaluated_shapes.append(distance.shape)  # Record here
            return np.exp(-distance / 4)

        def record_factor(
            coordinates: NDArrayf, magnitude: NDArrayf, correlation: Callable[..., NDArrayf], rank: int
        ) -> NDArrayf:
            ranks.append(rank)  # Record here
            return _gls_inducing_factor(coordinates, magnitude, correlation, rank)

        def record_neighbors(coordinates: NDArrayf, neighbors: int, block_size: int) -> Any:
            requested_neighbors.append(neighbors)  # Record here
            return _gls_neighbors(coordinates, neighbors, block_size)

        # Guard the optional exact solver and its dense conversions separately from correlation evaluation
        if backend is None:

            def fail_exact(**kwargs: Any) -> Any:
                """Fail if the default fit requests the optional exact backend."""

                pytest.fail("Default GLS must use the approximation backend.")

            monkeypatch.setattr("xdem.coreg.affine._gls_lazy_gpytorch", fail_exact)
        if backend in (None, "exact"):
            monkeypatch.setitem(sys.modules, "gpytorch", None)

        from linear_operator.operators import LinearOperator

        dense = LinearOperator.to_dense

        def guarded_dense(operator: Any) -> Any:
            """Fail if the fit requests the complete covariance of its fitting observations."""

            assert operator.shape[-2:] != (count, count)
            return dense(operator)

        monkeypatch.setattr(LinearOperator, "to_dense", guarded_dense)
        monkeypatch.setattr("xdem.coreg.affine._gls_inducing_factor", record_factor)
        monkeypatch.setattr("xdem.coreg.affine._gls_neighbors", record_neighbors)

        # 2/ Fit all observations with one/two error fields, and with default/custom settings
        fit_options = dict(options, sig_tba=0.1 + rows / 1000, corr_tba=(correlation, 4), force_opti="gls")
        if backend is not None:
            fit_options["gls_method"] = backend
        if backend != "exact":
            fit_options["sig_ref"] = 0.03 + columns / 2000
        if backend == "inducing":
            fit_options["corr_ref"] = (correlation, 4)
        fitted = method(only_translation=only_translation, subsample=1).fit(
            curved_dem, source, random_state=2, **fit_options
        )

        # 3/ Check expected output
        # We recover exactly the translation/rotation
        np.testing.assert_allclose(fitted.to_translations(), (0, 0, -1), rtol=0, atol=1e-7)
        np.testing.assert_allclose(fitted.to_matrix()[:3, :3], np.eye(3), rtol=0, atol=1e-8)
        assert fitted.meta["outputs"]["random"]["subsample_final"] == count
        standard_errors = fitted.meta["outputs"]["fitorbin"]["fit_perr"]
        count_parameters = 3 if only_translation else 6
        # We get positive uncertainties and bounded covariance allocations
        active_errors = standard_errors[:count_parameters]
        assert np.all(np.isfinite(active_errors) & (active_errors > 0))
        np.testing.assert_array_equal(standard_errors[count_parameters:], 0)
        assert evaluated_shapes
        if backend == "inducing":
            assert ranks and set(ranks) == {options.get("gls_rank", 512)}
            assert len(ranks) % 2 == 0
            assert requested_neighbors and set(requested_neighbors) == {options.get("gls_neighbors", 60)}
        if backend != "exact":
            assert any(len(shape) == 3 for shape in evaluated_shapes)


class TestSampledParameterCovariance:
    """Test module for helper functions used to estimate parameter covariance for both ICP/LZD."""

    def test_sampled_parameter_covariance__coincident_observations(self) -> None:
        """Checks that distinct sampled observations at identical coordinates have independent noise."""

        from xdem.coreg.affine import _propagate_sampled_covariance

        # Create three observations at one location with independent standard deviations of two
        coordinates = np.zeros((3, 2))
        sensitivity = np.ones((1, 3))
        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 2.0)])

        # Propagate their sum and compare with three independent variances
        covariance = _propagate_sampled_covariance(coordinates, sensitivity, errors)
        np.testing.assert_array_equal(covariance, [[3 * 2.0**2]])

    def test_sampled_parameter_covariance__components(self) -> None:
        """Checks that native covariance products match explicit heteroscedastic and correlated component matrices."""

        pytest.importorskip("skgstat")

        from xdem.coreg.affine import _propagate_sampled_covariance

        # Create 527 observations so point covariance has a shorter final block
        shape = (17, 31)
        rows, columns = np.indices(shape)
        coordinates = np.column_stack((2 * columns.ravel(), -3 * rows.ravel()))
        quality = np.linspace(0, 1, len(coordinates))

        # Create variable magnitudes and spatial components for the covariance comparison
        statistics = pd.DataFrame({"std": [0.5, 1.5], "count": [100, 100]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = geoutils.ErrorMagnitude.variable_from_grouped_stats(statistics)
        errors = geoutils.ErrorStructure(
            [
                geoutils.ErrorComponent("measurement", 0.2),
                geoutils.ErrorComponent(
                    "terrain", magnitude, VariogramModel("gaussian", effective_range=6, partial_sill=1)
                ),
                geoutils.ErrorComponent(
                    "long_range", 0.05, VariogramModel("exponential", effective_range=12, partial_sill=1)
                ),
            ]
        )
        sensitivity = np.random.default_rng(19).normal(size=(6, len(coordinates))) / len(coordinates)

        # Propagate native component covariances without changing observation locations or point IDs
        actual = _propagate_sampled_covariance(coordinates, sensitivity, errors, {"quality": quality})

        # Compare with the public error model's complete small covariance matrix
        expected_native = errors.to_covariance_matrix(coordinates, predictors={"quality": quality.ravel()})
        expected = sensitivity @ expected_native @ sensitivity.T
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)


class TestAffineCoregErrors:
    """Test module for errors/warnings common to all affine coregistrations."""

    def test_from_translations__error_nonfinite(self) -> None:
        """Checks an error is raised for a nonfinite translation in from_translations()."""

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

    @pytest.mark.parametrize("method", _RIGID_METHODS)
    @pytest.mark.parametrize("reverse", [False, True])
    def test_fit__equivalent_pixel_interpretations(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], reverse: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that equivalent "Area" and "Point" rasters align without introducing a half pixel shift."""

        # Change only the interpretation and enable warnings for differing metadata
        monkeypatch.setitem(geoutils.config, "warn_area_or_point", True)
        point_dem = curved_dem.copy()
        point_dem.set_area_or_point("Point")
        np.testing.assert_array_equal(point_dem.coords(grid=True), curved_dem.coords(grid=True))
        reference, moving = (point_dem, curved_dem) if reverse else (curved_dem, point_dem)

        # Check metadata warning during fit
        fitted = method(only_translation=True, subsample=1, tolerance_translation=1e-6)
        with pytest.warns(UserWarning, match="One raster has a pixel interpretation"):
            fitted.fit(reference, moving, random_state=41)

        # Should not be any translation (only the warning above)
        np.testing.assert_allclose(fitted.to_translations(), [0, 0, 0], rtol=0, atol=1e-5)

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

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize(
        "options, message",
        [
            ({"gls_method": "unknown"}, "gls_method"),
            ({"gls_neighbors": 0}, "gls_neighbors"),
            ({"gls_neighbors": 1.5}, "gls_neighbors"),
            ({"gls_neighbors": True}, "gls_neighbors"),
            ({"gls_rank": -1}, "gls_rank"),
            ({"gls_rank": 1.5}, "gls_rank"),
        ],
    )
    @pytest.mark.parametrize("gls_method", ["approximate", "exact", "inducing"])
    def test_fit__error_invalid_gls_options(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        monkeypatch: pytest.MonkeyPatch,
        options: dict[str, Any],
        message: str,
        gls_method: str,
    ) -> None:
        """Checks an error is raised for invalid GLS settings before any calculation."""

        # Fail if input validation reaches the gradient calculation
        def fail_gradients(**kwargs: Any) -> Any:
            pytest.fail("Invalid GLS settings must fail before gradients are calculated.")

        monkeypatch.setattr("xdem.coreg.affine._lzd_aux_vars", fail_gradients)
        monkeypatch.setattr("xdem.coreg.affine._dem_normals_curvature", fail_gradients)

        with pytest.raises(ValueError, match=message):
            method().fit(curved_dem, curved_dem + 1, **{"gls_method": gls_method, **options})

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize("options", [{"linearized": False}, {"fit_loss_func": "soft_l1"}, {"trim_residuals": True}])
    def test_fit__error_unsupported_error_model(
        self, curved_dem: Raster, method: Callable[..., AffineCoreg], options: dict[str, Any]
    ) -> None:
        """Checks an error is raised when error structure is used for ICP/LZD without linearized=True or linear loss."""

        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01)])
        fitted = method(**options)
        with pytest.raises(NotImplementedError, match="linearized|trimming"):
            fitted.fit(curved_dem, curved_dem + 0.2, error_structure=errors)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    @pytest.mark.parametrize(
        "options, native, error_type, message",
        [
            ({"force_opti": "unknown"}, False, ValueError, "force_opti"),
            ({"force_opti": "ols"}, True, ValueError, "force_opti"),
            ({"force_opti": "gls"}, False, ValueError, "requires observation errors"),
            ({"force_opti": "tls"}, False, ValueError, "requires observation errors"),
            ({"tls_method": "unknown"}, True, ValueError, "tls_method"),
            ({"tls_method": "spatial"}, False, ValueError, "requires input models"),
            ({"predictors": {}}, True, TypeError, "mappings keyed"),
            ({"error_applied_to": "tba"}, True, TypeError, "mappings keyed"),
        ],
    )
    def test_fit__error_invalid_error_options(
        self,
        curved_dem: Raster,
        method: Callable[..., AffineCoreg],
        options: dict[str, Any],
        native: bool,
        error_type: type[Exception],
        message: str,
    ) -> None:
        """Checks an error is raised for incompatible error inputs and GLS/TLS settings."""

        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01)]) if native else None
        with pytest.raises(error_type, match=message):
            method().fit(curved_dem, curved_dem + 0.2, error_structure=errors, **options)

    @pytest.mark.parametrize("method", [coreg.LZD, coreg.ICP])
    def test_fit__error_correlation_dimensions(self, curved_dem: Raster, method: Callable[..., AffineCoreg]) -> None:
        """Checks an error is raised when a correlation model requests an unavailable coordinate dimension."""

        # Create a correlation on a third axis while coregistration supplies X/Y locations
        correlation = VariogramModel("gaussian", effective_range=6, partial_sill=1, active_dims=(2,))
        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.01, correlation)])
        with pytest.raises(ValueError, match="X/Y coordinates"):
            method().fit(curved_dem, curved_dem + 0.2, error_structure=errors)


class TestVerticalShift:
    """Test module for VerticalShift coreg."""

    # Reuse raster/point inputs from the shared affine tests
    ref = TestAffineCoreg.ref
    all_fit_args = TestAffineCoreg.all_fit_args

    @pytest.mark.parametrize("subsample, expected", [(1, 1760), (0.5, 880), (37, 37)])
    def test_vertical_shift__sample_count(self, curved_dem: Raster, subsample: float, expected: int) -> None:
        """Checks that VerticalShift reports the number of sampled observations rather than coordinate axes."""

        # Add a constant Z offset
        shifted = curved_dem + 2.5

        # Fit with subsampling
        fitted = coreg.VerticalShift(subsample=subsample).fit(curved_dem, shifted, random_state=42)

        # Check sample count and Z offset
        assert fitted.meta["outputs"]["random"]["subsample_final"] == expected
        assert fitted.to_translations() == pytest.approx((0, 0, -2.5))

    def test_fit__error_unconverged_uncertainty(self, curved_dem: Raster) -> None:
        """Checks an error is raised for analytical LZD uncertainty before the fit converges."""

        # Limit a translated fit to one iteration, before its correction reaches tolerance
        shifted = curved_dem + 0.2
        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.1)])
        method = coreg.LZD(only_translation=True, subsample=60, max_iterations=1)

        # Check that an unfinished fit cannot report analytical uncertainty
        with pytest.raises(ValueError, match="Analytical LZD requires a converged fit"):
            method.fit(curved_dem, shifted, error_structure={"tba": errors}, random_state=12)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    def test_vertical_shift__planar_height(self, curved_dem: Raster, surface: str) -> None:
        """Checks that a Z-only fit works on flat/planar terrain."""

        # Create a flat surface or ramp
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Fit the Z offset
        fitted = coreg.VerticalShift().fit(reference, reference + 2.5)

        # Check the Z offset
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


class TestNuthKaab:
    """Test module for NuthKaab coregistration."""

    def test_nuthkaab__no_vertical_shift(self) -> None:
        """Checks that disabling the Nuth and Kääb vertical shift preserves horizontal shifts."""

        ref, tba = load_examples()[0:2]

        # Compare NK method with/without applying vertical shift
        coreg_method1 = coreg.NuthKaab(vertical_shift=True)
        coreg_method2 = coreg.NuthKaab(vertical_shift=False)

        coreg_method1.fit(ref, tba, random_state=42)
        coreg_method2.fit(ref, tba, random_state=42)

        # Recover shifts
        matrix1 = coreg_method1.to_matrix()
        matrix2 = coreg_method2.to_matrix()

        # Assert vertical shift is 0 for the 2nd coreg method
        assert matrix2[2, 3] == 0

        # Assert horizontal shifts are the same
        matrix2[2, 3] = matrix1[2, 3]
        assert np.array_equal(matrix1, matrix2)

    @pytest.mark.parametrize("count", [200, 300])
    def test_design_matrix__columns(self, count: int) -> None:
        """Checks that the design matrix contains cosine/sine/intercept columns."""

        # Create aspects and calculate the 3 design columns
        aspect = np.linspace(0, 2 * np.pi, count)
        X = _design_matrix_nuth_kaab(aspect)

        # Check one row per aspect and the cosine, sine and intercept columns
        assert X.shape == (count, 3)
        assert np.allclose(X[:, 0], np.cos(aspect))
        assert np.allclose(X[:, 1], np.sin(aspect))
        assert np.all(X[:, 2] == 1.0)

    @pytest.mark.parametrize(
        "random_state, count, coefficients", [(42, 500, (3.5, 1.2, -0.8)), (0, 1000, (2.1, 0.7, 1.3))]
    )
    def test_design_matrix__roundtrip_with_fit_func(
        self, random_state: int, count: int, coefficients: tuple[float, float, float]
    ) -> None:
        """Checks that OLS with a design matrix recovers the original coefficients."""

        # Create observations with _nuth_kaab_fit_func()
        rng = np.random.default_rng(random_state)
        aspect = rng.uniform(0, 2 * np.pi, count)
        a, b, c = coefficients
        y = _nuth_kaab_fit_func(aspect, a, b, c)  # type: ignore[arg-type]

        # Check design matrix predictions against _nuth_kaab_fit_func() before fitting
        X = _design_matrix_nuth_kaab(aspect)
        expected_coefficients = np.array([a * np.cos(b), a * np.sin(b), c])
        np.testing.assert_allclose(X @ expected_coefficients, y)
        A, B, fitted_c = np.linalg.lstsq(X, y, rcond=None)[0]

        # Easting = a*sin(b) = B, northing = a*cos(b) = A
        assert A == pytest.approx(a * np.cos(b), abs=1e-6)
        assert B == pytest.approx(a * np.sin(b), abs=1e-6)
        assert fitted_c == pytest.approx(c, abs=1e-6)

    def test_nuth_kaab__fit_optimizer_selection(self) -> None:
        """Checks that default OLS and an input curve_fit optimizer produce consistent X, Y and Z shifts."""

        # Instantiate both optimizer paths without fitting
        coreg_ols = coreg.NuthKaab(max_iterations=1)
        coreg_curve_fit = coreg.NuthKaab(fit_optimizer=scipy.optimize.curve_fit, max_iterations=1)

        # Check that metadata and info expose the optimizer that each instance will run
        assert coreg_ols.meta["inputs"]["fitorbin"]["fit_optimizer"] == "ols"
        assert "ols" in coreg_ols.info(as_str=True)
        assert coreg_curve_fit.meta["inputs"]["fitorbin"]["fit_optimizer"] is scipy.optimize.curve_fit
        assert "curve_fit" in coreg_curve_fit.info(as_str=True)

        # Fit both optimizers to the same example rasters
        ref, tba = load_examples()[0:2]
        coreg_ols.fit(ref, tba, random_state=42)
        coreg_curve_fit.fit(ref, tba, random_state=42)

        # Check that both optimizers produce consistent X, Y and Z shifts
        shift_keys = ["shift_x", "shift_y", "shift_z"]
        shifts_ols = [coreg_ols.meta["outputs"]["affine"][key] for key in shift_keys]  # type: ignore
        shifts_curve_fit = [coreg_curve_fit.meta["outputs"]["affine"][key] for key in shift_keys]  # type: ignore
        assert shifts_ols == pytest.approx(shifts_curve_fit)

    @pytest.mark.parametrize("trim", [False, True])
    def test_nuth_kaab__height_outliers(self, curved_dem: Raster, trim: bool) -> None:
        """Checks that sparse height outliers do not bias the binned NuthKaab translation."""

        # Translate the DEM and add outliers
        shifted = curved_dem.translate(1.2, -0.8) + 2.5
        shifted.data[::7, ::7] += 1000

        # Fit with and without trimming
        fitted = coreg.NuthKaab(subsample=1, trim_residuals=trim).fit(curved_dem, shifted, random_state=42)

        # Allow for subpixel interpolation error
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), atol=0.06)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    @pytest.mark.parametrize("bin_before_fit", [False, True])
    def test_nuth_kaab__error_constant_aspect(self, curved_dem: Raster, surface: str, bin_before_fit: bool) -> None:
        """Checks an error is raised when NuthKaab has no slopes or no variation in aspect."""

        # Create a flat surface or ramp with constant aspect
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Check failure
        method = coreg.NuthKaab(subsample=1, bin_before_fit=bin_before_fit)
        with pytest.raises(ValueError, match="no more valid values|the surface may be too flat or planar"):
            method.fit(reference, shifted, random_state=42)


class TestICP:
    """Test module for ICP coregistration."""

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    def test_icp__planar_p2point_identity(self, curved_dem: Raster, surface: str) -> None:
        """Checks that point-to-point ICP can exactly match a planar DEM."""

        # Create a flat surface or ramp
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Fit with point-to-point
        fitted = coreg.ICP(method="point-to-point", subsample=1).fit(reference, reference, random_state=42)

        # Check the identity transform
        np.testing.assert_allclose(fitted.to_matrix(), np.eye(4), atol=1e-7)

    @pytest.mark.parametrize(
        "axis_weights, plane_distance",
        [
            pytest.param(np.ones(3), 1 / np.sqrt(5), id="equal-axis-weights"),
            pytest.param(np.array([4.0, 1.0, 1.0]), 1 / np.sqrt(2), id="unequal-axis-weights"),
        ],
    )
    @pytest.mark.parametrize("linearized", [True, False])
    def test_icp_fit_func__weighted_plane_distance(
        self, linearized: bool, axis_weights: NDArrayf, plane_distance: float
    ) -> None:
        """Checks that ICP with axis and point weights."""

        # We use the tilted plane 2X + Z = 0 to test axis weights
        # From (0, 0, 1), the shortest path to the plane moves in both X and Z
        # With equal weights, this distance is 1/sqrt(2**2 + 1**2) = 1/sqrt(5)
        # An X weight of 4 counts an X change of 1 as a distance of 2, since 4*1**2 = 2**2
        # Using X_scaled = 2X, the plane equation is X_scaled + Z = 0
        # The point stays at (0, 0, 1), so its weighted distance is now 1/sqrt(1**2 + 1**2) = 1/sqrt(2)

        # Create offsets on plane 2X + Z = 0, above it, below it and twice as far above it
        displacements = np.array([[1, 2, -2], [0, 0, 1], [0, 0, -1], [1, 0, 0]], dtype=float).T
        normals = np.tile([2.0, 0.0, 1.0], (4, 1)).T / np.sqrt(5)
        reference = np.zeros_like(displacements)
        point_weights = np.array([1.0, 4.0, 9.0, 1.0])
        translation = np.array([0.01, -0.02, 0.03])
        moving = displacements - translation[:, None]

        # Evaluate the translation
        actual = _icp_fit_func(
            (reference, moving, normals),
            translation[0],
            translation[1],
            translation[2],
            0,
            0,
            0,
            method="point-to-plane",
            linearized=linearized,
            weights=point_weights,
            axis_weights=axis_weights,
        )

        # Due to weights, result should be zero on the plane, positive above it and negative below it
        # We calculate 2X + Z for each offset, then convert these values to expected distances
        equation_values = 2 * displacements[0] + displacements[2]
        plane_distances = plane_distance * equation_values
        expected = plane_distances * np.sqrt(point_weights)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("minimizer", ["ols", scipy.optimize.least_squares])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_icp_fit__weighted_linear_solution(
        self, minimizer: str | Callable[..., Any], only_translation: bool
    ) -> None:
        """Checks that direct and iterative ICP solvers match an independently weighted linear solution."""

        # Create noisy point pairs with unequal weights
        rng = np.random.default_rng(52)
        moving = rng.normal(size=(3, 30))
        normals = rng.normal(size=moving.shape)
        normals /= np.linalg.norm(normals, axis=0)
        reference = moving + rng.normal(scale=0.1, size=moving.shape)
        point_weights = np.linspace(0.2, 3, moving.shape[1])
        axis_weights = np.array([0.5, 1.0, 4.0])
        options = cast(InFitOrBinDict, {"fit_minimizer": minimizer, "fit_loss_func": "linear"})

        # Fit the point pairs
        fitted, _ = _icp_fit(
            reference, moving, normals, point_weights, axis_weights, "point-to-plane", options, only_translation, True
        )

        # Solve weighted normal equations independently, to check for consistency
        design = normals.T
        if not only_translation:
            design = np.column_stack((design, np.cross(moving.T, normals.T)))
        row_scale = np.sqrt(point_weights / np.sum(normals**2 / axis_weights[:, None], axis=0))
        target = np.sum(normals * (reference - moving), axis=0)
        expected, _, _, _ = np.linalg.lstsq(design * row_scale[:, None], target * row_scale, rcond=None)
        rotation = np.zeros(3) if only_translation else expected[3:]
        expected_matrix = matrix_from_translations_rotations(
            expected[0], expected[1], expected[2], rotation[0], rotation[1], rotation[2], use_degrees=False
        )
        np.testing.assert_allclose(fitted, expected_matrix, rtol=0, atol=1e-8)

    @pytest.mark.parametrize("method", ["point-to-point", "point-to-plane"])
    def test_icp_fit__trimmed_point_weights(self, method: Literal["point-to-point", "point-to-plane"]) -> None:
        """Checks that trimming removes point pairs and selects their weights properly before fitting."""

        # Create noisy point pairs with one outlier and unequal weights
        rng = np.random.default_rng(11)
        moving = rng.normal(size=(3, 40))
        reference = moving + np.array([0.01, -0.02, 0.03])[:, None] + rng.normal(scale=1e-4, size=moving.shape)
        reference[:, -1] += [20, 30, 40]
        normals = rng.normal(size=moving.shape)
        normals /= np.linalg.norm(normals, axis=0)
        weights = np.linspace(0.5, 2, moving.shape[1])
        options = cast(InFitOrBinDict, {"fit_minimizer": scipy.optimize.least_squares, "fit_loss_func": "linear"})
        trim_options = cast(
            InFitOrBinDict,
            dict(
                options,
                trim_residuals=True,
                trim_central_statistic=np.nanmedian,
                trim_spread_statistic=nmad,
                trim_spread_coverage=10,
                trim_iterative=False,
            ),
        )

        # Compare trimming with manual outlier removal
        actual, _ = _icp_fit(reference, moving, normals, weights, None, method, trim_options, True, False)
        expected, _ = _icp_fit(
            reference[:, :-1], moving[:, :-1], normals[:, :-1], weights[:-1], None, method, options, True, False
        )

        # Check matching translations
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("change", ["error_scale", "sample_order"])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("repeated", [False, True])
    @pytest.mark.parametrize("correlated", [False, True, pytest.param("x", id="x-only")])
    @pytest.mark.parametrize("error_options", _GLS_TLS_OPTIONS)
    def test_icp_fit__gls_tls(
        self, only_translation: bool, repeated: bool, correlated: bool | str, change: str, error_options: dict[str, Any]
    ) -> None:
        """
        Checks that GLS/TLS return the same fit after doubling error magnitudes or reordering samples.

        Covariance must grow by 4 when error magnitudes do by 2, and stay the same when samples are reordered.

        To verify the exactitude of uncertainty propagation in GLS/TLS, refer to test_uncertainty/test_coreg.py
        where we compare to numerical simulations.
        """

        if correlated:
            pytest.importorskip("skgstat")

        if error_options.get("gls_method") in ("approximate", "exact"):
            pytest.importorskip("linear_operator")
        elif error_options.get("gls_method") == "inducing":
            pytest.importorskip("gpytorch")

        # Create point pairs with a rotated moving Z axis
        rng = np.random.default_rng(52)
        count = 30
        reference = rng.normal(scale=4, size=(3, 24 if repeated else count))
        matched = np.arange(count) % reference.shape[1]
        selected = rng.permutation(count)
        current_moving = reference[:, matched] + rng.normal(scale=0.02, size=(3, count))
        rotation = scipy.spatial.transform.Rotation.from_euler("xyz", (0.2, -0.15, 0.1)).as_matrix()
        original_moving = np.empty_like(current_moving)
        original_moving[:, selected] = rotation.T @ (current_moving - np.array([5, -3, 1])[:, None])
        normals = np.vstack((rng.normal(scale=0.3, size=(2, reference.shape[1])), np.ones(reference.shape[1])))
        normals /= np.linalg.norm(normals, axis=0)
        normals = normals[:, matched]

        # We create and assign spatially varying height errors to both inputs
        models, predictors, statistics_by_side = {}, {}, {}
        correlation = (
            VariogramModel(
                "gaussian", effective_range=4, partial_sill=1, active_dims=(0,) if correlated == "x" else None
            )
            if correlated
            else None
        )
        for side, observations, magnitudes in (
            ("ref", reference, [0.01, 0.04]),
            ("tba", original_moving, [0.03, 0.08]),
        ):
            statistics = pd.DataFrame(
                {"std": magnitudes, "count": [100, 100]}, index=pd.Index([0.0, 1.0], name="quality")
            )
            magnitude = geoutils.ErrorMagnitude.variable_from_grouped_stats(statistics)
            models[side] = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", magnitude, correlation)])
            predictors[side] = {"quality": np.linspace(0, 1, observations.shape[1])}
            statistics_by_side[side] = statistics
        scale = 3.0
        fixed, moving = reference[:, matched] / scale, current_moving / scale
        settings = cast(InFitOrBinDict, {"fit_minimizer": "ols", "fit_loss_func": "linear"})

        # Fit with original sample indexes and the rotated Z axis
        fit_options: dict[str, Any] = dict(
            ref=fixed,
            tba=moving,
            norms=normals,
            weights=None,
            axis_weights=np.array([1.0, 2.0, 3.0]),
            method="point-to-plane",
            params_fit_or_bin=settings,
            only_translation=only_translation,
            linearized=True,
            error_structure=models,
            error_predictors=predictors,
            error_samples=(reference, original_moving),
            error_indices=(matched, selected),
            error_scale=scale,
            moving_axis=rotation[:, 2],
            pixel_size=(2.0, 3.0),
            **error_options,
        )
        fitted, covariance = _icp_fit(**fit_options)
        assert covariance is not None
        assert np.all(np.linalg.eigvalsh(covariance) > 0)

        if change == "error_scale":
            # Double input errors, so parameter covariance must grow by 2**2
            magnitude_multiplier = 2
            scaled_models = {}
            for side, statistics in statistics_by_side.items():
                magnitude = geoutils.ErrorMagnitude.variable_from_grouped_stats(statistics, scale=magnitude_multiplier)
                scaled_models[side] = geoutils.ErrorStructure(
                    [geoutils.ErrorComponent("measurement", magnitude, correlation)]
                )
            fit_options["error_structure"] = scaled_models
            expected_covariance = magnitude_multiplier**2 * covariance
        else:
            # Reorder original samples and predictors while selecting the same point pairs
            reference_order = rng.permutation(reference.shape[1])
            moving_order = rng.permutation(original_moving.shape[1])
            fit_options.update(
                error_samples=(reference[:, reference_order], original_moving[:, moving_order]),
                error_predictors={
                    "ref": {"quality": predictors["ref"]["quality"][reference_order]},
                    "tba": {"quality": predictors["tba"]["quality"][moving_order]},
                },
                # argsort() maps each old sample index to its new position
                error_indices=(np.argsort(reference_order)[matched], np.argsort(moving_order)[selected]),
            )
            expected_covariance = covariance

        # Fit again with the changed error magnitudes or sample order
        changed_fit, changed_covariance = _icp_fit(**fit_options)

        # Check fitted transform and covariance against what we expect
        assert changed_covariance is not None
        # TLS allows for ODR stopping differences after rescaling all observation weights
        fit_tolerance = 1e-8 if error_options["force_opti"] == "tls" else 1e-10
        covariance_tolerance = 1e-6 if error_options["force_opti"] == "tls" else 1e-9
        np.testing.assert_allclose(changed_fit, fitted, rtol=0, atol=fit_tolerance)
        np.testing.assert_allclose(changed_covariance, expected_covariance, rtol=covariance_tolerance, atol=1e-12)

    @pytest.mark.parametrize("surface", ["flat", "ramp"])
    @pytest.mark.parametrize("only_translation", [False, True])
    def test_icp__error_planar_p2plane(self, curved_dem: Raster, surface: str, only_translation: bool) -> None:
        """Checks an error is raised when planar normals cannot constrain the requested ICP transform."""

        # Create a flat surface or ramp
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)

        # Check failure
        method = coreg.ICP(subsample=1, only_translation=only_translation)
        with pytest.raises(ValueError, match="ICP cannot determine .*the surface may be too flat or planar"):
            method.fit(reference, reference, random_state=42)


class TestLZD:
    """Test module for LZD coregistration."""

    @pytest.mark.parametrize("pair", ["raster-raster", "raster-point", "point-raster"])
    @pytest.mark.parametrize("loss", [nmad, np.nanstd])
    def test_lzd__custom_nelder_mead(self, curved_dem: Raster, pair: str, loss: Callable[..., Any]) -> None:
        """Checks that a custom Nelder-Mead wrapper works to mimic old DhMinimize behaviour."""

        # Add a large Z offset
        reference: Any = curved_dem
        shifted: Any = curved_dem.translate(1.2, -0.8) + 500
        if pair == "raster-point":
            shifted = _point_samples(shifted)
        elif pair == "point-raster":
            reference = _point_samples(reference)

        # Fit residual spread in X/Y and median height in Z
        calls = []
        optimizer_options = {"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000}

        def fit_minimizer(
            residuals: Callable[[NDArrayf], NDArrayf], x0: NDArrayf, *, loss: Callable[..., Any], **kwargs: Any
        ) -> Any:
            """Custom minimize with Nelder-Mead and different horizontal/vertical logic."""

            calls.append((x0.copy(), loss, kwargs))

            def horizontal_residuals(xy: NDArrayf) -> NDArrayf:
                return residuals(np.array([xy[0], xy[1], x0[2]]))

            result = scipy.optimize.minimize(
                lambda xy: loss(horizontal_residuals(xy)), np.ones(2), method="Nelder-Mead", **kwargs
            )
            z_shift = x0[2] + np.median(horizontal_residuals(result.x))
            result.x = np.array([result.x[0], result.x[1], z_shift])
            return result

        # Fit DhMinimize and LZD with the same sample and loss
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

        # Allow interpolation error of 3%
        np.testing.assert_allclose(legacy.to_translations(), (-1.2, 0.8, -500), atol=0.06)
        np.testing.assert_allclose(replacement.to_translations(), (-1.2, 0.8, -500), atol=0.06)
        np.testing.assert_array_equal(replacement.to_matrix()[:3, :3], np.eye(3))

        # Check loss and optimizer options at each iteration
        assert calls
        for initial, received_loss, received_kwargs in calls:
            np.testing.assert_array_equal(initial, np.zeros(3))
            assert received_loss is loss
            assert received_kwargs == {"options": optimizer_options}

    @pytest.mark.parametrize("linearized", [False, True])
    @pytest.mark.parametrize("trim_iterative", [False, True])
    def test_fit__lzd_trimming(self, curved_dem: Raster, linearized: bool, trim_iterative: bool) -> None:
        """Checks that LZD trimming works properly."""

        # Translate DEM points and add Z outliers
        points = _point_samples(curved_dem)
        translations = np.array([0.6, -0.4, 0.8])
        matrix = matrix_from_translations_rotations(*translations)
        moving = EPC(coreg.apply_matrix(points.gdf, matrix=matrix), data_name="z")
        positions = np.random.default_rng(45).choice(len(moving.gdf), len(moving.gdf) // 12, replace=False)
        moving.gdf.loc[positions, "z"] += 10

        # Fit with trimming
        fitted = coreg.LZD(
            linearized=linearized,
            only_translation=True,
            trim_residuals=True,
            trim_iterative=trim_iterative,
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_rotation=1e-6,
        ).fit(curved_dem, moving, random_state=41)

        # Check that the fitted translations undo the applied translations
        np.testing.assert_allclose(fitted.to_translations(), -translations, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("magnitude", [0.05, 0.5])
    @pytest.mark.parametrize("error_options", _GLS_TLS_OPTIONS)
    def test_lzd__known_errors(self, only_translation: bool, magnitude: float, error_options: dict[str, Any]) -> None:
        """Checks that GLS/TLS return known parameter uncertainty even when the fitted residuals are zero."""

        if error_options.get("gls_method") in ("approximate", "exact"):
            pytest.importorskip("linear_operator")
        elif error_options.get("gls_method") == "inducing":
            pytest.importorskip("gpytorch")

        # Create a full-rank design with known parameters
        rng = np.random.default_rng(42)
        x, y, z, gx, gy = rng.normal(size=(5, 80))
        design = np.column_stack((-gx, -gy, np.ones_like(x)))
        expected = np.array([0.7, -0.4, 2.5])
        if not only_translation:
            rotations = np.column_stack((y + gy * z, -x - gx * z, gx * y - gy * x))
            design = np.column_stack((design, rotations))
            # Spatial TLS approaches OLS when rotations are zero
            rotation_values = np.zeros(3) if error_options.get("tls_method") == "spatial" else [0.02, -0.01, 0.03]
            expected = np.concatenate((expected, rotation_values))
        differences = design @ expected

        # Use small grid errors to approach OLS
        sigma_other = np.full(len(x), magnitude)
        sigma_grid = sigma_other * 1e-8
        models = tuple(
            geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", sigma)])
            for sigma in (magnitude, magnitude * 1e-8)
        )
        beta, standard_errors, covariance = _lzd_fit_gls_tls(
            x=x,
            y=y,
            z=z,
            dh=differences,
            gx=gx,
            gy=gy,
            pixel_size=2,
            sig_h_other=sigma_other,
            corr_h_other=None,
            sig_h_grid=sigma_grid,
            corr_h_grid=None,
            error_structure=models,
            only_translation=only_translation,
            **error_options,
        )

        # Check uncertainty from known errors
        expected_covariance = magnitude**2 * np.linalg.inv(design.T @ design)
        expected_standard_errors = np.sqrt(np.diag(expected_covariance))
        np.testing.assert_allclose(beta, expected, rtol=0, atol=1e-8)
        # Allow for grid errors and numerical derivatives
        np.testing.assert_allclose(standard_errors, expected_standard_errors, rtol=1e-3)
        # Compare covariance relative to parameter uncertainty, including entries close to zero
        covariance_scale = np.outer(expected_standard_errors, expected_standard_errors)
        np.testing.assert_allclose(
            covariance / covariance_scale, expected_covariance / covariance_scale, rtol=0, atol=1e-3
        )
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)

    @pytest.mark.parametrize("error_options", _GLS_TLS_OPTIONS)
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_lzd__error_aware_parameters(self, error_options: dict[str, Any], only_translation: bool) -> None:
        """Checks that GLS and TLS recover known LZD parameters and return uncertainties for each fitted parameter."""

        pytest.importorskip("skgstat")

        if error_options.get("gls_method") in ("approximate", "exact"):
            pytest.importorskip("linear_operator")
        elif error_options.get("gls_method") == "inducing":
            pytest.importorskip("gpytorch")

        # Create a full-rank design with known parameters
        rng = np.random.default_rng(42)
        x, y, z, gx, gy = rng.normal(size=(5, 80))
        design = np.column_stack((-gx, -gy, np.ones_like(x)))
        expected = np.array([0.7, -0.4, 2.5])
        if not only_translation:
            rotations = np.column_stack((y + gy * z, -x - gx * z, gx * y - gy * x))
            design = np.column_stack((design, rotations))
            expected = np.concatenate((expected, [0.02, -0.01, 0.03]))
        dh = design @ expected

        # Assign spatial errors to the height differences and, for TLS, the gradients
        correlation = VariogramModel("gaussian", effective_range=2, partial_sill=1)
        other_model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.2, correlation)])
        grid_model = None
        if error_options["force_opti"] == "tls":
            grid_model = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.1, correlation)])

        # Fit each GLS/TLS method with the same observations and errors
        beta, standard_errors, covariance = _lzd_fit_gls_tls(
            x=x,
            y=y,
            z=z,
            dh=dh,
            gx=gx,
            gy=gy,
            pixel_size=2,
            sig_h_other=None,
            corr_h_other=None,
            error_structure=(other_model, grid_model),
            only_translation=only_translation,
            **error_options,
        )

        # Check parameter order and covariance shape
        np.testing.assert_allclose(beta, expected, rtol=0, atol=1e-7)
        assert standard_errors.shape == expected.shape
        assert covariance.shape == (len(expected), len(expected))
        assert np.all(np.isfinite(standard_errors))
        assert np.all(standard_errors >= 0)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)

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

        # Translate the DEM and assign spatially varying errors
        shifted = curved_dem.translate(1.2, -0.8) + 2.5
        rows, columns = np.indices(curved_dem.shape)
        sigma_grid = 0.02 + rows / 2000
        sigma_other = 0.05 + columns / 2000
        if trim_residuals:
            shifted.data[:4, :] += 20
        correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)
        errors: dict[str, Any] = {"force_opti": optimizer, "gls_method": "exact"}
        if error_source in ("grid", "both"):
            errors.update(sig_ref=sigma_grid, corr_ref=correlation)
        if error_source in ("other", "both"):
            errors.update(sig_tba=sigma_other, corr_tba=correlation)

        # Fit translations with dense GLS or TLS
        monkeypatch.setattr("xdem.coreg.affine._gls_lazy_gpytorch", _dense_gls)
        fitted = coreg.LZD(only_translation=True, subsample=200, trim_residuals=trim_residuals).fit(
            curved_dem, shifted, random_state=42, **errors
        )

        # Allow interpolation error and check zero rotation
        np.testing.assert_allclose(fitted.to_translations(), (-1.2, 0.8, -2.5), rtol=0, atol=0.03)
        np.testing.assert_array_equal(fitted.to_matrix()[:3, :3], np.eye(3))
        standard_errors = fitted.meta["outputs"]["fitorbin"]["fit_perr"]
        assert standard_errors.shape == (3,)
        assert np.all(np.isfinite(standard_errors))
        assert np.all(standard_errors >= 0)

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("invert", [False, True])
    def test_fit__accumulated_covariance(
        self, curved_dem: Raster, monkeypatch: pytest.MonkeyPatch, only_translation: bool, invert: bool
    ) -> None:
        """Checks that LZD maps its final increment covariance to the accumulated transform in either input order."""

        # Create a translated transform with known uncertainty in its final local increment
        count = 3 if only_translation else 6
        local_covariance = np.diag(np.arange(1, count + 1, dtype=float)) * 1e-6
        matrix = matrix_from_translations_rotations(4, -6, 9, 0, 0, 0)

        def finish_iteration(**kwargs: Any) -> tuple[Any, ...]:
            """Return a converged transform and its known final increment covariance."""

            kwargs["constant_inputs"][-1]["error_output"]["fit_pcov"] = local_covariance
            output = dict(converged=True, last_iteration=2, iteration_stats=pd.DataFrame())
            return matrix, None, output, np.sqrt(np.diag(local_covariance))

        # Fit through the public API with raster or point reference data
        monkeypatch.setattr("xdem.coreg.affine._iterate_method", finish_iteration)
        reference = curved_dem.to_pointcloud() if invert else curved_dem
        errors = geoutils.ErrorStructure([geoutils.ErrorComponent("measurement", 0.1)])
        fitted = coreg.LZD(only_translation=only_translation, subsample=60).fit(
            reference, curved_dem, error_structure={"tba": errors}, random_state=12
        )

        # A left rotation changes the translation by delta_rotation cross translation
        jacobian = np.eye(count)
        if not only_translation and not invert:
            tx, ty, tz = matrix[:3, 3]
            jacobian[:3, 3:] = [[0, tz, -ty], [-tz, 0, tx], [ty, -tx, 0]]
        expected = jacobian @ local_covariance @ jacobian.T
        covariance = fitted.meta["outputs"]["fitorbin"]["fit_pcov"]
        # Allow roundoff from centered differences of the inverted transform
        np.testing.assert_allclose(covariance, expected, rtol=1e-8, atol=1e-14)
        np.testing.assert_allclose(fitted.meta["outputs"]["fitorbin"]["fit_perr"] ** 2, np.diag(expected), rtol=1e-8)

    @pytest.mark.parametrize("pixel_size", [2.0, (2.0, 3.0)])
    @pytest.mark.parametrize("active_dims", [None, (0,), (1,)])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_lzd__tls_gradient_variances(
        self,
        monkeypatch: pytest.MonkeyPatch,
        pixel_size: float | tuple[float, float],
        active_dims: tuple[int, ...] | None,
        only_translation: bool,
    ) -> None:
        """Checks that TLS gradient variances use each pixel spacing and the correlation's active dimensions."""

        pytest.importorskip("skgstat")

        # Create a grid error field that may be constant along one coordinate
        case = _tls_observation_case(only_translation, spatial=False)
        inputs: tuple[NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf] = case["covariance_inputs"]["inputs"]
        correlation = VariogramModel("gaussian", effective_range=6, partial_sill=1, active_dims=active_dims)
        component = geoutils.ErrorComponent("measurement", 0.1, correlation)
        model = geoutils.ErrorStructure([component])
        observed = []
        from xdem.coreg.affine import _fit_tls

        def capture_covariance(*args: Any, **kwargs: Any) -> Any:
            """Record the observation covariance used by the actual ODR fit."""

            observed.append(args[3])
            return _fit_tls(*args, **kwargs)

        # Fit TLS through the mapped error model, including its spatial covariance correction
        monkeypatch.setattr("xdem.coreg.affine._fit_tls", capture_covariance)
        beta, standard_errors, covariance = _lzd_fit_gls_tls(
            *inputs,
            pixel_size,
            None,
            None,
            only_translation=only_translation,
            error_structure=(None, model),
            force_opti="tls",
            tls_method="spatial",
        )

        # Centered differences have variance sigma^2 (1 - rho(2 * spacing)) / (2 * spacing^2)
        spacing = pixel_size if isinstance(pixel_size, tuple) else (pixel_size, pixel_size)
        for axis, distance in enumerate(spacing):
            lag = 2 * distance if active_dims is None or axis in active_dims else 0
            expected = 0.1**2 * (1 - component.predict_correlation(lag)) / (2 * distance**2)
            np.testing.assert_allclose(observed[0][axis, axis], expected, rtol=1e-12, atol=1e-18)
        np.testing.assert_allclose(beta, case["expected"], rtol=0, atol=1e-7)
        assert np.all(np.isfinite(covariance))
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)

    @pytest.mark.parametrize("pixel_size", [2.0, (2.0, 3.0)])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("residual", [0, 0.01])
    def test_tls_observation_jacobians__refits(
        self,
        monkeypatch: pytest.MonkeyPatch,
        only_translation: bool,
        residual: float,
        pixel_size: float | tuple[float, float],
    ) -> None:
        """Checks that sampled height and gradient sensitivities match actual TLS refits."""

        pytest.importorskip("skgstat")

        import scipy.odr

        # Tighten ODR tolerance to resolve small parameter changes
        original_odr = scipy.odr.ODR

        def precise_odr(*args: Any, **kwargs: Any) -> Any:
            """Resolve parameter changes below the ordinary estimator's stopping tolerance."""

            return original_odr(*args, **dict(kwargs, sstol=1e-12, partol=1e-12))

        monkeypatch.setattr(scipy.odr, "ODR", precise_odr)

        # Create a TLS fit with optional nonzero residuals
        case = _tls_observation_case(only_translation, spatial=True, residual=residual)
        options = case["covariance_inputs"]
        inputs: tuple[NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf] = options["inputs"]
        errors: _SampledErrors = options["errors"]
        beta = _lzd_fit_gls_tls(*inputs, pixel_size, *errors, only_translation=only_translation, force_opti="tls")[0]
        jacobians = _lzd_tls_observation_jacobians(beta, inputs, options["errors"], pixel_size)
        rng = np.random.default_rng(52)

        # Perturb each variable above the ODR stopping tolerance
        observed_derivatives, predicted_derivatives = [], []
        for variable, jacobian in zip((2, 3, 4, 5), jacobians):
            direction = rng.normal(scale=0.01, size=len(inputs[0]))
            fits = []
            step = 0.1
            for sign in (1, -1):
                perturbed = list(inputs)
                perturbed[variable] = inputs[variable] + sign * step * direction
                fit_inputs = (perturbed[0], perturbed[1], perturbed[2], perturbed[3], perturbed[4], perturbed[5])
                fits.append(
                    _lzd_fit_gls_tls(
                        *fit_inputs, pixel_size, *errors, only_translation=only_translation, force_opti="tls"
                    )[0]
                )

            # Calculate observed and predicted parameter changes
            observed = (fits[0] - fits[1]) / (2 * step)
            predicted = jacobian @ direction
            observed_derivatives.append(observed)
            predicted_derivatives.append(predicted)

        # Normalize by the combined response to avoid cancellation
        observed = np.column_stack(observed_derivatives)
        predicted = np.column_stack(predicted_derivatives)
        scale = np.linalg.norm(predicted, axis=1)[:, None]
        np.testing.assert_allclose((observed - predicted) / scale, 0, rtol=0, atol=0.002)

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("source", ["grid", "other", "both"])
    @pytest.mark.parametrize("spatial", [False, True])
    def test_tls_spatial_covariance__numerical_simulation(
        self, only_translation: bool, source: str, spatial: bool
    ) -> None:
        """Checks that sampled TLS covariance matches simulations of either or both error models."""

        if spatial:
            pytest.importorskip("skgstat")

        # 1/ Create small errors for first-order propagation
        case = _tls_observation_case(only_translation, spatial)
        options = dict(case["covariance_inputs"])
        other, grid = options["error_structure"]
        other = other if source != "grid" else None
        grid = grid if source != "other" else None
        options["error_structure"] = (other, grid)
        covariance = _lzd_tls_spatial_covariance(**options)

        # 2/ Simulate independent error fields from the model covariances
        rng = np.random.default_rng(43)
        n_simulations = 20
        draws = []
        for model, coordinates in ((grid, case["field_coordinates"]), (other, case["coordinates"])):
            if model is None:
                draws.append(np.zeros((len(coordinates), n_simulations)))
            else:
                field_covariance = model.to_covariance_matrix(coordinates)
                values, vectors = np.linalg.eigh(field_covariance)
                assert values.min() >= -1e-12
                draws.append((vectors * np.sqrt(np.maximum(values, 0))) @ rng.normal(size=(len(values), n_simulations)))
        estimates = np.array([case["fit"](draws[0][:, index], draws[1][:, index])[0] for index in range(n_simulations)])

        # 3/ Compare standard deviations with joint Monte Carlo intervals
        predicted_std = np.sqrt(np.diag(covariance))
        empirical_std = estimates.std(axis=0, ddof=1)
        tail = 0.001 / (2 * len(case["expected"]))
        bounds = np.sqrt(chi2.ppf([tail, 1 - tail], n_simulations - 1) / (n_simulations - 1))
        np.testing.assert_allclose(options["beta"], case["expected"], rtol=0, atol=1e-7)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-15)
        assert np.linalg.eigvalsh(covariance).min() > 0
        assert np.all((empirical_std / predicted_std >= bounds[0]) & (empirical_std / predicted_std <= bounds[1]))

    def test_lzd__gls_numerical_rank(self) -> None:
        """Checks that smooth spatial errors stop factorization before a roundoff pivot produces NaNs."""

        pytest.importorskip("linear_operator")

        # Create a large smooth field whose covariance has a small numerical rank
        coordinates = np.random.default_rng(174).uniform(0, 1, (10_000, 2))
        x, y = coordinates.T
        gx, gy = np.cos(6 * x), np.sin(4 * y)
        design = np.column_stack((-gx, -gy, np.ones(len(x))))
        expected = np.array([0.2, -0.3, 0.5])
        sigma = (0.5 + x) / 2
        correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)

        # Fit with a positive independent variance in addition to the smooth component
        beta, standard_errors, covariance = _lzd_fit_gls_tls(
            x,
            y,
            np.zeros(len(x)),
            design @ expected,
            gx,
            gy,
            1,
            sigma,
            correlation,
            np.full(len(x), 0.05),
            None,
            only_translation=True,
            force_opti="gls",
        )

        # Check a finite fit and positive parameter covariance for this positive definite model
        np.testing.assert_allclose(beta, expected, rtol=0, atol=1e-10)
        assert np.all(np.isfinite(covariance))
        assert np.linalg.eigvalsh(covariance).min() > 0
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)

    @pytest.mark.parametrize("model", ["independent", "exponential", "gaussian", "mixed", "two_inputs"])
    @pytest.mark.parametrize(
        "backend, neighbors",
        [
            pytest.param("approximate", None, id="approximate"),
            pytest.param("approximate", 192, id="all-neighbors"),
            pytest.param("inducing", None, id="inducing"),
        ],
    )
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("clustered", [False, True])
    def test_lzd__gls_methods(
        self,
        monkeypatch: pytest.MonkeyPatch,
        only_translation: bool,
        clustered: bool,
        model: str,
        backend: Literal["approximate", "exact", "inducing"],
        neighbors: int | None,
    ) -> None:
        """Checks that LZD GLS methods give similar parameters and covariance for the same spatial errors."""

        # Independent errors need no optional dependencies
        if model != "independent":
            pytest.importorskip("linear_operator")
            if backend == "inducing":
                pytest.importorskip("gpytorch")

        # Create curved terrain with regular or clustered observations
        rng = np.random.default_rng(26)
        x, y = np.meshgrid(np.linspace(0, 1, 14), np.linspace(0, 1, 14))
        x, y = x.ravel()[:193], y.ravel()[:193]
        if clustered:
            x, y = rng.uniform(0, 1, (2, len(x)))
            x[:80] *= 0.2
            y[:80] *= 0.2
        z = np.sin(x * 6) + np.cos(y * 4) + np.sin((x + y) * 5)
        gx = 6 * np.cos(x * 6) + 5 * np.cos((x + y) * 5)
        gy = -4 * np.sin(y * 4) + 5 * np.cos((x + y) * 5)
        design = np.column_stack((-gx, -gy, np.ones(len(x))))
        expected = np.array([0.2, -0.3, 500.0])
        if not only_translation:
            design = np.column_stack((design, y + gy * z, -x - gx * z, gx * y - gy * x))
            expected = np.concatenate((expected, [0.02, -0.01, 0.03]))
        values = design @ expected + rng.normal(0, 0.01, len(x))
        sigma = rng.uniform(0.5, 1.5, len(x))

        # Define short- and long-range correlations
        correlation: tuple[Callable[..., NDArrayf], float] | None = None
        if model == "exponential":
            correlation = (lambda d: np.exp(-d), 1)
        elif model == "gaussian":
            correlation = (lambda d: np.exp(-((d / 0.2) ** 2)), 0.2)
        elif model == "mixed":
            correlation = (lambda d: 0.9 * np.exp(-(d**2)) + 0.1 * np.exp(-d / 0.02), 1)
        elif model == "two_inputs":
            correlation = (lambda d: np.exp(-(d**2)), 1)
        sigma_grid = sigma / 3 if model in ("independent", "mixed", "two_inputs") else None
        grid_correlation = (lambda d: np.exp(-d / 0.02), 0.02) if model == "two_inputs" else None

        # Fit exact GLS as the reference for the same LZD observations and errors
        # Independent errors give the same result without optional dependencies
        fit_options: dict[str, Any] = dict(
            x=x,
            y=y,
            z=z,
            dh=values,
            gx=gx,
            gy=gy,
            pixel_size=2,
            sig_h_other=sigma,
            corr_h_other=correlation,
            sig_h_grid=sigma_grid,
            corr_h_grid=grid_correlation,
            force_opti="gls",
            only_translation=only_translation,
        )
        reference_method: Literal["approximate", "exact"] = "approximate" if model == "independent" else "exact"
        expected_beta, expected_errors, expected_covariance = _lzd_fit_gls_tls(
            **fit_options, gls_method=reference_method
        )

        # Independent errors must work without optional dependencies for every method
        if model == "independent":

            def fail_optional(name: str) -> Any:
                """Fail if independent GLS requests an optional dependency."""

                pytest.fail(f"Independent GLS must not import {name}.")

            monkeypatch.setattr("xdem.coreg.affine.import_optional", fail_optional)
        if backend == "approximate":
            monkeypatch.setitem(sys.modules, "gpytorch", None)

        # Run the selected GLS method through LZD, using a shorter final covariance block
        if backend == "approximate":
            monkeypatch.setattr("xdem.coreg.affine._gls_approximate", partial(_gls_approximate, block_size=32))
        beta, standard_errors, covariance = _lzd_fit_gls_tls(
            **fit_options, gls_method=backend, gls_neighbors=neighbors, gls_rank=64
        )

        # Conditioning on all neighbors should give the same result as exact GLS
        # Other approximations allow 10% uncertainty error and fit differences within 0.3 standard errors
        exact = model == "independent" or neighbors == len(x) - 1
        covariance_tolerance = 1e-9 if exact else 0.1
        fit_tolerance = 1e-8 if exact else 0.3
        np.testing.assert_allclose((beta - expected_beta) / expected_errors, 0, rtol=0, atol=fit_tolerance)
        np.testing.assert_allclose(standard_errors, expected_errors, rtol=covariance_tolerance)
        scale = np.outer(expected_errors, expected_errors)
        np.testing.assert_allclose(covariance / scale, expected_covariance / scale, rtol=0, atol=covariance_tolerance)
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)

    @pytest.mark.parametrize("model", ["gaussian", "mixed"])
    def test_lzd__gls_simulation_covariance(self, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
        """Checks that approximate GLS errors for all six parameters match simulations from the full covariance."""

        pytest.importorskip("linear_operator")

        # 1/ Create curved terrain with spatial errors
        rng = np.random.default_rng(26)
        x, y = rng.uniform(0, 1, (2, 193))
        z = np.sin(x * 6) + np.cos(y * 4) + np.sin((x + y) * 5)
        gx = 6 * np.cos(x * 6) + 5 * np.cos((x + y) * 5)
        gy = -4 * np.sin(y * 4) + 5 * np.cos((x + y) * 5)
        design = np.column_stack((-gx, -gy, np.ones(len(x)), y + gy * z, -x - gx * z, gx * y - gy * x))
        sigma = rng.uniform(0.5, 1.5, len(x))
        expected = np.array([0.2, -0.3, 500.0, 0.02, -0.01, 0.03])
        correlation: tuple[Callable[..., NDArrayf], float]
        if model == "gaussian":
            correlation = (lambda d: np.exp(-((d / 0.2) ** 2)), 0.2)
        else:
            correlation = (lambda d: 0.9 * np.exp(-(d**2)) + 0.1 * np.exp(-d / 0.02), 1)
        errors = (sigma, correlation, None, None)

        # Simulate errors from the full covariance
        distances = scipy.spatial.distance.cdist(np.column_stack((x, y)), np.column_stack((x, y)))
        covariance = (correlation[0](distances) + 1e-3 * np.eye(len(x))) * np.outer(sigma, sigma)
        n_simulations = 20
        draws = np.linalg.cholesky(covariance) @ rng.normal(size=(len(x), n_simulations))

        # 2/ Fit each simulation through LZD
        # Use small blocks with default neighbor count and rank
        monkeypatch.setattr("xdem.coreg.affine._gls_approximate", partial(_gls_approximate, block_size=32))
        estimates = []
        for index in range(n_simulations):
            values = design @ expected + draws[:, index]
            beta, standard_errors, _ = _lzd_fit_gls_tls(
                x, y, z, values, gx, gy, 2, *errors, force_opti="gls", gls_method="approximate"
            )
            estimates.append(beta)

        # Compare standard deviations with joint 0.1% chi-square bounds
        parameter_count = len(expected)
        tail_probability = 0.001 / (2 * parameter_count)
        bounds = np.sqrt(chi2.ppf([tail_probability, 1 - tail_probability], n_simulations - 1) / (n_simulations - 1))
        ratios = np.std(estimates, axis=0, ddof=1) / standard_errors
        assert np.all((bounds[0] <= ratios) & (ratios <= bounds[1])), ratios

    @pytest.mark.parametrize("neighbors", [1, 7, 30])
    @pytest.mark.parametrize("duplicates", [False, True])
    def test_gls__preceding_neighbors(self, neighbors: int, duplicates: bool) -> None:
        """Checks that local conditioning selects nearest preceding observations across merged KD-tree groups."""

        # Create observations spanning tree merges and a shorter final block
        coordinates = np.random.default_rng(42).normal(size=(193, 2))
        if duplicates:
            coordinates[80:90] = coordinates[:10]

        # Compare nearest preceding distances, allowing ties
        for rows, indexes, distances in _gls_neighbors(coordinates, neighbors=neighbors, block_size=32):
            if rows.start == 0:
                continue
            for offset, row in enumerate(range(rows.start, rows.stop)):
                expected = np.linalg.norm(coordinates[:row] - coordinates[row], axis=1)
                np.testing.assert_allclose(np.sort(distances[offset]), np.sort(expected)[:neighbors])
                assert np.all(indexes[offset] < row)
                assert len(np.unique(indexes[offset])) == neighbors
                actual = np.linalg.norm(coordinates[indexes[offset]] - coordinates[row], axis=1)
                np.testing.assert_allclose(distances[offset], actual)

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("model", ["gaussian", "exponential", "mixed"])
    @pytest.mark.parametrize("jitter", [0, 1e-3])
    def test_lzd__gls_covariance_blocks(
        self, monkeypatch: pytest.MonkeyPatch, only_translation: bool, model: str, jitter: float
    ) -> None:
        """Checks that LZD exact GLS uses small covariance blocks and preconditioning without a full allocation."""

        linear_operator = pytest.importorskip("linear_operator")
        from linear_operator.operators import AddedDiagLinearOperator, LinearOperator

        # Create LZD observations with a shorter final covariance block
        count, block_size = 193, 32
        rng = np.random.default_rng(81)
        x, y, z, gx, gy = rng.uniform(-1, 1, (5, count))
        values = rng.normal(size=count)
        sigma_other = rng.uniform(0.1, 0.3, count)
        sigma_grid = rng.uniform(0.2, 0.4, count) if model == "mixed" else None
        # Use a short Gaussian range for stability without regularization
        gaussian_range = 0.1

        # Guard covariance allocation and record preconditioning
        # Limit covariance evaluation to blocks or rows
        evaluations = []
        preconditioned = []

        def correlation(distance: NDArrayf) -> NDArrayf:
            """Evaluate correlation only for bounded blocks or individual covariance rows."""

            assert distance.size <= max(block_size**2, count)
            evaluations.append(distance.shape)
            if model == "gaussian":
                return np.exp(-((distance / gaussian_range) ** 2))
            return np.exp(-distance / 0.7)

        def grid_correlation(distance: NDArrayf) -> NDArrayf:
            """Evaluate the second input's Gaussian correlation with the same allocation bound."""

            assert distance.size <= max(block_size**2, count)
            return np.exp(-((distance / 0.4) ** 2))

        dense = LinearOperator.to_dense
        preconditioner = AddedDiagLinearOperator._preconditioner

        def guarded_dense(operator: Any) -> Any:
            """Fail if a solver requests the complete observation covariance."""

            assert operator.shape[-2:] != (count, count)
            return dense(operator)

        def checked_preconditioner(operator: Any) -> Any:
            """Record that linear_operator uses its preconditioner for the blockwise solve."""

            result = preconditioner(operator)
            preconditioned.append(result[0] is not None)
            return result

        monkeypatch.setattr(LinearOperator, "to_dense", guarded_dense)
        monkeypatch.setattr(AddedDiagLinearOperator, "_preconditioner", checked_preconditioner)
        correlation_range = gaussian_range if model == "gaussian" else 0.7
        errors = (
            sigma_other,
            (correlation, correlation_range),
            sigma_grid,
            (grid_correlation, 0.4) if model == "mixed" else None,
        )

        # Fit through LZD with dense Cholesky enabled externally
        monkeypatch.setattr(
            "xdem.coreg.affine._gls_lazy_gpytorch",
            partial(_gls_lazy_gpytorch, block_size=block_size, max_preconditioner_size=16, jitter=jitter),
        )
        with linear_operator.settings.max_cholesky_size(10_000):
            beta, standard_errors, covariance = _lzd_fit_gls_tls(
                x,
                y,
                z,
                values,
                gx,
                gy,
                2,
                *errors,
                force_opti="gls",
                gls_method="exact",
                only_translation=only_translation,
            )

        # Check finite outputs and the covariance blocks used during the fit
        assert beta.shape == standard_errors.shape == (3 if only_translation else 6,)
        assert np.all(np.isfinite(beta))
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
        np.testing.assert_allclose(standard_errors**2, np.diag(covariance), rtol=1e-12)
        assert evaluations
        assert (block_size, block_size) in evaluations
        # Check preconditioning is used only with jitter
        if jitter:
            assert any(preconditioned)
        else:
            assert not any(preconditioned)

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

        # Create a flat surface or ramp
        rows, columns = np.indices(curved_dem.shape)
        values = np.full(curved_dem.shape, 100.0) if surface == "flat" else 100.0 + columns / 2 + rows / 4
        reference = Raster.from_array(values, curved_dem.transform, curved_dem.crs, nodata=np.nan)
        shifted = reference.translate(1.2, -0.8) + 2.5

        # Check failure
        method = coreg.LZD(subsample=1, only_translation=only_translation, linearized=linearized, trim_residuals=trim)
        with pytest.raises(ValueError, match="LZD cannot determine .*the surface may be too flat or planar"):
            method.fit(reference, shifted, random_state=42)


class TestCPD:
    """Test module for CPD translation recovery, variance updates and local surface geometry least squares."""

    @pytest.mark.parametrize(
        "small_dem, height_scale, shifts, tolerance",
        [
            pytest.param("waves", 1, (0.6, -0.4, 0.8), 1e-7, id="waves"),
            pytest.param("smooth", 1, (0.6, -0.4, 0.8), 1e-7, id="smooth"),
            pytest.param("waves", 0, (0, 0, 0.8), 1e-5, id="flat"),
            pytest.param("waves", 0.001, (0, 0, 0.8), 1e-5, id="low-relief"),
        ],
        indirect=["small_dem"],
    )
    @pytest.mark.parametrize("lsg", [False, True])
    @pytest.mark.parametrize(
        "options",
        [
            pytest.param(dict(anisotropic=None, standardize=False, estep_knearest=None), id="isotropic"),
            pytest.param(dict(anisotropic=None, standardize=True, estep_knearest=64), id="isotropic-standardized"),
            pytest.param(dict(anisotropic=None, weight=0.5), id="outlier-weight"),
            pytest.param(dict(anisotropic="xy_vs_z", standardize=False, estep_knearest=64), id="xy-vs-z"),
            pytest.param(dict(anisotropic="xy_vs_z", standardize=True, estep_knearest=None), id="xy-vs-z-standardized"),
            pytest.param(dict(anisotropic="per_axis", standardize=False, estep_knearest=None), id="per-axis"),
            pytest.param(dict(anisotropic="per_axis", standardize=True, estep_knearest=64), id="per-axis-standardized"),
        ],
    )
    def test_fit__translated_dem(
        self,
        small_dem: Raster,
        height_scale: float,
        shifts: tuple[float, float, float],
        tolerance: float,
        lsg: bool,
        options: dict[str, Any],
    ) -> None:
        """Checks that CPD recovers a DEM translation across terrain and distance options, then applies its inverse."""

        # Scale terrain heights, then translate the DEM
        values = 40 + height_scale * (np.asarray(small_dem.data) - 40)
        reference = small_dem.copy(new_array=values)
        translations = np.array(shifts)
        moving = reference.translate(*translations[:2]) + translations[2]

        # Fit independently sampled point clouds with the selected distance and neighbor options
        fitted = coreg.CPD(
            lsg=lsg,
            only_translation=True,
            sampling_strategy="independent",
            subsample=1,
            max_iterations=100,
            tolerance_translation=1e-6,
            tolerance_objective_func=1e-6,
            **options,
        ).fit(reference, moving, random_state=41)
        aligned = fitted.apply(moving, resample=False)

        # Check the inverse translation, grid and convergence
        np.testing.assert_allclose(fitted.to_translations(), -translations, rtol=0, atol=tolerance)
        np.testing.assert_allclose(tuple(aligned.transform), tuple(reference.transform), rtol=0, atol=tolerance)
        # Allow for float32 rounding and the fitted translation tolerance
        np.testing.assert_allclose(aligned.data, reference.data, rtol=0, atol=max(4e-6, tolerance))
        assert fitted.meta["outputs"]["iterative"]["converged"]

    @pytest.mark.parametrize("loss", ["linear", "soft_l1"])
    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize("anisotropic", [False, True])
    def test_cpd_fit__expected_residual_variance(self, only_translation: bool, anisotropic: bool, loss: str) -> None:
        """Checks that classic CPD updates variance and objective from the full weighted pair residuals."""

        # Create point clouds of different sizes and extents
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(23, 3))
        moving = rng.normal(size=(17, 3)) * [0.7, 1.2, 0.5] + [0.2, -0.3, 0.5]
        weights = np.array([1.0, 0.5, 3.0]) if anisotropic else np.ones(3)
        variance = 0.8
        probabilities = _cpd_estep_classic(reference, moving, moving, 0.1, variance, weights)
        cache = _cpd_precompute(reference.T, moving.T, weight_cpd=0.1, axis_weights=weights)
        options = cast(InFitOrBinDict, {"fit_minimizer": scipy.optimize.least_squares, "fit_loss_func": loss})

        # Run a CPD step with fixed scale
        matrix, new_variance, objective = _cpd_fit(
            reference.T,
            moving.T,
            moving.T,
            options,
            0.1,
            variance,
            1e-12,
            only_translation=only_translation,
            cache=cache,
        )

        # Calculate weighted pair residuals independently
        transformed = moving @ matrix[:3, :3].T + matrix[:3, 3]
        squared_distances = np.sum((reference[None] - transformed[:, None]) ** 2 * weights, axis=2)
        expected_residual = float(np.sum(probabilities["P"] * squared_distances))
        weighted_dimensions = 3 * float(probabilities["Np"])
        expected_objective = expected_residual / (2 * variance) + weighted_dimensions / 2 * np.log(variance)
        assert new_variance == pytest.approx(expected_residual / weighted_dimensions, rel=1e-12)
        assert objective == pytest.approx(expected_objective, rel=1e-12)

    def test_lsg_build_whitened_terms__quadratic(self) -> None:
        """Checks that squared whitened residuals equal the intended local surface quadratic."""

        # Create tilted normals for off-diagonal precision terms
        rng = np.random.default_rng(17)
        normals = rng.normal(size=(9, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        precision = np.eye(3) + 20 * normals[:, :, None] * normals[:, None, :]
        targets = rng.normal(size=(9, 3))
        first_moment = np.einsum("mij,mj->mi", precision, targets)
        residuals = rng.normal(size=(9, 3))

        # Build LSG factors
        factors, means = _lsg_build_whitened_terms(precision, first_moment)

        # Allow for stabilization before factorization
        squared_whitened = np.sum(np.einsum("mij,mj->mi", factors, residuals) ** 2)
        squared_direct = np.einsum("mi,mij,mj->", residuals, precision, residuals)
        assert squared_whitened == pytest.approx(squared_direct, rel=1e-10)
        np.testing.assert_allclose(means, targets, rtol=0, atol=1e-9)

    def test_cpd_mstep_lsg__translation_optimum(self) -> None:
        """Checks that the LSG translation step equals the independently solved quadratic minimum."""

        # Create tilted normals and noisy targets
        rng = np.random.default_rng(42)
        normals = rng.normal(size=(9, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        precision = np.eye(3) + 20 * normals[:, :, None] * normals[:, None, :]
        moving = rng.normal(size=(9, 3))
        targets = rng.normal(size=(9, 3))
        first_moment = np.einsum("mij,mj->mi", precision, targets)
        options = cast(InFitOrBinDict, {"fit_minimizer": scipy.optimize.least_squares, "fit_loss_func": "linear"})

        # Fit the translation
        rotation, translation = _cpd_mstep_lsg_least_squares(
            moving, {"M0": precision, "M1": first_moment}, True, options
        )

        # Compare with normal equations within optimizer precision
        expected = np.linalg.solve(precision.sum(axis=0), np.einsum("mij,mj->i", precision, targets - moving))
        np.testing.assert_allclose(translation, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(rotation, np.eye(3))

    @pytest.mark.parametrize(
        "axis_weights, plane_distance",
        [
            pytest.param(np.ones(3), 1 / np.sqrt(5), id="equal-axis-weights"),
            pytest.param(np.array([4.0, 1.0, 1.0]), 1 / np.sqrt(2), id="unequal-axis-weights"),
        ],
    )
    def test_cpd_precompute__weighted_plane_distance(self, axis_weights: NDArrayf, plane_distance: float) -> None:
        """Checks that LSG combines weighted point distances with known distances to a tilted plane."""

        # (Same setup as for ICP weighted test)
        # We use the tilted plane 2X + Z = 0 to test axis weights
        # From (0, 0, 1), the shortest path to the plane moves in both X and Z
        # With equal weights, this distance is 1/sqrt(2**2 + 1**2) = 1/sqrt(5)
        # An X weight of 4 counts an X change of 1 as a distance of 2, since 4*1**2 = 2**2
        # Using X_scaled = 2X, the plane equation is X_scaled + Z = 0
        # The point stays at (0, 0, 1), so its weighted distance is now 1/sqrt(1**2 + 1**2) = 1/sqrt(2)

        # Create points on plane 2X + Z = 0 and offsets on, above and below it
        points = np.array([[0, 0, 0], [1, 0, -2], [0, 1, 0], [1, 1, -2]], dtype=float)
        normals = np.tile([2.0, 0.0, 1.0], (4, 1)) / np.sqrt(5)
        displacements = np.array([[1, 2, -2], [0, 0, 1], [0, 0, -1], [1, 0, 0]], dtype=float)

        # Build LSG precision matrices
        cache = _cpd_precompute(
            points.T,
            points.T,
            weight_cpd=0,
            lsg=True,
            axis_weights=axis_weights,
            ref_normals=normals,
            curvature=np.full(len(points), 0.05),
        )["lsg"]
        precision = cache["invSigma_flat"].reshape(-1, 3, 3)

        # Use SciPy for weighted point distances and the known plane distances for the LSG term
        point_squares = scipy.spatial.distance.cdist(
            displacements, np.zeros((1, 3)), metric="sqeuclidean", w=axis_weights
        ).ravel()

        # Calculate 2X + Z for each offset, then convert these values to squared plane distances
        # Multiply these by the LSG plane weight a, then add the weighted squared point distances
        equation_values = 2 * displacements[:, 0] + displacements[:, 2]
        plane_squares = (plane_distance * equation_values) ** 2
        expected = point_squares + cache["a"] * plane_squares

        # Check that each precision matrix gives the sum of the point and plane terms
        for index, (displacement, matrix) in enumerate(zip(displacements, precision)):
            assert displacement @ matrix @ displacement == pytest.approx(expected[index], rel=1e-12, abs=1e-14)

    @pytest.mark.parametrize("count", [1, 3, 7, 20])
    def test_cpd_precompute__lsg_weight(self, count: int) -> None:
        """Checks that a planar sample receives maximum LSG plane weight."""

        # Create points on a tilted plane
        rng = np.random.default_rng(33)
        x, y = rng.normal(size=(2, count))
        points = np.column_stack((x, y, 0.2 * x - 0.3 * y))
        normals = np.tile([-0.2, 0.3, 1.0], (count, 1))

        # Calculate PCA surface variation
        cache = _cpd_precompute(points.T, points.T, weight_cpd=0, lsg=True, ref_normals=normals)["lsg"]

        # Check zero surface variation and maximum plane weight
        np.testing.assert_allclose(cache["curvature"], 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(cache["a"], 30, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("lsg", [False, True])
    def test_cpd_fit__variance_floor(self, lsg: bool) -> None:
        """Checks the edge case behaviour: small residuals should not shrink CPD variance below its lower bound."""

        # Add noise below the variance floor
        rng = np.random.default_rng(69)
        reference = rng.normal(size=(12, 3))
        moving = reference + rng.normal(scale=1e-5, size=reference.shape)
        normals = np.tile([0.0, 0.0, 1.0], (len(reference), 1))
        cache = _cpd_precompute(
            reference.T, moving.T, weight_cpd=0, lsg=lsg, ref_normals=normals, curvature=np.full(len(reference), 0.05)
        )
        options = cast(InFitOrBinDict, {"fit_minimizer": scipy.optimize.least_squares, "fit_loss_func": "linear"})

        # Run one CPD step
        _, variance, _ = _cpd_fit(
            reference.T, moving.T, moving.T, options, 0, 1e-6, 1e-8, only_translation=True, lsg=lsg, cache=cache
        )

        # Check the variance floor
        assert variance == 1e-8

    @pytest.mark.parametrize("only_translation", [True, False])
    def test_cpd_mstep_lsg__zero_probability(self, only_translation: bool) -> None:
        """Checks edge case that points with zero correspondence probability contribute no residual to the LSG fit."""

        # Create a translated tetrahedron and an outlier with zero correspondence probability
        moving = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [100, -200, 300]], dtype=float)
        expected_translation = np.array([0.2, -0.3, 0.5])
        targets = moving + expected_translation
        precision = np.tile(np.eye(3), (len(moving), 1, 1))
        precision[-1] = 0
        first_moment = np.einsum("mij,mj->mi", precision, targets)
        options = cast(InFitOrBinDict, {"fit_minimizer": scipy.optimize.least_squares, "fit_loss_func": "linear"})

        # Build residual factors and fit with both matched points and the outlier
        factors, means = _lsg_build_whitened_terms(precision, first_moment)
        rotation, translation = _cpd_mstep_lsg_least_squares(
            moving, {"M0": precision, "M1": first_moment}, only_translation, options
        )

        # Check zero outlier residuals and the transformation constrained by the tetrahedron
        np.testing.assert_array_equal(factors[-1], 0)
        np.testing.assert_array_equal(means[-1], 0)
        assert np.all(np.isfinite(factors))
        assert np.all(np.isfinite(means))
        np.testing.assert_allclose(translation, expected_translation, rtol=0, atol=1e-7)
        np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-7)


class TestDhMinimizeDeprecation:
    """Test module for the DhMinimize deprecation notice."""

    def test_dh_minimize__deprecation_notice(self) -> None:
        """Checks DhMinimize deprecation."""

        with pytest.warns(DeprecationWarning, match="LZD with linearized=False, only_translation=True") as records:
            coreg.DhMinimize()
        assert "custom fit_minimizer wrapping Nelder-Mead" in str(records[0].message)
        assert "fit_loss_func=nmad" in str(records[0].message)
        assert "Z shift separately from the median residual" in str(records[0].message)
        assert records[0].filename == __file__

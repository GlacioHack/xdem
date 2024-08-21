"""Functions to test the coregistration base classes."""

from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, Callable

import numpy as np
import pytest
import rasterio as rio

import xdem
from tests.assert_function import assert_coreg_meta_equal
from tests.test_coreg.conftest import all_coregs
from xdem import coreg
from xdem._typing import NDArrayf
from xdem.coreg.base import Coreg


def test_init() -> None:
    """Test instantiation of Coreg"""

    c = coreg.Coreg()

    assert c._fit_called is False
    assert c._is_affine is None
    assert c._needs_vars is False


@pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
def test_copy(coreg_class: Callable[[], Coreg]) -> None:
    """Test that copying work expectedly (that no attributes still share references)."""

    # Create a coreg instance and copy it.
    corr = coreg_class()
    corr_copy = corr.copy()

    # Assign some attributes and .metadata after copying, respecting the CoregDict type class
    corr._meta["shift_z"] = 30
    # Make sure these don't appear in the copy
    assert corr_copy.meta != corr.meta
    assert not hasattr(corr_copy, "shift_z")


def test_error_method() -> None:
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
    vshiftcorr.meta["shift_z"] = 0
    # Now it should be equal to dem1 - dem2
    assert vshiftcorr.error(dem1, dem2, transform=affine, crs=crs, error_type="median") == -2

    # Create random noise and see if the standard deviation is equal (it should)
    rng = np.random.default_rng(42)
    dem3 = dem1.copy() + rng.random(size=dem1.size).reshape(dem1.shape)
    assert abs(vshiftcorr.error(dem1, dem3, transform=affine, crs=crs, error_type="std") - np.std(dem3)) < 1e-6


@pytest.mark.parametrize("subsample", [10, 10000, 0.5, 1])  # type: ignore
def test_get_subsample_on_valid_mask(subsample: float | int) -> None:
    """Test the subsampling function called by all subclasses"""

    # Define a valid mask
    width = height = 50
    rng = np.random.default_rng(42)
    valid_mask = rng.integers(low=0, high=2, size=(width, height), dtype=bool)

    # Define a class with a subsample and random_state in the .metadata
    coreg = Coreg(meta={"subsample": subsample, "random_state": 42})
    subsample_mask = coreg._get_subsample_on_valid_mask(valid_mask=valid_mask)

    # Check that it returns a same-shaped array that is boolean
    assert np.shape(valid_mask) == np.shape(subsample_mask)
    assert subsample_mask.dtype == bool
    # Check that the subsampled values are all within valid values
    assert all(valid_mask[subsample_mask])
    # Check that the number of subsampled value is coherent, or the maximum possible
    if subsample <= 1:
        # If value lower than 1, fraction of valid pixels
        subsample_val: float | int = int(subsample * np.count_nonzero(valid_mask))
    else:
        # Otherwise the number of pixels
        subsample_val = subsample
    assert np.count_nonzero(subsample_mask) == min(subsample_val, np.count_nonzero(valid_mask))


@pytest.mark.parametrize("coreg_class", all_coregs)  # type: ignore
def test_subsample(load_examples, coreg_class: Callable) -> None:  # type: ignore

    ref, tba, _, inlier_mask_data, fit_params = load_examples

    # Check that default value is set properly
    coreg_full = coreg_class()
    argspec = inspect.getfullargspec(coreg_class)
    assert coreg_full.meta["subsample"] == argspec.defaults[argspec.args.index("subsample") - 1]  # type: ignore

    # But can be overridden during fit
    coreg_full.fit(**fit_params, subsample=10000, random_state=42)
    assert coreg_full.meta["subsample"] == 10000
    # Check that the random state is properly set when subsampling explicitly or implicitly
    assert coreg_full.meta["random_state"] == 42

    # Test subsampled vertical shift correction
    coreg_sub = coreg_class(subsample=0.1)
    assert coreg_sub.meta["subsample"] == 0.1

    # Fit the vertical shift using 10% of the unmasked data using a fraction
    coreg_sub.fit(**fit_params, random_state=42)
    # Do the same but specify the pixel count instead.
    # They are not perfectly equal (np.count_nonzero(.mask) // 2 would be exact)
    # But this would just repeat the subsample code, so that makes little sense to test.
    coreg_sub = coreg_class(subsample=tba.data.size // 10)
    assert coreg_sub.meta["subsample"] == tba.data.size // 10
    coreg_sub.fit(**fit_params, random_state=42)

    # Add a few performance checks
    coreg_name = coreg_class.__name__
    if coreg_name == "VerticalShift":
        # Check that the estimated vertical shifts are similar
        assert abs(coreg_sub.meta["shift_z"] - coreg_full.meta["shift_z"]) < 0.1

    elif coreg_name == "NuthKaab":
        # Calculate the difference in the full vs. subsampled matrices
        matrix_diff = np.abs(coreg_full.to_matrix() - coreg_sub.to_matrix())
        # Check that the x/y/z differences do not exceed 30cm
        assert np.count_nonzero(matrix_diff > 0.5) == 0

    elif coreg_name == "Tilt":
        # Check that the estimated biases are similar
        assert coreg_sub.meta["fit_params"] == pytest.approx(coreg_full.meta["fit_params"], rel=1e-1)


def test_subsample__pipeline(load_examples) -> None:
    """Test that the subsample argument works as intended for pipelines"""

    _, _, _, _, fit_params = load_examples

    # Check definition during instantiation
    pipe = coreg.VerticalShift(subsample=200) + coreg.Deramp(subsample=5000)

    # Check the arguments are properly defined
    assert pipe.pipeline[0].meta["subsample"] == 200
    assert pipe.pipeline[1].meta["subsample"] == 5000

    # Check definition during fit
    pipe = coreg.VerticalShift() + coreg.Deramp()
    pipe.fit(**fit_params, subsample=1000)
    assert pipe.pipeline[0].meta["subsample"] == 1000
    assert pipe.pipeline[1].meta["subsample"] == 1000


def test_subsample__errors(load_examples) -> None:
    """Check proper errors are raised when using the subsample argument"""

    # A warning should be raised when overriding with fit if non-default parameter was passed during instantiation
    _, _, _, _, fit_params = load_examples
    vshift = coreg.VerticalShift(subsample=100)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Subsample argument passed to fit() will override non-default "
            "subsample value defined at instantiation. To silence this "
            "warning: only define 'subsample' in either fit(subsample=...) "
            "or instantiation e.g. VerticalShift(subsample=...)."
        ),
    ):
        vshift.fit(**fit_params, subsample=1000)

    # Same for a pipeline
    pipe = coreg.VerticalShift(subsample=200) + coreg.Deramp()
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Subsample argument passed to fit() will override non-default "
            "subsample values defined for individual steps of the pipeline. "
            "To silence this warning: only define 'subsample' in either "
            "fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
        ),
    ):
        pipe.fit(**fit_params, subsample=1000)

    # Same for a blockwise co-registration
    block = coreg.BlockwiseCoreg(coreg.VerticalShift(subsample=200), subdivision=4)
    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Subsample argument passed to fit() will override non-default subsample "
            "values defined in the step within the blockwise method. To silence this "
            "warning: only define 'subsample' in either fit(subsample=...) or "
            "instantiation e.g., VerticalShift(subsample=...)."
        ),
    ):
        block.fit(**fit_params, subsample=1000)


def test_coreg_raster_and_ndarray_args() -> None:

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
    dem2 = dem1.reproject(bounds=rio.coords.BoundingBox(1, 0, 6, 5), silent=True)
    dem2 += 1

    # Create a vertical shift correction for Rasters ("_r") and for arrays ("_a")
    vshiftcorr_r = coreg.VerticalShift()
    vshiftcorr_a = vshiftcorr_r.copy()

    # Fit the data
    vshiftcorr_r.fit(reference_elev=dem1, to_be_aligned_elev=dem2)
    vshiftcorr_a.fit(
        reference_elev=dem1.data,
        to_be_aligned_elev=dem2.reproject(dem1, silent=True).data,
        transform=dem1.transform,
        crs=dem1.crs,
    )

    # Validate that they ended up giving the same result.
    assert vshiftcorr_r.meta["shift_z"] == vshiftcorr_a.meta["shift_z"]

    # De-shift dem2
    dem2_r = vshiftcorr_r.apply(dem2)
    dem2_a, _ = vshiftcorr_a.apply(dem2.data, transform=dem2.transform, crs=dem2.crs)

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
def test_apply_resample(inputs: list[Any], load_examples) -> None:
    """
    Test that the option resample of coreg.apply works as expected.
    For vertical correction only (VerticalShift, Deramp...), option True or False should yield same results.
    For horizontal shifts (NuthKaab etc), georef should differ, but DEMs should be the same after resampling.
    For others, the method is not implemented.
    """
    # Get test inputs
    coreg_method, is_implemented, comp = inputs
    ref_dem, tba_dem, outlines, inlier_mask, _ = load_examples

    # Prepare coreg
    coreg_method.fit(tba_dem, ref_dem, inlier_mask=inlier_mask)

    # If not implemented, should raise an error
    if not is_implemented:
        with pytest.raises(NotImplementedError, match="Option `resample=False` not implemented for coreg method *"):
            coreg_method.apply(tba_dem, resample=False)
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
    coreg_method.apply(tba_dem, resample=True, resampling=rio.warp.Resampling.nearest)
    coreg_method.apply(tba_dem, resample=True, resampling=rio.warp.Resampling.cubic)
    with pytest.raises(ValueError, match="'None' is not a valid rasterio.enums.Resampling method.*"):
        coreg_method.apply(tba_dem, resample=True, resampling=None)


@pytest.mark.parametrize("coreg_class", all_coregs)  # type: ignore
def test_fit_and_apply(coreg_class: Callable, load_examples) -> None:  # type: ignore
    """Check that fit_and_apply returns the same results as using fit, then apply, for any coreg."""

    _, _, _, _, fit_params = load_examples

    # Initiate two similar coregs
    coreg_fit_then_apply = coreg_class()
    coreg_fit_and_apply = coreg_class()

    # Perform fit, then apply
    coreg_fit_then_apply.fit(**fit_params, subsample=10000, random_state=42)
    aligned_then = coreg_fit_then_apply.apply(elev=fit_params["to_be_aligned_elev"])

    # Perform fit and apply
    aligned_and = coreg_fit_and_apply.fit_and_apply(**fit_params, subsample=10000, random_state=42)

    # Check outputs are the same: aligned raster, and metadata keys and values

    assert list(coreg_fit_and_apply.meta.keys()) == list(coreg_fit_then_apply.meta.keys())

    # TODO: Fix randomness of directional bias...
    if coreg_class != coreg.DirectionalBias:
        assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
        assert all(
            assert_coreg_meta_equal(coreg_fit_and_apply.meta[k], coreg_fit_then_apply.meta[k])
            for k in coreg_fit_and_apply.meta.keys()
        )


def test_fit_and_apply__pipeline(load_examples) -> None:
    """Check if it works for a pipeline"""

    _, _, _, _, fit_params = load_examples

    # Initiate two similar coregs
    coreg_fit_then_apply = coreg.NuthKaab() + coreg.Deramp()
    coreg_fit_and_apply = coreg.NuthKaab() + coreg.Deramp()

    # Perform fit, then apply
    coreg_fit_then_apply.fit(**fit_params, subsample=10000, random_state=42)
    aligned_then = coreg_fit_then_apply.apply(elev=fit_params["to_be_aligned_elev"])

    # Perform fit and apply
    aligned_and = coreg_fit_and_apply.fit_and_apply(**fit_params, subsample=10000, random_state=42)

    assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
    assert list(coreg_fit_and_apply.pipeline[0].meta.keys()) == list(coreg_fit_then_apply.pipeline[0].meta.keys())
    assert all(
        assert_coreg_meta_equal(coreg_fit_and_apply.pipeline[0].meta[k], coreg_fit_then_apply.pipeline[0].meta[k])
        for k in coreg_fit_and_apply.pipeline[0].meta.keys()
    )
    assert list(coreg_fit_and_apply.pipeline[1].meta.keys()) == list(coreg_fit_then_apply.pipeline[1].meta.keys())
    assert all(
        assert_coreg_meta_equal(coreg_fit_and_apply.pipeline[1].meta[k], coreg_fit_then_apply.pipeline[1].meta[k])
        for k in coreg_fit_and_apply.pipeline[1].meta.keys()
    )


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
            "'reference_dem' .* overrides the given *",
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
        (
            "None",
            "None",
            "None",
            "None",
            "fit",
            "error",
            "Input elevation data should be a raster, " "an array or a geodataframe.",
        ),
        ("dem1 + np.nan", "dem2", "None", "None", "fit", "error", "'reference_dem' had only NaNs"),
        ("dem1", "dem2 + np.nan", "None", "None", "fit", "error", "'dem_to_be_aligned' had only NaNs"),
    ],
)  # type: ignore
def test_coreg_raises(combination: tuple[str, str, str, str, str, str, str]) -> None:
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

    def fit_func() -> Coreg:
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


def test_coreg_oneliner() -> None:
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

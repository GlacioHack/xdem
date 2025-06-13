"""Functions to test the coregistration base classes."""

from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, Callable, Iterable, Mapping

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import pytransform3d.rotations
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType
from scipy.ndimage import binary_dilation

import xdem
from xdem import coreg, examples
from xdem._typing import NDArrayf
from xdem.coreg.base import Coreg, apply_matrix, dict_key_to_str


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

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


def assert_coreg_meta_equal(input1: Any, input2: Any) -> bool:
    """Short test function to check equality of coreg dictionary values."""

    # Different equality check based on input: number, callable, array, dataframe
    if not isinstance(input1, type(input2)):
        return False
    elif isinstance(input1, (str, float, int, np.floating, np.integer, tuple, list)) or callable(input1):
        return input1 == input2
    elif isinstance(input1, np.ndarray):
        return np.array_equal(input1, input2, equal_nan=True)
    elif isinstance(input1, pd.DataFrame):
        return input1.equals(input2)
    # If input is a dictionary, we recursively call this function to check equality of all its sub-keys
    elif isinstance(input1, dict):
        return all(assert_coreg_meta_equal(input1[k], input2[k]) for k in input1.keys())
    else:
        raise TypeError(f"Input type {type(input1)} not supported for this test function.")


class TestCoregClass:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)
    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    def test_init(self) -> None:
        """Test instantiation of Coreg"""

        c = coreg.Coreg()

        assert c._fit_called is False
        assert c._is_affine is None
        assert c._needs_vars is False

    def test_info(self) -> None:
        """
        Test all coreg keys required for info() exist by mapping all sub-keys in CoregDict and comparing to
        coreg.base.dict_key_to_str.
        Check the info() string return contains the right text for a given key.
        """

        # This recursive function will find all sub-keys that are not TypedDict within a TypedDict
        def recursive_typeddict_items(typed_dict: Mapping[str, Any]) -> Iterable[str]:
            for key, value in typed_dict.__annotations__.items():
                try:
                    sub_typed_dict = getattr(coreg.base, value.__forward_arg__)
                    if type(sub_typed_dict) is type(typed_dict):
                        yield from recursive_typeddict_items(sub_typed_dict)
                except AttributeError:
                    yield key

        # All subkeys
        list_coregdict_keys = list(recursive_typeddict_items(coreg.base.CoregDict))  # type: ignore

        # Assert all keys exist in the mapping key to str dictionary used for info
        list_info_keys = list(dict_key_to_str.keys())

        # Temporary exceptions: pipeline/blockwise
        list_exceptions = [
            "step_meta",
            "pipeline",
        ]

        # Compare the two lists
        list_missing_keys = [k for k in list_coregdict_keys if (k not in list_info_keys and k not in list_exceptions)]
        if len(list_missing_keys) > 0:
            raise AssertionError(
                f"Missing keys in coreg.base.dict_key_to_str " f"for Coreg.info(): {', '.join(list_missing_keys)}"
            )

        # Check that info() contains the mapped string for an example
        c = coreg.Coreg(meta={"subsample": 10000})
        assert dict_key_to_str["subsample"] in c.info(as_str=True)

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:
        """Test that copying work expectedly (that no attributes still share references)."""

        # Create a coreg instance and copy it.
        corr = coreg_class()
        corr_copy = corr.copy()

        # Assign some attributes and .metadata after copying, respecting the CoregDict type class
        corr._meta["outputs"]["affine"] = {"shift_z": 30}
        # Make sure these don't appear in the copy
        assert corr_copy.meta != corr.meta

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
        vshiftcorr.meta["outputs"]["affine"]["shift_z"] = 0
        # Now it should be equal to dem1 - dem2
        assert vshiftcorr.error(dem1, dem2, transform=affine, crs=crs, error_type="median") == -2

        # Create random noise and see if the standard deviation is equal (it should)
        rng = np.random.default_rng(42)
        dem3 = dem1.copy() + rng.random(size=dem1.size).reshape(dem1.shape)
        assert abs(vshiftcorr.error(dem1, dem3, transform=affine, crs=crs, error_type="std") - np.std(dem3)) < 1e-6

    @pytest.mark.parametrize("subsample", [10, 10000, 0.5, 1])  # type: ignore
    def test_get_subsample_on_valid_mask(self, subsample: float | int) -> None:
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

    all_coregs = [
        coreg.VerticalShift,
        coreg.NuthKaab,
        coreg.ICP,
        coreg.Deramp,
        coreg.TerrainBias,
        coreg.DirectionalBias,
    ]

    @pytest.mark.parametrize("coreg_class", all_coregs)  # type: ignore
    def test_subsample(self, coreg_class: Callable) -> None:  # type: ignore

        # Check that default value is set properly
        coreg_full = coreg_class()
        argspec = inspect.getfullargspec(coreg_class)
        assert (
            coreg_full.meta["inputs"]["random"]["subsample"]
            == argspec.defaults[argspec.args.index("subsample") - 1]  # type: ignore
        )

        # But can be overridden during fit
        coreg_full.fit(**self.fit_params, subsample=10000, random_state=42)
        assert coreg_full.meta["inputs"]["random"]["subsample"] == 10000
        # Check that the random state is properly set when subsampling explicitly or implicitly
        assert coreg_full.meta["inputs"]["random"]["random_state"] == 42

        # Test subsampled vertical shift correction
        coreg_sub = coreg_class(subsample=0.1)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == 0.1

        # Fit the vertical shift using 10% of the unmasked data using a fraction
        coreg_sub.fit(**self.fit_params, random_state=42)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        coreg_sub = coreg_class(subsample=self.tba.data.size // 10)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == self.tba.data.size // 10
        coreg_sub.fit(**self.fit_params, random_state=42)

        # Add a few performance checks
        coreg_name = coreg_class.__name__
        if coreg_name == "VerticalShift":
            # Check that the estimated vertical shifts are similar
            assert (
                abs(coreg_sub.meta["outputs"]["affine"]["shift_z"] - coreg_full.meta["outputs"]["affine"]["shift_z"])
                < 0.1
            )

        elif coreg_name == "NuthKaab":
            # Calculate the difference in the full vs. subsampled matrices
            matrix_diff = np.abs(coreg_full.to_matrix() - coreg_sub.to_matrix())
            # Check that the x/y/z differences do not exceed 30cm
            assert np.count_nonzero(matrix_diff > 0.5) == 0

    def test_subsample__pipeline(self) -> None:
        """Test that the subsample argument works as intended for pipelines"""

        # Check definition during instantiation
        pipe = coreg.VerticalShift(subsample=200) + coreg.Deramp(subsample=5000)

        # Check the arguments are properly defined
        assert pipe.pipeline[0].meta["inputs"]["random"]["subsample"] == 200
        assert pipe.pipeline[1].meta["inputs"]["random"]["subsample"] == 5000

        # Check definition during fit
        pipe = coreg.VerticalShift() + coreg.Deramp()
        pipe.fit(**self.fit_params, subsample=1000)
        assert pipe.pipeline[0].meta["inputs"]["random"]["subsample"] == 1000
        assert pipe.pipeline[1].meta["inputs"]["random"]["subsample"] == 1000

    def test_subsample__errors(self) -> None:
        """Check proper errors are raised when using the subsample argument"""

        # A warning should be raised when overriding with fit if non-default parameter was passed during instantiation
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
            vshift.fit(**self.fit_params, subsample=1000)

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
            pipe.fit(**self.fit_params, subsample=1000)

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
        assert vshiftcorr_r.meta["outputs"]["affine"]["shift_z"] == vshiftcorr_a.meta["outputs"]["affine"]["shift_z"]

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
            [xdem.coreg.NuthKaab(), True, "approx"],
            [xdem.coreg.NuthKaab() + xdem.coreg.VerticalShift(), True, "approx"],
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
        # Ignore curve_fit potential warnings
        warnings.filterwarnings("ignore", "Covariance of the parameters could not be estimated*")

        # Get test inputs
        coreg_method, is_implemented, comp = inputs
        ref_dem, tba_dem, outlines = load_examples()  # Load example reference, to-be-aligned and mask.

        # Prepare coreg
        inlier_mask = ~outlines.create_mask(ref_dem)
        coreg_method.fit(tba_dem, ref_dem, inlier_mask=inlier_mask)

        # If not implemented, should raise an error
        if not is_implemented:
            with pytest.raises(NotImplementedError, match="Option `resample=False` not supported*"):
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
    def test_fit_and_apply(self, coreg_class: Callable) -> None:  # type: ignore
        """Check that fit_and_apply returns the same results as using fit, then apply, for any coreg."""

        # Initiate two similar coregs
        coreg_fit_then_apply = coreg_class()
        coreg_fit_and_apply = coreg_class()

        # Perform fit, then apply
        coreg_fit_then_apply.fit(**self.fit_params, subsample=10000, random_state=42)
        aligned_then = coreg_fit_then_apply.apply(elev=self.fit_params["to_be_aligned_elev"])

        # Perform fit and apply
        aligned_and = coreg_fit_and_apply.fit_and_apply(**self.fit_params, subsample=10000, random_state=42)

        # Check outputs are the same: aligned raster, and metadata keys and values

        assert list(coreg_fit_and_apply.meta.keys()) == list(coreg_fit_then_apply.meta.keys())

        # TODO: Fix randomness of directional bias...
        if coreg_class != coreg.DirectionalBias:
            assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
            assert all(
                assert_coreg_meta_equal(coreg_fit_and_apply.meta[k], coreg_fit_then_apply.meta[k])
                for k in coreg_fit_and_apply.meta.keys()
            )

    def test_fit_and_apply__pipeline(self) -> None:
        """Check if it works for a pipeline"""

        # Initiate two similar coregs
        coreg_fit_then_apply = coreg.NuthKaab() + coreg.Deramp()
        coreg_fit_and_apply = coreg.NuthKaab() + coreg.Deramp()

        # Perform fit, then apply
        coreg_fit_then_apply.fit(**self.fit_params, subsample=10000, random_state=42)
        aligned_then = coreg_fit_then_apply.apply(elev=self.fit_params["to_be_aligned_elev"])

        # Perform fit and apply
        aligned_and = coreg_fit_and_apply.fit_and_apply(**self.fit_params, subsample=10000, random_state=42)

        assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
        assert list(coreg_fit_and_apply.pipeline[0].meta.keys()) == list(coreg_fit_then_apply.pipeline[0].meta.keys())
        assert all(
            assert_coreg_meta_equal(
                coreg_fit_and_apply.pipeline[0].meta[k], coreg_fit_then_apply.pipeline[0].meta[k]  # type: ignore
            )
            for k in coreg_fit_and_apply.pipeline[0].meta.keys()
        )
        assert list(coreg_fit_and_apply.pipeline[1].meta.keys()) == list(coreg_fit_then_apply.pipeline[1].meta.keys())
        assert all(
            assert_coreg_meta_equal(
                coreg_fit_and_apply.pipeline[1].meta[k], coreg_fit_then_apply.pipeline[1].meta[k]  # type: ignore
            )
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


class TestCoregPipeline:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_elev=ref.data,
        to_be_aligned_elev=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:

        # Create a pipeline, add some .metadata, and copy it
        pipeline = coreg_class() + coreg_class()
        pipeline.pipeline[0]._meta["outputs"]["affine"] = {"shift_z": 1}

        pipeline_copy = pipeline.copy()

        # Add some more .metadata after copying (this should not be transferred)
        pipeline_copy.pipeline[0]._meta["outputs"]["affine"].update({"shift_y": 0.5 * 30})

        assert pipeline.pipeline[0].meta != pipeline_copy.pipeline[0].meta
        assert pipeline_copy.pipeline[0]._meta["outputs"]["affine"]["shift_z"]

    def test_pipeline(self) -> None:

        # Create a pipeline from two coreg methods.
        pipeline = coreg.CoregPipeline([coreg.VerticalShift(), coreg.NuthKaab()])
        pipeline.fit(**self.fit_params, subsample=5000, random_state=42)

        aligned_dem, _ = pipeline.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

        # Make a new pipeline with two vertical shift correction approaches.
        pipeline2 = coreg.CoregPipeline([coreg.VerticalShift(), coreg.VerticalShift()])
        # Set both "estimated" vertical shifts to be 1
        pipeline2.pipeline[0].meta["outputs"]["affine"] = {"shift_z": 1}
        pipeline2.pipeline[1].meta["outputs"]["affine"] = {"shift_z": 1}

        # Assert that the combined vertical shift is 2
        assert pipeline2.to_matrix()[2, 3] == 2.0

    # TODO: Figure out why DirectionalBias + DirectionalBias pipeline fails with Scipy error
    #  on bounds constraints on Mac only?
    all_coregs = [
        coreg.VerticalShift,
        coreg.NuthKaab,
        coreg.ICP,
        coreg.Deramp,
        coreg.TerrainBias,
        # coreg.DirectionalBias,
    ]

    @pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
    @pytest.mark.parametrize("coreg2", all_coregs)  # type: ignore
    def test_pipeline_combinations__nobiasvar(self, coreg1: Callable[[], Coreg], coreg2: Callable[[], Coreg]) -> None:
        """Test pipelines with all combinations of coregistration subclasses (without bias variables)"""

        # Create a pipeline from one affine and one biascorr methods.
        pipeline = coreg.CoregPipeline([coreg1(), coreg2()])
        pipeline.fit(**self.fit_params, subsample=5000, random_state=42)

        aligned_dem, _ = pipeline.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)
        assert aligned_dem.shape == self.ref.data.squeeze().shape

    @pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
    @pytest.mark.parametrize(
        "coreg2_init_kwargs",
        [
            dict(bias_var_names=["slope"], fit_or_bin="bin"),
            dict(bias_var_names=["slope", "aspect"], fit_or_bin="bin"),
        ],
    )  # type: ignore
    def test_pipeline_combinations__biasvar(
        self, coreg1: Callable[[], Coreg], coreg2_init_kwargs: dict[str, str]
    ) -> None:
        """Test pipelines with all combinations of coregistration subclasses with bias variables"""

        # Create a pipeline from one affine and one biascorr methods
        pipeline = coreg.CoregPipeline([coreg1(), coreg.BiasCorr(**coreg2_init_kwargs)])  # type: ignore
        bias_vars = {"slope": xdem.terrain.slope(self.ref), "aspect": xdem.terrain.aspect(self.ref)}
        pipeline.fit(**self.fit_params, bias_vars=bias_vars, subsample=5000, random_state=42)

        aligned_dem, _ = pipeline.apply(
            self.tba.data, transform=self.ref.transform, crs=self.ref.crs, bias_vars=bias_vars
        )
        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline__errors(self) -> None:
        """Test pipeline raises proper errors."""

        pipeline = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr()])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No `bias_vars` passed to .fit() for bias correction step "
                "<class 'xdem.coreg.biascorr.BiasCorr'> of the pipeline."
            ),
        ):
            pipeline.fit(**self.fit_params)

        pipeline2 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr(), coreg.BiasCorr()])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No `bias_vars` passed to .fit() for bias correction step <class 'xdem.coreg.biascorr.BiasCorr'> "
                "of the pipeline. As you are using several bias correction steps requiring"
                " `bias_vars`, don't forget to explicitly define their `bias_var_names` "
                "during instantiation, e.g. BiasCorr(bias_var_names=['slope'])."
            ),
        ):
            pipeline2.fit(**self.fit_params)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "When using several bias correction steps requiring `bias_vars` in a pipeline,"
                "the `bias_var_names` need to be explicitly defined at each step's "
                "instantiation, e.g. BiasCorr(bias_var_names=['slope'])."
            ),
        ):
            pipeline2.fit(**self.fit_params, bias_vars={"slope": xdem.terrain.slope(self.ref)})

        pipeline3 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr(bias_var_names=["slope"])])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step <class 'xdem.coreg.biascorr.BiasCorr'>: ['slope']."
            ),
        ):
            pipeline3.fit(**self.fit_params, bias_vars={"ncc": xdem.terrain.slope(self.ref)})

    def test_pipeline_pts(self) -> None:

        pipeline = coreg.NuthKaab() + coreg.DhMinimize()
        ref_points = self.ref.to_pointcloud(subsample=5000, random_state=42).ds
        ref_points["E"] = ref_points.geometry.x
        ref_points["N"] = ref_points.geometry.y
        ref_points.rename(columns={"b1": "z"}, inplace=True)

        # Check that this runs without error
        pipeline.fit(reference_elev=ref_points, to_be_aligned_elev=self.tba)

        for part in pipeline.pipeline:
            assert np.abs(part.meta["outputs"]["affine"]["shift_x"]) > 0

        assert (
            pipeline.pipeline[0].meta["outputs"]["affine"]["shift_x"]
            != pipeline.pipeline[1].meta["outputs"]["affine"]["shift_x"]
        )

    def test_coreg_add(self) -> None:

        # Test with a vertical shift of 4
        vshift = 4

        vshift1 = coreg.VerticalShift()
        vshift2 = coreg.VerticalShift()

        # Set the vertical shift attribute
        for vshift_corr in (vshift1, vshift2):
            vshift_corr.meta["outputs"]["affine"] = {"shift_z": vshift}

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

    def test_pipeline_consistency(self) -> None:
        """Check that pipelines properties are respected: reflectivity, fusion of same coreg"""

        # Test 1: Fusion of same coreg
        # Many vertical shifts
        many_vshifts = coreg.VerticalShift() + coreg.VerticalShift() + coreg.VerticalShift()
        many_vshifts.fit(**self.fit_params, random_state=42)
        aligned_dem, _ = many_vshifts.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        # The last steps should have shifts of EXACTLY zero
        assert many_vshifts.pipeline[1].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=10e-5)
        assert many_vshifts.pipeline[2].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=10e-5)

        # Many horizontal + vertical shifts
        many_nks = coreg.NuthKaab() + coreg.NuthKaab() + coreg.NuthKaab()
        many_nks.fit(**self.fit_params, random_state=42)
        aligned_dem, _ = many_nks.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        # The last steps should have shifts of NEARLY zero
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=0.05)
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_x"] == pytest.approx(0, abs=0.05)
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_y"] == pytest.approx(0, abs=0.05)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=0.05)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_x"] == pytest.approx(0, abs=0.05)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_y"] == pytest.approx(0, abs=0.05)

        # Test 2: Reflectivity
        # Those two pipelines should give almost the same result
        nk_vshift = coreg.NuthKaab() + coreg.VerticalShift()
        vshift_nk = coreg.VerticalShift() + coreg.NuthKaab()

        nk_vshift.fit(**self.fit_params, random_state=42)
        aligned_dem, _ = nk_vshift.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)
        vshift_nk.fit(**self.fit_params, random_state=42)
        aligned_dem, _ = vshift_nk.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        assert np.allclose(nk_vshift.to_matrix(), vshift_nk.to_matrix(), atol=10e-1)


class TestAffineManipulation:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.

    # Identity transformation
    matrix_identity = np.diag(np.ones(4, float))

    # Vertical shift
    matrix_vertical = matrix_identity.copy()
    matrix_vertical[2, 3] = 1

    # Vertical and horizontal shifts
    matrix_translations = matrix_identity.copy()
    matrix_translations[:3, 3] = [0.5, 1, 1.5]

    # Single rotation
    rotation = np.deg2rad(5)
    matrix_rotations = matrix_identity.copy()
    matrix_rotations[1, 1] = np.cos(rotation)
    matrix_rotations[2, 2] = np.cos(rotation)
    matrix_rotations[2, 1] = -np.sin(rotation)
    matrix_rotations[1, 2] = np.sin(rotation)

    # Mix of translations and rotations in all axes (X, Y, Z) simultaneously
    rotation_x = 5
    rotation_y = 10
    rotation_z = 3
    e = np.deg2rad(np.array([rotation_x, rotation_y, rotation_z]))
    # This is a 3x3 rotation matrix
    rot_matrix = pytransform3d.rotations.matrix_from_euler(e=e, i=0, j=1, k=2, extrinsic=True)
    matrix_all = matrix_rotations.copy()
    matrix_all[0:3, 0:3] = rot_matrix
    matrix_all[:3, 3] = [0.5, 1, 1.5]

    list_matrices = [matrix_identity, matrix_vertical, matrix_translations, matrix_rotations, matrix_all]

    @pytest.mark.parametrize("matrix", list_matrices)  # type: ignore
    def test_apply_matrix__points_geopandas(self, matrix: NDArrayf) -> None:
        """
        Test that apply matrix's exact transformation for points (implemented with NumPy matrix multiplication)
        is exactly the same as the one of GeoPandas.
        """

        # Create random points
        points = np.random.default_rng(42).normal(size=(10, 3))

        # Convert to a geodataframe and use apply_matrix for the point cloud
        epc = gpd.GeoDataFrame(data={"z": points[:, 2]}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]))
        trans_epc = apply_matrix(epc, matrix=matrix)

        # Compare to geopandas transformation
        # We first need to convert the 4x4 affine matrix into a 12-parameter affine matrix
        epc_3d = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1], z=points[:, 2]))
        mat_12params = np.zeros(12)
        mat_12params[:9] = matrix[:3, :3].flatten()
        mat_12params[9:] = matrix[:3, 3]
        trans_epc_gpd = epc_3d.affine_transform(matrix=mat_12params)

        # Check both transformations are equal
        assert np.allclose(trans_epc.geometry.x.values, trans_epc_gpd.geometry.x.values)
        assert np.allclose(trans_epc.geometry.y.values, trans_epc_gpd.geometry.y.values)
        assert np.allclose(trans_epc["z"].values, trans_epc_gpd.geometry.z.values)

    @pytest.mark.parametrize("regrid_method", [None, "iterative", "griddata"])  # type: ignore
    @pytest.mark.parametrize("matrix", list_matrices)  # type: ignore
    def test_apply_matrix__raster(self, regrid_method: None | str, matrix: NDArrayf) -> None:
        """Test that apply matrix gives consistent results between points and rasters (thus validating raster
        implementation, as point implementation is validated above), for all possible regridding methods."""

        # Create a synthetic raster and convert to point cloud
        # dem = gu.Raster(self.ref)
        dem_arr = np.linspace(0, 2, 25).reshape(5, 5)
        transform = rio.transform.from_origin(0, 5, 1, 1)
        dem = gu.Raster.from_array(dem_arr, transform=transform, crs=4326, nodata=100)
        epc = dem.to_pointcloud(data_column_name="z").ds

        # If a centroid was not given, default to the center of the DEM (at Z=0).
        centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

        # Apply affine transformation to both datasets
        trans_dem = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method=regrid_method)
        trans_epc = apply_matrix(epc, matrix=matrix, centroid=centroid)

        # Interpolate transformed DEM at coordinates of the transformed point cloud
        # Because the raster created as a constant slope (plan-like), the interpolated values should be very close
        z_points = trans_dem.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))
        valids = np.isfinite(z_points)
        assert np.count_nonzero(valids) > 0
        assert np.allclose(z_points[valids], trans_epc.z.values[valids], rtol=10e-5)

    def test_apply_matrix__raster_nodata(self) -> None:
        """Test the nodatas created by apply_matrix are consistent between methods"""

        # Use matrix with all transformations
        matrix = self.matrix_all

        # Create a synthetic raster, add NaNs, and convert to point cloud
        dem_arr = np.linspace(0, 2, 400).reshape(20, 20)
        dem_arr[10:14, 10:14] = np.nan
        dem_arr[5, 5] = np.nan
        dem_arr[:2, :] = np.nan
        transform = rio.transform.from_origin(0, 5, 1, 1)
        dem = gu.Raster.from_array(dem_arr, transform=transform, crs=4326, nodata=100)
        epc = dem.to_pointcloud(data_column_name="z").ds

        centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

        trans_dem_it = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="iterative")
        trans_dem_gd = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="griddata")

        # Get nodata mask
        mask_nodata_it = trans_dem_it.data.mask
        mask_nodata_gd = trans_dem_gd.data.mask

        # The iterative mask should be larger and contain the other (as griddata interpolates up to 1 pixel away)
        assert np.array_equal(np.logical_or(mask_nodata_gd, mask_nodata_it), mask_nodata_it)

        # Verify nodata masks are located within two pixels of each other (1 pixel can be added by griddata,
        # and 1 removed by regular-grid interpolation by the iterative method)
        smallest_mask = ~binary_dilation(
            ~mask_nodata_it, iterations=2
        )  # Invert before dilate to avoid spreading at the edges
        # All smallest mask value should exist in the mask of griddata
        assert np.array_equal(np.logical_or(smallest_mask, mask_nodata_gd), mask_nodata_gd)

    def test_apply_matrix__raster_realdata(self) -> None:
        """Testing real data no complex matrix only to avoid all loops"""

        # Use real data
        dem = self.ref
        dem.crop((dem.bounds.left, dem.bounds.bottom, dem.bounds.left + 2000, dem.bounds.bottom + 2000))
        epc = dem.to_pointcloud(data_column_name="z").ds

        # Only testing complex matrices for speed
        matrix = self.matrix_all

        # If a centroid was not given, default to the center of the DEM (at Z=0).
        centroid = (np.mean(epc.geometry.x.values), np.mean(epc.geometry.y.values), 0.0)

        # Apply affine transformation to both datasets
        trans_dem_it = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="iterative")
        trans_dem_gd = apply_matrix(dem, matrix=matrix, centroid=centroid, force_regrid_method="griddata")
        trans_epc = apply_matrix(epc, matrix=matrix, centroid=centroid)

        # Interpolate transformed DEM at coordinates of the transformed point cloud, and check values are very close
        z_points_it = trans_dem_it.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))
        z_points_gd = trans_dem_gd.interp_points(points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values))

        valids = np.logical_and(np.isfinite(z_points_it), np.isfinite(z_points_gd))
        assert np.count_nonzero(valids) > 0

        diff_it = z_points_it[valids] - trans_epc.z.values[valids]
        diff_gd = z_points_gd[valids] - trans_epc.z.values[valids]

        # Because of outliers, noise and slope near 90°, several solutions can exist
        # Additionally, contrary to the check in the __raster test which uses a constant slope DEM, the slopes vary
        # here so the interpolation check is less accurate so all values can vary a bit
        assert np.percentile(np.abs(diff_it), 90) < 1
        assert np.percentile(np.abs(diff_it), 50) < 0.2
        assert np.percentile(np.abs(diff_gd), 90) < 1
        assert np.percentile(np.abs(diff_gd), 50) < 0.2

        # But, between themselves, the two re-gridding methods should yield much closer results
        # (no errors due to 2D interpolation for checking)
        diff_it_gd = z_points_gd[valids] - z_points_it[valids]
        assert np.percentile(np.abs(diff_it_gd), 99) < 1  # 99% of values are within a meter (instead of 90%)
        assert np.percentile(np.abs(diff_it_gd), 50) < 0.02  # 10 times more precise than above

"""Functions to test the coregistration base classes."""

from __future__ import annotations

import inspect
import re
import sys
import warnings
from typing import Any, Callable, Iterable, Mapping

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import pytest
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

    ref_dem = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path_test("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


def load_examples_full() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


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

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:
        """Test that copying work expectedly (that no attributes still share references)."""

        # Create a coreg instance and copy it.
        corr = coreg_class()
        corr_copy = corr.copy()

        # Assign some attributes and .metadata after copying, respecting the CoregDict type class
        corr._meta["outputs"]["affine"] = {"shift_z": 30}
        # Make sure these don't appear in the copy
        assert corr_copy.meta != corr.meta

    @pytest.mark.parametrize("subsample", [10, 10000, 0.5, 1])
    def test_get_subsample_on_valid_mask(self, subsample: float | int) -> None:
        """Test the subsampling function called by all subclasses"""

        # Define a valid mask
        width = height = 50
        rng = np.random.default_rng(42)
        valid_mask = rng.integers(low=0, high=2, size=(width, height), dtype=bool)

        # Define a class with a subsample and random_state in the .metadata
        params_random={"subsample": subsample, "random_state": 42}

        subsample_mask = xdem.coreg.base._get_subsample_on_valid_mask(params_random=params_random,
                                                                      valid_mask=valid_mask)

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

    @pytest.mark.skipif(sys.platform != "linux", reason="Basinhopping from DirectionalBias fails on Mac")
    @pytest.mark.parametrize("coreg_class", all_coregs)
    def test_subsample(self, coreg_class: Any) -> None:

        # Check that default value is set properly
        coreg_full = coreg_class()
        argspec = inspect.getfullargspec(coreg_class)
        assert (
            coreg_full.meta["inputs"]["random"]["subsample"]
            == argspec.defaults[argspec.args.index("subsample") - 1]  # type: ignore
        )

        # Add keyword arguments for speed on basinhopping method
        if coreg_class == coreg.DirectionalBias:
            fit_kwargs = {"niter": 1}  # Only run one iteration
        else:
            fit_kwargs = {}

        # But can be overridden during fit
        coreg_full.fit(**self.fit_params, subsample=10000, random_state=42, **fit_kwargs)
        assert coreg_full.meta["inputs"]["random"]["subsample"] == 10000
        # Check that the random state is properly set when subsampling explicitly or implicitly
        assert coreg_full.meta["inputs"]["random"]["random_state"] == 42

        # Test subsampled vertical shift correction
        coreg_sub = coreg_class(subsample=0.1)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == 0.1

        # Fit the vertical shift using 10% of the unmasked data using a fraction
        coreg_sub.fit(**self.fit_params, random_state=42, **fit_kwargs)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        coreg_sub = coreg_class(subsample=self.tba.data.size // 10)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == self.tba.data.size // 10
        coreg_sub.fit(**self.fit_params, random_state=42, **fit_kwargs)

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
    )
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

    @pytest.mark.skipif(sys.platform != "linux", reason="Basinhopping from DirectionalBias fails on Mac")
    @pytest.mark.parametrize("coreg_class", all_coregs)
    def test_fit_and_apply(self, coreg_class: Any) -> None:
        """Check that fit_and_apply returns the same results as using fit, then apply, for any coreg."""

        # Initiate two similar coregs
        coreg_fit_then_apply = coreg_class()
        coreg_fit_and_apply = coreg_class()

        # Add keyword arguments for speed on basinhopping method
        if coreg_class == coreg.DirectionalBias:
            fit_kwargs = {"niter": 1}  # Only run one iteration
        else:
            fit_kwargs = {}

        # Perform fit, then apply
        coreg_fit_then_apply.fit(**self.fit_params, subsample=10000, random_state=42, **fit_kwargs)
        aligned_then = coreg_fit_then_apply.apply(elev=self.fit_params["to_be_aligned_elev"])

        # Perform fit and apply
        aligned_and = coreg_fit_and_apply.fit_and_apply(
            **self.fit_params, subsample=10000, random_state=42, fit_kwargs=fit_kwargs
        )

        # Check outputs are the same: aligned raster, and metadata keys and values

        assert list(coreg_fit_and_apply.meta.keys()) == list(coreg_fit_then_apply.meta.keys())

        # TODO: Fix randomness of directional bias...
        if coreg_class != coreg.DirectionalBias:
            assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
            assert all(
                assert_coreg_meta_equal(coreg_fit_and_apply.meta[k], coreg_fit_then_apply.meta[k])
                for k in coreg_fit_and_apply.meta.keys()
            )

    @pytest.mark.parametrize(
        "combination",
        [
            ("raster", "raster", False, "raster", "passes", ""),
            ("raster", "raster", False, "array", "error", "Input mask array"),
            ("raster", "raster", True, "raster", "passes", ""),
            ("array", "raster", True, "raster", "passes", ""),
            ("raster", "array", True, "raster", "passes", ""),
            ("array", "array", True, "raster", "passes", ""),
            ("pc", "raster", False, "raster", "passes", ""),
            ("raster", "pc", False, "raster", "passes", ""),
            ("pc", "array", True, "array", "error", "Input mask array"),
            ("array", "pc", True, "array", "error", "Input mask array"),
        ],
    )
    def test_fit_and_apply__cropped_mask(self, combination: tuple[str, str, str, str, str, str]) -> None:
        """
        Assert that the same mask, no matter its projection, gives the same results after a fit_and_apply (by shift
        output values). NuthKaab has been chosen if this case but the method doesn't change anything.

        The 'combination' param contains this in order:
            1. The ref_type : raster, array or pc for pointclouds
            2. The tba_type : raster, array or pc for pointclouds
            3. If the fit_and_apply needs ref_dem.transform and ref_dem.crs
            4. The mask_type : raster or array
            6. The expected outcome of the test
            7. The error message (if applicable)
        """

        ref_type, tba_type, info, mask_type, result, text = combination

        # Init data
        ref_dem, tba_dem, mask = load_examples()
        inlier_mask = ~mask.create_mask(ref_dem)

        # Load dem_ref info if needed
        transform = None
        crs = None
        if info:
            transform = ref_dem.transform
            crs = ref_dem.crs

        # Crop mask
        nrows, ncols = inlier_mask.shape
        inlier_mask_crop = inlier_mask.icrop((0, 0, ncols - 10, nrows - 10))

        # And reprojected the cropped mask to have the same size as before
        inlier_mask_crop_proj = inlier_mask_crop.reproject(ref_dem, resampling=rio.warp.Resampling.nearest)

        # Evaluate the type of the inputs
        if ref_type == "array":
            ref_dem = ref_dem.data
        elif ref_type == "pc":
            ref_dem = ref_dem.to_pointcloud().ds
            ref_dem.rename(columns={"b1": "z"}, inplace=True)
        if tba_type == "array":
            tba_dem = tba_dem.data
        elif tba_type == "pc":
            tba_dem = tba_dem.to_pointcloud().ds
            tba_dem.rename(columns={"b1": "z"}, inplace=True)
        if mask_type == "array":
            inlier_mask_crop = inlier_mask_crop.data

        list_shift = ["shift_x", "shift_y", "shift_z"]
        warnings.filterwarnings("ignore")  # to do the process until the end

        # Use VerticalShift as a representative example.
        nuthkaab_ref = xdem.coreg.NuthKaab()
        nuthkaab_ref.fit_and_apply(
            ref_dem, tba_dem, inlier_mask=inlier_mask_crop_proj, transform=transform, crs=crs, random_state=42
        )
        shifts_ref = [nuthkaab_ref.meta["outputs"]["affine"][k] for k in list_shift]  # type: ignore

        nuthkaab_crop = xdem.coreg.NuthKaab()
        if result == "error":
            with pytest.raises(ValueError, match=re.escape(text)):
                nuthkaab_crop.fit_and_apply(
                    ref_dem, tba_dem, inlier_mask=inlier_mask_crop, transform=transform, crs=crs, random_state=42
                )
            return
        else:
            nuthkaab_crop.fit_and_apply(
                ref_dem, tba_dem, inlier_mask=inlier_mask_crop, transform=transform, crs=crs, random_state=42
            )
        shifts_crop = [nuthkaab_crop.meta["outputs"]["affine"][k] for k in list_shift]  # type: ignore

        # Check the output shifts match
        assert shifts_ref == pytest.approx(shifts_crop)

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
                "Input elevation data should be a raster, array, geodataframe or point cloud.*",
            ),
            ("dem1 + np.nan", "dem2", "None", "None", "fit", "error", "'reference_dem' had only NaNs"),
            ("dem1", "dem2 + np.nan", "None", "None", "fit", "error", "'dem_to_be_aligned' had only NaNs"),
        ],
    )
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
    trans_x = 0.5
    trans_y = 1
    trans_z = 1.5
    # This is a 3x3 rotation matrix
    matrix_all = xdem.coreg.matrix_from_translations_rotations(
        trans_x, trans_y, trans_z, rotation_x, rotation_y, rotation_z
    )

    list_matrices = [matrix_identity, matrix_vertical, matrix_translations, matrix_rotations, matrix_all]

    @pytest.mark.parametrize("matrix", list_matrices)
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

    @pytest.mark.parametrize("regrid_method", [None, "iterative", "griddata"])
    @pytest.mark.parametrize("matrix", list_matrices)
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
        z_points = trans_dem.interp_points(
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values), as_array=True
        )
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
        z_points_it = trans_dem_it.interp_points(
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values), as_array=True
        )
        z_points_gd = trans_dem_gd.interp_points(
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values), as_array=True
        )

        valids = np.logical_and(np.isfinite(z_points_it), np.isfinite(z_points_gd))
        assert np.count_nonzero(valids) > 0

        diff_it = z_points_it[valids] - trans_epc.z.values[valids]
        diff_gd = z_points_gd[valids] - trans_epc.z.values[valids]

        # Because of outliers, noise and slope near 90°, several solutions can exist
        # Additionally, contrary to the check in the __raster test which uses a constant slope DEM, the slopes vary
        # here so the interpolation check is less accurate so all values can vary a bit
        assert np.percentile(np.abs(diff_it), 90) < 1.2
        assert np.percentile(np.abs(diff_it), 50) < 0.3
        assert np.percentile(np.abs(diff_gd), 90) < 1.2
        assert np.percentile(np.abs(diff_gd), 50) < 0.3

        # But, between themselves, the two re-gridding methods should yield much closer results
        # (no errors due to 2D interpolation for checking)
        diff_it_gd = z_points_gd[valids] - z_points_it[valids]
        assert np.percentile(np.abs(diff_it_gd), 99) < 1.2  # 99% of values are within a 1.20 meter (instead of 90%)
        assert np.percentile(np.abs(diff_it_gd), 50) < 0.03  # 10 times more precise than above

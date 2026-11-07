"""Functions to test the coregistration base classes."""

from __future__ import annotations

import inspect
import re
import sys
import warnings
from typing import Any, Callable, Iterable, Literal, Mapping
from unittest.mock import patch

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
    if input1 is None:
        return input2 is None
    elif not isinstance(input1, type(input2)):
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
    def test_coreg_cosampling_support(self, subsample: float | int) -> None:
        """Checks that coregistration sampling preserves valid locations, observation order and auxiliary values."""

        # Give each observation a unique value so sampled coordinates and auxiliaries can be checked together
        rng = np.random.default_rng(42)
        valid_mask = rng.integers(0, 2, size=(50, 50), dtype=bool)
        values = np.arange(2500.0).reshape(valid_mask.shape)
        raster = Raster.from_array(values, rio.transform.from_origin(0, 50, 1, 1), 32632, area_or_point="Area")
        options = dict(
            ref_elev=raster,
            tba_elev=raster + 2,
            inlier_mask=raster.copy(new_array=valid_mask),
            subsample=subsample,
            random_state=42,
            aux_vars={"quality": values * 2},
        )

        # Draw the same sample twice to check reproducible correspondence between the datasets
        reference, aligned, auxiliary = xdem.coreg.base._sample_coreg_inputs(**options)
        repeated = xdem.coreg.base._sample_coreg_inputs(**options)[0]

        # Preserve the sample budget, common finite population and reproducible ordering
        expected = int(subsample * np.count_nonzero(valid_mask)) if subsample <= 1 else int(subsample)
        assert reference.shape == (3, min(expected, np.count_nonzero(valid_mask)))
        assert valid_mask.ravel()[reference[2].astype(int)].all()
        np.testing.assert_array_equal(reference, repeated)

        # Check shared coordinates and the known two-unit difference between paired elevations
        np.testing.assert_array_equal(reference[:2], aligned[:2])
        np.testing.assert_array_equal(reference[2] + 2, aligned[2])
        assert np.all(np.diff(reference[2]) > 0)

        # Auxiliary values must follow the same selected observations and ordering
        assert auxiliary is not None
        np.testing.assert_array_equal(auxiliary["quality"], reference[2] * 2)

    @pytest.mark.parametrize("sampling_strategy", ["same_xy", "nearest_xy", "independent"])
    def test_sample_coreg_inputs__raster_strategies(
        self, sampling_strategy: Literal["independent", "same_xy", "nearest_xy"]
    ) -> None:
        """Checks that sampling uses same X/Y, nearest X/Y or independent raster coordinates."""

        # 1/ Create shifted raster grids
        rows, columns = np.indices((6, 8))
        values = (rows * 8 + columns).astype(float)
        reference = Raster.from_array(values, rio.transform.from_origin(0, 6, 1, 1), 32632, area_or_point="Area")
        moving = reference.translate(0.3, -0.2) + 2

        # 2/ Sample elevations with the requested strategy
        first, second, auxiliary = xdem.coreg.base._sample_coreg_inputs(
            reference,
            moving,
            None,
            subsample=7,
            sampling_strategy=sampling_strategy,
            random_state=42,
            aux_vars={"quality": moving},  # We duplicate "moving" here, to check sampling of auxiliary variables
            aux_tied_to="tba",
        )

        # 3/ Checks
        # Shape should contain X/Y/Z
        assert first.shape[0] == second.shape[0] == 3

        # Auxiliary values should be sampled like "second", thus equal because we passed them above
        assert auxiliary is not None
        np.testing.assert_array_equal(auxiliary["quality"], second[2])
        # For same_xy, both rasters are sampled at shared coordinates
        if sampling_strategy == "same_xy":
            np.testing.assert_array_equal(first[:2], second[:2])
        else:
            # For nearest_xy and independent, elevations minus 2 identify the original moving pixels
            expected = moving.ij2xy(*np.unravel_index((second[2] - 2).astype(int), moving.shape))
            np.testing.assert_array_equal(second[:2], expected)
            # For nearest_xy, paired pixel centers differ by the raster grid translation
            if sampling_strategy == "nearest_xy":
                np.testing.assert_allclose(
                    second[:2] - first[:2], np.tile([[0.3], [-0.2]], (1, first.shape[1])), rtol=0, atol=1e-14
                )

    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize("sampling_strategy", ["same_xy", "nearest_xy"])
    def test_sample_coreg_inputs__point_target(
        self, reverse: bool, sampling_strategy: Literal["independent", "same_xy", "nearest_xy"]
    ) -> None:
        """Checks that point cloud coordinates define the sample in either raster and point cloud order."""

        # 1/ Create points between pixel centers
        rows, columns = np.indices((6, 8))
        raster = Raster.from_array(
            (2 * columns + 3 * rows).astype(float), rio.transform.from_origin(0, 6, 1, 1), 32632, area_or_point="Area"
        )
        x, y = raster.ij2xy([1, 2, 3, 4], [1, 3, 4, 6])
        points = gu.PointCloud.from_xyz(x + 0.3, y - 0.2, np.arange(4.0), crs=raster.crs)
        inputs = (points, raster) if reverse else (raster, points)
        point_owner = "ref" if reverse else "tba"

        # 2/ Sample elevations with the requested strategy
        first, second, auxiliary = xdem.coreg.base._sample_coreg_inputs(
            *inputs,
            None,
            subsample=3,
            sampling_strategy=sampling_strategy,
            random_state=42,
            aux_vars={"quality": (points, "z")},  # Reuse point elevations to check auxiliary sampling
            aux_tied_to=point_owner,
        )

        # 3/ Checks
        raster_sample, point_sample = (second, first) if reverse else (first, second)
        # Point elevations store their original indexes
        indexes = point_sample[2].astype(int)
        # Shape should contain X/Y/Z for three sampled points
        assert point_sample.shape == raster_sample.shape == (3, 3)
        # Point coordinates should be unchanged in either input order
        np.testing.assert_array_equal(point_sample[:2], np.vstack((x + 0.3, y - 0.2))[:, indexes])

        # Auxiliary values should match point elevations because we passed the same column
        assert auxiliary is not None
        np.testing.assert_array_equal(auxiliary["quality"], point_sample[2])
        # For same_xy, raster elevations are interpolated at the point coordinates
        if sampling_strategy == "same_xy":
            np.testing.assert_array_equal(raster_sample[:2], point_sample[:2])
            # Pixel centers give column = X - 0.5 and row = 5.5 - Y for the planar raster
            np.testing.assert_allclose(raster_sample[2], 2 * (point_sample[0] - 0.5) + 3 * (5.5 - point_sample[1]))
        else:
            # For nearest_xy, use the nearest pixel centers and their original raster values
            np.testing.assert_array_equal(raster_sample[:2], np.vstack((x, y))[:, indexes])
            np.testing.assert_array_equal(raster_sample[2], raster.data[[1, 2, 3, 4], [1, 3, 4, 6]][indexes])

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
        coreg_full.fit(**self.fit_params, subsample=1000, random_state=42, **fit_kwargs)
        assert coreg_full.meta["inputs"]["random"]["subsample"] == 1000
        # Check that the random state is properly set when subsampling explicitly or implicitly
        assert coreg_full.meta["inputs"]["random"]["random_state"] == 42

        # Use a larger fraction for iterative NuthKaab fitting
        fraction = 0.5 if coreg_class == coreg.NuthKaab else 0.1
        coreg_sub = coreg_class(subsample=fraction)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == fraction

        # Fit using a fraction of the unmasked data
        coreg_sub.fit(**self.fit_params, random_state=42, **fit_kwargs)
        # Do the same but specify the pixel count instead.
        # A count from the full raster is not exactly the same as a fraction of valid cells
        sample_count = int(self.tba.data.size * fraction)
        coreg_sub = coreg_class(subsample=sample_count)
        assert coreg_sub.meta["inputs"]["random"]["subsample"] == sample_count
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

    @pytest.mark.parametrize("factory", [coreg.Coreg, coreg.AffineCoreg, coreg.BiasCorr])
    @pytest.mark.parametrize(
        "reference_kind, aligned_kind, fit_method",
        [
            ("raster", "raster", "_fit_rst_rst"),
            ("raster", "points", "_fit_rst_pts"),
            ("points", "raster", "_fit_rst_pts"),
            ("points", "points", "_fit_pts_pts"),
        ],
    )
    def test_fit__spatial_dispatch(
        self, factory: Callable[..., Coreg], reference_kind: str, aligned_kind: str, fit_method: str
    ) -> None:
        """Checks that fit() calls the fitting method matching each raster and point cloud pair."""

        inputs: dict[str, Any] = {"raster": self.ref, "points": gu.PointCloud(self.points, data_name="z")}
        reference, aligned = inputs[reference_kind], inputs[aligned_kind]
        estimator = factory(meta={"subsample": 1}) if factory is Coreg else factory()

        # Check the underlying function is called once
        with patch.object(estimator, fit_method) as fitted:
            estimator.fit(reference, aligned)
        fitted.assert_called_once()
        assert fitted.call_args.kwargs["ref_elev"] is reference
        assert fitted.call_args.kwargs["tba_elev"] is aligned

    def test_fit__point_elevation_columns(self) -> None:
        """Checks that fit() uses each point cloud's selected elevation column independently."""

        frame = gpd.GeoDataFrame(
            {"reference_height": [1.0, 2.0, 3.0], "aligned_height": [3.0, 4.0, 5.0]},
            geometry=gpd.points_from_xy([0, 1, 2], [0, 1, 2]),
            crs=32632,
        )
        reference = gu.PointCloud(frame.copy(), data_name="reference_height")
        aligned = gu.PointCloud(frame.copy(), data_name="aligned_height")

        estimator = coreg.VerticalShift().fit(reference, aligned)

        assert estimator.to_matrix()[2, 3] == -2

    @pytest.mark.parametrize("operation", ["fit", "apply", "fit_and_apply"])
    def test_methods__legacy_point_column(self, operation: str) -> None:
        """Checks that fit(), apply() and fit_and_apply() use the requested elevation column."""

        reference = gpd.GeoDataFrame(
            {"height": [1.0, 2.0, 3.0]}, geometry=gpd.points_from_xy([0, 1, 2], [0, 1, 2]), crs=32632
        )
        aligned = reference.assign(height=reference.height + 2)

        if operation == "apply":
            result = coreg.AffineCoreg.from_translations(0, 0, -2).apply(aligned, z_name="height")
        else:
            estimator = coreg.VerticalShift()
            result = getattr(estimator, operation)(reference, aligned, z_name="height")
            assert estimator.to_matrix()[2, 3] == -2
        if operation != "fit":
            np.testing.assert_array_equal(result.height, reference.height)

    @pytest.mark.parametrize("kind", ["array", "raster", "points"])
    def test_fit__initial_shift(self, kind: str) -> None:
        """Checks that fit() shifts the reference coordinates and aligns an array mask before fitting."""

        # Create a raster or points
        reference: Any
        aligned: Any
        values = np.ma.masked_equal(np.arange(30.0).reshape(5, 6), 0)
        raster = Raster.from_array(values, rio.Affine(2, 0, 500000, 0, -2, 7000000), 32632, nodata=np.nan)
        options: dict[str, Any] = {}
        if kind == "array":
            reference = aligned = values
            options = dict(transform=raster.transform, crs=raster.crs, inlier_mask=np.ones(raster.shape, dtype=bool))
        elif kind == "raster":
            reference = aligned = raster.to_xarray()
        else:
            reference = aligned = gpd.GeoDataFrame(
                {"z": [10.0, 20.0]}, geometry=gpd.points_from_xy([0, 1], [2, 3]), crs=32632
            )
        estimator = coreg.NuthKaab(initial_shift=(4, -2))
        fit_method = "_fit_pts_pts" if kind == "points" else "_fit_rst_rst"

        # Inspect inputs before fitting
        with patch.object(estimator, fit_method) as fitted:
            estimator.fit(reference, aligned, **options)
        arguments = fitted.call_args.kwargs
        first, second = arguments["ref_elev"], arguments["tba_elev"]

        # Check the reference shift
        if kind == "points":
            expected = np.array([[-4, -3], [4, 5], [10, 20]])
            np.testing.assert_array_equal(first.to_xyz(), expected)
            np.testing.assert_array_equal(second.to_xyz(), [[0, 1], [2, 3], [10, 20]])
        else:
            assert first.transform == rio.Affine.translation(-4, 2) @ raster.transform
            assert second.transform == raster.transform
            np.testing.assert_array_equal(first.to_nanarray().squeeze(), raster.to_nanarray())
            if kind == "array":
                assert arguments["inlier_mask"].georeferenced_grid_equal(first)
                np.testing.assert_array_equal(arguments["inlier_mask"].data, options["inlier_mask"])

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

    @pytest.mark.parametrize("raster_output", [False, True])
    def test_apply__raster_output(self, raster_output: bool) -> None:
        """Checks that apply() formats arrays or rasters returned by _apply_rst()."""

        raster = Raster.from_array(np.arange(12.0).reshape(3, 4), rio.Affine(2, 0, 0, 0, -2, 6), 32632)
        estimator = coreg.Coreg(meta={"subsample": 1})
        with patch.object(estimator, "_fit_rst_rst"):
            estimator.fit(raster, raster)
        expected = raster.to_nanarray() + 2
        output = raster.copy(new_array=expected) if raster_output else expected

        with patch.object(estimator, "_apply_rst", return_value=(output, raster.transform)) as applied:
            actual = estimator.apply(raster)

        assert applied.call_args.kwargs["elev"] is raster
        np.testing.assert_array_equal(actual.to_nanarray(), expected)

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
            assert dem_coreg_resample.raster_equal(dem_coreg_noresample)
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
        coreg_fit_then_apply.fit(**self.fit_params, **fit_kwargs)
        aligned_then = coreg_fit_then_apply.apply(elev=self.fit_params["to_be_aligned_elev"])

        # Perform fit and apply
        aligned_and = coreg_fit_and_apply.fit_and_apply(**self.fit_params, fit_kwargs=fit_kwargs)

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
            ref_dem = ref_dem.to_pointcloud(data_name="z")
        if tba_type == "array":
            tba_dem = tba_dem.data
        elif tba_type == "pc":
            tba_dem = tba_dem.to_pointcloud(data_name="z")
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
            (
                "dem1.data",
                "dem2",
                "dem1.transform",
                "dem1.crs",
                "fit",
                "warns",
                "'dem_to_be_aligned' .* overrides .*",
            ),
            (
                "dem1.data",
                "dem2.data",
                "None",
                "dem1.crs",
                "fit",
                "error",
                "'transform' must be given if any DEM is array-like.",
            ),
            (
                "dem1.data",
                "dem2.data",
                "dem1.transform",
                "None",
                "fit",
                "error",
                "'crs' must be given if any DEM is array-like.",
            ),
            (
                "dem1",
                "dem2.data",
                "None",
                "dem1.crs",
                "apply",
                "error",
                "'transform' must be given if any DEM is array-like.",
            ),
            (
                "dem1",
                "dem2.data",
                "dem1.transform",
                "None",
                "apply",
                "error",
                "'crs' must be given if any DEM is array-like.",
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
            ("dem1 + np.nan", "dem2", "None", "None", "fit", "error", "no finite data common"),
            ("dem1", "dem2 + np.nan", "None", "None", "fit", "error", "no finite data common"),
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
            # Supply complete metadata while fitting when this case checks apply() with an array
            fit_transform = dem1.transform if testing_step == "apply" else transform
            fit_crs = dem1.crs if testing_step == "apply" else crs
            return vshiftcorr.fit(ref_dem, tba_dem, transform=fit_transform, crs=fit_crs)

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


class TestCoregClassChunked:
    """Test module for raster and array loading during fit()."""

    @pytest.mark.parametrize("as_array", [False, True])
    def test_fit__loading_laziness(self, as_array: bool) -> None:
        """Checks that fit() passes raster and array inputs to _fit_rst_rst() without loading them."""

        pytest.importorskip("dask")
        raster = Raster.from_array(np.arange(30.0).reshape(5, 6), rio.Affine(2, 0, 0, 0, -2, 12), 32632)
        first = raster.to_xarray().chunk({"y": 3, "x": 4})
        second = (raster + 2).to_xarray().chunk({"y": 2, "x": 3})
        inputs = (first.data, second.data) if as_array else (first, second)
        options = dict(transform=raster.transform, crs=raster.crs) if as_array else {}
        estimator = coreg.Coreg(meta={"subsample": 1})

        with patch.object(estimator, "_fit_rst_rst") as fitted:
            estimator.fit(*inputs, **options)

        for name in ("ref_elev", "tba_elev"):
            prepared = fitted.call_args.kwargs[name]
            assert not prepared.is_loaded
            assert prepared.transform == raster.transform
        assert not first.rst.is_loaded and not second.rst.is_loaded


class TestCoregClassErrors:
    """Test module for mixed GeoUtils and accessor inputs to coregistration."""

    @pytest.mark.parametrize(
        "operation, argument",
        [
            ("fit", "reference"),
            ("fit", "aligned"),
            ("fit", "mask"),
            ("fit", "predictor"),
            ("fit", "error_predictor"),
            ("fit_and_apply", "reference"),
            ("apply", "reference"),
            ("apply", "predictor"),
        ],
    )
    def test_methods__error_mixed_families(self, argument: str, operation: str) -> None:
        """Checks an error is raised for mixed GeoUtils and accessor inputs before Dask evaluates any arrays."""

        pytest.importorskip("dask")

        # Combine GeoUtils objects and lazy accessors
        raster = Raster.from_array(np.ones((5, 6)), rio.Affine(2, 0, 500000, 0, -2, 7000000), 32632)
        lazy = raster.to_xarray().chunk({"y": 3, "x": 4})
        first = raster if argument == "reference" else lazy
        second = raster if argument == "aligned" else lazy

        # Select the mixed input argument
        options: dict[str, Any] = {}
        if argument == "mask":
            options["inlier_mask"] = raster > 0
        if argument == "predictor":
            options["bias_vars"] = {"height": raster}
        if argument == "error_predictor":
            fit_options = {"error_predictors": {"ref": {"quality": raster}}}
            options.update({"fit_kwargs": fit_options} if operation == "fit_and_apply" else fit_options)
        if operation == "apply":
            options["bias_vars"] = {"height": lazy if argument == "reference" else raster}
        arguments = (first,) if operation == "apply" else (first, second)
        estimator = coreg.AffineCoreg.from_translations(0, 0, 1) if operation == "apply" else coreg.VerticalShift()

        # Check mixed input families
        with pytest.raises(TypeError, match="Cannot mix Raster/PointCloud"):
            getattr(estimator, operation)(*arguments, **options)

        # Check input loading
        assert not lazy.rst.is_loaded


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
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values),
            as_array=True,
            nodata_propagation="propagate",
        )
        # Exclude partial interpolation windows, where edge clamping cannot reproduce an inclined plane
        rows, cols = trans_dem.xy2ij(trans_epc.geometry.x.values, trans_epc.geometry.y.values, op=np.float64)
        interior = (cols >= 1) & (cols <= trans_dem.width - 2) & (rows >= 1) & (rows <= trans_dem.height - 2)
        valids = np.isfinite(z_points) & interior
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
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values),
            as_array=True,
            nodata_propagation="propagate",
        )
        z_points_gd = trans_dem_gd.interp_points(
            points=(trans_epc.geometry.x.values, trans_epc.geometry.y.values),
            as_array=True,
            nodata_propagation="propagate",
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

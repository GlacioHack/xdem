"""Functions to test the coregistration base classes."""

from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, Callable

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import xdem
    from xdem import coreg, examples, misc, spatialstats
    from xdem._typing import NDArrayf
    from xdem.coreg.base import Coreg, apply_matrix


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

    def test_init(self) -> None:
        """Test instantiation of Coreg"""

        c = coreg.Coreg()

        assert c._fit_called is False
        assert c._is_affine is None
        assert c._needs_vars is False

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:
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

    @pytest.mark.parametrize("subsample", [10, 10000, 0.5, 1])  # type: ignore
    def test_get_subsample_on_valid_mask(self, subsample: float | int) -> None:
        """Test the subsampling function called by all subclasses"""

        # Define a valid mask
        width = height = 50
        np.random.seed(42)
        valid_mask = np.random.randint(low=0, high=2, size=(width, height), dtype=bool)

        # Define a class with a subsample and random_state in the metadata
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

    @pytest.mark.parametrize("coreg", all_coregs)  # type: ignore
    def test_subsample(self, coreg: Callable) -> None:  # type: ignore
        warnings.simplefilter("error")

        # Check that default value is set properly
        coreg_full = coreg()
        argspec = inspect.getfullargspec(coreg)
        assert coreg_full._meta["subsample"] == argspec.defaults[argspec.args.index("subsample") - 1]  # type: ignore

        # But can be overridden during fit
        coreg_full.fit(**self.fit_params, subsample=10000, random_state=42)
        assert coreg_full._meta["subsample"] == 10000
        # Check that the random state is properly set when subsampling explicitly or implicitly
        assert coreg_full._meta["random_state"] == 42

        # Test subsampled vertical shift correction
        coreg_sub = coreg(subsample=0.1)
        assert coreg_sub._meta["subsample"] == 0.1

        # Fit the vertical shift using 10% of the unmasked data using a fraction
        coreg_sub.fit(**self.fit_params, random_state=42)
        # Do the same but specify the pixel count instead.
        # They are not perfectly equal (np.count_nonzero(self.mask) // 2 would be exact)
        # But this would just repeat the subsample code, so that makes little sense to test.
        coreg_sub = coreg(subsample=self.tba.data.size // 10)
        assert coreg_sub._meta["subsample"] == self.tba.data.size // 10
        coreg_sub.fit(**self.fit_params, random_state=42)

        # Add a few performance checks
        coreg_name = coreg.__name__
        if coreg_name == "VerticalShift":
            # Check that the estimated vertical shifts are similar
            assert abs(coreg_sub._meta["vshift"] - coreg_full._meta["vshift"]) < 0.1

        elif coreg_name == "NuthKaab":
            # Calculate the difference in the full vs. subsampled matrices
            matrix_diff = np.abs(coreg_full.to_matrix() - coreg_sub.to_matrix())
            # Check that the x/y/z differences do not exceed 30cm
            assert np.count_nonzero(matrix_diff > 0.5) == 0

        elif coreg_name == "Tilt":
            # Check that the estimated biases are similar
            assert coreg_sub._meta["coefficients"] == pytest.approx(coreg_full._meta["coefficients"], rel=1e-1)

    def test_subsample__pipeline(self) -> None:
        """Test that the subsample argument works as intended for pipelines"""

        # Check definition during instantiation
        pipe = coreg.VerticalShift(subsample=200) + coreg.Deramp(subsample=5000)

        # Check the arguments are properly defined
        assert pipe.pipeline[0]._meta["subsample"] == 200
        assert pipe.pipeline[1]._meta["subsample"] == 5000

        # Check definition during fit
        pipe = coreg.VerticalShift() + coreg.Deramp()
        pipe.fit(**self.fit_params, subsample=1000)
        assert pipe.pipeline[0]._meta["subsample"] == 1000
        assert pipe.pipeline[1]._meta["subsample"] == 1000

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
            block.fit(**self.fit_params, subsample=1000)

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
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
        verbose=True,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:

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

    def test_pipeline(self) -> None:
        warnings.simplefilter("error")

        # Create a pipeline from two coreg methods.
        pipeline = coreg.CoregPipeline([coreg.VerticalShift(), coreg.NuthKaab()])
        pipeline.fit(**self.fit_params)

        aligned_dem, _ = pipeline.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        assert aligned_dem.shape == self.ref.data.squeeze().shape

        # Make a new pipeline with two vertical shift correction approaches.
        pipeline2 = coreg.CoregPipeline([coreg.VerticalShift(), coreg.VerticalShift()])
        # Set both "estimated" vertical shifts to be 1
        pipeline2.pipeline[0]._meta["vshift"] = 1
        pipeline2.pipeline[1]._meta["vshift"] = 1

        # Assert that the combined vertical shift is 2
        assert pipeline2.to_matrix()[2, 3] == 2.0

    all_coregs = [
        coreg.VerticalShift(),
        coreg.NuthKaab(),
        coreg.ICP(),
        coreg.Deramp(),
        coreg.TerrainBias(),
        coreg.DirectionalBias(),
    ]

    @pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
    @pytest.mark.parametrize("coreg2", all_coregs)  # type: ignore
    def test_pipeline_combinations__nobiasvar(self, coreg1: Coreg, coreg2: Coreg) -> None:
        """Test pipelines with all combinations of coregistration subclasses (without bias variables)"""

        # Create a pipeline from one affine and one biascorr methods.
        pipeline = coreg.CoregPipeline([coreg1, coreg2])
        pipeline.fit(**self.fit_params)

        aligned_dem, _ = pipeline.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)
        assert aligned_dem.shape == self.ref.data.squeeze().shape

    @pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
    @pytest.mark.parametrize(
        "coreg2",
        [
            coreg.BiasCorr1D(bias_var_names=["slope"], fit_or_bin="bin"),
            coreg.BiasCorr2D(bias_var_names=["slope", "aspect"], fit_or_bin="bin"),
        ],
    )  # type: ignore
    def test_pipeline_combinations__biasvar(self, coreg1: Coreg, coreg2: Coreg) -> None:
        """Test pipelines with all combinations of coregistration subclasses with bias variables"""

        # Create a pipeline from one affine and one biascorr methods.
        pipeline = coreg.CoregPipeline([coreg1, coreg2])
        bias_vars = {"slope": xdem.terrain.slope(self.ref), "aspect": xdem.terrain.aspect(self.ref)}
        pipeline.fit(**self.fit_params, bias_vars=bias_vars)

        aligned_dem, _ = pipeline.apply(
            self.tba.data, transform=self.ref.transform, crs=self.ref.crs, bias_vars=bias_vars
        )
        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline__errors(self) -> None:
        """Test pipeline raises proper errors."""

        pipeline = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr1D()])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No `bias_vars` passed to .fit() for bias correction step "
                "<class 'xdem.coreg.biascorr.BiasCorr1D'> of the pipeline."
            ),
        ):
            pipeline.fit(**self.fit_params)

        pipeline2 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr1D(), coreg.BiasCorr1D()])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No `bias_vars` passed to .fit() for bias correction step <class 'xdem.coreg.biascorr.BiasCorr1D'> "
                "of the pipeline. As you are using several bias correction steps requiring"
                " `bias_vars`, don't forget to explicitly define their `bias_var_names` "
                "during instantiation, e.g. BiasCorr1D(bias_var_names=['slope'])."
            ),
        ):
            pipeline2.fit(**self.fit_params)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "When using several bias correction steps requiring `bias_vars` in a pipeline,"
                "the `bias_var_names` need to be explicitly defined at each step's "
                "instantiation, e.g. BiasCorr1D(bias_var_names=['slope'])."
            ),
        ):
            pipeline2.fit(**self.fit_params, bias_vars={"slope": xdem.terrain.slope(self.ref)})

        pipeline3 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr1D(bias_var_names=["slope"])])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step <class 'xdem.coreg.biascorr.BiasCorr1D'>: ['slope']."
            ),
        ):
            pipeline3.fit(**self.fit_params, bias_vars={"ncc": xdem.terrain.slope(self.ref)})

    def test_pipeline_pts(self) -> None:
        warnings.simplefilter("ignore")

        pipeline = coreg.NuthKaab() + coreg.GradientDescending()
        ref_points = self.ref.to_points(as_array=False, subset=5000, pixel_offset="center").ds
        ref_points["E"] = ref_points.geometry.x
        ref_points["N"] = ref_points.geometry.y
        ref_points.rename(columns={"b1": "z"}, inplace=True)

        # Check that this runs without error
        pipeline.fit_pts(reference_dem=ref_points, dem_to_be_aligned=self.tba)

        for part in pipeline.pipeline:
            assert np.abs(part._meta["offset_east_px"]) > 0

        assert pipeline.pipeline[0]._meta["offset_east_px"] != pipeline.pipeline[1]._meta["offset_east_px"]

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

    def test_pipeline_consistency(self):
        """Check that pipelines properties are respected: reflectivity, fusion of same coreg"""

        # Test 1: Fusion of same coreg
        # Many vertical shifts
        many_vshifts = coreg.VerticalShift() + coreg.VerticalShift() + coreg.VerticalShift()
        many_vshifts.fit(**self.fit_params)
        aligned_dem, _ = many_vshifts.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        # The last steps should have shifts of EXACTLY zero
        assert many_vshifts.pipeline[1]._meta["vshift"] == pytest.approx(0, abs=10e-5)
        assert many_vshifts.pipeline[2]._meta["vshift"] == pytest.approx(0, abs=10e-5)

        # Many horizontal + vertical shifts
        many_nks = coreg.NuthKaab() + coreg.NuthKaab() + coreg.NuthKaab()
        many_nks.fit(**self.fit_params)
        aligned_dem, _ = many_nks.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        # The last steps should have shifts of NEARLY zero
        assert many_nks.pipeline[1]._meta["vshift"] == pytest.approx(0, abs=0.01)
        assert many_nks.pipeline[1]._meta["offset_east_px"] == pytest.approx(0, abs=0.01)
        assert many_nks.pipeline[1]._meta["offset_north_px"] == pytest.approx(0, abs=0.01)
        assert many_nks.pipeline[2]._meta["vshift"] == pytest.approx(0, abs=0.01)
        assert many_nks.pipeline[2]._meta["offset_east_px"] == pytest.approx(0, abs=0.01)
        assert many_nks.pipeline[2]._meta["offset_north_px"] == pytest.approx(0, abs=0.01)

        # Test 2: Reflectivity
        # Those two pipelines should give almost the same result
        nk_vshift = coreg.NuthKaab() + coreg.VerticalShift()
        vshift_nk = coreg.VerticalShift() + coreg.NuthKaab()

        nk_vshift.fit(**self.fit_params)
        aligned_dem, _ = nk_vshift.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)
        vshift_nk.fit(**self.fit_params)
        aligned_dem, _ = vshift_nk.apply(self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

        assert np.allclose(nk_vshift.to_matrix(), vshift_nk.to_matrix(), atol=10e-1)

class TestBlockwiseCoreg:
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

    @pytest.mark.parametrize(
        "pipeline", [coreg.VerticalShift(), coreg.VerticalShift() + coreg.NuthKaab()]
    )  # type: ignore
    @pytest.mark.parametrize("subdivision", [4, 10])  # type: ignore
    def test_blockwise_coreg(self, pipeline: Coreg, subdivision: int) -> None:
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

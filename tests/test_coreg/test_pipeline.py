"""Functions to test the coregistration base classes."""

from __future__ import annotations

import re
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geoutils import Raster, Vector
from geoutils.raster import RasterType

import xdem
from xdem import coreg, examples
from xdem.coreg.base import Coreg


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path_test("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

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


class TestCoregPipeline:

    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_elev=ref,
        to_be_aligned_elev=tba,
        inlier_mask=inlier_mask,
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
        ref_points = self.ref.to_pointcloud(subsample=5000, random_state=42)

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

        # The last steps should have shifts of EXACTLY zero
        assert many_vshifts.pipeline[1].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=10e-5)
        assert many_vshifts.pipeline[2].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=10e-5)

        # Many horizontal + vertical shifts
        many_nks = coreg.LZD() + coreg.LZD() + coreg.LZD()
        many_nks.fit(**self.fit_params, random_state=42)

        # The last steps should have shifts of NEARLY zero, like 0.1 pixel
        abs_trans = 0.1 * self.ref.res[0]
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=abs_trans)
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_x"] == pytest.approx(0, abs=abs_trans)
        assert many_nks.pipeline[1].meta["outputs"]["affine"]["shift_y"] == pytest.approx(0, abs=abs_trans)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_z"] == pytest.approx(0, abs=abs_trans)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_x"] == pytest.approx(0, abs=abs_trans)
        assert many_nks.pipeline[2].meta["outputs"]["affine"]["shift_y"] == pytest.approx(0, abs=abs_trans)

        # Test 2: Reflectivity
        # Those two pipelines should give almost the same result
        nk_vshift = coreg.NuthKaab() + coreg.VerticalShift()
        vshift_nk = coreg.VerticalShift() + coreg.NuthKaab()

        nk_vshift.fit(**self.fit_params, random_state=42)
        vshift_nk.fit(**self.fit_params, random_state=42)

        # TODO: See after merge of #890
        nk_vshift_tr = coreg.translations_rotations_from_matrix(nk_vshift.to_matrix())
        vshift_nk_tr = coreg.translations_rotations_from_matrix(vshift_nk.to_matrix())
        assert np.allclose(nk_vshift_tr, vshift_nk_tr)

    def test_subsample_pipeline(self) -> None:
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

    def test_subsample_pipeline__exceptions(self) -> None:
        """Test that the subsample exceptions work as intended for pipelines"""

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

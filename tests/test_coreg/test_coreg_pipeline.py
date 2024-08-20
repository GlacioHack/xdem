import re
from typing import Callable

import numpy as np
import pytest

import xdem
from tests.test_coreg.conftest import all_coregs
from xdem import coreg
from xdem.coreg.base import Coreg


@pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
def test_copy(coreg_class: Callable[[], Coreg]) -> None:

    # Create a pipeline, add some .metadata, and copy it
    pipeline = coreg_class() + coreg_class()
    pipeline.pipeline[0]._meta["shift_z"] = 1

    pipeline_copy = pipeline.copy()

    # Add some more .metadata after copying (this should not be transferred)
    pipeline_copy.pipeline[0]._meta["shift_y"] = 0.5 * 30

    assert pipeline.pipeline[0].meta != pipeline_copy.pipeline[0].meta
    assert pipeline_copy.pipeline[0]._meta["shift_z"]


def test_pipeline(load_examples) -> None:

    ref, tba, _, _, fit_params = load_examples

    # Create a pipeline from two coreg methods.
    pipeline = coreg.CoregPipeline([coreg.VerticalShift(), coreg.NuthKaab()])
    pipeline.fit(**fit_params, subsample=5000, random_state=42)

    aligned_dem, _ = pipeline.apply(tba.data, transform=ref.transform, crs=ref.crs)

    assert aligned_dem.shape == ref.data.squeeze().shape

    # Make a new pipeline with two vertical shift correction approaches.
    pipeline2 = coreg.CoregPipeline([coreg.VerticalShift(), coreg.VerticalShift()])
    # Set both "estimated" vertical shifts to be 1
    pipeline2.pipeline[0].meta["shift_z"] = 1
    pipeline2.pipeline[1].meta["shift_z"] = 1

    # Assert that the combined vertical shift is 2
    assert pipeline2.to_matrix()[2, 3] == 2.0


@pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
@pytest.mark.parametrize("coreg2", all_coregs)  # type: ignore
def test_pipeline_combinations__nobiasvar(
    coreg1: Callable[[], Coreg], coreg2: Callable[[], Coreg], load_examples
) -> None:
    """Test pipelines with all combinations of coregistration subclasses (without bias variables)"""

    ref, tba, _, _, fit_params = load_examples

    # Create a pipeline from one affine and one biascorr methods.
    pipeline = coreg.CoregPipeline([coreg1(), coreg2()])
    pipeline.fit(**fit_params, subsample=5000, random_state=42)

    aligned_dem, _ = pipeline.apply(tba.data, transform=ref.transform, crs=ref.crs)
    assert aligned_dem.shape == ref.data.squeeze().shape


@pytest.mark.parametrize("coreg1", all_coregs)  # type: ignore
@pytest.mark.parametrize(
    "coreg2_init_kwargs",
    [
        dict(bias_var_names=["slope"], fit_or_bin="bin"),
        dict(bias_var_names=["slope", "aspect"], fit_or_bin="bin"),
    ],
)  # type: ignore
def test_pipeline_combinations__biasvar(
    coreg1: Callable[[], Coreg], coreg2_init_kwargs: dict[str, str], load_examples
) -> None:
    """Test pipelines with all combinations of coregistration subclasses with bias variables"""

    ref, tba, _, _, fit_params = load_examples

    # Create a pipeline from one affine and one biascorr methods
    pipeline = coreg.CoregPipeline([coreg1(), coreg.BiasCorr(**coreg2_init_kwargs)])
    print(pipeline.pipeline[0].meta["subsample"])
    print(pipeline.pipeline[1].meta["subsample"])
    bias_vars = {"slope": xdem.terrain.slope(ref), "aspect": xdem.terrain.aspect(ref)}
    pipeline.fit(**fit_params, bias_vars=bias_vars, subsample=5000, random_state=42)

    aligned_dem, _ = pipeline.apply(tba.data, transform=ref.transform, crs=ref.crs, bias_vars=bias_vars)
    assert aligned_dem.shape == ref.data.squeeze().shape


def test_pipeline__errors(load_examples) -> None:
    """Test pipeline raises proper errors."""

    ref, _, _, _, fit_params = load_examples

    pipeline = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr()])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No `bias_vars` passed to .fit() for bias correction step "
            "<class 'xdem.coreg.biascorr.BiasCorr'> of the pipeline."
        ),
    ):
        pipeline.fit(**fit_params)

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
        pipeline2.fit(**fit_params)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "When using several bias correction steps requiring `bias_vars` in a pipeline,"
            "the `bias_var_names` need to be explicitly defined at each step's "
            "instantiation, e.g. BiasCorr(bias_var_names=['slope'])."
        ),
    ):
        pipeline2.fit(**fit_params, bias_vars={"slope": xdem.terrain.slope(ref)})

    pipeline3 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr(bias_var_names=["slope"])])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
            "instantiation of the bias correction step <class 'xdem.coreg.biascorr.BiasCorr'>: ['slope']."
        ),
    ):
        pipeline3.fit(**fit_params, bias_vars={"ncc": xdem.terrain.slope(ref)})


def test_pipeline_pts(load_examples) -> None:
    ref, tba, _, _, _ = load_examples

    pipeline = coreg.NuthKaab() + coreg.GradientDescending()
    ref_points = ref.to_pointcloud(subsample=5000, random_state=42).ds
    ref_points["E"] = ref_points.geometry.x
    ref_points["N"] = ref_points.geometry.y
    ref_points.rename(columns={"b1": "z"}, inplace=True)

    # Check that this runs without error
    pipeline.fit(reference_elev=ref_points, to_be_aligned_elev=tba)

    for part in pipeline.pipeline:
        assert np.abs(part.meta["shift_x"]) > 0

    assert pipeline.pipeline[0].meta["shift_x"] != pipeline.pipeline[1].meta["shift_x"]


def test_coreg_add() -> None:

    # Test with a vertical shift of 4
    vshift = 4

    vshift1 = coreg.VerticalShift()
    vshift2 = coreg.VerticalShift()

    # Set the vertical shift attribute
    for vshift_corr in (vshift1, vshift2):
        vshift_corr.meta["shift_z"] = vshift

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


def test_pipeline_consistency(load_examples) -> None:
    """Check that pipelines properties are respected: reflectivity, fusion of same coreg"""

    ref, tba, _, _, fit_params = load_examples

    # Test 1: Fusion of same coreg
    # Many vertical shifts
    many_vshifts = coreg.VerticalShift() + coreg.VerticalShift() + coreg.VerticalShift()
    many_vshifts.fit(**fit_params, random_state=42)
    aligned_dem, _ = many_vshifts.apply(tba.data, transform=ref.transform, crs=ref.crs)

    # The last steps should have shifts of EXACTLY zero
    assert many_vshifts.pipeline[1].meta["shift_z"] == pytest.approx(0, abs=10e-5)
    assert many_vshifts.pipeline[2].meta["shift_z"] == pytest.approx(0, abs=10e-5)

    # Many horizontal + vertical shifts
    many_nks = coreg.NuthKaab() + coreg.NuthKaab() + coreg.NuthKaab()
    many_nks.fit(**fit_params, random_state=42)
    aligned_dem, _ = many_nks.apply(tba.data, transform=ref.transform, crs=ref.crs)

    # The last steps should have shifts of NEARLY zero
    assert many_nks.pipeline[1].meta["shift_z"] == pytest.approx(0, abs=0.02)
    assert many_nks.pipeline[1].meta["shift_x"] == pytest.approx(0, abs=0.02)
    assert many_nks.pipeline[1].meta["shift_y"] == pytest.approx(0, abs=0.02)
    assert many_nks.pipeline[2].meta["shift_z"] == pytest.approx(0, abs=0.02)
    assert many_nks.pipeline[2].meta["shift_x"] == pytest.approx(0, abs=0.02)
    assert many_nks.pipeline[2].meta["shift_y"] == pytest.approx(0, abs=0.02)

    # Test 2: Reflectivity
    # Those two pipelines should give almost the same result
    nk_vshift = coreg.NuthKaab() + coreg.VerticalShift()
    vshift_nk = coreg.VerticalShift() + coreg.NuthKaab()

    nk_vshift.fit(**fit_params, random_state=42)
    aligned_dem, _ = nk_vshift.apply(tba.data, transform=ref.transform, crs=ref.crs)
    vshift_nk.fit(**fit_params, random_state=42)
    aligned_dem, _ = vshift_nk.apply(tba.data, transform=ref.transform, crs=ref.crs)

    assert np.allclose(nk_vshift.to_matrix(), vshift_nk.to_matrix(), atol=10e-1)

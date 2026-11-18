"""Tests for sequential coregistration pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from geoutils import Raster, Vector
from geoutils.multiproc import MultiprocConfig
from geoutils.raster import RasterType

import xdem
from tests.test_coreg.test_base import assert_coreg_meta_equal
from xdem import DEM, coreg, examples, open_dem
from xdem.coreg.base import Coreg


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path_test("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


class _CoregPipelineInputs:
    """Provide shared example elevations, masks and points for pipeline tests."""

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


class TestCoregPipeline(_CoregPipelineInputs):
    """Test module for construction, fitting, application and composition of in-memory pipelines."""

    @pytest.mark.parametrize("coreg_class", [coreg.VerticalShift, coreg.ICP, coreg.NuthKaab])  # type: ignore
    def test_copy(self, coreg_class: Callable[[], Coreg]) -> None:
        """Checks that copying a pipeline keeps subsequent metadata changes independent."""

        # Create a pipeline, add some .metadata, and copy it
        pipeline = coreg_class() + coreg_class()
        pipeline.pipeline[0]._meta["outputs"]["affine"] = {"shift_z": 1}

        pipeline_copy = pipeline.copy()

        # Add some more .metadata after copying (this should not be transferred)
        pipeline_copy.pipeline[0]._meta["outputs"]["affine"].update({"shift_y": 0.5 * 30})

        assert pipeline.pipeline[0].meta != pipeline_copy.pipeline[0].meta
        assert pipeline_copy.pipeline[0]._meta["outputs"]["affine"]["shift_z"]

    def test_pipeline(self) -> None:
        """Checks that a fitted pipeline preserves the grid shape and combines affine shifts."""

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
        """Checks that combinations without bias variables fit and preserve the raster shape."""

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
        """Checks that combinations using explicit bias variables fit and preserve the raster shape."""

        # Create a pipeline from one affine and one biascorr methods
        pipeline = coreg.CoregPipeline([coreg1(), coreg.BiasCorr(**coreg2_init_kwargs)])  # type: ignore
        bias_vars = {"slope": xdem.terrain.slope(self.ref), "aspect": xdem.terrain.aspect(self.ref)}
        pipeline.fit(**self.fit_params, bias_vars=bias_vars, subsample=5000, random_state=42)

        aligned_dem, _ = pipeline.apply(
            self.tba.data, transform=self.ref.transform, crs=self.ref.crs, bias_vars=bias_vars
        )
        assert aligned_dem.shape == self.ref.data.squeeze().shape

    def test_pipeline_pts(self) -> None:
        """Checks that a pipeline fits a point reference and passes its aligned raster between steps."""

        pipeline = coreg.NuthKaab() + coreg.DhMinimize()
        ref_points = self.ref.to_pointcloud(subsample=3000, random_state=42)

        # Check that this runs without error
        pipeline.fit(reference_elev=ref_points, to_be_aligned_elev=self.tba)

        for part in pipeline.pipeline:
            assert np.abs(part.meta["outputs"]["affine"]["shift_x"]) > 0

        assert (
            pipeline.pipeline[0].meta["outputs"]["affine"]["shift_x"]
            != pipeline.pipeline[1].meta["outputs"]["affine"]["shift_x"]
        )

    def test_coreg_add(self) -> None:
        """Checks that adding coregistration steps and pipelines combines their transformations in order."""

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

        # Try to add a Coreg step to an already existing CoregPipeline
        vshift4 = vshift3 + vshift1
        assert vshift4.to_matrix()[2, 3] == vshift * 3

        # Try to add two CoregPipelines
        vshift5 = vshift3 + vshift3
        assert vshift5.to_matrix()[2, 3] == vshift * 4

    def test_pipeline_consistency(self) -> None:
        """Checks that repeated steps converge and equivalent step orders give similar transformations."""

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
        nk_vshift = coreg.NuthKaab(vertical_shift=False) + coreg.VerticalShift()
        vshift_nk = coreg.VerticalShift() + coreg.NuthKaab(vertical_shift=False)

        nk_vshift.fit(**self.fit_params, random_state=42)
        vshift_nk.fit(**self.fit_params, random_state=42)

        # TODO: See after merge of #890
        nk_vshift_tr = coreg.translations_rotations_from_matrix(nk_vshift.to_matrix())
        vshift_nk_tr = coreg.translations_rotations_from_matrix(vshift_nk.to_matrix())
        assert np.allclose(nk_vshift_tr[:3], vshift_nk_tr[:3], atol=0.2 * self.ref.res[0])
        assert np.allclose(nk_vshift_tr[3:], vshift_nk_tr[3:], atol=10e-6)

    def test_subsample_pipeline(self) -> None:
        """Checks that pipeline and step subsample values are stored and propagated during fitting."""

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

    def test_fit_and_apply__pipeline(self) -> None:
        """Checks that fit_and_apply() matches separate fit() and apply() calls for a pipeline."""

        # Initiate two similar coregs
        coreg_fit_then_apply = coreg.NuthKaab() + coreg.Deramp()
        coreg_fit_and_apply = coreg.NuthKaab() + coreg.Deramp()

        # Perform fit, then apply
        coreg_fit_then_apply.fit(**self.fit_params, subsample=10000, random_state=42)
        aligned_then = coreg_fit_then_apply.apply(elev=self.fit_params["to_be_aligned_elev"])

        # Perform fit and apply
        aligned_and = coreg_fit_and_apply.fit_and_apply(**self.fit_params, subsample=10000, random_state=42)

        assert aligned_and.raster_equal(aligned_then, warn_failure_reason=True)
        for combined, separate in zip(coreg_fit_and_apply.pipeline, coreg_fit_then_apply.pipeline):
            assert list(combined.meta) == list(separate.meta)
            assert assert_coreg_meta_equal(combined.meta, separate.meta)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("shape", [(5, 6), (1, 5, 6)])
    def test_apply__array_representation(self, masked: bool, shape: tuple[int, ...]) -> None:
        """Checks that pipeline application preserves array shape, masks and updated grid metadata."""

        # Create elevations and a fitted pipeline
        values = np.arange(30.0).reshape(shape)
        values.reshape(-1)[8] = np.nan
        if masked:
            values = np.ma.masked_invalid(values)
        raster = Raster.from_array(values, Affine(2, 0, 500000, 0, -2, 7000000), 32632, nodata=np.nan)
        pipeline = (coreg.VerticalShift() + coreg.VerticalShift()).fit(raster, raster + 2)
        expected = pipeline.apply(raster, resample=False)

        # Apply the pipeline to the array
        actual, transform = pipeline.apply(values, transform=raster.transform, crs=raster.crs, resample=False)

        # Check values, masks and grid metadata
        assert actual.shape == shape and transform == expected.transform
        assert np.ma.isMaskedArray(actual) == masked
        np.testing.assert_array_equal(np.ma.filled(actual, np.nan), expected.to_nanarray().reshape(shape))
        np.testing.assert_array_equal(np.ma.filled(values, np.nan), raster.to_nanarray().reshape(shape))

    def test_apply__dataframe_elevation_column(self) -> None:
        """Checks that a pipeline applies a named elevation column while preserving other point attributes."""

        # Create a fitted pipeline and points
        raster = Raster.from_array(np.ones((5, 6)), Affine(2, 0, 500000, 0, -2, 7000000), 32632)
        pipeline = (coreg.VerticalShift() + coreg.VerticalShift()).fit(raster, raster + 2)
        frame = gpd.GeoDataFrame(
            {"height": [10.0, 20.0], "quality": [1, 2]},
            geometry=gpd.points_from_xy([0, 1], [2, 3]),
            index=[4, 7],
            crs=32632,
        )

        # Apply the named elevation column
        actual = pipeline.apply(frame, z_name="height")

        # Check corrected heights and unchanged attributes
        np.testing.assert_array_equal(actual.height, frame.height - 2)
        np.testing.assert_array_equal(actual.quality, frame.quality)
        np.testing.assert_array_equal(actual.index, frame.index)
        np.testing.assert_array_equal(actual.geometry, frame.geometry)


@pytest.fixture
def pipeline_dem_paths(tmp_path: Path, bias_correction: bool) -> tuple[Path, ...]:
    """Write curved DEMs with a ramp or horizontal shift for pipeline comparisons."""

    # Create terrain with a known shift
    rows, columns = np.indices((20, 25))
    values = (100 + 5 * np.sin(columns / 4) + 3 * np.cos(rows / 5)).astype("float32")
    reference: Any = Raster.from_array(values, Affine(2, 0, 500000, 0, -2, 7000000), 32632, nodata=np.nan)
    source: Any
    if bias_correction:
        source = reference + 0.5 + 0.03 * rows + 0.02 * columns
    else:
        source = reference.translate(0.3, -0.2) + 0.5

    paths = tuple(tmp_path / f"input-{index}.tif" for index in range(2))
    for raster, path in zip((reference, source), paths):
        raster.to_file(path)
    return paths


class TestCoregPipelineChunked:
    """Test module for pipeline fit_and_apply() with Dask or multiprocessing."""

    @pytest.mark.parametrize(
        "backend, bias_correction",
        [("dask", True), ("mp", False), ("dask_array", False)],
    )
    def test_fit_and_apply__chunk_invariance(
        self, backend: str, bias_correction: bool, pipeline_dem_paths: tuple[Path, ...], tmp_path: Path
    ) -> None:
        """Checks that chunked pipelines match in-memory steps, values and transforms without loading inputs."""

        from geoutils.multiproc import MpCluster

        # 1/ Open inputs and select the pipeline
        reference: Any
        source: Any
        reference, source = (DEM(path, load_data=True) for path in pipeline_dem_paths)
        estimator = (
            coreg.VerticalShift(subsample=80) + coreg.Deramp(poly_order=1, subsample=80)
            if bias_correction
            else coreg.VerticalShift(subsample=100) + coreg.ICP(subsample=100) + coreg.VerticalShift(subsample=100)
        )
        options: dict[str, Any] = dict(random_state=42)
        # 2/ Fit and apply the chunked pipeline
        if backend in ("dask", "dask_array"):
            # 2.1/ If using Dask
            pytest.importorskip("dask")
            if backend == "dask_array":
                import dask.array as da

                estimator = coreg.VerticalShift(subsample=20) + coreg.VerticalShift(subsample=20)
                values = reference.to_nanarray()
                reference, source = values, values + 0.5
                inputs = [da.from_array(reference, chunks=(7, 9)), da.from_array(source, chunks=(6, 8))]
                options.update(transform=Affine(2, 0, 500000, 0, -2, 7000000), crs=32632)
            else:
                # Use different layouts and shorter final tiles
                inputs = [
                    open_dem(path, chunks={"y": rows, "x": columns}).dem
                    for path, (rows, columns) in zip(pipeline_dem_paths, ((7, 9), (6, 8)))
                ]
                assert all(not value.is_loaded for value in inputs)
            expected = estimator.copy()
            in_memory = expected.fit_and_apply(reference, source, **options)
            actual = estimator.copy()
            output = actual.fit_and_apply(inputs[0], inputs[1], **options)
            if backend == "dask_array":
                values, transform = output
                assert isinstance(values, da.Array) and transform == in_memory[1]
                computed = values.compute()
                expected_values = in_memory[0]
                assert all(isinstance(value, da.Array) for value in inputs)
            else:
                assert not output.dem.is_loaded
                computed = output.compute().data.squeeze()
                expected_values = in_memory.to_nanarray()
                assert output.dem.transform == in_memory.transform and output.dem.crs == in_memory.crs
                assert not output.dem.is_loaded and all(not value.is_loaded for value in inputs)
        else:
            # 2.2/ If using MP
            inputs = [DEM(path) for path in pipeline_dem_paths]
            config = MultiprocConfig(chunks=(7, 9), outfile=str(tmp_path / "output.tif"))
            assert all(not value.is_loaded for value in inputs)
            expected = estimator.copy()
            in_memory = expected.fit_and_apply(reference, source, **options)
            actual = estimator.copy()
            with MpCluster({"nb_workers": 2, "max_tasks_per_child": None}) as workers:
                config.cluster = workers
                output = actual.fit_and_apply(inputs[0], inputs[1], mp_config=config, **options)
            assert not output.is_loaded
            computed = output.to_nanarray()
            expected_values = in_memory.to_nanarray()
            assert output.transform == in_memory.transform and output.crs == in_memory.crs
            assert all(not value.is_loaded for value in inputs)

        # 3/ Check equality of step metadata and corrected values
        for actual_step, expected_step in zip(actual.pipeline, expected.pipeline):
            assert assert_coreg_meta_equal(actual_step.meta["outputs"], expected_step.meta["outputs"])
        np.testing.assert_array_equal(computed, expected_values)


class TestCoregPipelineErrors(_CoregPipelineInputs):
    """Test module for invalid bias variables, additions and conflicting pipeline sampling options."""

    def test_fit__error_invalid_bias_variables(self) -> None:
        """Checks that each bias correction step requires the bias variables declared for that step."""

        # Reject an omitted bias variable for a single correction step
        pipeline = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr()])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "No `bias_vars` passed to .fit() for bias correction step "
                "<class 'xdem.coreg.biascorr.BiasCorr'> of the pipeline."
            ),
        ):
            pipeline.fit(**self.fit_params)

        # Require explicit variable names when several correction steps need bias variables
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

        # Reject variables whose names do not match the step declaration
        pipeline3 = coreg.CoregPipeline([coreg.NuthKaab(), coreg.BiasCorr(bias_var_names=["slope"])])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step <class 'xdem.coreg.biascorr.BiasCorr'>: ['slope']."
            ),
        ):
            pipeline3.fit(**self.fit_params, bias_vars={"ncc": xdem.terrain.slope(self.ref)})

    def test_add__error_invalid_type(self) -> None:
        """Checks that adding an object other than a coregistration step or pipeline raises an error."""

        with pytest.raises(ValueError, match="Incompatible add type"):
            coreg.VerticalShift() + 1  # type: ignore

    def test_fit__warning_subsample_override(self) -> None:
        """Checks that a fit subsample warns when it overrides a non-default step subsample."""

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

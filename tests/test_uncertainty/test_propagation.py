"""Tests for public elevation uncertainty estimation and propagation workflows."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from geoutils.multiproc import MultiprocConfig

import xdem


class TestElevationErrorEstimation:
    """Test module for error estimation methods on DEMs and elevation point clouds."""

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("entrypoint", ["function", "method", "accessor"])
    @pytest.mark.parametrize("other_precision", [None, "negligible", "same"])
    def test_estimate_error_structure__array_comparison(
        self, kind: str, entrypoint: str, other_precision: str | None
    ) -> None:
        """Checks that masked array comparisons give the expected magnitude for each precision assumption."""

        # 1/ Prepare inputs
        # Create errors and exclude the last row
        values = np.arange(36, dtype=float).reshape(6, 6)
        mask = np.indices(values.shape)[0] < 5
        if kind == "raster":
            source = xdem.DEM.from_array(values, Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999)
        else:
            x, y = np.meshgrid(np.arange(6), np.arange(6))
            values, mask = values.ravel(), mask.ravel()
            source = xdem.EPC(gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32632))

        # Set comparison and precision options
        options: dict[str, Any] = {
            "other": np.zeros_like(values),
            "predictors": {},
            "mask": mask,
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }
        if other_precision is not None:
            options["other_precision"] = other_precision

        # 2/ Estimate and compare magnitudes
        # Run the function, method or accessor
        if entrypoint == "function":
            result = xdem.uncertainty.estimate_error_structure(source, **options)
        elif entrypoint == "method":
            result = source.estimate_error_structure(**options)
        else:
            accessor = source.to_xarray().dem if kind == "raster" else source.gdf.epc
            result = accessor.estimate_error_structure(**options)

        # Equal independent errors increase difference variance by a factor of two
        scale = np.sqrt(2) if other_precision == "same" else 1.0
        assert type(result) is gu.ErrorStructure
        assert result.predict_magnitude() == pytest.approx(np.std(values[mask]) / scale)
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(mask)

    @pytest.mark.parametrize("other_precision", ["negligible", "same"])
    @pytest.mark.parametrize(
        "source_kind, reference_kind", [("raster", "raster"), ("raster", "point"), ("point", "raster")]
    )
    def test_estimate_error_structure__terrain_predictors(
        self, source_kind: str, reference_kind: str, other_precision: Literal["negligible", "same"]
    ) -> None:
        """Checks that terrain predictor names give the same estimates as explicit GeoUtils predictors."""

        # 1/ Prepare raster or point inputs
        # Create curved terrain with measurement errors
        rows, columns = np.indices((12, 12))
        elevations = rows + columns + 0.1 * rows * columns
        errors = np.random.default_rng(7).normal(size=elevations.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        source_raster = xdem.DEM.from_array(elevations + errors, transform, 32632, nodata=-9999)
        reference_raster = xdem.DEM.from_array(elevations, transform, 32632, nodata=-9999)

        # Sample sparse points to distinguish point and raster comparisons
        source = source_raster if source_kind == "raster" else source_raster.to_pointcloud(subsample=30, random_state=2)
        reference = (
            reference_raster
            if reference_kind == "raster"
            else reference_raster.to_pointcloud(subsample=30, random_state=2)
        )
        terrain_source = source_raster if source_kind == "raster" else reference_raster
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 3,
            "min_count": 2,
            "spread_estimator": np.std,
            "random_state": 4,
        }

        # 2/ Estimate and compare magnitudes
        # Run GeoUtils with slope values and xDEM with the predictor name
        expected = gu.ErrorStructure.estimate(
            source,
            other=reference,
            other_precision=other_precision,
            at=None,
            align="reproject",
            predictors={"slope": terrain_source.slope()},
            **options,
        )
        actual = xdem.uncertainty.estimate_error_structure(
            source, reference, predictors=("slope",), other_precision=other_precision, **options
        )

        # Check grouped statistics
        assert type(actual) is gu.ErrorStructure
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(actual_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)

    def test_estimate_uncertainty__deprecated(self) -> None:
        """Checks that the deprecated DEM method warns and returns magnitude and correlation."""

        pytest.importorskip("skgstat")

        # Load example DEMs
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))

        # Check the deprecation warning links to the migration guide
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
                magnitude, correlation = source.estimate_uncertainty(reference, random_state=42)

        # Check the magnitude grid and correlation at zero distance
        assert isinstance(magnitude, gu.Raster)
        assert magnitude.georeferenced_grid_equal(source)
        assert correlation(0) == pytest.approx(1)

    @pytest.mark.parametrize("at", ["self", "other", None])
    def test_estimate_error_structure__locations(self, at: Literal["self", "other"] | None) -> None:
        """Checks that masked comparisons match GeoUtils on the selected grid."""

        # Create different grids and mask the first four columns
        values = np.arange(36, dtype=float).reshape(6, 6)
        source = xdem.DEM.from_array(values, Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999)
        reference = xdem.DEM.from_array(np.zeros((3, 3)), Affine(20, 0, 0, 0, -20, 60), 32632, nodata=-9999)
        mask = np.indices(values.shape)[1] < 4
        options: dict[str, Any] = {
            "predictors": {},
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
            "other_precision": "negligible",
            "at": at,
            "align": "reproject",
        }

        # Compare a raster mask in GeoUtils with an array mask in xDEM
        expected = gu.ErrorStructure.estimate(source, other=reference, mask=source.copy(new_array=mask), **options)
        actual = source.estimate_error_structure(other=reference, mask=mask, **options)

        # Check magnitude and selected observations
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        assert actual.fit_diagnostics == expected.fit_diagnostics

    def test_estimate_error_structure__point_attributes(self) -> None:
        """Checks that point value and predictor columns give the same estimates as GeoUtils."""

        # Create points with height, Z and quality columns
        values = np.array([-2.0, -1.0, 1.0, 2.0, -4.0, -2.0, 2.0, 4.0])
        points = gu.PointCloud.from_xyz(np.arange(8), np.zeros(8), values, crs=32632, data_name="height")
        frame = points.gdf.copy()
        frame["Z"] = 100.0
        frame["quality"] = np.repeat([1.0, 2.0], 4)
        frame.pc.set_data_name("height")

        # Select height values and group by quality
        options: dict[str, Any] = {
            "other": np.zeros(8),
            "other_precision": "negligible",
            "predictors": {"quality": "quality"},
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "spread_estimator": np.std,
            "bins": 2,
            "min_count": 2,
        }

        # Run xDEM and GeoUtils
        expected = gu.ErrorStructure.estimate(frame, **options)
        actual = xdem.uncertainty.estimate_error_structure(frame, **options)

        # Check grouped height statistics
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(actual_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)


class TestCoregistrationSummary:
    """Test module for optional simulated Coregs and plots of translation and rotation uncertainty."""

    @pytest.mark.parametrize("return_coregs", [False, True])
    @pytest.mark.parametrize("return_samples", [False, True])
    def test_propagate_uncertainty_coreg__saved_coregs(self, return_coregs: bool, return_samples: bool) -> None:
        """Checks that saved Coregs match the parameter summaries and store each fit's metadata."""

        # 1/ Simulate vertical shifts
        # Create a vertical offset
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        source = reference + 5
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Select Coreg and sample outputs independently
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            source,
            method,
            error_structure=errors,
            n_samples=4,
            random_state=2,
            return_coregs=return_coregs,
            return_samples=return_samples,
            at=["tz"],
        )

        # 2/ Check saved fits and summaries
        # Check optional outputs and the unchanged method
        assert (summary.samples is not None) == return_samples
        assert not method._fit_called
        if not return_coregs:
            assert summary.coregs is None
            return

        # Check separate Coregs and their vertical shifts
        assert summary.coregs is not None
        assert list(summary.coregs) == [1, 2, 3, 4]
        assert len({id(fitted) for fitted in summary.coregs.values()}) == 4
        parameters = pd.DataFrame(0.0, index=list(summary.coregs), columns=summary.estimate.index)
        for number, fitted in summary.coregs.items():
            assert fitted is not method
            assert fitted._fit_called
            shift = fitted.meta["outputs"]["affine"]["shift_z"]
            assert fitted.to_matrix()[2, 3] == shift
            parameters.loc[number, "tz"] = shift

        # Compare means, uncertainties and samples with saved fits
        np.testing.assert_allclose(parameters.mean(), summary.mean)
        np.testing.assert_allclose(parameters.std(), summary.std)
        if summary.samples is not None:
            np.testing.assert_array_equal(parameters.loc[:, summary.samples.columns], summary.samples)

    @pytest.mark.parametrize("supplied_axes", [False, True])
    def test_plot__parameters(self, supplied_axes: bool) -> None:
        """Checks that parameter plots show simulated means, standard deviations and the original estimate."""

        import matplotlib.pyplot as plt

        # 1/ Simulate vertical shifts
        # Create a vertical offset
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        source = reference + 5
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Save parameter samples
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            source,
            method,
            error_structure=errors,
            n_samples=4,
            random_state=2,
            return_samples=True,
        )

        # Check summaries and the unchanged method
        assert isinstance(summary, gu.PropagationSummary)
        assert isinstance(summary, xdem.uncertainty.PropagationSummary)
        assert summary.coregs is None
        assert summary.estimate["tz"] == pytest.approx(-5)
        assert summary.output["crs"] == reference.crs
        assert not method._fit_called
        assert summary.samples is not None
        np.testing.assert_allclose(summary.std, summary.samples.std())

        # 2/ Plot and check parameters
        # Create or reuse plot axes
        axes = tuple(plt.subplots(1, 2)[1]) if supplied_axes else None
        plotted_axes = summary.plot(axes=axes)
        if axes is not None:
            assert plotted_axes == axes

        # Check means, error bars and the original fit
        for axis, names, label in zip(
            plotted_axes, (("tx", "ty", "tz"), ("rx", "ry", "rz")), ("Translation (metre)", "Rotation (radian)")
        ):
            mean = summary.mean.loc[list(names)].to_numpy()
            std = summary.std.loc[list(names)].to_numpy()
            endpoints = np.asarray(axis.collections[0].get_segments())[:, :, 1]
            np.testing.assert_allclose(axis.lines[0].get_ydata(orig=False), mean)
            np.testing.assert_allclose(endpoints, np.column_stack((mean - std, mean + std)))
            np.testing.assert_allclose(axis.lines[-1].get_ydata(orig=False), summary.estimate.loc[list(names)])
            assert axis.get_ylabel() == label
        plt.close(plotted_axes[0].figure)


class TestCoregistrationAnalyticalPropagation:
    """Test module for analytical summaries of GLS/TLS fit uncertainties."""

    @pytest.mark.parametrize("optimizer", ["gls", "tls"])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_propagate_uncertainty_coreg__lzd(self, optimizer: str, only_translation: bool) -> None:
        """Checks that analytical summaries return LZD fit uncertainties unchanged."""

        if optimizer == "gls" and find_spec("gpytorch") is None:
            pytest.skip("GLS requires optional GPyTorch.")

        # 1/ Prepare the fit
        # Create curved terrain to constrain translations and rotations
        rows, columns = np.indices((15, 17))
        values = 5 * np.sin(columns / 3) + 3 * np.cos(rows / 4) + 2 * np.sin((columns + rows) / 5)
        reference = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 30), 32632)
        aligned = reference + 1
        sigma = np.full(reference.shape, 0.1)
        correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        fit_options: dict[str, Any] = {
            "sig_tba": sigma,
            "corr_tba": correlation,
            "force_opti": optimizer,
        }
        if optimizer == "tls":
            fit_options.update(sig_ref=sigma, corr_ref=correlation)

        # 2/ Compare analytical summaries with fit uncertainties
        # Fit LZD and summarize its uncertainties
        fitted = xdem.coreg.LZD(only_translation=only_translation, subsample=100).fit(
            reference, aligned, random_state=2, **fit_options
        )
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, aligned, fitted, error_structure=errors, method="analytical"
        )

        # Check fitted uncertainties and zero uncertainty for inactive rotations
        count = 3 if only_translation else 6
        np.testing.assert_array_equal(summary.std.iloc[:count], fitted.meta["outputs"]["fitorbin"]["fit_perr"])
        np.testing.assert_array_equal(summary.std.iloc[count:], 0)
        np.testing.assert_allclose(summary.estimate.iloc[:3], fitted.to_translations())
        assert summary.output["centroid"] == fitted.centroid()
        assert type(summary) is xdem.uncertainty.PropagationSummary
        assert summary.coregs is None


class TestCoregistrationPropagation:
    """Test module for simulating fitted coregistration parameters."""

    def test_propagate_uncertainty_coreg__parameters(self) -> None:
        """Checks that coregistration propagation returns six parameter summaries and saved samples."""

        # Load example DEMs and define independent errors
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.5)])

        # Run two simulations to estimate parameter uncertainty
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            summary = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                source,
                xdem.coreg.LZD(),
                n_samples=2,
                error_structure=structure,
                random_state=5,
                return_samples=True,
            )

        # Check six parameters and two sample rows
        assert isinstance(summary.estimate, pd.Series)
        assert isinstance(summary.mean, pd.Series)
        assert isinstance(summary.std, pd.Series)
        assert summary.estimate.shape == summary.mean.shape == summary.std.shape == (6,)
        assert summary.samples is not None
        assert summary.samples.shape == (2, 6)
        assert "centroid" in summary.output

    @pytest.mark.parametrize("point_input", [None, "ref", "tba"])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    @pytest.mark.parametrize("precoreg", [False, True])
    def test_propagate_uncertainty_coreg__reproducibility(
        self, point_input: str | None, error_applied_to: Literal["ref", "tba"], precoreg: bool
    ) -> None:
        """Checks that repeated coregistration simulations match across input types and initial alignments."""

        # 1/ Prepare inputs
        # Create a vertical offset with raster or point inputs
        elevations = np.arange(48.0).reshape(6, 8)
        transform = Affine(10, 0, 0, 0, -10, 60)
        reference: Any = xdem.DEM(gu.Raster.from_array(elevations, transform, 32632))
        source: Any = reference + 5
        if point_input == "ref":
            reference = reference.to_pointcloud()
        elif point_input == "tba":
            source = source.to_pointcloud()

        # Copy inputs to check for changes after fitting
        original_reference = reference.data.copy()
        original_source = source.data.copy()
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.2)])

        # 2/ Run twice with the same seed
        result = xdem.uncertainty.propagate_uncertainty(
            source,
            "coreg",
            error_structure=structure,
            n_samples=3,
            random_state=4,
            return_samples=True,
            return_coregs=True,
            inputs={"coreg": {"reference_elev": reference}},
            operation_options={
                "coreg_method": method,
                "error_applied_to": error_applied_to,
                "precoreg": precoreg,
            },
        )
        repeated = xdem.uncertainty.propagate_uncertainty(
            source,
            "coreg",
            error_structure=structure,
            n_samples=3,
            random_state=4,
            return_samples=True,
            return_coregs=True,
            inputs={"coreg": {"reference_elev": reference}},
            operation_options={
                "coreg_method": method,
                "error_applied_to": error_applied_to,
                "precoreg": precoreg,
            },
        )
        assert isinstance(result.estimate, pd.Series)
        assert isinstance(result.mean, pd.Series)
        assert isinstance(result.std, pd.Series)
        assert isinstance(result.n_valid, pd.Series)
        assert isinstance(repeated.mean, pd.Series)
        assert isinstance(repeated.std, pd.Series)

        # 3/ Compare results
        # Initial alignment removes the original vertical shift
        expected_nominal_shift = 0 if precoreg else -5
        assert result.estimate["tz"] == pytest.approx(expected_nominal_shift)
        assert result.mean["tz"] == pytest.approx(expected_nominal_shift, abs=0.1)
        assert "centroid" in result.output

        # Check equal means and uncertainties
        pd.testing.assert_series_equal(result.mean, repeated.mean)
        pd.testing.assert_series_equal(result.std, repeated.std)

        # Check all three fits contribute to summaries and samples
        assert result.n_success == result.n_samples == 3
        assert result.n_valid.eq(3).all()
        assert result.samples is not None
        assert result.samples.index.tolist() == [1, 2, 3]
        assert result.samples.columns.tolist() == ["tx", "ty", "tz", "rx", "ry", "rz"]

        # Compare saved transformations
        assert result.coregs is not None
        assert repeated.coregs is not None
        assert list(result.coregs) == list(repeated.coregs) == [1, 2, 3]
        for number, fitted in result.coregs.items():
            np.testing.assert_array_equal(fitted.to_matrix(), repeated.coregs[number].to_matrix())

        # Check unchanged method and elevations
        assert "shift_z" not in method.meta["outputs"].get("affine", {})
        np.testing.assert_array_equal(reference.data, original_reference)
        np.testing.assert_array_equal(source.data, original_source)

    @pytest.mark.parametrize("input_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    def test_propagate_uncertainty_coreg__xarray(
        self, input_kind: str, error_applied_to: Literal["ref", "tba"]
    ) -> None:
        """Checks that Dataset variables and native inputs give the same LZD uncertainty."""

        # 1/ Prepare inputs
        # Create curved terrain to constrain all translations
        rows, columns = np.mgrid[:15, :17]
        values = 5 * np.sin(columns / 7) + 3 * np.cos(rows / 6) + 2 * np.sin((columns + rows) / 11)
        transform = Affine(1, 0, 500_000, 0, -1, 7_000_000)
        reference = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        source = reference + 1
        if input_kind == "pointcloud":
            source = source.to_pointcloud()
        errors = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.01)])
        coreg_method = xdem.coreg.LZD(
            only_translation=True, subsample=1, tolerance_translation=1e-6, tolerance_rotation=None
        )

        # Create a Dataset with separate raster and point coordinates
        source_array = source.to_xarray()
        if input_kind == "pointcloud":
            source_array = source_array.rename({"x": "x_point", "y": "y_point"})
        dataset = xr.Dataset({"reference": reference.to_xarray(), "elevation": source_array})
        original = dataset.copy(deep=True)
        options: dict[str, Any] = dict(
            error_structure=errors,
            method="numerical",
            error_applied_to=error_applied_to,
            random_state=2,
            n_samples=3,
            return_samples=True,
        )

        # 2/ Run and compare outputs
        # Run propagation on native inputs and Dataset variables
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, source, coreg_method, **options)
        actual = xdem.uncertainty.propagate_uncertainty_coreg(
            dataset["reference"], dataset["elevation"], coreg_method, **options
        )

        # Check equal summaries and samples
        pd.testing.assert_series_equal(actual.mean, expected.mean)
        pd.testing.assert_series_equal(actual.std, expected.std)
        assert actual.samples is not None and expected.samples is not None
        pd.testing.assert_frame_equal(actual.samples, expected.samples)

        # Check unchanged inputs
        xr.testing.assert_identical(dataset, original)
        assert not coreg_method._fit_called


class TestElevationErrorEstimationChunked:
    """Test module for elevation error estimation with Dask and multiprocessing."""

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param("dask", marks=pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")),
            "multiproc",
        ],
    )
    def test_estimate_error_structure__terrain_predictors(self, backend: str, tmp_path: Path) -> None:
        """Checks that Dask/MP terrain predictors give the same grouped statistics as in-memory inputs."""

        # 1/ Create inputs and expected result
        # Create curved terrain with random errors
        rng = np.random.default_rng(32)
        y, x = np.mgrid[:24, :24]
        elevation = 0.3 * x + 0.5 * y + 0.04 * x * y
        transform = Affine(10, 0, 0, 0, -10, 240)
        source = xdem.DEM(gu.Raster.from_array(elevation + rng.normal(size=x.shape), transform, 32632, nodata=-9999))
        reference = xdem.DEM(gu.Raster.from_array(elevation, transform, 32632, nodata=-9999))

        # Set magnitude estimation options
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "random_state": 4,
        }

        expected = source.estimate_error_structure(reference, **options)

        # 2/ Run Dask or MP estimation
        if backend == "dask":
            # Use different 6 x 8 and 8 x 6 chunks to check alignment
            chunked_source = source.to_xarray().chunk({"x": 8, "y": 6})
            chunked_reference = reference.to_xarray().chunk({"x": 6, "y": 8})
            mp_config = None
            assert not chunked_source.rst.is_loaded
            assert not chunked_reference.rst.is_loaded
        else:
            # Write rasters for MP reads in 8 x 8 tiles
            source_path = tmp_path / "source.tif"
            reference_path = tmp_path / "reference.tif"
            source.to_file(source_path)
            reference.to_file(reference_path)
            chunked_source = xdem.DEM(source_path)
            chunked_reference = xdem.DEM(reference_path)
            mp_config = MultiprocConfig(chunks=(8, 8))
            assert not chunked_source.is_loaded
            assert not chunked_reference.is_loaded

        # Estimate from slope and curvature without loading either input
        result = xdem.uncertainty.estimate_error_structure(
            chunked_source, chunked_reference, mp_config=mp_config, **options
        )
        source_loaded = chunked_source.rst.is_loaded if backend == "dask" else chunked_source.is_loaded
        reference_loaded = chunked_reference.rst.is_loaded if backend == "dask" else chunked_reference.is_loaded
        assert not source_loaded
        assert not reference_loaded

        # 3/ Compare grouped statistics
        magnitude = result.components["measurement"].magnitude
        eager_magnitude = expected.components["measurement"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert isinstance(eager_magnitude, gu.ErrorMagnitude)
        assert result.empirical_variogram is None
        assert isinstance(magnitude.grouped_statistics, pd.DataFrame)

        # Allow rounding from reductions across chunks
        pd.testing.assert_frame_equal(
            magnitude.grouped_statistics, eager_magnitude.grouped_statistics, check_exact=False, rtol=0, atol=1e-12
        )

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Requires Dask-GeoPandas")
    def test_estimate_error_structure__point_raster(self) -> None:
        """Checks that a Dask EPC matches in-memory estimation against a DEM and stays lazy."""

        import dask_geopandas

        from xdem.epc.pd_accessor import _register_dask_epc_accessor

        # Create points inside a reference DEM
        columns, rows = np.meshgrid(np.arange(7), np.arange(5))
        values = np.linspace(-1, 1, columns.size)
        source = xdem.EPC(gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values, crs=32632))
        reference = xdem.DEM.from_array(np.zeros((7, 9)), Affine(1, 0, -1, 0, -1, 6), 32632, nodata=-9999)
        options: dict[str, Any] = {
            "predictors": {},
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }
        assert source.is_loaded and reference.is_loaded

        # Split 35 points into three uneven partitions to check row alignment
        _register_dask_epc_accessor()
        lazy = dask_geopandas.from_geopandas(source.gdf, npartitions=3, sort=False)
        graph = lazy.expr
        assert not lazy.epc.is_loaded

        # Run in-memory and Dask estimation
        expected = source.estimate_error_structure(reference, **options)
        actual = lazy.epc.estimate_error_structure(reference, **options)

        # Allow rounding from reductions across partitions
        assert isinstance(actual, gu.ErrorStructure)
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude(), rel=0, abs=1e-12)

        # Check lazy points and loaded source data
        assert lazy.expr is graph and not lazy.epc.is_loaded
        assert source.is_loaded and reference.is_loaded


class TestElevationErrorEstimationErrors:
    """Test module for grid validation during error estimation."""

    def test_estimate_error_structure__error_mismatched_grids(self) -> None:
        """Checks an error is raised for mismatched grids when alignment is disabled."""

        # Create grids with different resolutions
        source = xdem.DEM.from_array(
            np.arange(36, dtype=float).reshape(6, 6), Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999
        )
        reference = xdem.DEM.from_array(np.zeros((3, 3)), Affine(20, 0, 0, 0, -20, 60), 32632, nodata=-9999)

        # Check the error with alignment disabled
        with pytest.raises(ValueError, match="does not share the selected support grid"):
            source.estimate_error_structure(other=reference, predictors={}, align="raise")


class TestPropagationErrors:
    """Test module for unsupported analytical fits and explicit numerical sample selection."""

    def test_propagate_uncertainty_coreg__error_analytical_coregs(self) -> None:
        """Checks an error is raised for simulated Coregs requested with analytical propagation."""

        # Create an error model
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        method = xdem.coreg.VerticalShift()

        # Check analytical propagation cannot return simulated Coregs
        with pytest.raises(ValueError, match="Analytical propagation cannot return simulated Coreg objects"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                None, None, method, error_structure=errors, method="analytical", return_coregs=True
            )

    def test_propagate_uncertainty_coreg__error_failed_draw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that saved Coregs have the correct simulation numbers after a failed draw."""

        # Create a vertical offset and fail the first error draw
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])
        generate = gu.ErrorStructure.generate_random_field
        draws = iter([False, True, True])

        def draw(self: gu.ErrorStructure, *args: Any, **kwargs: Any) -> Any:
            """Fail the first draw and generate the remaining error fields normally."""

            if not next(draws):
                raise RuntimeError("failed draw")
            return generate(self, *args, **kwargs)

        # Run simulations and check the failure warning
        monkeypatch.setattr(gu.ErrorStructure, "generate_random_field", draw)
        with pytest.warns(UserWarning, match="failed draw"):
            summary = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                reference + 5,
                xdem.coreg.VerticalShift(vshift_reduc_func=np.mean),
                error_structure=errors,
                n_samples=3,
                random_state=2,
                return_coregs=True,
                return_samples=True,
            )

        # Check the failed draw and saved fits
        assert summary.failures == {1: "failed draw"}
        assert summary.n_success == 2
        assert summary.coregs is not None
        assert list(summary.coregs) == [2, 3]
        assert summary.samples is not None
        assert summary.samples.loc[1].isna().all()
        for number, fitted in summary.coregs.items():
            assert fitted.to_matrix()[2, 3] == summary.samples.loc[number, "tz"]

    @pytest.mark.parametrize("with_source", [False, True])
    def test_propagate_uncertainty__error_custom_analytical(self, with_source: bool, tmp_path: Path) -> None:
        """Checks an error is raised for analytical custom propagation before loading elevations."""

        # Open an optional raster without loading it
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        source = None
        if with_source:
            raster = gu.Raster.from_array(np.ones((5, 7)), Affine(1, 0, 0, 0, -1, 5), 32632)
            source_path = tmp_path / "source.tif"
            raster.to_file(source_path)
            source = gu.Raster(source_path)
            assert not source.is_loaded

        # Request a custom analytical calculation with a terrain predictor
        with pytest.raises(NotImplementedError, match="Custom callables only support numerical propagation"):
            xdem.uncertainty.propagate_uncertainty(
                source,
                np.mean,
                error_structure=errors,
                predictors={"slope": "slope"},
                method="analytical",
            )

        # Check the raster stays unloaded
        if source is not None:
            assert not source.is_loaded

    def test_propagate_uncertainty_coreg__error_ols(self) -> None:
        """Checks an error is raised for analytical coregistration without GLS/TLS fit uncertainties."""

        # Create curved terrain for LZD
        rows, columns = np.indices((9, 11))
        values = np.sin(columns / 2) + np.cos(rows / 3)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 18), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Check OLS cannot provide analytical fit uncertainty
        with pytest.raises(ValueError, match="GLS/TLS uncertainties"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                raster,
                raster + 1,
                xdem.coreg.LZD(only_translation=True, subsample=1),
                error_structure=errors,
                method="analytical",
            )

    def test_propagate_uncertainty__error_sample_selection(self) -> None:
        """Checks an error is raised when raster simulation draws are requested without selecting pixels."""

        # Create a raster and error model
        raster = gu.Raster.from_array(np.ones((5, 7)), Affine(1, 0, 0, 0, -1, 5), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Check saving samples requires selected pixels
        with pytest.raises(ValueError, match="Select outputs with at"):
            xdem.uncertainty.propagate_uncertainty(raster, error_structure=errors, return_samples=True)


class TestCoregistrationPropagationErrors:
    """Test module for failed coregistration fits during uncertainty propagation."""

    def test_propagate_uncertainty_coreg__error_failed_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that failed fits are recorded and excluded from parameter summaries."""

        # 1/ Prepare a fitting failure
        # Create flat terrain for a vertical shift
        reference = xdem.DEM(gu.Raster.from_array(np.ones((4, 5)), Affine(1, 0, 0, 0, -1, 4), 32632))
        method = xdem.coreg.VerticalShift()
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.2)])

        # Fail the first simulation after the original fit
        original_fit = xdem.coreg.VerticalShift.fit
        calls = iter([True, False, True, True])

        def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Fail the selected fit and otherwise use the original coregistration method."""

            if not next(calls):
                raise RuntimeError("failed fit")
            return original_fit(self, *args, **kwargs)

        # 2/ Run simulations
        # Check the fitting failure emits a warning
        monkeypatch.setattr(xdem.coreg.VerticalShift, "fit", fit)
        with pytest.warns(UserWarning, match="failed fit"):
            summary = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                reference + 5,
                method,
                error_structure=structure,
                n_samples=3,
                random_state=4,
                return_samples=True,
                return_coregs=True,
            )

        # 3/ Check successful fits and samples
        # Check counts and missing samples for the failed fit
        assert summary.n_samples == 3
        assert summary.n_success == 2
        assert summary.failures == {1: "failed fit"}
        assert summary.samples is not None
        assert summary.samples.index.tolist() == [1, 2, 3]
        assert summary.samples.loc[1].isna().all()

        # Compare saved Coregs with successful samples
        assert summary.coregs is not None
        assert list(summary.coregs) == [2, 3]
        for number, fitted in summary.coregs.items():
            assert fitted.to_matrix()[2, 3] == summary.samples.loc[number, "tz"]

        # Check uncertainty from successful fits
        np.testing.assert_allclose(summary.std, summary.samples.std())

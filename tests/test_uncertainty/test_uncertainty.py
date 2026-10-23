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
    """Test module for public estimation methods on DEMs and elevation point clouds."""

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("entrypoint", ["function", "method", "accessor"])
    @pytest.mark.parametrize("other_precision", [None, "negligible", "same"])
    def test_estimate_error_structure__array_comparison(
        self, kind: str, entrypoint: str, other_precision: str | None
    ) -> None:
        """Checks that array comparisons, masks and precision assumptions give the expected error magnitude."""

        # Create errors with a known spread and exclude the last row from estimation
        values = np.arange(36, dtype=float).reshape(6, 6)
        mask = np.indices(values.shape)[0] < 5
        if kind == "raster":
            source = xdem.DEM.from_array(values, Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999)
        else:
            x, y = np.meshgrid(np.arange(6), np.arange(6))
            values, mask = values.ravel(), mask.ravel()
            source = xdem.EPC(gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32632))
        options: dict[str, Any] = {
            "other": np.zeros_like(values),
            "predictors": {},
            "mask": mask,
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }
        if other_precision is not None:
            options["other_precision"] = other_precision

        # Run each public entry point with the same GeoUtils argument names
        if entrypoint == "function":
            result = xdem.uncertainty.estimate_error_structure(source, **options)
        elif entrypoint == "method":
            result = source.estimate_error_structure(**options)
        else:
            accessor = source.to_xarray().dem if kind == "raster" else source.gdf.epc
            result = accessor.estimate_error_structure(**options)

        # Equal independent errors require sqrt(2) scaling; omitted precision uses xDEM's negligible default
        scale = np.sqrt(2) if other_precision == "same" else 1.0
        assert type(result) is gu.ErrorStructure
        assert result.predict_magnitude() == pytest.approx(np.std(values[mask]) / scale)
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(mask)

    @pytest.mark.parametrize("other_precision", ["negligible", "same"])
    @pytest.mark.parametrize(
        "source_kind, reference_kind", [("raster", "raster"), ("raster", "point"), ("point", "raster")]
    )
    def test_estimate_error_structure__terrain_predictors_match_geoutils(
        self, source_kind: str, reference_kind: str, other_precision: Literal["negligible", "same"]
    ) -> None:
        """Checks that terrain predictor names give GeoUtils results for raster and point comparisons."""

        # Create curved terrain with measurement errors
        rows, columns = np.indices((12, 12))
        elevations = rows + columns + 0.1 * rows * columns
        errors = np.random.default_rng(7).normal(size=elevations.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        source_raster = xdem.DEM.from_array(elevations + errors, transform, 32632, nodata=-9999)
        reference_raster = xdem.DEM.from_array(elevations, transform, 32632, nodata=-9999)

        # Select sparse point locations so mixed comparisons differ from comparisons on the full grid
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

        # Compare terrain shorthand with an explicit slope predictor passed to GeoUtils on the same locations
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

        # Check every group fitted at the common finite locations
        assert type(actual) is gu.ErrorStructure
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(actual_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)

    def test_deprecated_dem_uncertainty_adapter(self) -> None:
        """Checks that the deprecated DEM method warns and returns its documented magnitude and correlation outputs."""

        # Require the optional backend used to fit spatial correlation
        pytest.importorskip("skgstat")

        # Load the same example pair used by the public estimation workflow
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))

        # Check that the legacy call directs the user to the migration guide
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
                magnitude, correlation = source.estimate_uncertainty(reference, random_state=42)

        # Check the magnitude grid and unit correlation at coincident locations
        assert isinstance(magnitude, gu.Raster)
        assert magnitude.georeferenced_grid_equal(source)
        assert correlation(0) == pytest.approx(1)

    @pytest.mark.parametrize("at", ["self", "other", None])
    def test_estimate_error_structure__comparison_locations(self, at: Literal["self", "other"] | None) -> None:
        """Checks that comparison locations and grid alignment are forwarded with a spatial mask."""

        # Create different source and reference grids and a mask covering the first four source columns
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

        # Give GeoUtils a spatial mask explicitly and let xDEM attach the same grid to the Boolean array
        expected = gu.ErrorStructure.estimate(source, other=reference, mask=source.copy(new_array=mask), **options)
        actual = source.estimate_error_structure(other=reference, mask=mask, **options)

        # Check the selected observations as well as their spread on each comparison grid
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        assert actual.fit_diagnostics == expected.fit_diagnostics

    def test_estimate_error_structure__point_attributes(self) -> None:
        """Checks that active point values and predictor columns are passed unchanged to GeoUtils."""

        # Create points with a selected height column, an unused Z column, and two quality groups
        values = np.array([-2.0, -1.0, 1.0, 2.0, -4.0, -2.0, 2.0, 4.0])
        points = gu.PointCloud.from_xyz(np.arange(8), np.zeros(8), values, crs=32632, data_name="height")
        frame = points.gdf.copy()
        frame["Z"] = 100.0
        frame["quality"] = np.repeat([1.0, 2.0], 4)
        frame.pc.set_data_name("height")
        options: dict[str, Any] = {
            "other": np.zeros(8),
            "other_precision": "negligible",
            "predictors": {"quality": "quality"},
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "spread_estimator": np.std,
            "bins": 2,
            "min_count": 2,
        }

        # Fit through xDEM and GeoUtils using the dataframe's selected attribute
        expected = gu.ErrorStructure.estimate(frame, **options)
        actual = xdem.uncertainty.estimate_error_structure(frame, **options)

        # Check both quality groups and the selected height values rather than the constant Z column
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(actual_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)


class TestElevationErrorEstimationChunked:
    """Test module for Dask and multiprocessing elevation error estimation against eager inputs."""

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param("dask", marks=pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")),
            "multiproc",
        ],
    )
    def test_estimate_error_structure__default_terrain_predictors_match_eager(
        self, backend: str, tmp_path: Path
    ) -> None:
        """Checks that default terrain predictors produce the eager grouped magnitude in Dask and MP."""

        # A curved surface gives slope and curvature variation while random errors define its spread
        rng = np.random.default_rng(32)
        y, x = np.mgrid[:24, :24]
        elevation = 0.3 * x + 0.5 * y + 0.04 * x * y
        transform = Affine(10, 0, 0, 0, -10, 240)
        source = xdem.DEM(gu.Raster.from_array(elevation + rng.normal(size=x.shape), transform, 32632, nodata=-9999))
        reference = xdem.DEM(gu.Raster.from_array(elevation, transform, 32632, nodata=-9999))
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "random_state": 4,
        }

        # Fit the eager reference, then prepare different Dask block layouts or file-backed MP tiles
        expected = source.estimate_error_structure(reference, **options)
        if backend == "dask":
            chunked_source = source.to_xarray().chunk({"x": 8, "y": 6})
            chunked_reference = reference.to_xarray().chunk({"x": 6, "y": 8})
            mp_config = None
            assert not chunked_source.rst.is_loaded
            assert not chunked_reference.rst.is_loaded
        else:
            source_path = tmp_path / "source.tif"
            reference_path = tmp_path / "reference.tif"
            source.to_file(source_path)
            reference.to_file(reference_path)
            chunked_source = xdem.DEM(source_path)
            chunked_reference = xdem.DEM(reference_path)
            mp_config = MultiprocConfig(chunks=(8, 8))
            assert not chunked_source.is_loaded
            assert not chunked_reference.is_loaded

        # Fit the same magnitude from derived slope and curvature without loading either input
        result = xdem.uncertainty.estimate_error_structure(
            chunked_source, chunked_reference, mp_config=mp_config, **options
        )
        source_loaded = chunked_source.rst.is_loaded if backend == "dask" else chunked_source.is_loaded
        reference_loaded = chunked_reference.rst.is_loaded if backend == "dask" else chunked_reference.is_loaded
        assert not source_loaded
        assert not reference_loaded

        # Check the complete grouped table rather than one interpolated location
        magnitude = result.components["measurement"].magnitude
        eager_magnitude = expected.components["measurement"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert isinstance(eager_magnitude, gu.ErrorMagnitude)
        assert result.empirical_variogram is None
        assert isinstance(magnitude.grouped_statistics, pd.DataFrame)
        pd.testing.assert_frame_equal(
            magnitude.grouped_statistics, eager_magnitude.grouped_statistics, check_exact=False, rtol=0, atol=1e-12
        )

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Requires Dask-GeoPandas")
    def test_estimate_error_structure__lazy_points_with_native_reference(self) -> None:
        """Checks that a Dask EPC can use a native DEM reference while its source graph stays lazy."""

        import dask_geopandas

        from xdem.epc.pd_accessor import _register_dask_epc_accessor

        # Create 35 elevation points inside a native reference raster
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

        # Split points into three uneven partitions to reveal dropped rows during interface conversion
        _register_dask_epc_accessor()
        lazy = dask_geopandas.from_geopandas(source.gdf, npartitions=3, sort=False)
        graph = lazy.expr
        assert not lazy.epc.is_loaded

        # Compare the compact model with eager estimation while mixing the Pandas and native elevation interfaces
        expected = source.estimate_error_structure(reference, **options)
        actual = lazy.epc.estimate_error_structure(reference, **options)

        # Allow rounding from variance reductions across partitions; neither calculation replaces source data
        assert isinstance(actual, gu.ErrorStructure)
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude(), rel=0, abs=1e-12)
        assert lazy.expr is graph and not lazy.epc.is_loaded
        assert source.is_loaded and reference.is_loaded


class TestElevationErrorEstimationErrors:
    """Test module for GeoUtils validation reached through the elevation estimation wrapper."""

    def test_estimate_error_structure__error_mismatched_grids(self) -> None:
        """Checks that mismatched grids raise an error when alignment is disabled."""

        # Create two grids with different pixel spacing over the same extent
        source = xdem.DEM.from_array(
            np.arange(36, dtype=float).reshape(6, 6), Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999
        )
        reference = xdem.DEM.from_array(np.zeros((3, 3)), Affine(20, 0, 0, 0, -20, 60), 32632, nodata=-9999)

        # Ask GeoUtils to validate the grids rather than using xDEM's default reprojection
        with pytest.raises(ValueError, match="does not share the selected support grid"):
            source.estimate_error_structure(other=reference, predictors={}, align="raise")


class TestCoregistrationPropagation:
    """Test module for simulating fitted coregistration parameters."""

    def test_error_structure_propagates_through_coregistration(self) -> None:
        """Checks that an ErrorStructure produces translation and rotation summaries through coregistration."""

        # Load the example DEM pair and assign a small independent measurement error
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.5)])

        # Run two realizations so the returned parameter spread is defined
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

        # Check the six native transformation parameters and one retained value row per realization
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
    def test_coreg_propagation_support_reproducibility_and_prealignment(
        self, point_input: str | None, error_applied_to: Literal["ref", "tba"], precoreg: bool
    ) -> None:
        """Checks that coregistration propagation is reproducible for either error side and raster or point inputs."""

        # Create a known vertical offset and select which dataset, if any, uses point elevations
        elevations = np.arange(48.0).reshape(6, 8)
        transform = Affine(10, 0, 0, 0, -10, 60)
        reference: Any = xdem.DEM(gu.Raster.from_array(elevations, transform, 32632))
        source: Any = reference + 5
        if point_input == "ref":
            reference = reference.to_pointcloud()
        elif point_input == "tba":
            source = source.to_pointcloud()

        # Save the inputs so the simulations can be checked for unintended modification
        original_reference = reference.data.copy()
        original_source = source.data.copy()
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.2)])

        # Repeat the complete workflow with identical settings
        result = xdem.uncertainty.propagate_uncertainty(
            source,
            "coreg",
            error_structure=structure,
            n_samples=3,
            random_state=4,
            return_samples=True,
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

        # Initial alignment removes the original correction before the residual fit
        expected_nominal_shift = 0 if precoreg else -5
        assert result.estimate["tz"] == pytest.approx(expected_nominal_shift)
        assert result.mean["tz"] == pytest.approx(expected_nominal_shift, abs=0.1)
        assert "centroid" in result.output

        # Check reproducibility of the ensemble mean and spread
        pd.testing.assert_series_equal(result.mean, repeated.mean)
        pd.testing.assert_series_equal(result.std, repeated.std)

        # Check that all requested fits contribute to each parameter summary and retained table
        assert result.n_success == result.n_samples == 3
        assert result.n_valid.eq(3).all()
        assert result.samples is not None
        assert result.samples.index.tolist() == [1, 2, 3]
        assert result.samples.columns.tolist() == ["tx", "ty", "tz", "rx", "ry", "rz"]

        # Check that fitting copies preserve the original method and both elevation datasets
        assert "shift_z" not in method.meta["outputs"].get("affine", {})
        np.testing.assert_array_equal(reference.data, original_reference)
        np.testing.assert_array_equal(source.data, original_source)

    @pytest.mark.parametrize("input_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    def test_propagate_uncertainty_coreg__xarray_variables_match_native(
        self, input_kind: str, error_applied_to: Literal["ref", "tba"]
    ) -> None:
        """Checks that raster and point variables selected from Datasets reproduce native LZD uncertainty."""

        # Use curved terrain so all three translations are observable
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

        # Put both representations in a Dataset with separate raster and point coordinates
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

        # Compare selected variables with the same fixed-support fit on native inputs
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, source, coreg_method, **options)
        actual = xdem.uncertainty.propagate_uncertainty_coreg(
            dataset["reference"], dataset["elevation"], coreg_method, **options
        )
        pd.testing.assert_series_equal(actual.mean, expected.mean)
        pd.testing.assert_series_equal(actual.std, expected.std)
        assert actual.samples is not None and expected.samples is not None
        pd.testing.assert_frame_equal(actual.samples, expected.samples)
        xr.testing.assert_identical(dataset, original)
        assert not coreg_method._fit_called


class TestCoregistrationPropagationErrors:
    """Test module for failed coregistration fits during uncertainty propagation."""

    def test_propagate_uncertainty_coreg__error_failed_fit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that a failed coregistration is omitted from the successful count and parameter summaries."""

        # Use a constant reference so the successful fits have only a vertical translation
        reference = xdem.DEM(gu.Raster.from_array(np.ones((4, 5)), Affine(1, 0, 0, 0, -1, 4), 32632))
        method = xdem.coreg.VerticalShift()
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.2)])

        # Allow the original estimate, fail the first simulation and allow the remaining two
        original_fit = xdem.coreg.VerticalShift.fit
        calls = iter([True, False, True, True])

        def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Fail the selected fit and otherwise use the original coregistration method."""

            if not next(calls):
                raise RuntimeError("failed fit")
            return original_fit(self, *args, **kwargs)

        # Run with one controlled fitting failure and check that it emits a warning
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
            )

        # Check that success counts and retained simulation numbers preserve the failed first realization
        assert summary.n_samples == 3
        assert summary.n_success == 2
        assert summary.failures == {1: "failed fit"}
        assert summary.samples is not None
        assert summary.samples.index.tolist() == [1, 2, 3]
        assert summary.samples.loc[1].isna().all()

        # Compare the reported spread with the sample standard deviation of successful fits only
        np.testing.assert_allclose(summary.std, summary.samples.std())

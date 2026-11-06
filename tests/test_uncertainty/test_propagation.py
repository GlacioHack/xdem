"""Tests for public elevation uncertainty estimation and propagation workflows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from geoutils._dispatch import _get_pointcloud_interface, _get_raster_interface
from geoutils.multiproc import MultiprocConfig
from geoutils.raster.xr_accessor import DataArrayRasterAccessor

import xdem
from xdem import DEM, open_dem


class TestElevationErrorEstimation:
    """Test module for error estimation methods on DEMs and EPCs."""

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize(
        "entrypoint, other_precision, comparison",
        [
            pytest.param("function", None, False, id="function-proxy-default"),
            pytest.param("function", None, True, id="function-comparison-default"),
            pytest.param("method", "same", False, id="method-proxy-same"),
            pytest.param("method", "same", True, id="method-comparison-same"),
            pytest.param("accessor", "negligible", False, id="accessor-proxy-negligible"),
            pytest.param("accessor", "negligible", True, id="accessor-comparison-negligible"),
        ],
    )
    def test_estimate_error_structure__masked_inputs(
        self, kind: str, entrypoint: str, other_precision: str | None, comparison: bool
    ) -> None:
        """Checks that error structure estimation works the same through every API entry point."""

        # 1/ Create error proxy, or differences source/other from a zero array
        # Exclude the last row with a mask, and one NaN
        values = np.arange(-18.0, 18.0).reshape(6, 6)
        if not comparison:
            values[2, 3] = np.nan
        mask = np.indices(values.shape)[0] < 5
        if kind == "raster":
            source = xdem.DEM.from_array(values, Affine(2, 0, 0, 0, -2, 12), 32632, nodata=-9999)
        else:
            rows, columns = np.indices(values.shape)
            values, mask = values.ravel(), mask.ravel()
            source = xdem.EPC(gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), values, crs=32632))

        # We set precision options only when passed by user, preserving the defaults for a proxy
        options: dict[str, Any] = {
            "mask": mask,
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }
        if comparison:
            options.update(other=np.zeros_like(values), predictors={}, precoreg=False)
        if other_precision is not None:
            options["other_precision"] = other_precision

        # 2/ Estimate through the function, method or accessor
        if entrypoint == "function":
            result = xdem.uncertainty.estimate_error_structure(source=source, **options)
        elif entrypoint == "method":
            result = source.estimate_error_structure(**options)
        else:
            accessor = source.to_xarray().dem if kind == "raster" else source.gdf.epc
            result = accessor.estimate_error_structure(**options)

        # 3/ Check equality
        # Equal independent errors increase variance by factor of 2
        # A proxy already describes errors, so a precision assumption must not rescale it
        valid = mask & np.isfinite(values)
        scale = np.sqrt(2) if comparison and other_precision == "same" else 1.0
        assert type(result) is gu.ErrorStructure
        assert result.predict_magnitude() == pytest.approx(np.std(values[valid]) / scale)
        assert result.components["measurement"].magnitude.kind == "constant"
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(valid)

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("entrypoint", ["function", "method", "accessor"])
    def test_estimate_error_structure__proxy_predictors(self, kind: str, entrypoint: str) -> None:
        """Checks that a proxy and explicit terrain predictors match GeoUtils grouped statistics."""

        # 1/ Create a proxy and terrain predictors
        # Derive slope from elevations
        rows, columns = np.mgrid[:18, :20]
        grid = Affine(2, 0, 0, 0, -2, 36)
        elevations = xdem.DEM.from_array(rows + columns + 0.1 * rows * columns, grid, 32632, nodata=-9999)
        slope = elevations.slope().data
        errors = (1 + slope / 10) * np.random.default_rng(42).normal(size=rows.shape)
        proxy = elevations.copy(new_array=errors)
        source = proxy if kind == "raster" else proxy.to_pointcloud()
        # Match predictor rows to the finite pixels exported as points
        valid_pixels = np.isfinite(proxy.data.filled(np.nan)).ravel()
        predictors = {"slope": slope if kind == "raster" else slope.ravel()[valid_pixels]}
        options: dict[str, Any] = {
            "predictors": predictors,
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "spread_estimator": np.std,
            "bins": 3,
            "min_count": 10,
            "random_state": 4,
        }

        # 2/ Estimate and compare grouped statistics
        expected = gu.ErrorStructure.estimate(source, **options)
        if entrypoint == "function":
            actual = xdem.uncertainty.estimate_error_structure(source=source, **options)
        elif entrypoint == "method":
            actual = source.estimate_error_structure(**options)
        else:
            accessor = source.to_xarray().dem if kind == "raster" else source.gdf.epc
            actual = accessor.estimate_error_structure(**options)

        # Check equality
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert actual_magnitude.predictor_names == ("slope",)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)

    @pytest.mark.parametrize(
        "kind, entrypoint, override",
        [
            pytest.param("raster", "function", False, id="raster-function-defaults"),
            pytest.param("point", "function", True, id="points-function-overrides"),
            pytest.param("point", "method", False, id="points-method-defaults"),
            pytest.param("raster", "method", True, id="raster-method-overrides"),
            pytest.param("raster", "accessor", False, id="raster-accessor-defaults"),
            pytest.param("point", "accessor", True, id="points-accessor-overrides"),
        ],
    )
    def test_estimate_error_structure__estimators(self, kind: str, entrypoint: str, override: bool) -> None:
        """Checks that NMAD and Dowd are defaults and explicit estimators override them through every API."""

        pytest.importorskip("skgstat")

        # Create correlated errors with outliers
        rows, columns = np.mgrid[:20, :24]
        values = np.sin(columns / 3) + np.cos(rows / 4)
        values[::5, ::6] += 100
        proxy = xdem.DEM.from_array(values, Affine(2, 0, 0, 0, -2, 40), 32632, nodata=-9999)
        source = proxy if kind == "raster" else proxy.to_pointcloud()
        estimators: dict[str, Any] = {
            "spread_estimator": np.std if override else gu.stats.nmad,
            "variogram_estimator": "matheron" if override else "dowd",
        }
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "constant", "correlation": "spherical"}},
            "n_pairs": 5_000,
            "n_lags": 8,
            "pair_sampling": "random_xy",
            "random_state": 42,
        }
        expected = gu.ErrorStructure.estimate(source, **options, **estimators)
        if override:
            options.update(estimators)

        # Estimate through each public API
        if entrypoint == "function":
            actual = xdem.uncertainty.estimate_error_structure(source, **options)
        elif entrypoint == "method":
            actual = source.estimate_error_structure(**options)
        else:
            accessor = source.to_xarray().dem if kind == "raster" else source.gdf.epc
            actual = accessor.estimate_error_structure(**options)

        # Compare magnitude and the empirical variogram before model fitting
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        assert actual.metadata["spread_estimator"] == ("std" if override else "nmad")
        assert actual.metadata["variogram_estimator"] == estimators["variogram_estimator"]
        assert actual.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        np.testing.assert_array_equal(actual.empirical_variogram.counts, expected.empirical_variogram.counts)
        np.testing.assert_array_equal(
            actual.empirical_variogram.semivariance, expected.empirical_variogram.semivariance
        )

    @pytest.mark.parametrize("kind", ["raster", "point"])
    def test_estimate_error_structure__variable_precision(self, kind: str) -> None:
        """Checks that equal precision divides variable magnitudes by sqrt(2) without changing correlation."""

        pytest.importorskip("skgstat")

        # 1/ Create variable errors
        # Increase error magnitude across 4 groups
        rows, columns = np.mgrid[:24, :28]
        quality = np.broadcast_to(np.arange(28) // 7, rows.shape).astype(float)
        residual = (1 + quality) * (
            np.sin(columns / 3) + np.cos(rows / 4) + np.random.default_rng(42).normal(0, 0.2, rows.shape)
        )
        grid = Affine(2, 0, 0, 0, -2, 48)
        reference = xdem.DEM.from_array(100 + rows + columns, grid, 32632, nodata=-9999)
        measured = reference.copy(new_array=reference.data + residual)

        # Select raster or point observations with the same predictor values
        source = measured if kind == "raster" else measured.to_pointcloud()
        other = reference if kind == "raster" else reference.to_pointcloud()
        predictors = {"quality": quality if kind == "raster" else quality.ravel()}
        options: dict[str, Any] = {
            "predictors": predictors,
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
            "bins": 4,
            "min_count": 10,
            "spread_estimator": np.std,
            # Supply enough pairs for fitting
            "n_pairs": 5_000,
            "n_lags": 8,
            "pair_sampling": "random_xy",
            "random_state": 42,
        }

        # 2/ Estimate both precision assumptions
        negligible = xdem.uncertainty.estimate_error_structure(
            source, other=other, other_precision="negligible", precoreg=False, **options
        )
        same = xdem.uncertainty.estimate_error_structure(
            source, other=other, other_precision="same", precoreg=False, **options
        )

        # Equal precision splits difference variance between the two inputs
        magnitude = negligible.predict_magnitude(predictors)
        assert np.ptp(magnitude) > 0
        np.testing.assert_allclose(same.predict_magnitude(predictors), magnitude / np.sqrt(2))

        # Normalizing by magnitude should give the same empirical variogram and fitted correlation
        assert negligible.empirical_variogram is not None
        assert same.empirical_variogram is not None
        np.testing.assert_array_equal(same.empirical_variogram.counts, negligible.empirical_variogram.counts)
        np.testing.assert_allclose(same.empirical_variogram.semivariance, negligible.empirical_variogram.semivariance)
        distances = np.array([0.0, 5.0, 10.0, 30.0])
        # Allow rounding near zero correlation
        np.testing.assert_allclose(
            same.predict_correlation(distances), negligible.predict_correlation(distances), atol=1e-10
        )

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
        sampled = source.cosample(reference, auxiliary={"slope": terrain_source.slope()}, align="reproject")
        scale = np.sqrt(2) if other_precision == "same" else 1
        expected_predictors: dict[str, Any]
        if isinstance(sampled, gu.Raster):
            residuals = sampled.copy(new_array=(sampled.data[0] - sampled.data[1]) / scale)
            expected_predictors = {"slope": (sampled, 3)}
        else:
            difference = (sampled.gdf["self"].to_numpy() - sampled.gdf["other"].to_numpy()) / scale
            residuals = sampled.copy(new_array=difference)
            expected_predictors = {"slope": "slope"}
        expected = gu.ErrorStructure.estimate(residuals, predictors=expected_predictors, **options)
        actual = xdem.uncertainty.estimate_error_structure(
            source, reference, predictors=("slope",), other_precision=other_precision, precoreg=False, **options
        )

        # 3/ Check equality of grouped statistics
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
            with pytest.warns(DeprecationWarning, match="release_notes.html#uncertainty-module-migration"):
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
        sampled = source.cosample(reference, mask=source.copy(new_array=mask), at=at, align="reproject")
        expected = gu.ErrorStructure.estimate(
            sampled, predictors={}, components=options["components"], spread_estimator=np.std
        )
        actual = source.estimate_error_structure(other=reference, mask=mask, precoreg=False, **options)

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
        actual = xdem.uncertainty.estimate_error_structure(frame, precoreg=False, **options)

        # Check grouped height statistics
        expected_magnitude = expected.components["measurement"].magnitude
        actual_magnitude = actual.components["measurement"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(actual_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)


class TestElevationErrorEstimationPrecoreg:
    """Test module for optional alignment before estimating elevation errors."""

    @pytest.mark.parametrize("precoreg", [False, True])
    def test_estimate_error_structure__proxy_coreg(self, coreg_method: xdem.coreg.AffineCoreg, precoreg: bool) -> None:
        """Checks that a proxy bypasses every coregistration method, including with prealignment enabled."""

        # Create errors with a known spread
        values = np.arange(12.0).reshape(3, 4)
        proxy = xdem.DEM.from_array(values, Affine(2, 0, 0, 0, -2, 6), 32632, nodata=-9999)

        # Supply a method without a reference dataset
        result = proxy.estimate_error_structure(
            coreg_method=coreg_method,
            precoreg=precoreg,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Check the unscaled proxy spread and unchanged method
        assert result.predict_magnitude() == pytest.approx(np.std(values))
        assert not coreg_method._fit_called

    def test_estimate_error_structure__default_coreg(self, curved_dem: xdem.DEM) -> None:
        """Checks that prealignment defaults to LZD."""

        # Create curved terrain with a vertical offset and small measurement errors
        reference = curved_dem
        elevations = reference.data.copy()
        source = reference.copy(new_array=elevations + 1 + np.random.default_rng(42).normal(0, 0.01, reference.shape))
        options: dict[str, Any] = {
            "predictors": {},
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
            "random_state": 4,
        }

        # Compare the default with an explicit LZD method
        actual = source.estimate_error_structure(reference, **options)
        expected = source.estimate_error_structure(reference, coreg_method=xdem.coreg.LZD(), **options)
        assert actual.predict_magnitude() == expected.predict_magnitude()
        assert actual.fit_diagnostics == expected.fit_diagnostics

    @pytest.mark.parametrize("shift", [(10, -10), (3.7, -2.4), (5, -5)])
    @pytest.mark.parametrize(
        "rotation, point_input",
        [(0, None), (0.2, None), (0, "ref"), (0, "tba")],
        ids=["dem-dem", "dem-dem-rotation", "dem-epc", "epc-dem"],
    )
    def test_estimate_error_structure__resampling_noise(
        self, shift: tuple[float, float], rotation: float, point_input: str | None
    ) -> None:
        """Checks that precoreg with DEM/EPC references does not resample, to preserve source measurement noise."""

        # Create curved terrain and 0.5 m white noise on 10 m pixels
        rows, columns = np.mgrid[:64, :70]
        values = 0.02 * columns**2 + 0.03 * rows**2 + 0.01 * rows * columns
        noise = np.random.default_rng(42).normal(0, 0.5, values.shape)
        grid = Affine(10, 0, 500_000, 0, -10, 7_000_000)
        reference: Any = xdem.DEM.from_array(values, grid, 32632, nodata=-9999)

        # Rotate the source grid about its center and add the known horizontal and vertical offsets
        center = (values.shape[1] / 2, values.shape[0] / 2)
        rotation_grid = (
            Affine.translation(*center) * Affine.rotation(rotation) * Affine.translation(-center[0], -center[1])
        )
        source_grid = Affine.translation(*shift) * grid * rotation_grid
        source: Any = xdem.DEM.from_array(values + noise + 1, source_grid, 32632, nodata=-9999)

        # Store the mask on the source grid so it also selects source points
        mask = np.zeros(values.shape, dtype=bool)
        mask[5:-5, 5:-5] = True
        comparison_mask = source.copy(new_array=mask)
        method: xdem.coreg.AffineCoreg = xdem.coreg.LZD(only_translation=rotation == 0, subsample=1)
        if point_input == "ref":
            reference = reference.to_pointcloud(force_pixel_offset="center")

            # Start ICP at the known translation to isolate the effect of raster resampling
            method = xdem.coreg.ICP(only_translation=True, initial_shift=(-shift[0], -shift[1]), subsample=1)
        elif point_input == "tba":
            # Convert source pixels to points with the same measurement noise
            source = source.to_pointcloud(force_pixel_offset="center")
        original_source = source.copy()

        # Estimate residual noise without interpolating the noisy source onto a raster grid
        errors = source.estimate_error_structure(
            reference,
            coreg_method=method,
            mask=comparison_mask,
            predictors={},
            random_state=12,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Allow 2% for the fitted transformation and pixels selected near the mask boundary
        expected = np.std(noise[mask])
        assert errors.predict_magnitude() == pytest.approx(expected, rel=0.02)

        # Check unchanged source values, coordinates and method
        assert not method._fit_called
        np.testing.assert_array_equal(source.data, original_source.data)
        if point_input == "tba":
            assert source.georeferenced_coords_equal(original_source)
        else:
            assert source.transform == source_grid

    @pytest.mark.parametrize("point_input", [None, "ref", "tba"])
    def test_estimate_error_structure__precoreg(
        self,
        curved_dem: xdem.DEM,
        precoreg_case: tuple[xdem.coreg.AffineCoreg, str, bool],
        point_input: str | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that optional prealignment matches manual residual estimation."""

        # 1/ Create inputs
        # Add a vertical offset and noise to curved terrain
        coreg_method, entrypoint, precoreg = precoreg_case
        reference: Any = curved_dem
        source: Any = reference.copy(
            new_array=reference.data + 1 + np.random.default_rng(42).normal(0, 0.01, reference.shape)
        )
        mask = reference.copy(new_array=np.indices(reference.shape)[0] < 14)
        if point_input == "ref":
            reference = reference.to_pointcloud(force_pixel_offset="center")
        elif point_input == "tba":
            source = source.to_pointcloud(force_pixel_offset="center")
        original_source, original_reference = source.copy(), reference.copy()

        # 2/ Compare automatic and manual alignment
        # Estimate constant independent errors from the selected residuals
        options: dict[str, Any] = {
            "predictors": {},
            "mask": mask,
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
            "random_state": 4,
        }
        aligned, comparison, comparison_mask = source, reference, mask
        if precoreg:
            fitted = coreg_method.copy().fit(reference, source, inlier_mask=mask, random_state=4)
            if point_input == "ref":
                # Transform reference points back to the unchanged source DEM after applying the mask
                observations = reference.subsample(1, mask=mask)
                frame = xdem.coreg.apply_matrix(
                    observations.gdf,
                    fitted.to_matrix(),
                    invert=True,
                    centroid=fitted.centroid(),
                    z_name=observations.data_name,
                )
                comparison = gu.PointCloud(frame, data_name=observations.data_name)
                comparison_mask = None
            else:
                observations = (
                    source.to_pointcloud(force_pixel_offset="center") if isinstance(source, gu.Raster) else source
                )
                aligned = fitted.apply(observations)
        else:

            def fail_fit(*args: Any, **kwargs: Any) -> None:
                """Fail if precoreg=False actually calls Coreg.fit."""

                pytest.fail("Disabled precoreg must not call fit().")

            monkeypatch.setattr(type(coreg_method), "fit", fail_fit)
        sampled = aligned.cosample(comparison, mask=comparison_mask, align="reproject")
        if isinstance(sampled, gu.Raster):
            residuals = sampled.copy(new_array=sampled.data[0] - sampled.data[1])
        else:
            residuals = sampled.copy(new_array=sampled.gdf["self"].to_numpy() - sampled.gdf["other"].to_numpy())
        expected = gu.ErrorStructure.estimate(
            residuals, **{key: value for key, value in options.items() if key != "mask"}
        )

        # Run the function, method or accessor with the same alignment options
        if entrypoint == "function":
            actual = xdem.uncertainty.estimate_error_structure(
                source, reference, precoreg=precoreg, coreg_method=coreg_method, **options
            )
        else:
            target = source
            accessor_reference = reference
            if entrypoint == "accessor":
                target = source.gdf.epc if point_input == "tba" else source.to_xarray().dem
                accessor_reference = reference.gdf if point_input == "ref" else reference.to_xarray()
                accessor_mask = DataArrayRasterAccessor.from_array(mask.data.filled(False), mask.transform, mask.crs)
                options = {**options, "mask": accessor_mask}
            actual = target.estimate_error_structure(
                accessor_reference, precoreg=precoreg, coreg_method=coreg_method, **options
            )

        # 3/ Check estimates and unchanged inputs
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        assert actual.fit_diagnostics == expected.fit_diagnostics
        assert not coreg_method._fit_called
        np.testing.assert_array_equal(source.data, original_source.data)
        np.testing.assert_array_equal(reference.data, original_reference.data)
        for elevation, original in ((source, original_source), (reference, original_reference)):
            if isinstance(elevation, gu.Raster):
                assert elevation.georeferenced_grid_equal(original)
            else:
                assert elevation.georeferenced_coords_equal(original)

    @pytest.mark.parametrize(
        "reference_crs, pipeline, variable, entrypoint",
        [
            pytest.param(32632, False, False, "function", id="projected-affine-constant-function"),
            pytest.param(4326, False, True, "accessor", id="geographic-affine-variable-accessor"),
            pytest.param(32632, True, False, "accessor", id="projected-pipeline-constant-accessor"),
            pytest.param(4326, True, True, "function", id="geographic-pipeline-variable-function"),
        ],
    )
    def test_estimate_error_structure__inverse_rotation(
        self,
        curved_dem: xdem.DEM,
        reference_crs: int,
        pipeline: bool,
        variable: bool,
        entrypoint: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that inverse rotations preserve point errors, masks and source terrain predictors."""

        # 1/ Create reference points with known errors and rotations
        # Exclude raster edges so roundoff in transformed coordinates cannot select missing terrain derivatives
        source = curved_dem
        interior = np.zeros(source.shape, dtype=bool)
        interior[2:-2, 2:-2] = True
        observations = source.to_pointcloud(force_pixel_offset="center").subsample(
            1, mask=source.copy(new_array=interior)
        )
        noise = np.random.default_rng(42).normal(0, 0.5, len(observations.data))
        reference = observations.copy(new_array=observations.data + noise)

        # Use rotations about all three axes and different centroids to check inverse pipeline order
        matrix = xdem.coreg.matrix_from_translations_rotations(0.7, -0.4, 1.3, 1.0, -2.0, 3.0)
        first = xdem.coreg.AffineCoreg.from_matrix(matrix)
        first.meta["outputs"]["affine"]["centroid"] = (500_020.0, 6_999_982.0, 0.0)
        steps: list[xdem.coreg.Coreg] = [first]
        if pipeline:
            second = xdem.coreg.AffineCoreg.from_rotations(0.4, -0.6, 1.2)
            second.meta["outputs"]["affine"]["centroid"] = (500_027.0, 6_999_977.0, 2.0)
            steps.append(second)
        method = xdem.coreg.CoregPipeline(steps) if pipeline else first
        for step in steps:
            reference = step.apply(reference)
        original_values = source.data.copy()

        # Return the known transform so fitting error cannot hide interpolation of the point errors
        def use_prescribed_fit(self: xdem.coreg.Coreg, *args: Any, **kwargs: Any) -> xdem.coreg.Coreg:
            """Return the prescribed transform to isolate residual calculation from fitting errors."""

            return self

        monkeypatch.setattr(type(method), "fit", use_prescribed_fit)

        # 2/ Estimate errors with a mask defined in the reference frame
        # Select the right half of the raster before inverse transformation moves the reference points
        mask = source.copy(new_array=np.indices(source.shape)[1] >= source.shape[1] // 2)
        cutoff = source.bounds.left + source.res[0] * (source.shape[1] // 2)
        selected = reference.gdf.geometry.x.to_numpy() >= cutoff

        # Reproject the reference to check that inverse transforms use the DEM CRS
        reference = reference.reproject(crs=reference_crs)
        original_reference = reference.copy()
        options: dict[str, Any] = {
            "components": {
                "measurement": {"magnitude": "heteroscedastic" if variable else "constant", "correlation": None}
            },
            "spread_estimator": np.std,
            "bins": {"slope": np.array([0.0, 20.0, 40.0, 60.0, 90.0])} if variable else None,
            "min_count": 3,
        }
        target = source.to_xarray().dem if entrypoint == "accessor" else source
        comparison = reference.gdf if entrypoint == "accessor" else reference
        comparison_mask = mask
        if entrypoint == "accessor":
            comparison_mask = DataArrayRasterAccessor.from_array(mask.data.filled(False), mask.transform, mask.crs)
        actual = target.estimate_error_structure(
            comparison, coreg_method=method, mask=comparison_mask, predictors=("slope",) if variable else {}, **options
        )

        # 3/ Compare with the original errors and predictors at source pixel centers
        residuals = observations.copy(new_array=-noise).subsample(1, mask=selected)
        expected_predictors = {"slope": source.slope()} if variable else {}
        expected = gu.ErrorStructure.estimate(residuals, predictors=expected_predictors, **options)

        # Allow roundoff from rotations, CRS conversion and predictor interpolation
        if variable:
            actual_magnitude = actual.components["measurement"].magnitude
            expected_magnitude = expected.components["measurement"].magnitude
            pd.testing.assert_frame_equal(
                actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics, rtol=1e-7, atol=1e-8
            )
        else:
            assert actual.predict_magnitude() == pytest.approx(np.std(noise[selected]), rel=1e-8)
        assert actual.fit_diagnostics == expected.fit_diagnostics
        np.testing.assert_array_equal(source.data, original_values)
        pd.testing.assert_frame_equal(reference.gdf, original_reference.gdf)
        assert not method._fit_called

    @pytest.mark.parametrize("case", ["array", "terrain_predictors"])
    def test_estimate_error_structure__precoreg_inputs(
        self, curved_dem: xdem.DEM, coreg_method: xdem.coreg.AffineCoreg, case: str
    ) -> None:
        """Checks that array references and terrain predictors match manual alignment."""

        # 1/ Prepare inputs and fitting options
        # Create curved terrain with a vertical offset and measurement noise
        reference = curved_dem
        elevations = reference.data.copy()
        source: Any = reference.copy(
            new_array=elevations + 1 + np.random.default_rng(42).normal(0, 0.01, reference.shape)
        )
        other = reference.data if case == "array" else reference

        # Use constant magnitudes except when checking terrain predictors after alignment
        variable = case == "terrain_predictors"
        options: dict[str, Any] = {
            "components": {
                "measurement": {"magnitude": "heteroscedastic" if variable else "constant", "correlation": None}
            },
            "spread_estimator": np.std,
            "bins": 3,
            "min_count": 10,
            "random_state": 4,
        }

        # 2/ Compare automatic and manual estimation
        # Derive slope at the source pixels and transform those observations without resampling
        fitted = coreg_method.copy().fit(reference, source, random_state=4)
        observations = source.to_pointcloud(force_pixel_offset="center")
        expected_predictors = {}
        if variable:
            observations.gdf["slope"] = source.slope().data.filled(np.nan).ravel()
            expected_predictors = {"slope": "slope"}
        aligned = fitted.apply(observations)
        expected = gu.ErrorStructure.estimate(
            aligned, other=reference, other_precision="negligible", predictors=expected_predictors, **options
        )
        actual = xdem.uncertainty.estimate_error_structure(
            source, other, coreg_method=coreg_method, predictors=("slope",) if variable else {}, **options
        )

        # Check the residual spread or grouped terrain statistics
        if variable:
            expected_magnitude = expected.components["measurement"].magnitude
            actual_magnitude = actual.components["measurement"].magnitude
            pd.testing.assert_frame_equal(actual_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)
        else:
            assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        assert actual.fit_diagnostics == expected.fit_diagnostics


class TestPropagation:
    """Test module for coregistration dispatch and workflows in propagate()."""

    @pytest.mark.parametrize("point_input", [None, "ref", "tba"])
    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    def test_propagate_uncertainty_coreg__reproducibility(
        self,
        curved_dem: xdem.DEM,
        coreg_method: xdem.coreg.AffineCoreg,
        point_input: str | None,
        error_applied_to: Literal["ref", "tba"],
    ) -> None:
        """Checks that repeated coregistration simulations match across raster and point inputs."""

        # 1/ Prepare inputs
        # Create a vertical offset with raster or point inputs
        reference: Any = curved_dem
        source: Any = reference + 5
        if point_input == "ref":
            reference = reference.to_pointcloud(force_pixel_offset="center")
        elif point_input == "tba":
            source = source.to_pointcloud(force_pixel_offset="center")

        # Copy inputs to check for changes after fitting
        original_reference = reference.data.copy()
        original_source = source.data.copy()
        method = coreg_method
        # Small errors let repeated rigid fits converge within the local linear approximation
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.001)])

        # 2/ Run twice with the same seed
        result, repeated = (
            xdem.uncertainty.propagate(
                source,
                "coreg",
                error_structure={error_applied_to: structure},
                n_simulations=3,
                random_state=4,
                return_samples=True,
                return_coregs=True,
                inputs={"coreg": {"reference_elev": reference}},
                operation_options={
                    "coreg_method": method,
                },
            )
            for _ in range(2)
        )
        assert isinstance(result.estimate, pd.Series)
        assert isinstance(result.mean, pd.Series)
        assert isinstance(result.std, pd.Series)
        assert isinstance(result.n_valid, pd.Series)
        assert isinstance(repeated.mean, pd.Series)
        assert isinstance(repeated.std, pd.Series)

        # 3/ Compare results
        # Allow numerical fitting tolerance on the known vertical offset
        assert result.estimate["tz"] == pytest.approx(-5, abs=1e-4)
        assert result.mean["tz"] == pytest.approx(-5, abs=0.1)
        assert "centroid" in result.output

        # Check equal means and uncertainties
        pd.testing.assert_series_equal(result.mean, repeated.mean)
        pd.testing.assert_series_equal(result.std, repeated.std)

        # Check all three fits contribute to summaries and samples
        assert result.n_success == result.n_simulations == 3
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

    @pytest.mark.parametrize("error_applied_to", ["ref", "tba"])
    def test_coregistration_uncertainty_workflow(
        self, curved_dem: xdem.DEM, coreg_method: xdem.coreg.AffineCoreg, error_applied_to: Literal["ref", "tba"]
    ) -> None:
        """Checks that errors estimated after alignment give uncertainty on the original transformation."""

        # 1/ Fit and estimate errors
        # Create curved terrain with a vertical offset and measurement noise
        reference = curved_dem
        elevations = reference.data.copy()
        # Small errors let repeated rigid fits converge within the local linear approximation
        noise = np.random.default_rng(42).normal(0, 0.001, reference.shape)
        source = reference.copy(new_array=elevations + 1 + noise)

        # Estimate independent errors from aligned residuals on the selected dataset
        fitted = coreg_method.copy().fit(reference, source, random_state=42)
        aligned = fitted.apply(source, resample=True)
        uncertain, other = (reference, aligned) if error_applied_to == "ref" else (aligned, reference)
        errors = xdem.uncertainty.estimate_error_structure(
            uncertain,
            other=other,
            predictors={},
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
            precoreg=False,
        )

        # Check magnitude against finite residuals, excluding borders masked during resampling
        residuals = (aligned - reference).data.compressed()
        assert errors.predict_magnitude() == pytest.approx(np.std(residuals))
        assert errors.predict_magnitude() > 0

        # 2/ Propagate on the original pair
        options: dict[str, Any] = {
            "error_structure": {error_applied_to: errors},
            "n_simulations": 3,
            "random_state": 42,
            "return_samples": True,
            "return_coregs": True,
        }
        direct = xdem.uncertainty.propagate_uncertainty_coreg(reference, source, coreg_method, **options)
        generic = xdem.uncertainty.propagate(
            source,
            "coreg",
            inputs={"coreg": {"reference_elev": reference}},
            operation_options={"coreg_method": coreg_method},
            **options,
        )

        # 3/ Compare results
        # The estimate describes the original offset and both entry points give the same simulations
        assert direct.estimate["tz"] == pytest.approx(-1, abs=0.05)
        assert direct.n_success == generic.n_success == 3
        assert direct.error_structure[error_applied_to] is errors
        assert np.isfinite(direct.std).all()
        assert direct.std["tz"] > 0
        pd.testing.assert_series_equal(direct.estimate, generic.estimate)
        pd.testing.assert_series_equal(direct.mean, generic.mean)
        pd.testing.assert_series_equal(direct.std, generic.std)
        assert direct.samples is not None and generic.samples is not None
        pd.testing.assert_frame_equal(direct.samples, generic.samples)

        # Check unchanged method and input elevations
        assert not coreg_method._fit_called
        np.testing.assert_array_equal(reference.data, elevations)
        np.testing.assert_array_equal(source.data, elevations + 1 + noise)


class TestPropagationAccessors:
    """Test module for propagation outputs and custom calculations in each in-memory object family."""

    @pytest.mark.parametrize("point_input", [False, True])
    @pytest.mark.parametrize("method", ["analytical", "numerical"])
    def test_propagate__source_family(self, point_input: bool, method: str) -> None:
        """Checks that propagation preserves accessor output families and matches GeoUtils object values."""

        # Create equivalent GeoUtils and accessor inputs
        raster = gu.Raster.from_array(np.arange(30.0).reshape(5, 6), Affine(2, 0, 0, 0, -2, 10), 32632)
        source = raster.to_pointcloud() if point_input else raster
        target = source.gdf.copy() if point_input else raster.to_xarray()
        if point_input:
            target.attrs["data_name"] = source.data_name
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        options: dict[str, Any] = {"error_structure": errors, "method": method, "at": "all"}
        if method == "numerical":
            options.update(n_simulations=3, random_state=42, return_samples=True)

        # Propagate the same error model
        expected = xdem.uncertainty.propagate(source, **options)
        actual = xdem.uncertainty.propagate(target, **options)

        # Check output types, values and source data
        assert actual.estimate is target
        for name in ("estimate", "mean", "std"):
            result = getattr(actual, name)
            assert isinstance(result, gpd.GeoDataFrame if point_input else xr.DataArray)
            interface = _get_pointcloud_interface(result) if point_input else _get_raster_interface(result)
            np.testing.assert_array_equal(np.asarray(interface.data).squeeze(), getattr(expected, name).data)
            assert interface.crs == source.crs
        pd.testing.assert_frame_equal(actual.selection, expected.selection)
        if method == "numerical":
            assert actual.samples is not None and expected.samples is not None
            pd.testing.assert_frame_equal(actual.samples, expected.samples)
        np.testing.assert_array_equal(source.data, np.arange(30.0) if point_input else np.arange(30.0).reshape(5, 6))

    @pytest.mark.parametrize("point_input", [False, True])
    def test_propagate__custom_family(self, point_input: bool) -> None:
        """Checks that custom calculations receive the same object family for original and perturbed values."""

        # Create equivalent GeoUtils and accessor inputs
        raster = gu.Raster.from_array(np.arange(30.0).reshape(5, 6), Affine(2, 0, 0, 0, -2, 10), 32632)
        source = raster.to_pointcloud() if point_input else raster
        target = source.gdf.copy() if point_input else raster.to_xarray()
        if point_input:
            target.attrs["data_name"] = source.data_name
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Record the inputs received by the calculation
        calls = []

        def calculate(value: Any) -> Any:
            """Return the elevations and record the received container type."""

            calls.append(type(value))
            return value

        # Propagate through the custom calculation
        actual = xdem.uncertainty.propagate(
            target, calculate, error_structure=errors, n_simulations=3, random_state=42, at="all", return_samples=True
        )
        expected = xdem.uncertainty.propagate(
            source, error_structure=errors, n_simulations=3, random_state=42, at="all", return_samples=True
        )

        # Check one original call and three perturbed calls
        assert calls == [type(target)] * 4
        for name in ("estimate", "mean", "std"):
            result = getattr(actual, name)
            interface = _get_pointcloud_interface(result) if point_input else _get_raster_interface(result)
            np.testing.assert_array_equal(np.asarray(interface.data).squeeze(), getattr(expected, name).data)
        assert actual.samples is not None and expected.samples is not None
        pd.testing.assert_frame_equal(actual.samples, expected.samples)


@pytest.fixture
def terrain_dem_paths(tmp_path: Path) -> tuple[Path, ...]:
    """Write curved DEMs with seeded measurement noise for terrain error estimation."""

    # Create curved terrain with random errors
    rng = np.random.default_rng(32)
    y, x = np.mgrid[:24, :24]
    elevation = 0.3 * x + 0.5 * y + 0.04 * x * y
    transform = Affine(10, 0, 0, 0, -10, 240)
    source = xdem.DEM(gu.Raster.from_array(elevation + rng.normal(size=x.shape), transform, 32632, nodata=-9999))
    reference = xdem.DEM(gu.Raster.from_array(elevation, transform, 32632, nodata=-9999))

    paths = tuple(tmp_path / f"input-{index}.tif" for index in range(2))
    for raster, path in zip(
        (
            source,
            reference,
        ),
        paths,
    ):
        raster.to_file(path)
    return paths


@pytest.fixture
def error_proxy_paths(tmp_path: Path, backend: str) -> tuple[Path, Path]:
    """Write a raster or point error proxy and a flat DEM for optional prealignment."""

    if backend == "dask_geopandas":
        pytest.importorskip("pyarrow")
        columns, rows = np.meshgrid(np.arange(7), np.arange(5))
        proxy = xdem.EPC(gu.PointCloud.from_xyz(columns.ravel(), rows.ravel(), np.linspace(-1, 1, 35), crs=32632))
        path = tmp_path / "proxy.parquet"
        proxy.to_parquet(str(path))
    else:
        values = np.arange(63.0).reshape(7, 9)
        raster = DEM.from_array(values, Affine(2, 0, 0, 0, -2, 14), 32632, nodata=-9999)
        path = tmp_path / "proxy.tif"
        raster.to_file(path)
    reference = DEM.from_array(np.zeros((7, 9)), Affine(1, 0, -1, 0, -1, 6), 32632, nodata=-9999)
    reference_path = tmp_path / "reference.tif"
    reference.to_file(reference_path)
    return path, reference_path


class TestElevationErrorEstimationChunked:
    """Test module for elevation error estimation with Dask and multiprocessing."""

    @pytest.mark.parametrize(
        "backend, precoreg, override",
        [
            ("dask", None, False),
            ("multiproc", None, True),
            ("dask_geopandas", None, False),
            ("dask_geopandas", False, True),
            ("dask_geopandas", True, True),
        ],
    )
    def test_estimate_error_structure__chunk_invariance_proxy(
        self, backend: str, precoreg: bool | None, override: bool, error_proxy_paths: tuple[Path, Path]
    ) -> None:
        """Checks that raster and point proxies match in-memory error estimates without loading their inputs."""

        # 1/ Open the proxy and configure optional prealignment
        path, reference_path = error_proxy_paths
        reference = DEM(reference_path, load_data=True) if precoreg is not None else None
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
        }
        if override:
            options["spread_estimator"] = np.std
        if precoreg is not None:
            options.update(predictors={}, precoreg=precoreg)
            # A flat reference constrains only the vertical shift
            if precoreg:
                options["coreg_method"] = xdem.coreg.VerticalShift()

        # Calculate the in-memory reference
        proxy: Any = xdem.EPC(path) if backend == "dask_geopandas" else DEM(path)
        proxy.load()
        expected = proxy.estimate_error_structure(reference, **options)

        # 2/ Estimate errors from the chunked proxy
        source: Any
        if backend in ("dask", "dask_geopandas"):
            # 2.1/ If using Dask
            if backend == "dask_geopandas":
                pytest.importorskip("dask_geopandas")
                # Split 35 points into three uneven partitions
                source = xdem.open_epc(path, chunks=12).epc
            else:
                pytest.importorskip("dask")
                source = open_dem(path, chunks={"y": 3, "x": 4}).dem
            other = open_dem(reference_path) if reference is not None else None
            assert not source.is_loaded
            actual = source.estimate_error_structure(other, **options)
            assert not source.is_loaded
        else:
            # 2.2/ If using MP
            source = DEM(path)
            config = MultiprocConfig(chunks=(3, 4))
            other = open_dem(reference_path) if reference is not None else None
            assert not source.is_loaded
            actual = source.estimate_error_structure(other, mp_config=config, **options)
            assert not source.is_loaded

        # 3/ Check equality of magnitudes and input loading
        assert type(actual) is gu.ErrorStructure
        # Allow rounding from reductions across chunks
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude(), rel=0, abs=1e-12)
        assert proxy.is_loaded
        assert reference is None or reference.is_loaded

    @pytest.mark.parametrize("backend, override, precoreg", [("dask", False, False), ("multiproc", True, True)])
    def test_estimate_error_structure__chunk_invariance_terrain_predictors(
        self, backend: str, override: bool, precoreg: bool, terrain_dem_paths: tuple[Path, ...]
    ) -> None:
        """Checks that Dask/MP terrain predictors give the same grouped statistics as in-memory inputs."""

        # 1/ Open DEMs and configure terrain predictors
        source, reference = (DEM(path, load_data=True) for path in terrain_dem_paths)
        # Set magnitude estimation options
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 3,
            "min_count": 10,
            "random_state": 4,
        }

        # These statistics checks need only a vertical correction on their smooth terrain
        if precoreg:
            options["coreg_method"] = xdem.coreg.VerticalShift()
        if override:
            options["spread_estimator"] = np.std
        expected = source.estimate_error_structure(reference, precoreg=precoreg, **options)

        # 2/ Estimate errors from the chunked DEMs
        if backend == "dask":
            # 2.1/ If using Dask
            pytest.importorskip("dask")
            # Use different layouts and shorter final chunks
            chunks = ((6, 8), (8, 6))
            chunked_source, chunked_reference = (
                open_dem(path, chunks={"y": rows, "x": columns}).dem
                for path, (rows, columns) in zip(terrain_dem_paths, chunks)
            )
            mp_config = None
            assert not chunked_source.is_loaded and not chunked_reference.is_loaded
            result = xdem.uncertainty.estimate_error_structure(
                chunked_source, chunked_reference, mp_config=mp_config, precoreg=precoreg, **options
            )
            assert not chunked_source.is_loaded and not chunked_reference.is_loaded
        else:
            # 2.2/ If using MP
            chunked_source, chunked_reference = (DEM(path) for path in terrain_dem_paths)
            mp_config = MultiprocConfig(chunks=(8, 8))
            assert not chunked_source.is_loaded and not chunked_reference.is_loaded
            result = xdem.uncertainty.estimate_error_structure(
                chunked_source, chunked_reference, mp_config=mp_config, precoreg=precoreg, **options
            )
            assert not chunked_source.is_loaded and not chunked_reference.is_loaded

        # 3/ Check equality of grouped statistics and input loading
        magnitude = result.components["measurement"].magnitude
        in_memory_magnitude = expected.components["measurement"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert isinstance(in_memory_magnitude, gu.ErrorMagnitude)
        assert result.empirical_variogram is None
        assert isinstance(magnitude.grouped_statistics, pd.DataFrame)

        # Allow rounding from reductions across chunks
        pd.testing.assert_frame_equal(
            magnitude.grouped_statistics, in_memory_magnitude.grouped_statistics, check_exact=False, rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("point_input", [None, "ref", "tba"])
    def test_estimate_error_structure__chunk_invariance_precoreg(
        self, curved_dem: xdem.DEM, point_input: str | None
    ) -> None:
        """Checks that Dask prealignment matches in-memory estimation without loading rasters or point clouds."""

        pytest.importorskip("dask")

        # 1/ Prepare elevations with a height shift and measurement noise
        reference: Any = curved_dem
        source: Any = reference.copy(
            new_array=reference.data + 1 + np.random.default_rng(42).normal(0, 0.01, reference.shape)
        )
        if point_input == "ref":
            reference = reference.to_pointcloud(force_pixel_offset="center")
        elif point_input == "tba":
            source = source.to_pointcloud(force_pixel_offset="center")
        options: dict[str, Any] = {
            "predictors": {},
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
            "random_state": 4,
        }

        # 2/ Estimate errors with automatic prealignment
        expected = xdem.uncertainty.estimate_error_structure(source, reference, **options)

        # Use different raster chunks and uneven point partitions
        lazy = []
        for index, elevation in enumerate((source, reference)):
            array = elevation.to_xarray()
            chunks = {"x": 7 + index, "y": 5 + index} if isinstance(elevation, gu.Raster) else {array.dims[0]: 37}
            lazy.append(array.chunk(chunks))
        actual = xdem.uncertainty.estimate_error_structure(*lazy, **options)

        # 3/ Check equality of magnitudes and unchanged inputs
        # Allow rounding from reductions across chunks
        assert actual.predict_magnitude() == pytest.approx(expected.predict_magnitude(), rel=0, abs=1e-12)
        for array in lazy:
            interface = _get_raster_interface(array)
            if interface is None:
                interface = _get_pointcloud_interface(array)
            assert not interface.is_loaded


class TestElevationErrorEstimationErrors:
    """Test module for grid validation during error estimation."""

    @pytest.mark.parametrize("predictors", [("slope",), {"slope": "slope"}])
    def test_estimate_error_structure__error_proxy_predictors(self, predictors: Any) -> None:
        """Checks an error is raised for terrain attribute names without elevation inputs."""

        # Create a proxy whose values describe errors, not terrain
        proxy = xdem.DEM.from_array(np.ones((5, 7)), Affine(2, 0, 0, 0, -2, 10), 32632, nodata=-9999)

        # Require predictor values derived from actual elevations
        with pytest.raises(
            ValueError, match="Terrain predictors for an error proxy must be input as arrays or rasters"
        ):
            proxy.estimate_error_structure(predictors=predictors)

    def test_estimate_error_structure__error_mismatched_grids(self) -> None:
        """Checks an error is raised for mismatched grids when alignment is disabled."""

        # Create grids with different resolutions
        source = xdem.DEM.from_array(
            np.arange(36, dtype=float).reshape(6, 6), Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999
        )
        reference = xdem.DEM.from_array(np.zeros((3, 3)), Affine(20, 0, 0, 0, -20, 60), 32632, nodata=-9999)

        # Check the error with alignment disabled
        with pytest.raises(ValueError, match="does not share the selected support grid"):
            source.estimate_error_structure(other=reference, predictors={}, align="raise", precoreg=False)

    def test_estimate_error_structure__error_coreg_method(self) -> None:
        """Checks an error is raised for a coregistration method that is not a Coreg instance."""

        # Create a raster and supply an invalid method
        source = xdem.DEM.from_array(np.ones((5, 7)), Affine(2, 0, 0, 0, -2, 10), 32632, nodata=-9999)
        options: dict[str, Any] = {"coreg_method": "NuthKaab"}

        # Check the method type
        with pytest.raises(TypeError, match="coreg_method must be a Coreg instance or None"):
            xdem.uncertainty.estimate_error_structure(source, source, **options)

    def test_estimate_error_structure__error_mixed_precoreg(self) -> None:
        """Checks an error is raised for prealignment with mixed raster families before loading values."""

        pytest.importorskip("dask")

        # Split a raster into uneven chunks and preserve its lazy array
        source = xdem.DEM.from_array(np.ones((9, 11)), Affine(2, 0, 0, 0, -2, 18), 32632, nodata=-9999)
        lazy = source.to_xarray().chunk({"x": 4, "y": 5})
        assert not lazy.rst.is_loaded

        with pytest.raises(TypeError, match="Cannot mix Raster/PointCloud"):
            xdem.uncertainty.estimate_error_structure(lazy, source)
        assert not lazy.rst.is_loaded


class TestPropagationErrors:
    """Test module for unsupported analytical fits and explicit numerical sample selection."""

    @pytest.mark.parametrize("with_source", [False, True])
    def test_propagate__error_custom_analytical(self, with_source: bool, tmp_path: Path) -> None:
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
            xdem.uncertainty.propagate(
                source,
                np.mean,
                error_structure=errors,
                predictors={"slope": "slope"},
                method="analytical",
            )

        # Check the raster stays unloaded
        if source is not None:
            assert not source.is_loaded

    @pytest.mark.parametrize("kind", ["array", "raster", "point"])
    def test_propagate__error_lazy_source(self, kind: str) -> None:
        """Checks an error is raised for Dask source propagation before any values are computed."""

        pytest.importorskip("dask")

        import dask.array as da

        # Create a lazy array, raster or point cloud
        raster = gu.Raster.from_array(np.ones((5, 7)), Affine(2, 0, 0, 0, -2, 10), 32632)
        if kind == "array":
            source = da.from_array(raster.data.filled(np.nan), chunks=(3, 4))
        elif kind == "raster":
            source = raster.to_xarray().chunk({"y": 3, "x": 4})
        else:
            source = raster.to_pointcloud().to_xarray()
            source = source.chunk({source.dims[0]: 4})
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Check unsupported Dask inputs
        with pytest.raises(NotImplementedError, match="Dask propagation for this operation"):
            xdem.uncertainty.propagate(source, error_structure=errors)

        # Check input loading
        if kind == "array":
            assert isinstance(source, da.Array)
        else:
            interface = _get_raster_interface(source) if kind == "raster" else _get_pointcloud_interface(source)
            assert not interface.is_loaded

    def test_propagate__error_sample_selection(self) -> None:
        """Checks an error is raised when raster simulation draws are requested without selecting pixels."""

        # Create a raster and error model
        raster = gu.Raster.from_array(np.ones((5, 7)), Affine(1, 0, 0, 0, -1, 5), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Check saving samples requires selected pixels
        with pytest.raises(ValueError, match="Select outputs with at"):
            xdem.uncertainty.propagate(raster, error_structure=errors, return_samples=True)

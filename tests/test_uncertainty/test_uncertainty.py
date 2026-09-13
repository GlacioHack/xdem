"""Tests for public elevation uncertainty estimation and propagation workflows."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
from affine import Affine

import xdem


class TestElevationErrorEstimation:
    """Public estimation methods on DEMs and elevation point clouds."""

    def test_elevation_methods_apply_error_attribution_on_common_support(self) -> None:
        """Checks that DEM and EPC estimation apply the requested error attribution and preserve spatial types."""

        # Create a zero reference and a source whose elevation differences contain only random error
        rng = np.random.default_rng(8)
        transform = Affine(10, 0, 0, 0, -10, 240)
        reference = xdem.DEM(gu.Raster.from_array(np.zeros((24, 24)), transform, 32632, nodata=-9999))
        source = xdem.DEM(gu.Raster.from_array(rng.normal(size=(24, 24)), transform, 32632, nodata=-9999))

        # Use independent constant errors to isolate attribution from correlation fitting
        components = {"measurement": {"magnitude": "constant", "correlation": None}}

        # Estimate source errors assuming either negligible or equal errors in the reference
        negligible = source.estimate_error_structure(reference, predictors={}, components=components)
        same = source.estimate_error_structure(reference, predictors={}, components=components, other_error="same")

        # Repeat estimation on point elevations and evaluate both magnitudes and random fields
        points = source.to_pointcloud(subsample=250, random_state=3)
        point_structure = points.estimate_error_structure(reference, predictors={}, components=components)
        point_magnitude = point_structure.predict_magnitude(like=points)
        point_error = point_structure.generate_random_field(points, random_state=4)

        # Equal independent error contributions each have half the variance of their difference
        assert same.predict_magnitude() == pytest.approx(negligible.predict_magnitude() / np.sqrt(2))

        # Check that the selected locations and output types follow the elevation input
        assert negligible.metadata["support"] == "raster"
        assert point_structure.metadata["support"] == "pointcloud"
        assert isinstance(point_magnitude, xdem.EPC)
        assert isinstance(point_error, xdem.EPC)

    def test_deprecated_dem_uncertainty_adapter(self) -> None:
        """Checks that the deprecated DEM method warns and returns its documented magnitude and correlation outputs."""

        # Require the optional backend used to fit spatial correlation
        pytest.importorskip("skgstat")

        # Load the same example pair used by the public estimation workflow
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))

        # Check that the legacy call directs the user to the migration guide
        with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
            magnitude, correlation = source.estimate_uncertainty(reference, random_state=42)

        # Check the magnitude grid and unit correlation at coincident locations
        assert isinstance(magnitude, gu.Raster)
        assert magnitude.georeferenced_grid_equal(source)
        assert correlation(0) == pytest.approx(1)

    @pytest.mark.skipif(find_spec("skgstat") is not None, reason="Only runs if scikit-gstat is missing.")
    def test_estimate_error_structure_missing_variography_dependency(self) -> None:
        """Checks that correlated estimation reports a clear import error when scikit-gstat is unavailable."""

        # Load a valid DEM pair so the missing backend is the only intended failure
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))

        # Check that the public API names the optional dependency needed by correlation fitting
        with pytest.raises(ImportError, match="Optional dependency 'scikit-gstat' required"):
            source.estimate_error_structure(reference)


class TestCoregistrationPropagation:
    """Simulation of fitted coregistration parameters and failure handling."""

    def test_error_structure_propagates_through_coregistration(self) -> None:
        """Checks that an ErrorStructure produces translation and rotation summaries through coregistration."""

        # Load the example DEM pair and assign a small independent measurement error
        reference = xdem.DEM(xdem.examples.get_path_test("longyearbyen_ref_dem"))
        source = xdem.DEM(xdem.examples.get_path_test("longyearbyen_tba_dem"))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.5)])

        # Run two realizations so the returned parameter spread is defined
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            summary, simulations, fitted = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                source,
                xdem.coreg.LZD(),
                nsim=2,
                error_structure=structure,
                random_state=5,
            )

        # Check the six transformation parameters and retain one fitted object per realization
        assert summary.shape == (6, 2)
        assert len(simulations) == len(fitted) == 2

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

        # Use a fixed seed for both realizations and fitting, with optional initial alignment
        options = dict(
            reference_elev=reference,
            coreg_method=method,
            error_structure=structure,
            nsim=3,
            error_applied_to=error_applied_to,
            precoreg=precoreg,
            random_state=4,
        )

        # Repeat the complete workflow with identical settings
        result = xdem.uncertainty.propagate_uncertainty(source, "coreg", **options)
        repeated = xdem.uncertainty.propagate_uncertainty(source, "coreg", **options)

        # Initial alignment removes the five-unit offset before residual simulations
        expected_shift = 0 if precoreg else -5
        assert result.estimate["tz"] == pytest.approx(expected_shift)
        assert result.mean["tz"] == pytest.approx(expected_shift, abs=0.1)

        # Check reproducibility of the ensemble mean and spread
        pd.testing.assert_series_equal(result.mean, repeated.mean)
        pd.testing.assert_series_equal(result.std, repeated.std)

        # Check that all requested fits contribute to each parameter summary and retained table
        assert result.n_success == result.nsim == 3
        assert result.n_valid.eq(3).all()
        assert len(result.metadata["fitted_coregs"]) == 3
        assert result.metadata["simulation_table"].nsim.tolist() == [1, 2, 3]

        # Check that fitting copies preserve the original method and both elevation datasets
        assert "shift_z" not in method.meta["outputs"].get("affine", {})
        np.testing.assert_array_equal(reference.data, original_reference)
        np.testing.assert_array_equal(source.data, original_source)

    def test_coreg_propagation_counts_failed_fits(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            summary, simulations, fitted = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                reference + 5,
                method,
                error_structure=structure,
                nsim=3,
                random_state=4,
            )

        # Check that success counts and retained simulation numbers exclude the failed first realization
        assert summary.attrs == {"nsim": 3, "n_success": 2, "frac_success": 2 / 3}
        assert simulations.nsim.tolist() == [2, 3]
        assert len(fitted) == 2

        # Compare the reported spread with the sample standard deviation of successful fits only
        np.testing.assert_allclose(summary["std"], simulations[["tx", "ty", "tz", "rx", "ry", "rz"]].std())

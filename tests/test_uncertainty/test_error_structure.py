"""Tests for error components, magnitude models and their combined covariance."""

from __future__ import annotations

from importlib.util import find_spec

import geoutils as gu
import numpy as np
import pandas as pd
import pytest

import xdem


class TestErrorStructure:
    """Combination of component magnitudes and spatial correlations."""

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_error_components_separate_magnitude_and_correlation(self) -> None:
        """Checks that component variances add correctly and their correlation models retain unit variance."""

        # Combine two correlated components and one independent component with known magnitudes
        short = xdem.ErrorComponent("short", 2, gu.VariogramModel("gaussian", 10, 3))
        long = xdem.ErrorComponent("long", 1, gu.VariogramModel("spherical", 100, 2))
        independent = xdem.ErrorComponent("independent", 0.5)
        structure = xdem.ErrorStructure([short, long, independent])

        # Check that input variogram sills are normalized independently of the assigned magnitudes
        assert short.correlation is not None and short.correlation.sill == 1
        assert long.correlation is not None and long.correlation.sill == 1

        # The total variance is the sum of squared magnitudes: 2**2 + 1**2 + 0.5**2
        assert structure.predict_variance() == pytest.approx(5.25)
        assert structure.predict_correlation(0) == pytest.approx(1)
        assert "short" in repr(structure) and "long" in repr(structure)

        # Compare pairwise and matrix covariance through the same component equation
        coordinates = np.array([[0, 0], [10, 0]], dtype=float)
        covariance = structure.to_covariance_matrix(coordinates)
        assert covariance.shape == (2, 2)
        assert covariance == pytest.approx(covariance.T)
        assert np.diag(covariance) == pytest.approx(np.full(2, 5.25))
        assert covariance[0, 1] == pytest.approx(structure.predict_covariance(10))

    def test_grouped_component_preserves_total_local_magnitude(self) -> None:
        """Checks that allocating fixed variance to a separate component preserves the total local magnitude."""

        # Define two reliable bins whose total error magnitudes are one and two
        intervals = pd.IntervalIndex.from_breaks([0, 1, 2], name="slope")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        statistics = pd.DataFrame([[1.0, 100], [2.0, 100]], index=intervals, columns=columns)

        # Remove the fixed component variance from the grouped magnitude before combining them
        variable = xdem.ErrorMagnitude.grouped(statistics, variance_offset=0.25)
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent("variable", variable),
                xdem.ErrorComponent("fixed", 0.5),
            ]
        )

        # Evaluate at the bin centers, where the fitted totals must equal the input statistics
        predictors = {"slope": np.array([0.5, 1.5])}
        assert structure.predict_magnitude(predictors) == pytest.approx([1, 2])
        assert structure.predict_magnitude(predictors, component="variable") == pytest.approx(
            np.sqrt(np.array([1, 4]) - 0.25)
        )

        # Check that the covariance diagonal contains the same total local variances
        covariance = structure.to_covariance_matrix(np.array([[0, 0], [1, 0]]), predictors=predictors)
        assert np.diag(covariance) == pytest.approx([1, 4])

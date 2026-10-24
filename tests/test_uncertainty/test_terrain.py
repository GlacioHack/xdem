"""Tests for analytical and numerical uncertainty propagation through terrain attributes."""

from __future__ import annotations

from typing import Any

import geoutils as gu
import numpy as np
import pytest
from affine import Affine
from geoutils import Raster
from geoutils.stats.variography import VariogramModel

import xdem


class TestTerrainPropagation:
    """Test module for terrain filter derivatives and analytical uncertainty at flat terrain."""

    @pytest.mark.parametrize("surface_fit", ["Horn", "ZevenbergThorne", "Florinsky"])
    @pytest.mark.parametrize("attribute", ["slope", "aspect"])
    @pytest.mark.parametrize("degrees", [False, True])
    @pytest.mark.parametrize("correlated", [False, True])
    def test_propagate_uncertainty__finite_differences(
        self, surface_fit: str, attribute: str, degrees: bool, correlated: bool
    ) -> None:
        """Checks that analytical terrain uncertainty matches finite differences."""

        # 1/ Run analytical propagation
        # Create curved terrain with defined slope and aspect derivatives
        rows, columns = np.indices((11, 13))
        values = columns / 2 + rows / 3 + np.sin(columns / 4)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 22), 32632)
        model = VariogramModel("gaussian", effective_range=5, partial_sill=1) if correlated else None
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.02, correlation=model)])
        options: dict[str, Any] = {
            "attribute": attribute,
            "surface_fit": surface_fit,
            "degrees": degrees,
            "engine": "scipy",
        }

        # Select an interior pixel for comparison
        summary = xdem.uncertainty.propagate_uncertainty(
            raster,
            "terrain",
            error_structure=errors,
            operation_options=options,
            method="analytical",
            at=[(5, 6)],
        )

        # 2/ Calculate finite differences
        # Use 3 x 3 windows, or 5 x 5 for Florinsky
        radius = 2 if surface_fit == "Florinsky" else 1
        locations = [(row, col) for row in range(5 - radius, 6 + radius) for col in range(6 - radius, 7 + radius)]
        derivatives = []
        epsilon = 1e-4
        for row, col in locations:
            plus, minus = values.copy(), values.copy()
            plus[row, col] += epsilon
            minus[row, col] -= epsilon
            upper = xdem.terrain.get_terrain_attribute(raster.copy(new_array=plus), out_dtype=np.float64, **options)
            lower = xdem.terrain.get_terrain_attribute(raster.copy(new_array=minus), out_dtype=np.float64, **options)
            derivatives.append((upper.data[5, 6] - lower.data[5, 6]) / (2 * epsilon))

        # 3/ Compare uncertainties
        # Combine finite differences with observation covariance
        x, y = raster.ij2xy(*np.asarray(locations).T)
        covariance = errors.to_covariance_matrix(np.column_stack((x, y)))
        jacobian = np.asarray(derivatives)
        expected_std = np.sqrt(jacobian @ covariance @ jacobian)

        # Check uncertainty and unchanged elevations
        assert summary.std.data[5, 6] == pytest.approx(expected_std, rel=1e-6)
        assert summary.metadata["approximation"] == "first_order"
        np.testing.assert_array_equal(raster.data, values)

    def test_propagate_uncertainty__flat(self) -> None:
        """Checks that flat slope has undefined analytical uncertainty and positive numerical uncertainty."""

        # Create flat terrain where the slope derivative is undefined
        raster = gu.Raster.from_array(np.ones((7, 9)), Affine(2, 0, 0, 0, -2, 14), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        options = {"attribute": "slope", "engine": "scipy"}

        # Run analytical and numerical propagation
        analytical = xdem.uncertainty.propagate_uncertainty(
            raster,
            "terrain",
            error_structure=errors,
            method="analytical",
            operation_options=options,
        )
        numerical = xdem.uncertainty.propagate_uncertainty(
            raster,
            "terrain",
            error_structure=errors,
            operation_options=options,
            n_samples=4,
            random_state=2,
        )

        # Check simulations give positive slope and uncertainty
        assert analytical.std.get_mask()[3, 4]
        assert numerical.mean.data[3, 4] > 0
        assert numerical.std.data[3, 4] > 0

    def test_propagate_uncertainty__slope_mean(self) -> None:
        """Checks that slope simulations preserve the DEM grid and give a positive mean on flat terrain."""

        # Create flat terrain
        values = np.ones((10, 12))
        transform = Affine(10, 0, 0, 0, -10, 100)
        dem = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.5)])

        # Save slope samples from four simulations
        result = xdem.uncertainty.propagate_uncertainty(
            dem,
            "terrain",
            error_structure=structure,
            operation_options={"attribute": "slope", "engine": "scipy"},
            n_samples=4,
            random_state=3,
            at="all",
            return_samples=True,
        )

        # Check raster outputs
        assert isinstance(result.estimate, Raster)
        assert isinstance(result.mean, Raster)
        assert isinstance(result.std, Raster)

        # Check the original grid and positive mean slope
        assert result.std.georeferenced_grid_equal(dem)
        assert np.nanmean(result.estimate.data) == pytest.approx(0)
        assert np.nanmean(result.mean.data) > 0

        # Compare uncertainty with saved samples
        assert result.samples is not None
        selected_std = np.asarray(result.std.data).reshape(-1)[result.selection["flat_index"]]
        np.testing.assert_allclose(selected_std, result.samples.std(axis=0, ddof=1), atol=1e-12)

        # Check unchanged elevations
        assert np.all(dem.data == 1)

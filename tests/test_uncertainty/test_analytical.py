"""Tests for effective sample sizes and analytical uncertainty of spatial averages."""

from __future__ import annotations

from importlib.util import find_spec

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

import xdem
from xdem.uncertainty import analytical


class TestEffectiveSampleSize:
    """Exact and approximate integration of spatial correlation models."""

    @pytest.mark.parametrize("range1", [10**i for i in range(3)])
    @pytest.mark.parametrize("psill1", [0.1, 1, 10])
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.parametrize("area", [10 ** (2 * i) for i in range(3)])
    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_circular_single_range(self, range1: float, psill1: float, model1: str, area: float) -> None:
        """Checks that radial quadrature matches the circular formula for each supported correlation model."""

        # Convert the model partial sill to an error magnitude while keeping unit correlation variance
        structure = xdem.ErrorStructure(
            [xdem.ErrorComponent("error", np.sqrt(psill1), gu.VariogramModel(model1, range1, 1))]
        )

        # Evaluate the closed formula for the equivalent circular area
        neff_circ_exact = analytical.neff_circular_approx_theoretical(area=area, error_structure=structure)
        # Evaluate the same radial integral by numerical quadrature
        neff_circ_numer = analytical.neff_circular_approx_numerical(area=area, error_structure=structure)

        # Allow a 0.1 percent tolerance for numerical quadrature
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.parametrize("range1", [10**i for i in range(2)])
    @pytest.mark.parametrize("range2", [10**i for i in range(2)])
    @pytest.mark.parametrize("range3", [10**i for i in range(2)])
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.parametrize("model2", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_circular_three_ranges(
        self, range1: float, range2: float, range3: float, model1: str, model2: str
    ) -> None:
        """Checks that radial quadrature and circular formulas agree for the sum of three correlation models."""

        # Use equal component variances to compare combinations of short and long correlation ranges
        area = 1000
        psill1 = 1
        psill2 = 1
        psill3 = 1
        model3 = "spherical"

        # Represent the three independent contributions as separate error components
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent(f"range{i}", np.sqrt(sill), gu.VariogramModel(model, scale, 1))
                for i, (model, scale, sill) in enumerate(
                    zip([model1, model2, model3], [range1, range2, range3], [psill1, psill2, psill3])
                )
            ]
        )

        # Evaluate the closed formula for the equivalent circular area
        neff_circ_exact = analytical.neff_circular_approx_theoretical(area=area, error_structure=structure)
        # Evaluate the same radial integral by numerical quadrature
        neff_circ_numer = analytical.neff_circular_approx_numerical(area=area, error_structure=structure)

        # Allow a 0.1 percent tolerance for numerical quadrature
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_exact_and_approx_hugonnet(self) -> None:
        """Checks that sampled covariance sums approximate the exact effective sample size within ten percent."""

        # Create a regular grid with the same error magnitude at every location
        shape = (15, 15)
        errors = np.ones(shape)

        # Locate each observation on a grid with unit spacing
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        xx, yy = np.meshgrid(x, y)

        # Store one row of coordinates per observation for the covariance sum
        coords = np.dstack((xx.ravel(), yy.ravel())).squeeze()
        errors = errors.ravel()

        # Combine short and long range errors with equal variance contributions
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent("short", np.sqrt(0.5), gu.VariogramModel("spherical", 5, 1)),
                xdem.ErrorComponent("long", np.sqrt(0.5), gu.VariogramModel("gaussian", 50, 1)),
            ]
        )

        # Compute the exact effective sample size from all observation pairs
        neff_exact = analytical.neff_exact(coords=coords, error_structure=structure)

        # Check that the non-vectorized version gives the same result
        neff_exact_nv = analytical.neff_exact(coords=coords, error_structure=structure, vectorized=False)
        assert neff_exact == pytest.approx(neff_exact_nv, rel=0.001)

        # Check that the approximation function runs with default parameters, sampling 100 out of 225 samples
        neff_approx = analytical.neff_hugonnet_approx(
            coords=coords, error_structure=structure, subsample=100, random_state=42
        )

        # Check that the non-vectorized version gives the same result, sampling 100 out of 225 samples
        neff_approx_nv = analytical.neff_hugonnet_approx(
            coords=coords,
            error_structure=structure,
            subsample=100,
            vectorized=False,
            random_state=42,
        )

        assert neff_approx == pytest.approx(neff_approx_nv, rel=0.001)

        # Check that the approximation is about the same as the original estimate within 10%
        assert neff_approx == pytest.approx(neff_exact, rel=0.1)


class TestSpatialErrorPropagation:
    """Propagation of component covariances over masks and vector areas."""

    def test_point_weights_with_duplicate_labels(self) -> None:
        """Checks that selected point weights stay aligned when user index labels are duplicated or unordered."""

        # Give each observation a distinct weight and keep duplicate labels in the source table
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, np.ones(5), crs=32632)
        points.ds.index = ["b", "a", "b", "c", "a"]
        original_index = points.ds.index.copy()
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])

        # Independent unit errors have a mean variance equal to the sum of squared normalized weights
        structure = xdem.ErrorStructure([xdem.ErrorComponent("independent", 1.0)])
        normalized = weights[mask] / weights[mask].sum()
        expected = np.sqrt(np.sum(normalized**2))
        result = xdem.uncertainty.spatial_error_propagation(
            [mask], structure, support=points, weights=weights, subsample=None
        )

        # Preserve both the analytical result and the user's original point labels
        assert result[0] == pytest.approx(expected)
        pd.testing.assert_index_equal(points.ds.index, original_index)

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_component_covariance_matches_area_propagation(self) -> None:
        """Checks that area propagation combines local component covariances using both endpoint magnitudes."""

        # Create a small grid with a predictor that varies between observations
        raster = gu.Raster.from_array(np.ones((3, 4)), Affine(10, 0, 0, 0, -10, 30), 32632)
        quality = np.arange(12).reshape(3, 4) / 12

        # Define a magnitude that increases from 0.5 to 2 across the predictor range
        index = pd.Index([0.0, 1.0], name="quality")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        table = pd.DataFrame([[0.5, 100], [2, 100]], index=index, columns=columns)

        # Combine the variable component with constant correlated and independent contributions
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent(
                    "variable", xdem.ErrorMagnitude.grouped(table), gu.VariogramModel("gaussian", 30, 1)
                ),
                xdem.ErrorComponent("fixed", 0.7, gu.VariogramModel("spherical", 100, 1)),
                xdem.ErrorComponent("independent", 0.3),
            ]
        )

        # Exclude one observation and obtain matching coordinates and predictor values
        mask = np.ones(raster.shape, dtype=bool)
        mask[0, 0] = False
        coordinates = np.column_stack(raster.ij2xy(*np.where(mask)))
        covariance = structure.to_covariance_matrix(coordinates, predictors={"quality": quality[mask]})

        # Compute the expected uncertainty directly as the square root of the weighted covariance sum
        weights = np.arange(1, 13).reshape(raster.shape).astype(float)
        normalized = weights[mask] / weights[mask].sum()
        expected = np.sqrt(normalized @ covariance @ normalized)

        # Propagate through the public area API using the same mask and weights
        result = xdem.uncertainty.spatial_error_propagation(
            [mask],
            structure,
            support=raster,
            predictors={"quality": quality},
            weights=weights,
            subsample=None,
        )

        # Check the propagated uncertainty against the independent matrix calculation
        assert result == pytest.approx([expected])

        # Derive effective sample size from the weighted local magnitude and area uncertainty
        mean_sigma = np.sum(normalized * structure.predict_magnitude({"quality": quality[mask]}))
        assert xdem.uncertainty.number_effective_samples(
            mask,
            structure,
            support=raster,
            predictors={"quality": quality},
            weights=weights,
            subsample=None,
        ) == pytest.approx((mean_sigma / expected) ** 2)

        # A numeric area alone cannot locate a spatially variable magnitude predictor
        with pytest.raises(ValueError, match="constant magnitudes"):
            xdem.uncertainty.number_effective_samples(100, structure)

    def test_independent_area_and_point_support(self) -> None:
        """Checks that independent area errors decrease with the square root of the raster or point count."""

        # Assign a magnitude of two to twelve independent observations
        raster = gu.Raster.from_array(np.ones((3, 4)), Affine(10, 0, 0, 0, -10, 30), 32632)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])

        # Both spatial representations must give twelve effective samples and the same standard error
        for support in [raster, raster.to_pointcloud()]:
            assert xdem.uncertainty.number_effective_samples(None, structure, support=support) == pytest.approx(12)
            assert xdem.uncertainty.spatial_error_propagation([None], structure, support=support) == pytest.approx(
                [2 / np.sqrt(12)]
            )

        # A numeric area without observation locations cannot define an independent sample count
        with pytest.raises(ValueError, match="explicit observation support"):
            xdem.uncertainty.number_effective_samples(100, structure)

    def test_vector_area_matches_mask_and_rasterization(self) -> None:
        """Checks that vector areas select the same observations as explicit masks for area propagation."""

        # Define a polygon covering half of a grid with independent errors
        raster = gu.Raster.from_array(np.ones((6, 8)), Affine(10, 0, 0, 0, -10, 60), 32632)
        vector = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 40, 60)], crs=raster.crs))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])

        # Compute the expected standard error from the number of selected raster cells
        mask = vector.create_mask(raster, as_array=True)
        expected = 2 / np.sqrt(np.count_nonzero(mask))

        # Compare Vector and GeoDataFrame inputs on raster and point observations
        for support in [raster, raster.to_pointcloud()]:
            support_mask = vector.create_mask(support, as_array=True)
            expected_on_support = 2 / np.sqrt(np.count_nonzero(support_mask))
            for area in [vector, vector.ds]:
                result = xdem.uncertainty.spatial_error_propagation([area], structure, support=support, subsample=None)
                assert result == pytest.approx([expected_on_support])

        # Rasterize the polygon at the original resolution when no observation grid is supplied
        rasterized = xdem.uncertainty.spatial_error_propagation(
            [vector],
            structure,
            rasterize_resolution=10,
            subsample=None,
        )

        # Check that rasterization reproduces the uncertainty on the explicit grid
        assert rasterized == pytest.approx([expected])

        # A plain mask must have a spatial dataset to provide its observation coordinates
        with pytest.raises(ValueError, match="support"):
            xdem.uncertainty.number_effective_samples(mask, structure)

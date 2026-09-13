"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import warnings
from importlib.util import find_spec

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter

import xdem
from xdem._typing import NDArrayf


class TestErrorStructureEstimation:
    """Estimation of independent and correlated error components."""

    def test_estimate_independent_variable_component_without_variography(self) -> None:
        """Checks that an independent component recovers increasing magnitudes without fitting a variogram."""

        # Create errors whose spread increases with a known quality predictor
        rng = np.random.default_rng(5)
        predictor = np.broadcast_to(np.linspace(0, 1, 40), (40, 40))
        values = 20 + (0.5 + predictor) * rng.normal(size=predictor.shape)

        # Place the proxy and predictor on the same raster grid
        transform = Affine(10, 0, 0, 0, -10, 400)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit only the variable magnitude by declaring spatially independent errors
        structure = xdem.ErrorStructure.estimate(
            proxy,
            predictors={"quality": predictor_raster},
            components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            bins=5,
            min_count=50,
            random_state=4,
        )

        # Check that high predictor values retain the greater input spread
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert predicted[1] > predicted[0]

        # Check that independent estimation has no variogram or pair refinement diagnostics
        assert structure.empirical_variogram is None
        assert structure.fit_diagnostics["refinement"]["success"] is None

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_estimate_separates_variable_short_and_fixed_long_components(self) -> None:
        """Checks that estimation separates a variable short range error from a constant long range error."""

        # Generate two random fields with distinct smoothing lengths
        rng = np.random.default_rng(9)
        predictor = np.broadcast_to(np.linspace(0, 1, 48), (48, 48))
        short_field = gaussian_filter(rng.normal(size=predictor.shape), 1)
        long_field = gaussian_filter(rng.normal(size=predictor.shape), 7)

        # Normalize both fields before assigning their separate error magnitudes
        short_field = (short_field - short_field.mean()) / short_field.std()
        long_field = (long_field - long_field.mean()) / long_field.std()
        values = (0.5 + 1.5 * predictor) * short_field + 0.7 * long_field

        # Place the combined errors and quality predictor on the same raster grid
        transform = Affine(10, 0, 0, 0, -10, 480)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit the default two-component model to the combined proxy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = xdem.ErrorStructure.estimate(
                proxy,
                predictors={"quality": predictor_raster},
                bins=4,
                min_count=100,
                n_pairs=5_000,
                n_lags=8,
                random_state=2,
            )

        # Check that each fitted component retains its intended magnitude model
        short = structure["short_range"]
        long = structure["long_range"]
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert isinstance(short.magnitude, xdem.ErrorMagnitude)
        assert isinstance(long.magnitude, xdem.ErrorMagnitude)
        assert short.correlation is not None and long.correlation is not None
        assert short.magnitude.kind == "grouped"
        assert long.magnitude.kind == "constant"

        # Check that the spatial scales remain ordered and the total spread increases with quality
        assert short.correlation.effective_range < long.correlation.effective_range
        assert predicted[1] > predicted[0]
        assert structure.fit_diagnostics["refinement"]["success"]

    def test_variogram_estimation_retains_only_compact_pair_diagnostics(self) -> None:
        """Checks that estimation retains grouped diagnostics and can refit a model without storing sampled pairs."""

        # Require the optional fitting backend for this correlated error model
        pytest.importorskip("skgstat")

        # Create smooth variation at two spatial scales on a fixed grid
        y, x = np.mgrid[:40, :40]
        values = np.sin(x / 4) + 0.5 * np.cos(y / 10)
        proxy = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 400), 32632, nodata=-9999)

        # Estimate a Gaussian correlation and refit the retained bins with a spherical model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = xdem.ErrorStructure.estimate(
                proxy,
                components={"spatial": {"magnitude": "constant", "correlation": "gaussian"}},
                n_pairs=1_000,
                n_lags=6,
                refine=True,
                random_state=42,
            )
            refitted = structure.refit("spherical")

        # Check that retained diagnostics summarize at most one row per requested lag bin
        refinement = structure.fit_diagnostics["refinement"]
        conditional = refinement["conditional_statistics"]
        assert refinement["success"]
        assert isinstance(conditional, pd.DataFrame)
        assert set(conditional) == {"lag", "mean_magnitude", "semivariance", "fitted_semivariance", "count"}
        assert len(conditional) <= 6

        # Check that raw pair arrays are discarded while the replacement model remains usable
        assert not any("pair" in name for name in vars(structure))
        assert refitted["spatial"].correlation is not None
        assert refitted["spatial"].correlation.model_name == "spherical"


class TestStandardization:
    """Calibration of magnitudes and rejection of standardized outliers."""

    def test_two_step_standardization(self) -> None:
        """Checks that two-step standardization removes an outlier and rescales the remaining errors to unit spread."""

        from xdem.uncertainty.estimation import two_step_standardization

        # Create variable error magnitudes and one outlier far beyond the normal spread
        rng = np.random.default_rng(1)
        quality = rng.uniform(0, 1, 1000)
        values = rng.normal(size=1000) * (1 + quality)
        values[0] = 1000

        # Deliberately double the magnitude model so the second step must correct its scale
        def unscaled(predictors: tuple[ArrayLike, ...]) -> NDArrayf:
            """Return an intentionally oversized magnitude for the quality predictor."""

            return 2.0 * (1.0 + np.asarray(predictors[0], dtype=float))

        # Standardize the errors and retain the corrected magnitude function
        standardized, model = two_step_standardization(values, [quality], unscaled)

        # Check outlier rejection, unit spread and recovery of the original errors from the model
        assert np.isnan(standardized[0])
        assert gu.stats.nmad(standardized) == pytest.approx(1)
        assert model((quality,))[1:] == pytest.approx(values[1:] / standardized[1:])

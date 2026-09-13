"""Compatibility checks for the entirely deprecated spatialstats module."""

import inspect

import geoutils as gu
import numpy as np
import pandas as pd
import pytest

import xdem


class TestSpatialstatsDeprecation:
    """Compatibility and migration guidance for the deprecated public module."""

    def test_every_spatialstats_function_is_deprecated(self) -> None:
        """Checks that every legacy function warns and retains its documented call signature."""

        # Select functions defined in this module, excluding imports and the optional Numba fallback
        functions = {
            name: function
            for name, function in inspect.getmembers(xdem.spatialstats, inspect.isfunction)
            if not name.startswith("_") and name != "njit" and function.__module__ == xdem.spatialstats.__name__
        }

        # Check that all original public entry points remain available during deprecation
        assert len(functions) == 24

        # Check migration guidance and signature preservation for each legacy entry point
        for name, function in functions.items():
            doc = inspect.getdoc(function)
            assert doc is not None and "uncertainty_migration.html" in doc
            assert inspect.signature(function) == inspect.signature(function.__wrapped__)

            # Check migration guidance even when the legacy call is missing required arguments
            with pytest.warns(DeprecationWarning, match=f"{name}.*uncertainty_migration.html"):
                with pytest.raises(TypeError):
                    function()

    def test_deprecated_grouping_and_fit(self) -> None:
        """Checks that deprecated binning and fitting retain their original tables, predictions and warnings."""

        # Create simple increasing values with two perfectly related predictors
        values = np.arange(100, dtype=float)
        with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
            table = xdem.spatialstats.nd_binning(values, [values, values / 10], ["x", "y"], list_var_bins=3)

        # Check that the original table still includes marginal and joint groups
        assert set(table.nd) == {1, 2}
        with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
            interpolator = xdem.spatialstats.interp_nd_binning(table, ["x"], statistic="nanmedian", min_count=1)

        # Evaluate interpolation with the original positional predictor interface
        assert np.isfinite(interpolator((np.array([10, 50]),))).all()
        with pytest.warns(DeprecationWarning):

            # Look up the two-dimensional group statistic for each observation
            result = xdem.spatialstats.get_perbin_nd_binning(
                table, [values, values / 10], ["x", "y"], statistic="nanmedian"
            )

        # The first bin contains values zero through 32, whose median is 16
        assert result.shape == values.shape
        np.testing.assert_array_equal(result[:33], np.full(33, 16.0))
        with pytest.warns(DeprecationWarning):

            # Check that the original NMAD entry point still delegates to GeoUtils
            assert xdem.spatialstats.nmad(values) == gu.stats.nmad(values)

    def test_deprecated_variogram_and_neff(self) -> None:
        """Checks that deprecated variogram functions preserve parameter tables and sampling controls."""

        # Require the original optional variogram backend
        pytest.importorskip("skgstat")

        # Define a legacy parameter table with known covariance at zero distance
        params = pd.DataFrame({"model": ["spherical"], "range": [20], "psill": [4]})
        with pytest.warns(DeprecationWarning):
            covariance = xdem.spatialstats.covariance_from_variogram(params)

        # The zero-distance covariance must equal the supplied partial sill
        assert covariance(0) == pytest.approx(4)
        with pytest.warns(DeprecationWarning):
            value = xdem.spatialstats.number_effective_samples(100, params)

        # Compare legacy effective samples with the same model expressed as an ErrorStructure
        structure = xdem.ErrorStructure([xdem.ErrorComponent("error", 2, gu.VariogramModel("spherical", 20, 1))])
        assert value == pytest.approx(xdem.uncertainty.number_effective_samples(100, structure))
        with pytest.warns(DeprecationWarning):

            # Exercise the original sampler controls and explicit distance bins
            empirical = xdem.spatialstats.sample_empirical_variogram(
                np.random.default_rng(0).normal(size=(12, 12)),
                gsd=10,
                subsample=30,
                bin_func=[15, 30, 60, 90, 160],
                runs=2,
                samples=5,
                random_state=4,
            )

        # Check the legacy column names and removal of the last undersampled distance bin
        assert set(empirical) == {"exp", "lags", "count", "err_exp"}
        np.testing.assert_array_equal(empirical["lags"], [15, 30, 60, 90])

    def test_deprecated_plotting(self) -> None:
        """Checks that deprecated plotting accepts the original empirical tables and fitted curves."""

        # Require plotting support and construct a small legacy variogram table
        pyplot = pytest.importorskip("matplotlib.pyplot")
        empirical = pd.DataFrame({"exp": [1, 2, 3], "lags": [10, 20, 30], "count": [50, 60, 70], "err_exp": np.nan})

        # Plot a fitted curve across split distance ranges while checking migration guidance
        with pytest.warns(DeprecationWarning, match="uncertainty_migration.html"):
            xdem.spatialstats.plot_variogram(
                empirical,
                [lambda distance: distance / 10],
                ["model"],
                xscale_range_split=[15],
                xlabel="Distance",
            )

        # Create a legacy bin table and pass it directly to the original one-dimensional plot
        values = np.arange(100.0)
        with pytest.warns(DeprecationWarning):
            table = xdem.spatialstats.nd_binning(values, [values], ["quality"], list_var_bins=3)
        with pytest.warns(DeprecationWarning):
            xdem.spatialstats.plot_1d_binning(table, "quality", "nanmedian", min_count=1)

        # Close the figures so this compatibility check leaves no plotting state behind
        pyplot.close("all")

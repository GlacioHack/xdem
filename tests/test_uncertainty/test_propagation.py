"""Tests for the thin terrain, spatial and coregistration propagation adapters."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geoutils as gu
import numpy as np
import pytest
from affine import Affine
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel

import xdem


class TestSpatialPropagation:
    """Test module for area summaries and delegation to GeoUtils spatial propagation."""

    @pytest.mark.parametrize("statistic", ["mean", "integral"])
    @pytest.mark.parametrize("point_input", [False, True])
    def test_propagate_uncertainty__weighted_areas(self, statistic: str, point_input: bool) -> None:
        """Checks that weighted means and integrals use the selected original observations and their errors."""

        # Create a labelled area with unequal weights so normalization and area scaling can be checked
        values = np.arange(48, dtype=float).reshape(6, 8)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 12), 32632)
        source = raster.to_pointcloud() if point_input else raster
        mask = np.zeros(source.data.shape, dtype=bool)
        mask.reshape(-1)[5:15] = True
        weights = np.linspace(1, 2, values.size).reshape(source.data.shape)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Propagate a fixed weighted sum through GeoUtils' analytical calculation
        summary = xdem.uncertainty.propagate_uncertainty(
            source,
            "spatial",
            error_structure=structure,
            method="analytical",
            inputs={"spatial": {"areas": {"selected": mask}, "weights": weights}},
            operation_options={"statistic": statistic},
        )

        # Independent observation errors give a variance equal to the sum of squared weighted magnitudes
        coefficients = weights[mask].copy()
        if statistic == "mean":
            coefficients /= coefficients.sum()
        elif not point_input:
            coefficients *= 4
        expected = coefficients @ source.data[mask]
        assert type(summary) is gu.PropagationSummary
        assert summary.estimate["selected"] == pytest.approx(expected)
        assert summary.std["selected"] == pytest.approx(0.2 * np.linalg.norm(coefficients))

    def test_propagate_uncertainty__bound_spatial_method(self) -> None:
        """Checks that passing a bound reprojection method returns the same summary as GeoUtils."""

        # Create a fine raster and a coarser reference for area averaging
        source = gu.Raster.from_array(np.arange(64, dtype=float).reshape(8, 8), Affine(1, 0, 0, 0, -1, 8), 32632)
        reference = gu.Raster.from_array(np.zeros((4, 4)), Affine(2, 0, 0, 0, -2, 8), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        options: dict[str, Any] = {
            "ref": reference,
            "resampling": Mean(),
            "nodata_propagation": "ignore",
        }

        # Call the same bound method through both public propagation entry points
        expected = gu.uncertainty.propagate(
            source.reproject,
            error_structure=errors,
            operation_kwargs=options,
            method="analytical",
        )
        actual = xdem.uncertainty.propagate_uncertainty(
            operation=source.reproject,
            error_structure=errors,
            operation_options=options,
            method="analytical",
        )

        # Compare native raster layouts and every propagated marginal
        assert type(actual) is gu.PropagationSummary
        assert actual.std.georeferenced_grid_equal(expected.std)
        np.testing.assert_array_equal(actual.estimate.data, expected.estimate.data)
        np.testing.assert_array_equal(actual.std.data, expected.std.data)


class TestTerrainPropagation:
    """Test module for terrain filter derivatives and analytical uncertainty at flat terrain."""

    @pytest.mark.parametrize("surface_fit", ["Horn", "ZevenbergThorne", "Florinsky"])
    @pytest.mark.parametrize("attribute", ["slope", "aspect"])
    @pytest.mark.parametrize("degrees", [False, True])
    @pytest.mark.parametrize("correlated", [False, True])
    def test_propagate_uncertainty__terrain_matches_finite_differences(
        self, surface_fit: str, attribute: str, degrees: bool, correlated: bool
    ) -> None:
        """Checks that analytical terrain uncertainty matches derivatives of the public terrain calculation."""

        # Use a sloping curved surface so slope and aspect have defined derivatives away from the edges
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

        # Propagate to the whole map and select one interior pixel for an independent comparison
        summary = xdem.uncertainty.propagate_uncertainty(
            raster,
            "terrain",
            error_structure=errors,
            operation_options=options,
            method="analytical",
            at=[(5, 6)],
        )

        # Perturb the public API's actual stencil, including Florinsky's wider 5 x 5 neighborhood
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

        # Combine those numerical derivatives with the same source error model
        x, y = raster.ij2xy(*np.asarray(locations).T)
        covariance = errors.to_covariance_matrix(np.column_stack((x, y)))
        jacobian = np.asarray(derivatives)
        expected_std = np.sqrt(jacobian @ covariance @ jacobian)
        assert summary.std.data[5, 6] == pytest.approx(expected_std, rel=1e-6)
        assert summary.metadata["approximation"] == "first_order"
        np.testing.assert_array_equal(raster.data, values)

    def test_propagate_uncertainty__flat_terrain(self) -> None:
        """Checks that flat slope has undefined analytical uncertainty and positive numerical uncertainty."""

        # Create a flat raster where small errors change slope but its derivative is undefined
        raster = gu.Raster.from_array(np.ones((7, 9)), Affine(2, 0, 0, 0, -2, 14), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        options = {"attribute": "slope", "engine": "scipy"}

        # Compare the derivative approximation with direct terrain simulations
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

        # Numerical calculations describe the nonlinear slope response even at zero gradient
        assert analytical.std.get_mask()[3, 4]
        assert numerical.mean.data[3, 4] > 0
        assert numerical.std.data[3, 4] > 0


class TestCoregistrationSummary:
    """Test module for storing fitted transforms and plotting translations and rotations with their own units."""

    @pytest.mark.parametrize("supplied_axes", [False, True])
    def test_plot__parameters_and_fitted_transform(self, supplied_axes: bool) -> None:
        """Checks that a coregistration summary stores its fit and plots means with their parameter uncertainties."""

        import matplotlib.pyplot as plt

        # Create a vertical offset so the original transformation and sampled uncertainty have known meanings
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        source = reference + 5
        method = xdem.coreg.VerticalShift(vshift_reduc_func=np.mean)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Store the fitted transformation with the ordinary GeoUtils simulation statistics and draws
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            source,
            method,
            error_structure=errors,
            n_samples=4,
            random_state=2,
            return_samples=True,
        )
        assert isinstance(summary, gu.PropagationSummary)
        assert isinstance(summary, xdem.uncertainty.PropagationSummary)
        assert summary.coreg is not method
        assert summary.coreg.to_matrix()[2, 3] == pytest.approx(-5)
        assert summary.output["crs"] == reference.crs
        assert not method._fit_called
        assert summary.samples is not None
        np.testing.assert_allclose(summary.std, summary.samples.std())

        # Plot on new axes or reuse supplied ones without mixing translations and angles
        axes = tuple(plt.subplots(1, 2)[1]) if supplied_axes else None
        plotted_axes = summary.plot(axes=axes)
        if axes is not None:
            assert plotted_axes == axes

        # Check the plotted means, uncertainty endpoints and original fit against the stored results
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
    """Test module for exposing existing GLS/TLS LZD standard errors without another propagation calculation."""

    @pytest.mark.parametrize("optimizer", ["gls", "tls"])
    @pytest.mark.parametrize("only_translation", [True, False])
    def test_propagate_uncertainty_coreg__uses_lzd_uncertainty(self, optimizer: str, only_translation: bool) -> None:
        """Checks that analytical summaries return exactly the standard errors already calculated by LZD."""

        if optimizer == "gls" and find_spec("gpytorch") is None:
            pytest.skip("GLS requires optional GPyTorch.")

        # Create curved terrain with enough variation to fit translations and rotations
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

        # Run the existing error-aware fit, then wrap its output without refitting
        fitted = xdem.coreg.LZD(only_translation=only_translation, subsample=100).fit(
            reference, aligned, random_state=2, **fit_options
        )
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, aligned, fitted, error_structure=errors, method="analytical"
        )

        # Preserve the native parameter order and exact fitted errors; inactive rotations have zero spread
        count = 3 if only_translation else 6
        np.testing.assert_array_equal(summary.std.iloc[:count], fitted.meta["outputs"]["fitorbin"]["fit_perr"])
        np.testing.assert_array_equal(summary.std.iloc[count:], 0)
        np.testing.assert_allclose(summary.estimate.iloc[:3], fitted.to_translations())
        assert summary.output["centroid"] == fitted.centroid()
        assert type(summary) is xdem.uncertainty.PropagationSummary
        assert summary.coreg is not fitted
        np.testing.assert_array_equal(summary.coreg.to_matrix(), fitted.to_matrix())


class TestPropagationErrors:
    """Test module for unsupported analytical fits and explicit numerical sample selection."""

    def test_propagate_uncertainty_coreg__error_ols_has_no_analytical_result(
        self,
    ) -> None:
        """Checks an error is raised for analytical coregistration without GLS/TLS fit uncertainties."""

        # Use observable terrain so the unsupported uncertainty method is the source of the error
        rows, columns = np.indices((9, 11))
        values = np.sin(columns / 2) + np.cos(rows / 3)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 18), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # OLS fitting remains available, but no separate analytical propagation is added around it
        with pytest.raises(ValueError, match="GLS/TLS uncertainties"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                raster,
                raster + 1,
                xdem.coreg.LZD(only_translation=True, subsample=1),
                error_structure=errors,
                method="analytical",
            )

    def test_propagate_uncertainty__error_spatial_samples_need_selection(self) -> None:
        """Checks an error is raised when raster simulation draws are requested without selecting pixels."""

        # Create a small source whose spatial output still requires an explicit draw selection
        raster = gu.Raster.from_array(np.ones((5, 7)), Affine(1, 0, 0, 0, -1, 5), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Require a selection before allocating retained simulations
        with pytest.raises(ValueError, match="Select outputs with at"):
            xdem.uncertainty.propagate_uncertainty(raster, error_structure=errors, return_samples=True)

"""Tests for coregistration parameter uncertainty and propagation summaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine
from geoutils.multiproc import MultiprocConfig
from geoutils.stats.variography import VariogramModel
from scipy.stats import chi2, norm, rice

import xdem
from xdem import DEM, open_dem


@pytest.fixture
def icp_dem() -> gu.Raster:
    """Create a small curved DEM to work on coreg with rotation/translation."""

    rows, columns = np.indices((10, 12))
    x, y = (columns - 5.5) * 2, (4.5 - rows) * 3
    values = 40 + 5 * np.sin(x / 6) + 3 * np.cos(y / 8) + 2 * np.sin((x + y) / 7)
    return gu.Raster.from_array(
        values, Affine(2, 0, 500_000, 0, -3, 5_100_000), 32632, nodata=np.nan, area_or_point="Point"
    )


@pytest.fixture
def parameter_summary() -> xdem.uncertainty.PropagationSummary:
    """Create six correlated parameters with stored covariance and deterministic simulation samples."""

    labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])
    covariance = np.diag([4.0, 1.0, 0.25, 0.04, 0.01, 0.0025])
    covariance[0, 1] = covariance[1, 0] = 1.0
    covariance[3, 4] = covariance[4, 3] = 0.01
    samples = pd.DataFrame(np.random.default_rng(42).multivariate_normal(np.zeros(6), covariance, 200), columns=labels)
    mean = samples.mean()
    estimate = mean + samples.std() / 4
    selection = pd.DataFrame({"flat_index": np.arange(6), "estimate": estimate}, index=labels)
    return xdem.uncertainty.PropagationSummary(
        estimate=estimate,
        mean=mean,
        std=samples.std(),
        error_structure=gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 0.2)]),
        method="analytical",
        selection=selection,
        samples=samples,
        output={"translation_units": "metre", "parameter_covariance": samples.cov()},
    )


def _compare_simulated_uncertainty(
    reference: gu.Raster | gu.PointCloud,
    moving: gu.Raster | gu.PointCloud,
    method: xdem.coreg.AffineCoreg,
    options: dict[str, Any],
    only_translation: bool,
    *,
    n_simulations: int = 20,
    check_mean: bool = False,
) -> xdem.uncertainty.PropagationSummary:
    """Compare analytical parameter uncertainties with repeated fits under the same source errors."""

    # Propagate identical errors and fitting options analytically and numerically
    analytical = xdem.uncertainty.propagate_uncertainty_coreg(reference, moving, method, method="analytical", **options)

    # Request the same GLS/TLS fit explicitly for the numerical comparison
    numerical_options = dict(options)
    fit_options = dict(options.get("fit_kwargs", {}))
    if not any(name in fit_options for name in ("sig_ref", "sig_tba", "corr_ref", "corr_tba")):
        fit_options.update(error_structure=options["error_structure"], error_predictors=options.get("error_predictors"))
    numerical_options["fit_kwargs"] = fit_options
    numerical = xdem.uncertainty.propagate_uncertainty_coreg(
        reference,
        moving,
        method,
        n_simulations=n_simulations,
        on_error="raise",
        random_field_options={"backend": "gpytorch", "gpytorch_inducing_points": None},
        **numerical_options,
    )

    # We use a chi-square interval to quantitatively constrain the allowed variation due to small number of simulations,
    # with an additional 0.1% error across parameters
    count = 3 if only_translation else 6
    probability = 0.001 / (2 * count)
    bounds = np.sqrt(chi2.ppf([probability, 1 - probability], n_simulations - 1) / (n_simulations - 1))
    ratios = numerical.std.iloc[:count] / analytical.std.iloc[:count]
    assert np.all((bounds[0] <= ratios) & (ratios <= bounds[1])), ratios

    # Check that every draw contributes
    assert numerical.n_success == n_simulations
    assert numerical.n_valid.eq(n_simulations).all()
    if only_translation:
        np.testing.assert_array_equal(numerical.std.iloc[3:], 0)

    # Check simulated means against the original fit within 4.5 standard errors
    if check_mean:
        mean_error = (numerical.mean.iloc[:count] - analytical.estimate.iloc[:count]) / analytical.std.iloc[:count]
        np.testing.assert_allclose(mean_error, 0, rtol=0, atol=4.5 / np.sqrt(n_simulations))
    return analytical


class TestCoregistrationPropagationSummary:
    """Test module for the PropagationSummary subclass adapted to coreg."""

    @pytest.mark.parametrize("input_axes", [False, True])
    def test_plot(self, input_axes: bool) -> None:
        """Checks that plot show the right data."""

        plt = pytest.importorskip("matplotlib.pyplot")

        # We create a synthetic propagation summary
        labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])
        estimate = pd.Series([1, 2, 3, 0.1, 0.2, 0.3], index=labels)
        expected_mean = estimate + pd.Series([0.1, -0.2, 0.3, -0.01, 0.02, -0.03], index=labels)
        expected_std = pd.Series([0.2, 0.3, 0.4, 0.02, 0.03, 0.04], index=labels)
        selection = pd.DataFrame({"flat_index": np.arange(6), "estimate": estimate}, index=labels)
        summary = xdem.uncertainty.PropagationSummary(
            estimate=estimate,
            mean=expected_mean,
            std=expected_std,
            error_structure=gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)]),
            method="numerical",
            selection=selection,
            output={"translation_units": "metre"},
        )

        # THen create or reuse plot axes
        axes = tuple(plt.subplots(1, 2)[1]) if input_axes else None
        plotted_axes = summary.plot(axes=axes)
        if axes is not None:
            assert plotted_axes == axes

        # Check means, error bars and the original fit are all displayed properly
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

    @pytest.mark.parametrize(
        "source, input_axes, coverage",
        [
            pytest.param("analytical", False, 0.95, id="analytical-new-axes-95"),
            pytest.param("analytical", True, 0.68, id="analytical-supplied-axes-68"),
            pytest.param("samples", False, 0.68, id="samples-new-axes-68"),
            pytest.param("samples", True, 0.95, id="samples-supplied-axes-95"),
        ],
    )
    def test_plot__ellipses(
        self,
        parameter_summary: xdem.uncertainty.PropagationSummary,
        input_axes: bool,
        source: str,
        coverage: float,
    ) -> None:
        """Checks that joint X/Y regions and separate Z intervals reproduce the parameter covariance."""

        plt = pytest.importorskip("matplotlib.pyplot")
        summary = parameter_summary
        if source == "samples":
            changes: dict[str, Any] = {"method": "numerical", "output": {"translation_units": "metre"}}
            summary = replace(summary, **changes)

        # Draw both layouts from the same correlated parameters
        axes = tuple(plt.subplots(2, 2)[1].flat) if input_axes else None
        plotted = summary.plot(axes=axes, kind="ellipses", coverage=coverage, capsize=5, color="C2")
        assert len(plotted) == 4
        if axes is not None:
            assert plotted == axes

        # A transformed unit circle has the requested covariance after removing the joint probability scale
        covariance = parameter_summary.output["parameter_covariance"]
        for index, names in enumerate((("tx", "ty", "tz"), ("rx", "ry", "rz"))):
            horizontal, vertical = plotted[2 * index : 2 * index + 2]
            ellipse = horizontal.patches[0]
            transform = ellipse.get_patch_transform().get_matrix()[:2, :2]
            expected = covariance.loc[list(names[:2]), list(names[:2])].to_numpy()
            np.testing.assert_allclose(transform @ transform.T / chi2.ppf(coverage, 2), expected)
            np.testing.assert_allclose(ellipse.center, summary.mean.loc[list(names[:2])])
            np.testing.assert_allclose(
                np.asarray(horizontal.lines[-1].get_data()).ravel(), summary.estimate.loc[list(names[:2])]
            )

            # Z has a one-dimensional normal interval, with its own axis and units
            bounds = (
                summary.mean.loc[names[2]]
                + np.array([-1, 1]) * norm.ppf((1 + coverage) / 2) * summary.std.loc[names[2]]
            )
            np.testing.assert_allclose(vertical.collections[0].get_segments()[0][:, 1], bounds)
            unit = "metre" if index == 0 else "radian"
            assert horizontal.get_xlabel() == f"X {'translation' if index == 0 else 'rotation'} ({unit})"
            assert vertical.get_xticklabels()[0].get_text() == names[2]
        plt.close(plotted[0].figure)

    @pytest.mark.parametrize(
        "source, kind, input_axes, coverage",
        [
            pytest.param("analytical", None, False, 0.95, id="analytical-default-new-axes-95"),
            pytest.param("analytical", "horizontal_vertical", True, 0.68, id="analytical-explicit-supplied-axes-68"),
            pytest.param("samples", None, True, 0.68, id="samples-default-supplied-axes-68"),
            pytest.param("samples", "horizontal_vertical", False, 0.95, id="samples-explicit-new-axes-95"),
        ],
    )
    def test_plot__horizontal_vertical(
        self,
        parameter_summary: xdem.uncertainty.PropagationSummary,
        input_axes: bool,
        source: str,
        coverage: float,
        kind: str | None,
    ) -> None:
        """Checks that ellipses show horizontal magnitudes, vertical parameters and their covariance."""

        plt = pytest.importorskip("matplotlib.pyplot")
        summary = parameter_summary
        if source == "samples":
            changes: dict[str, Any] = {"method": "numerical", "output": {"translation_units": "metre"}}
            summary = replace(summary, **changes)
        else:
            # Equal independent horizontal variances give a Rice magnitude
            covariance = summary.output["parameter_covariance"].copy()
            for names in (("tx", "ty", "tz"), ("rx", "ry", "rz")):
                variance = covariance.loc[names[0], names[0]]
                covariance.loc[list(names), list(names)] = np.diag(
                    [variance, variance, covariance.loc[names[2], names[2]]]
                )
            changes = {"output": {**summary.output, "parameter_covariance": covariance}}
            summary = replace(summary, **changes)
        axes = tuple(plt.subplots(1, 2)[1]) if input_axes else None

        options: dict[str, Any] = {} if kind is None else {"kind": kind}
        plotted = summary.plot(axes=axes, coverage=coverage, **options)

        assert len(plotted) == 2
        if axes is not None:
            assert plotted == axes

        # Compare transformed covariance and means
        covariance = summary.output.get("parameter_covariance")
        if covariance is None:
            covariance = summary.samples.cov()
        for axis, names, scale, xlabel, ylabel in zip(
            plotted,
            (("tx", "ty", "tz"), ("rx", "ry", "rz")),
            (1.0, 1e6),
            ("Horizontal translation (metre)", "Tilt (µrad)"),
            ("Vertical translation (metre)", "Yaw (µrad)"),
        ):
            assert len(axis.patches) == 1
            ellipse = axis.patches[0]
            block = covariance.loc[list(names), list(names)].to_numpy()
            mean_x, mean_y, mean_z = summary.mean.loc[list(names)]
            horizontal = np.hypot(mean_x, mean_y)
            sigma = np.sqrt(block[0, 0])
            magnitude_mean = rice.mean(horizontal / sigma, scale=sigma)
            magnitude_variance = rice.var(horizontal / sigma, scale=sigma)
            expected_center = np.array([magnitude_mean, mean_z]) * scale
            expected = np.diag([magnitude_variance, block[2, 2]])
            if source == "samples":
                values = summary.samples.loc[:, list(names)].to_numpy()
                magnitudes = np.sqrt(values[:, 0] ** 2 + values[:, 1] ** 2)
                expected_center = np.array([magnitudes.mean(), values[:, 2].mean()]) * scale
                expected = np.cov(magnitudes, values[:, 2])
            transform = ellipse.get_patch_transform().get_matrix()[:2, :2]
            np.testing.assert_allclose(transform @ transform.T / (chi2.ppf(coverage, 2) * scale**2), expected)
            np.testing.assert_allclose(ellipse.center, expected_center)
            np.testing.assert_array_equal(axis.lines[0].get_ydata(), [0, 0])
            np.testing.assert_array_equal(axis.lines[1].get_xdata(), [0, 0])
            assert axis.get_legend() is None
            assert axis.get_xlabel() == xlabel
            assert axis.get_ylabel() == ylabel
            assert 0 in axis.get_xticks()
            assert 0 in axis.get_yticks()
            assert axis.get_xlim()[0] < 0 < axis.get_xlim()[1]
            assert axis.get_ylim()[0] < 0 < axis.get_ylim()[1]
        plt.close(plotted[0].figure)

    @pytest.mark.parametrize(
        "horizontal_std, other_std, expected_mean",
        [(0.0, 0.0, 0.0), (2.0, 2.0, np.sqrt(2 * np.pi)), (2.0, 0.0, np.sqrt(8 / np.pi))],
    )
    def test_plot__zero_horizontal_mean(
        self,
        parameter_summary: xdem.uncertainty.PropagationSummary,
        horizontal_std: float,
        other_std: float,
        expected_mean: float,
    ) -> None:
        """Checks that zero horizontal means give finite ellipses with Gaussian magnitude moments."""

        plt = pytest.importorskip("matplotlib.pyplot")
        labels = parameter_summary.mean.index
        covariance = pd.DataFrame(np.diag([horizontal_std**2, other_std**2, 1, 0, 0, 0]), labels, labels)
        changes: dict[str, Any] = {
            "mean": pd.Series(0.0, index=labels),
            "samples": None,
            "output": {"parameter_covariance": covariance},
        }
        summary = replace(parameter_summary, **changes)

        axes = summary.plot()

        # Independent X/Y errors give Rayleigh or half-normal magnitudes
        expected_covariance = np.diag([horizontal_std**2 + other_std**2 - expected_mean**2, 1])
        ellipse = axes[0].patches[0]
        transform = ellipse.get_patch_transform().get_matrix()[:2, :2]
        np.testing.assert_allclose(ellipse.center, [expected_mean, 0])
        np.testing.assert_allclose(transform @ transform.T / chi2.ppf(0.95, 2), expected_covariance, atol=1e-10)
        np.testing.assert_array_equal(axes[1].patches[0].center, [0, 0])
        plt.close(axes[0].figure)

    @pytest.mark.parametrize("horizontal_mean", [0.0, 1.0, 1000.0])
    def test_plot__correlated_gaussian_magnitudes(
        self, parameter_summary: xdem.uncertainty.PropagationSummary, horizontal_mean: float
    ) -> None:
        """Checks that Gaussian magnitudes include X/Y correlation and covariance with the vertical parameter."""

        plt = pytest.importorskip("matplotlib.pyplot")
        labels = parameter_summary.mean.index
        means = horizontal_mean * np.array([1.0, 2.0, 3.0])
        block = 4 * np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) + np.diag([0, 0, 1])
        covariance = np.zeros((6, 6))
        covariance[:3, :3], covariance[3:, 3:] = block, block * 1e-12
        changes: dict[str, Any] = {
            "mean": pd.Series(np.concatenate((means, means * 1e-6)), index=labels),
            "std": pd.Series(np.sqrt(np.diag(covariance)), index=labels),
            "samples": None,
            "output": {"parameter_covariance": pd.DataFrame(covariance, labels, labels)},
        }
        summary = replace(parameter_summary, **changes)

        axes = summary.plot()

        # With Y = 2X, the magnitude is sqrt(5) times a folded normal variable
        signed_probability = 2 * norm.cdf(horizontal_mean / 2) - 1
        absolute_mean = 2 * np.sqrt(2 / np.pi) * np.exp(-(horizontal_mean**2) / 8)
        absolute_mean += horizontal_mean * signed_probability
        expected_mean = np.sqrt(5) * absolute_mean
        magnitude_variance = 5 * (horizontal_mean**2 + 4 - absolute_mean**2)
        cross_covariance = np.sqrt(5) * 12 * signed_probability
        expected_covariance = np.array([[magnitude_variance, cross_covariance], [cross_covariance, 37]])
        for axis in axes:
            ellipse = axis.patches[0]
            transform = ellipse.get_patch_transform().get_matrix()[:2, :2]
            np.testing.assert_allclose(ellipse.center, [expected_mean, means[2]], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(
                transform @ transform.T / chi2.ppf(0.95, 2), expected_covariance, rtol=1e-9, atol=1e-9
            )
        plt.close(axes[0].figure)

    @pytest.mark.parametrize("show_samples", [False, True])
    def test_plot__numerical_draws(
        self, parameter_summary: xdem.uncertainty.PropagationSummary, show_samples: bool
    ) -> None:
        """Checks that numerical points show one horizontal magnitude and vertical value per simulation."""

        plt = pytest.importorskip("matplotlib.pyplot")

        # Set distinct translations and rotations
        draws = pd.DataFrame(
            [[8, 20, 30, 1e-6, 2e-6, 3e-6], [10, 24, 32, 3e-6, 6e-6, 5e-6], [12, 28, 34, 5e-6, 10e-6, 7e-6]],
            columns=parameter_summary.selection.index,
        )
        changes: dict[str, Any] = {
            "method": "numerical",
            "samples": draws,
            "mean": draws.mean(),
            "std": draws.std(),
            "output": {"translation_units": "metre"},
        }
        summary = replace(parameter_summary, **changes)

        axes = summary.plot(show_samples=show_samples)

        # Check magnitudes in displayed units
        expected = (
            [[np.sqrt(464), 30], [26, 32], [np.sqrt(928), 34]],
            [[np.sqrt(5), 3], [np.sqrt(45), 5], [np.sqrt(125), 7]],
        )
        for axis, values in zip(axes, expected):
            np.testing.assert_allclose(axis.patches[0].center, np.mean(values, axis=0))
            assert len(axis.collections) == int(show_samples)
            if show_samples:
                np.testing.assert_allclose(axis.collections[0].get_offsets(), values, atol=1e-14)
                assert axis.collections[0].get_alpha() == 0.5
        plt.close(axes[0].figure)

    def test_plot__compare_summaries(self, parameter_summary: xdem.uncertainty.PropagationSummary) -> None:
        """Checks that plot() overlays labelled analytical and numerical ellipses with zero ticks."""

        plt = pytest.importorskip("matplotlib.pyplot")
        changes: dict[str, Any] = {"method": "numerical", "output": {"translation_units": "metre"}}
        numerical = replace(parameter_summary, **changes)

        axes = parameter_summary.plot(color="C0", label="Analytical")
        plotted = numerical.plot(axes=axes, color="C1", label="Numerical")

        assert plotted == axes
        for axis in axes:
            assert len(axis.patches) == 2
            assert len(axis.collections) == 1
            assert axis.collections[0].get_alpha() == 0.5
            assert axis.patches[0].get_edgecolor() != axis.patches[1].get_edgecolor()
            assert axis.get_legend_handles_labels()[1] == ["Analytical", "Numerical"]
            assert 0 in axis.get_xticks()
            assert 0 in axis.get_yticks()
        plt.close(axes[0].figure)


class TestPropagateUncertaintyCoreg:
    """Test module for the propagate_uncertainty_coreg() function."""

    @pytest.mark.parametrize(
        "return_coregs, return_samples, at",
        [
            (False, False, ["tz"]),
            (False, True, ["tz"]),
            (True, False, ["tz"]),
            (True, True, ["tz"]),
            (True, True, None),
        ],
    )
    def test_propagate_uncertainty_coreg__saved_coregs(
        self, return_coregs: bool, return_samples: bool, at: list[str] | None
    ) -> None:
        """Checks that saved Coregs match the parameter summaries and store each fit metadata."""

        # 1/ Simulate vertical shifts
        # Create DEM with a vertical offset, and an error structure
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
            n_simulations=4,
            random_state=2,
            return_coregs=return_coregs,
            return_samples=return_samples,
            at=at,
        )

        # 2/ Check saved fits and summaries
        # Check the 6-param summary
        labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
        assert type(summary) is xdem.uncertainty.PropagationSummary
        for quantity in (summary.estimate, summary.mean, summary.std, summary.n_valid):
            assert isinstance(quantity, pd.Series)
            assert quantity.index.tolist() == labels
        assert summary.estimate["tz"] == pytest.approx(-5)

        # Check successful draws, metadata and selected sample columns
        assert summary.n_success == 4
        assert summary.n_valid.eq(4).all()
        assert summary.output["crs"] == reference.crs
        assert "centroid" in summary.output
        if return_samples:
            assert summary.samples is not None
            assert summary.samples.shape == (4, len(labels if at is None else at))
            assert summary.samples.columns.tolist() == (labels if at is None else at)

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

    @pytest.mark.parametrize(
        "solver, options",
        [
            (xdem.coreg.LZD, {}),
            (xdem.coreg.ICP, {}),
            (xdem.coreg.LZD, {"trim_residuals": True}),
            (xdem.coreg.ICP, {"trim_residuals": True}),
            (xdem.coreg.LZD, {"linearized": False}),
            (xdem.coreg.ICP, {"linearized": False}),
            (xdem.coreg.LZD, {"fit_loss_func": "soft_l1"}),
            (xdem.coreg.ICP, {"fit_loss_func": "soft_l1"}),
            (xdem.coreg.ICP, {"method": "point-to-point"}),
            (xdem.coreg.ICP, {"sampling_strategy": "same_xy", "iterative_sampling": True}),
            (xdem.coreg.ICP, {"sampling_strategy": "nearest_xy", "iterative_sampling": True}),
        ],
    )
    def test_propagate_uncertainty_coreg__numerical_estimators(
        self, icp_dem: gu.Raster, solver: type[xdem.coreg.AffineCoreg], options: dict[str, Any]
    ) -> None:
        """Checks that numerical propagation works with any coreg setup (linear or not, custom funcs, etc)."""

        # Create a vertical offset and small independent measurement errors
        source = icp_dem + 0.2
        method = solver(only_translation=True, subsample=1, **options)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.01)])

        # Run fit and 3 random draws
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            icp_dem,
            source,
            method,
            error_structure=errors,
            n_simulations=3,
            random_state=12,
            on_error="raise",
        )

        # Check the known correction and successful draws
        np.testing.assert_allclose(summary.estimate.iloc[:3], (0, 0, -0.2), rtol=0, atol=1e-7)
        assert summary.n_success == 3
        assert summary.std["tz"] > 0

    @pytest.mark.parametrize(
        "solver, side, predictor_kind, propagation_method",
        [
            pytest.param(xdem.coreg.LZD, "ref", "array", "analytical", id="lzd-reference-array-analytical"),
            pytest.param(xdem.coreg.LZD, "tba", "raster", "numerical", id="lzd-moving-raster-numerical"),
            pytest.param(xdem.coreg.LZD, "ref", "terrain_name", "numerical", id="lzd-reference-terrain-numerical"),
            pytest.param(xdem.coreg.ICP, "tba", "array", "numerical", id="icp-moving-array-numerical"),
            pytest.param(xdem.coreg.ICP, "ref", "raster", "analytical", id="icp-reference-raster-analytical"),
            pytest.param(xdem.coreg.ICP, "tba", "terrain_name", "analytical", id="icp-moving-terrain-analytical"),
        ],
    )
    def test_propagate_uncertainty_coreg__nan_predictors(
        self,
        curved_dem: xdem.DEM,
        solver: type[xdem.coreg.AffineCoreg],
        side: str,
        predictor_kind: str,
        propagation_method: Literal["analytical", "numerical"],
    ) -> None:
        """Checks that predictors with NaNs still provide proper fits and numerical simulations."""

        # Add a NaN in the DEM, which will propagate to slope
        reference = curved_dem.copy()
        reference.data[4, 5] = np.ma.masked
        source = reference + 0.2
        slope = (reference if side == "ref" else source).slope()

        # Define small slope-dependent errors across the possible range of slopes in degrees
        statistics = pd.DataFrame(
            {"std": [0.001, 0.002], "count": [100, 100]}, index=pd.Index([0.0, 90.0], name="slope")
        )
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        predictor = {"array": slope.data, "raster": slope, "terrain_name": "slope"}[predictor_kind]
        original_reference, original_source = reference.copy(), source.copy()

        # Select analytical covariance or three numerical draws using the input predictor representation
        propagation_options: dict[str, Any] = {"method": propagation_method, "random_state": 12}
        if propagation_method == "numerical":
            propagation_options.update(n_simulations=3, on_error="raise")
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            source,
            solver(only_translation=True, subsample=1),
            error_structure={side: errors},
            error_predictors={side: {"slope": predictor}},
            **propagation_options,
        )

        # Check the correction and uncertainty from the finite slope observations
        np.testing.assert_allclose(summary.estimate.iloc[:3], (0, 0, -0.2), rtol=0, atol=1e-7)
        assert np.all(summary.std.iloc[:3] > 0)
        if propagation_method == "numerical":
            assert summary.n_success == 3
        else:
            # Analytical uncertainty matches a direct fit, without the mask used to prepare numerical fields
            expected = solver(only_translation=True, subsample=1).fit(
                reference,
                source,
                error_structure={side: errors},
                error_predictors={side: {"slope": slope}},
                random_state=12,
            )
            expected_errors = expected.meta["outputs"]["fitorbin"]["fit_perr"][:3]
            np.testing.assert_allclose(summary.std.iloc[:3], expected_errors, rtol=1e-12)

        # Check unchanged input elevations and slope values
        np.testing.assert_array_equal(reference.data, original_reference.data)
        np.testing.assert_array_equal(source.data, original_source.data)
        np.testing.assert_array_equal(slope.data, (reference if side == "ref" else source).slope().data)

    @pytest.mark.parametrize("solver", [xdem.coreg.LZD, xdem.coreg.ICP])
    def test_propagate_uncertainty_coreg__matching_gls_estimates(
        self, icp_dem: gu.Raster, solver: type[xdem.coreg.AffineCoreg]
    ) -> None:
        """Checks that analytical and numerical GLS fits have identical unperturbed estimates and centroids."""

        # Leave elevations finite where predictors are missing
        source = icp_dem + 0.2
        predictor = np.ones(icp_dem.shape)
        predictor[:2] = np.nan
        statistics = pd.DataFrame(
            {"std": [0.01, 0.02], "count": [100, 100]}, index=pd.Index([0.0, 1.0], name="position")
        )
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        errors = {side: gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)]) for side in ("ref", "tba")}
        predictors = {side: {"position": predictor} for side in errors}
        options: dict[str, Any] = {
            "error_structure": errors,
            "error_predictors": predictors,
            "fit_kwargs": {"error_structure": errors, "error_predictors": predictors},
            "random_state": 42,
        }
        method = solver(only_translation=True, subsample=40)

        # Use the same GLS fit for both propagation methods
        analytical = xdem.uncertainty.propagate_uncertainty_coreg(
            icp_dem, source, method, method="analytical", **options
        )
        numerical = xdem.uncertainty.propagate_uncertainty_coreg(
            icp_dem,
            source,
            method,
            method="numerical",
            n_simulations=2,
            on_error="raise",
            **options,
        )

        pd.testing.assert_series_equal(analytical.estimate, numerical.estimate)
        np.testing.assert_array_equal(analytical.output["centroid"], numerical.output["centroid"])

    @pytest.mark.parametrize("only_translation", [True, False])
    @pytest.mark.parametrize(
        "solver, optimizer, legacy, seed",
        [
            pytest.param(xdem.coreg.LZD, "gls", False, 12, id="lzd-native"),
            pytest.param(xdem.coreg.LZD, "gls", True, 2, id="lzd-legacy-gls"),
            pytest.param(xdem.coreg.LZD, "tls", True, 2, id="lzd-legacy-tls"),
            pytest.param(xdem.coreg.ICP, "gls", False, 12, id="icp-integer-seed"),
            pytest.param(xdem.coreg.ICP, "gls", False, None, id="icp-no-seed"),
            pytest.param(xdem.coreg.ICP, "gls", False, np.random.default_rng(12), id="icp-generator"),
        ],
    )
    def test_propagate_uncertainty_coreg__refit(
        self,
        icp_dem: gu.Raster,
        only_translation: bool,
        solver: type[xdem.coreg.AffineCoreg],
        optimizer: str,
        legacy: bool,
        seed: int | np.random.Generator | None,
    ) -> None:
        """Checks that analytical propagation refits (when the method is already fitted)."""

        if legacy and optimizer == "gls":
            pytest.importorskip("linear_operator")

        # 1/ Fit curved terrain using native models or the legacy LZD error arguments
        reference = icp_dem
        source = reference + 0.2
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])
        options: dict[str, Any] = {"force_opti": optimizer}
        if legacy:
            sigma = np.full(reference.shape, 0.1)
            correlation = (lambda distance: np.exp(-((distance / 2) ** 2)), 2)
            options.update(sig_tba=sigma, corr_tba=correlation)
            if optimizer == "tls":
                options.update(sig_ref=sigma, corr_ref=correlation)
        else:
            options["error_structure"] = {"tba": errors}
        method = solver(
            only_translation=only_translation, subsample=60, tolerance_translation=1e-6, tolerance_rotation=1e-6
        )
        fitted = method.fit(reference, source, random_state=seed, **options)
        matrix = fitted.to_matrix().copy()

        # 2/ Change both the elevations and errors, using the same seed for a fresh fit and the saved method
        changed_source = source + 0.2
        changed_errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.3)])
        fresh = solver(
            only_translation=only_translation, subsample=60, tolerance_translation=1e-6, tolerance_rotation=1e-6
        )
        arguments: dict[str, Any] = dict(
            error_structure=changed_errors, method="analytical", random_state=12, at=["tx", "rz"]
        )
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, changed_source, fresh, **arguments)
        summary = xdem.uncertainty.propagate_uncertainty_coreg(reference, changed_source, fitted, **arguments)

        # 3/ Compare the saved method's summary with a fresh fit using the input errors
        labels = ["tx", "ty", "tz", "rx", "ry", "rz"]
        expected_covariance = expected.output["parameter_covariance"]
        expected_std = expected.std
        covariance = summary.output["parameter_covariance"]

        # Check covariance labels, symmetry and standard deviations
        assert type(summary) is xdem.uncertainty.PropagationSummary
        assert list(covariance.index) == list(covariance.columns) == labels
        np.testing.assert_array_equal(covariance, expected_covariance)
        np.testing.assert_array_equal(summary.std, expected_std)
        np.testing.assert_allclose(np.diag(covariance), summary.std**2, rtol=1e-12, atol=1e-20)
        np.testing.assert_allclose(covariance, covariance.T, rtol=0, atol=1e-15)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-15

        # Check the original fit, selected outputs and parameter estimates
        np.testing.assert_array_equal(summary.mean, summary.estimate)
        np.testing.assert_allclose(summary.estimate, expected.estimate)
        np.testing.assert_array_equal(fitted.to_matrix(), matrix)
        assert summary.output["centroid"] == fitted.centroid()
        assert list(summary.selection.index) == ["tx", "rz"]
        assert summary.coregs is None
        if solver is xdem.coreg.ICP:
            assert summary.output["fixed_correspondences"] is True

    @pytest.mark.parametrize("solver", [xdem.coreg.LZD, xdem.coreg.ICP])
    def test_propagate_uncertainty_coreg__refit_without_saved_errors(
        self, icp_dem: gu.Raster, solver: type[xdem.coreg.AffineCoreg]
    ) -> None:
        """Checks that a fit without saved uncertainty is refitted with the input error model."""

        # Fit without an error model and mark the old convergence result as unsuitable for reuse
        fitted = solver(only_translation=True, subsample=60).fit(icp_dem, icp_dem + 0.2, random_state=12)
        fitted.meta["outputs"]["iterative"]["converged"] = False
        original_matrix = fitted.to_matrix().copy()
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Calculate analytical uncertainty from a new fit to the inputs
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            icp_dem, icp_dem + 0.2, fitted, error_structure=errors, method="analytical", random_state=12
        )

        # Check nonzero translation errors and preserve the original method's metadata
        assert np.all(summary.std.iloc[:3] > 0)
        np.testing.assert_array_equal(fitted.to_matrix(), original_matrix)
        assert not fitted.meta["outputs"]["iterative"]["converged"]
        assert "fit_pcov" not in fitted.meta["outputs"]["fitorbin"]

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

        # Exclude edges so simulated horizontal shifts do not change the valid observation sample
        mask = np.zeros(reference.shape, dtype=bool)
        mask[1:-1, 1:-1] = True

        # Create a Dataset with separate raster and point coordinates
        source_array = source.to_xarray()
        if input_kind == "pointcloud":
            source_array = source_array.rename({"x": "x_point", "y": "y_point"})
        dataset = xr.Dataset({"reference": reference.to_xarray(), "elevation": source_array})
        original = dataset.copy(deep=True)
        options: dict[str, Any] = dict(
            error_structure={error_applied_to: errors},
            method="numerical",
            random_state=2,
            n_simulations=3,
            return_samples=True,
            inlier_mask=mask,
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

    @pytest.mark.parametrize("crs, unit", [(32632, "metre"), (2227, "US survey foot")])
    def test_propagate_uncertainty_coreg__file_crs_units(self, tmp_path: Path, crs: int, unit: str) -> None:
        """Checks that a file-backed raster gives a parameter summary with its projected coordinate units."""

        # Write a projected raster and reopen it with the CRS read by Rasterio
        values = np.arange(48.0).reshape(6, 8)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 12), crs)
        filename = tmp_path / "reference.tif"
        raster.to_file(filename)
        reference = gu.Raster(filename)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Simulate the correction of a known vertical offset using the file-backed reference
        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference,
            reference + 5,
            xdem.coreg.VerticalShift(vshift_reduc_func=np.mean),
            error_structure=errors,
            n_simulations=4,
            random_state=2,
        )

        # Check that unit lookup succeeds without changing the CRS or the fitted correction
        assert summary.output["translation_units"] == unit
        assert summary.output["crs"] == reference.crs
        assert summary.estimate["tz"] == pytest.approx(-5)


class TestCoregUncertaintyAnalyticalVsNumerical:
    """Test module for analytical uncertainty compared with numerical uncertainty from simulations for coreg."""

    @pytest.mark.parametrize("only_translation", [True, False], ids=["translation", "rigid"])
    @pytest.mark.parametrize(
        "gls_method, correlated",
        [
            pytest.param("approximate", False, id="independent"),
            pytest.param("approximate", True, id="correlated-approximate"),
            pytest.param("exact", True, id="correlated-exact"),
            pytest.param("inducing", True, id="correlated-inducing"),
        ],
    )
    def test_propagate_uncertainty_coreg__gls_matches_simulation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        correlated: bool,
        only_translation: bool,
        gls_method: str,
    ) -> None:
        """Checks that analytical GLS standard errors agree with repeated fits under the same elevation errors."""

        if correlated:
            pytest.importorskip("skgstat")

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        from functools import partial

        from xdem.coreg.affine import (
            _gls_approximate,
            _gls_inducing,
            _gls_lazy_gpytorch,
        )

        # Force GPyTorch block size solves even for this small DEM
        if gls_method == "exact":
            monkeypatch.setattr(
                "xdem.coreg.affine._gls_lazy_gpytorch",
                partial(_gls_lazy_gpytorch, block_size=32, max_preconditioner_size=16),
            )
        elif gls_method == "approximate":
            monkeypatch.setattr("xdem.coreg.affine._gls_approximate", partial(_gls_approximate, block_size=32))
        else:
            monkeypatch.setattr("xdem.coreg.affine._gls_inducing", partial(_gls_inducing, block_size=32))

        # 1/ Create a small curved DEM and its error model
        # Curvature constrains all six parameters, and the border excludes observations that could leave the grid
        rows, columns = np.indices((12, 13))
        values = 5 * np.sin(columns / 3) + 3 * np.cos(rows / 4) + 2 * np.sin((columns + rows) / 5)
        reference = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 24), 32632, nodata=np.nan)
        source = reference.copy()
        mask = np.zeros(reference.shape, dtype=bool)
        mask[2:-2, 2:-2] = True

        # Errors below the fit tolerances check the linear approximation before further resampling
        magnitude = 1e-5
        model = VariogramModel("gaussian", effective_range=4, partial_sill=1) if correlated else None
        component = gu.ErrorComponent("measurement", magnitude, correlation=model)
        errors = gu.ErrorStructure([component])
        correlation = (component.predict_correlation, 4) if correlated else None
        fit_options = {
            "sig_tba": np.full(reference.shape, magnitude),
            "corr_tba": correlation,
            "force_opti": "gls",
            "gls_method": gls_method,
            "gls_neighbors": 10,
            "gls_rank": 8,
        }
        options: dict[str, Any] = {
            "error_structure": errors,
            "inlier_mask": mask,
            "fit_kwargs": fit_options,
            "random_state": 42,
        }
        method = xdem.coreg.LZD(only_translation=only_translation, subsample=1)

        # 2/ Compare fitted standard errors with exact native Gaussian simulations
        _compare_simulated_uncertainty(
            reference, source, method, options, only_translation, n_simulations=20, check_mean=True
        )

    @pytest.mark.parametrize(
        "solver, only_translation, point_input",
        [
            pytest.param(xdem.coreg.LZD, True, None, id="lzd-translation-raster"),
            pytest.param(xdem.coreg.LZD, False, "tba", id="lzd-rigid-points"),
            pytest.param(xdem.coreg.ICP, False, None, id="icp-rigid-raster"),
            pytest.param(xdem.coreg.ICP, True, "tba", id="icp-translation-points"),
        ],
    )
    def test_propagate_uncertainty_coreg__both_inputs_match_simulation(
        self,
        icp_dem: gu.Raster,
        solver: Any,
        only_translation: bool,
        point_input: str | None,
    ) -> None:
        """Checks that both input models predict simulated translation and rotation errors."""

        pytest.importorskip("skgstat")

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # 1/ Create two independent spatial error models with different magnitude predictors
        reference = icp_dem
        moving = reference + 0.2
        if point_input == "tba":
            # Use the raster's native coordinates so pixel interpretation does not add a horizontal shift
            x, y = reference.coords(grid=True)
            moving = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), moving.data.ravel(), crs=reference.crs)
        sources = {"ref": reference, "tba": moving}
        models = {}
        predictors = {}
        for name, source in sources.items():
            quality = np.linspace(0, 1, source.data.size)
            predictor = "reference_quality" if name == "ref" else "moving_quality"
            magnitudes = [1e-4, 2e-4] if name == "ref" else [2e-4, 3e-4]
            statistics = pd.DataFrame(
                {"std": magnitudes, "count": [100, 100]}, index=pd.Index([0.0, 1.0], name=predictor)
            )
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
            correlation = VariogramModel("gaussian", effective_range=6 if name == "ref" else 4, partial_sill=1)
            model = gu.ErrorStructure(
                [
                    gu.ErrorComponent("measurement", 5e-5),
                    gu.ErrorComponent("terrain", magnitude, correlation),
                ]
            )
            models[name] = model
            predictors[name] = {predictor: quality.reshape(source.data.shape)}

        # 2/ Propagate both fields through identical analytical and numerical fits
        mask = np.zeros(icp_dem.shape, dtype=bool)
        mask[2:-2, 2:-2] = True
        method = solver(
            only_translation=only_translation,
            subsample=1,
            max_iterations=50,
            tolerance_translation=1e-7,
            tolerance_rotation=1e-7,
        )
        options: dict[str, Any] = dict(
            error_structure=models, error_predictors=predictors, inlier_mask=mask, random_state=42
        )
        analytical = _compare_simulated_uncertainty(reference, moving, method, options, only_translation)

        # Check the original models and all six covariance dimensions
        assert analytical.error_structure is models
        assert analytical.output["parameter_covariance"].shape == (6, 6)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize(
        "layout, only_translation, error_applied_to, correlated",
        [
            pytest.param("raster", True, "ref", False, id="raster-translation-reference-independent"),
            pytest.param("raster", False, "tba", True, id="raster-rigid-moving-correlated"),
            pytest.param("points", False, "ref", True, id="points-rigid-reference-correlated"),
            pytest.param("points", True, "tba", False, id="points-translation-moving-independent"),
            pytest.param("inverted", True, "ref", True, id="inverted-translation-reference-correlated"),
            pytest.param("inverted", False, "tba", False, id="inverted-rigid-moving-independent"),
        ],
    )
    def test_propagate_uncertainty_coreg__icp_matches_simulation(
        self,
        icp_dem: gu.Raster,
        layout: str,
        only_translation: bool,
        error_applied_to: Literal["ref", "tba"],
        correlated: bool,
    ) -> None:
        """Checks that analytical ICP translation and rotation errors agree with exact native Gaussian simulations."""

        if correlated:
            pytest.importorskip("skgstat")

        pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # 1/ Create stable correspondences with a nonzero transform and an interior inlier mask
        # A raster shift requires interpolation, while transformed native points preserve exact geometry
        reference = icp_dem
        if layout == "raster":
            moving = reference.translate(0.15, -0.12) + 0.1
        else:
            x, y = reference.coords(grid=True)
            points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), reference.data.ravel(), crs=reference.crs)
            angles = (0, 0, 0) if only_translation else (0.12, -0.09, 0.15)
            planted = xdem.coreg.matrix_from_translations_rotations(0.18, -0.12, 0.24, *angles)
            center = (float(x.mean()), float(y.mean()), float(reference.data.mean()))
            transformed = xdem.coreg.apply_matrix(points.gdf, planted, centroid=center)
            moving = gu.PointCloud(transformed, data_name="z")
            if layout == "inverted":
                reference, moving = moving, reference
        mask = np.zeros(icp_dem.shape, dtype=bool)
        mask[2:-2, 2:-2] = True
        method = xdem.coreg.ICP(
            only_translation=only_translation,
            subsample=1,
            sampling_strategy="nearest_xy" if layout == "raster" else "independent",
            tolerance_translation=1e-9,
            tolerance_rotation=1e-8,
            max_iterations=50,
        )
        magnitude = 1e-4
        model = VariogramModel("gaussian", effective_range=6, partial_sill=1) if correlated else None
        component = gu.ErrorComponent("measurement", magnitude, correlation=model)
        errors = gu.ErrorStructure([component])

        # 2/ Draw native Gaussian fields independently of the analytical covariance implementation
        options: dict[str, Any] = dict(error_structure={error_applied_to: errors}, inlier_mask=mask, random_state=42)
        _compare_simulated_uncertainty(reference, moving, method, options, only_translation, check_mean=True)


@pytest.fixture
def coreg_dem_paths(icp_dem: gu.Raster, tmp_path: Path) -> tuple[Path, ...]:
    """Write elevations and error predictors for coregistration uncertainty comparisons."""

    quality = np.linspace(0, 1, icp_dem.data.size).reshape(icp_dem.shape)
    quality[3, 4] = np.nan
    paths = tuple(tmp_path / f"input-{index}.tif" for index in range(4))
    rasters = (icp_dem, icp_dem + 0.2, icp_dem.copy(new_array=quality), icp_dem.copy(new_array=1 - quality))
    for raster, path in zip(rasters, paths):
        raster.to_file(path)
    return paths


class TestCoregistrationPropagationChunked:
    """Test module for analytical and numerical coregistration uncertainty with Dask and multiprocessing."""

    @pytest.mark.parametrize(
        "backend, coreg_method, fit_options",
        [
            pytest.param("dask", xdem.coreg.ICP, {"force_opti": "gls", "gls_method": "approximate"}, id="gls-dask"),
            pytest.param("mp", xdem.coreg.LZD, {"force_opti": "tls", "tls_method": "spatial"}, id="tls-mp"),
        ],
    )
    def test_propagate_uncertainty_coreg__chunk_invariance_analytical(
        self,
        backend: str,
        coreg_method: type[xdem.coreg.AffineCoreg],
        fit_options: dict[str, Any],
        coreg_dem_paths: tuple[Path, ...],
    ) -> None:
        """Checks that chunked heights, gradients and predictors give exactly the in-memory parameter covariance."""

        if fit_options.get("gls_method") in ("approximate", "exact"):
            pytest.importorskip("linear_operator")

        # 1/ Open inputs and define variable error models
        reference, source = (DEM(path, load_data=True) for path in coreg_dem_paths[:2])
        quality = np.linspace(0, 1, reference.data.size).reshape(reference.shape)
        quality[3, 4] = np.nan
        statistics = pd.DataFrame(
            {"std": [0.01, 0.02], "count": [100, 100]}, index=pd.Index([0.0, 1.0], name="quality")
        )
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        fit_options = dict(fit_options)
        # Use the same spatial correlation for both elevation errors
        correlation = VariogramModel("gaussian", effective_range=8, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude, correlation)])
        models = {"ref": errors, "tba": errors}
        predictors = {"ref": {"quality": quality}, "tba": {"quality": 1 - quality}}
        estimator = coreg_method(subsample=40, only_translation=False)
        options: dict[str, Any] = dict(
            error_structure=models, method="analytical", random_state=12, fit_kwargs=fit_options
        )

        # Calculate the in-memory reference
        expected = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, source, estimator, error_predictors=predictors, **options
        )

        # 2/ Propagate errors from the chunked elevations
        if backend == "dask":
            # 2.1/ If using Dask
            pytest.importorskip("dask")
            # Use different layouts and shorter final chunks
            chunks = ((3, 5), (4, 7), (4, 4), (5, 3))
            inputs = [
                open_dem(path, chunks={"y": rows, "x": columns}).dem
                for path, (rows, columns) in zip(coreg_dem_paths, chunks)
            ]
            config = None
            first, second, ref_quality, tba_quality = inputs
            predictors = {"ref": {"quality": ref_quality}, "tba": {"quality": tba_quality}}
            assert all(not value.is_loaded for value in inputs)
            actual = xdem.uncertainty.propagate_uncertainty_coreg(
                first, second, estimator, error_predictors=predictors, mp_config=config, **options
            )
            assert all(not value.is_loaded for value in inputs)
        else:
            # 2.2/ If using MP
            inputs = [DEM(path) for path in coreg_dem_paths]
            config = MultiprocConfig(chunks=(3, 5))
            first, second, ref_quality, tba_quality = inputs
            predictors = {"ref": {"quality": ref_quality}, "tba": {"quality": tba_quality}}
            assert all(not value.is_loaded for value in inputs)
            actual = xdem.uncertainty.propagate_uncertainty_coreg(
                first, second, estimator, error_predictors=predictors, mp_config=config, **options
            )
            assert all(not value.is_loaded for value in inputs)

        # 3/ Check equality of estimates and parameter covariance
        pd.testing.assert_series_equal(actual.estimate, expected.estimate, check_exact=True)
        pd.testing.assert_series_equal(actual.std, expected.std, check_exact=True)
        pd.testing.assert_frame_equal(
            actual.output["parameter_covariance"], expected.output["parameter_covariance"], check_exact=True
        )

    @pytest.mark.parametrize(
        "backend, coreg_method, predictor_kind, fit_errors",
        [
            ("dask", xdem.coreg.ICP, "array", True),
            ("mp", xdem.coreg.LZD, "none", False),
            ("dask", xdem.coreg.CPD, "scalar", False),
        ],
    )
    def test_propagate_uncertainty_coreg__chunk_invariance_numerical(
        self,
        backend: str,
        coreg_dem_paths: tuple[Path, ...],
        coreg_method: type[xdem.coreg.AffineCoreg],
        predictor_kind: str,
        fit_errors: bool,
    ) -> None:
        """Checks that uncertainty propag for coreg on chunked data gives exactly the in-memory result."""

        # 1/ Open inputs and define independent error models
        reference, source = (DEM(path, load_data=True) for path in coreg_dem_paths[:2])
        predictors = None
        # Small errors for fits to converge
        scalar_magnitude = 0.001 if fit_errors else 0.01
        magnitude: float | gu.ErrorMagnitude = scalar_magnitude
        if predictor_kind != "none":
            quality: Any = 0.5
            if predictor_kind == "array":
                # Add a NaN
                quality = np.linspace(1, 0, reference.data.size).reshape(reference.shape)
                quality[3, 4] = np.nan
            statistics = pd.DataFrame(
                {"std": [scalar_magnitude, 2 * scalar_magnitude], "count": [100, 100]},
                index=pd.Index([0.0, 1.0], name="quality"),
            )
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
            predictors = {"ref": {"quality": quality}, "tba": {"quality": 1 - quality}}
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        estimator = coreg_method(subsample=40, only_translation=True)
        options: dict[str, Any] = dict(
            error_structure={"ref": errors, "tba": errors},
            error_predictors=predictors,
            n_simulations=3,
            random_state=42,
            return_samples=True,
            return_coregs=True,
            on_error="raise",
        )
        # Use the same error models in fitting and propagation
        if fit_errors:
            options["fit_kwargs"] = dict(error_structure=options["error_structure"], error_predictors=predictors)

        # Calculate the in-memory reference with the same seed
        expected = xdem.uncertainty.propagate_uncertainty_coreg(reference, source, estimator, **options)

        # 2/ Propagate errors from the chunked elevations
        if backend == "dask":
            # 2.1/ If using Dask
            pytest.importorskip("dask")
            # Use different layouts and shorter final chunks
            chunks = ((3, 5), (4, 7))
            inputs = [
                open_dem(path, chunks={"y": rows, "x": columns}).dem
                for path, (rows, columns) in zip(coreg_dem_paths[:2], chunks)
            ]
            config = None
            first, second = inputs
            assert all(not value.is_loaded for value in inputs)
            actual = xdem.uncertainty.propagate_uncertainty_coreg(first, second, estimator, mp_config=config, **options)
            assert all(not value.is_loaded for value in inputs)
        else:
            # 2.2/ If using MP
            inputs = [DEM(path) for path in coreg_dem_paths[:2]]
            config = MultiprocConfig(chunks=(3, 5))
            first, second = inputs
            assert all(not value.is_loaded for value in inputs)
            actual = xdem.uncertainty.propagate_uncertainty_coreg(first, second, estimator, mp_config=config, **options)
            assert all(not value.is_loaded for value in inputs)

        # 3/ Check equality of summaries, samples and fitted transforms
        pd.testing.assert_series_equal(actual.estimate, expected.estimate, check_exact=True)
        pd.testing.assert_series_equal(actual.mean, expected.mean, check_exact=True)
        pd.testing.assert_series_equal(actual.std, expected.std, check_exact=True)
        pd.testing.assert_frame_equal(actual.samples, expected.samples, check_exact=True)

        # Fits with error models use the original elevations
        if fit_errors:
            analytical = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                source,
                estimator,
                error_structure=options["error_structure"],
                error_predictors=predictors,
                fit_kwargs=options["fit_kwargs"],
                random_state=42,
                method="analytical",
            )
            pd.testing.assert_series_equal(actual.estimate, analytical.estimate, check_exact=True)
            np.testing.assert_array_equal(actual.output["centroid"], analytical.output["centroid"])

        # Compare every saved simulation fit
        assert actual.coregs is not None and expected.coregs is not None
        for number, fitted in actual.coregs.items():
            reference_fit = expected.coregs[number]
            assert isinstance(fitted, xdem.coreg.AffineCoreg)
            assert isinstance(reference_fit, xdem.coreg.AffineCoreg)
            np.testing.assert_array_equal(fitted.to_matrix(), reference_fit.to_matrix())


class TestCoregistrationPropagationSummaryErrors:
    """Test module for missing covariance and invalid options in coregistration summary plots."""

    @pytest.mark.parametrize(
        "changes, options, message",
        [
            ({"output": {}, "samples": None}, {}, "require parameter covariance"),
            ({}, {"kind": "unknown"}, "kind must be"),
            ({}, {"coverage": 0}, "coverage must be"),
            ({}, {"coverage": 1}, "coverage must be"),
            ({}, {"coverage": np.nan}, "coverage must be"),
            ({}, {"axes": (None, None)}, "require four axes"),
        ],
    )
    def test_plot__error_ellipse_inputs(
        self,
        parameter_summary: xdem.uncertainty.PropagationSummary,
        changes: dict[str, Any],
        options: dict[str, Any],
        message: str,
    ) -> None:
        """Checks an error is raised for missing covariance, invalid coverage or an invalid ellipse layout."""

        pytest.importorskip("matplotlib")
        summary = replace(parameter_summary, **changes)

        with pytest.raises(ValueError, match=message):
            summary.plot(**{"kind": "ellipses", **options})

    @pytest.mark.parametrize(
        "changes, options, message",
        [
            ({"output": {}, "samples": None}, {}, "require parameter covariance"),
            ({}, {"coverage": 0}, "coverage must be"),
            ({}, {"coverage": 1}, "coverage must be"),
            ({}, {"coverage": np.nan}, "coverage must be"),
            ({}, {"axes": (None,)}, "require two axes"),
        ],
    )
    def test_plot__error_horizontal_vertical_inputs(
        self,
        parameter_summary: xdem.uncertainty.PropagationSummary,
        changes: dict[str, Any],
        options: dict[str, Any],
        message: str,
    ) -> None:
        """Checks an error is raised for missing covariance or invalid horizontal/vertical plot options."""

        pytest.importorskip("matplotlib")
        summary = replace(parameter_summary, **changes)

        with pytest.raises(ValueError, match=message):
            summary.plot(**{"kind": "horizontal_vertical", **options})


class TestPropagateUncertaintyCoregErrors:
    """Test module for unsupported coregistration uncertainty inputs, stored fits and failed simulations."""

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

    @pytest.mark.parametrize("stage", ["draw", "fit"])
    def test_propagate_uncertainty_coreg__error_failed_simulation(
        self, monkeypatch: pytest.MonkeyPatch, stage: str
    ) -> None:
        """Checks that failed draws or fits are recorded and excluded from summaries and saved Coregs."""

        # 1/ Prepare a failure in the first simulation, after the original fit when applicable
        reference = gu.Raster.from_array(np.arange(48, dtype=float).reshape(6, 8), Affine(2, 0, 0, 0, -2, 12), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])
        owner = gu.ErrorStructure if stage == "draw" else xdem.coreg.VerticalShift
        name = "generate_random_field" if stage == "draw" else "fit"
        original = getattr(owner, name)
        calls = iter([True, False, True, True] if stage == "fit" else [False, True, True])
        message = f"failed {stage}"

        def fail_first_simulation(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Fail the first simulation and otherwise use the original draw or fit."""

            if not next(calls):
                raise RuntimeError(message)
            return original(self, *args, **kwargs)

        # 2/ Run simulations and check the failure warning
        monkeypatch.setattr(owner, name, fail_first_simulation)
        with pytest.warns(UserWarning, match=message):
            summary = xdem.uncertainty.propagate_uncertainty_coreg(
                reference,
                reference + 5,
                xdem.coreg.VerticalShift(vshift_reduc_func=np.mean),
                error_structure=errors,
                n_simulations=3,
                random_state=2,
                return_coregs=True,
                return_samples=True,
            )

        # 3/ Compare saved Coregs, sample rows and uncertainty from successful fits
        assert summary.n_simulations == 3
        assert summary.n_success == 2
        assert summary.failures == {1: message}
        assert summary.samples is not None
        assert summary.samples.index.tolist() == [1, 2, 3]
        assert summary.samples.loc[1].isna().all()
        assert summary.coregs is not None
        assert list(summary.coregs) == [2, 3]
        for number, fitted in summary.coregs.items():
            assert fitted.to_matrix()[2, 3] == summary.samples.loc[number, "tz"]
        np.testing.assert_allclose(summary.std, summary.samples.std())

    @pytest.mark.parametrize(
        "options",
        [
            {"method": "point-to-point"},
            {"fit_loss_func": "soft_l1"},
            {"trim_residuals": True},
            {"linearized": False},
            {"sampling_strategy": "same_xy", "iterative_sampling": True},
            {"sampling_strategy": "nearest_xy", "iterative_sampling": True},
        ],
    )
    def test_propagate_uncertainty_coreg__error_icp_options(self, icp_dem: gu.Raster, options: dict[str, Any]) -> None:
        """Checks an error is raised for unsupported analytical ICP settings during fitting."""

        # Create a supported error model and an ICP option that changes the analytical estimator
        method = xdem.coreg.ICP(**options)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Fit the input elevations with analytical uncertainty
        with pytest.raises(NotImplementedError, match="Analytical ICP"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                icp_dem, icp_dem + 0.2, method, error_structure=errors, method="analytical"
            )

    @pytest.mark.parametrize("options", [{"trim_residuals": True}, {"linearized": False}, {"fit_loss_func": "soft_l1"}])
    def test_propagate_uncertainty_coreg__error_lzd_options(self, icp_dem: gu.Raster, options: dict[str, Any]) -> None:
        """Checks an error is raised for unsupported analytical LZD settings during fitting."""

        # Create an LZD estimator requiring numerical propagation
        method = xdem.coreg.LZD(**options)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.1)])

        # Request analytical uncertainty and use the existing LZD validation
        with pytest.raises(NotImplementedError, match="LZD error models require"):
            xdem.uncertainty.propagate_uncertainty_coreg(
                icp_dem, icp_dem + 0.2, method, error_structure=errors, method="analytical"
            )

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"error_structure": {}}, "nonempty mapping"),
            ({"error_structure": {"moving": None}}, "keys must"),
            ({"error_structure": {"tba": None}}, "Each error_structure value"),
            ({"error_predictors": {"ref": {}}}, "keys must name inputs"),
            ({"error_predictors": {"tba": []}}, "Each error_predictors value"),
        ],
    )
    def test_propagate_uncertainty_coreg__error_invalid_mapping(self, options: dict[str, Any], message: str) -> None:
        """Checks an error is raised for invalid mappings before reading elevation inputs."""

        # Create default moving errors and replace one mapping with invalid values
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.01)])
        options = {"error_structure": {"tba": errors}, **options}

        # Check validation before fitting or resolving either elevation
        with pytest.raises((TypeError, ValueError), match=message):
            xdem.uncertainty.propagate_uncertainty_coreg(None, None, xdem.coreg.LZD(), **options)

# Copyright (c) 2026 GeoUtils developers
#
# Adapted for xDEM from the GeoUtils uncertainty simulation module:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tools to help manage coregistration uncertainty."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
from geoutils._misc import import_optional
from geoutils.uncertainty import PropagationSummary as GeoUtilsPropagationSummary
from geoutils.uncertainty.predictors import _spatial_predictors
from geoutils.uncertainty.propagation import simulate
from numpy.typing import NDArray
from pyproj import CRS
from scipy.integrate import quad_vec
from scipy.stats import chi2, norm

from xdem.coreg import Coreg
from xdem.coreg.affine import ICP, LZD, _coreg_error_inputs
from xdem.coreg.base import _as_eager_elevation, translations_rotations_from_matrix
from xdem.uncertainty.propagation import _select_outputs


@dataclass(frozen=True, kw_only=True)
class PropagationSummary(GeoUtilsPropagationSummary):  # type: ignore[misc]
    """
    Fitted coregistration parameters and their uncertainty, with optional Coregs from numerical simulations.

    The summary includes estimate, mean and std as pandas Series indexed by tx, ty, tz, rx, ry and rz.
    Translations use coordinate units, rotations are Euler angles in radians about the centroid in output.

    With return_coregs=True, coregs stores the fitted Coreg objects and their fit metadata for each simulation
    number.

    plot() displays horizontal/vertical uncertainty ellipses when covariance is available, or parameter
    error bars otherwise. With kind="parameters", it shows the original fit and mean +/- one standard deviation.
    With kind="ellipses", it shows joint X/Y uncertainty and separate Z intervals.
    With kind="horizontal_vertical", one ellipse summarizes horizontal and vertical translation uncertainty,
    and another summarizes tilt and yaw uncertainty. Horizontal translation and tilt use sqrt(X² + Y²),
    with ellipses centered on their means. Rotations are displayed in microradians.
    Both ellipse views use analytical parameter covariance or samples saved with return_samples=True. For example::

        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, moving, xdem.coreg.LZD(), error_structure={"tba": errors}, return_coregs=True,
        )
        summary.plot()
        for number, fitted in summary.coregs.items():
            print(number, fitted.meta["outputs"])

    :param coregs: Optional fitted Coreg objects indexed by simulation number, starting at 1.
    """

    coregs: Mapping[int, Coreg] | None = None

    def plot(
        self,
        axes: Sequence[Any] | None = None,
        *,
        kind: Literal["parameters", "ellipses", "horizontal_vertical"] | None = None,
        coverage: float = 0.95,
        show_samples: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """
        Plot coregistration uncertainty for translations and rotations.

        By default, horizontal/vertical ellipses show parameter means, with gray zero reference lines.
        Without parameter covariance or all six saved parameter samples, the default shows parameter error bars.
        Use kind="parameters" to plot means with one standard deviation and the original fitted values.

        The ellipse view uses a Gaussian approximation to the parameter covariance. Each X/Y ellipse has
        the requested joint coverage, while each Z interval has the same marginal coverage.
        The horizontal/vertical view uses sqrt(X² + Y²) for horizontal translation and tilt.
        Numerical ellipses summarize these magnitudes and Z values across simulations. Analytical ellipses
        use deterministic quadrature of the joint Gaussian parameters to calculate magnitude moments.
        Coverage uses a Gaussian approximation. Rotations are converted to microradians for display.

        :param axes: Optional Matplotlib axes for translations and rotations.
            The ellipse view takes four axes: translation X/Y, translation Z, rotation X/Y and rotation Z.
            The horizontal/vertical view takes two axes, for translations and rotations.
        :param kind: None selects horizontal/vertical ellipses when covariance is available, otherwise error bars.
            "parameters" shows individual error bars. "ellipses" separates joint X/Y and Z uncertainty.
            "horizontal_vertical" shows horizontal magnitudes against vertical parameters.
        :param coverage: Probability enclosed by each ellipse or Z interval in the ellipse views.
        :param show_samples: Show saved simulations in the horizontal/vertical view as points with 50% opacity.
            Each simulation contributes its horizontal magnitude and vertical value.
        :param kwargs: Additional arguments for Matplotlib errorbar(). In the horizontal/vertical view,
            color and label also describe each ellipse, for comparing summaries on the same axes.

        :returns: Two parameter or horizontal/vertical axes, or four axes for the ellipse view.
        """

        if kind is None:
            has_covariance = self.output.get("parameter_covariance") is not None
            has_samples = self.samples is not None and {"tx", "ty", "tz", "rx", "ry", "rz"}.issubset(
                self.samples.columns
            )
            kind = "horizontal_vertical" if has_covariance or has_samples else "parameters"
        if kind not in {"parameters", "ellipses", "horizontal_vertical"}:
            raise ValueError("kind must be 'parameters', 'ellipses' or 'horizontal_vertical'.")
        if kind == "horizontal_vertical":
            return _plot_parameter_horizontal_vertical(self, axes, coverage, show_samples=show_samples, **kwargs)
        if kind == "ellipses":
            return _plot_parameter_ellipses(self, axes, coverage, **kwargs)

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        translation_axis, rotation_axis = axes
        units = (self.output.get("translation_units", "coordinate units"), "radian")
        kwargs = {"fmt": "o", "capsize": 3, "label": "Mean +/- 1 std", **kwargs}

        # Separate coordinate and angular units
        for axis, names, title, unit in zip(
            (translation_axis, rotation_axis),
            (("tx", "ty", "tz"), ("rx", "ry", "rz")),
            ("Translation", "Rotation"),
            units,
        ):
            parameters = list(names)
            axis.errorbar(parameters, self.mean.loc[parameters], yerr=self.std.loc[parameters], **kwargs)
            axis.plot(parameters, self.estimate.loc[parameters], "x", label="Original fit")
            axis.set_ylabel(f"{title} ({unit})")
            axis.legend()
        return translation_axis, rotation_axis


def _parameter_covariance(summary: PropagationSummary, coverage: float) -> pd.DataFrame:
    """Read finite parameter covariance from an analytical result or saved simulation samples."""

    # Analytical fits store covariance, while numerical fits can provide complete parameter draws
    if not np.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be finite and strictly between zero and one.")
    covariance = summary.output.get("parameter_covariance")
    if covariance is None and summary.samples is not None:
        covariance = summary.samples.dropna().cov()
    if not isinstance(covariance, pd.DataFrame):
        raise ValueError("Ellipse plots require parameter covariance or samples saved with return_samples=True.")
    parameters = ["tx", "ty", "tz", "rx", "ry", "rz"]
    if not set(parameters).issubset(covariance.index) or not set(parameters).issubset(covariance.columns):
        raise ValueError("Ellipse plots require covariance for all six coregistration parameters.")
    if not np.isfinite(covariance.loc[parameters, parameters].to_numpy()).all():
        raise ValueError("Ellipse plots require finite parameter covariance.")
    return covariance


def _parameter_covariance_ellipse(
    block: NDArray[np.float64], means: NDArray[np.float64], coverage: float, *, color: Any, label: str
) -> Any:
    """Construct a joint Gaussian confidence ellipse from a pair of parameter means and their covariance."""

    from matplotlib.patches import Ellipse

    joint_scale = chi2.ppf(coverage, df=2)
    # Eigenvectors orient the ellipse, and covariance eigenvalues give its squared radii
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    radii = np.sqrt(joint_scale * np.maximum(eigenvalues, 0))
    direction = eigenvectors[:, -1]
    angle = np.degrees(np.arctan2(direction[1], direction[0]))
    return Ellipse(
        xy=means,
        width=2 * radii[-1],
        height=2 * radii[0],
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=0.25,
        label=label,
    )


def _gaussian_horizontal_vertical_moments(
    means: NDArray[np.float64], covariance: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate the mean and covariance of (sqrt(X² + Y²), Z) for joint Gaussian parameters.

    Integrate the Gaussian Laplace transform to obtain the magnitude mean and its expected gradient.
    The second moment is E[X² + Y²], and Stein's identity gives covariance with Z from that gradient.
    Far from zero, Hermite quadrature of centered magnitude changes avoids cancellation in the variance.
    """

    # Rotate the horizontal covariance to independent Gaussian components
    horizontal_mean = means[:2]
    horizontal = np.hypot(*horizontal_mean)
    variances, directions = np.linalg.eigh(covariance[:2, :2])
    variances = np.maximum(variances, 0)
    total_variance = variances.sum()
    if total_variance == 0:
        return np.array([horizontal, means[2]]), np.diag([0.0, covariance[2, 2]])

    # A smooth magnitude can be integrated accurately using a small Hermite grid
    if horizontal**2 > 100 * total_variance:
        nodes, weights = np.polynomial.hermite.hermgauss(12)
        first, second = np.meshgrid(nodes, nodes, indexing="ij")
        independent = np.column_stack((first.ravel(), second.ravel())) * np.sqrt(2)
        weights = np.outer(weights, weights).ravel() / np.pi
        deviations = independent @ (directions * np.sqrt(variances)).T
        coordinates = horizontal_mean + deviations
        magnitudes = np.hypot(coordinates[:, 0], coordinates[:, 1])

        # Subtract squared lengths before division to resolve small changes around a large mean
        changes = (2 * (deviations @ horizontal_mean) + np.sum(deviations**2, axis=1)) / (magnitudes + horizontal)
        mean_change = weights @ changes
        magnitude_mean = horizontal + mean_change
        magnitude_variance = weights @ (changes - mean_change) ** 2
        gradient = weights @ (coordinates / magnitudes[:, None])
    else:
        # Normalize the second moment to one so the integral is independent of coordinate units
        scale = np.sqrt(horizontal**2 + total_variance)
        scaled_variances = variances / scale**2
        scaled_mean = directions.T @ horizontal_mean / scale

        def integrand(value: float) -> NDArray[np.float64]:
            """Evaluate the Gaussian Laplace transform for the magnitude mean and expected gradient."""

            squared = value**2
            denominator = 1 + 2 * squared * scaled_variances
            log_transform = -0.5 * np.log1p(2 * squared * scaled_variances).sum()
            log_transform -= squared * np.sum(scaled_mean**2 / denominator)
            mean_term = -np.expm1(log_transform) / squared
            gradient_terms = 2 * scaled_mean / denominator * np.exp(log_transform)
            return np.concatenate(([mean_term], gradient_terms))

        moments, _ = quad_vec(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-12)
        moments /= np.sqrt(np.pi)
        magnitude_mean = scale * moments[0]
        magnitude_variance = scale**2 * max(1 - moments[0] ** 2, 0)
        gradient = directions @ moments[1:]

    # The Gaussian cross covariance depends on the expected gradient of the magnitude
    cross_covariance = gradient @ covariance[:2, 2]
    transformed_covariance = np.array([[magnitude_variance, cross_covariance], [cross_covariance, covariance[2, 2]]])
    return np.array([magnitude_mean, means[2]]), transformed_covariance


def _plot_parameter_ellipses(
    summary: PropagationSummary, axes: Sequence[Any] | None, coverage: float, **kwargs: Any
) -> tuple[Any, ...]:
    """Draw Gaussian X/Y uncertainty ellipses and separate Z intervals for translations and rotations."""

    covariance = _parameter_covariance(summary, coverage)
    parameters = ["tx", "ty", "tz", "rx", "ry", "rz"]
    if axes is not None and len(axes) != 4:
        raise ValueError("Ellipse plots require four axes.")

    # Separate horizontal and vertical parameters so different error magnitudes have their own scales
    import_optional("matplotlib")
    import matplotlib.pyplot as plt

    if axes is None:
        _, grid = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True, width_ratios=[2, 1])
        axes = tuple(grid.flat)
    point_options = {"fmt": "o", "color": "C0", "label": "Mean", **kwargs}
    interval_options = {
        **point_options,
        "capsize": kwargs.get("capsize", 3),
        "label": kwargs.get("label", f"{coverage:.0%} interval"),
    }
    marginal_scale = norm.ppf((1 + coverage) / 2)
    units = (summary.output.get("translation_units", "coordinate units"), "radian")

    for index, (names, title, unit) in enumerate(
        zip((parameters[:3], parameters[3:]), ("Translation", "Rotation"), units)
    ):
        horizontal_axis, vertical_axis = axes[2 * index : 2 * index + 2]
        horizontal_names = names[:2]
        means = summary.mean.loc[horizontal_names].to_numpy(dtype=float)
        block = covariance.loc[horizontal_names, horizontal_names].to_numpy(dtype=float)

        ellipse = _parameter_covariance_ellipse(
            block, means, coverage, color=point_options["color"], label=f"{coverage:.0%} joint region"
        )
        horizontal_axis.add_patch(ellipse)
        horizontal_axis.errorbar(*means, **point_options)
        horizontal_axis.plot(*summary.estimate.loc[horizontal_names], "x", color="0.2", label="Original fit")
        horizontal_axis.set(
            xlabel=f"X {title.lower()} ({unit})", ylabel=f"Y {title.lower()} ({unit})", title=f"{title} in X/Y"
        )
        horizontal_axis.set_aspect("equal", adjustable="datalim")
        horizontal_axis.autoscale_view()
        horizontal_axis.margins(0.2)

        # Z uses the normal marginal interval rather than the two-dimensional ellipse scale
        vertical_name = names[2]
        interval = marginal_scale * summary.std.loc[vertical_name]
        vertical_axis.errorbar([0], [summary.mean.loc[vertical_name]], yerr=[interval], **interval_options)
        vertical_axis.plot([0], [summary.estimate.loc[vertical_name]], "x", color="0.2", label="Original fit")
        vertical_axis.set(
            xticks=[0], xticklabels=[vertical_name], ylabel=f"Z {title.lower()} ({unit})", title=f"{title} in Z"
        )
        for axis in (horizontal_axis, vertical_axis):
            axis.legend(fontsize="small")
    return tuple(axes)


def _plot_parameter_horizontal_vertical(
    summary: PropagationSummary, axes: Sequence[Any] | None, coverage: float, *, show_samples: bool, **kwargs: Any
) -> tuple[Any, ...]:
    """
    Plot horizontal magnitudes and vertical parameters with uncertainty ellipses around their means.

    _parameter_covariance() reads the parameter covariance. _gaussian_horizontal_vertical_moments()
    transforms the Gaussian parameters to magnitude moments, or we summarize saved simulations directly.
    _parameter_covariance_ellipse() draws the Gaussian approximation from these moments.
    Rotations are displayed in microradians.
    """

    # Validate the covariance before allocating the plotting panels
    covariance = _parameter_covariance(summary, coverage)
    if axes is not None and len(axes) != 2:
        raise ValueError("Horizontal/vertical plots require two axes.")

    import_optional("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import MaxNLocator

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    point_options = {"fmt": "x", "color": "0.2", "zorder": 4, **kwargs, "label": "_nolegend_"}
    color = kwargs.get("color", "C0")
    label = kwargs.get("label", "Horizontal/vertical")
    unit = summary.output.get("translation_units", "coordinate units")
    labels = (
        (f"Horizontal translation ({unit})", f"Vertical translation ({unit})"),
        ("Tilt (µrad)", "Yaw (µrad)"),
    )

    # Convert translations and rotations to horizontal magnitudes and vertical parameters
    for axis, names, title, scale, (xlabel, ylabel) in zip(
        axes,
        (("tx", "ty", "tz"), ("rx", "ry", "rz")),
        ("Translation", "Rotation"),
        (1.0, 1e6),
        labels,
    ):
        parameters = list(names)
        means = summary.mean.loc[parameters].to_numpy(dtype=float) * scale
        parameter_covariance = covariance.loc[parameters, parameters].to_numpy(dtype=float) * scale**2
        samples = None

        # Summarize one horizontal magnitude and vertical value per simulation
        if (
            summary.method == "numerical"
            and summary.samples is not None
            and set(names).issubset(summary.samples.columns)
        ):
            values = summary.samples.loc[:, parameters].dropna().to_numpy(dtype=float) * scale
            samples = np.column_stack((np.hypot(values[:, 0], values[:, 1]), values[:, 2]))
            center = samples.mean(axis=0)
            block = np.cov(samples, rowvar=False)
        else:
            center, block = _gaussian_horizontal_vertical_moments(means, parameter_covariance)

        # Show the Gaussian region and optional simulations around the parameter means
        ellipse = _parameter_covariance_ellipse(block, center, coverage, color=color, label=label)
        ellipse.set(alpha=None, facecolor=to_rgba(color, 0.15), linewidth=1.5)
        axis.add_patch(ellipse)
        if show_samples and samples is not None:
            axis.scatter(*samples.T, color=color, alpha=0.5, s=12, edgecolors="none", zorder=2)

        # Mark zero independently of the simulations
        axis.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        axis.axvline(0, color="0.85", linewidth=0.8, zorder=0)
        axis.errorbar(*center, **point_options)
        axis.set(xlabel=xlabel, ylabel=ylabel, title=f"{title} uncertainty ({coverage:.0%})")
        axis.autoscale_view()
        axis.margins(0.15)

        # Use readable ticks without forcing symmetric limits around zero
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    return tuple(axes)


def propagate_uncertainty_coreg(
    reference_elev: Any,
    to_be_aligned_elev: Any,
    coreg_method: Coreg,
    *,
    error_structure: gu.ErrorStructure | Mapping[str, gu.ErrorStructure],
    error_predictors: Mapping[str, Mapping[str, Any]] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    inlier_mask: Any = None,
    weights: Any = None,
    subsample: int | float | None = None,
    random_state: int | np.random.Generator | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    return_coregs: bool = False,
    **kwargs: Any,
) -> PropagationSummary:
    """
    Propagate and summarize uncertainties on translations and rotations from coregistration.

    Numerical propagation adds source errors over many simulations, repeating fit() with a fixed sampling seed.
    Analytical propagation supports linearized LZD and point-to-plane ICP.

    :param reference_elev: Reference raster or point cloud.
    :param to_be_aligned_elev: Raster or point cloud to align.
    :param coreg_method: Coregistration method, copied before fitting.
    :param error_structure: Independent models keyed by "ref" and/or "tba". A single model describes tba.
    :param error_predictors: Magnitude predictors keyed by the same input names.
    :param method: Repeated numerical fits, LZD fit uncertainties, or analytical point-to-plane ICP uncertainty.
    :param inlier_mask: Stable observations passed to fit().
    :param weights: Observation weights passed to fit().
    :param subsample: Optional replacement for the method's sample size.
    :param random_state: Seed controlling error draws and the fixed fit sample.
    :param fit_kwargs: Additional arguments passed unchanged to fit().
    :param return_coregs: Store each successful numerical fit as a Coreg, including its fit metadata.
    :param kwargs: GeoUtils simulation options, including n_simulations, at, return_samples and quantiles.
        random_field_options passes options to generate_random_field(), which approximates correlated fields by
        default.

    :returns: A propagation summary with six labelled parameters and optional fitted Coregs.
    """

    # 1/ Check inputs
    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if method == "analytical" and return_coregs:
        raise ValueError("Analytical propagation cannot return simulated Coreg objects.")
    models, predictor_inputs = _coreg_error_inputs(error_structure, error_predictors)
    if not models:
        raise TypeError("Supply at least one coregistration error model.")
    if "error_applied_to" in kwargs or "predictors" in kwargs:
        raise TypeError("Use error_structure and error_predictors mappings keyed by 'ref' and 'tba'.")

    # 2/ Prepare elevations and fitting options
    reference = _as_eager_elevation(reference_elev)
    aligned = _as_eager_elevation(to_be_aligned_elev)
    options = dict(fit_kwargs or {})
    options.update(inlier_mask=inlier_mask, weights=weights)
    if subsample is not None:
        options["subsample"] = subsample
    # Resolve terrain predictors separately on each uncertain DEM
    from xdem import terrain

    sources = {"ref": reference, "tba": aligned}
    resolved = {}
    for name, values in predictor_inputs.items():
        resolved[name] = {
            key: (
                terrain.get_terrain_attribute(sources[name], value)
                if isinstance(value, str)
                and isinstance(sources[name], gu.Raster)
                and value in terrain.available_attributes
                else value
            )
            for key, value in values.items()
        }

    if method == "analytical":
        # Fit with the error models and filter missing predictors during co-sampling
        legacy_errors = any(key in options for key in ("sig_ref", "sig_tba", "corr_ref", "corr_tba"))
        if isinstance(coreg_method, (LZD, ICP)) and not legacy_errors:
            options.update(error_structure=models, error_predictors=resolved)
        if isinstance(coreg_method, ICP) and reference.crs != aligned.crs:
            raise ValueError("Analytical ICP requires inputs in the same CRS.")
    else:
        # Select the same finite observations for the unperturbed fit and every numerical draw
        draw_predictors = {}
        for name, values in resolved.items():
            source = sources[name]
            aligned_values = _spatial_predictors(source, values, size=source.data.size)
            assert aligned_values is not None
            valid = np.isfinite(np.ma.asarray(source.data, dtype=float).filled(np.nan)).reshape(-1)
            for value in aligned_values.values():
                valid &= np.isfinite(value)
            if not np.any(valid):
                raise ValueError(f"Input {name!r} has no finite elevations and error predictors.")

            # Fill masked values to pass input checks
            # The elevation mask excludes these cells from fitting
            draw_predictors[name] = {
                key: value if np.isscalar(value) else np.where(valid, value, value[valid][0])
                for key, value in aligned_values.items()
            }

            # GLS/TLS filters predictors during co-sampling without changing elevation gradients
            if options.get("error_structure") is None:
                sources[name] = source.copy(
                    new_array=np.ma.masked_where(~valid.reshape(source.data.shape), source.data)
                )
        reference, aligned = sources["ref"], sources["tba"]

    operation_seed = int(np.random.default_rng(random_state).integers(0, 2**32))

    def fit(ref: Any, tba: Any) -> Coreg:
        """We fit a copy using the same observation sample for every realization."""

        return coreg_method.copy().fit(ref, tba, random_state=operation_seed, **options)

    # 3/ Fit unperturbed inputs for the output "estimate" (base output value of PropagationSummary)
    original = fit(reference, aligned)

    # 4/ Extract transformation parameters and metadata
    centroid = getattr(original, "centroid", lambda: None)()
    center = np.zeros(3) if centroid is None else np.asarray(centroid)
    labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])

    def parameters(fitted: Coreg) -> pd.Series:
        """Read matrix parameters, and correct to original centroid."""

        matrix = fitted.to_matrix().copy()
        fitted_centroid = getattr(fitted, "centroid", lambda: None)()
        fitted_center = np.zeros(3) if fitted_centroid is None else np.asarray(fitted_centroid)
        matrix[:3, 3] += (np.eye(3) - matrix[:3, :3]) @ (fitted_center - center)
        return pd.Series(
            translations_rotations_from_matrix(matrix, return_degrees=False),
            index=labels,
        )

    estimate = parameters(original)
    # Read coordinate units from either Rasterio or PyProj CRS objects
    crs = None if reference.crs is None else CRS.from_user_input(reference.crs)
    translation_units = crs.axis_info[0].unit_name if crs is not None and crs.is_projected else "coordinate units"
    output = {
        "centroid": centroid,
        "crs": reference.crs,
        "translation_units": translation_units,
        "rotation_units": "radian",
    }

    # 5/ Return analytical uncertainty
    if method == "analytical":
        if isinstance(original, ICP):
            covariance = original.meta["outputs"]["fitorbin"]["fit_pcov"]
            output["parameter_covariance"] = pd.DataFrame(covariance, index=labels, columns=labels)
            output["fixed_correspondences"] = True
        elif not isinstance(original, LZD) or "fit_perr" not in original.meta["outputs"].get("fitorbin", {}):
            raise ValueError("Analytical coregistration uncertainty requires an LZD fit with GLS/TLS uncertainties.")
        if isinstance(original, LZD) and "fit_pcov" in original.meta["outputs"]["fitorbin"]:
            native_covariance = original.meta["outputs"]["fitorbin"]["fit_pcov"]
            covariance = np.zeros((6, 6))
            covariance[: len(native_covariance), : len(native_covariance)] = native_covariance
            output["parameter_covariance"] = pd.DataFrame(covariance, index=labels, columns=labels)
        standard_errors = np.zeros(6)
        errors = original.meta["outputs"]["fitorbin"]["fit_perr"]
        standard_errors[: len(errors)] = errors
        return PropagationSummary(
            estimate=estimate,
            mean=estimate.copy(),
            std=pd.Series(standard_errors, index=labels),
            error_structure=error_structure,
            method="analytical",
            output=output,
            selection=_select_outputs(estimate, kwargs.pop("at", None)),
        )

    # 6/ Simulate numerical uncertainty
    fitted_coregs: list[Coreg] | None = [] if return_coregs else None

    def calculate(values: Mapping[str, Any]) -> pd.Series:
        """Refit the perturbed inputs and extract the six native parameters."""

        fitted = fit(values.get("ref", reference), values.get("tba", aligned))
        result = parameters(fitted)

        # Save fits that GeoUtils includes in its parameter summaries
        if fitted_coregs is not None and np.any(np.isfinite(result.to_numpy())):
            fitted_coregs.append(fitted)
        return result

    field_options = dict(kwargs.pop("random_field_options", {}) or {})

    def draw_error(generator: np.random.Generator) -> dict[str, Any]:
        """Draw independent native fields for the named elevation inputs."""

        values = {}
        for name, model in models.items():
            source = sources[name]
            error = model.generate_random_field(
                like=source, predictors=draw_predictors.get(name), random_state=generator, **field_options
            )
            values[name] = source.copy(new_array=source.data + error.data)
        return values

    selection = _select_outputs(estimate, kwargs.pop("at", None))
    if (kwargs.get("return_samples", False) or kwargs.get("quantiles")) and selection.empty:
        raise ValueError("Select outputs with at when requesting samples or quantiles.")
    kwargs.setdefault("on_error", "warn")
    kwargs.setdefault("n_simulations", 100)
    summary = simulate(
        estimate=estimate,
        draw_error=draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        random_state=random_state,
        output=output,
        **kwargs,
    )

    # Match saved fits to successful simulation numbers, including gaps after failed draws
    stored_coregs = None
    if fitted_coregs is not None:
        assert summary.n_simulations is not None
        numbers = [number for number in range(1, summary.n_simulations + 1) if number not in summary.failures]
        stored_coregs = dict(zip(numbers, fitted_coregs, strict=True))
    return PropagationSummary(
        **{item.name: getattr(summary, item.name) for item in fields(summary)}, coregs=stored_coregs
    )

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

from xdem.coreg import Coreg
from xdem.coreg.affine import LZD
from xdem.coreg.base import _as_eager_elevation, translations_rotations_from_matrix
from xdem.uncertainty.propagation import _select_outputs, _simulate


@dataclass(frozen=True, kw_only=True)
class PropagationSummary(GeoUtilsPropagationSummary):  # type: ignore[misc]
    """
    Fitted coregistration parameters and their uncertainty, with the original transformation available as coreg.

    estimate, mean and std are Series indexed by tx, ty, tz, rx, ry and rz. Translations use coordinate units;
    rotations are Euler angles in radians about coreg.centroid(). The fitted coreg also provides to_matrix()
    and apply(). GeoUtils supplies the usual sample tables, quantiles and uncertainty intervals.

    plot() displays translations and rotations on separate axes, with the original fit and mean +/- one
    standard deviation. For example::

        summary = xdem.uncertainty.propagate_uncertainty_coreg(
            reference, moving, xdem.coreg.LZD(), error_structure=errors,
        )
        summary.plot()
        aligned = summary.coreg.apply(moving)

    :param coreg: Transformation fitted to the original elevations before drawing numerical errors.
    """

    coreg: Coreg

    def plot(self, axes: Sequence[Any] | None = None, **kwargs: Any) -> tuple[Any, Any]:
        """
        Plot parameter means with one standard deviation and mark the original fitted values.

        :param axes: Two Matplotlib axes for translations and rotations; create them when omitted.
        :param kwargs: Additional arguments for Matplotlib errorbar().
        :returns: The translation and rotation axes.
        """

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
        translation_axis, rotation_axis = axes
        units = (self.output.get("translation_units", "coordinate units"), "radian")
        kwargs = {"fmt": "o", "capsize": 3, "label": "Mean +/- 1 std", **kwargs}

        # Separate the coordinate and angular units while comparing the original fit with propagated results
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


def propagate_uncertainty_coreg(
    reference_elev: Any,
    to_be_aligned_elev: Any,
    coreg_method: Coreg,
    *,
    error_structure: gu.ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    error_applied_to: Literal["ref", "tba"] = "tba",
    inlier_mask: Any = None,
    weights: Any = None,
    subsample: int | float | None = None,
    precoreg: bool = False,
    random_state: int | np.random.Generator | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PropagationSummary:
    """
    Summarize translations and Euler rotations from repeated fits or an existing GLS/TLS LZD fit.

    Numerical propagation adds source errors and repeats fit() with a fixed sampling seed. Translations use the
    coordinate units and rotations use radians, about the original fit's centroid recorded in output.
    Analytical propagation reads LZD's fit_perr directly: pass a fitted GLS/TLS LZD, or its sig_ref/sig_tba and
    corr_ref/corr_tba fitting arguments in fit_kwargs. An OLS fit has no analytical uncertainty here.

    :param reference_elev: Reference raster or point cloud.
    :param to_be_aligned_elev: Raster or point cloud to align.
    :param coreg_method: Coregistration method, copied before fitting.
    :param error_structure: GeoUtils error model used for numerical draws and recorded with the result.
    :param predictors: Source variables controlling error magnitude.
    :param method: Repeated numerical fits, or the uncertainties already returned by GLS/TLS LZD.
    :param error_applied_to: Dataset receiving numerical errors, "ref" or "tba".
    :param inlier_mask: Stable observations passed to fit().
    :param weights: Observation weights passed to fit().
    :param subsample: Optional replacement for the method's sample size.
    :param precoreg: Align the original moving dataset before repeating fits.
    :param random_state: Seed controlling error draws and the fixed fit sample.
    :param fit_kwargs: Additional arguments passed unchanged to fit().
    :param kwargs: GeoUtils simulation options, including n_samples, at, return_samples and quantiles.
    :returns: A coregistration PropagationSummary with the fitted transformation and six labelled parameters.
    """

    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if error_applied_to not in {"ref", "tba"}:
        raise ValueError("error_applied_to must be 'ref' or 'tba'.")
    reference = _as_eager_elevation(reference_elev)
    aligned = _as_eager_elevation(to_be_aligned_elev)
    options = dict(fit_kwargs or {})
    options.update(inlier_mask=inlier_mask, weights=weights)
    if subsample is not None:
        options["subsample"] = subsample
    operation_seed = int(np.random.default_rng(random_state).integers(0, 2**32))

    def fit(ref: Any, tba: Any) -> Coreg:
        """Fit a copy using the same observation sample for every realization."""

        return coreg_method.copy().fit(ref, tba, random_state=operation_seed, **options)

    original = coreg_method.copy() if method == "analytical" and coreg_method._fit_called else fit(reference, aligned)
    if precoreg:
        aligned = original.apply(aligned, resample=True)
        original = fit(reference, aligned)
    centroid = getattr(original, "centroid", lambda: None)()
    center = np.zeros(3) if centroid is None else np.asarray(centroid)
    labels = pd.Index(["tx", "ty", "tz", "rx", "ry", "rz"])

    def parameters(fitted: Coreg) -> pd.Series:
        """Read the native matrix parameters, expressing each fit about the original centroid."""

        matrix = fitted.to_matrix().copy()
        fitted_centroid = getattr(fitted, "centroid", lambda: None)()
        fitted_center = np.zeros(3) if fitted_centroid is None else np.asarray(fitted_centroid)
        matrix[:3, 3] += (np.eye(3) - matrix[:3, :3]) @ (fitted_center - center)
        return pd.Series(
            translations_rotations_from_matrix(matrix, return_degrees=False),
            index=labels,
        )

    estimate = parameters(original)
    translation_units = (
        reference.crs.axis_info[0].unit_name if reference.crs and reference.crs.is_projected else "coordinate units"
    )
    output = {
        "centroid": centroid,
        "crs": reference.crs,
        "translation_units": translation_units,
        "rotation_units": "radian",
    }
    if method == "analytical":
        if not isinstance(original, LZD) or "fit_perr" not in original.meta["outputs"].get("fitorbin", {}):
            raise ValueError("Analytical coregistration uncertainty requires an LZD fit with GLS/TLS uncertainties.")
        standard_errors = np.zeros(6)
        errors = original.meta["outputs"]["fitorbin"]["fit_perr"]
        standard_errors[: len(errors)] = errors
        return PropagationSummary(
            estimate=estimate,
            mean=estimate.copy(),
            std=pd.Series(standard_errors, index=labels),
            error_structure=error_structure,
            method="analytical",
            coreg=original,
            output=output,
            selection=_select_outputs(estimate, kwargs.pop("at", None)),
        )

    def calculate(value: Any) -> pd.Series:
        """Refit the perturbed dataset and extract the six native parameters."""

        fitted = fit(value, aligned) if error_applied_to == "ref" else fit(reference, value)
        return parameters(fitted)

    source = reference if error_applied_to == "ref" else aligned
    kwargs.setdefault("on_error", "warn")
    summary = _simulate(
        source,
        calculate,
        error_structure=error_structure,
        predictors=predictors,
        random_state=random_state,
        output=output,
        estimate=estimate,
        **kwargs,
    )
    return PropagationSummary(**{item.name: getattr(summary, item.name) for item in fields(summary)}, coreg=original)

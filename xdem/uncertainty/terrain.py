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

"""Propagate elevation errors through terrain attributes using GeoUtils."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import geoutils as gu
import numpy as np
from geoutils.operators import LocalData
from geoutils.operators.reducer import Sum
from geoutils.uncertainty import PropagationSummary
from geoutils.uncertainty.predictors import _spatial_predictors
from geoutils.uncertainty.propagation import _numeric_values, _wrap_like

from xdem.terrain import get_terrain_attribute
from xdem.terrain.surfit import _divider_method_coef, all_coefs
from xdem.uncertainty.propagation import _simulate


def _propagate_terrain(
    source: gu.Raster,
    *,
    attribute: str = "slope",
    error_structure: gu.ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    terrain_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PropagationSummary:
    """
    Repeat terrain calculations numerically or propagate slope/aspect filter weights analytically.

    get_terrain_attribute() supplies the original output and its surface filters supply the X/Y gradient weights.
    For analytical slope/aspect, their angle derivatives weight those same filters; GeoUtils handles the source
    covariance. The derivative is undefined at zero gradient, where analytical standard deviations are NaN.
    """

    options = dict(terrain_options or {})

    def calculate(value: Any) -> Any:
        """Calculate the requested terrain attribute with the same settings for every draw."""

        return get_terrain_attribute(value, attribute=attribute, **options)

    degrees = options.get("degrees", True)
    period = (360.0 if degrees else 2 * np.pi) if attribute == "aspect" else None
    if method == "numerical":
        return _simulate(
            source,
            calculate,
            error_structure=error_structure,
            predictors=predictors,
            output={"circular_period": period},
            **kwargs,
        )
    if attribute not in {"slope", "aspect"}:
        raise NotImplementedError("Analytical terrain propagation supports slope and aspect.")

    # Reuse the selected terrain method's convolution filters and pixel-spacing normalization
    estimate = calculate(source)
    names = {
        "Horn": ("h2", "h1"),
        "ZevenbergThorne": ("zt_h", "zt_g"),
        "Florinsky": ("fl_p", "fl_q"),
    }
    filter_names = names[options.get("surface_fit", "Horn")]
    filters = [np.flip(all_coefs[name]) / _divider_method_coef(source.res[0], name) for name in filter_names]
    radius = filters[0].shape[0] // 2
    values = _numeric_values(source)
    ids = np.arange(values.size).reshape(values.shape)
    x, y = source.coords(grid=True, force_offset="center")
    coordinates = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    empty = LocalData(np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=int))
    targets = [empty] * values.size
    nominal = _numeric_values(estimate).reshape(-1)

    # Differentiate the angle only where the terrain function has a complete, non-flat neighborhood
    for row, col in np.argwhere(np.isfinite(_numeric_values(estimate))):
        window = np.s_[row - radius : row + radius + 1, col - radius : col + radius + 1]
        patch = values[window]
        if patch.shape != filters[0].shape or not np.all(np.isfinite(patch)):
            continue
        gx, gy = (float(np.sum(kernel * patch)) for kernel in filters)
        squared_gradient = gx**2 + gy**2
        if squared_gradient == 0:
            continue
        if attribute == "slope":
            derivative = (gx * filters[0] + gy * filters[1]) / (np.sqrt(squared_gradient) * (1 + squared_gradient))
        else:
            derivative = (gy * filters[0] - gx * filters[1]) / squared_gradient
        if degrees:
            derivative *= 180 / np.pi
        positions = ids[window].ravel()
        targets[row * values.shape[1] + col] = LocalData(
            patch.ravel(),
            np.ones(patch.size, dtype=bool),
            positions,
            coordinates=coordinates[positions],
            sample_weights=derivative.ravel(),
        )

    # GeoUtils computes each weighted covariance sum; only restore the original terrain grid afterward
    aligned = _spatial_predictors(source, predictors, size=values.size)
    by_id = {name: dict(enumerate(value)) if np.ndim(value) else value for name, value in (aligned or {}).items()}
    operator = Sum()
    operator._uses_error_covariance = False
    at = kwargs.pop("at", None)
    labels = list(np.ndindex(values.shape))
    summary = gu.uncertainty.propagate(
        operator,
        targets,
        error_structure,
        predictors=by_id,
        method="analytical",
        nominal_estimate=nominal,
        output_labels=labels,
        at=[] if at is None else at,
        **kwargs,
    )
    moments = {
        name: _wrap_like(estimate, np.asarray(getattr(summary, name)).reshape(values.shape)) for name in ("mean", "std")
    }
    return replace(
        summary,
        estimate=estimate,
        **moments,
        output={"circular_period": period},
        metadata={"approximation": "first_order", "output_distribution": "unknown"},
    )

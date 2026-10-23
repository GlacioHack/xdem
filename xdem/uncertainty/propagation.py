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

"""Adapt elevation calculations to GeoUtils uncertainty simulation."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
from geoutils.operators import LocalData
from geoutils.operators.reducer import Sum
from geoutils.uncertainty import PropagationSummary
from geoutils.uncertainty.predictors import _spatial_predictors
from geoutils.uncertainty.propagation import (
    _numeric_values,
    _output_selection,
    simulate,
)


def _select_outputs(estimate: Any, at: Literal["all"] | Sequence[Hashable] | None) -> pd.DataFrame:
    """Select labelled outputs for saved draws without changing the simulation's spatial support."""

    values = _numeric_values(estimate)
    if at is None and (isinstance(estimate, (gu.Raster, gu.PointCloud)) or values.size > 256):
        return pd.DataFrame(columns=["flat_index", "estimate"])
    if isinstance(estimate, pd.Series):
        labels = estimate.index
    elif isinstance(estimate, gu.PointCloud):
        labels = estimate.gdf.index
    elif isinstance(estimate, pd.DataFrame):
        labels = pd.Index(
            [(row, col) for row in estimate.index for col in estimate.columns],
            tupleize_cols=False,
        )
    elif values.ndim:
        labels = pd.Index(list(np.ndindex(values.shape)), tupleize_cols=False)
    else:
        labels = pd.Index(["value"])
    if not labels.is_unique:
        raise ValueError("Uncertainty output labels must be unique.")

    # Spatial outputs require an explicit selection; small parameter tables select all outputs by default
    selection = _output_selection(labels, values.reshape(-1))
    if at is None:
        return selection
    if isinstance(at, str):
        if at != "all":
            raise ValueError("at must be 'all' or a sequence of output labels.")
        return selection[np.isfinite(selection.estimate)]
    keys = list(at)
    if len(set(keys)) != len(keys) or any(key not in labels for key in keys):
        raise ValueError("at must name distinct existing output labels.")
    return selection.iloc[[labels.get_loc(key) for key in keys]]


def _simulate(
    source: gu.Raster | gu.PointCloud,
    operation: Callable[[Any], Any],
    *,
    error_structure: gu.ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    n_samples: int = 100,
    random_state: int | np.random.Generator | None = None,
    at: Literal["all"] | Sequence[Hashable] | None = None,
    return_samples: bool = False,
    quantiles: Sequence[float] = (),
    max_sample_bytes: int = 268_435_456,
    on_error: Literal["raise", "warn"] = "raise",
    output: Mapping[str, Any] | None = None,
    estimate: Any = None,
) -> PropagationSummary:
    """
    Repeat an elevation operation with random fields and return GeoUtils' simulation summary.

    generate_random_field() handles the error components and their spatial correlation. simulate() accumulates
    moments, handles failed calculations and optionally stores selected draws or quantiles.
    """

    estimate = operation(source) if estimate is None else estimate
    selection = _select_outputs(estimate, at)
    if (return_samples or quantiles) and selection.empty:
        raise ValueError("Select outputs with at when requesting samples or quantiles.")

    def draw_error(generator: np.random.Generator) -> Any:
        """Draw one error field on the unchanged elevation support."""

        return error_structure.generate_random_field(like=source, predictors=predictors, random_state=generator)

    def calculate(error: Any) -> Any:
        """Add one error field to a copy of the original elevations and repeat the operation."""

        return operation(source.copy(new_array=source.data + error.data))

    return simulate(
        estimate=estimate,
        draw_error=draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        n_samples=n_samples,
        random_state=random_state,
        output=output,
        return_samples=return_samples,
        quantiles=quantiles,
        max_sample_bytes=max_sample_bytes,
        on_error=on_error,
    )


def _propagate_areas(
    source: gu.Raster | gu.PointCloud,
    *,
    areas: Mapping[Hashable, Any],
    error_structure: gu.ErrorStructure,
    statistic: Literal["mean", "integral"] = "mean",
    weights: Any = None,
    predictors: Mapping[str, Any] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    **kwargs: Any,
) -> PropagationSummary:
    """Propagate named area means or integrals using GeoUtils sampling and weighted sums."""

    from xdem.uncertainty.analytical import _area_support

    if statistic not in {"mean", "integral"}:
        raise ValueError("statistic must be 'mean' or 'integral'.")
    if isinstance(source, gu.PointCloud) and statistic == "integral" and weights is None:
        raise ValueError("A point cloud integral requires explicit point area weights.")
    values = _numeric_values(source).reshape(-1)
    area_weights = (
        np.ones(values.size)
        if weights is None
        else np.broadcast_to(_numeric_values(weights), source.data.shape).reshape(-1)
    )
    if np.any(~np.isfinite(area_weights)) or np.any(area_weights < 0):
        raise ValueError("Area weights must be finite and nonnegative.")

    # Build observation groups through the same GeoUtils sampling used by the older area functions
    targets = []
    for area in areas.values():
        coordinates, mask, _ = _area_support(area, source, None)
        positions = np.flatnonzero(mask)
        coefficients = area_weights[positions].copy()
        if coefficients.sum() <= 0:
            raise ValueError("Each area must contain observations with positive weight.")
        if statistic == "mean":
            coefficients /= coefficients.sum()
        elif isinstance(source, gu.Raster):
            coefficients *= abs(source.transform.determinant)
        targets.append(
            LocalData(
                values[positions],
                np.ones(len(positions), dtype=bool),
                positions,
                coordinates=coordinates,
                sample_weights=coefficients,
            )
        )
    labels = pd.Index(list(areas), tupleize_cols=False, name="area")
    operator = Sum()
    operator._uses_error_covariance = False

    def calculate(value: Any) -> pd.Series:
        """Sum each area's fixed observation weights after perturbing the source elevations."""

        raw = _numeric_values(value).reshape(-1)
        groups = [replace(local, values=raw[local.source_ids]) for local in targets]
        return pd.Series(operator.reduce_batch(groups, nodata_propagation="propagate"), index=labels)

    if method == "numerical":
        return _simulate(
            source,
            calculate,
            error_structure=error_structure,
            predictors=predictors,
            **kwargs,
        )

    # GeoUtils matches predictor values to the shared source IDs and calculates the covariance sums
    aligned = _spatial_predictors(source, predictors, size=values.size)
    by_id = {name: dict(enumerate(value)) if np.ndim(value) else value for name, value in (aligned or {}).items()}
    summary = gu.uncertainty.propagate(
        operator,
        targets,
        error_structure,
        predictors=by_id,
        method="analytical",
        output_labels=list(labels),
        **kwargs,
    )
    moments = {name: pd.Series(getattr(summary, name), index=labels) for name in ("estimate", "mean", "std")}
    return replace(summary, **moments)

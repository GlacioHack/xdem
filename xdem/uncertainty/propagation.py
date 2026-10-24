# Copyright (c) 2024 xDEM developers
# Copyright (c) 2026 GeoUtils developers
#
# Adapted for xDEM from the GeoUtils uncertainty simulation module:
# https://github.com/glaciohack/geoutils
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Estimate elevation errors and propagate them through terrain, area, and coregistration calculations."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal

import geoutils as gu
import numpy as np
import pandas as pd
from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_raster,
)
from geoutils.uncertainty import PropagationSummary
from geoutils.uncertainty.propagation import (
    _numeric_values,
    _output_selection,
    simulate,
)

from xdem import terrain
from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import _as_eager_elevation
from xdem.uncertainty.error_structure import ErrorStructure

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterLike
    from geoutils.vector.base import VectorLike

    from xdem.dem.base import DEMLike
    from xdem.epc.base import EPCLike

__all__ = ["estimate_error_structure", "propagate_uncertainty"]


############################################
# 1/ ESTIMATION FROM ELEVATION DIFFERENCES
############################################


def _estimation_input_family(values: Sequence[Any]) -> list[Any]:
    """Convert mixed raster and point inputs to GeoUtils accessors while preserving lazy data."""

    # Identify raster and point inputs; arrays and vector outlines need no conversion
    interfaces = []
    for value in values:
        interface = _get_raster_interface(value)
        if interface is None:
            interface = _get_pointcloud_interface(value)
        interfaces.append(interface)
    use_accessors = any(getattr(item, "_is_xr", False) or getattr(item, "_is_pd", False) for item in interfaces)
    if not use_accessors:
        return list(values)

    # Convert GeoUtils objects when another input uses an Xarray or GeoPandas accessor
    result = []
    for value, interface in zip(values, interfaces):
        if interface is None or interface._is_xr or getattr(interface, "_is_pd", False):
            result.append(value)
        elif _is_raster(interface):
            result.append(interface.to_xarray())
        else:
            frame = interface.gdf.copy()
            frame.attrs["data_name"] = interface.data_name
            result.append(frame)
    return result


def estimate_error_structure(
    source_elev: DEMLike | EPCLike,
    other: DEMLike | EPCLike | NDArrayf,
    *,
    predictors: Mapping[str, Any] | tuple[str, ...] | None = None,
    mask: RasterLike | VectorLike | NDArrayb | None = None,
    other_precision: Literal["negligible", "same"] = "negligible",
    at: Literal["self", "other"] | None = None,
    align: Literal["raise", "reproject"] = "reproject",
    mp_config: MultiprocConfig | None = None,
    **kwargs: Any,
) -> ErrorStructure:
    """
    Estimate elevation error magnitudes and spatial correlation using terrain predictors.

    Slope and maximum curvature are used as predictors from an input DEM by default.
    Pass predictors={} for constant magnitudes.

    :param source_elev: Elevation dataset whose error structure is estimated.
    :param other: Comparison elevation dataset, or an array matching the source pixels or point order.
    :param predictors: Named predictors or a tuple of terrain attribute names. None uses terrain defaults when
        possible.
    :param mask: Spatial or Boolean mask selecting stable terrain where differences represent error.
    :param other_precision: "negligible" assigns difference errors to the source. "same" assumes independent errors
        with the same magnitude and correlation in both datasets; GeoUtils divides differences by the square root of
        two.
    :param at: Comparison locations: "self", "other", or None to prefer point locations when present.
    :param align: "raise" reports mismatched grids or coordinate systems; "reproject" aligns them for comparison.
    :param mp_config: Worker and tile settings for multiprocessing estimation.
    :param kwargs: Additional options for GeoUtils ErrorStructure.estimate(), such as components, bins and
        random_state.
    :returns: The fitted GeoUtils error structure, with magnitudes in elevation units.
    """

    # Choose the DEM for slope/curv
    source_raster = _get_raster_interface(source_elev)
    other_raster = _get_raster_interface(other)
    terrain_source = source_raster if source_raster is not None else other_raster
    if predictors is None:
        predictors = ("slope", "max_curvature") if terrain_source is not None else ()
    predictor_specs = dict(predictors) if isinstance(predictors, Mapping) else {name: name for name in predictors}
    terrain_names = []
    for value in predictor_specs.values():
        if isinstance(value, str) and terrain_source is not None and value in terrain.available_attributes:
            if value not in terrain_names:
                terrain_names.append(value)

    # Preserve grid coordinates for boolean raster masks
    if isinstance(mask, np.ndarray) and mask.ndim > 1:
        for raster in (source_raster, other_raster):
            if raster is not None and mask.squeeze().shape == raster.shape:
                mask = raster.from_array(
                    mask.squeeze(), raster.transform, raster.crs, area_or_point=raster.area_or_point
                )
                break

    # Leave multiprocessing files available until fitting finishes
    with ExitStack() as temporary_files:
        if terrain_names:
            terrain_config = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
            attributes = terrain.get_terrain_attribute(terrain_source, terrain_names, mp_config=terrain_config)
            attributes = [attributes] if len(terrain_names) == 1 else attributes
            terrain_predictors = dict(zip(terrain_names, attributes))
            predictor_specs = {
                name: terrain_predictors.get(value, value) if isinstance(value, str) else value
                for name, value in predictor_specs.items()
            }

        source_elev, other, mask, *predictor_values = _estimation_input_family(
            [source_elev, other, mask, *predictor_specs.values()]
        )
        return ErrorStructure.estimate(
            source_elev,
            other=other,
            other_precision=other_precision,
            at=at,
            align=align,
            predictors=dict(zip(predictor_specs, predictor_values)),
            mask=mask,
            mp_config=mp_config,
            **kwargs,
        )


###################################
# 2/ SIMULATION HELPERS
###################################


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


###################################
# 3/ UNCERTAINTY PROPAGATION
###################################


def propagate_uncertainty(
    data: Any = None,
    operation: Literal["terrain", "spatial", "coreg"] | Callable[..., Any] | None = None,
    *,
    error_structure: gu.ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Mapping[str, Any]] | None = None,
    operation_options: Mapping[str, Any] | None = None,
    method: Literal["analytical", "numerical"] = "numerical",
    **kwargs: Any,
) -> PropagationSummary:
    """
    Propagate elevation errors through one terrain, area or coregistration calculation.

    Numerical propagation use random fields and simulation summaries directly through GeoUtils.
    Analytical propagation use derivatives for terrain, LZD coregistration uses a GLS fit.

    For example, propagate errors to a mean over all valid DEM pixels::

        summary = xdem.uncertainty.propagate_uncertainty(
            dem, "spatial", error_structure=errors, method="analytical",
            inputs={"spatial": {"areas": {"whole_dem": None}}},
        )
        summary.std["whole_dem"]

    :param data: Original raster, point cloud or supported elevation accessor.
    :param operation: "terrain", "spatial", "coreg", a callable, or None to summarize the source errors.
        A callable receives the perturbed elevation object and the supplied operation arguments.
    :param error_structure: GeoUtils error model for the uncertain elevations.
    :param predictors: Source magnitude predictors, including terrain attribute names for raster inputs.
    :param inputs: Fixed inputs under the operation name (or "custom" for a callable). Terrain accepts its usual
        arguments; spatial accepts an areas mapping and optional weights; coreg requires reference_elev.
    :param operation_options: Operation arguments, such as attribute for terrain, statistic for area means or
        integrals, and coreg_method for coregistration. Spatial integrals on point clouds require area weights.
    :param method: "numerical" for repeated calculations or "analytical" for existing propagation formulas.
    :param kwargs: GeoUtils propagation/simulation options, including n_samples, random_state, at,
        return_samples, quantiles, max_sample_bytes and on_error. Spatial outputs require at to save draws.
    :returns: GeoUtils PropagationSummary with the original result, propagated mean and standard deviation,
        and any requested selected draws or quantiles.
    """

    # 1/ Check inputs
    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if not isinstance(error_structure, gu.ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")

    # Fixed inputs for the selected operation override arguments in operation_options
    name = operation if isinstance(operation, str) else "custom"
    options = {**dict(operation_options or {}), **dict((inputs or {}).get(name, {}))}

    # 2/ Delegate methods to GeoUtils
    spatial_methods = {
        "grid",
        "reproject",
        "krige",
        "interp_points",
        "interp_at_points",
        "resample_at_points",
        "reduce_points",
        "reduce_at_points",
    }
    if (
        callable(operation)
        and getattr(operation, "__self__", None) is not None
        and operation.__name__ in spatial_methods
    ):
        return gu.uncertainty.propagate(
            operation,
            error_structure=error_structure,
            predictors=predictors,
            operation_kwargs=options,
            method=method,
            **kwargs,
        )

    # Custom callables require numerical propagation
    if method == "analytical" and callable(operation):
        raise NotImplementedError("Custom callables only support numerical propagation.")

    # 3/ Prepare elevation + potential predictors

    source = _as_eager_elevation(data)
    if not isinstance(source, (gu.Raster, gu.PointCloud)):
        raise TypeError("Uncertainty propagation requires a raster or point cloud source.")

    resolved = dict(predictors or {})
    for key, value in resolved.items():
        if isinstance(value, str) and isinstance(source, gu.Raster) and value in terrain.available_attributes:
            resolved[key] = terrain.get_terrain_attribute(source, value)

    # 4/ Run propagation workflow

    # For coreg
    if operation == "coreg":
        from xdem.uncertainty.coreg import propagate_uncertainty_coreg

        reference = options.pop("reference_elev")
        coreg_method = options.pop("coreg_method")
        return propagate_uncertainty_coreg(
            reference,
            source,
            coreg_method,
            error_structure=error_structure,
            predictors=resolved,
            method=method,
            **options,
            **kwargs,
        )

    # For terrain
    if operation == "terrain":
        from xdem.uncertainty.terrain import _propagate_terrain

        if not isinstance(source, gu.Raster):
            raise TypeError("Terrain uncertainty requires a raster source.")
        attribute = options.pop("attribute", "slope")
        return _propagate_terrain(
            source,
            attribute=attribute,
            error_structure=error_structure,
            predictors=resolved,
            terrain_options=options,
            method=method,
            **kwargs,
        )

    # For area (spatial) sums/averages
    if operation == "spatial":
        from xdem.uncertainty.spatial import _propagate_areas

        return _propagate_areas(
            source,
            error_structure=error_structure,
            predictors=resolved,
            method=method,
            **options,
            **kwargs,
        )

    # 5/ Now, we handle analytical propagation outside the named workflows

    # Check for an unrecognized operation name
    if operation is not None and not callable(operation):
        raise ValueError("operation must be 'terrain', 'spatial', 'coreg', a callable or None.")

    if method == "analytical":
        # Report the source elevations and predict their error magnitudes at each pixel or point
        std = error_structure.predict_magnitude(resolved, like=source)
        return PropagationSummary(
            estimate=source,
            mean=source.copy(),
            std=std,
            error_structure=error_structure,
            method="analytical",
            selection=_select_outputs(source, kwargs.pop("at", None)),
            metadata={"output_distribution": "normal"},
        )

    # 6/ Run numerical propagation for source errors or a custom callable

    # Pass the fixed operation arguments to each custom calculation
    def calculate(value: Any) -> Any:
        """Return perturbed elevations or apply the custom operation."""
        return value if operation is None else operation(value, **options)

    # Draw elevation errors and summarize the resulting elevations or custom outputs
    return _simulate(
        source,
        calculate,
        error_structure=error_structure,
        predictors=resolved,
        **kwargs,
    )

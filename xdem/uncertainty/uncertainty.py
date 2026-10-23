# Copyright (c) 2024 xDEM developers
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

from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal

import geoutils as gu
import numpy as np
from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_raster,
)
from geoutils.uncertainty import PropagationSummary

from xdem import terrain
from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import _as_eager_elevation
from xdem.uncertainty.analytical import (
    number_effective_samples,
    spatial_error_propagation,
)
from xdem.uncertainty.coreg import propagate_uncertainty_coreg
from xdem.uncertainty.error_structure import ErrorStructure
from xdem.uncertainty.numerical import patches_method
from xdem.uncertainty.propagation import _propagate_areas, _select_outputs, _simulate
from xdem.uncertainty.terrain import _propagate_terrain

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.raster.base import RasterLike
    from geoutils.vector.base import VectorLike

    from xdem.dem.base import DEMLike
    from xdem.epc.base import EPCLike

__all__ = [
    "estimate_error_structure",
    "propagate_uncertainty",
    "propagate_uncertainty_coreg",
    "number_effective_samples",
    "spatial_error_propagation",
    "patches_method",
]


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

        # Give GeoUtils compatible spatial inputs and return its fitted error structure
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
# 2/ UNCERTAINTY PROPAGATION
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

    # 1/ Validate the propagation options

    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if not isinstance(error_structure, gu.ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")

    # Fixed inputs for the selected operation override arguments in operation_options
    name = operation if isinstance(operation, str) else "custom"
    options = {**dict(operation_options or {}), **dict((inputs or {}).get(name, {}))}

    # 2/ Delegate bound spatial methods to GeoUtils

    # Bound GeoUtils operations already own their data, neighborhoods and uncertainty calculations
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

    # 3/ Prepare the elevations and magnitude predictors

    # Convert elevation accessors to the eager objects required by the xDEM workflows
    source = _as_eager_elevation(data)
    if not isinstance(source, (gu.Raster, gu.PointCloud)):
        raise TypeError("Uncertainty propagation requires a raster or point cloud source.")

    # Named terrain predictors are calculated once and remain fixed throughout the simulations
    resolved = dict(predictors or {})
    for key, value in resolved.items():
        if isinstance(value, str) and isinstance(source, gu.Raster) and value in terrain.available_attributes:
            resolved[key] = terrain.get_terrain_attribute(source, value)

    # 4/ Run the named propagation workflow

    # Separate the reference elevations and fitting method from the remaining coregistration options
    if operation == "coreg":
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

    # Require a raster before propagating errors through a terrain attribute
    if operation == "terrain":
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

    # Use the supplied regions and weights for area means or integrals
    if operation == "spatial":
        return _propagate_areas(
            source,
            error_structure=error_structure,
            predictors=resolved,
            method=method,
            **options,
            **kwargs,
        )

    # 5/ Summarize source errors analytically

    # Named workflows have returned, so only a callable or no operation is valid here
    if operation is not None and not callable(operation):
        raise ValueError("operation must be 'terrain', 'spatial', 'coreg', a callable or None.")

    # Without an operation, the result is the source elevation and its pointwise error magnitude
    if method == "analytical":
        if operation is not None:
            raise NotImplementedError("Analytical custom propagation requires a supported GeoUtils operator.")
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

    # 6/ Simulate source errors or a custom calculation

    # Adapt the optional callable to a simulation callback with fixed operation arguments
    def calculate(value: Any) -> Any:
        """Run calculation on perturbed elevations."""
        return value if operation is None else operation(value, **options)

    # Draw elevation errors and summarize the repeated outputs with GeoUtils
    return _simulate(
        source,
        calculate,
        error_structure=error_structure,
        predictors=resolved,
        **kwargs,
    )

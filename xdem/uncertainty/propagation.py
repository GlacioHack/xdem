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
import xarray as xr
from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils.uncertainty import PropagationSummary
from geoutils.uncertainty.estimation import _difference_proxy
from geoutils.uncertainty.propagation import (
    _numeric_values,
    _output_selection,
    simulate,
)

from xdem import terrain
from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    _apply_affine_elevation,
    _check_coreg_input_families,
)
from xdem.coreg.pipeline import CoregPipeline
from xdem.uncertainty.error_structure import ErrorStructure

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterLike
    from geoutils.vector.base import VectorLike

    from xdem.dem.base import DEMLike
    from xdem.epc.base import EPCLike

__all__ = ["estimate_error_structure", "propagate"]


############################################
# 1/ ESTIMATION FROM ELEVATION DIFFERENCES
############################################


def _prepare_estimation_predictors(
    source: DEMLike | EPCLike,
    other: DEMLike | EPCLike | NDArrayf | None,
    predictors: Mapping[str, Any] | tuple[str, ...] | None,
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> dict[str, Any]:
    """
    We prepare input predictors, either as array and/or terrain attributes.

    We use the source DEM when available, otherwise the reference DEM. With a raster reference, precoregistration
    calls this function before transforming source so predictors can be sampled at their original locations.
    """

    # Choose the DEM for slope/curv
    source_raster = _get_raster_interface(source)
    other_raster = _get_raster_interface(other)
    terrain_source = None
    if other is not None:
        terrain_source = source_raster if source_raster is not None else other_raster
    if predictors is None:
        predictors = ("slope", "max_curvature") if terrain_source is not None else ()
    predictor_specs = dict(predictors) if isinstance(predictors, Mapping) else {name: name for name in predictors}

    # Identify terrain attributes among input arrays, rasters and point columns
    terrain_names = []
    for value in predictor_specs.values():
        if other is None and source_raster is not None and isinstance(value, str):
            raise ValueError("Terrain predictors for an error proxy must be input as arrays or rasters.")
        if isinstance(value, str) and terrain_source is not None and value in terrain.available_attributes:
            if value not in terrain_names:
                terrain_names.append(value)

    # Derive terrain attributes
    if terrain_names:
        terrain_config = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
        attributes = terrain.get_terrain_attribute(terrain_source, terrain_names, mp_config=terrain_config)
        attributes = [attributes] if len(terrain_names) == 1 else attributes
        terrain_predictors = dict(zip(terrain_names, attributes))
        predictor_specs = {
            name: terrain_predictors.get(value, value) if isinstance(value, str) else value
            for name, value in predictor_specs.items()
        }
    return predictor_specs


def estimate_error_structure(
    source: DEMLike | EPCLike,
    other: DEMLike | EPCLike | NDArrayf | None = None,
    *,
    predictors: Mapping[str, Any] | tuple[str, ...] | None = None,
    mask: RasterLike | VectorLike | NDArrayb | None = None,
    precoreg: bool = True,
    coreg_method: Coreg | None = None,
    other_precision: Literal["negligible", "same"] = "negligible",
    at: Literal["self", "other"] | None = None,
    align: Literal["raise", "reproject"] = "reproject",
    spread_estimator: Callable[[Any], Any] | None = gu.stats.nmad,
    variogram_estimator: str | Callable[[Any], float] = "dowd",
    mp_config: MultiprocConfig | None = None,
    **kwargs: Any,
) -> ErrorStructure:
    """
    Estimate elevation error magnitudes and spatial correlation using terrain predictors.

    With two elevation inputs, slope and maximal curvature are used as predictors by default.
    Pass predictors={} for constant magnitudes.

    With ``other=None``, source is considered an error proxy, used without precoregistration or precision scaling.
    With two inputs, we align ``source`` to ``other`` with LZD by default before estimating errors from their
    residual differences. Pass ``precoreg=False`` for data that are already aligned.

    With a raster reference, we transform source pixels or points and interpolate the reference at their new
    coordinates by default, preserving source noise. With a point cloud reference, we apply the inverse affine
    transform to its points and interpolate the unchanged source DEM. This applies rotations exactly to points
    without regridding a DEM, although interpolation can still smooth noise in the sampled DEM.

    Pass predictor values explicitly for variable magnitudes.

    :param source: Raster or point cloud containing elevations, or an error proxy when other is None.
    :param other: Comparison elevation dataset, or an array matching the source pixels or point order.
        None uses source as an error proxy.
    :param predictors: Named predictor values or, for paired inputs, a tuple of terrain attribute names.
        None uses terrain defaults for paired elevations and constant magnitudes for a proxy.
    :param mask: Spatial or Boolean mask selecting stable terrain where differences represent error.
    :param precoreg: Align the source when the other input is present.
    :param coreg_method: Optional Coreg method or pipeline, copied before fitting. None uses LZD.
        Ignored when precoreg=False or other is None.
    :param other_precision: "negligible" assigns difference errors to the source. "same" assumes independent errors
        with the same magnitude and correlation in both datasets, dividing differences by the square root of
        two. Only used with two inputs.
    :param at: Comparison locations: "self", "other", or None to prefer point locations when present.
        Choosing "other" can interpolate source elevations and smooth their noise.
    :param align: "raise" reports mismatched grids or coordinate systems; "reproject" aligns them for comparison.
    :param spread_estimator: Error magnitude estimator, defaulting to NMAD for robustness to outliers.
        None uses the GeoUtils default, numpy.std().
    :param variogram_estimator: Empirical variogram estimator, defaulting to Dowd for robustness to outliers.
    :param mp_config: Worker and tile settings for multiprocessing estimation.
    :param kwargs: Additional options for GeoUtils ErrorStructure.estimate(), such as components, bins and
        random_state.

    :returns: Error structure, with magnitudes in elevation units.
    """

    _check_coreg_input_families(source, other, mask, predictors, mp_config=mp_config)

    # 1/ Prepare the mask
    source_raster = _get_raster_interface(source)
    other_raster = _get_raster_interface(other)

    if isinstance(mask, np.ndarray) and mask.ndim > 1:
        for raster in (source_raster, other_raster):
            if raster is not None and mask.squeeze().shape == raster.shape:
                mask = raster.from_array(
                    mask.squeeze(), raster.transform, raster.crs, area_or_point=raster.area_or_point
                )
                break

    # Leave multiprocessing files available until fitting finishes
    with ExitStack() as temporary_files:
        # 2/ Precoregister elevations
        predictor_specs = None
        if other is not None and precoreg:
            # Default to LZD for precoreg
            from xdem.coreg import LZD

            if coreg_method is None:
                coreg_method = LZD()
            if not isinstance(coreg_method, Coreg):
                raise TypeError("coreg_method must be a Coreg instance or None.")

            # Give a reference array the source coordinates
            source_interface = source_raster if source_raster is not None else _get_pointcloud_interface(source)
            if isinstance(other, np.ndarray):
                other = source_interface.copy(new_array=other)

            # Copy the method so estimating errors leaves the input fit unchanged
            fitted = coreg_method.copy().fit(
                other,
                source,
                inlier_mask=mask,
                random_state=kwargs.get("random_state"),
                mp_config=mp_config,
            )
            other_raster = _get_raster_interface(other)

            # Avoid resampling the source after precoregistration (which would smooth white or short-range noise)
            if other_raster is not None:
                predictor_specs = _prepare_estimation_predictors(source, other, predictors, mp_config, temporary_files)
                auxiliary = {}
                for index, value in enumerate(predictor_specs.values()):
                    if not isinstance(value, str):
                        raster = _get_raster_interface(value)
                        value = raster if raster is not None else value
                    auxiliary[f"_xdem_predictor_{index}"] = (source, value) if isinstance(value, str) else value

                # Attach predictors at the original source pixels or points
                sampling_config = (
                    temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
                )
                sampled = source_interface.cosample(
                    source, auxiliary=auxiliary, auxiliary_at="self", at="self", align=align, mp_config=sampling_config
                )

                # Convert all valid pixels to points and apply the forward fit without resampling
                # The reference DEM is interpolated at these transformed points below
                sampled_raster = _get_raster_interface(sampled)
                if sampled_raster is not None:
                    bands = {"self": 1, **{name: index + 3 for index, name in enumerate(auxiliary)}}
                    point_config = (
                        temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
                    )
                    sampled = sampled_raster.subsample(1, bands=bands, mp_config=point_config)
                    # Use the same number of points as raster cells per multiprocessing tile
                    if mp_config is not None:
                        mp_config = mp_config.copy()
                        mp_config.chunks = int(np.prod(mp_config.chunks))
                source = fitted.apply(sampled)
                predictor_specs = dict(zip(predictor_specs, auxiliary))

            # Move reference points into the source DEM frame with the inverse affine fit
            # We interpolate the unchanged DEM there to avoid raster resampling, even for rotations
            elif source_raster is not None and fitted.is_affine:
                # Use the fit CRS, then apply the mask before the inverse transform changes point locations
                reference_points = _get_pointcloud_interface(other)
                if reference_points.crs != source_raster.crs:
                    reference_points = _get_pointcloud_interface(reference_points.reproject(crs=source_raster.crs))
                reference_points = _get_pointcloud_interface(
                    reference_points.cosample(reference_points, mask=mask, at="self", align=align)
                )
                other = reference_points

                # Invert pipeline steps in reverse order, using each fitted rotation centroid
                steps = fitted.pipeline if isinstance(fitted, CoregPipeline) else [fitted]
                for step in reversed(steps):
                    other = _apply_affine_elevation(
                        _get_pointcloud_interface(other),
                        matrix=np.linalg.inv(step.to_matrix()),
                        centroid=step.meta["outputs"]["affine"].get("centroid"),
                        mp_config=mp_config,
                    )
                mask = None
            else:
                # Apply methods without an affine inverse to the source, or transform a source point cloud
                source = fitted.apply(source)

        # 3/ Prepare terrain predictors
        # With a raster reference, precoregistration already attached predictors before transforming the source
        if predictor_specs is None:
            predictor_specs = _prepare_estimation_predictors(source, other, predictors, mp_config, temporary_files)

        # 4/ Compare elevations
        _check_coreg_input_families(source, other, mask, predictor_specs, mp_config=mp_config)

        # Compare at the selected coordinates and derive residuals for error estimation
        if other is not None:
            if other_precision not in ("same", "negligible"):
                raise ValueError("other_precision must be 'same' or 'negligible'.")
            raster = _get_raster_interface(source)
            points = _get_pointcloud_interface(source) if raster is None else None
            sampler = raster if raster is not None else points
            if sampler is None:
                raise TypeError("source must be a raster or point cloud.")
            auxiliary = {
                f"_xdem_predictor_{index}": (source, value) if isinstance(value, str) else value
                for index, value in enumerate(predictor_specs.values())
            }
            sampling_config = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
            sampled = sampler.cosample(
                other,
                auxiliary=auxiliary,
                auxiliary_at="self",
                at=at,
                mask=mask,
                align=align,
                mp_config=sampling_config,
            )
            scale = np.sqrt(2.0) if other_precision == "same" else 1.0
            source = _difference_proxy(sampled, scale, mp_config, temporary_files)
            if _get_raster_interface(source) is not None:
                predictor_specs = {name: (source, index + 3) for index, name in enumerate(predictor_specs)}
            else:
                predictor_specs = dict(zip(predictor_specs, auxiliary))
            mask = None

        # 5/ Estimate the error structure
        return ErrorStructure.estimate(
            source,
            predictors=predictor_specs,
            mask=mask,
            spread_estimator=spread_estimator,
            variogram_estimator=variogram_estimator,
            mp_config=mp_config,
            **kwargs,
        )


###################################
# 2/ SIMULATION HELPERS
###################################


def _select_outputs(estimate: Any, at: Literal["all"] | Sequence[Hashable] | None) -> pd.DataFrame:
    """Select labelled outputs for saved draws without changing the simulation's spatial support."""

    raster = _get_raster_interface(estimate)
    points = _get_pointcloud_interface(estimate) if raster is None else None
    interface = raster if raster is not None else points
    values = _numeric_values(estimate)
    if raster is not None and raster.count == 1:
        values = values.reshape(raster.shape)
    if at is None and (interface is not None or values.size > 256):
        return pd.DataFrame(columns=["flat_index", "estimate"])
    if points is not None:
        dataset = points._dataset
        labels = dataset.get_index(next(iter(dataset.dims))) if points._is_xr else dataset.index
    elif isinstance(estimate, pd.Series):
        labels = estimate.index
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
    source: RasterLike | PointCloudLike,
    operation: Callable[[Any], Any],
    *,
    error_structure: gu.ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    random_field_options: Mapping[str, Any] | None = None,
    n_simulations: int = 100,
    random_state: int | np.random.Generator | None = None,
    at: Literal["all"] | Sequence[Hashable] | None = None,
    return_samples: bool = False,
    quantiles: Sequence[float] = (),
    max_sample_bytes: int = 268_435_456,
    on_error: Literal["raise", "warn"] = "raise",
    output: Mapping[str, Any] | None = None,
    estimate: Any = None,
) -> PropagationSummary:
    """Repeat an elevation operation with random fields and return GeoUtils propagation summary."""

    raster = _get_raster_interface(source)
    source_interface = raster if raster is not None else _get_pointcloud_interface(source)
    estimate = operation(source) if estimate is None else estimate
    selection = _select_outputs(estimate, at)
    if (return_samples or quantiles) and selection.empty:
        raise ValueError("Select outputs with at when requesting samples or quantiles.")

    field_options = dict(random_field_options or {})

    def draw_error(generator: np.random.Generator) -> Any:
        """Draw one error field on the unchanged elevation support."""

        return error_structure.generate_random_field(
            like=source_interface, predictors=predictors, random_state=generator, **field_options
        )

    def calculate(error: Any) -> Any:
        """Add one error field to a copy of the original elevations and repeat the operation."""

        perturbed = source_interface.copy(new_array=source_interface.data + _numeric_values(error))
        return operation(perturbed)

    return simulate(
        estimate=estimate,
        draw_error=draw_error,
        calculate=calculate,
        error_structure=error_structure,
        selection=selection,
        n_simulations=n_simulations,
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


def propagate(
    data: Any = None,
    operation: Literal["terrain", "spatial", "coreg"] | Callable[..., Any] | None = None,
    *,
    error_structure: gu.ErrorStructure | Mapping[str, gu.ErrorStructure],
    predictors: Mapping[str, Any] | None = None,
    error_predictors: Mapping[str, Mapping[str, Any]] | None = None,
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

        summary = xdem.uncertainty.propagate(
            dem, "spatial", error_structure=errors, method="analytical",
            inputs={"spatial": {"areas": {"whole_dem": None}}},
        )
        summary.std["whole_dem"]

    :param data: Original raster, point cloud or supported elevation accessor.
    :param operation: "terrain", "spatial", "coreg", a callable, or None to summarize the source errors.
        A callable receives the perturbed elevation object and the input operation arguments.
    :param error_structure: GeoUtils error model, or models keyed by "ref" and "tba" for coregistration.
    :param error_predictors: Coregistration magnitude predictors keyed by "ref" and "tba".
    :param predictors: Source magnitude predictors, including terrain attribute names for raster inputs.
    :param inputs: Fixed inputs under the operation name (or "custom" for a callable). Terrain accepts its usual
        arguments; spatial accepts an areas mapping and optional weights; coreg requires reference_elev.
    :param operation_options: Operation arguments, such as attribute for terrain, statistic for area means or
        integrals, and coreg_method for coregistration. Spatial integrals on point clouds require area weights.
    :param method: "numerical" for repeated calculations or "analytical" for existing propagation formulas.
    :param kwargs: GeoUtils propagation/simulation options, including n_simulations, random_state, at,
        return_samples, quantiles, max_sample_bytes and on_error. Spatial outputs require at to save draws.
        random_field_options passes options to generate_random_field(), which approximates correlated fields by
        default. Pass {"gpytorch_inducing_points": None} for exact in-memory draws.
    :returns: GeoUtils PropagationSummary with the original result, propagated mean and standard deviation,
        and any requested selected draws or quantiles.
    """

    # 1/ Check inputs
    if method not in {"analytical", "numerical"}:
        raise ValueError("method must be 'analytical' or 'numerical'.")
    if method == "analytical" and kwargs.get("return_samples", False):
        raise ValueError("Analytical propagation cannot return simulation samples.")
    if operation == "coreg":
        from xdem.uncertainty.coreg import propagate_uncertainty_coreg

        options = {**dict(operation_options or {}), **dict((inputs or {}).get("coreg", {}))}
        reference = options.pop("reference_elev")
        coreg_method = options.pop("coreg_method")
        if predictors is not None:
            raise TypeError("Coregistration uses error_predictors keyed by 'ref' and 'tba'.")
        return propagate_uncertainty_coreg(
            reference,
            data,
            coreg_method,
            error_structure=error_structure,
            error_predictors=error_predictors,
            method=method,
            **options,
            **kwargs,
        )
    if error_predictors is not None:
        raise TypeError("error_predictors is only used for coregistration. Use predictors for other operations.")
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

    if isinstance(data, xr.Dataset):
        raise TypeError("Select an elevation DataArray from the Dataset before coregistration or uncertainty analysis.")
    raster = _get_raster_interface(data)
    points = _get_pointcloud_interface(data) if raster is None else None
    if (
        is_dask_array(data)
        or is_dask_dataframe(data)
        or (raster is not None and raster._chunks is not None)
        or (points is not None and points._is_dask)
    ):
        raise NotImplementedError("Dask propagation for this operation is not supported yet.")
    source_interface = raster if raster is not None else points
    if source_interface is None:
        raise TypeError("Uncertainty propagation requires a raster or point cloud source.")

    # Use public containers for the original and perturbed calculations
    source = source_interface
    if raster is not None and raster._is_xr:
        source = raster._obj
    elif points is not None and (points._is_xr or points._is_pd):
        source = points._dataset

    resolved = dict(predictors or {})
    for key, value in resolved.items():
        if isinstance(value, str) and raster is not None and value in terrain.available_attributes:
            resolved[key] = terrain.get_terrain_attribute(raster, value)

    # 4/ Run propagation workflow

    # For terrain
    if operation == "terrain":
        from xdem.uncertainty.terrain import _propagate_terrain

        if raster is None:
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
        std = error_structure.predict_magnitude(resolved, like=source_interface)
        return PropagationSummary(
            estimate=source,
            mean=source_interface.copy(),
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

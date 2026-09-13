# Copyright (c) 2026 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
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

"""Estimate error component magnitudes and correlations from one spatial proxy.

The workflow first prepares aligned values and estimates their total magnitude. It then fits a nested variogram to
standardized errors, converts its variance fractions to named components and optionally refines those contributions
against conditional pair semivariances. Only grouped summaries, variogram bins and small optimizer diagnostics survive.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal, cast

import geoutils as gu
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from xdem._misc import import_optional
from xdem._typing import NDArrayf
from xdem.uncertainty.error_structure import (
    ErrorComponent,
    ErrorMagnitude,
    ErrorStructure,
)

############################
# 1/ INPUT AND MASK ALIGNMENT
############################


def _prepare_proxy_inputs(
    error_proxy: Any,
    predictors: Mapping[str, Any],
    mask: Any | None,
) -> tuple[NDArray[np.floating[Any]], dict[str, NDArray[np.floating[Any]]], NDArray[np.bool_]]:
    """Read an error proxy, aligned predictors and one common eligible mask."""

    # Identify public raster and point cloud interfaces without constraining xDEM subclasses
    is_raster = hasattr(error_proxy, "ij2xy")
    is_pointcloud = hasattr(error_proxy, "georeferenced_coords_equal") and hasattr(error_proxy, "data_column")
    if not is_raster and not is_pointcloud:
        raise TypeError("error_proxy must be a GeoUtils raster or point cloud.")

    if is_raster:
        # Select one raster band and replace masked cells by NaN for shared finite checks
        values = error_proxy.data
        if hasattr(values, "data") and not isinstance(values, np.ndarray) and not np.ma.isMaskedArray(values):
            values = values.data
        if hasattr(values, "compute"):
            values = values.compute()
        if np.ma.isMaskedArray(values):
            values = np.ma.asarray(values, dtype=float).filled(np.nan)

        # Require one elevation value per raster cell after resolving the input array
        values = np.asarray(values, dtype=float).squeeze()
        if values.ndim != 2:
            raise ValueError("Error proxy rasters must contain one two dimensional band.")

        # Align every predictor explicitly so grouped and pairwise calculations share cell positions
        predictor_arrays: dict[str, NDArray[np.floating[Any]]] = {}
        for name, predictor in predictors.items():
            predictor_raster = predictor if hasattr(predictor, "ij2xy") else getattr(predictor, "rst", None)
            if predictor_raster is not None:
                if not error_proxy.georeferenced_grid_equal(predictor_raster):
                    raise ValueError(f"Magnitude predictor {name!r} must share the error proxy grid.")
                predictor = predictor_raster.data
            elif isinstance(predictor, str):
                raise TypeError("Raster magnitude predictors cannot be column names.")
            elif hasattr(predictor, "data") and not isinstance(predictor, np.ndarray):
                predictor = predictor.data

            # Materialize the aligned raster predictor and retain its invalid cells for the common mask
            if hasattr(predictor, "compute"):
                predictor = predictor.compute()
            if np.ma.isMaskedArray(predictor):
                predictor = np.ma.asarray(predictor, dtype=float).filled(np.nan)
            predictor_array = np.asarray(predictor, dtype=float).squeeze()
            if predictor_array.shape != values.shape:
                raise ValueError(f"Magnitude predictor {name!r} must match the error proxy shape.")
            predictor_arrays[name] = predictor_array

        # Convert spatial and array masks to the same raster grid
        if mask is None:
            eligible = np.ones(values.shape, dtype=bool)
        elif hasattr(mask, "create_mask"):
            eligible = np.asarray(mask.create_mask(ref=error_proxy, as_array=True), dtype=bool).squeeze()
        else:
            mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
            if mask_raster is not None:
                if not error_proxy.georeferenced_grid_equal(mask_raster):
                    raise ValueError("A raster mask must share the error proxy grid.")
                mask_values = mask_raster.data
                if hasattr(mask_values, "compute"):
                    mask_values = mask_values.compute()
                if np.ma.isMaskedArray(mask_values):
                    mask_values = np.ma.asarray(mask_values).filled(False)
                eligible = np.isfinite(np.asarray(mask_values).squeeze()) & np.asarray(mask_values).squeeze().astype(
                    bool
                )
            else:
                eligible = np.asarray(mask).squeeze()

        # Reject masks whose shape or dtype could select a different raster population
        if eligible.shape != values.shape or not np.issubdtype(eligible.dtype, np.bool_):
            raise ValueError("mask must be Boolean and match the error proxy shape.")
    else:
        # Materialize one point table because pair indexes refer to its original row order
        dataframe = error_proxy.ds.compute() if hasattr(error_proxy.ds, "compute") else error_proxy.ds
        source_values = (
            dataframe[error_proxy.data_column].to_numpy()
            if error_proxy.data_column is not None
            else dataframe.geometry.z.to_numpy()
        )
        values = np.asarray(source_values, dtype=float).squeeze()
        predictor_arrays = {}

        # Retain native point coordinates for interpolation of raster predictors and masks
        points = (dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy())

        # Read native columns and aligned point arrays, or interpolate raster predictors at points
        for name, predictor in predictors.items():
            predictor_raster = predictor if hasattr(predictor, "ij2xy") else getattr(predictor, "rst", None)
            predictor_pointcloud = (
                predictor
                if hasattr(predictor, "georeferenced_coords_equal") and hasattr(predictor, "data_column")
                else getattr(predictor, "pc", None)
            )

            # Resolve each point predictor from a named column, a raster or an aligned point cloud
            if isinstance(predictor, str):
                if predictor not in dataframe:
                    raise ValueError(f"Point cloud has no magnitude predictor column {predictor!r}.")
                predictor_values = dataframe[predictor].to_numpy()
            elif predictor_raster is not None:
                predictor_values = predictor_raster.interp_points(points=points, as_array=True)
            elif predictor_pointcloud is not None:
                if not error_proxy.georeferenced_coords_equal(predictor_pointcloud):
                    raise ValueError(f"Point predictor {name!r} must share ordered coordinates with error_proxy.")
                predictor_values = predictor_pointcloud.data
            else:
                predictor_values = predictor.to_numpy() if hasattr(predictor, "to_numpy") else predictor

            # Convert sampled point predictors to one finite-checkable value per observation
            if hasattr(predictor_values, "compute"):
                predictor_values = predictor_values.compute()
            if np.ma.isMaskedArray(predictor_values):
                predictor_values = np.ma.asarray(predictor_values, dtype=float).filled(np.nan)
            predictor_array = np.asarray(predictor_values, dtype=float).squeeze()
            if predictor_array.ndim != 1 or len(predictor_array) != len(values):
                raise ValueError(f"Magnitude predictor {name!r} must contain one value per point.")
            predictor_arrays[name] = predictor_array

        # Evaluate vector and raster masks at points before accepting plain Boolean arrays
        if mask is None:
            eligible = np.ones(len(values), dtype=bool)
        elif hasattr(mask, "create_mask"):
            eligible = np.asarray(mask.create_mask(ref=error_proxy, as_array=True), dtype=bool).squeeze()
        else:
            mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
            if mask_raster is not None:
                mask_values = mask_raster.interp_points(points=points, method="nearest", as_array=True)
                eligible = np.isfinite(np.asarray(mask_values).squeeze()) & np.asarray(mask_values).squeeze().astype(
                    bool
                )
            else:
                eligible = np.asarray(mask).squeeze()

        # Reject point masks that cannot select the original row order exactly
        if eligible.ndim != 1 or len(eligible) != len(values) or not np.issubdtype(eligible.dtype, np.bool_):
            raise ValueError("mask must be Boolean with one value per error proxy point.")

    # Exclude missing errors and predictors once so every fitted step uses a common population
    eligible = eligible & np.isfinite(values)
    for predictor_array in predictor_arrays.values():
        eligible &= np.isfinite(predictor_array)
    if np.count_nonzero(eligible) < 2:
        raise ValueError("At least two finite error proxy observations are required.")
    return values, predictor_arrays, eligible


################################
# 2/ TOTAL MAGNITUDE ESTIMATION
################################


def _estimate_total_magnitude(
    values: NDArray[np.floating[Any]],
    predictors: Mapping[str, NDArray[np.floating[Any]]],
    eligible: NDArray[np.bool_],
    *,
    bins: Mapping[str, Any] | int | None,
    spread_estimator: Callable[[Any], Any],
    min_count: int,
    outlier_factor: float | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
) -> tuple[ErrorMagnitude, Mapping[str, Any]]:
    """Estimate a constant or grouped total magnitude and its scaling diagnostics."""

    # Validate statistical controls before invoking grouped reductions
    if min_count < 1:
        raise ValueError("min_count must be a positive integer.")
    if outlier_factor is not None and (not np.isfinite(outlier_factor) or outlier_factor <= 0):
        raise ValueError("outlier_factor must be finite and positive, or None.")

    # Remove a constant bias before relating the remaining error spread to predictors
    statistic = getattr(spread_estimator, "__name__", "spread")
    centered = values - float(np.nanmedian(values[eligible]))

    # Use the direct spread when no variable magnitude was requested
    if not predictors:
        magnitude = float(spread_estimator(centered[eligible]))
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("spread_estimator returned a non-positive or non-finite magnitude.")
        return ErrorMagnitude.constant(magnitude), {"center": float(np.nanmedian(values[eligible])), "scale": 1.0}

    # Expand one shared bin count or the default across all named predictors
    if bins is None:
        grouped_bins = dict.fromkeys(predictors, 10)
    elif isinstance(bins, (int, np.integer)):
        grouped_bins = {name: int(bins) for name in predictors}
    else:
        grouped_bins = dict(bins)
    if set(grouped_bins) != set(predictors):
        raise ValueError("bins must define every magnitude predictor and no unknown names.")

    # Avoid asking the shared sampler for more observations than this fitted population contains
    grouped_subsample = subsample
    if isinstance(subsample, (int, np.integer)) and subsample >= np.count_nonzero(eligible):
        grouped_subsample = 1

    # Keep unobserved predictor combinations so the fitted magnitude grid can fill their gaps
    table = gu.stats.grouped_stats(
        {"error": centered},
        dict(predictors),
        bins=grouped_bins,
        statistics=[spread_estimator],
        mask=eligible,
        subsample=grouped_subsample,
        random_state=random_state,
        observed=False,
    )

    # Build an initial interpolation model using reliable group statistics
    unscaled = ErrorMagnitude(
        kind="grouped",
        predictor_names=tuple(predictors),
        grouped_statistics=table,
        statistic=statistic,
        min_count=min_count,
    )

    # Correct interpolation bias through a second spread estimate on standardized errors
    initial = np.asarray(unscaled.predict(predictors), dtype=float)
    standardized = centered[eligible] / initial[eligible]
    finite = np.isfinite(standardized)

    # Exclude extreme standardized errors before estimating the final multiplicative correction
    if outlier_factor is not None and np.any(finite):
        standardized_center = float(np.nanmedian(standardized[finite]))
        preliminary_spread = float(spread_estimator(standardized[finite]))
        finite &= np.abs(standardized - standardized_center) <= outlier_factor * preliminary_spread

    # Rescale the predicted magnitudes so the retained standardized errors have unit spread
    scale = float(spread_estimator(standardized[finite]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The second magnitude scaling returned a non-positive or non-finite value.")
    return replace(unscaled, scale=scale), {
        "center": float(np.nanmedian(values[eligible])),
        "scale": scale,
        "retained_for_scaling": int(np.count_nonzero(finite)),
    }


#################################
# 3/ COMPONENT INITIALIZATION
#################################


def _normalize_component_configuration(
    components: Mapping[str, Mapping[str, Any]] | None,
    *,
    has_predictors: bool,
) -> list[dict[str, Any]]:
    """Validate ordered component specifications for the variogram estimator."""

    # Use the common DEM model when callers provide terrain or quality predictors
    if components is None:
        if has_predictors:
            components = {
                "short_range": {"magnitude": "heteroscedastic", "correlation": "gaussian"},
                "long_range": {"magnitude": "constant", "correlation": "spherical"},
            }
        else:
            components = {"short_range": {"magnitude": "constant", "correlation": "gaussian"}}
    if not components:
        raise ValueError("components must define at least one named error contribution.")

    # Normalize accessible aliases while preserving declared short to long order
    normalized: list[dict[str, Any]] = []
    variable_count = 0
    for name, configuration in components.items():
        if not isinstance(name, str) or not name or not isinstance(configuration, Mapping):
            raise TypeError("components must map non-empty names to configuration mappings.")
        unknown = set(configuration).difference({"magnitude", "correlation"})
        if unknown:
            raise ValueError(f"Unknown configuration for component {name!r}: {sorted(unknown)!r}.")

        # Resolve magnitude aliases and require predictors for a variable component
        magnitude = configuration.get("magnitude", "constant")
        if magnitude == "variable":
            magnitude = "heteroscedastic"
        if magnitude not in {"constant", "heteroscedastic"}:
            raise ValueError("Estimated component magnitude must be 'constant' or 'heteroscedastic'.")
        if magnitude == "heteroscedastic":
            variable_count += 1
            if not has_predictors:
                raise ValueError("A heteroscedastic component requires at least one magnitude predictor.")

        # Keep independent errors distinct from components that require a named correlation model
        correlation = configuration.get("correlation")
        if correlation is not None and not isinstance(correlation, str):
            raise TypeError("Estimated component correlation must be a variogram model name or None.")
        normalized.append({"name": name, "magnitude": magnitude, "correlation": correlation})

    # Limit fitting to the component decomposition supported by the current variogram estimator
    if variable_count > 1:
        raise NotImplementedError(
            "Variogram estimation currently supports at most one heteroscedastic component; "
            "manually constructed ErrorStructure objects can contain more."
        )
    return normalized


def _initialize_components(
    configuration: list[dict[str, Any]],
    total_magnitude: ErrorMagnitude,
    fitted_variogram: gu.Variogram,
) -> list[ErrorComponent]:
    """Convert standardized partial sills into component magnitude and correlation models."""

    # Separate fitted structured models from their optional shared independent nugget
    if fitted_variogram.model is None:
        raise ValueError("A fitted variogram is required to initialize error components.")
    fitted_model = fitted_variogram.model
    structured_models = list(fitted_model.components) if fitted_model.model_name == "sum" else [fitted_model]
    correlated = [item for item in configuration if item["correlation"] is not None]
    independent = [item for item in configuration if item["correlation"] is None]
    if len(structured_models) != len(correlated):
        raise RuntimeError("The fitted variogram does not match the requested correlated components.")
    if len(independent) > 1:
        raise ValueError("Only one independent component can be identified from a shared variogram nugget.")

    # Order fitted contributions by spatial range because component declarations follow short to long range
    fitted_ranges = np.array([float(model.effective_range) for model in structured_models])
    fitted_order = np.argsort(fitted_ranges)
    if not np.array_equal(fitted_order, np.arange(len(structured_models))):
        warnings.warn(
            "Fitted nested variogram ranges crossed; assigning their contributions from short to long range.",
            UserWarning,
        )
    ordered_ranges = fitted_ranges[fitted_order]
    ordered_sills = np.array([float(model.partial_sill or 0.0) for model in structured_models])[fitted_order]

    # Associate ordered parameters with declared model forms and retain the fitted nugget separately
    model_by_name: dict[str, gu.VariogramModel | None] = {}
    sill_by_name: dict[str, float] = {}
    model_position = 0
    for item in configuration:
        if item["correlation"] is None:
            model_by_name[item["name"]] = None
            sill_by_name[item["name"]] = float(fitted_model.nugget)
        else:
            model = replace(
                structured_models[model_position],
                effective_range=float(ordered_ranges[model_position]),
                partial_sill=float(ordered_sills[model_position]),
            )
            model_position += 1
            model_by_name[item["name"]] = replace(model, nugget=0.0)
            sill_by_name[item["name"]] = float(model.partial_sill or 0.0)

    # Normalize variance shares only after all structured and independent contributions are assigned
    total_sill = sum(sill_by_name.values())
    if total_sill <= 0:
        raise ValueError("The fitted variogram has no positive component variance.")

    # Derive initial magnitudes from normalized variance fractions
    reference_variance = total_magnitude.reference_value**2
    initial_variance = {
        item["name"]: reference_variance * sill_by_name[item["name"]] / total_sill for item in configuration
    }
    variable = next((item for item in configuration if item["magnitude"] == "heteroscedastic"), None)
    if variable is not None:
        fixed_variance = sum(
            initial_variance[item["name"]] for item in configuration if item["magnitude"] == "constant"
        )

        # Reserve at least five percent of representative variance for the variable component
        maximum_fixed = 0.95 * reference_variance
        if fixed_variance > maximum_fixed:
            factor = maximum_fixed / fixed_variance
            for item in configuration:
                if item["magnitude"] == "constant":
                    initial_variance[item["name"]] *= factor
            fixed_variance = maximum_fixed
    else:
        fixed_variance = 0.0

    # Build immutable public components while keeping their fitted fractions as diagnostics
    output: list[ErrorComponent] = []
    for item in configuration:
        # Allocate the remaining local variance to the variable component and fixed scalars to the others
        magnitude = (
            replace(total_magnitude, variance_offset=fixed_variance)
            if item["magnitude"] == "heteroscedastic"
            else ErrorMagnitude.constant(np.sqrt(initial_variance[item["name"]]))
        )
        output.append(
            ErrorComponent(
                name=item["name"],
                magnitude=magnitude,
                correlation=model_by_name[item["name"]],
                metadata={
                    "magnitude_kind": item["magnitude"],
                    "initial_variance_fraction": sill_by_name[item["name"]] / total_sill,
                },
            )
        )
    return output


#################################
# 4/ CONDITIONAL PAIR REFINEMENT
#################################


def _refine_components_from_pairs(
    error_proxy: Any,
    components: list[ErrorComponent],
    total_magnitude: ErrorMagnitude,
    predictor_arrays: Mapping[str, NDArray[np.floating[Any]]],
    *,
    mask: Any | None,
    estimator: str | Callable[[Any], float],
    n_pairs: int,
    pair_sampling: str,
    n_lags: int,
    min_lag: float | None,
    max_lag: float | None,
    pair_sampling_kwargs: Mapping[str, Any],
    random_state: int | np.random.Generator | None,
) -> tuple[list[ErrorComponent], Mapping[str, Any]]:
    """
    Refine component magnitudes and ranges using error pairs grouped by distance and local magnitude.

    A variogram of standardized errors initializes the components, but different magnitudes at the two
    locations change the pair semivariance. Compare robust observed semivariances with the full component
    equation in groups of similar distance and magnitude. Constrain variance shares and ranges during fitting,
    then retain grouped diagnostics while discarding the sampled pair arrays.
    """

    # Draw one temporary pair sample and recover total magnitudes at both endpoints
    pairs = error_proxy.sample_pairs(
        n_pairs=n_pairs,
        sampling=pair_sampling,
        min_distance=min_lag,
        max_distance=max_lag,
        mask=mask,
        random_state=random_state,
        **pair_sampling_kwargs,
    )

    # Extract pair distances, original values and indexes needed to evaluate both local magnitudes
    distances = np.asarray(pairs["distance"], dtype=float)
    endpoint_values = np.asarray(pairs["value"], dtype=float)
    total_array = np.asarray(total_magnitude.predict(predictor_arrays), dtype=float)
    indexes = np.asarray(pairs["index"], dtype=np.int64)

    # Expand constant magnitudes directly because they do not occupy the source support
    if total_array.ndim == 0:
        endpoint_total = np.full(endpoint_values.shape, float(total_array), dtype=float)
    elif pairs.attrs.get("source") == "raster":
        rows = np.asarray(pairs["row"], dtype=np.int64)
        columns = np.asarray(pairs["column"], dtype=np.int64)
        endpoint_total = total_array[rows, columns]
    else:
        endpoint_total = total_array[indexes]

    # Remove any interpolation failure before constructing robust conditional bins
    differences = np.abs(endpoint_values[:, 0] - endpoint_values[:, 1])
    valid = (
        np.isfinite(distances)
        & (distances > 0)
        & np.isfinite(differences)
        & np.all(np.isfinite(endpoint_total), axis=1)
        & np.all(endpoint_total > 0, axis=1)
    )

    # Apply the same finite-pair selection to distances, differences and endpoint magnitudes
    distances = distances[valid]
    differences = differences[valid]
    endpoint_total = endpoint_total[valid]
    if len(distances) < 20:
        return components, {"success": False, "message": "Too few finite pairs for conditional refinement."}

    # Resolve the same robust estimator used for the ordinary empirical variogram
    if callable(estimator):
        estimator_function = estimator
    else:
        skgstat = import_optional("skgstat", package_name="scikit-gstat")
        if not hasattr(skgstat.estimators, estimator):
            raise ValueError(f"Unknown SciKit-GStat variogram estimator {estimator!r}.")
        estimator_function = getattr(skgstat.estimators, estimator)

    # Cross distance classes with total magnitude quantiles to expose component dependence
    distance_edges = np.geomspace(float(np.min(distances)), float(np.max(distances)), n_lags + 1)
    mean_magnitude = np.mean(endpoint_total, axis=1)

    # Use magnitude quantiles to distinguish variable error spread within each distance class
    magnitude_edges = np.unique(np.quantile(mean_magnitude, np.linspace(0, 1, 5)))
    if len(magnitude_edges) < 2:
        magnitude_edges = np.array([mean_magnitude.min(), np.nextafter(mean_magnitude.max(), np.inf)])
    else:
        magnitude_edges[-1] = np.nextafter(magnitude_edges[-1], np.inf)

    # Assign pairs to the crossed distance and magnitude groups, including boundary values
    distance_group = np.clip(np.digitize(distances, distance_edges, right=True) - 1, 0, n_lags - 1)
    magnitude_group = np.clip(
        np.digitize(mean_magnitude, magnitude_edges, right=False) - 1,
        0,
        len(magnitude_edges) - 2,
    )
    group_ids = distance_group * (len(magnitude_edges) - 1) + magnitude_group

    # Retain groups large enough for stable robust empirical estimates
    unique_groups, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    retained_groups = unique_groups[counts >= 20]
    selected = np.isin(group_ids, retained_groups)
    distances = distances[selected]
    differences = differences[selected]
    endpoint_total = endpoint_total[selected]
    group_ids = group_ids[selected]
    unique_groups, inverse, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
    if len(unique_groups) < max(3, len(components)):
        return components, {"success": False, "message": "Too few populated conditional groups for refinement."}

    # Estimate robust semivariance independently in each populated conditional group
    observed = np.array(
        [float(estimator_function(differences[inverse == index])) for index in range(len(unique_groups))]
    )

    # Parameterize fixed variances so one variable component always remains physically valid
    variable_index = next(
        (
            index
            for index, component in enumerate(components)
            if cast(ErrorMagnitude, component.magnitude).kind == "grouped"
        ),
        None,
    )

    # Separate constant variance parameters from the component carrying spatially varying magnitude
    fixed_indexes = [index for index in range(len(components)) if index != variable_index]
    initial_variances = np.array(
        [cast(ErrorMagnitude, component.magnitude).reference_value ** 2 for component in components]
    )
    magnitude_parameters: list[float] = []
    if variable_index is not None and fixed_indexes:
        # Bound fixed variance using a low local-variance quantile so it cannot exhaust the variable component
        fixed_limit = float(np.quantile(endpoint_total.ravel() ** 2, 0.02))
        shares = initial_variances[fixed_indexes] / fixed_limit
        if np.sum(shares) >= 0.95:
            shares *= 0.95 / np.sum(shares)

        # Express variance shares as log ratios with a positive reserve for variable error
        reserve = max(1 - float(np.sum(shares)), 1e-6)
        magnitude_parameters = np.log(np.maximum(shares, 1e-12) / reserve).tolist()
    elif variable_index is None and len(components) > 1:
        fixed_limit = total_magnitude.reference_value**2

        # Use relative shares that sum to the known total variance when every component is constant
        shares = initial_variances / np.sum(initial_variances)
        magnitude_parameters = np.log(np.maximum(shares[:-1], 1e-12) / max(shares[-1], 1e-12)).tolist()
    else:
        fixed_limit = total_magnitude.reference_value**2

    # Constrain declared correlation ranges to ordered nonoverlapping intervals
    correlated_indexes = [index for index, component in enumerate(components) if component.correlation is not None]
    initial_ranges_list: list[float] = []
    for index in correlated_indexes:
        correlation = cast(gu.VariogramModel, components[index].correlation)
        if correlation.effective_range is None:
            raise AssertionError("A fitted component correlation must define an effective range.")
        initial_ranges_list.append(correlation.effective_range)
    initial_ranges = np.asarray(initial_ranges_list, dtype=float)
    minimum_distance = float(np.min(distances))
    maximum_distance = float(np.max(distances))
    if maximum_distance <= minimum_distance:
        return components, {"success": False, "message": "Sampled pairs contain no usable distance range."}

    # Keep initial ranges inside the observed domain and restore separation if a fit collapsed them
    initial_ranges = np.clip(initial_ranges, minimum_distance, maximum_distance)
    if len(initial_ranges) > 1 and np.any(np.diff(initial_ranges) <= np.finfo(float).eps * initial_ranges[:-1]):
        initial_ranges = np.geomspace(minimum_distance, maximum_distance, len(initial_ranges) + 2)[1:-1]

    # Separate neighboring fitted ranges at geometric midpoints and optimize in log-distance units
    boundaries = [minimum_distance]
    boundaries.extend(np.sqrt(initial_ranges[:-1] * initial_ranges[1:]).tolist())
    boundaries.append(maximum_distance)
    range_lower = np.log(np.maximum(np.asarray(boundaries[:-1]), np.finfo(float).eps))
    range_upper = np.log(np.maximum(np.asarray(boundaries[1:]), np.asarray(boundaries[:-1]) * 1.0001))
    initial_log_ranges = np.log(np.clip(initial_ranges, np.exp(range_lower), np.exp(range_upper)))

    # Decode constrained magnitudes and ranges from one compact optimizer vector
    n_magnitude_parameters = len(magnitude_parameters)

    def decode(parameters: NDArray[np.floating[Any]]) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Convert bounded optimizer parameters to physically valid component variances and positive ranges."""

        # Start with the initial variance allocation and replace only fitted shares
        component_variances = initial_variances.copy()
        if variable_index is not None and fixed_indexes:
            # Map log ratios to non-negative fixed variances below the available local variance limit
            exponentials = np.exp(np.clip(parameters[:n_magnitude_parameters], -30, 30))
            fixed_variances = fixed_limit * exponentials / (1 + np.sum(exponentials))
            component_variances[fixed_indexes] = fixed_variances
            component_variances[variable_index] = max(
                total_magnitude.reference_value**2 - float(np.sum(fixed_variances)),
                0.0,
            )

        # Normalize constant component shares to preserve their total variance
        elif variable_index is None and len(components) > 1:
            logits = np.r_[parameters[:n_magnitude_parameters], 0.0]
            exponentials = np.exp(logits - np.max(logits))
            component_variances = fixed_limit * exponentials / np.sum(exponentials)

        # Recover positive spatial ranges from the remaining log-distance parameters
        ranges = np.exp(parameters[n_magnitude_parameters:])
        return component_variances, ranges

    # Evaluate the exact endpoint magnitude equation before reducing predictions by group
    skgstat = import_optional("skgstat", package_name="scikit-gstat")

    def objective(parameters: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Compare grouped observations with the predicted pair semivariance of the complete component model."""

        # Recover the trial component model before evaluating its covariance at sampled pairs
        component_variances, ranges = decode(parameters)
        fixed_variance_sum = float(np.sum(component_variances[fixed_indexes]))
        predicted_pairs = np.zeros(len(distances), dtype=float)
        range_position = 0

        # Evaluate separate local magnitudes at both pair locations for the variable component
        for index, component in enumerate(components):
            if index == variable_index:
                first_magnitude = np.sqrt(np.maximum(endpoint_total[:, 0] ** 2 - fixed_variance_sum, 0.0))
                second_magnitude = np.sqrt(np.maximum(endpoint_total[:, 1] ** 2 - fixed_variance_sum, 0.0))
            else:
                first_magnitude = second_magnitude = np.sqrt(component_variances[index])

            # Evaluate unit sill correlation or the independent error limit
            if component.correlation is None:
                correlation = np.zeros(len(distances), dtype=float)
            else:
                model = component.correlation
                arguments = [ranges[range_position], 1.0]
                range_position += 1
                if model.model_name == "matern":
                    arguments.append(float(model.smoothness))
                elif model.model_name == "stable":
                    arguments.append(float(model.shape))
                correlation = 1 - np.asarray(
                    getattr(skgstat.models, model.model_name)(distances, *arguments),
                    dtype=float,
                )

            # Use half the difference variance: (sigma1**2 + sigma2**2 - 2*sigma1*sigma2*rho) / 2
            predicted_pairs += 0.5 * (
                first_magnitude**2 + second_magnitude**2 - 2 * first_magnitude * second_magnitude * correlation
            )

        # Compare robust observations with mean model predictions using count and scale weights
        predicted = np.bincount(inverse, weights=predicted_pairs, minlength=len(unique_groups)) / counts
        reference = max(float(np.nanmedian(observed)), float(np.finfo(float).eps))
        scale = np.maximum(np.abs(observed), 0.1 * reference)
        weight = np.sqrt(counts / np.median(counts))
        return weight * (predicted - observed) / scale

    # Fit all variance shares and ranges together with bounded robust least squares
    initial_parameters = np.r_[magnitude_parameters, initial_log_ranges]
    lower = np.r_[np.full(n_magnitude_parameters, -30.0), range_lower]
    upper = np.r_[np.full(n_magnitude_parameters, 30.0), range_upper]
    result = least_squares(
        objective,
        initial_parameters,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
    )
    fitted_variances, fitted_ranges = decode(result.x)

    # Retain one compact record per conditional bin for decomposition diagnostics
    residual = objective(result.x)
    reference = max(float(np.nanmedian(observed)), float(np.finfo(float).eps))
    residual_scale = np.maximum(np.abs(observed), 0.1 * reference)
    residual_weight = np.sqrt(counts / np.median(counts))
    predicted = observed + residual * residual_scale / residual_weight

    # Store observed and fitted semivariances with their mean distance, magnitude and pair count
    conditional_statistics = pd.DataFrame(
        {
            "lag": np.bincount(inverse, weights=distances, minlength=len(unique_groups)) / counts,
            "mean_magnitude": (
                np.bincount(inverse, weights=np.mean(endpoint_total, axis=1), minlength=len(unique_groups)) / counts
            ),
            "semivariance": observed,
            "fitted_semivariance": predicted,
            "count": counts,
        },
        index=pd.Index(unique_groups, name="conditional_group"),
    )

    # Estimate local parameter uncertainty from the final robust least squares Jacobian
    parameter_error = np.full(len(result.x), np.nan)
    if result.jac.shape[0] > result.jac.shape[1]:
        information = result.jac.T @ result.jac
        if np.linalg.matrix_rank(information) == information.shape[0]:
            residual_variance = 2 * result.cost / (result.jac.shape[0] - result.jac.shape[1])
            parameter_error = np.sqrt(np.diag(np.linalg.inv(information) * residual_variance))

    # Transfer optimized contributions back to immutable public component objects
    fixed_variance_sum = float(np.sum(fitted_variances[fixed_indexes]))
    refined: list[ErrorComponent] = []
    range_position = 0
    for index, component in enumerate(components):
        magnitude = (
            replace(total_magnitude, variance_offset=fixed_variance_sum)
            if index == variable_index
            else ErrorMagnitude.constant(np.sqrt(fitted_variances[index]))
        )

        # Update only correlated components with their fitted range, preserving independent components
        correlation = component.correlation
        if correlation is not None:
            correlation = replace(correlation, effective_range=float(fitted_ranges[range_position]))
            range_position += 1
        refined.append(
            ErrorComponent(
                component.name,
                magnitude,
                correlation,
                metadata={**component.metadata, "refined": True},
            )
        )

    # Retain optimizer convergence and grouped evidence without keeping raw observation pairs
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "cost": float(result.cost),
        "optimality": float(result.optimality),
        "conditional_group_count": int(len(unique_groups)),
        "pair_count": int(len(distances)),
        "parameters": result.x.tolist(),
        "parameter_error": parameter_error.tolist(),
        "conditional_statistics": conditional_statistics,
    }
    return refined, diagnostics


################################
# 5/ COMPLETE ESTIMATION WORKFLOW
################################


def _representative_variogram(
    empirical: gu.Variogram,
    components: list[ErrorComponent],
) -> gu.Variogram:
    """Attach the representative normalized component model to empirical bins."""

    # Normalize reference component variances so the combined model retains unit sill
    reference_variances = np.array(
        [cast(ErrorMagnitude, component.magnitude).reference_value ** 2 for component in components]
    )
    fractions = reference_variances / np.sum(reference_variances)
    structured: list[gu.VariogramModel] = []
    nugget = 0.0

    # Represent independent variance as a nugget and retain normalized structured contributions
    for component, fraction in zip(components, fractions):
        if component.correlation is None:
            nugget += float(fraction)
        else:
            structured.append(replace(component.correlation, partial_sill=float(fraction)))
    if not structured:
        raise ValueError("At least one spatially correlated component is required for variogram fitting.")

    # Attach the combined model and its predictions at the original empirical lag centers
    model = (
        replace(structured[0], nugget=nugget)
        if len(structured) == 1
        else gu.VariogramModel.sum(structured, nugget=nugget)
    )
    return replace(empirical, model=model, fitted_semivariance=model.variogram(empirical.lags))


def _estimate_error_structure(
    error_proxy: Any,
    *,
    predictors: Mapping[str, Any] | None,
    components: Mapping[str, Mapping[str, Any]] | None,
    mask: Any | None,
    bins: Mapping[str, Any] | int | None,
    spread_estimator: Callable[[Any], Any],
    min_count: int,
    outlier_factor: float | None,
    subsample_magnitude: int | float,
    variogram_estimator: str | Callable[[Any], float],
    n_pairs: int,
    pair_sampling: Literal["loglag", "random_xy"],
    n_lags: int,
    min_lag: float | None,
    max_lag: float | None,
    n_runs: int,
    fit_method: Literal["variogram"],
    refine: bool,
    fit_kwargs: Mapping[str, Any] | None,
    pair_sampling_kwargs: Mapping[str, Any] | None,
    random_state: int | np.random.Generator | None,
) -> ErrorStructure:
    """Implement :meth:`ErrorStructure.estimate` without retaining spatial pairs."""

    # Normalize configuration and aligned source arrays before statistical estimation
    if fit_method != "variogram":
        raise NotImplementedError("Only fit_method='variogram' is currently implemented.")
    predictor_mapping = {} if predictors is None else dict(predictors)
    configuration = _normalize_component_configuration(components, has_predictors=bool(predictor_mapping))
    values, predictor_arrays, eligible = _prepare_proxy_inputs(error_proxy, predictor_mapping, mask)

    # Draw separate seeds so magnitude estimation, variography and refinement are reproducible stages
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    magnitude_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    variogram_seed = int(rng.integers(0, np.iinfo(np.int32).max))
    refinement_seed = int(rng.integers(0, np.iinfo(np.int32).max))

    # Estimate the total local magnitude before separating component variances
    total_magnitude, magnitude_diagnostics = _estimate_total_magnitude(
        values,
        predictor_arrays,
        eligible,
        bins=bins,
        spread_estimator=spread_estimator,
        min_count=min_count,
        outlier_factor=outlier_factor,
        subsample=subsample_magnitude,
        random_state=magnitude_seed,
    )

    # Finish after magnitude estimation when only independent errors were requested
    correlated_models = [item["correlation"] for item in configuration if item["correlation"] is not None]
    if not correlated_models:
        if len(configuration) != 1:
            raise ValueError("Only one independent component can be estimated without a spatial correlation model.")
        independent_component = ErrorComponent(configuration[0]["name"], total_magnitude, None)
        return ErrorStructure(
            [independent_component],
            fit_diagnostics={
                "magnitude": dict(magnitude_diagnostics),
                "refinement": {"success": None, "message": "No correlated component was requested."},
                "identifiability": [],
            },
            metadata={
                "fit_method": fit_method,
                "spread_estimator": getattr(spread_estimator, "__name__", "callable"),
                "variogram_estimator": None,
                "component_configuration": configuration,
                "total_magnitude": total_magnitude,
                "n_runs": 0,
            },
        )

    # Center before local standardization so varying magnitudes do not turn a constant bias into spatial structure
    total_array = np.asarray(total_magnitude.predict(predictor_arrays), dtype=float)
    standardized_values = np.full(values.shape, np.nan, dtype=float)
    np.divide(
        values - float(magnitude_diagnostics["center"]),
        total_array,
        out=standardized_values,
        where=np.isfinite(total_array) & (total_array > 0),
    )
    standardized_proxy = error_proxy.copy(new_array=standardized_values)

    # Fit the requested nested models through GeoUtils lightweight variography
    fit_options = dict(fit_kwargs or {})
    independent_count = sum(item["correlation"] is None for item in configuration)
    if independent_count:
        fit_options.setdefault("use_nugget", True)

    # Estimate the standardized variogram on the same selected error population
    pair_options = dict(pair_sampling_kwargs or {})
    empirical = standardized_proxy.variogram(
        n_pairs=n_pairs,
        sampling=pair_sampling,
        estimator=variogram_estimator,
        n_lags=n_lags,
        min_lag=min_lag,
        max_lag=max_lag,
        n_runs=n_runs,
        model=correlated_models,
        fit_kwargs=fit_options,
        random_state=variogram_seed,
        mask=mask,
        **pair_options,
    )
    initialized = _initialize_components(configuration, total_magnitude, empirical)

    # Refine range and magnitude separation against conditional raw error pairs
    if refine:
        fitted_components, refinement_diagnostics = _refine_components_from_pairs(
            error_proxy,
            initialized,
            total_magnitude,
            predictor_arrays,
            mask=mask,
            estimator=variogram_estimator,
            n_pairs=n_pairs,
            pair_sampling=pair_sampling,
            n_lags=n_lags,
            min_lag=min_lag,
            max_lag=max_lag,
            pair_sampling_kwargs=pair_options,
            random_state=refinement_seed,
        )
    else:
        fitted_components = initialized
        refinement_diagnostics = {"success": None, "message": "Conditional refinement was disabled."}

    # Update displayed variogram predictions to match the final component decomposition
    empirical = _representative_variogram(empirical, fitted_components)

    # Report weak range separation and domain limited long range estimates explicitly
    messages: list[str] = []
    ranges = [
        float(component.correlation.effective_range)
        for component in fitted_components
        if component.correlation is not None
    ]
    if len(ranges) > 1 and any(second / first < 1.5 for first, second in zip(ranges[:-1], ranges[1:])):
        messages.append("Some fitted correlation ranges overlap and their contributions may be weakly identified.")

    # Flag a long correlation range near the sampled extent, where it may be difficult to distinguish from trend
    sampled_maximum = float(empirical.attrs.get("max_distance", np.nanmax(empirical.lags)))
    if ranges and np.isfinite(sampled_maximum) and ranges[-1] >= 0.9 * sampled_maximum:
        messages.append("The longest correlation range approaches the sampled extent and may represent a trend.")
    for message in messages:
        warnings.warn(message, UserWarning)

    # Return only compact grouped statistics, variogram bins and optimizer summaries
    return ErrorStructure(
        fitted_components,
        empirical_variogram=empirical,
        fit_diagnostics={
            "magnitude": dict(magnitude_diagnostics),
            "refinement": dict(refinement_diagnostics),
            "identifiability": messages,
        },
        metadata={
            "fit_method": fit_method,
            "spread_estimator": getattr(spread_estimator, "__name__", "callable"),
            "variogram_estimator": (
                getattr(variogram_estimator, "__name__", "callable")
                if callable(variogram_estimator)
                else variogram_estimator
            ),
            "component_configuration": configuration,
            "total_magnitude": total_magnitude,
            "n_runs": n_runs,
        },
    )


def _refit_error_structure(
    structure: ErrorStructure,
    *,
    correlation_models: str | Sequence[str] | None,
    fit_kwargs: Mapping[str, Any] | None,
) -> ErrorStructure:
    """Refit retained variogram bins and update every dependent component contribution."""

    # Require compact fit metadata needed to rebuild component magnitudes consistently
    if structure.empirical_variogram is None:
        raise ValueError("Refitting requires a retained empirical variogram.")
    configuration_value = structure.metadata.get("component_configuration")
    total_magnitude = structure.metadata.get("total_magnitude")
    if not isinstance(configuration_value, list) or not isinstance(total_magnitude, ErrorMagnitude):
        raise ValueError("This error structure does not retain the estimation metadata required for refitting.")
    configuration = [dict(item) for item in configuration_value]

    # Default to current models while preserving the declared short to long component order
    correlated = [component for component in structure.values() if component.correlation is not None]
    models: str | list[str]
    if correlation_models is None:
        models = [component.correlation.model_name for component in correlated if component.correlation is not None]
    elif isinstance(correlation_models, str):
        models = correlation_models
    else:
        models = list(correlation_models)

    # Require the replacement models to describe the same number of correlated components
    expected = sum(item["correlation"] is not None for item in configuration)
    model_count = len(models.split("+")) if isinstance(models, str) else len(models)
    if model_count != expected:
        raise ValueError(f"correlation_models must contain {expected} model(s).")

    # Preserve nugget fitting when the structure contains an independent component
    options = dict(fit_kwargs or {})
    if any(item["correlation"] is None for item in configuration):
        options.setdefault("use_nugget", True)
    empirical = structure.empirical_variogram.fit(models, **options)

    # Update component definitions before reallocating their magnitudes from the refitted variogram
    for item, model_name in zip(
        [item for item in configuration if item["correlation"] is not None],
        models.split("+") if isinstance(models, str) else models,
    ):
        item["correlation"] = model_name
    components = _initialize_components(configuration, total_magnitude, empirical)
    empirical = _representative_variogram(empirical, components)

    # Mark the absence of conditional pair refinement rather than retaining stale diagnostics
    diagnostics = {
        **structure.fit_diagnostics,
        "refinement": {
            "success": None,
            "message": "Refit from empirical bins; call estimate again for conditional pair refinement.",
        },
    }
    return ErrorStructure(
        components,
        empirical_variogram=empirical,
        fit_diagnostics=diagnostics,
        metadata={**structure.metadata, "component_configuration": configuration},
    )


############################
# 6/ ERROR STANDARDIZATION
############################


def two_step_standardization(
    dvalues: NDArrayf,
    list_var: list[NDArrayf],
    unscaled_error_fun: Callable[[tuple[ArrayLike, ...]], NDArrayf],
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    fac_spread_outliers: float | None = 7,
) -> tuple[NDArrayf, Callable[[tuple[ArrayLike, ...]], NDArrayf]]:
    """
    Standardize the proxy differenced values using the modelled heteroscedasticity, re-scaled to the spread statistic,
    and generate the final standardization function.

    :param dvalues: Proxy values as array of size (N,) (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param unscaled_error_fun: Function of the spread with explanatory variables not yet re-scaled
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Standardized values array of size (N,), Function to destandardize
    """

    # Standardize a first time with the function
    zscores = dvalues / unscaled_error_fun(tuple(list_var))

    # Set large outliers that might have been created by the standardization to NaN, central tendency should already be
    # around zero so only need to take the absolute value
    if fac_spread_outliers is not None:
        if np.ma.isMaskedArray(zscores):
            zscores[np.abs(zscores) > fac_spread_outliers * spread_statistic(zscores)] = np.ma.masked
        else:
            zscores[np.abs(zscores) > fac_spread_outliers * spread_statistic(zscores)] = np.nan

    # Re-compute the spread statistic to re-standardize, as dividing by the function will not necessarily bring the
    # z-score exactly equal to one due to approximations of N-D binning, interpolating and due to the outlier filtering
    zscore_nmad = spread_statistic(zscores)

    # Re-standardize
    zscores /= zscore_nmad

    # Define the exact function for de-standardization to pass as output
    def error_fun(*args: tuple[ArrayLike, ...]) -> NDArrayf:
        """Evaluate the corrected magnitude after calibrating the standardized error spread."""

        return zscore_nmad * unscaled_error_fun(*args)

    return zscores, error_fun

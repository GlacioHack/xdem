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

"""Numerical propagation of elevation errors and empirical uncertainty sampling from spatial patches."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
from geoutils import PointCloud, Raster
from geoutils._dispatch import _is_pointcloud, _is_raster
from geoutils.filters import _create_circular_mask, mean_filter
from geoutils.raster import RasterType
from geoutils.vector.vector import VectorType
from numpy.typing import NDArray

from xdem._misc import import_optional
from xdem._typing import NDArrayf
from xdem.uncertainty.error_structure import ErrorStructure

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike

##########################
# 1/ RANDOM ERROR FIELDS
##########################


def generate_random_field(
    like: RasterBase | PointCloudBase,
    error_structure: ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None = None,
    n_fields: int = 1,
    random_state: int | np.random.Generator | None = None,
) -> RasterLike | PointCloudLike | list[RasterLike | PointCloudLike]:
    """Generate independent error realizations on raster or point cloud support.

    :param like: Raster or point cloud defining output coordinates and type.
    :param error_structure: Components defining magnitudes and spatial correlations.
    :param predictors: Named values required by grouped component magnitudes on ``like``.
    :param n_fields: Number of independent realizations.
    :param random_state: Random generator or seed used for reproducible fields.
    :returns: One spatial object or a list when ``n_fields`` is greater than one.
    """

    # Validate supported spatial interfaces before importing an optional simulation backend
    is_raster = _is_raster(like)
    is_pointcloud = _is_pointcloud(like)
    if not is_raster and not is_pointcloud:
        raise TypeError("like must be a GeoUtils raster or point cloud.")

    # Require a whole number of fields and share one generator across all component draws
    if not isinstance(n_fields, (int, np.integer)) or isinstance(n_fields, bool) or n_fields < 1:
        raise ValueError("n_fields must be a positive integer.")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)

    # Prepare coordinates once because every component and realization shares the same support
    if is_raster:
        output_shape = tuple(like.shape)
        x = np.arange(output_shape[1], dtype=float) * abs(float(like.res[0]))
        y = np.arange(output_shape[0], dtype=float) * abs(float(like.res[1]))
        coordinates: Any = (x, y)
        mesh_type = "structured"
    else:
        # Use the native point coordinates when no regular raster grid is available
        dataframe = like.ds.compute() if hasattr(like.ds, "compute") else like.ds
        output_shape = (len(dataframe),)
        coordinates = (dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy())
        mesh_type = "unstructured"

    realizations: list[Any] = []
    for _ in range(n_fields):
        # Generate every component independently so their covariances add by construction
        combined = np.zeros(output_shape, dtype=float)
        for component in error_structure.values():
            seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            if component.correlation is None:
                unit_field = np.random.default_rng(seed).standard_normal(output_shape)
            else:
                # Convert the unit variance correlation model through GeoUtils before simulating its spatial field
                gstools = import_optional("gstools")
                variogram = gu.Variogram(
                    lags=np.empty(0),
                    semivariance=np.empty(0),
                    counts=np.empty(0, dtype=np.int64),
                    model=component.correlation,
                )
                covariance_model = variogram.to_gstools(dim=2).model
                unit_field = np.asarray(gstools.SRF(covariance_model, seed=seed)(coordinates, mesh_type=mesh_type))

                # Convert GSTools X/Y structured order to the raster row/column convention
                if is_raster:
                    unit_field = unit_field.T
                if unit_field.shape != output_shape:
                    raise RuntimeError("GSTools returned a random field that does not match the target support.")

            # Scale the unit variance field by this component magnitude on the target support
            magnitude = component.predict_magnitude(predictors)
            combined += np.asarray(magnitude) * unit_field

        # Preserve invalid raster cells because the target object defines both grid and valid support
        if is_raster:
            combined = np.ma.masked_array(combined, mask=np.asarray(like.get_mask()).reshape(output_shape))
        else:
            combined = np.where(np.isfinite(np.asarray(like.data)), combined, np.nan)

        # Wrap each realization in a copy so the source elevations and spatial metadata remain intact
        realizations.append(like.copy(new_array=combined))
    return realizations[0] if n_fields == 1 else realizations


################################################
# 2/ SIMULATION RESULTS AND SHARED CALCULATION
################################################


@dataclass
class PropagationResult:
    """Summary of a calculation repeated on simulated elevation realizations.

    ``estimate`` is the unperturbed calculation; ``mean`` and ``std`` summarize the ensemble. Spatial and labelled
    outputs retain their coordinates and labels. Standard deviations require two valid realizations per output.

    :param estimate: Calculation on the original elevation data.
    :param mean: Ensemble mean, or circular mean when a period is supplied.
    :param std: Sample standard deviation, or circular standard deviation.
    :param nsim: Requested number of realizations.
    :param n_success: Number of successful calculations.
    :param n_valid: Number of finite realizations for each output value.
    :param samples: Individual outputs when requested.
    :param failures: Failed simulation numbers and their error messages.
    :param metadata: Additional diagnostics from a specialized workflow.
    """

    estimate: Any
    mean: Any
    std: Any
    nsim: int
    n_success: int
    n_valid: Any
    samples: list[Any] | None = None
    failures: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _simulation_values(value: Any, template: Any | None = None) -> NDArray[np.floating[Any]]:
    """Extract numerical outputs while checking that every realization uses the same locations or labels."""

    # Keep output labels and spatial coordinates identical throughout the ensemble
    if template is not None:
        if isinstance(template, Raster):
            if not isinstance(value, Raster) or not template.georeferenced_grid_equal(value):
                raise ValueError("Simulation outputs must retain the same raster grid.")

        # Require identical point ordering so each output location is summarized consistently
        elif isinstance(template, PointCloud):
            if not isinstance(value, PointCloud) or not template.georeferenced_coords_equal(value):
                raise ValueError("Simulation outputs must retain the same point coordinates.")

        # Preserve named outputs such as coregistration parameters across all realizations
        elif isinstance(template, (pd.Series, pd.DataFrame)):
            if type(value) is not type(template) or not value.index.equals(template.index):
                raise ValueError("Simulation outputs must retain the same index.")
            if isinstance(template, pd.DataFrame) and not value.columns.equals(template.columns):
                raise ValueError("Simulation outputs must retain the same columns.")

    # Compute only the current realization and preserve masked values as NaN
    if isinstance(value, (Raster, PointCloud)):
        value = value.data
    if hasattr(value, "compute"):
        value = value.compute()
    values = np.ma.asarray(value, dtype=float).filled(np.nan)
    if not np.any(np.isfinite(values)):
        raise ValueError("The operation returned no finite values.")
    return values


def _run_simulations(
    elev: Raster | PointCloud,
    operation: Callable[[Any], Any],
    *,
    error_structure: ErrorStructure,
    predictors: Mapping[str, Any] | None = None,
    nsim: int = 100,
    random_state: int | np.random.Generator | None = None,
    return_samples: bool = False,
    circular_period: float | None = None,
    on_error: Literal["raise", "warn"] = "raise",
) -> PropagationResult:
    """
    Repeat a calculation on perturbed elevations and accumulate its mean and standard deviation.

    Magnitude predictors stay fixed while complete random error fields are added to the elevations. Maintain
    a separate finite count for every output location. Linear outputs use an online mean and variance; circular
    outputs use summed unit vectors. Retain individual results only when requested.
    """

    # Validate the ensemble controls before executing a potentially expensive operation
    if not isinstance(nsim, (int, np.integer)) or isinstance(nsim, bool) or nsim < 2:
        raise ValueError("nsim must be an integer of at least two.")
    if on_error not in {"raise", "warn"}:
        raise ValueError("on_error must be 'raise' or 'warn'.")

    # Validate angular units separately from the model and elevation input types
    if circular_period is not None and (not np.isfinite(circular_period) or circular_period <= 0):
        raise ValueError("circular_period must be finite and positive.")
    if not isinstance(error_structure, ErrorStructure):
        raise TypeError("error_structure must be an ErrorStructure.")
    if not isinstance(elev, (Raster, PointCloud)):
        raise TypeError("elev must be a Raster or PointCloud.")
    rng = np.random.default_rng(random_state)

    # Evaluate the original data separately so nonlinear shifts in the ensemble mean remain visible
    estimate = operation(elev.copy())
    original = _simulation_values(estimate)

    # Allocate per-output counts and running linear moments on the original result shape
    count = np.zeros(original.shape, dtype=np.int64)
    mean = np.zeros(original.shape, dtype=float)
    moment = np.zeros(original.shape, dtype=float)

    # Track circular vectors and optional diagnostics without storing all realizations by default
    sine = np.zeros(original.shape, dtype=float)
    cosine = np.zeros(original.shape, dtype=float)
    failures: dict[int, str] = {}
    samples: list[Any] | None = [] if return_samples else None
    n_success = 0

    # Draw complete component fields so spatial dependence is retained inside every calculation
    for index in range(nsim):
        logging.info("Propagating uncertainty: simulation %d of %d", index + 1, nsim)
        error = generate_random_field(elev, error_structure, predictors=predictors, random_state=rng)
        try:
            # Recompute the full operation and require each result to match the original output locations
            output = operation(elev + error)
            values = _simulation_values(output, template=estimate)
            if values.shape != original.shape:
                raise ValueError("The operation must return the same shape for every realization.")

        # Skip failed calculations only when requested and preserve their simulation number and message
        except Exception as err:
            if on_error == "raise":
                raise
            failures[index + 1] = str(err)
            warnings.warn(f"Simulation {index + 1} of {nsim} failed and was skipped: {err}", stacklevel=2)
            continue

        # Update each finite output independently without retaining the full simulation stack
        valid = np.isfinite(values)
        count[valid] += 1
        if circular_period is None:
            # Use the online variance update to avoid subtracting nearly equal large sums
            delta = values[valid] - mean[valid]
            mean[valid] += delta / count[valid]
            moment[valid] += delta * (values[valid] - mean[valid])
        else:
            # Accumulate unit vectors so angles on opposite sides of zero remain close
            angles = values[valid] * (2 * np.pi / circular_period)
            sine[valid] += np.sin(angles)
            cosine[valid] += np.cos(angles)

        # Count successful calculations separately from the number of finite values at each output
        n_success += 1
        if samples is not None:
            samples.append(output.copy() if hasattr(output, "copy") else output)

    # Require a meaningful ensemble and mask locations with insufficient observations
    if n_success < 2:
        raise RuntimeError(f"Only {n_success} of {nsim} simulations succeeded; at least two are required.")

    # Use the sample variance denominator and leave outputs with fewer than two values undefined
    std = np.full(original.shape, np.nan, dtype=float)
    if circular_period is None:
        std[count > 1] = np.sqrt(np.maximum(moment[count > 1], 0) / (count[count > 1] - 1))
    else:
        # Recover circular mean and spread from vector direction and length, then restore the requested units
        mean = np.asarray(np.mod(np.arctan2(sine, cosine), 2 * np.pi) * circular_period / (2 * np.pi))
        length = np.divide(np.hypot(sine, cosine), count, out=np.zeros_like(mean), where=count > 0)
        length = np.clip(length, np.finfo(float).tiny, 1)
        std[count > 1] = np.sqrt(-2 * np.log(length[count > 1])) * circular_period / (2 * np.pi)
    mean[count == 0] = np.nan

    # Restore the operation's output representation for all compact summaries
    summaries = []
    for summary_values in (mean, std, count):
        if isinstance(estimate, (Raster, PointCloud)):
            summary_values = np.ma.masked_invalid(np.asarray(summary_values, dtype=float))
            summaries.append(estimate.copy(new_array=summary_values))

        # Restore table labels so parameter names and other named outputs remain attached to their statistics
        elif isinstance(estimate, pd.DataFrame):
            summaries.append(pd.DataFrame(summary_values, index=estimate.index, columns=estimate.columns))
        elif isinstance(estimate, pd.Series):
            summaries.append(pd.Series(summary_values, index=estimate.index, name=estimate.name))
        else:
            summaries.append(summary_values.item() if summary_values.ndim == 0 else summary_values)

    # Return the original calculation alongside ensemble summaries and any retained diagnostics
    return PropagationResult(
        estimate, summaries[0], summaries[1], int(nsim), n_success, summaries[2], samples, failures
    )


####################################################
# 3/ SPATIAL, TERRAIN AND COREGISTRATION WORKFLOWS
####################################################


def _propagate_uncertainty_spatial(
    elev: Raster | PointCloud,
    *,
    areas: list[Any],
    weights: Any | None = None,
    **kwargs: Any,
) -> PropagationResult:
    """Propagate elevation errors through area averages with fixed masks and averaging weights."""

    from xdem.uncertainty.analytical import _area_support

    # Resolve the requested geometry once so every realization uses the same spatial support
    masks = [_area_support(area, elev, None)[1] for area in areas]

    # Resolve weights once so every simulated average uses the same observation contributions
    raw_weights = weights.data if isinstance(weights, (Raster, PointCloud)) else weights
    weight_array = np.ones(np.shape(elev.data)) if raw_weights is None else np.asarray(raw_weights, dtype=float)
    if weight_array.shape != np.shape(elev.data) or np.any(~np.isfinite(weight_array)) or np.any(weight_array < 0):
        raise ValueError("weights must be finite, non-negative and match the elevation support.")

    # Reject empty weighted averages before starting the simulations
    if any(not np.any(weight_array[mask] > 0) for mask in masks):
        raise ValueError("Every area must contain a positive total weight.")

    def average(realization: Any) -> NDArray[np.floating[Any]]:
        """Compute one weighted average per fixed area for a single elevation realization."""

        # Reduce each complete realization before estimating the spread between realizations
        values = np.ma.asarray(realization.data, dtype=float)
        return np.array([np.ma.average(values[mask], weights=weight_array[mask]) for mask in masks])

    # Summarize variation between complete area averages, including their spatially correlated errors
    return _run_simulations(elev, average, **kwargs)


def _propagate_uncertainty_terrain(
    elev: Raster,
    *,
    attribute: str = "slope",
    terrain_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PropagationResult:
    """Recompute one terrain attribute for every elevation realization."""

    from xdem.terrain import get_terrain_attribute

    # Keep terrain settings fixed and choose circular summaries for aspect
    if not isinstance(elev, Raster):
        raise TypeError("Terrain uncertainty propagation requires a raster elevation dataset.")

    # Use a circular summary for aspect in the same units as the terrain calculation
    options = dict(terrain_kwargs or {})
    if attribute == "aspect":
        kwargs.setdefault("circular_period", 360.0 if options.get("degrees", True) else 2 * np.pi)

    def derive(realization: Any) -> Any:
        """Recalculate the selected terrain attribute with the fixed terrain options."""

        return get_terrain_attribute(realization, attribute=attribute, **options)

    # Recompute the terrain attribute before accumulating statistics for each output cell
    return _run_simulations(elev, derive, **kwargs)


def _propagate_uncertainty_coreg(
    elev: Raster | PointCloud,
    *,
    reference_elev: Raster | PointCloud,
    coreg_method: Any,
    error_applied_to: Literal["ref", "tba"] = "tba",
    inlier_mask: Any = None,
    precoreg: bool = False,
    kwargs_coreg_fit: Mapping[str, Any] | None = None,
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> PropagationResult:
    """
    Propagate errors through repeated coregistration fits and retain their transformation diagnostics.

    Optionally align once before simulating residual errors. Each subsequent fit receives an independent copy
    of the method. The first calculation gives the original estimate; only simulated fits enter the retained
    parameter table and successful realization list.
    """

    # Share reproducible randomness between initial alignment and independent fit copies
    if error_applied_to not in {"ref", "tba"}:
        raise ValueError("error_applied_to must be 'ref' or 'tba'.")
    rng = np.random.default_rng(random_state)
    options = dict(kwargs_coreg_fit or {})
    if "random_state" in options:
        raise ValueError("Pass random_state to propagation, not kwargs_coreg_fit.")

    # Remove the initial misalignment when the requested uncertainty concerns residual fitting errors
    if precoreg:
        initial = coreg_method.copy()
        initial.fit(reference_elev, elev, inlier_mask=inlier_mask, random_state=rng, **options)
        elev = initial.apply(elev)

    # Retain fit artefacts separately from the six numeric outputs summarized by the shared engine
    rows: list[pd.DataFrame] = []
    fitted: list[Any] = []
    iteration = -1

    # Perturb the selected error source while holding the other elevation dataset fixed
    source = elev if error_applied_to == "tba" else reference_elev

    def fit(realization: Any) -> pd.Series:
        """Fit one realization and return its six translation and rotation parameters."""

        nonlocal iteration
        iteration += 1

        # Place the perturbed realization on the chosen side of the fit
        reference = realization if error_applied_to == "ref" else reference_elev
        aligned = realization if error_applied_to == "tba" else elev

        # Fit a fresh method so one realization cannot alter the next fit or the caller's method
        method = coreg_method.copy()
        method.fit(reference, aligned, inlier_mask=inlier_mask, random_state=rng, **options)
        row = _postproc_coreg_metadata(method)

        # Exclude the original unperturbed calculation, which is evaluated before the simulation loop
        if iteration > 0:
            row["nsim"] = iteration
            rows.append(row)
            fitted.append(method)
        return row.loc[0, ["tx", "ty", "tz", "rx", "ry", "rz"]]

    # Preserve the coregistration policy of reporting failed fits and summarizing successful ones
    kwargs.setdefault("on_error", "warn")
    result = _run_simulations(source, fit, random_state=rng, **kwargs)

    # Attach complete successful fits to the compact transformation summaries
    result.metadata.update({"simulation_table": pd.concat(rows, ignore_index=True), "fitted_coregs": fitted})
    return result


def _postproc_coreg_metadata(c: Any) -> pd.DataFrame:
    """Collect fitted translations, rotations, centroid and optional convergence statistics in one row."""

    from xdem import coreg

    # Get metadata: translations, rotations, centroid, last iteration and last translation/rotation
    output_matrix = c.to_matrix()
    trans_rot_names = ["tx", "ty", "tz", "rx", "ry", "rz"]
    trans_rot = coreg.translations_rotations_from_matrix(output_matrix)
    if "centroid" in c.meta["outputs"]["affine"]:
        output_centroid = c.meta["outputs"]["affine"]["centroid"]
    else:
        output_centroid = (np.nan, np.nan, np.nan)

    # Leave convergence fields undefined for methods that do not perform iterative fitting
    last_statistic_t, last_statistic_r = np.nan, np.nan
    iteration = c.meta["outputs"].get("iterative", {})
    last_it = iteration.get("last_iteration", np.nan)
    df_stats = iteration.get("iteration_stats")

    # Read the final iteration while allowing methods that report translation without rotation
    if df_stats is not None and len(df_stats):
        last_stats = df_stats.loc[df_stats["iteration"] == last_it]
        if len(last_stats):
            last_statistic_t = last_stats["translation"].iloc[0]
            if "rotation" in last_stats:
                last_statistic_r = last_stats["rotation"].iloc[0]

    # Build one labelled row used by both simulation tables and transformation summaries
    output_dict = {trans_rot_names[i]: trans_rot[i] for i in range(len(trans_rot_names))}
    output_dict.update({"ocx": output_centroid[0], "ocy": output_centroid[1], "ocz": output_centroid[2]})
    df = pd.DataFrame(data=[output_dict])
    df["last_iteration"] = last_it
    df["last_translation"] = last_statistic_t
    df["last_rotation"] = last_statistic_r

    return df


################################
# 4/ EMPIRICAL PATCH ESTIMATION
################################


def _patches_convolution(
    values: NDArrayf,
    gsd: float,
    area: float,
    perc_min_valid: float = 80.0,
    patch_shape: str = "circular",
    method: str = "scipy",
    statistic_between_patches: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    return_in_patch_statistics: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, pd.DataFrame]:
    """
    Estimate area uncertainty from moving windows over the error proxy.

    Convert each requested area to a square or circular kernel, compute window means and finite counts, and
    discard windows with too little valid terrain. Select sets of windows spaced one kernel width apart so
    patches do not overlap, then average the spread estimates from the different sets.
    """

    # Get kernel size to match area
    # If circular, it corresponds to the diameter
    if patch_shape.lower() == "circular":
        kernel_size = int(np.round(2 * np.sqrt(area / np.pi) / gsd, decimals=0))
    # If square, to the side length
    elif patch_shape.lower() == "square":
        kernel_size = int(np.round(np.sqrt(area) / gsd, decimals=0))

    else:
        raise ValueError('Kernel shape should be "square" or "circular".')

    logging.info("Computing the convolution on the entire array...")
    mean_img, nb_valid_img, nb_pixel_per_kernel = mean_filter(
        array=values,
        size=kernel_size,
        kernel_shape=patch_shape,
        engine=method,
        preserve_nodata=False,
        return_counts=True,
    )

    # Exclude mean values if number of valid pixels is less than a percentage of the kernel size
    mean_img[nb_valid_img < nb_pixel_per_kernel * perc_min_valid / 100.0] = np.nan

    # A problem with the convolution method compared to the quadrant one is that patches are not independent, which
    # can bias the estimation of spread. To remedy this, we compute spread statistics on patches separated by the
    # kernel size (i.e., the diameter of the circular patch, or side of the square patch) to ensure no dependency

    # There are as many combinations for this calculation as the square of the kernel_size
    logging.info("Computing statistic between patches for all independent combinations...")
    list_statistic_estimates = []
    list_nb_independent_patches = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            statistic = statistic_between_patches(mean_img[i::kernel_size, j::kernel_size].ravel())
            nb_patches = np.count_nonzero(np.isfinite(mean_img[i::kernel_size, j::kernel_size]))
            list_statistic_estimates.append(statistic)
            list_nb_independent_patches.append(nb_patches)

    if return_in_patch_statistics:
        # Create dataframe of independent patches for one independent setting
        df = pd.DataFrame(
            data={
                "nanmean": mean_img[::kernel_size, ::kernel_size].ravel(),
                "count": nb_valid_img[::kernel_size, ::kernel_size].ravel(),
            }
        )

    # We then use the average of the statistic computed for different sets of independent patches to get a more robust
    # estimate
    average_statistic = float(np.nanmean(np.asarray(list_statistic_estimates)))
    nb_independent_patches = float(np.nanmean(np.asarray(list_nb_independent_patches)))
    exact_area = nb_pixel_per_kernel * gsd**2

    if return_in_patch_statistics:
        return average_statistic, nb_independent_patches, exact_area, df
    else:
        return average_statistic, nb_independent_patches, exact_area


def _patches_loop_quadrants(
    values: NDArrayf,
    gsd: float,
    area: float,
    patch_shape: str = "circular",
    n_patches: int = 1000,
    perc_min_valid: float = 80.0,
    statistics_in_patch: Iterable[Callable[[NDArrayf], np.floating[Any]] | str] = (np.nanmean,),
    statistic_between_patches: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    random_state: int | np.random.Generator | None = None,
    return_in_patch_statistics: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, pd.DataFrame]:
    """
    Estimate area uncertainty from terrain patches sampled on a grid of candidate centers.

    Tile the raster for each requested area, retain patches with enough valid data and summarize their central
    values. Return both the spread between patches and their individual statistics for inspection.
    """

    list_statistics_in_patch = list(statistics_in_patch)
    # Add count by default
    list_statistics_in_patch.append("count")

    # Get statistic name
    statistics_name = [f if isinstance(f, str) else f.__name__ for f in list_statistics_in_patch]

    # Define random state
    rng = np.random.default_rng(random_state)

    # Divide raster in quadrants where we can sample
    nx, ny = np.shape(values)

    kernel_size = int(np.round(np.sqrt(area) / gsd, decimals=0))

    # For rectangular quadrants
    nx_sub = int(np.floor((nx - 1) / kernel_size))
    ny_sub = int(np.floor((ny - 1) / kernel_size))
    # For circular patches
    rad = int(np.round(np.sqrt(area / np.pi) / gsd, decimals=0))

    # Compute exact area to provide to checks and return
    if patch_shape.lower() == "square":
        nb_pixel_exact = nx_sub * ny_sub
        exact_area = nb_pixel_exact * gsd**2
    elif patch_shape.lower() == "circular":
        nb_pixel_exact = np.count_nonzero(_create_circular_mask(shape=(nx, ny), radius=rad))
        exact_area = nb_pixel_exact * gsd**2

    # Create list of all possible quadrants
    list_quadrant = [[i, j] for i in range(nx_sub) for j in range(ny_sub)]
    u = 0
    # Keep sampling while there is quadrants left and below maximum number of patch to sample
    remaining_nsamp = n_patches
    list_df = []
    while len(list_quadrant) > 0 and u < n_patches:
        # Draw a random coordinate from the list of quadrants, select more than enough random points to avoid drawing
        # randomly and differencing lists several times
        list_idx_quadrant = rng.choice(len(list_quadrant), size=min(len(list_quadrant), 10 * remaining_nsamp))

        for idx_quadrant in list_idx_quadrant:

            logging.info("Working on a new quadrant")

            # Select center coordinates
            i = list_quadrant[idx_quadrant][0]
            j = list_quadrant[idx_quadrant][1]

            # Get patch by masking the square or circular quadrant
            if patch_shape.lower() == "square":
                patch = values[
                    kernel_size * i : kernel_size * (i + 1), kernel_size * j : kernel_size * (j + 1)
                ].flatten()
            elif patch_shape.lower() == "circular":
                center_x = np.floor(kernel_size * (i + 1 / 2))
                center_y = np.floor(kernel_size * (j + 1 / 2))
                mask = _create_circular_mask((nx, ny), center=(center_x, center_y), radius=rad)
                patch = values[mask]
            else:
                raise ValueError("Patch method must be square or circular.")

            # Check that the patch is complete and has the minimum number of valid values
            nb_pixel_total = len(patch)
            nb_pixel_valid = len(patch[np.isfinite(patch)])
            if nb_pixel_valid >= np.ceil(perc_min_valid / 100.0 * nb_pixel_total) and nb_pixel_total == nb_pixel_exact:
                u = u + 1
                if u > n_patches:
                    break
                logging.info("Found valid quadrant " + str(u) + " (maximum: " + str(n_patches) + ")")

                df = pd.DataFrame()
                df = df.assign(tile=[str(i) + "_" + str(j)])
                for j, statistic in enumerate(list_statistics_in_patch):
                    if isinstance(statistic, str):
                        if statistic == "count":
                            df[statistic] = [nb_pixel_valid]
                        else:
                            raise ValueError('No other string than "count" are supported for named statistics.')
                    else:
                        df[statistics_name[j]] = [statistic(patch[np.isfinite(patch)].astype("float64"))]

                list_df.append(df)

        # Get remaining samples to draw
        remaining_nsamp = n_patches - u
        # Remove quadrants already sampled from list
        list_quadrant = [c for j, c in enumerate(list_quadrant) if j not in list_idx_quadrant]

    if len(list_df) > 0:
        df_all = pd.concat(list_df)
        # The average statistic is computed on the first in-patch statistic
        average_statistic = float(statistic_between_patches(df_all[statistics_name[0]].values))
        nb_independent_patches = np.count_nonzero(np.isfinite(df_all[statistics_name[0]].values))
    else:
        df_all = pd.DataFrame()
        for j, _ in enumerate(list_statistics_in_patch):
            df_all[statistics_name[j]] = [np.nan]
        average_statistic = np.nan
        nb_independent_patches = 0
        warnings.warn("No valid patch found covering this area size, returning NaN for statistic.")

    if return_in_patch_statistics:
        return average_statistic, nb_independent_patches, exact_area, df_all
    else:
        return average_statistic, nb_independent_patches, exact_area


@overload
def patches_method(
    values: NDArrayf | RasterType,
    areas: list[float],
    gsd: float = None,
    stable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    statistics_in_patch: tuple[Callable[[NDArrayf], np.floating[Any]] | str] = (np.nanmean,),
    statistic_between_patches: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    perc_min_valid: float = 80.0,
    patch_shape: str = "circular",
    vectorized: bool = True,
    convolution_method: str = "scipy",
    n_patches: int = 1000,
    *,
    return_in_patch_statistics: Literal[False] = False,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame: ...


@overload
def patches_method(
    values: NDArrayf | RasterType,
    areas: list[float],
    gsd: float = None,
    stable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    statistics_in_patch: tuple[Callable[[NDArrayf], np.floating[Any]] | str] = (np.nanmean,),
    statistic_between_patches: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    perc_min_valid: float = 80.0,
    patch_shape: str = "circular",
    vectorized: bool = True,
    convolution_method: str = "scipy",
    n_patches: int = 1000,
    *,
    return_in_patch_statistics: Literal[True],
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def patches_method(
    values: NDArrayf | RasterType,
    areas: list[float],
    gsd: float = None,
    stable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | VectorType | gpd.GeoDataFrame = None,
    statistics_in_patch: tuple[Callable[[NDArrayf], np.floating[Any]] | str] = (np.nanmean,),
    statistic_between_patches: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    perc_min_valid: float = 80.0,
    patch_shape: str = "circular",
    vectorized: bool = True,
    convolution_method: str = "scipy",
    n_patches: int = 1000,
    return_in_patch_statistics: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monte Carlo patches method that samples multiple patches of terrain, square or circular, of a certain area and
    computes a statistic in each patch. Then, another statistic is computed between all patches. Typically, a statistic
    of central tendency (e.g., the mean) is computed for each patch, then a statistic of spread (e.g., the NMAD)
    is computed on the central tendency of all the patches. This specific procedure gives an empirical estimate of the
    standard error of the mean.

    The function returns the exact areas of the patches, which might differ from the input due to rasterization of the
    shapes.

    By default, the fast vectorized method based on a convolution of all pixels is used, but only works with the mean.
    To compute other statistics (possibly a list), the non-vectorized method that randomly samples quadrants of the
    input array up to a certain number of patches "n_patches" can be used.

    The per-patch statistics can be returned as a concatenated dataframe using the "return_in_patch_statistics"
    argument, not done by default due to large sizes.

    :param values: Values as array or Raster
    :param areas: List of patch areas to process (squared unit of ground sampling distance; exact patch areas might not
        always match these accurately due to rasterization, and are returned as outputs)
    :param gsd: Ground sampling distance
    :param stable_mask: Vector shapefile of stable terrain (if values is Raster), or boolean array of same shape as
        values
    :param unstable_mask: Vector shapefile of unstable terrain (if values is Raster), or boolean array of same shape
        as values
    :param statistics_in_patch: List of statistics to compute in each patch (count is computed by default;
        only mean and count supported for vectorized version)
    :param statistic_between_patches: Statistic to compute between all patches, typically a measure of spread, applied
        to the first in-patch statistic, which is typically the mean
    :param perc_min_valid: Minimum valid area in the patch
    :param patch_shape: Shape of patch, either "circular" or "square"
    :param vectorized: Whether to use the vectorized (convolution) method or the for loop in quadrants
    :param convolution_method: Convolution method to use, either "scipy" or "numba" (only for vectorized)
    :param n_patches: Maximum number of patches to sample (only for non-vectorized)
    :param return_in_patch_statistics: Whether to return the dataframe of statistics for all patches and areas
    :param random_state: Random state or seed number to use for calculations (only for non-vectorized, for testing)

    :return: Dataframe of statistic between patches with independent patches count and exact areas,
        (Optional) Dataframe of per-patch statistics
    """

    if isinstance(values, Raster):
        # Boolean raster and array masks use True for eligible cells, so invert an unstable selection explicitly
        unstable_selection: Any = unstable_mask
        unstable_mode = "outside"
        if isinstance(unstable_mask, Raster):
            unstable_values = np.ma.asarray(unstable_mask.data).filled(False).astype(bool)
            unstable_selection = unstable_mask.copy(new_array=~unstable_values)
            unstable_mode = "inside"
        elif isinstance(unstable_mask, np.ndarray):
            unstable_selection = ~np.ma.asarray(unstable_mask).filled(False).astype(bool)
            unstable_mode = "inside"

        # Select finite stable raster cells through the shared spatial sampling interface
        selected_mask = stable_mask if stable_mask is not None else unstable_selection
        mask_mode = "inside" if stable_mask is not None else unstable_mode
        sampled = values.cosample(values, mask=selected_mask, mask_mode=mask_mode)

        # Apply an additional exclusion when stable and unstable masks were both supplied
        if stable_mask is not None and unstable_mask is not None:
            sampled = sampled.cosample(
                sampled,
                band=1,
                other_band=1,
                mask=unstable_selection,
                mask_mode=unstable_mode,
            )
        values_arr = np.ma.asarray(sampled.data[0], dtype=float).filled(np.nan)
        if gsd is None:
            gsd = values.res[0]
    elif isinstance(values, np.ndarray):
        # Plain arrays have no spatial support for cosample(), so accept only aligned array or raster masks
        if gsd is None:
            raise ValueError("The ground sampling distance must be provided if no Raster object is passed.")
        values_arr = np.ma.asarray(values, dtype=float).filled(np.nan)
        if values_arr.ndim != 2:
            raise ValueError("Patch values must be a two dimensional array.")

        # Combine the optional inclusion and exclusion masks on the unchanged array grid
        selected = np.ones(values_arr.shape, dtype=bool)
        for name, supplied_mask, keep_inside in (
            ("stable", stable_mask, True),
            ("unstable", unstable_mask, False),
        ):
            if supplied_mask is None:
                continue
            if isinstance(supplied_mask, Raster):
                mask_values = supplied_mask.data
            elif isinstance(supplied_mask, np.ndarray):
                mask_values = supplied_mask
            else:
                raise ValueError(f"The {name} mask must be a Raster or NumPy array when values is an array.")

            # Require masks to identify the same cells before applying their Boolean selection
            mask_array = np.ma.asarray(mask_values).filled(False).squeeze()
            if mask_array.shape != values_arr.shape:
                raise ValueError(f"The {name} mask must match the values shape.")
            selected &= mask_array.astype(bool) if keep_inside else ~mask_array.astype(bool)
        values_arr[~selected] = np.nan
    else:
        raise ValueError("The values must be a Raster or NumPy array.")

    # Initialize list of dataframe for the statistic on all patches
    list_stats = []
    list_nb_patches = []
    list_exact_areas = []

    # Initialize a list to concatenate full dataframes if we want to return them
    if return_in_patch_statistics:
        list_df = []

    # Looping on areas
    for area in areas:
        # If vectorized, we run the convolution which only supports mean and count statistics
        if vectorized:
            outputs = _patches_convolution(
                values=values_arr,
                gsd=gsd,
                area=area,
                perc_min_valid=perc_min_valid,
                patch_shape=patch_shape,
                method=convolution_method,
                statistic_between_patches=statistic_between_patches,
                return_in_patch_statistics=return_in_patch_statistics,
            )

        # If not, we run the quadrant loop method that supports any statistic
        else:
            outputs = _patches_loop_quadrants(
                values=values_arr,
                gsd=gsd,
                area=area,
                patch_shape=patch_shape,
                n_patches=n_patches,
                perc_min_valid=perc_min_valid,
                statistics_in_patch=statistics_in_patch,
                statistic_between_patches=statistic_between_patches,
                return_in_patch_statistics=return_in_patch_statistics,
                random_state=random_state,
            )

        list_stats.append(outputs[0])
        list_nb_patches.append(outputs[1])
        list_exact_areas.append(outputs[2])
        if return_in_patch_statistics:
            # Here we'd need to write overload for all the patch subfunctions... maybe we can do this more easily with
            # the function behaviour, ignoring for now.
            df: pd.DataFrame = outputs[3]  # type: ignore
            df["areas"] = area
            df["exact_areas"] = outputs[2]
            list_df.append(df)

    # Produce final dataframe of statistic between patches per area
    df_statistic = pd.DataFrame(
        data={
            statistic_between_patches.__name__: list_stats,
            "nb_indep_patches": list_nb_patches,
            "exact_areas": list_exact_areas,
            "areas": areas,
        }
    )

    if return_in_patch_statistics:
        # Concatenate the complete dataframe
        df_tot = pd.concat(list_df)
        return df_statistic, df_tot
    else:
        return df_statistic

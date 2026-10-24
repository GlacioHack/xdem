# Copyright (c) 2024 xDEM developers
# Copyright (c) 2026 GeoUtils developers
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

"""Estimate spatial uncertainty from covariance models and empirical patches."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import replace
from typing import Any, Literal, overload

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
from geoutils import ErrorMagnitude, Raster
from geoutils.filters import mean_filter
from geoutils.filters.regular import _create_circular_mask
from geoutils.operators import LocalData
from geoutils.operators.reducer import Sum
from geoutils.raster import RasterType
from geoutils.uncertainty import PropagationSummary
from geoutils.uncertainty.predictors import _spatial_predictors
from geoutils.uncertainty.propagation import _numeric_values
from geoutils.vector.vector import VectorType
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.spatial.distance import cdist

from xdem._typing import NDArrayf
from xdem.uncertainty.propagation import _simulate

######################################
# 1/ AREA INPUTS AND COVARIANCE SUMS
######################################


def _area_support(
    area: Any, support: gu.Raster | gu.PointCloud, predictors: Mapping[str, Any] | None
) -> tuple[NDArray[Any], NDArray[np.bool_], dict[str, NDArray[Any]]]:
    """
    Select finite observations and their magnitude predictors within one spatial area.

    This function relies primarily on GeoUtils cosampling.
    """

    if isinstance(area, (int, float, np.number)):
        raise TypeError("A numeric area has no location; use a spatial mask or vector with support.")

    # Reset point indexes to match weights by row
    sampling_support = support
    if isinstance(support, gu.PointCloud):
        sampling_support = gu.PointCloud(support.gdf.reset_index(drop=True), data_column=support.data_name)

    # Select observations and predictors with the same mask
    sampled = sampling_support.cosample(
        sampling_support, auxiliary=predictors, auxiliary_at="self", mask=area, align="reproject", subsample=1
    )

    # Return coordinates and predictors in observation order
    if isinstance(sampled, gu.Raster):
        mask = ~np.ma.getmaskarray(sampled.data[0])
        coordinates = sampled.ij2xy(*np.where(mask))
        values = {name: sampled.data[index + 2][mask].data for index, name in enumerate(predictors or {})}
    else:
        # Match the mask to the original observation order
        mask = np.zeros(len(support.gdf), dtype=bool)
        mask[sampled.ds.index.to_numpy()] = True
        coordinates = (sampled.ds.geometry.x.to_numpy(), sampled.ds.geometry.y.to_numpy())
        values = {name: sampled.ds[name].to_numpy() for name in predictors or {}}
    return np.column_stack(coordinates), mask, values


def _mean_variance(
    coords: Any,
    error_structure: gu.ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None = None,
    weights: Any | None = None,
    subsample: int | None = 1000,
    vectorized: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Compute the variance of a weighted spatial mean by summing component covariances.

    For each observation pair, covariance is the product of its two local magnitudes and its correlation.
    The optional subsample selects only the second observation of each pair; a scaling factor accounts for the
    unselected observations. Also return the squared weighted mean magnitude used to define effective samples.
    """

    # 1/ Prepare coordinates, weights and error magnitudes
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or len(coords) == 0 or np.any(~np.isfinite(coords)):
        raise ValueError("coords must contain finite observation coordinates.")

    # Normalize weights for the mean
    n = len(coords)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and have a positive sum.")
    weights = weights / weights.sum()

    # Select observations for the covariance sum
    if subsample is not None and (not isinstance(subsample, (int, np.integer)) or subsample < 1):
        raise ValueError("subsample must be a positive integer or None.")
    size = n if subsample is None else min(n, subsample)
    indices = np.arange(n) if size == n else np.random.default_rng(random_state).choice(n, size, replace=False)

    # Calculate each component's error magnitude
    magnitudes = []
    for component in error_structure.components.values():
        magnitude = np.asarray(component.predict_magnitude(predictors), dtype=float)
        magnitude = np.broadcast_to(magnitude, (n,))
        if np.any(~np.isfinite(magnitude)):
            raise ValueError("Magnitude predictions must be finite throughout the selected area.")
        magnitudes.append(magnitude)

    # 2/ Sum covariance between observation pairs
    # Process observations in blocks to limit memory use
    variance = 0.0
    block_size = max(1, min(512, 1_000_000 // size)) if vectorized else 1
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances = cdist(coords[start:stop], coords[indices])
        covariance = np.zeros_like(distances)

        # Combine component error magnitudes and correlations
        for component, magnitude in zip(error_structure.components.values(), magnitudes):
            covariance += (
                magnitude[start:stop, None] * magnitude[None, indices] * component.predict_correlation(distances)
            )

        # Apply the weights of both observations in each pair
        variance += float(np.sum(covariance * weights[start:stop, None] * weights[None, indices]))

    # Account for observation pairs omitted by subsampling
    variance *= n / size

    # 3/ Calculate the reference variance for effective sample size
    # Add component variances to get each observation's error magnitude
    local_std = np.sqrt(np.sum(np.square(magnitudes), axis=0))

    # Square the weighted mean error magnitude
    reference_variance = float(np.sum(weights * local_std) ** 2)
    return max(variance, 0.0), reference_variance


#####################################
# 2/ DISCRETE EFFECTIVE SAMPLE SIZE
#####################################


def neff_exact(
    coords: Any,
    error_structure: gu.ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None = None,
    weights: Any | None = None,
    vectorized: bool = True,
) -> float:
    """Compute effective samples from the complete discrete covariance sum.

    :param coords: Observation coordinates shaped as observations by dimensions.
    :param error_structure: Error magnitudes and component correlations.
    :param predictors: Magnitude predictors at the observations.
    :param weights: Non-negative averaging weights, normalized internally.
    :param vectorized: Whether to evaluate covariance rows in blocks.
    :returns: Squared average local standard deviation divided by the variance of the mean.
    """

    # Compute the variance of the spatial mean and the squared average local magnitude
    variance, reference = _mean_variance(
        coords, error_structure, predictors=predictors, weights=weights, subsample=None, vectorized=vectorized
    )

    # Convert their ratio to effective samples; a zero-magnitude model has no defined ratio
    if reference == 0:
        raise ValueError("Effective sample size is undefined for zero error magnitude.")
    return reference / variance if variance > 0 else float("inf")


def neff_hugonnet_approx(
    coords: Any,
    error_structure: gu.ErrorStructure,
    *,
    predictors: Mapping[str, Any] | None = None,
    weights: Any | None = None,
    subsample: int = 1000,
    vectorized: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Approximate effective samples by subsampling one endpoint of the covariance sum.

    :param coords: Observation coordinates shaped as observations by dimensions.
    :param error_structure: Error magnitudes and component correlations.
    :param predictors: Magnitude predictors at the observations.
    :param weights: Non-negative averaging weights, normalized internally.
    :param subsample: Maximum number of second endpoints in the double sum.
    :param vectorized: Whether to evaluate covariance rows in blocks.
    :param random_state: Seed or generator for the endpoint subsample.
    :returns: Effective sample size using the sampled covariance sum.
    """

    # Compute the variance of the spatial mean and the squared average local magnitude
    variance, reference = _mean_variance(
        coords,
        error_structure,
        predictors=predictors,
        weights=weights,
        subsample=subsample,
        vectorized=vectorized,
        random_state=random_state,
    )

    # Convert their ratio to effective samples; a zero-magnitude model has no defined ratio
    if reference == 0:
        raise ValueError("Effective sample size is undefined for zero error magnitude.")
    return reference / variance if variance > 0 else float("inf")


###################################
# 3/ CIRCULAR AREA APPROXIMATIONS
###################################


def _stationary_models(error_structure: gu.ErrorStructure) -> list[tuple[Any, float]]:
    """Extract constant component variances and correlation models for integration over a numeric area."""

    # Require spatially constant magnitudes before integrating without observation coordinates
    models = []
    for component in error_structure.components.values():
        assert isinstance(component.magnitude, ErrorMagnitude)
        if component.magnitude.kind != "constant":
            raise ValueError(
                "Numeric areas require constant magnitudes; supply a spatial area and support for predictors."
            )

        # Ignore zero contributions and reject independent errors that require a sample count
        variance = float(component.predict_magnitude()) ** 2
        if variance == 0:
            continue
        if component.correlation is None:
            raise ValueError("Independent errors require explicit observation support for area propagation.")

        # Expand summed models while carrying their component variance into each base covariance
        pending = [(component.correlation, variance)]
        while pending:
            model, scale = pending.pop()
            if model.nugget:
                raise ValueError("A nugget requires explicit observation support for area propagation.")
            if model.model_name == "sum":
                pending.extend((child, scale) for child in model.components)
            else:
                models.append((model, scale))
    return models


def neff_circular_approx_numerical(area: float, error_structure: gu.ErrorStructure) -> float:
    """Integrate stationary covariance radially over an equivalent circular area.

    This retains the Rolstad-style approximation previously in spatialstats. Numerical quadrature evaluates an
    analytical covariance model; it does not simulate realizations. Use discrete support for independent errors.

    :param area: Positive area in squared coordinate units.
    :param error_structure: Stationary error structure without an independent component or nugget.
    :returns: Effective sample size for the circular approximation.
    """

    # Validate the area before converting it to the radius of an equivalent disk
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area must be finite and positive.")
    models = _stationary_models(error_structure)
    radius = np.sqrt(area / np.pi)

    # Compute the local variance that will be compared with the integrated area variance
    total = sum(scale * model.sill for model, scale in models)
    if total <= 0:
        raise ValueError("Effective sample size is undefined for zero error magnitude.")

    # Weight covariance by radial distance to integrate concentric rings from the center to the edge
    integrated = sum(
        scale * quad(lambda distance, model=model: distance * model.covariance(distance), 0, radius)[0]
        for model, scale in models
    )

    # Normalize by disk area and express the reduction in variance as effective samples
    variance = 2 * integrated / radius**2
    return total / variance


def neff_circular_approx_theoretical(area: float, error_structure: gu.ErrorStructure) -> float:
    """Evaluate the closed circular formulas for spherical, exponential, Gaussian and cubic components.

    :param area: Positive area in squared coordinate units.
    :param error_structure: Stationary error structure using the supported base models.
    :returns: Effective sample size for the circular approximation.
    """

    # Validate the disk area and extract constant component covariance models
    if not np.isfinite(area) or area <= 0:
        raise ValueError("area must be finite and positive.")
    models = _stationary_models(error_structure)
    # Lag l_equiv equal to the radius needed for a disk of equivalent area A
    l_equiv = np.sqrt(area / np.pi)

    # Below, we list exact integral functions over an area A assumed a disk integrated radially from the center

    # Formulas of h * covariance = h * ( psill - variogram ) for each form, then its integral for each form to yield
    # the standard error SE. a1 = range and c1 = partial sill.

    # Spherical: h * covariance = c1 * h * ( 1 - 3/2 * h / a1 + 1/2 * (h/a1)**3 )
    # = c1 * (h - 3/2 * h**2 / a1 + 1/2 * h**4 / a1**3)
    # Spherical: radial integral of above from 0 to L:
    # SE**2 = 2 / (L**2) * c1 * (L**2 / 2 - 3/2 * L**3 / 3 / a1 + 1/2 * 1/5 * L**5 / a1**3)
    # which leads to SE**2 =  c1 * (1 - L / a1 + 1/5 * (L/a1)**3 )
    # If spherical model is above the spherical range a1: SE**2 = c1 /5 * (a1/L)**2

    def spherical_exact_integral(a1: float, c1: float, L: float) -> float:
        """Evaluate the radial covariance integral for a spherical model."""

        if l_equiv <= a1:
            squared_se = c1 * (1 - L / a1 + 1 / 5 * (L / a1) ** 3)
        else:
            squared_se = c1 / 5 * (a1 / L) ** 2
        return squared_se

    # Exponential: h * covariance = c1 * h * exp(-h/a); a = a1/3
    # Exponential: radial integral of above from 0 to L: SE**2 =  2 / (L**2) * c1 * a * (a - exp(-L/a) * (a + L))

    def exponential_exact_integral(a1: float, c1: float, L: float) -> float:
        """Evaluate the radial covariance integral for an exponential model."""

        a = a1 / 3
        squared_se = 2 * c1 * (a / L) ** 2 * (1 - np.exp(-L / a) * (1 + L / a))
        return squared_se

    # Gaussian: h * covariance = c1 * h * exp(-h**2/a**2) ; a = a1/2
    # Gaussian: radial integral of above from 0 to L: SE**2 = 2 / (L**2) * c1 * 1/2 * a**2 * (1 - exp(-L**2/a**2))

    def gaussian_exact_integral(a1: float, c1: float, L: float) -> float:
        """Evaluate the radial covariance integral for a Gaussian model."""

        a = a1 / 2
        squared_se = c1 * (a / L) ** 2 * (1 - np.exp(-(L**2) / a**2))
        return squared_se

    # Cubic: h * covariance = c1 * h * (1 - (7 * (h**2 / a**2)) + ((35 / 4) * (h**3 / a**3)) -
    #                          ((7 / 2) * (h**5 / a**5)) + ((3 / 4) * (h**7 / a**7)))
    # Cubic: radial integral of above from 0 to L:
    # SE**2 = c1 * (6*a**7 -21*a**5*L**2 + 21*a**4*L**3 - 6*a**2*L**5 + L**7) / (6*a**7)

    def cubic_exact_integral(a1: float, c1: float, L: float) -> float:
        """Evaluate the radial covariance integral for a cubic model."""

        if l_equiv <= a1:
            squared_se = (
                c1 * (6 * a1**7 - 21 * a1**5 * L**2 + 21 * a1**4 * L**3 - 6 * a1**2 * L**5 + L**7) / (6 * a1**7)
            )
        else:
            squared_se = 1 / 6 * c1 * a1**2 / L**2
        return squared_se

    # Select the closed formula for each component before combining their area variances
    integrals = {
        "spherical": spherical_exact_integral,
        "exponential": exponential_exact_integral,
        "gaussian": gaussian_exact_integral,
        "cubic": cubic_exact_integral,
    }
    if any(model.model_name not in integrals for model, _ in models):
        raise ValueError("Use numerical quadrature for models without a closed circular formula.")
    variance = sum(
        integrals[model.model_name](model.effective_range, scale * model.partial_sill, l_equiv)
        for model, scale in models
    )
    total = sum(scale * model.sill for model, scale in models)
    if total <= 0:
        raise ValueError("Effective sample size is undefined for zero error magnitude.")
    return total / variance


#####################################
# 4/ PROPAGATION OVER SPATIAL AREAS
#####################################


def _area_variance(
    area: Any,
    error_structure: gu.ErrorStructure,
    *,
    support: Any = None,
    predictors: Mapping[str, Any] | None = None,
    weights: Any = None,
    rasterize_resolution: Any = None,
    **kwargs: Any,
) -> tuple[float, float]:
    """Resolve one area and propagate its full component covariance."""

    # 1/ Use the circular approximation when only the area size is known
    if isinstance(area, (int, float, np.number)):
        if predictors is not None or weights is not None:
            raise ValueError("Numeric areas cannot locate predictors or weights; supply a spatial mask or vector.")
        neff = neff_circular_approx_numerical(float(area), error_structure)
        reference = float(error_structure.predict_variance())
        return reference / neff, reference

    # 2/ Prepare a raster if needed
    if support is None:
        if isinstance(rasterize_resolution, gu.Raster):
            support = rasterize_resolution
        elif isinstance(area, (gu.Vector, gpd.GeoDataFrame)):
            vector = gu.Vector(area) if isinstance(area, gpd.GeoDataFrame) else area
            resolution = rasterize_resolution

            # Default to five pixels across the shortest correlation range
            if resolution is None:
                models = _stationary_models(error_structure)
                ranges = [model.effective_range for model, _ in models if model.effective_range is not None]
                if not ranges:
                    raise ValueError("Supply support or rasterize_resolution for this error structure.")
                resolution = min(ranges) / 5
                warnings.warn("Rasterization resolution set to 20% of the shortest correlation range.", stacklevel=3)

            # Rasterize the polygon
            support = vector.create_mask(res=resolution)
        else:
            raise ValueError("A mask requires a raster or point cloud support.")

    # 3/ Select observations and calculate the variance
    coords, mask, sampled_predictors = _area_support(area, support, predictors)

    # Select weights for the same observations
    selected_weights = None
    if weights is not None:
        raw = weights.data if isinstance(weights, (gu.Raster, gu.PointCloud)) else weights
        raw = np.asarray(raw, dtype=float)
        if raw.shape != mask.shape:
            raise ValueError("weights must match the support shape.")
        selected_weights = raw[mask]

    return _mean_variance(coords, error_structure, predictors=sampled_predictors, weights=selected_weights, **kwargs)


def number_effective_samples(
    area: Any,
    error_structure: gu.ErrorStructure,
    *,
    support: Any = None,
    predictors: Mapping[str, Any] | None = None,
    weights: Any = None,
    rasterize_resolution: Any = None,
    **kwargs: Any,
) -> float:
    """Compute effective samples from an error structure over an area.

    A numeric area uses a stationary circular approximation. A polygon or Boolean mask uses the observation support
    and full component covariance, including spatially varying magnitudes.

    :param area: Numeric area, vector geometry, Boolean mask, or None for all support observations.
    :param error_structure: Error magnitudes and component correlations.
    :param support: Raster or point cloud defining observation coordinates.
    :param predictors: Magnitude predictors evaluated on support.
    :param weights: Non-negative averaging weights on support.
    :param rasterize_resolution: Resolution or raster used to rasterize a vector when support is omitted.
    :param kwargs: Covariance summation options: subsample, vectorized and random_state.
    :returns: Effective sample size relative to the average local standard deviation.
    """

    # Resolve the area and calculate its mean variance using the appropriate integration method
    variance, reference = _area_variance(
        area,
        error_structure,
        support=support,
        predictors=predictors,
        weights=weights,
        rasterize_resolution=rasterize_resolution,
        **kwargs,
    )

    # Compare the squared average local magnitude with the uncertainty of the spatial mean
    if reference == 0:
        raise ValueError("Effective sample size is undefined for zero error magnitude.")
    return reference / variance if variance > 0 else float("inf")


def spatial_error_propagation(
    areas: list[Any],
    error_structure: gu.ErrorStructure,
    *,
    support: Any = None,
    predictors: Mapping[str, Any] | None = None,
    weights: Any = None,
    **kwargs: Any,
) -> list[float]:
    """Propagate component covariance to the standard uncertainty of spatial averages.

    :param areas: Numeric areas, vector geometries or Boolean masks; None selects all support observations.
    :param error_structure: Error magnitudes and component correlations.
    :param support: Raster or point cloud defining observation coordinates.
    :param predictors: Magnitude predictors evaluated on support.
    :param weights: Non-negative averaging weights on support.
    :param kwargs: Additional area rasterization and covariance summation options.
    :returns: Standard uncertainty of the average for each area.
    """

    # Evaluate each area separately because its mask and averaging weights determine its covariance sum
    uncertainties = []
    for area in areas:
        variance, _ = _area_variance(
            area, error_structure, support=support, predictors=predictors, weights=weights, **kwargs
        )

        # Convert the variance of the average back to a standard uncertainty
        uncertainties.append(float(np.sqrt(variance)))
    return uncertainties


################################
# 5/ EMPIRICAL PATCH ESTIMATION
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


#########################################
# 6/ AREA MEAN AND INTEGRAL PROPAGATION
#########################################


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
    """Propagate area means or integrals using GeoUtils sampling and weighted sums."""

    # 1/ Check inputs
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

    # 2/ Select observations and weights for each area
    targets = []
    for area in areas.values():
        coordinates, mask, _ = _area_support(area, source, None)
        positions = np.flatnonzero(mask)
        coefficients = area_weights[positions].copy()
        if coefficients.sum() <= 0:
            raise ValueError("Each area must contain observations with positive weight.")

        # Normalize mean weights or apply pixel areas for raster integrals
        if statistic == "mean":
            coefficients /= coefficients.sum()
        elif isinstance(source, gu.Raster):
            coefficients *= abs(source.transform.determinant)

        # Store selected observations with their weights and original indexes
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

    # Use the weights directly when summing each area
    operator = Sum()
    operator._uses_error_covariance = False

    # 3/ Simulate area means or integrals
    def calculate(value: Any) -> pd.Series:
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

    # 4/ Propagate area uncertainty analytically
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

    # Restore area labels on the results
    moments = {name: pd.Series(getattr(summary, name), index=labels) for name in ("estimate", "mean", "std")}
    return replace(summary, **moments)

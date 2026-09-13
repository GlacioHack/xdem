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

"""Analytical propagation of component covariances to spatial averages.

Circular approximations retain the original radial integration formulas. Discrete propagation sums the full
covariance of each ErrorStructure component, including its magnitudes at both endpoints. The module presents
support preparation, covariance sums, circular approximations and public area workflows in that order.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import geoutils as gu
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.spatial.distance import cdist

from xdem.uncertainty.error_structure import ErrorMagnitude, ErrorStructure

######################################
# 1/ AREA INPUTS AND COVARIANCE SUMS
######################################


def _area_support(
    area: Any, support: gu.Raster | gu.PointCloud, predictors: Mapping[str, Any] | None
) -> tuple[NDArray[Any], NDArray[np.bool_], dict[str, NDArray[Any]]]:
    """
    Select finite observations and their magnitude predictors within one spatial area.

    GeoUtils aligns the area and predictors to the raster or point cloud. Return coordinates, a mask in the
    original observation shape and predictor values in the same order as the selected coordinates.
    """

    # Let GeoUtils align spatial masks and predictors to the observation coordinates
    if isinstance(area, (int, float, np.number)):
        raise TypeError("A numeric area has no location; use a spatial mask or vector with support.")

    # Give point rows positional labels locally so duplicate user indexes cannot misalign averaging weights
    sampling_support = support
    if isinstance(support, gu.PointCloud):
        sampling_support = gu.PointCloud(support.ds.reset_index(drop=True), data_column=support.data_column)

    # Sample the dataset against itself to keep one shared selection for values and predictors
    sampled = sampling_support.cosample(
        sampling_support, auxiliary=predictors, auxiliary_at="self", mask=area, align="reproject", subsample=1
    )

    # Recover grid coordinates and predictor bands in the same order as Boolean weight indexing
    if isinstance(sampled, gu.Raster):
        mask = ~np.ma.getmaskarray(sampled.data[0])
        coordinates = sampled.ij2xy(*np.where(mask))
        values = {name: sampled.data[index + 2][mask].data for index, name in enumerate(predictors or {})}
    else:
        # Retain native point order when selecting weights from the original observation array
        mask = np.zeros(len(support.ds), dtype=bool)
        mask[sampled.ds.index.to_numpy()] = True
        coordinates = (sampled.ds.geometry.x.to_numpy(), sampled.ds.geometry.y.to_numpy())
        values = {name: sampled.ds[name].to_numpy() for name in predictors or {}}
    return np.column_stack(coordinates), mask, values


def _mean_variance(
    coords: Any,
    error_structure: ErrorStructure,
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

    # Normalize spatial coordinates and weights before drawing a covariance subsample
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or len(coords) == 0 or np.any(~np.isfinite(coords)):
        raise ValueError("coords must contain finite observation coordinates.")

    # Normalize non-negative weights so their sum represents one spatial average
    n = len(coords)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (n,) or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be finite, non-negative and have a positive sum.")
    weights = weights / weights.sum()

    # Select the second observation of each pair uniformly, or retain all pairs for the exact result
    if subsample is not None and (not isinstance(subsample, (int, np.integer)) or subsample < 1):
        raise ValueError("subsample must be a positive integer or None.")
    size = n if subsample is None else min(n, subsample)
    indices = np.arange(n) if size == n else np.random.default_rng(random_state).choice(n, size, replace=False)

    # Evaluate each component magnitude once; combining correlations first would lose endpoint information
    magnitudes = []
    for component in error_structure.values():
        magnitude = np.asarray(component.predict_magnitude(predictors), dtype=float)
        magnitude = np.broadcast_to(magnitude, (n,))
        if np.any(~np.isfinite(magnitude)):
            raise ValueError("Magnitude predictions must be finite throughout the selected area.")
        magnitudes.append(magnitude)

    # Process rows in blocks to avoid an allocation proportional to the square of the full support
    variance = 0.0
    block_size = max(1, min(512, 1_000_000 // size)) if vectorized else 1
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances = cdist(coords[start:stop], coords[indices])
        covariance = np.zeros_like(distances)

        # Combine each component using its magnitude at both locations and its own distance correlation
        for component, magnitude in zip(error_structure.values(), magnitudes):
            covariance += (
                magnitude[start:stop, None] * magnitude[None, indices] * component.predict_correlation(distances)
            )
        variance += float(np.sum(covariance * weights[start:stop, None] * weights[None, indices]))

    # Account for unselected second observations in the sampled double sum
    variance *= n / size

    # Retain the historical effective-sample convention using the average local standard deviation
    local_std = np.sqrt(np.sum(np.square(magnitudes), axis=0))
    reference_variance = float(np.sum(weights * local_std) ** 2)
    return max(variance, 0.0), reference_variance


#####################################
# 2/ DISCRETE EFFECTIVE SAMPLE SIZE
#####################################


def neff_exact(
    coords: Any,
    error_structure: ErrorStructure,
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
    error_structure: ErrorStructure,
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


def _stationary_models(error_structure: ErrorStructure) -> list[tuple[Any, float]]:
    """
    Extract constant component variances and correlation models for integration over a numeric area.

    A numeric area has no observation locations. It therefore cannot describe variable magnitudes or the
    number of independent errors. Expand nested sums so each supported covariance can be integrated separately.
    """

    # Require spatially constant magnitudes before integrating without observation coordinates
    models = []
    for component in error_structure.values():
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


def neff_circular_approx_numerical(area: float, error_structure: ErrorStructure) -> float:
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


def neff_circular_approx_theoretical(area: float, error_structure: ErrorStructure) -> float:
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
    error_structure: ErrorStructure,
    *,
    support: Any = None,
    predictors: Mapping[str, Any] | None = None,
    weights: Any = None,
    rasterize_resolution: Any = None,
    **kwargs: Any,
) -> tuple[float, float]:
    """Resolve one area and propagate its full component covariance."""

    # Keep the historical circular approximation for a bare numeric area
    if isinstance(area, (int, float, np.number)):
        if predictors is not None or weights is not None:
            raise ValueError("Numeric areas cannot locate predictors or weights; supply a spatial mask or vector.")
        neff = neff_circular_approx_numerical(float(area), error_structure)
        reference = float(error_structure.predict_variance())
        return reference / neff, reference

    # Rasterize a polygon when no observation support was supplied
    if support is None:
        if isinstance(rasterize_resolution, gu.Raster):
            support = rasterize_resolution
        elif isinstance(area, (gu.Vector, gpd.GeoDataFrame)):
            vector = gu.Vector(area) if isinstance(area, gpd.GeoDataFrame) else area
            resolution = rasterize_resolution

            # Choose a grid fine enough to resolve the shortest correlation range when none is specified
            if resolution is None:
                models = _stationary_models(error_structure)
                ranges = [model.effective_range for model, _ in models if model.effective_range is not None]
                if not ranges:
                    raise ValueError("Supply support or rasterize_resolution for this error structure.")
                resolution = min(ranges) / 5
                warnings.warn("Rasterization resolution set to 20% of the shortest correlation range.", stacklevel=3)

            # Use the rasterized polygon as the integration grid
            support = vector.create_mask(res=resolution)
        else:
            raise ValueError("A mask requires a raster or point cloud support.")

    # Sample complete component predictors before summing endpoint covariances
    coords, mask, sampled_predictors = _area_support(area, support, predictors)

    # Select weights with the same mask and ordering as the component predictors
    selected_weights = None
    if weights is not None:
        raw = weights.data if isinstance(weights, (gu.Raster, gu.PointCloud)) else weights
        raw = np.asarray(raw, dtype=float)
        if raw.shape != mask.shape:
            raise ValueError("weights must match the support shape.")
        selected_weights = raw[mask]

    # Evaluate the weighted covariance sum on the selected observation locations
    return _mean_variance(coords, error_structure, predictors=sampled_predictors, weights=selected_weights, **kwargs)


def number_effective_samples(
    area: Any,
    error_structure: ErrorStructure,
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
    error_structure: ErrorStructure,
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

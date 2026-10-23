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

"""Propagate elevation errors with random simulations and estimate uncertainty from spatial patches."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
from geoutils import Raster
from geoutils.filters import mean_filter
from geoutils.filters.regular import _create_circular_mask
from geoutils.raster import RasterType
from geoutils.vector.vector import VectorType

from xdem._typing import NDArrayf

################################
# EMPIRICAL PATCH ESTIMATION
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

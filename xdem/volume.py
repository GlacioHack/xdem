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

"""Volume change calculation tools (aimed for glaciers)."""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.fill
import scipy.interpolate
from geoutils.raster import RasterType
from geoutils.raster.array import (
    get_array_and_mask,
    get_mask_from_array,
    get_valid_extent,
)
from tqdm import tqdm

import xdem
from xdem._typing import MArrayf, NDArrayf


def hypsometric_binning(
    ddem: NDArrayf,
    ref_dem: NDArrayf,
    bins: float | np.ndarray[Any, np.dtype[np.floating[Any] | np.integer[Any]]] = 50.0,
    kind: str = "fixed",
    aggregation_function: Callable[[NDArrayf], float] = np.median,
) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.
    The elevation bins will be calculated based on all ref_dem valid pixels.
    ddem may contain NaN/masked values over the same area, they will be excluded before the aggregation.

    It is assumed that the dDEM is calculated as 'ref_dem - dem' (not 'dem - ref_dem').

    :param ddem: The dDEM as a 2D or 1D array.
    :param ref_dem: The reference DEM as a 2D or 1D array.
    :param bins: The bin size, count, or array, depending on the binning method ('kind').
    :param kind: The kind of binning to do. Choices: ['fixed', 'count', 'quantile', 'custom'].
    :param aggregation_function: The function to aggregate the elevation values within a bin. Defaults to the median.

    :returns: A Pandas DataFrame with elevation bins and dDEM statistics.
    """
    assert ddem.shape == ref_dem.shape

    # Convert ddem mask into NaN
    ddem, _ = get_array_and_mask(ddem)

    # Extract only the valid values, i.e. valid in ref_dem
    valid_mask = ~get_mask_from_array(ref_dem)
    ddem = np.array(ddem[valid_mask])
    ref_dem = np.array(ref_dem.squeeze()[valid_mask])

    if isinstance(bins, np.ndarray):
        zbins = bins
    elif kind == "fixed":
        zbins = np.arange(ref_dem.min(), ref_dem.max() + bins + 1e-6, step=bins)  # +1e-6 in case min=max (1 point)
    elif kind == "count":
        # Make bins between mean_dem.min() and a little bit above mean_dem.max().
        # The bin count has to be bins + 1 because zbins[0] will be a "below min value" bin, which will be irrelevant.
        zbins = np.linspace(ref_dem.min(), ref_dem.max() + 1e-6 / bins, num=int(bins + 1))
    elif kind == "quantile":
        # Make the percentile steps. The bins + 1 is explained above.
        steps = np.linspace(0, 100, num=int(bins) + 1)
        zbins = np.fromiter((np.percentile(ref_dem, step) for step in steps), dtype=float)
        # The uppermost bin needs to be a tiny amount larger than the highest value to include it.
        zbins[-1] += 1e-6
    elif kind == "custom":
        zbins = bins  # type: ignore
    else:
        raise ValueError(f"Invalid bin kind: {kind}. Choices: ['fixed', 'count', 'quantile', 'custom'].")

    # Generate bins and get bin indices from the mean DEM
    indices = np.digitize(ref_dem, bins=zbins)

    nb_bins = zbins.shape[0] - 1
    # Calculate statistics for each bin.
    # If no values exist, all stats should be nans (except count with should be 0)
    # medians, means, stds, nmads = (np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan, ) * 4
    values = np.full(shape=nb_bins, fill_value=np.nan, dtype=ddem.dtype)
    counts = np.zeros_like(values, dtype=int)
    for i in range(nb_bins):

        values_in_bin = ddem[indices == i + 1]

        # Remove possible Nans
        values_in_bin = values_in_bin[np.isfinite(values_in_bin)]

        # Skip if no values are in the bin.
        if values_in_bin.shape[0] == 0:
            continue

        try:
            values[i - 1] = aggregation_function(values_in_bin)
            counts[i - 1] = values_in_bin.shape[0]
        except IndexError as exception:
            # If custom bins were added, i may exceed the bin range, which will be silently ignored.
            if kind == "custom" and "out of bounds" in str(exception):
                continue
            raise exception

    # Collect the results in a dataframe
    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(zbins), data=np.vstack([values, counts]).T, columns=["value", "count"]
    )

    return output


def interpolate_hypsometric_bins(
    hypsometric_bins: pd.DataFrame,
    value_column: str = "value",
    method: str = "polynomial",
    order: int = 3,
    count_threshold: int | None = None,
) -> pd.DataFrame:
    """
    Interpolate hypsometric bins using any valid Pandas interpolation technique.

    NOTE: It will not extrapolate!

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param value_column: The name of the column in 'hypsometric_bins' to use as values.
    :param method: Any valid Pandas interpolation technique (e.g. 'linear', 'polynomial', 'ffill', 'bfill').
    :param order: The polynomial order to use. Only used if method='polynomial'.
    :param count_threshold: Optional. A pixel count threshold to exclude during the curve fit (requires a 'count'
        column).
    :returns: Bins interpolated with the chosen interpolation method.
    """
    # Copy the bins that will be filled.
    bins = hypsometric_bins.copy()
    # Temporarily set the index to be the midpoint (for interpolation)
    bins.index = bins.index.mid

    # Set all bins that are under a (potentially) specified count to NaN (they should be excluded from interpolation)
    if count_threshold is not None:
        assert "count" in hypsometric_bins.columns, "'count' not a column in the dataframe"
        bins_under_threshold = bins["count"] < count_threshold
        bins.loc[bins_under_threshold, value_column] = np.nan

    # Count number of valid (finite) values
    nvalids = np.count_nonzero(np.isfinite(bins[value_column]))

    if nvalids <= order + 1:
        # Cannot interpolate -> leave as it is
        warnings.warn("Not enough valid bins for interpolation -> returning copy", UserWarning)
        return hypsometric_bins.copy()
    else:
        # Interpolate all bins that are NaN.
        bins[value_column] = bins[value_column].interpolate(method=method, order=order, limit_direction="both")

    # If some points were temporarily set to NaN (to exclude from the interpolation), re-set them.
    if count_threshold is not None:
        bins.loc[bins_under_threshold, value_column] = hypsometric_bins.loc[bins_under_threshold.values, value_column]

    # Return the index to intervals instead of the midpoint.
    bins.index = hypsometric_bins.index

    return bins


def fit_hypsometric_bins_poly(
    hypsometric_bins: pd.DataFrame,
    value_column: str = "value",
    degree: int = 3,
    iterations: int = 1,
    count_threshold: int | None = None,
) -> pd.Series:
    """
    Fit a polynomial to the hypsometric bins.

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param value_column: The name of the column in 'hypsometric_bins' to use as values.
    :param degree: The degree of the polynomial to use.
    :param iterations: The number of iterations to run. \
 At each iteration, values with residuals larger than 3 times the residuals' standard deviation are excluded.
    :param count_threshold: Optional. A pixel count threshold to exclude during the curve fit (requires a 'count'
        column).
    :returns: Bins replaced by the polynomial fit.
    """
    bins = hypsometric_bins.copy()
    bins.index = bins.index.mid

    if count_threshold is not None:
        assert "count" in hypsometric_bins.columns, "'count' not a column in the dataframe"
        bins_under_threshold = bins["count"] < count_threshold
        bins.loc[bins_under_threshold, value_column] = np.nan

    # Remove invalid bins
    valids = np.isfinite(np.asarray(bins[value_column]))

    for _k in range(iterations):

        # Fit polynomial
        x = bins.index[valids]
        y = bins[value_column][valids]
        pcoeff = np.polyfit(x, y, deg=degree)

        # Calculate residuals
        interpolated_values = np.polyval(pcoeff, bins.index)
        residuals = interpolated_values - bins[value_column]
        residuals_std = np.nanstd(residuals.values)

        # Filter outliers further than 3 std
        valids_old = np.copy(valids)
        valids[np.abs(residuals.values) > 3 * residuals_std] = False
        if np.array_equal(valids, valids_old):
            break

    # Save as pandas' DataFrame
    output = pd.DataFrame(
        index=hypsometric_bins.index, data=np.vstack([interpolated_values, bins["count"]]).T, columns=["value", "count"]
    )

    return output


def calculate_hypsometry_area(
    ddem_bins: pd.Series | pd.DataFrame,
    ref_dem: NDArrayf,
    pixel_size: float | tuple[float, float],
    timeframe: str = "reference",
) -> pd.Series:
    """
    Calculate the associated representative area of the given dDEM bins.

    By default, the area bins will be representative of the mean timing between the reference and nonreference DEM:
    elevations = ref_dem - (h_vs_dh_funcion(ref_dem) / 2)
    This can be changed to either "reference":
    elevations = ref_dem
    Or "nonreference":
    elevations = ref_dem - h_vs_dh_function(ref_dem)

    :param ddem_bins: A Series or DataFrame of dDEM values. If a DataFrame is given, the column 'value' will be used.
    :param ref_dem: The reference DEM. This should not have any NaNs.
    :param pixel_size: The xy or (x, y) size of the reference DEM pixels in georeferenced coordinates.
    :param timeframe: The time at which the area bins are representative. Choices: "reference", "nonreference", "mean"

    :returns: The representative area within the given dDEM bins.
    """
    assert not np.any(np.isnan(ref_dem)), "The given reference DEM has NaNs. No NaNs are allowed to calculate area!"

    if timeframe not in ["reference", "nonreference", "mean"]:
        raise ValueError(
            f"Argument 'timeframe={timeframe}' is invalid. Choices: ['reference', 'nonreference', 'mean']."
        )

    if isinstance(ddem_bins, pd.DataFrame):
        ddem_bins = ddem_bins["value"]

    # For timeframe "mean" or "nonreference", check that ddem_bins values can be interpolated at any altitude
    if timeframe in ["mean", "nonreference"]:
        assert not np.any(np.isnan(ddem_bins.values)), "The dDEM bins cannot contain NaNs. Remove or fill them first."

        # Generate a continuous elevation vs. dDEM function
        ddem_func = scipy.interpolate.interp1d(
            ddem_bins.index.mid, ddem_bins.values, kind="linear", fill_value="extrapolate"
        )

    # Generate average elevations by subtracting half of the dDEM's values to the reference DEM
    if timeframe == "mean":
        elevations = ref_dem - (ddem_func(ref_dem.data) / 2)
    elif timeframe == "reference":
        elevations = ref_dem
    elif timeframe == "nonreference":
        elevations = ref_dem - ddem_func(ref_dem.data)

    # Extract the bins from the dDEM series and compute the frequency of points in the bins.
    bins = np.r_[[ddem_bins.index.left[0]], ddem_bins.index.right]
    bin_counts = np.histogram(elevations, bins=bins)[0]

    # Multiply the bin counts with the pixel area to get the full area.
    bin_area = bin_counts * (pixel_size**2 if not isinstance(pixel_size, tuple) else pixel_size[0] * pixel_size[1])

    # Put this in a series which will be returned.
    output = pd.Series(index=ddem_bins.index, data=bin_area)

    return output


def idw_interpolation(
    array: NDArrayf | MArrayf,
    max_search_distance: int = 10,
    extrapolate: bool = False,
    force_fill: bool = False,
) -> NDArrayf:
    """
    Interpolate a 2D array using rasterio's fillnodata.

    :param array: An array with NaNs or a masked array to interpolate.
    :param max_search_distance: The maximum number of pixels to search in all directions to find values \
to interpolate from. The default is 10.
    :param extrapolate: if False, will remove pixels that have been extrapolated by fillnodata. Default is False.
    :param force_fill: if True, will replace all remaining gaps by the median of all valid values. Default is False.

    :returns: A filled array with no NaNs
    """

    # Create a mask for where nans exist
    nan_mask = get_mask_from_array(array)

    interpolated_array = rasterio.fill.fillnodata(
        array.copy(), mask=(~nan_mask).astype("uint8"), max_search_distance=max_search_distance
    )

    # Remove extrapolated values: gaps up to the size of max_search_distance are kept,
    # but surfaces that artificially grow on the edges are removed
    if not extrapolate:
        interp_mask = scipy.ndimage.binary_closing(
            (~nan_mask).squeeze().astype("uint8"), structure=np.ones((max_search_distance - 1,) * 2)
        ).astype("bool")
        if np.ndim(array) == 3:
            interpolated_array[:, ~interp_mask] = np.nan
        else:
            interpolated_array[~interp_mask] = np.nan

    # Fill the nans (values outside of the value boundaries) with the median value
    # This triggers a warning with np.masked_array's because it ignores the mask
    if force_fill:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpolated_array[np.isnan(interpolated_array)] = np.nanmedian(array)
    else:
        # If input is masked array, return a masked array
        extrap_mask = interpolated_array != array.data
        if isinstance(array, np.ma.masked_array):
            interpolated_array = np.ma.masked_array(interpolated_array, mask=(nan_mask & ~extrap_mask))

    return interpolated_array.reshape(array.shape)


def hypsometric_interpolation(
    voided_ddem: NDArrayf | MArrayf,
    ref_dem: NDArrayf | MArrayf,
    mask: NDArrayf,
) -> MArrayf:
    """
    Interpolate a dDEM using hypsometric interpolation within the given mask.

    Using `ref_dem`, elevation bins of constant height (hard-coded to 50 m for now) are created.
    Gaps in `voided-ddem`, within the provided `mask`, are filled with the median dDEM value within that bin.

    :param voided_ddem: A dDEM with voids (either an array with nans or a masked array).
    :param ref_dem: The reference DEM in the dDEM comparison.
    :param mask: A mask to delineate the area that will be interpolated (True means hypsometric will be used).
    """
    # Get ddem array with invalid pixels converted to NaN and mask of invalid pixels
    ddem, ddem_mask = get_array_and_mask(voided_ddem)

    # Get ref_dem array with invalid pixels converted to NaN and mask of invalid pixels
    dem, dem_mask = get_array_and_mask(ref_dem)

    # Make sure the mask does not have e.g. the shape (1, height, width)
    mask = mask.squeeze()

    # A mask of inlier values: The union of the mask and the inverted exclusion masks of both rasters.
    inlier_mask = mask & (~ddem_mask & ~dem_mask)
    if np.count_nonzero(inlier_mask) == 0:
        warnings.warn("No valid data found within mask, returning copy", UserWarning)
        return np.ma.masked_array(data=voided_ddem)

    # Estimate the elevation dependent gradient.
    gradient = xdem.volume.hypsometric_binning(ddem[inlier_mask], dem[inlier_mask])

    # Interpolate possible missing elevation bins in 1D - no extrapolation done here
    interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(gradient)

    gradient_model = scipy.interpolate.interp1d(
        interpolated_gradient.index.mid, interpolated_gradient["value"].values, fill_value="extrapolate"
    )

    # Create an idealized dDEM using the relationship between elevation and dDEM
    idealized_ddem = np.zeros_like(dem)
    idealized_ddem[mask] = gradient_model(dem[mask])

    # Replace ddem gaps with idealized hypsometric ddem, but only within mask
    corrected_ddem = np.where(ddem_mask & mask, idealized_ddem, ddem)

    output = np.ma.masked_array(corrected_ddem, mask=~np.isfinite(corrected_ddem))

    assert output is not None

    return output


def local_hypsometric_interpolation(
    voided_ddem: NDArrayf | MArrayf,
    ref_dem: NDArrayf | MArrayf,
    mask: NDArrayf,
    min_coverage: float = 0.2,
    count_threshold: int | None = 1,
    nodata: float | int = -9999,
    plot: bool = False,
) -> MArrayf:
    """
    Interpolate a dDEM using local hypsometric interpolation.
    The algorithm loops through each features in the vector file.

    The dDEM is assumed to have been created as "voided_ddem = reference_dem - other_dem".

    :param voided_ddem: A dDEM with voids (either an array with nans or a masked array).
    :param ref_dem: The reference DEM in the dDEM comparison.
    :param mask: A raster of same shape as voided_ddem and ref_dem, containing a different non-0 pixel value for \
each geometry on which to loop.
    :param min_coverage: Optional. The minimum coverage fraction to be considered for interpolation.
    :param count_threshold: Optional. A pixel count threshold to exclude during the hypsometric curve fit.
    :param nodata: Optional. No data value to be used for the output masked_array.
    :param plot: Set to True to display intermediate plots.

    :returns: A dDEM with gaps filled by applying a hypsometric interpolation for each geometry in mask, \
for areas filling the min_coverage criterion.
    """
    # Remove any unnecessary dimension
    orig_shape = voided_ddem.shape
    voided_ddem = voided_ddem.squeeze()
    ref_dem = ref_dem.squeeze()
    mask = mask.squeeze()

    # Check that all arrays have same dimensions
    assert voided_ddem.shape == ref_dem.shape == mask.shape

    # Get ddem array with invalid pixels converted to NaN and mask of invalid pixels
    ddem, ddem_mask = get_array_and_mask(voided_ddem)

    # Get ref_dem array with invalid pixels converted to NaN and mask of invalid pixels
    dem, dem_mask = get_array_and_mask(ref_dem)

    # A mask of inlier values: The union of the mask and the inverted exclusion masks of both rasters.
    inlier_mask = (mask != 0) & (~ddem_mask & ~dem_mask)
    if np.count_nonzero(inlier_mask) == 0:
        warnings.warn("No valid data found within mask, returning copy", UserWarning)
        return np.ma.masked_array(voided_ddem)

    if plot:
        plt.matshow(inlier_mask)
        plt.title("inlier mask")
        plt.show()

    # List of indexes to loop on
    geometry_index = np.unique(mask[mask != 0])
    logging.info("Found %d geometries", len(geometry_index))

    # Get fraction of valid pixels for each geometry
    coverage = np.zeros(len(geometry_index))
    for k, index in enumerate(geometry_index):
        local_inlier_mask = inlier_mask & (mask == index)
        total_pixels = np.count_nonzero(mask == index)
        valid_pixels = np.count_nonzero(local_inlier_mask)
        coverage[k] = valid_pixels / float(total_pixels)

    # Filter geometries with too little coverage
    valid_geometry_index = geometry_index[coverage >= min_coverage]
    logging.info("Found %d geometries with sufficient coverage", len(valid_geometry_index))

    idealized_ddem = nodata * np.ones_like(dem)

    for _k, index in enumerate(valid_geometry_index):

        # Mask of valid pixel within geometry
        local_mask = mask == index
        local_inlier_mask = inlier_mask & (local_mask)

        # Estimate the elevation dependent gradient
        gradient = xdem.volume.hypsometric_binning(ddem[local_mask], dem[local_mask])

        # Remove bins with loo low count
        filt_gradient = gradient.copy()
        if count_threshold is not None:
            if count_threshold > 1:
                bins_under_threshold = filt_gradient["count"] < count_threshold
                filt_gradient.loc[bins_under_threshold, "value"] = np.nan

        # Interpolate missing elevation bins
        interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(filt_gradient)

        # At least 2 points needed for interp1d, if not skip feature
        nvalues = len(interpolated_gradient["value"].values)
        if nvalues < 2:
            warnings.warn(
                f"Not enough valid bins for feature with index {index:d} -> skipping interpolation", UserWarning
            )
            continue

        # Create a model for 2D interpolation
        gradient_model = scipy.interpolate.interp1d(
            interpolated_gradient.index.mid, interpolated_gradient["value"].values, fill_value="extrapolate"
        )

        if plot:
            local_ddem = np.where(local_inlier_mask, ddem, np.nan)
            vmax = max(np.abs(np.nanpercentile(local_ddem, [2, 98])))
            rowmin, rowmax, colmin, colmax = get_valid_extent(mask == index)

            plt.figure(figsize=(12, 8))
            plt.subplot(121)
            plt.imshow(
                (mask == index)[rowmin:rowmax, colmin:colmax], cmap="Greys", vmin=0, vmax=2, interpolation="none"
            )

            plt.imshow(
                local_ddem[rowmin:rowmax, colmin:colmax], cmap="RdYlBu", vmin=-vmax, vmax=vmax, interpolation="none"
            )
            plt.colorbar()
            plt.title(f"ddem for geometry # {index:d}")

            plt.subplot(122)
            plt.plot(gradient["value"], gradient.index.mid, label="raw")
            plt.plot(interpolated_gradient, gradient.index.mid, label="interpolated", ls="--")
            plt.xlabel("ddem")
            plt.ylabel("Elevation")
            plt.legend()
            plt.title("Average ddem per elevation bin")
            plt.tight_layout()
            plt.show()

        # Create an idealized dDEM (only considering the dH gradient)
        idealized_ddem[mask == index] = gradient_model(dem[mask == index])

    # Measure the difference between the original dDEM and the idealized dDEM
    assert ddem.shape == idealized_ddem.shape
    ddem_difference = ddem.astype("float32") - idealized_ddem.astype("float32")
    ddem_difference[idealized_ddem == nodata] = np.nan

    # Spatially interpolate the difference between these two products.
    interpolated_ddem_diff = idw_interpolation(np.where(ddem_mask, np.nan, ddem_difference))
    interpolated_ddem_diff[np.isnan(interpolated_ddem_diff)] = 0

    # Correct the idealized dDEM with the difference to the original dDEM.
    corrected_ddem = idealized_ddem + interpolated_ddem_diff

    # Set Nans to nodata
    corrected_ddem[~np.isfinite(corrected_ddem)] = nodata

    output = np.ma.masked_array(
        corrected_ddem, mask=(corrected_ddem == nodata)  # mask=((mask != 0) & (ddem_mask | dem_mask))
    ).reshape(orig_shape)

    assert output is not None

    return output


def get_regional_hypsometric_signal(
    ddem: NDArrayf | MArrayf | RasterType,
    ref_dem: NDArrayf | MArrayf | RasterType,
    glacier_index_map: NDArrayf | RasterType,
    n_bins: int = 20,
    min_coverage: float = 0.05,
) -> pd.DataFrame:
    """
    Get the normalized regional hypsometric elevation change signal, read "the general shape of it".

    :param ddem: The dDEM to analyse.
    :param ref_dem: A void-free reference DEM.
    :param glacier_index_map: An array glacier indices of the same shape as the previous inputs.
    n_bins = 20  # TODO: This should be an argument.
    :param n_bins: The number of elevation bins to subdivide each glacier in.

    :returns: A DataFrame of bin statistics, scaled by elevation and elevation change.
    """
    # Extract the array and mask representations of the arrays.
    ddem_arr, ddem_mask = get_array_and_mask(ddem)
    ref_arr, ref_mask = get_array_and_mask(ref_dem)
    glacier_index_map, _ = get_array_and_mask(glacier_index_map)

    # The reference DEM should be void free
    assert np.count_nonzero(ref_mask) == 0, "Reference DEM has voids"

    # The unique indices are the unique glaciers.
    unique_indices = np.unique(glacier_index_map)

    # Create empty (ddem) value and (pixel) count arrays which will be filled iteratively.
    values = np.full((n_bins, unique_indices.shape[0]), fill_value=np.nan, dtype=float)
    counts = np.full((n_bins, unique_indices.shape[0]), fill_value=np.nan, dtype=float)

    # Start a counter of glaciers that are actually processed.
    count = 0
    # Loop over each unique glacier.
    for i in tqdm(
        np.unique(glacier_index_map),
        desc="Finding regional signal",
        disable=logging.getLogger().getEffectiveLevel() > logging.INFO,
    ):
        # If i ==0, it's assumed to be periglacial.
        if i == 0:
            continue
        # Create a mask representing a particular glacier.
        glacier_values = glacier_index_map == i

        # Stop if the "glacier" is tiny. It might be a cropped glacier outline for example.
        if np.count_nonzero(glacier_values) < 10:
            continue

        # The inlier mask is where that particular glacier is and where nans don't exist.
        inlier_mask = glacier_values & ~ddem_mask

        # Skip if the coverage is below the threshold
        if (np.count_nonzero(inlier_mask) / np.count_nonzero(glacier_values)) < min_coverage:
            continue

        # Extract only the difference and elevation values that correspond to the glacier.
        differences = ddem_arr[inlier_mask]
        elevations = ref_arr[inlier_mask]

        # Run the hypsometric binning.
        try:
            bins = hypsometric_binning(differences, elevations, bins=n_bins, kind="count")
        except ValueError:
            # ValueError: zero-size array to reduction operation minimum which has no identity on "zbins=" call
            continue

        # Min-max scale by elevation.
        bins.index = (bins.index.mid - bins.index.left.min()) / (bins.index.right.max() - bins.index.left.min())

        # Scale by difference.
        bins["value"] = (bins["value"] - np.nanmin(bins["value"])) / (
            np.nanmax(bins["value"]) - np.nanmin(bins["value"])
        )

        # Assign the values and counts to the output array.
        values[:, count] = bins["value"]
        counts[:, count] = bins["count"]

        count += 1

    output = pd.DataFrame(
        data={
            "w_mean": np.nansum(values * counts, axis=1) / np.nansum(counts, axis=1),
            "median": np.nanmedian(values, axis=1),
            "std": np.nanstd(values, axis=1),
            "sigma-1-lower": np.nanpercentile(values, 16, axis=1),
            "sigma-1-upper": np.nanpercentile(values, 84, axis=1),
            "sigma-2-lower": np.nanpercentile(values, 2.5, axis=1),
            "sigma-2-upper": np.nanpercentile(values, 97.5, axis=1),
            "count": np.nansum(counts, axis=1).astype(int),
        },
        index=pd.IntervalIndex.from_breaks(np.linspace(0, 1, n_bins + 1, dtype="float64")),
    )

    return output


def norm_regional_hypsometric_interpolation(
    voided_ddem: NDArrayf | MArrayf | RasterType,
    ref_dem: NDArrayf | MArrayf | RasterType,
    glacier_index_map: NDArrayf | RasterType,
    min_coverage: float = 0.1,
    regional_signal: pd.DataFrame | None = None,
    min_elevation_range: float = 0.33,
    idealized_ddem: bool = False,
) -> NDArrayf:
    """
    Interpolate missing values by scaling the normalized regional hypsometric signal to each glacier separately.

    Only missing values are interpolated. The rest of the glacier's values are fixed.

    :param voided_ddem: The voided dDEM to fill NaNs in.
    :param ref_dem: A void-free reference DEM.
    :param glacier_index_map: An array glacier indices of the same shape as the previous inputs.
    :param min_coverage: The minimum fractional coverage of a glacier to interpolate. Defaults to 10%.
    :param regional_signal: A regional signal is already estimate. Otherwise one will be estimated.
    :param min_elevation_range: The minimum allowed min/max bin range to scale a signal from.\
            Default: 1/3 of the elevation range needs to be present.
    :param idealized_ddem: Replace observed glacier values with the hypsometric signal. Good for error assessments.

    :raises AssertionError: If `ref_dem` has voids.

    :returns: A dDEM where glacier's that fit the min_coverage criterion are interpolated.
    """
    # Extract the array and nan parts of the inputs.
    ddem_arr, ddem_nans = get_array_and_mask(voided_ddem)
    ref_arr, ref_nans = get_array_and_mask(ref_dem)
    glacier_index_map, _ = get_array_and_mask(glacier_index_map)

    # The reference DEM should be void free
    assert np.count_nonzero(ref_nans) == 0, "Reference DEM has voids"

    # If the regional signal was not given as an argument, find it from the dDEM.
    if regional_signal is None:
        regional_signal = get_regional_hypsometric_signal(
            ddem=ddem_arr, ref_dem=ref_arr, glacier_index_map=glacier_index_map
        )

    # The unique indices are the unique glaciers.
    unique_indices = np.unique(glacier_index_map)

    # Make a copy of the dDEM which will be filled iteratively.
    ddem_filled = ddem_arr.copy()
    # Loop over all glaciers and fill the dDEM accordingly.
    for i in tqdm(
        unique_indices, desc="Interpolating dDEM", disable=logging.getLogger().getEffectiveLevel() > logging.INFO
    ):
        if i == 0:  # i==0 is assumed to mean stable ground.
            continue
        # Create a mask representing a particular glacier.
        glacier_values = glacier_index_map == i

        # The inlier mask is where that particular glacier is and where nans don't exist.
        inlier_mask = glacier_values & ~ddem_nans

        # If the fractional coverage is smaller than the given threshold, skip the glacier.
        if (np.count_nonzero(inlier_mask) / np.count_nonzero(glacier_values)) < min_coverage:
            continue

        # Extract only the finite difference and elevation values that correspond to the glacier.
        differences = ddem_arr[inlier_mask]
        elevations = ref_arr[inlier_mask]

        # Get the reference elevation min and max
        elev_min = ref_arr[glacier_values].min()
        elev_max = ref_arr[glacier_values].max()

        # Copy the signal
        signal = regional_signal["w_mean"].copy()
        # Scale the signal elevation midpoints to the glacier elevation range.
        midpoints = signal.index.mid
        midpoints *= elev_max - elev_min
        midpoints += elev_min
        step = midpoints[1] - midpoints[0]
        # Create an interval structure from the midpoints and the step size.
        signal.index = pd.IntervalIndex.from_arrays(left=midpoints - step / 2, right=midpoints + step / 2)

        # Find the hypsometric bins of the glacier.
        hypsometric_bins = hypsometric_binning(
            ddem=differences,
            ref_dem=elevations,
            bins=np.r_[[signal.index.left[0]], signal.index.right],  # This will generate the same steps as the signal.
            kind="custom",
        )
        bin_stds = hypsometric_binning(
            ddem=differences,
            ref_dem=elevations,
            bins=np.r_[[signal.index.left[0]], signal.index.right],
            kind="custom",
            aggregation_function=np.nanstd,
        )
        # Check which of the bins were non-empty.
        non_empty_bins = np.isfinite(hypsometric_bins["value"])

        non_empty_range = np.sum(non_empty_bins[non_empty_bins].index.length)
        full_range = np.sum(hypsometric_bins.index.length)

        if (non_empty_range / full_range) < min_elevation_range:
            continue

        # A theoretical minimum of 2 bins are needed for the curve fit.
        if np.count_nonzero(non_empty_bins) < 2:
            continue

        # The weights are the squared inverse of the standard deviation of each bin.
        bin_weights = bin_stds["value"].values[non_empty_bins] / np.sqrt(
            hypsometric_bins["count"].values[non_empty_bins]
        )
        bin_weights[bin_weights == 0.0] = 1e-8  # Avoid divide by zero problems.

        # Fit linear coefficients to scale the regional signal to the hypsometric bins properly.
        # The inverse of the pixel counts are used as weights, to properly disregard poorly constrained bins.
        with warnings.catch_warnings():
            # curve_fit will sometimes say "can't estimate covariance". This is okay.
            warnings.filterwarnings("ignore", message="covariance")
            coeffs = scipy.optimize.curve_fit(
                f=lambda x, a, b: a * x + b,  # Estimate a linear function "f(x) = ax + b".
                xdata=signal.values[non_empty_bins],  # The xdata is the normalized regional signal
                ydata=hypsometric_bins["value"].values[non_empty_bins],  # The ydata is the actual values.
                p0=[1, 0],  # The initial guess of a and b (doesn't matter too much)
                sigma=bin_weights,
            )[0]

        # Create a linear model from the elevations and the scaled regional signal.
        model = scipy.interpolate.interp1d(
            signal.index.mid, np.poly1d(coeffs)(signal.values), bounds_error=False, fill_value="extrapolate"
        )

        # Find which values to fill using the model (all nans within the glacier extent)
        if not idealized_ddem:
            values_to_fill = glacier_values & ddem_nans
        # If it should be idealized, replace all glacier values with the model
        else:
            values_to_fill = glacier_values
        # Fill the nans using the scaled regional signal.
        ddem_filled[values_to_fill] = model(ref_arr[values_to_fill])

    return ddem_filled

"""Functions to calculate changes in volume (aimed for glaciers)."""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import rasterio.fill
import scipy.interpolate

import xdem


def hypsometric_binning(ddem: np.ndarray, ref_dem: np.ndarray, bins: Union[float, np.ndarray] = 50.0,
                        kind: str = "fixed", aggregation_function: Callable = np.median) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.

    It is assumed that the dDEM is calculated as 'ref_dem - dem' (not 'dem - ref_dem').

    :param ddem: The dDEM as a 2D or 1D array.
    :param ref_dem: The reference DEM as a 2D or 1D array.
    :param bins: The bin size, count, or array, depending on the binning method ('kind').
    :param kind: The kind of binning to do. Choices: ['fixed', 'count', 'quantile', 'custom'].
    :param aggregation_function: The function to aggregate the elevation values within a bin. Defaults to the median.

    :returns: A Pandas DataFrame with elevation bins and dDEM statistics.
    """
    assert ddem.shape == ref_dem.shape

    # Remove all nans, and flatten the inputs.
    nan_mask = np.logical_or(
        np.isnan(ddem) if not isinstance(ddem, np.ma.masked_array) else ddem.mask,
        np.isnan(ref_dem) if not isinstance(ref_dem, np.ma.masked_array) else ref_dem.mask
    )
    # Extract only the valid values and (if relevant) convert from masked_array to array
    ddem = np.array(ddem[~nan_mask])
    ref_dem = np.array(ref_dem[~nan_mask])

    # If the bin size should be seen as a percentage.
    if kind == "fixed":
        zbins = np.arange(ref_dem.min(), ref_dem.max() + bins + 1e-6, step=bins)  # +1e-6 in case min=max (1 point)
    elif kind == "count":
        # Make bins between mean_dem.min() and a little bit above mean_dem.max().
        # The bin count has to be bins + 1 because zbins[0] will be a "below min value" bin, which will be irrelevant.
        zbins = np.linspace(ref_dem.min(), ref_dem.max() + 1e-6 / bins, num=int(bins + 1))
    elif kind == "quantile":
        # Make the percentile steps. The bins + 1 is explained above.
        steps = np.linspace(0, 100, num=int(bins) + 1)
        zbins = np.fromiter(
            (np.percentile(ref_dem, step) for step in steps),
            dtype=float
        )
        # The uppermost bin needs to be a tiny amount larger than the highest value to include it.
        zbins[-1] += 1e-6
    elif kind == "custom":
        zbins = bins
    else:
        raise ValueError(f"Invalid bin kind: {kind}. Choices: ['fixed', 'count', 'quantile', 'custom']")

    # Generate bins and get bin indices from the mean DEM
    indices = np.digitize(ref_dem, bins=zbins)

    # Calculate statistics for each bin.
    # If no values exist, all stats should be nans (except count with should be 0)
    # medians, means, stds, nmads = (np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan, ) * 4
    values = np.zeros(shape=zbins.shape[0] - 1, dtype=ddem.dtype) * np.nan
    counts = np.zeros_like(values, dtype=int)
    for i in np.arange(indices.min(), indices.max() + 1):
        values_in_bin = ddem[indices == i]
        # Skip if no values are in the bin.
        if values_in_bin.shape[0] == 0:
            continue

        values[i - 1] = aggregation_function(values_in_bin)
        # medians[i - 1] = np.median(values_in_bin)
        # means[i - 1] = np.mean(values_in_bin)
        # stds[i - 1] = np.std(values_in_bin)
        counts[i - 1] = values_in_bin.shape[0]
        # nmads[i - 1] = xdem.spatial_tools.nmad(values_in_bin)

    # Collect the results in a dataframe
    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(zbins),
        data=np.vstack([
            values, counts
        ]).T,
        columns=["value", "count"]
    )

    return output


def interpolate_hypsometric_bins(hypsometric_bins: pd.DataFrame, value_column="value", method="polynomial", order=3,
                                 count_threshold: Optional[int] = None) -> pd.Series:
    """
    Interpolate hypsometric bins using any valid Pandas interpolation technique.

    NOTE: It will not extrapolate!

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param value_column: The name of the column in 'hypsometric_bins' to use as values.
    :param method: Any valid Pandas interpolation technique (e.g. 'linear', 'polynomial', 'ffill', 'bfill').
    :param order: The polynomial order to use. Only used if method='polynomial'.
    :param count_threshold: Optional. A pixel count threshold to exclude during the curve fit (requires a 'count' column).
    :returns: Bins interpolated with the chosen interpolation method.
    """
    bins = hypsometric_bins.copy()
    bins.index = bins.index.mid

    if count_threshold is not None:
        assert "count" in hypsometric_bins.columns, f"'count' not a column in the dataframe"
        bins_under_threshold = bins["count"] < count_threshold
        bins.loc[bins_under_threshold, value_column] = np.nan

    interpolated_values = bins[value_column].interpolate(method=method, order=order, limit_direction="both")

    if count_threshold is not None:
        interpolated_values.loc[bins_under_threshold] = hypsometric_bins.loc[bins_under_threshold.values, value_column]
    interpolated_values.index = hypsometric_bins.index

    return interpolated_values


def calculate_hypsometry_area(ddem_bins: Union[pd.Series, pd.DataFrame], ref_dem: np.ndarray,
                              pixel_size: Union[float, tuple[float, float]],
                              timeframe: str = "reference") -> pd.Series:
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
        raise ValueError(f"Argument 'timeframe={timeframe}' is invalid. Choices: ['reference', 'nonreference', 'mean']")

    if isinstance(ddem_bins, pd.DataFrame):
        ddem_bins = ddem_bins["value"]
    assert not np.any(np.isnan(ddem_bins.values)), "The dDEM bins cannot contain NaNs. Remove or fill them first."
    # Generate a continuous elevation vs. dDEM function
    ddem_func = scipy.interpolate.interp1d(ddem_bins.index.mid, ddem_bins.values,
                                           kind="linear", fill_value="extrapolate")
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
    bin_area = bin_counts * (pixel_size ** 2 if not isinstance(pixel_size, tuple) else pixel_size[0] * pixel_size[1])

    # Put this in a series which will be returned.
    output = pd.Series(index=ddem_bins.index, data=bin_area)

    return output


def linear_interpolation(array: Union[np.ndarray, np.ma.masked_array]) -> np.ndarray:
    """
    Interpolate a 2D array using bilinear interpolation.

    :param array: An array with NaNs or a masked array to interpolate.
    :returns: A filled array with no NaNs
    """
    # Create a mask for where nans exist
    nan_mask = (array.mask | np.isnan(array)) if isinstance(array, np.ma.masked_array) else np.isnan(array)

    interpolated_array = rasterio.fill.fillnodata(array.copy(), mask=~nan_mask.astype("uint8"))

    # Fill the nans (values outside of the value boundaries) with the median value
    # This triggers a warning with np.masked_array's because it ignores the mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interpolated_array[np.isnan(interpolated_array)] = np.nanmedian(array)

    return interpolated_array.reshape(array.shape)


def hypsometric_interpolation(voided_ddem: Union[np.ndarray, np.ma.masked_array],
                              ref_dem: Union[np.ndarray, np.ma.masked_array],
                              mask: np.ndarray) -> np.ma.masked_array:
    """
    Interpolate a dDEM using hypsometric interpolation within the given mask.

    The dDEM is assumed to have been created as "voided_ddem = reference_dem - other_dem".
    Areas outside the mask will be linearly interpolated, but are masked out.

    :param voided_ddem: A dDEM with voids (either an array with nans or a masked array).
    :param ref_dem: The reference DEM in the dDEM comparison.
    :param mask: A mask to delineate the area that will be interpolated (True means hypsometric will be used).
    """
    ddem = np.asarray(voided_ddem)
    # The exclusion mask is the union of the nan mask and the potential masked_array mask
    ddem_mask = np.isnan(ddem) | (voided_ddem.mask if isinstance(voided_ddem, np.ma.masked_array) else False)
    # Maybe temporary: Make sure that interpolation works as it should by messing up the masked values.
    ddem[ddem_mask] = np.nan

    dem = np.asarray(ref_dem)
    # The exclusion mask is the union of the nan mask and the potential masked_array mask
    dem_mask = np.isnan(dem) | (ref_dem.mask if isinstance(ref_dem, np.ma.masked_array) else False)

    # A mask of inlier values: The union of the mask and the inverted exclusion masks of both rasters.
    inlier_mask = mask & (~ddem_mask & ~dem_mask)
    if np.count_nonzero(inlier_mask) == 0:
        warnings.warn("No valid data found within mask, returning copy", UserWarning)
        return np.copy(ddem)

    # Estimate the elevation dependent gradient.
    gradient = xdem.volume.hypsometric_binning(
        ddem[inlier_mask],
        dem[inlier_mask]
    )

    #
    interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(gradient)

    gradient_model = scipy.interpolate.interp1d(
        interpolated_gradient.index.mid,
        interpolated_gradient.values,
        fill_value="extrapolate"
    )

    # Create an idealized dDEM (only considering the dH gradient)
    idealized_ddem = gradient_model(dem)
    idealized_ddem[~mask] = 0.0

    # Measure the difference between the original dDEM and the idealized dDEM
    assert ddem.shape == idealized_ddem.shape
    ddem_difference = ddem.astype("float64") - idealized_ddem.astype("float64")

    # Spatially interpolate the difference between these two products.
    #interpolated_ddem_diff = ddem_difference.copy()
    #interpolated_ddem_diff[ddem_mask] = np.nan
    # rasterio.fill.fillnodata(
    #    interpolated_ddem_diff, mask=~np.isnan(interpolated_ddem_diff))
    interpolated_ddem_diff = linear_interpolation(np.where(ddem_mask, np.nan, ddem_difference))

    # Correct the idealized dDEM with the difference to the original dDEM.
    corrected_ddem = idealized_ddem + interpolated_ddem_diff

    output = np.ma.masked_array(
        corrected_ddem,
        mask=(~mask & (ddem_mask | dem_mask))
    )

    assert output is not None

    return output

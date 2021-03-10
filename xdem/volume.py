"""Functions to calculate changes in volume (aimed for glaciers)."""
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import scipy.interpolate


def hypsometric_binning(ddem: np.ndarray, ref_dem: np.ndarray, bin_size=50,
                        normalized_bin_size: bool = False, aggregation_function: Callable = np.median) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.

    It is assumed that the dDEM is calculated as 'ref_dem - dem' (not 'dem - ref_dem').

    :param ddem: The dDEM as a 2D or 1D array.
    :param ref_dem: The reference DEM as a 2D or 1D array.
    :param bin_size: The bin interval size in georeferenced units (or percent; 0-100, if normalized_bin_size=True)
    :param normalized_bin_size: If the given bin size should be parsed as a percentage of the glacier's elevation range.
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

    # Calculate the mean representative elevations between the two DEMs
    mean_dem = ref_dem - (ddem / 2)

    # If the bin size should be seen as a percentage.
    if normalized_bin_size:
        assert 0 < bin_size < 100

        # Get the statistical elevation range to normalize the bin size with
        elevation_range = np.percentile(mean_dem, 99) - np.percentile(mean_dem, 1)
        bin_size = elevation_range / bin_size

    # Generate bins and get bin indices from the mean DEM
    bins = np.arange(mean_dem.min(), mean_dem.max() + bin_size, step=bin_size)
    indices = np.digitize(mean_dem, bins=bins)

    # Calculate statistics for each bin.
    # If no values exist, all stats should be nans (except count with should be 0)
    # medians, means, stds, nmads = (np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan, ) * 4
    values = np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan
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
        index=pd.IntervalIndex.from_breaks(bins),
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
                              pixel_size: Union[float, tuple[float, float]]) -> pd.Series:
    """
    Calculate the associated representative area of the given dDEM bins.

    :param ddem_bins: A Series or DataFrame of dDEM values. If a DataFrame is given, the column 'value' will be used.
    :param ref_dem: The reference DEM. This should not have any NaNs.
    :param pixel_size: The xy or (x, y) size of the reference DEM pixels in georeferenced coordinates.

    :returns: The representative area within the given dDEM bins.
    """
    assert not np.any(np.isnan(ref_dem)), "The given reference DEM has NaNs. No NaNs are allowed to calculate area!"

    if isinstance(ddem_bins, pd.DataFrame):
        ddem_bins = ddem_bins["value"]
    assert not np.any(np.isnan(ddem_bins.values)), "The dDEM bins cannot contain NaNs. Remove or fill them first."
    # Generate a continuous elevation vs. dDEM function
    ddem_func = scipy.interpolate.interp1d(ddem_bins.index.mid, ddem_bins.values,
                                           kind="linear", fill_value="extrapolate")
    # Generate average elevations by subtracting half of the dDEM's values to the reference DEM
    mean_dem = ref_dem - (ddem_func(ref_dem) / 2)

    # Extract the bins from the dDEM series and compute the frequency of points in the bins.
    bins = np.r_[[ddem_bins.index.left[0]], ddem_bins.index.right]
    bin_counts = np.histogram(mean_dem, bins=bins)[0]

    # Multiply the bin counts with the pixel area to get the full area.
    bin_area = bin_counts * (pixel_size ** 2 if not isinstance(pixel_size, tuple) else pixel_size[0] * pixel_size[1])

    # Put this in a series which will be returned.
    output = pd.Series(index=ddem_bins.index, data=bin_area)

    return output

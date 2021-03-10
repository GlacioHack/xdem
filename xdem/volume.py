"""Functions to calculate changes in volume (aimed for glaciers)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import xdem


def hypsometric_binning(ddem: np.ndarray, ref_dem: np.ndarray, bin_size=50,
                        normalized_bin_size: bool = False) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.

    It is assumed that the dDEM is calculated as 'ref_dem - dem' (not 'dem - ref_dem').

    :param ddem: The dDEM as a 2D or 1D array.
    :param ref_dem: The reference DEM as a 2D or 1D array.
    :param bin_size: The bin interval size in georeferenced units (or percent; 0-100, if normalized_bin_size=True)
    :param normalized_bin_size: If the given bin size should be parsed as a percentage of the glacier's elevation range.

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
    medians, means, stds, nmads = (np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan, ) * 4
    counts = np.zeros_like(medians, dtype=int)
    for i in np.arange(indices.min(), indices.max() + 1):
        values_in_bin = ddem[indices == i]
        # Skip if no values are in the bin.
        if values_in_bin.shape[0] == 0:
            continue

        medians[i - 1] = np.median(values_in_bin)
        means[i - 1] = np.mean(values_in_bin)
        stds[i - 1] = np.std(values_in_bin)
        counts[i - 1] = values_in_bin.shape[0]
        nmads[i - 1] = xdem.spatial_tools.nmad(values_in_bin)

    # Collect the results in a dataframe
    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(bins),
        data=np.vstack([
            medians, means, stds, counts, nmads
        ]).T,
        columns=["median", "mean", "std", "count", "nmad"]
    )

    return output


def interpolate_hypsometric_bins(hypsometric_bins: pd.DataFrame, height_column="median", method="polynomial", order=3,
                                 count_threshold: Optional[int] = None) -> pd.Series:
    """
    Interpolate hypsometric bins using any valid Pandas interpolation technique.

    NOTE: It will not extrapolate!

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param height_column: The name of the column in 'hypsometric_bins' to use as heights.
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
        bins.loc[bins_under_threshold, height_column] = np.nan

    interpolated_values = bins[height_column].interpolate(method=method, order=order, limit_direction="both")

    if count_threshold is not None:
        interpolated_values.loc[bins_under_threshold] = hypsometric_bins.loc[bins_under_threshold.values, height_column]
    interpolated_values.index = hypsometric_bins.index

    return interpolated_values

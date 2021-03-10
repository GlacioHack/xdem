from __future__ import annotations

import numpy as np
import pandas as pd


def hypsometric_binning(ddem: np.ndarray, dem: np.ndarray, bin_size=50,
                        normalized_bin_size: bool = False) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.

    :param ddem: The dDEM as a 2D or 1D array.
    :param dem: The reference DEM as a 2D or 1D array.
    :param bin_size: The bin interval size in georeferenced units (or percent; 0-100, if normalized_bin_size=True)
    :param normalized_bin_size: If the given bin size should be parsed as a percentage of the glacier's elevation range.

    :returns: A Pandas DataFrame with elevation bins and dDEM statistics.
    """

    assert ddem.shape == dem.shape
    # Remove all nans, and flatten the inputs.
    nan_mask = np.isnan(ddem) | np.isnan(dem)
    ddem = ddem[~nan_mask]
    dem = dem[~nan_mask]

    # Calculate the mean representative elevations between the two DEMs
    mean_dem = dem - (ddem / 2)

    # If the bin size should be seen as a percentage.
    if normalized_bin_size:
        assert bin_size > 0 and bin_size < 100

        # Get the statistical elevation range to normalize the bin size with
        elevation_range = np.percentile(mean_dem, 99) - np.percentile(mean_dem, 1)
        bin_size = elevation_range / bin_size

    # Generate bins and get bin indices from the mean DEM
    bins = np.arange(mean_dem.min(), mean_dem.max() + bin_size, step=bin_size)
    indices = np.digitize(mean_dem, bins=bins)

    # Calculate statistics for each bin.
    # If no values exist, all stats should be nans (except count with should be 0)
    medians = np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) + np.nan
    means = medians.copy()
    stds = medians.copy()
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

    # Collect the results in a dataframe
    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(bins),
        data=np.vstack([
            medians, means, stds, counts
        ]).T,
        columns=["median", "mean", "std", "count"]
    )

    return output

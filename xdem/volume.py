from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hypsometric_binning(ddem: np.ndarray, dem: np.ndarray):

    assert ddem.shape == dem.shape
    nan_mask = np.isnan(ddem) | np.isnan(dem)
    ddem = ddem[~nan_mask]
    dem = dem[~nan_mask]

    old_dem = dem - ddem
    elevation_range = np.percentile(old_dem, 99) - np.percentile(old_dem, 1)

    bin_size = 50 if elevation_range > 500 else elevation_range / 10

    bins = np.arange(old_dem.min(), old_dem.max() + bin_size, step=bin_size)
    indices = np.digitize(old_dem, bins=bins)

    medians = np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) + np.nan
    means = medians.copy()
    stds = medians.copy()
    counts = np.zeros_like(medians, dtype=int)
    for i in np.unique(indices):
        values_in_bin = ddem[indices == i]
        if values_in_bin.shape[0] == 0:
            continue

        medians[i - 1] = np.median(values_in_bin)
        means[i - 1] = np.mean(values_in_bin)
        stds[i - 1] = np.std(values_in_bin)
        counts[i - 1] = values_in_bin.shape[0]

    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(bins),
        data=np.vstack([
            medians, means, stds, counts
        ]).T,
        columns=["median", "mean", "std", "count"]
    )

    # more options can be specified also
    print(output.to_string())
    raise ValueError

    return output

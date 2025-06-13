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

"""Spatial statistical tools to estimate uncertainties related to DEMs"""
from __future__ import annotations

import inspect
import itertools
import logging
import math as m
import multiprocessing as mp
import warnings
from typing import Any, Callable, Iterable, Literal, TypedDict, overload

import geopandas as gpd
import geoutils as gu
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
import scipy.ndimage
from geoutils.raster import Mask, Raster, RasterType, subsample_array
from geoutils.raster.array import get_array_and_mask
from geoutils.vector.vector import Vector, VectorType
from numba import prange
from numpy.typing import ArrayLike
from packaging.version import Version
from scipy import integrate
from scipy.interpolate import RegularGridInterpolator, griddata
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd

from xdem._typing import NDArrayb, NDArrayf
from xdem.misc import deprecate

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    try:
        import skgstat as skg

        _has_skgstat = True
        if Version(skg.__version__) < Version("1.0.18"):
            raise ImportWarning(f"scikit-gstat>=1.0.18 is recommended, current version is {skg.__version__}.")
    except ImportError:
        _has_skgstat = False


@deprecate(
    removal_version=Version("0.4"), details="xdem.spatialstats.nmad is being deprecated in favor of geoutils.stats.nmad"
)
def nmad(data: NDArrayf, nfact: float = 1.4826) -> np.floating[Any]:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.
    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. Höhle and Höhle (2009), http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    :param data: Input array or raster
    :param nfact: Normalization factor for the data

    :returns nmad: (normalized) median absolute deviation of data.
    """
    return gu.stats.nmad(data, nfact)


def nd_binning(
    values: NDArrayf,
    list_var: list[NDArrayf],
    list_var_names: list[str],
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf, ...] | None = None,
    statistics: Iterable[str | Callable[[NDArrayf], np.floating[Any]]] = ("count", np.nanmedian, gu.stats.nmad),
    list_ranges: list[float] | None = None,
) -> pd.DataFrame:
    """
    N-dimensional binning of values according to one or several explanatory variables with computed statistics in
    each bin. By default, the sample count, the median and the normalized absolute median deviation (NMAD). The count
    is always computed, no matter user input.
    Values input is a (N,) array and variable input is a L-sized list of flattened arrays of similar dimensions (N,).
    For more details on the format of input variables, see documentation of scipy.stats.binned_statistic_dd.

    :param values: Values array of size (N,)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param list_var_names: List of size (L) of names of the explanatory variables
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory
        variables; defaults to 10 bins
    :param statistics: List of size (X) of statistics to be computed; defaults to count, median and nmad
    :param list_ranges: List of size (L) of minimum and maximum ranges to bin the explanatory variables; defaults to
        min/max of the data
    :return:
    """

    # We separate 1d, 2d and nd binning, because propagating statistics between different dimensional binning is not
    # always feasible using scipy because it allows for several dimensional binning, while it's not straightforward in
    # pandas
    if list_var_bins is None:
        list_var_bins = (10,) * len(list_var_names)
    elif isinstance(list_var_bins, (int, np.integer)):
        list_var_bins = (list_var_bins,) * len(list_var_names)

    # Flatten the arrays if this has not been done by the user
    values = values.ravel()
    list_var = [var.ravel() for var in list_var]

    # Remove no data values
    valid_data = np.logical_and.reduce([np.isfinite(values)] + [np.isfinite(var) for var in list_var])
    values = values[valid_data]
    list_var = [var[valid_data] for var in list_var]

    statistics = list(statistics)
    # In case the statistics are user-defined, and they forget count, we add it for later calculation or plotting
    if "count" not in statistics:
        statistics.insert(0, "count")

    statistics_name = [f if isinstance(f, str) else f.__name__ for f in statistics]

    # Get binned statistics in 1d: a simple loop is sufficient
    list_df_1d = []
    for i, var in enumerate(list_var):
        df_stats_1d = pd.DataFrame()
        # Get statistics
        for j, statistic in enumerate(statistics):
            stats_binned_1d, bedges_1d = binned_statistic(
                x=var, values=values, statistic=statistic, bins=list_var_bins[i], range=list_ranges
            )[:2]
            # Save in a dataframe
            df_stats_1d[statistics_name[j]] = stats_binned_1d
        # We need to get the middle of the bins from the edges, to get the same dimension length
        df_stats_1d[list_var_names[i]] = pd.IntervalIndex.from_breaks(bedges_1d, closed="left")
        # Report number of dimensions used
        df_stats_1d.insert(0, "nd", 1)

        list_df_1d.append(df_stats_1d)

    # Get binned statistics in 2d: all possible 2d combinations
    list_df_2d = []
    if len(list_var) > 1:
        combs = list(itertools.combinations(list_var_names, 2))
        for _, comb in enumerate(combs):
            var1_name, var2_name = comb
            # Corresponding variables indexes
            i1, i2 = list_var_names.index(var1_name), list_var_names.index(var2_name)
            df_stats_2d = pd.DataFrame()
            for j, statistic in enumerate(statistics):
                stats_binned_2d, bedges_var1, bedges_var2 = binned_statistic_2d(
                    x=list_var[i1],
                    y=list_var[i2],
                    values=values,
                    statistic=statistic,
                    bins=[list_var_bins[i1], list_var_bins[i2]],
                    range=list_ranges,
                )[:3]
                # Get statistics
                df_stats_2d[statistics_name[j]] = stats_binned_2d.flatten()
            # Derive interval indexes and convert bins into 2d indexes
            ii1 = pd.IntervalIndex.from_breaks(bedges_var1, closed="left")
            ii2 = pd.IntervalIndex.from_breaks(bedges_var2, closed="left")
            df_stats_2d[var1_name] = [i1 for i1 in ii1 for i2 in ii2]
            df_stats_2d[var2_name] = [i2 for i1 in ii1 for i2 in ii2]
            # Report number of dimensions used
            df_stats_2d.insert(0, "nd", 2)

            list_df_2d.append(df_stats_2d)

    # Get binned statistics in nd, without redoing the same stats
    df_stats_nd = pd.DataFrame()
    if len(list_var) > 2:
        for j, statistic in enumerate(statistics):
            stats_binned_2d, list_bedges = binned_statistic_dd(
                sample=list_var, values=values, statistic=statistic, bins=list_var_bins, range=list_ranges
            )[0:2]
            df_stats_nd[statistics_name[j]] = stats_binned_2d.flatten()
        list_ii = []
        # Loop through the bin edges and create IntervalIndexes from them (to get both
        for bedges in list_bedges:
            list_ii.append(pd.IntervalIndex.from_breaks(bedges, closed="left"))

        # Create nd indexes in nd-array and flatten for each variable
        iind = np.meshgrid(*list_ii)
        for i, var_name in enumerate(list_var_names):
            df_stats_nd[var_name] = iind[i].flatten()

        # Report number of dimensions used
        df_stats_nd.insert(0, "nd", len(list_var_names))

    # Concatenate everything
    list_all_dfs = list_df_1d + list_df_2d + [df_stats_nd]
    df_concat = pd.concat(list_all_dfs)
    # commenting for now: pd.MultiIndex can be hard to use
    # df_concat = df_concat.set_index(list_var_names)

    return df_concat


# Function to convert IntervalIndex written to str in csv back to pd.Interval
# from: https://github.com/pandas-dev/pandas/issues/28210
def _pandas_str_to_interval(istr: str) -> float | pd.Interval:
    if isinstance(istr, float):
        return np.nan
    else:
        c_left = istr[0] == "["
        c_right = istr[-1] == "]"
        closed = {(True, False): "left", (False, True): "right", (True, True): "both", (False, False): "neither"}[
            c_left, c_right
        ]
        left, right = map(float, istr[1:-1].split(","))
        try:
            return pd.Interval(left, right, closed)
        except Exception:
            return np.nan


def interp_nd_binning(
    df: pd.DataFrame,
    list_var_names: str | list[str],
    statistic: str | Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    interpolate_method: Literal["nearest"] | Literal["linear"] = "linear",
    min_count: int | None = 100,
) -> Callable[[tuple[ArrayLike, ...]], NDArrayf]:
    """
    Estimate an interpolant function for an N-dimensional binning. Preferably based on the output of nd_binning.
    For more details on the input dataframe, and associated list of variable name and statistic, see nd_binning.

    First, interpolates nodata values of the irregular N-D binning grid with scipy.griddata.
    Then, extrapolates nodata values on the N-D binning grid with scipy.griddata with "nearest neighbour"
    (necessary to get a regular grid without NaNs).
    Finally, creates an interpolant function (linear by default) to interpolate between points of the grid using
    scipy.RegularGridInterpolator. Extrapolation is fixed to nearest neighbour by duplicating edge bins along each
    dimension (linear extrapolation of two equal bins = nearest neighbour).

    If the variable pd.DataSeries corresponds to an interval (as the output of nd_binning), uses the middle of the
    interval. Otherwise, uses the variable as such.

    :param df: Dataframe with statistic of binned values according to explanatory variables.
    :param list_var_names: Explanatory variable data series to select from the dataframe.
    :param statistic: Statistic to interpolate, stored as a data series in the dataframe.
    :param interpolate_method: Method to interpolate inside of edge bins, "nearest", "linear" (default).
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata).

    :return: N-dimensional interpolant function.

    :examples
    # Using a dataframe created from scratch
    >>> df = pd.DataFrame({"var1": [1, 2, 3, 1, 2, 3, 1, 2, 3], "var2": [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ... "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9]})

    # In 2 dimensions, the statistic array looks like this
    # array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    #     ])
    >>> fun = interp_nd_binning(df, list_var_names=["var1", "var2"], statistic="statistic", min_count=None)

    # Right on point.
    >>> fun((2, 2))
    array(5.)

    # Interpolated linearly inside the 2D frame.
    >>> fun((1.5, 1.5))
    array(3.)

    # Extrapolated linearly outside the 2D frame: nearest neighbour.
    >>> fun((-1, 1))
    array(1.)
    """
    # If list of variable input is simply a string
    if isinstance(list_var_names, str):
        list_var_names = [list_var_names]

    # Check that the dataframe contains what we need
    for var in list_var_names:
        if var not in df.columns:
            raise ValueError('Variable "' + var + '" does not exist in the provided dataframe.')
    statistic_name = statistic if isinstance(statistic, str) else statistic.__name__
    if statistic_name not in df.columns:
        raise ValueError('Statistic "' + statistic_name + '" does not exist in the provided dataframe.')
    if min_count is not None and "count" not in df.columns:
        raise ValueError('Statistic "count" is not in the provided dataframe, necessary to use the min_count argument.')
    if df.empty:
        raise ValueError("Dataframe is empty.")

    df_sub = df.copy()

    # If the dataframe is an output of nd_binning, keep only the dimension of interest
    if "nd" in df_sub.columns:
        df_sub = df_sub[df_sub.nd == len(list_var_names)]

    # Compute the middle values instead of bin interval if the variable is a pandas interval type
    for var in list_var_names:

        # Check if all value are numeric (NaN counts as integer), if yes leave as is
        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in df_sub[var].values):
            pass
        # Check if any value is a pandas interval (NaN do not count, so using any), if yes compute the middle values
        elif any(isinstance(x, pd.Interval) for x in df_sub[var].values):
            df_sub[var] = pd.IntervalIndex(df_sub[var]).mid.values
        # Check for any unformatted interval (saving and reading a pd.DataFrame without MultiIndexing transforms
        # pd.Interval into strings)
        elif any(isinstance(_pandas_str_to_interval(x), pd.Interval) for x in df_sub[var].values):
            intervalindex_vals = [_pandas_str_to_interval(x) for x in df_sub[var].values]
            df_sub[var] = pd.IntervalIndex(intervalindex_vals).mid.values
        else:
            raise ValueError("The variable columns must be provided as numerical mid values, or pd.Interval values.")

    # Check that explanatory variables have valid binning values which coincide along the dataframe
    df_sub = df_sub[np.logical_and.reduce([np.isfinite(df_sub[var].values) for var in list_var_names])]
    if df_sub.empty:
        raise ValueError(
            "Dataframe does not contain a nd binning with the variables corresponding to the list of variables."
        )
    # Check that the statistic data series contain valid data
    if all(~np.isfinite(df_sub[statistic_name].values)):
        raise ValueError("Dataframe does not contain any valid statistic values.")

    # Remove statistic values calculated with a sample count under the minimum count
    if min_count is not None:
        df_sub.loc[df_sub["count"] < min_count, statistic_name] = np.nan

    values = df_sub[statistic_name].values
    ind_valid = np.isfinite(values)

    # Re-check that the statistic data series contain valid data after filtering with min_count
    if all(~ind_valid):
        raise ValueError(
            "Dataframe does not contain any valid statistic values after filtering with min_count = "
            + str(min_count)
            + "."
        )

    # Get a list of middle values for the binning coordinates, to define a nd grid
    list_bmid = []
    shape = []
    for var in list_var_names:
        bmid = sorted(np.unique(df_sub[var][ind_valid]))
        list_bmid.append(bmid)
        shape.append(len(bmid))

    # The workflow below is a bit complicated because of the irregular grid and handling of nodata values!
    # Steps 1/ and 2/ fill the nodata values in the irregular grid, and step 3/ creates the interpolant object to
    # get a value at any point inside or outside the bin edges

    # 1/ Use griddata first to perform interpolation for nodata values within the N-D convex hull of the irregular grid

    # Valid values
    values = values[ind_valid]
    # Coordinates of valid values
    points_valid = tuple(df_sub[var].values[ind_valid] for var in list_var_names)
    # Coordinates of all grid points (convex hull points will be detected automatically and interpolated)
    bmid_grid = np.meshgrid(*list_bmid, indexing="ij")
    points_grid = tuple(bmid_grid[i].flatten() for i in range(len(list_var_names)))
    # Interpolate on grid within convex hull with interpolation method
    values_grid = griddata(points_valid, values, points_grid, method=interpolate_method)

    # 2/ Use griddata to extrapolate nodata values with nearest neighbour on the N-D grid and remove all NaNs

    # Valid values after above interpolation in convex hull
    ind_valid_interp = np.isfinite(values_grid)
    values_interp = values_grid[ind_valid_interp]
    # Coordinate of valid values
    points_valid_interp = tuple(points_grid[i][ind_valid_interp] for i in range(len(points_grid)))

    # First extrapolate once with nearest neighbour on the original grid,
    # this ensures that when the grid is extended (below) the nearest neighbour
    # extrapolation will work as expected (else, there would be problems with
    # extrapolation happening in a diagonal direction (along more than one grid dimension,
    # as that could be the nearest bin in some cases), resulting in a final extrapolation
    # that would not actually use nearest neighbour (when using interpolate_method = "linear"):
    # sometimes producing unphysical negative uncertainties.
    values_grid_nearest1 = griddata(points_valid_interp, values_interp, points_grid, method="nearest")

    # Extend grid by a value of "1" the point coordinates in all directions to ensure
    # that 3/ will extrapolate linearly as for "nearest"
    list_bmid_extended = []
    for i in range(len(list_bmid)):
        bmid_bin = list_bmid[i]
        # Add bin before first edge and decrease coord value
        bmid_bin_extlow = np.insert(bmid_bin, 0, bmid_bin[0] - 1)
        # Add bin after last edge and increase coord value
        bmid_bin_extboth = np.append(bmid_bin_extlow, bmid_bin[-1] + 1)
        list_bmid_extended.append(bmid_bin_extboth)
    bmid_grid_extended = np.meshgrid(*list_bmid_extended, indexing="ij")
    points_grid_extended = tuple(bmid_grid_extended[i].flatten() for i in range(len(list_var_names)))
    # Update shape
    shape_extended = tuple(x + 2 for x in shape)

    # Extrapolate on extended grid with nearest neighbour
    values_grid_nearest2 = griddata(points_grid, values_grid_nearest1, points_grid_extended, method="nearest")
    values_grid_nearest2 = values_grid_nearest2.reshape(shape_extended)

    # 3/ Use RegularGridInterpolator to perform interpolation **between points** of the grid, with extrapolation forced
    # to nearest neighbour by duplicating edge points
    # (does not support NaNs, hence the need for 2/ above)
    interp_fun = RegularGridInterpolator(
        tuple(list_bmid_extended), values_grid_nearest2, method="linear", bounds_error=False, fill_value=None
    )

    return interp_fun  # type: ignore


def get_perbin_nd_binning(
    df: pd.DataFrame,
    list_var: list[NDArrayf],
    list_var_names: str | list[str],
    statistic: str | Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
    min_count: int | None = 0,
) -> NDArrayf:
    """
    Get per-bin array statistic for a list of array input variables, based on the results of an independent N-D binning.

    For example, get a 2D array of elevation uncertainty based on 2D arrays of slope and curvature and a related binning
    (for uncertainty analysis) or get a 2D array of elevation bias based on 2D arrays of rotated X coordinates (for
    an along-track bias correction).

    :param list_var: List of size (L) of explanatory variables array of size (N,).
    :param list_var_names: List of size (L) of names of the explanatory variables.
    :param df: Dataframe with statistic of binned values according to explanatory variables.
    :param statistic: Statistic to use, stored as a data series in the dataframe.
    :param min_count: Minimum number of samples to be used as a valid statistic (otherwise not applying operation).

    :return: The array of statistic values corresponding to the input variables.
    """

    # Prepare output
    values_out = np.zeros(np.shape(list_var[0])) * np.nan

    # If list of variable input is simply a string
    if isinstance(list_var_names, str):
        list_var_names = [list_var_names]

    if len(list_var) != len(list_var_names):
        raise ValueError("The lists of variables and variable names should be the same length.")

    # Check that the dataframe contains what we need
    for var in list_var_names:
        if var not in df.columns:
            raise ValueError('Variable "' + var + '" does not exist in the provided dataframe.')
    statistic_name = statistic if isinstance(statistic, str) else statistic.__name__
    if statistic_name not in df.columns:
        raise ValueError('Statistic "' + statistic_name + '" does not exist in the provided dataframe.')
    if min_count is not None and "count" not in df.columns:
        raise ValueError('Statistic "count" is not in the provided dataframe, necessary to use the min_count argument.')
    if df.empty:
        raise ValueError("Dataframe is empty.")

    df_sub = df.copy()

    # If the dataframe is an output of nd_binning, keep only the dimension of interest
    if "nd" in df_sub.columns:
        df_sub = df_sub[df_sub.nd == len(list_var_names)]

    # Check for any unformatted interval (saving and reading a pd.DataFrame without MultiIndexing transforms
    # pd.Interval into strings)
    for var_name in list_var_names:
        if any(isinstance(x, pd.Interval) for x in df_sub[var_name].values):
            continue
        elif any(isinstance(_pandas_str_to_interval(x), pd.Interval) for x in df_sub[var_name]):
            df_sub[var_name] = [_pandas_str_to_interval(x) for x in df_sub[var_name]]
        else:
            ValueError("The bin intervals of the dataframe should be pandas.Interval.")

    # Apply operator in the nd binning
    # We compute the masks linked to each 1D bin in a single for loop, to optimize speed
    L = len(list_var)
    all_mask_vars = []
    all_interval_vars = []
    for k in range(L):
        # Get variable name and list of intervals in the dataframe
        var_name = list_var_names[k]
        list_interval_var = np.unique(df_sub[var_name].values)

        # Get a list of mask for every bin of the variable
        list_mask_var = [
            np.logical_and(list_var[k] >= list_interval_var[j].left, list_var[k] < list_interval_var[j].right)
            for j in range(len(list_interval_var))
        ]

        # Save those in lists to later combine them
        all_mask_vars.append(list_mask_var)
        all_interval_vars.append(list_interval_var)

    # We perform the K-D binning by logically combining the masks
    all_ranges = [range(len(all_interval_vars[k])) for k in range(L)]
    for indices in itertools.product(*all_ranges):

        # Get mask of the specific bin, skip if empty
        mask_bin = np.logical_and.reduce([all_mask_vars[k][indices[k]] for k in range(L)])
        if np.count_nonzero(mask_bin) == 0:
            continue

        # Get the statistic
        index_bin = np.logical_and.reduce(
            [df_sub[list_var_names[k]] == all_interval_vars[k][indices[k]] for k in range(L)]
        )
        statistic_bin = df_sub[statistic_name][index_bin].values[0]

        # Get count value of the statistic and use it if above the threshold
        count_bin = df_sub["count"][index_bin].values[0]
        if count_bin > min_count:
            # Write out to the output array
            values_out[mask_bin] = statistic_bin

    return values_out


def two_step_standardization(
    dvalues: NDArrayf,
    list_var: list[NDArrayf],
    unscaled_error_fun: Callable[[tuple[ArrayLike, ...]], NDArrayf],
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    fac_spread_outliers: float | None = 7,
) -> tuple[NDArrayf, Callable[[tuple[ArrayLike, ...]], NDArrayf]]:
    """
    Standardize the proxy differenced values using the modelled heteroscedasticity, re-scaled to the spread statistic,
    and generate the final standardization function.

    :param dvalues: Proxy values as array of size (N,) (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param unscaled_error_fun: Function of the spread with explanatory variables not yet re-scaled
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Standardized values array of size (N,), Function to destandardize
    """

    # Standardize a first time with the function
    zscores = dvalues / unscaled_error_fun(tuple(list_var))

    # Set large outliers that might have been created by the standardization to NaN, central tendency should already be
    # around zero so only need to take the absolute value
    if fac_spread_outliers is not None:
        zscores[np.abs(zscores) > fac_spread_outliers * spread_statistic(zscores)] = np.nan

    # Re-compute the spread statistic to re-standardize, as dividing by the function will not necessarily bring the
    # z-score exactly equal to one due to approximations of N-D binning, interpolating and due to the outlier filtering
    zscore_nmad = spread_statistic(zscores)

    # Re-standardize
    zscores /= zscore_nmad

    # Define the exact function for de-standardization to pass as output
    def error_fun(*args: tuple[ArrayLike, ...]) -> NDArrayf:
        return zscore_nmad * unscaled_error_fun(*args)

    return zscores, error_fun


def _estimate_model_heteroscedasticity(
    dvalues: NDArrayf,
    list_var: list[NDArrayf],
    list_var_names: list[str],
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf] | None = None,
    min_count: int | None = 100,
    fac_spread_outliers: float | None = 7,
) -> tuple[pd.DataFrame, Callable[[tuple[NDArrayf, ...]], NDArrayf]]:
    """
    Estimate and model the heteroscedasticity (i.e., variability in error) according to a list of explanatory variables
    from a proxy of differenced values (e.g., elevation differences), if possible compared to a source of higher
    precision.

    This function performs N-D data binning with the list of explanatory variable for a spread statistic, then
    performs N-D interpolation on this statistic, scales the output with a two-step standardization to return an error
    function of the explanatory variables.

    The functions used are `nd_binning`, `interp_nd_binning` and `two_step_standardization`.

    :param dvalues: Proxy values as array of size (N,) (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param list_var_names: List of size (L) of names of the explanatory variables
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory
        variables; defaults to 10 bins
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata)
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Dataframe of binned spread statistic with explanatory variables, Error function with explanatory variables
    """

    # Perform N-D binning with the differenced values computing the spread statistic
    df = nd_binning(
        values=dvalues,
        list_var=list_var,
        list_var_names=list_var_names,
        statistics=[spread_statistic],
        list_var_bins=list_var_bins,
    )

    # Perform N-D linear interpolation for the spread statistic
    fun = interp_nd_binning(df, list_var_names=list_var_names, statistic=spread_statistic.__name__, min_count=min_count)

    # Get the final function based on a two-step standardization
    final_fun = two_step_standardization(
        dvalues=dvalues,
        list_var=list_var,
        unscaled_error_fun=fun,
        spread_statistic=spread_statistic,
        fac_spread_outliers=fac_spread_outliers,
    )[1]

    return df, final_fun


@overload
def _preprocess_values_with_mask_to_array(  # type: ignore
    values: list[NDArrayf | RasterType],
    include_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    exclude_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    gsd: float | None = None,
    preserve_shape: bool = True,
) -> tuple[list[NDArrayf], float]: ...


@overload
def _preprocess_values_with_mask_to_array(
    values: NDArrayf | RasterType,
    include_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    exclude_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    gsd: float | None = None,
    preserve_shape: bool = True,
) -> tuple[NDArrayf, float]: ...


def _preprocess_values_with_mask_to_array(
    values: list[NDArrayf | RasterType] | NDArrayf | RasterType,
    include_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    exclude_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    gsd: float | None = None,
    preserve_shape: bool = True,
) -> tuple[list[NDArrayf] | NDArrayf, float]:
    """
    Preprocess input values provided as Raster or ndarray with a stable and/or unstable mask provided as Vector or
    ndarray into an array of stable values.

    By default, the shape is preserved and the masked values converted to NaNs.

    :param values: Values or list of values as a Raster, array or a list of Raster/arrays
    :param include_mask: Vector shapefile of mask to include (if values is Raster), or boolean array of same shape as
        values
    :param exclude_mask: Vector shapefile of mask to exclude (if values is Raster), or boolean array of same shape
        as values
    :param gsd: Ground sampling distance, if all the input values are provided as array
    :param preserve_shape: If True, masks unstable values with NaN. If False, returns a 1D array of stable values.

    :return: Array of stable terrain values, Ground sampling distance
    """

    # Check inputs: needs to be Raster, array or a list of those
    if not isinstance(values, (Raster, np.ndarray, list)) or (
        isinstance(values, list) and not all(isinstance(val, (Raster, np.ndarray)) for val in values)
    ):
        raise ValueError("The values must be a Raster or NumPy array, or a list of those.")
    # Masks need to be an array, Vector or GeoPandas dataframe
    if include_mask is not None and not isinstance(include_mask, (np.ndarray, Vector, Mask, gpd.GeoDataFrame)):
        raise ValueError("The stable mask must be a Vector, Mask, GeoDataFrame or NumPy array.")
    if exclude_mask is not None and not isinstance(exclude_mask, (np.ndarray, Vector, Mask, gpd.GeoDataFrame)):
        raise ValueError("The unstable mask must be a Vector, Mask, GeoDataFrame or NumPy array.")

    # Check that input stable mask can only be a georeferenced vector if the proxy values are a Raster to project onto
    if isinstance(values, list):
        any_raster = any(isinstance(val, Raster) for val in values)
    else:
        any_raster = isinstance(values, Raster)
    if not any_raster and isinstance(include_mask, (Vector, gpd.GeoDataFrame)):
        raise ValueError(
            "The stable mask can only passed as a Vector or GeoDataFrame if the input values contain a Raster."
        )

    # If there is only one array or Raster, put alone in a list
    if not isinstance(values, list):
        return_unlist = True
        values = [values]
    else:
        return_unlist = False

    # Get the arrays
    values_arr = [get_array_and_mask(val)[0] if isinstance(val, Raster) else val for val in values]

    # Get the ground sampling distance from the first Raster if there is one
    if gsd is None and any_raster:
        for i in range(len(values)):
            if isinstance(values[i], Raster):
                first_raster = values[i]
                break
        # Looks like mypy cannot trace the isinstance here... ignoring
        gsd = first_raster.res[0]  # type: ignore
    elif gsd is not None:
        gsd = gsd
    else:
        raise ValueError("The ground sampling distance must be provided if no Raster object is passed.")

    # If the stable mask is not an array, create it
    if include_mask is None:
        include_mask_arr = np.ones(np.shape(values_arr[0]), dtype=bool)
    elif isinstance(include_mask, (Vector, gpd.GeoDataFrame)):

        # If the stable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(include_mask, gpd.GeoDataFrame):
            stable_vector = Vector(include_mask)
        else:
            stable_vector = include_mask

        # Create the mask
        include_mask_arr = stable_vector.create_mask(first_raster, as_array=True)
    # If the mask is a Mask
    elif isinstance(include_mask, Mask):
        include_mask_arr = include_mask.data.filled(False)
    # If the mask is already an array, just pass it
    else:
        include_mask_arr = include_mask

    # If the unstable mask is not an array, create it
    if exclude_mask is None:
        exclude_mask_arr = np.zeros(np.shape(values_arr[0]), dtype=bool)
    elif isinstance(exclude_mask, (Vector, gpd.GeoDataFrame)):

        # If the unstable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(exclude_mask, gpd.GeoDataFrame):
            unstable_vector = Vector(exclude_mask)
        else:
            unstable_vector = exclude_mask

        # Create the mask
        exclude_mask_arr = unstable_vector.create_mask(first_raster, as_array=True)
    # If the mask is already an array, just pass it
    # If the mask is a Mask
    elif isinstance(exclude_mask, Mask):
        exclude_mask_arr = exclude_mask.data.filled(False)
    else:
        exclude_mask_arr = exclude_mask

    include_mask_arr = np.logical_and(include_mask_arr, ~exclude_mask_arr).squeeze()

    if preserve_shape:
        # Need to preserve the shape, so setting as NaNs all points not on stable terrain
        values_stable_arr = []
        for val in values_arr:
            val_stable = val.copy()
            val_stable[~include_mask_arr] = np.nan
            values_stable_arr.append(val_stable)
    else:
        values_stable_arr = [val_arr[include_mask_arr] for val_arr in values_arr]

    # If input was a list, give a list. If it was a single array, give a single array.
    if return_unlist:
        values_stable_arr = values_stable_arr[0]  # type: ignore

    return values_stable_arr, gsd


@overload
def infer_heteroscedasticity_from_stable(
    dvalues: NDArrayf,
    list_var: list[NDArrayf | RasterType],
    stable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    list_var_names: list[str] = None,
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf] | None = None,
    min_count: int | None = 100,
    fac_spread_outliers: float | None = 7,
) -> tuple[NDArrayf, pd.DataFrame, Callable[[tuple[NDArrayf, ...]], NDArrayf]]: ...


@overload
def infer_heteroscedasticity_from_stable(
    dvalues: RasterType,
    list_var: list[NDArrayf | RasterType],
    stable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    list_var_names: list[str] = None,
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf] | None = None,
    min_count: int | None = 100,
    fac_spread_outliers: float | None = 7,
) -> tuple[RasterType, pd.DataFrame, Callable[[tuple[NDArrayf, ...]], NDArrayf]]: ...


def infer_heteroscedasticity_from_stable(
    dvalues: NDArrayf | RasterType,
    list_var: list[NDArrayf | RasterType],
    stable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    list_var_names: list[str] = None,
    spread_statistic: Callable[[NDArrayf], np.floating[Any]] = gu.stats.nmad,
    list_var_bins: int | tuple[int, ...] | tuple[NDArrayf] | None = None,
    min_count: int | None = 100,
    fac_spread_outliers: float | None = 7,
) -> tuple[NDArrayf | RasterType, pd.DataFrame, Callable[[tuple[NDArrayf, ...]], NDArrayf]]:
    """
    Infer heteroscedasticity from differenced values on stable terrain and a list of explanatory variables.

    This function returns an error map, a dataframe of spread values and the error function with explanatory variables.
    It is a convenience wrapper for `estimate_model_heteroscedasticity` to work on either Raster or array, compute the
    stable mask and return an error map.

    If no stable or unstable mask is provided to mask in or out the values, all terrain is used.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables as array or Raster of same shape as dvalues
    :param stable_mask: Vector shapefile of stable terrain (if dvalues is Raster), or boolean array of same shape as
        dvalues
    :param unstable_mask: Vector shapefile of unstable terrain (if dvalues is Raster), or boolean array of same shape
        as dvalues
    :param list_var_names: List of size (L) of names of the explanatory variables, otherwise named var1, var2, etc.
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory
        variables; defaults to 10 bins
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata)
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Inferred error map (array or Raster, same as input proxy values),
        Dataframe of binned spread statistic with explanatory variables,
        Error function with explanatory variables
    """

    # Create placeholder variables names if those don't exist
    if list_var_names is None:
        list_var_names = ["var" + str(i + 1) for i in range(len(list_var))]

    # Get the arrays for proxy values and explanatory variables
    list_all_arr, gsd = _preprocess_values_with_mask_to_array(
        values=[dvalues] + list_var, include_mask=stable_mask, exclude_mask=unstable_mask, preserve_shape=False
    )
    dvalues_stable_arr = list_all_arr[0]
    list_var_stable_arr = list_all_arr[1:]

    # Estimate and model the heteroscedasticity using only stable terrain
    df, fun = _estimate_model_heteroscedasticity(
        dvalues=dvalues_stable_arr,
        list_var=list_var_stable_arr,
        list_var_names=list_var_names,
        spread_statistic=spread_statistic,
        list_var_bins=list_var_bins,
        min_count=min_count,
        fac_spread_outliers=fac_spread_outliers,
    )

    # Use the standardization function to get the error array for the entire input array (not only stable)
    list_var_arr = [get_array_and_mask(var)[0] if isinstance(var, Raster) else var for var in list_var]
    error = fun(tuple(list_var_arr))

    # Return the right type, depending on dvalues input
    if isinstance(dvalues, Raster):
        return dvalues.copy(new_array=error), df, fun
    else:
        return error, df, fun


def _create_circular_mask(
    shape: tuple[int, int], center: tuple[int, int] | None = None, radius: float | None = None
) -> NDArrayb:
    """
    Create circular mask on a raster, defaults to the center of the array and its half width

    :param shape: shape of array
    :param center: center
    :param radius: radius
    :return:
    """

    w, h = shape

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    # Manual solution
    Y, X = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center < radius

    return mask


def _create_ring_mask(
    shape: tuple[int, int],
    center: tuple[int, int] | None = None,
    in_radius: float = 0,
    out_radius: float | None = None,
) -> NDArrayb:
    """
    Create ring mask on a raster, defaults to the center of the array and a circle mask of half width of the array

    :param shape: shape of array
    :param center: center
    :param in_radius: inside radius
    :param out_radius: outside radius
    :return:
    """

    w, h = shape

    if center is None:
        center = (int(w / 2), int(h / 2))
    if out_radius is None:
        out_radius = min(center[0], center[1], w - center[0], h - center[1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in *divide")
        mask_inside = _create_circular_mask((w, h), center=center, radius=in_radius)
        mask_outside = _create_circular_mask((w, h), center=center, radius=out_radius)

    mask_ring = np.logical_and(~mask_inside, mask_outside)

    return mask_ring


def _subsample_wrapper(
    values: NDArrayf,
    coords: NDArrayf,
    shape: tuple[int, int],
    subsample: int = 10000,
    subsample_method: str = "pdist_ring",
    inside_radius: float = 0,
    outside_radius: float = None,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArrayf, NDArrayf]:
    """
    (Not used by default)
    Wrapper for subsampling pdist methods
    """
    nx, ny = shape

    rng = np.random.default_rng(random_state)

    # Subsample spatially for disk/ring methods
    if subsample_method in ["pdist_disk", "pdist_ring"]:
        # Select random center coordinates
        center_x = rng.choice(nx, 1)[0]
        center_y = rng.choice(ny, 1)[0]
        if subsample_method == "pdist_ring":
            subindex = _create_ring_mask(
                (nx, ny), center=(center_x, center_y), in_radius=inside_radius, out_radius=outside_radius
            )
        else:
            subindex = _create_circular_mask((nx, ny), center=(center_x, center_y), radius=outside_radius)

        index = subindex.flatten()
        values_sp = values[index]
        coords_sp = coords[index, :]

    else:
        values_sp = values
        coords_sp = coords

    index = subsample_array(values_sp, subsample=subsample, return_indices=True, random_state=random_state)
    values_sub = values_sp[index[0]]
    coords_sub = coords_sp[index[0], :]

    return values_sub, coords_sub


def _aggregate_pdist_empirical_variogram(
    values: NDArrayf,
    coords: NDArrayf,
    subsample: int,
    shape: tuple[int, int],
    subsample_method: str,
    gsd: float,
    pdist_multi_ranges: list[float] | None = None,
    # **kwargs: **EmpiricalVariogramKArgs, # This will work in Python 3.12, fails in the meantime, some ignore will be
    # removable then in this function
    **kwargs: Any,
) -> pd.DataFrame:
    """
    (Not used by default)
    Aggregating subfunction of sample_empirical_variogram for pdist methods.
    The pairwise differences are calculated within each subsample.
    """

    # If no multi_ranges are provided, define a logical default behaviour with the pixel size and grid size
    if subsample_method in ["pdist_disk", "pdist_ring"]:

        if pdist_multi_ranges is None:

            # Define list of ranges as exponent 2 of the resolution until the maximum range
            pdist_multi_ranges = []
            # We start at 10 times the ground sampling distance
            new_range = gsd * 10
            while new_range < kwargs.get("maxlag") / 2:  # type: ignore
                pdist_multi_ranges.append(new_range)
                new_range *= 2
            pdist_multi_ranges.append(kwargs.get("maxlag"))  # type: ignore

        # Define subsampling parameters
        list_inside_radius = []
        list_outside_radius: list[float | None] = []
        binned_ranges = [0.0] + pdist_multi_ranges
        for i in range(len(binned_ranges) - 1):

            # Radiuses need to be passed as pixel sizes, dividing by ground sampling distance
            outside_radius = binned_ranges[i + 1] / gsd
            if subsample_method == "pdist_ring":
                inside_radius = binned_ranges[i] / gsd
            else:
                inside_radius = 0.0

            list_outside_radius.append(outside_radius)
            list_inside_radius.append(inside_radius)
    else:
        # For random point selection, no need for multi-range parameters
        pdist_multi_ranges = [kwargs.get("maxlag")]  # type: ignore
        list_outside_radius = [None]
        list_inside_radius = [0.0]

    # Estimate variogram with specific subsampling at multiple ranges
    list_df_range = []
    for j in range(len(pdist_multi_ranges)):

        values_sub, coords_sub = _subsample_wrapper(
            values,
            coords,
            shape=shape,
            subsample=subsample,
            subsample_method=subsample_method,
            inside_radius=list_inside_radius[j],
            outside_radius=list_outside_radius[j],
            random_state=kwargs.get("random_state"),
        )
        if len(values_sub) == 0:
            continue
        df_range = _get_pdist_empirical_variogram(values=values_sub, coords=coords_sub, **kwargs)

        # Aggregate runs
        list_df_range.append(df_range)

    df = pd.concat(list_df_range)

    return df


def _get_pdist_empirical_variogram(values: NDArrayf, coords: NDArrayf, **kwargs: Any) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object calculating pairwise distances within the sample

    :param values: Values
    :param coords: Coordinates
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """

    # Remove random_state keyword argument that is not used
    kwargs.pop("random_state")

    # Get arguments of Variogram class init function
    variogram_args = skg.Variogram.__init__.__code__.co_varnames[: skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in variogram_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn("Keyword arguments: " + ",".join(list(remaining_kwargs.keys())) + " were not used.")
    # Filter corresponding arguments before passing
    filtered_kwargs = {k: kwargs[k] for k in variogram_args if k in kwargs}

    # Derive variogram with default MetricSpace (equivalent to scipy.pdist)
    V = skg.Variogram(coordinates=coords, values=values, normalize=False, fit_method=None, **filtered_kwargs)

    # Get bins, empirical variogram values, and bin count
    bins, exp = V.get_empirical()
    count = V.bin_count

    # Write to dataframe
    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df


def _choose_cdist_equidistant_sampling_parameters(**kwargs: Any) -> tuple[int, int, float]:
    """
    Add a little calculation to partition the "subsample" argument automatically into the "run" and "samples"
    arguments of RasterEquidistantMetricSpace, to have a similar number of points than with a classic pdist method.

    We compute the arguments to match a N0**2/2 number of pairwise comparison, N0 being the "subsample" input, and
    forcing the number of rings to 10 by default. This corresponds to 10 independent rings with equal number of samples
    compared pairwise against a central disk. We force this number of sample to be at least 2 (skgstat raises an error
    if there is only one). Additionally, if samples permit, we compute 10 independent runs, maximum 100 to limit
    processing times when aggregating different runs in sparse matrixes. If even more sample permit (default case), we
    increase the number of subsamples in rings and runs simultaneously.

    The number of pairwise samples for a classic pdist is N0(N0-1)/2 with N0 the number of samples of the ensemble. For
    the cdist equidistant calculation it is M*N*R where N are the subsamples in the center disk, M is the number of
    samples in the rings which amounts to X*N where X is the number of rings in the grid extent, as each ring draws N
    samples. And R is the number of runs with a different random center point.
    X is fixed by the extent and ratio_subsample parameters, and so N0**2/2 = R*X*N**2, and we want at least 10 rings
    and, if possible, 10 runs.

    !! Different variables: !! The "samples" of RasterEquidistantMetricSpace is N, while the "subsample" passed is N0.
    """

    # First, we extract the extent, shape and subsample values from the keyword arguments
    extent = kwargs["extent"]
    shape = kwargs["shape"]
    subsample = kwargs["subsample"]

    # We define the number of rings to 10 in order to get a decent equidistant sampling, we'll later adjust the
    # ratio_sampling to force that number to 10
    if "nb_rings" in kwargs.keys():
        nb_rings = kwargs["nb_rings"]
    else:
        nb_rings = 10
    # For one run (R=1), and two samples per disk/ring (N=2), and the number of rings X=10, this requires N0 to be at
    # least 10:
    min_subsample = np.ceil(np.sqrt(2 * nb_rings * 2**2) + 1)
    if subsample < min_subsample:
        raise ValueError(f"The number of subsamples needs to be at least {min_subsample:.0f}.")

    # The pairwise comparisons can be deduced from the number of rings: R * N**2 = N0**2/(2*X)
    pairwise_comp_per_disk = np.ceil(subsample**2 / (2 * nb_rings))

    # With R*N**2 = N0**2/2, and minimum 2 samples N, we compute the number of runs R forcing at least
    # 10 runs and maximum 100
    if pairwise_comp_per_disk < 10:
        runs = int(pairwise_comp_per_disk / 2**2)
    else:
        runs = int(min(100, 10 * np.ceil((pairwise_comp_per_disk / (2**2 * 10)) ** (1 / 3))))

    # Now we can derive the number of samples N, which will always be at least 2
    subsample_per_disk_per_run = int(np.ceil(np.sqrt(pairwise_comp_per_disk / runs)))

    # Finally, we need to force the ratio_subsample to get exactly 10 rings

    # We first derive the maximum distance and resolution the same way as in skgstat.RasterEquidistantMetricSpace
    maxdist = np.sqrt((extent[1] - extent[0]) ** 2 + (extent[3] - extent[2]) ** 2)
    res = np.mean([(extent[1] - extent[0]) / (shape[0] - 1), (extent[3] - extent[2]) / (shape[1] - 1)])

    # Then, we derive the subsample ratio. We have:
    # 1) radius * sqrt(2)**X = maxdist, and
    # 2) pi * radius**2 = res**2 * N / sub_ratio
    # From which we can deduce: sub_ratio = res**2 * N / (pi * maxdist**2 / sqrt(2)**(2X) )
    ratio_subsample = res**2 * subsample_per_disk_per_run / (np.pi * maxdist**2 / np.sqrt(2) ** (2 * nb_rings))

    # And the number of total pairwise comparison
    total_pairwise_comparison = runs * subsample_per_disk_per_run**2 * nb_rings

    logging.info(
        "Equidistant circular sampling will be performed for %d runs (random center points) with pairwise "
        "comparison between %d samples (points) of the central disk and again %d samples times %d independent "
        "rings centered on the same center point. This results in approximately %d pairwise comparisons (duplicate "
        "pairwise points randomly selected will be removed).",
        runs,
        subsample_per_disk_per_run,
        subsample_per_disk_per_run,
        nb_rings,
        total_pairwise_comparison,
    )

    return runs, subsample_per_disk_per_run, ratio_subsample


def _get_cdist_empirical_variogram(
    values: NDArrayf, coords: NDArrayf, subsample_method: str, **kwargs: Any
) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object calculating pairwise distances between two sample collections
    of a MetricSpace (see scikit-gstat documentation for more details)

    :param values: Values
    :param coords: Coordinates
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """

    if subsample_method == "cdist_equidistant":

        if "runs" not in kwargs.keys() and "samples" not in kwargs.keys():
            # We define subparameters for the equidistant technique to match the number of pairwise comparison
            # that would have a classic "subsample" with pdist, except if those parameters are already user-defined
            runs, samples, ratio_subsample = _choose_cdist_equidistant_sampling_parameters(**kwargs)
            kwargs["ratio_subsample"] = ratio_subsample
            kwargs["runs"] = runs
            # The "samples" argument is used by skgstat Metric subclasses (and not "subsample")
            kwargs["samples"] = samples

        kwargs.pop("subsample")

    elif subsample_method == "cdist_point":

        # We set the samples to match the subsample argument if the method is random points
        kwargs["samples"] = kwargs.pop("subsample")

    # Rename the "random_state" argument into "rng", also used by skgstat Metric subclasses
    kwargs["rnd"] = kwargs.pop("random_state")

    # Define MetricSpace function to be used, fetch possible keywords arguments
    if subsample_method == "cdist_point":
        # List keyword arguments of the Probabilistic class init function
        ms_args = skg.ProbabalisticMetricSpace.__init__.__code__.co_varnames[
            : skg.ProbabalisticMetricSpace.__init__.__code__.co_argcount
        ]
        ms = skg.ProbabalisticMetricSpace
    else:
        # List keyword arguments of the RasterEquidistant class init function
        ms_args = skg.RasterEquidistantMetricSpace.__init__.__code__.co_varnames[
            : skg.RasterEquidistantMetricSpace.__init__.__code__.co_argcount
        ]
        ms = skg.RasterEquidistantMetricSpace

    # Get arguments of Variogram class init function
    variogram_args = skg.Variogram.__init__.__code__.co_varnames[: skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed, accounting for MetricSpace arguments
    remaining_kwargs = kwargs.copy()
    for arg in variogram_args + ms_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn("Keyword arguments: " + ", ".join(list(remaining_kwargs.keys())) + " were not used.")

    # Filter corresponding arguments before passing to MetricSpace function
    filtered_ms_kwargs = {k: kwargs[k] for k in ms_args if k in kwargs}
    M = ms(coords=coords, **filtered_ms_kwargs)

    # Filter corresponding arguments before passing to Variogram function
    filtered_var_kwargs = {k: kwargs[k] for k in variogram_args if k in kwargs}
    V = skg.Variogram(M, values=values, normalize=False, fit_method=None, **filtered_var_kwargs)

    # Get bins, empirical variogram values, and bin count
    bins, exp = V.get_empirical(bin_center=False)
    count = V.bin_count

    # Write to dataframe
    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df


def _wrapper_get_empirical_variogram(argdict: dict[str, Any]) -> pd.DataFrame:
    """
    Multiprocessing wrapper for get_pdist_empirical_variogram and get_cdist_empirical variogram

    :param argdict: Keyword argument to pass to get_pdist/cdist_empirical_variogram
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """
    logging.debug("Working on run " + str(argdict["i"]) + " out of " + str(argdict["imax"]))
    argdict.pop("i")
    argdict.pop("imax")

    if argdict["subsample_method"] in ["cdist_equidistant", "cdist_point"]:
        # Simple wrapper for the skgstat Variogram function for cdist methods
        return _get_cdist_empirical_variogram(**argdict)
    else:
        # Aggregating several skgstat Variogram after iterative subsampling of specific points in the Raster
        return _aggregate_pdist_empirical_variogram(**argdict)


class EmpiricalVariogramKArgs(TypedDict, total=False):
    runs: int
    pdist_multi_ranges: list[float]
    ratio_subsample: float
    samples: int
    nb_rings: int
    maxlag: float
    bin_func: Any
    estimator: str


def sample_empirical_variogram(
    values: NDArrayf | RasterType,
    gsd: float = None,
    coords: NDArrayf = None,
    subsample: int = 1000,
    subsample_method: str = "cdist_equidistant",
    n_variograms: int = 1,
    n_jobs: int = 1,
    random_state: int | np.random.Generator | None = None,
    # **kwargs: **EmpiricalVariogramKArgs, # This will work in Python 3.12, fails in the meantime, we'll be able to
    # remove some type ignores from this function in the future
    **kwargs: int | list[float] | float | str | Any,
) -> pd.DataFrame:
    """
    Sample empirical variograms with binning adaptable to multiple ranges and spatial subsampling adapted for raster
    data.
    Returns an empirical variogram (empirical variance, upper bound of spatial lag bin, count of pairwise samples).

    If values are provided as a Raster subclass, nothing else is required.
    If values are provided as a 2D array (M,N), a ground sampling distance is sufficient to derive the pairwise
    distances.
    If values are provided as a 1D array (N), an array of coordinates (N,2) or (2,N) is expected. If the coordinates
    do not correspond to points of a grid, a ground sampling distance is needed to correctly get the grid size.

    By default, the subsampling is based on RasterEquidistantMetricSpace implemented in scikit-gstat. This method
    samples more effectively large grid data by isolating pairs of spatially equidistant ensembles for distributed
    pairwise comparison. In practice, two subsamples are drawn for pairwise comparison: one from a disk of certain
    radius within the grid, and another one from rings of larger radii that increase steadily between the pixel size
    and the extent of the raster. Those disks and rings are sampled several times across the grid using random centers.
    See more details in Hugonnet et al. (2022), https://doi.org/10.1109/jstars.2022.3188922, in particular on
    Supplementary Fig. 13. for the subsampling scheme.

    The "subsample" argument determines the number of samples for each method to yield a number of pairwise comparisons
    close to that of a pdist calculation, that is N*(N-1)/2 where N is the subsample argument.
    For the cdist equidistant method, the "runs" (random centers) and "samples" (subsample of a disk/ring) are set
    automatically to get close to N*(N-1)/2 pairwise samples, fixing a number of rings "nb_rings" to 10. Those can be
    more finely adjusted by passing the argument "runs", "samples" and "nb_rings" to kwargs. Further details can be
    found in the description of skgstat.MetricSpace.RasterEquidistantMetricSpace or
    _choose_cdist_equidistant_sampling_parameters.

    Spatial subsampling method argument subsample_method can be one of "cdist_equidistant", "cdist_point",
    "pdist_point", "pdist_disk" and "pdist_ring".
    The cdist methods use MetricSpace classes of scikit-gstat and do pairwise comparison between two distinct ensembles
    as in scipy.spatial.cdist. For the cdist methods, the variogram is estimated in a single run from the MetricSpace.

    The pdist methods use methods to subsample the Raster points directly and do pairwise comparison within a single
    ensemble as in scipy.spatial.pdist. For the pdist methods, an iterative process is required: a list of ranges
    subsampled independently is used.

    Variograms are derived independently for several runs and ranges using each pairwise sample, and later aggregated.
    If the subsampling method selected is "random_point", the multi-range argument is ignored as range has no effect on
    this subsampling method.

    For pdist methods, keyword arguments are passed to skgstat.Variogram.
    For cdist methods, keyword arguments are passed to both skgstat.Variogram and skgstat.MetricSpace.

    :param values: Values of studied variable
    :param gsd: Ground sampling distance
    :param coords: Coordinates
    :param subsample: Number of samples to randomly draw from the values
    :param subsample_method: Spatial subsampling method
    :param n_variograms: Number of independent empirical variogram estimations (to estimate empirical variogram spread)
    :param n_jobs: Number of processing cores
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Empirical variogram (variance, upper bound of lag bin, counts)
    """

    if not _has_skgstat:
        raise ValueError("Optional dependency needed. Install 'scikit-gstat'.")

    # First, check all that the values provided are OK
    if isinstance(values, Raster):
        gsd = values.res[0]
        values, mask = get_array_and_mask(values)
    elif isinstance(values, (np.ndarray, np.ma.masked_array)):
        values, mask = get_array_and_mask(values)
    else:
        raise ValueError("Values must be of type NDArrayf, np.ma.masked_array or Raster subclass.")
    values = values.squeeze()

    # Then, check if the logic between values, coords and gsd is respected
    if (gsd is not None or subsample_method in ["cdist_equidistant", "pdist_disk", "pdist_ring"]) and values.ndim == 1:
        raise ValueError(
            'Values array must be 2D when using any of the "cdist_equidistant", "pdist_disk" and '
            '"pdist_ring" methods, or providing a ground sampling distance instead of coordinates.'
        )
    elif coords is not None and values.ndim != 1:
        raise ValueError("Values array must be 1D when providing coordinates.")
    elif coords is not None and (coords.shape[0] != 2 and coords.shape[1] != 2):
        raise ValueError("The coordinates array must have one dimension with length equal to 2")
    elif values.ndim == 2 and gsd is None:
        raise ValueError("The ground sampling distance must be defined when passing a 2D values array.")

    # Check the subsample method provided exists, otherwise list options
    if subsample_method not in ["cdist_equidistant", "cdist_point", "pdist_point", "pdist_disk", "pdist_ring"]:
        raise TypeError(
            'The subsampling method must be one of "cdist_equidistant, "cdist_point", "pdist_point", '
            '"pdist_disk" or "pdist_ring".'
        )
    # Check that, for several runs, the binning function is an Iterable, otherwise skgstat might provide variogram
    # values over slightly different binnings due to randomly changing subsample maximum lags
    if n_variograms > 1 and "bin_func" in kwargs.keys() and not isinstance(kwargs.get("bin_func"), Iterable):
        warnings.warn(
            "Using a named binning function of scikit-gstat might provide different binnings for each "
            "independent run. To remediate that issue, pass bin_func as an Iterable of right bin edges, "
            "(or use default bin_func)."
        )

    # Defaulting to coordinates if those are provided
    if coords is not None:
        nx = None
        ny = None
        # Making the shape of coordinates consistent if they are transposed
        if coords.shape[0] == 2 and coords.shape[1] != 2:
            coords = np.transpose(coords)
    # If no coordinates provided, we use the shape of the array and the provided ground sampling distance to derive
    # relative coordinates (starting at zero)
    else:
        nx, ny = np.shape(values)
        x, y = np.meshgrid(np.arange(0, values.shape[0] * gsd, gsd), np.arange(0, values.shape[1] * gsd, gsd))
        coords = np.dstack((x.flatten(), y.flatten())).squeeze()
        values = values.flatten()

    # Get the ground sampling distance from the coordinates before keeping only valid data, if it was not provided
    if gsd is None:
        gsd = np.mean([coords[0, 0] - coords[0, 1], coords[0, 0] - coords[1, 0]])
    # Get extent
    extent = (np.min(coords[:, 0]), np.max(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 1]))

    # Get the maximum lag from the coordinates before keeping only valid data, if it was not provided
    if "maxlag" not in kwargs.keys():
        # We define maximum lag as the maximum distance between coordinates (needed to provide custom bins, otherwise
        # skgstat rewrites the maxlag with the subsample of coordinates provided)
        maxlag = np.sqrt(
            (np.max(coords[:, 0]) - np.min(coords[:, 0])) ** 2 + (np.max(coords[:, 1]) - np.min(coords[:, 1])) ** 2
        )
        kwargs.update({"maxlag": maxlag})

    # Keep only valid data for cdist methods, remove later for pdist methods
    if "cdist" in subsample_method:
        ind_valid = np.isfinite(values)
        values = values[ind_valid]
        coords = coords[ind_valid, :]

    if "bin_func" not in kwargs.keys():
        # If no bin_func is provided, we provide an Iterable to provide a custom binning function to skgstat,
        # because otherwise bins might be inconsistent across runs
        bin_func = []
        right_bin_edge = np.sqrt(2) * gsd
        while right_bin_edge < kwargs.get("maxlag"):
            bin_func.append(right_bin_edge)
            # We use the default exponential increasing factor of RasterEquidistantMetricSpace, adapted for grids
            right_bin_edge *= np.sqrt(2)
        bin_func.append(kwargs.get("maxlag"))
        kwargs.update({"bin_func": bin_func})

    # Prepare necessary arguments to pass to variogram subfunctions
    args = {
        "values": values,
        "coords": coords,
        "subsample_method": subsample_method,
        "subsample": subsample,
    }
    if subsample_method in ["cdist_equidistant", "pdist_ring", "pdist_disk", "pdist_point"]:
        # The shape is needed for those three methods
        args.update({"shape": (nx, ny)})
        if subsample_method == "cdist_equidistant":
            # The coordinate extent is needed for this method
            args.update({"extent": extent})
        else:
            args.update({"gsd": gsd})

    # If a random_state is passed, each run needs to be passed an independent child random state, otherwise they will
    # provide exactly the same sampling and results
    if random_state is not None:
        # Define the random state if only a seed is provided
        rng = np.random.default_rng(random_state)

        # Create a list of child random states per number of variograms
        list_random_state: list[None | np.random.Generator] = list(
            rng.choice(n_variograms, n_variograms, replace=False)
        )
    else:
        list_random_state = [None for i in range(n_variograms)]

    # Derive the variogram
    # Differentiate between 1 core and several cores for multiple runs
    # All variogram runs have random sampling inherent to their subfunctions, so we provide the same input arguments
    if n_jobs == 1:
        logging.info("Using 1 core...")

        list_df_run = []
        for i in range(n_variograms):

            argdict = {
                "i": i,
                "imax": n_variograms,
                "random_state": list_random_state[i],
                **args,
                **kwargs,  # type: ignore
            }
            df_run = _wrapper_get_empirical_variogram(argdict=argdict)

            list_df_run.append(df_run)
    else:
        logging.info("Using " + str(n_jobs) + " cores...")

        pool = mp.Pool(n_jobs, maxtasksperchild=1)
        list_argdict = [
            {"i": i, "imax": n_variograms, "random_state": list_random_state[i], **args, **kwargs}  # type: ignore
            for i in range(n_variograms)
        ]
        list_df_run = pool.map(_wrapper_get_empirical_variogram, list_argdict, chunksize=1)
        pool.close()
        pool.join()

    # Aggregate multiple ranges subsampling
    df = pd.concat(list_df_run)

    # For a single run, no multi-run sigma estimated
    if n_variograms == 1:
        df = df.rename(columns={"bins": "lags"})
        df["err_exp"] = np.nan
    # For several runs, group results, use mean as empirical variogram, estimate sigma, and sum the counts
    else:
        df_grouped = df.groupby("bins", dropna=False)
        df_mean = df_grouped[["exp"]].mean()
        df_std = df_grouped[["exp"]].std()
        df_count = df_grouped[["count"]].sum()
        df_mean["lags"] = df_mean.index.values
        df_mean["err_exp"] = df_std["exp"] / np.sqrt(n_variograms)
        df_mean["count"] = df_count["count"]
        df = df_mean

    # Fix variance error for Dowd's variogram in SciKit-GStat

    # If skgstat > 1.0, we can use Dowd's without correcting, otherwise we correct
    from packaging.version import Version

    if Version(skg.__version__) <= Version("1.0.0"):
        if "estimator" in kwargs.keys() and kwargs["estimator"] == "dowd":
            # Correction: we divide all experimental variance values by 2
            df.exp.values /= 2
            df.err_exp.values /= 2

    # Remove the last spatial lag bin which is always undersampled
    df.drop(df.tail(1).index, inplace=True)

    # Force output dtype (default differs on different OS)
    df = df.astype({"exp": "float64", "err_exp": "float64", "lags": "float64", "count": "int64"})

    return df


def _get_skgstat_variogram_model_name(model: str | Callable[[NDArrayf, float, float], NDArrayf]) -> str:
    """Function to identify a SciKit-GStat variogram model from a string or a function"""

    if not _has_skgstat:
        raise ValueError("Optional dependency needed. Install 'scikit-gstat'.")

    list_supported_models = ["spherical", "gaussian", "exponential", "cubic", "stable", "matern"]

    if callable(model):
        if inspect.getmodule(model).__name__ == "skgstat.models":  # type: ignore
            model_name = model.__name__
        else:
            raise ValueError("Variogram models can only be passed as functions of the skgstat.models package.")

    elif isinstance(model, str):
        model_name = "None"
        for supp_model in list_supported_models:
            if model.lower() in [supp_model[0:3], supp_model]:
                model_name = supp_model.lower()
        if model_name == "None":
            raise ValueError(
                f"Variogram model name {model} not recognized. Supported models are: "
                + ", ".join(list_supported_models)
                + "."
            )

    else:
        raise ValueError(
            "Variogram models can be passed as strings or skgstat.models function. "
            "Supported models are: " + ", ".join(list_supported_models) + "."
        )

    return model_name


def get_variogram_model_func(params_variogram_model: pd.DataFrame) -> Callable[[NDArrayf], NDArrayf]:
    """
    Construct the sum of spatial variogram function from a dataframe of variogram parameters.

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).

    :return: Function of sum of variogram with spatial lags.
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Define the function of sum of variogram models of h (spatial lag) to return
    def sum_model(h: NDArrayf) -> NDArrayf:

        fn = np.zeros(np.shape(h))

        for i in range(len(params_variogram_model)):
            # Get scikit-gstat model from name or Callable
            model_name = _get_skgstat_variogram_model_name(params_variogram_model["model"].values[i])
            model_function = getattr(skg.models, model_name)
            r = params_variogram_model["range"].values[i]
            p = params_variogram_model["psill"].values[i]
            # For models that expect 2 parameters
            if model_name in ["spherical", "gaussian", "exponential", "cubic"]:
                fn += model_function(h, r, p)
            # For models that expect 3 parameters
            elif model_name in ["stable", "matern"]:
                s = params_variogram_model["smooth"].values[i]
                fn += model_function(h, r, p, s)
        return fn

    return sum_model


def covariance_from_variogram(params_variogram_model: pd.DataFrame) -> Callable[[NDArrayf], NDArrayf]:
    """
    Construct the spatial covariance function from a dataframe of variogram parameters.
    The covariance function is the sum of partial sills "PS" minus the sum of associated variograms "gamma":
    C = PS - gamma

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).

    :return: Covariance function with spatial lags
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get total sill
    total_sill = np.sum(params_variogram_model["psill"])

    # Get function from sum of variogram
    sum_variogram = get_variogram_model_func(params_variogram_model)

    def cov(h: NDArrayf) -> NDArrayf:
        return total_sill - sum_variogram(h)

    return cov


def correlation_from_variogram(params_variogram_model: pd.DataFrame) -> Callable[[NDArrayf], NDArrayf]:
    """
    Construct the spatial correlation function from a dataframe of variogram parameters.
    The correlation function is the covariance function "C" divided by the sum of partial sills "PS": rho = C / PS

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).

    :return: Correlation function with spatial lags
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get total sill
    total_sill = np.sum(params_variogram_model["psill"].values)

    # Get covariance from sum of variogram
    cov = covariance_from_variogram(params_variogram_model)

    def rho(h: NDArrayf) -> NDArrayf:
        return cov(h) / total_sill

    return rho


def fit_sum_model_variogram(
    list_models: list[str | Callable[[NDArrayf, float, float], NDArrayf]],
    empirical_variogram: pd.DataFrame,
    bounds: list[tuple[float, float]] = None,
    p0: list[float] = None,
    maxfev: int = None,
) -> tuple[Callable[[NDArrayf], NDArrayf], pd.DataFrame]:
    """
    Fit a sum of variogram models to an empirical variogram, with weighted least-squares based on sampling errors. To
    use preferably with the empirical variogram dataframe returned by the `sample_empirical_variogram` function.

    :param list_models: List of K variogram models to sum for the fit in order from short to long ranges. Can either be
        a 3-letter string, full string of the variogram name or SciKit-GStat model function (e.g., for a
        spherical model "Sph", "Spherical" or skgstat.models.spherical).
    :param empirical_variogram: Empirical variogram, formatted as a dataframe with count (pairwise sample count), lags
        (upper bound of spatial lag bin), exp (experimental variance), and err_exp (error on experimental variance).
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper, sill
        lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).
    :param maxfev: Maximum number of function evaluations before the termination, passed to scipy.optimize.curve_fit().
        Convergence problems can sometimes be fixed by changing this value (default None: automatically determine the
        number).

    :return: Function of sum of variogram, Dataframe of optimized coefficients.
    """

    # Define a function of a sum of variogram model forms, with undetermined arguments
    def variogram_sum(h: float, *args: list[float]) -> float:
        fn = 0.0
        i = 0
        for model in list_models:
            # Get the model name and convert to SciKit-GStat function
            model_name = _get_skgstat_variogram_model_name(model)
            model_function = getattr(skg.models, model_name)
            # For models that expect 2 parameters
            if model_name in ["spherical", "gaussian", "exponential", "cubic"]:
                fn += model_function(h, args[i], args[i + 1])
                i += 2
            # For models that expect 3 parameters
            elif model_name in ["stable", "matern"]:
                fn += model_function(h, args[i], args[i + 1], args[i + 2])
                i += 3

        return fn

    # First, filter outliers
    empirical_variogram = empirical_variogram[np.isfinite(empirical_variogram.exp.values)]

    # Use shape of empirical variogram to assess rough boundaries/first estimates
    n_average = np.ceil(len(empirical_variogram.exp.values) / 10)
    exp_movaverage = np.convolve(empirical_variogram.exp.values, np.ones(int(n_average)) / n_average, mode="valid")
    # Maximum variance of the process
    max_var = np.max(exp_movaverage)

    # Simplify things for scipy: let's provide boundaries and first guesses
    if bounds is None:
        bounds = [(0, empirical_variogram.lags.values[-1]), (0, max_var)] * len(list_models)

    if p0 is None:
        p0 = []
        for i in range(len(list_models)):
            # Use psill evenly distributed
            psill_p0 = ((i + 1) / len(list_models)) * max_var

            # Use corresponding ranges
            # !! This fails when no empirical value crosses this (too wide binning/nugget)
            # ind = np.array(np.abs(exp_movaverage-psill_p0)).argmin()
            # range_p0 = empirical_variogram.lags.values[ind]
            range_p0 = ((i + 1) / len(list_models)) * empirical_variogram.lags.values[-1]

            p0.append(range_p0)
            p0.append(psill_p0)

    final_bounds = np.transpose(np.array(bounds))

    # If the error provided is all NaNs (single variogram run), or all zeros (two variogram runs), run without weights
    if np.all(np.isnan(empirical_variogram.err_exp.values)) or np.all(empirical_variogram.err_exp.values == 0):
        cof, cov = curve_fit(
            variogram_sum,
            empirical_variogram.lags.values,
            empirical_variogram.exp.values,
            method="trf",
            p0=p0,
            bounds=final_bounds,
            maxfev=maxfev,
        )
    # Otherwise, use a weighted fit
    else:
        # We need to filter for possible no data in the error
        valid = np.isfinite(empirical_variogram.err_exp.values)
        cof, cov = curve_fit(
            variogram_sum,
            empirical_variogram.lags.values[valid],
            empirical_variogram.exp.values[valid],
            method="trf",
            p0=p0,
            bounds=final_bounds,
            sigma=empirical_variogram.err_exp.values[valid],
            maxfev=maxfev,
        )

    # Store optimized parameters
    list_df = []
    i = 0
    for model in list_models:
        model_name = _get_skgstat_variogram_model_name(model)
        # For models that expect 2 parameters
        if model_name in ["spherical", "gaussian", "exponential", "cubic"]:
            df = pd.DataFrame()
            df = df.assign(model=[model_name], range=[cof[i]], psill=[cof[i + 1]])
            i += 2
        # For models that expect 3 parameters
        elif model_name in ["stable", "matern"]:
            df = pd.DataFrame()
            df = df.assign(model=[model_name], range=[cof[i]], psill=[cof[i + 1]], smooth=[cof[i + 2]])
            i += 3
        list_df.append(df)
    df_params = pd.concat(list_df)

    # Also pass the function of sum of variogram
    variogram_sum_fit = get_variogram_model_func(df_params)

    return variogram_sum_fit, df_params


def _estimate_model_spatial_correlation(
    dvalues: NDArrayf | RasterType,
    list_models: list[str | Callable[[NDArrayf, float, float], NDArrayf]],
    estimator: str = "dowd",
    gsd: float = None,
    coords: NDArrayf = None,
    subsample: int = 1000,
    subsample_method: str = "cdist_equidistant",
    n_variograms: int = 1,
    n_jobs: int = 1,
    random_state: int | np.random.Generator | None = None,
    bounds: list[tuple[float, float]] = None,
    p0: list[float] = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Callable[[NDArrayf], NDArrayf]]:
    """
    Estimate and model the spatial correlation of the input variable by empirical variogram sampling and fitting of a
    sum of variogram model.

    The spatial correlation is returned as a function of spatial lags (in units of the input coordinates) which gives a
    correlation value between 0 and 1.

    This function samples an empirical variogram using skgstat.Variogram, then optimizes by weighted least-squares the
    sum of a defined number of models, using the functions `sample_empirical_variogram` and `fit_sum_model_variogram`.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_models: List of K variogram models to sum for the fit in order from short to long ranges. Can either be
        a 3-letter string, full string of the variogram name or SciKit-GStat model function (e.g., for a
        spherical model "Sph", "Spherical" or skgstat.models.spherical).
    :param estimator: Estimator for the empirical variogram; default to Dowd's variogram (see skgstat.Variogram for
        the list of available estimators).
    :param gsd: Ground sampling distance
    :param coords: Coordinates
    :param subsample: Number of samples to randomly draw from the values
    :param subsample_method: Spatial subsampling method
    :param n_variograms: Number of independent empirical variogram estimations (to estimate empirical variogram spread)
    :param n_jobs: Number of processing cores
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper,
        sill lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).

    :return: Dataframe of empirical variogram, Dataframe of optimized model parameters, Function of spatial correlation
        (0 to 1) with spatial lags
    """

    empirical_variogram = sample_empirical_variogram(
        values=dvalues,
        estimator=estimator,
        gsd=gsd,
        coords=coords,
        subsample=subsample,
        subsample_method=subsample_method,
        n_variograms=n_variograms,
        n_jobs=n_jobs,
        random_state=random_state,
        **kwargs,
    )

    params_variogram_model = fit_sum_model_variogram(
        list_models=list_models, empirical_variogram=empirical_variogram, bounds=bounds, p0=p0
    )[1]

    spatial_correlation_func = correlation_from_variogram(params_variogram_model=params_variogram_model)

    return empirical_variogram, params_variogram_model, spatial_correlation_func


def infer_spatial_correlation_from_stable(
    dvalues: NDArrayf | RasterType,
    list_models: list[str | Callable[[NDArrayf, float, float], NDArrayf]],
    stable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    unstable_mask: NDArrayf | Mask | VectorType | gpd.GeoDataFrame = None,
    errors: NDArrayf | RasterType = None,
    estimator: str = "dowd",
    gsd: float = None,
    coords: NDArrayf = None,
    subsample: int = 1000,
    subsample_method: str = "cdist_equidistant",
    n_variograms: int = 1,
    n_jobs: int = 1,
    bounds: list[tuple[float, float]] = None,
    p0: list[float] = None,
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, Callable[[NDArrayf], NDArrayf]]:
    """
    Infer spatial correlation of errors from differenced values on stable terrain and a list of variogram model to fit
    as a sum.

    This function returns a dataframe of the empirical variogram, a dataframe of optimized model parameters, and a
    spatial correlation function. The spatial correlation is returned as a function of spatial lags
    (in units of the input coordinates) which gives a correlation value between 0 and 1.
    It is a convenience wrapper for `estimate_model_spatial_correlation` to work on either Raster or array and compute
    the stable mask.

    If no stable or unstable mask is provided to mask in or out the values, all terrain is used.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as
        elevation differences on stable terrain)
    :param list_models: List of K variogram models to sum for the fit in order from short to long ranges. Can either be
        a 3-letter string, full string of the variogram name or SciKit-GStat model function (e.g., for a
        spherical model "Sph", "Spherical" or skgstat.models.spherical).
    :param stable_mask: Vector shapefile of stable terrain (if dvalues is Raster), or boolean array of same shape as
        dvalues
    :param unstable_mask: Vector shapefile of unstable terrain (if dvalues is Raster), or boolean array of same shape
        as dvalues
    :param errors: Error values to account for heteroscedasticity (ignored if None).
    :param estimator: Estimator for the empirical variogram; default to Dowd's variogram (see skgstat.Variogram for
        the list of available estimators).
    :param gsd: Ground sampling distance, if input values are provided as array
    :param coords: Coordinates
    :param subsample: Number of samples to randomly draw from the values
    :param subsample_method: Spatial subsampling method
    :param n_variograms: Number of independent empirical variogram estimations (to estimate empirical variogram spread)
    :param n_jobs: Number of processing cores
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper,
        sill lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Dataframe of empirical variogram, Dataframe of optimized model parameters, Function of spatial correlation
        (0 to 1) with spatial lags
    """

    dvalues_stable_arr, gsd = _preprocess_values_with_mask_to_array(
        values=dvalues, include_mask=stable_mask, exclude_mask=unstable_mask, gsd=gsd
    )

    # Perform standardization if error array is provided
    if errors is not None:
        if isinstance(errors, Raster):
            errors_arr = get_array_and_mask(errors)[0]
        else:
            errors_arr = errors

        # Standardize
        dvalues_stable_arr /= errors_arr

    # Estimate and model spatial correlations
    empirical_variogram, params_variogram_model, spatial_correlation_func = _estimate_model_spatial_correlation(
        dvalues=dvalues_stable_arr,
        list_models=list_models,
        estimator=estimator,
        gsd=gsd,
        coords=coords,
        subsample=subsample,
        subsample_method=subsample_method,
        n_variograms=n_variograms,
        n_jobs=n_jobs,
        random_state=random_state,
        bounds=bounds,
        p0=p0,
        **kwargs,
    )

    return empirical_variogram, params_variogram_model, spatial_correlation_func


def _check_validity_params_variogram(params_variogram_model: pd.DataFrame) -> None:
    """Check the validity of the modelled variogram parameters dataframe (mostly in the case it is passed manually)."""

    # Check that expected columns exists
    expected_columns = ["model", "range", "psill"]
    if not all(c in params_variogram_model for c in expected_columns):
        raise ValueError(
            'The dataframe with variogram parameters must contain the columns "model", "range" and "psill".'
        )

    # Check that the format of variogram models are correct
    for model in params_variogram_model["model"].values:
        _get_skgstat_variogram_model_name(model)

    # Check that the format of ranges, sills are correct
    for r in params_variogram_model["range"].values:
        if not isinstance(r, (float, np.floating, int, np.integer)):
            raise ValueError("The variogram ranges must be float or integer.")
        if r <= 0:
            raise ValueError("The variogram ranges must have non-zero, positive values.")

    # Check that the format of ranges, sills are correct
    for p in params_variogram_model["psill"].values:
        if not isinstance(p, (float, np.floating, int, np.integer)):
            raise ValueError("The variogram partial sills must be float or integer.")
        if p <= 0:
            raise ValueError("The variogram partial sills must have non-zero, positive values.")

    # Check that the matern smoothness factor exist and is rightly formatted
    if ["stable"] in params_variogram_model["model"].values or ["matern"] in params_variogram_model["model"].values:
        if "smooth" not in params_variogram_model:
            raise ValueError(
                'The dataframe with variogram parameters must contain the column "smooth" for '
                "the smoothness factor when using Matern or Stable models."
            )
        for i in range(len(params_variogram_model)):
            if params_variogram_model["model"].values[i] in ["stable", "matern"]:
                s = params_variogram_model["smooth"].values[i]
                if not isinstance(s, (float, np.floating, int, np.integer)):
                    raise ValueError("The variogram smoothness parameter must be float or integer.")
                if s <= 0:
                    raise ValueError("The variogram smoothness parameter must have non-zero, positive values.")


def neff_circular_approx_theoretical(area: float, params_variogram_model: pd.DataFrame) -> float:
    """
    Number of effective samples approximated from exact disk integration of a sum of any number of variogram models
    of spherical, gaussian, exponential or cubic form over a disk of a certain area. This approximation performs best
    for areas with a shape close to that of a disk.
    Inspired by Rolstad et al. (2009): http://dx.doi.org/10.3189/002214309789470950.
    The input variogram parameters match the format of the dataframe returned by `fit_sum_variogram_models`, also
    detailed in the parameter description to be passed manually.

    This function contains the exact integrated formulas and is mostly used for testing the numerical integration
    of any number and forms of variograms provided by the function `neff_circular_approx_numerical`.

    The number of effective samples serves to convert between standard deviation and standard error. For example, with
    two models: if SE is the standard error, SD the standard deviation and N_eff the number of effective samples:
    SE = SD / sqrt(N_eff) => N_eff = SD^2 / SE^2 => N_eff = (PS1 + PS2)/SE^2 where PS1 and PS2 are the partial sills
    estimated from the variogram models, and SE is estimated by integrating the variogram models with parameters PS1/PS2
    and R1/R2 where R1/R2 are the correlation ranges.

    :param area: Area (in square unit of the variogram ranges)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).

    :return: Number of effective samples
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

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
        if l_equiv <= a1:
            squared_se = c1 * (1 - L / a1 + 1 / 5 * (L / a1) ** 3)
        else:
            squared_se = c1 / 5 * (a1 / L) ** 2
        return squared_se

    # Exponential: h * covariance = c1 * h * exp(-h/a); a = a1/3
    # Exponential: radial integral of above from 0 to L: SE**2 =  2 / (L**2) * c1 * a * (a - exp(-L/a) * (a + L))

    def exponential_exact_integral(a1: float, c1: float, L: float) -> float:
        a = a1 / 3
        squared_se = 2 * c1 * (a / L) ** 2 * (1 - np.exp(-L / a) * (1 + L / a))
        return squared_se

    # Gaussian: h * covariance = c1 * h * exp(-h**2/a**2) ; a = a1/2
    # Gaussian: radial integral of above from 0 to L: SE**2 = 2 / (L**2) * c1 * 1/2 * a**2 * (1 - exp(-L**2/a**2))

    def gaussian_exact_integral(a1: float, c1: float, L: float) -> float:
        a = a1 / 2
        squared_se = c1 * (a / L) ** 2 * (1 - np.exp(-(L**2) / a**2))
        return squared_se

    # Cubic: h * covariance = c1 * h * (1 - (7 * (h**2 / a**2)) + ((35 / 4) * (h**3 / a**3)) -
    #                          ((7 / 2) * (h**5 / a**5)) + ((3 / 4) * (h**7 / a**7)))
    # Cubic: radial integral of above from 0 to L:
    # SE**2 = c1 * (6*a**7 -21*a**5*L**2 + 21*a**4*L**3 - 6*a**2*L**5 + L**7) / (6*a**7)

    def cubic_exact_integral(a1: float, c1: float, L: float) -> float:
        if l_equiv <= a1:
            squared_se = (
                c1 * (6 * a1**7 - 21 * a1**5 * L**2 + 21 * a1**4 * L**3 - 6 * a1**2 * L**5 + L**7) / (6 * a1**7)
            )
        else:
            squared_se = 1 / 6 * c1 * a1**2 / L**2
        return squared_se

    squared_se = 0.0
    valid_models = ["spherical", "exponential", "gaussian", "cubic"]
    exact_integrals = [
        spherical_exact_integral,
        exponential_exact_integral,
        gaussian_exact_integral,
        cubic_exact_integral,
    ]
    for i in np.arange(len(params_variogram_model)):
        model_name = _get_skgstat_variogram_model_name(params_variogram_model["model"].values[i])
        r = params_variogram_model["range"].values[i]
        p = params_variogram_model["psill"].values[i]
        if model_name in valid_models:
            exact_integral = exact_integrals[valid_models.index(model_name)]
            squared_se += exact_integral(r, p, l_equiv)

    # We sum all partial sill to get the total sill
    total_sill = np.nansum(params_variogram_model.psill)
    # The number of effective sample is the fraction of total sill by squared standard error
    neff = total_sill / squared_se

    return neff


def _integrate_fun(fun: Callable[[NDArrayf], NDArrayf], low_b: float, upp_b: float) -> float:
    """
    Numerically integrate function between an upper and lower bounds
    :param fun: Function to integrate
    :param low_b: Lower bound
    :param upp_b: Upper bound

    :return: Integral between lower and upper bound
    """
    return integrate.quad(fun, low_b, upp_b)[0]


def neff_circular_approx_numerical(area: float | int, params_variogram_model: pd.DataFrame) -> float:
    """
    Number of effective samples derived from numerical integration for any sum of variogram models over a circular area.
    This is a generalization of Rolstad et al. (2009): http://dx.doi.org/10.3189/002214309789470950, which is verified
    against exact integration of `neff_circular_approx_theoretical`. This approximation performs best for areas with
    a shape close to that of a disk.
    The input variogram parameters match the format of the dataframe returned by `fit_sum_variogram_models`, also
    detailed in the parameter description to be passed manually.

    The number of effective samples N_eff serves to convert between standard deviation and standard error
    over the area: SE = SD / sqrt(N_eff) if SE is the standard error, SD the standard deviation.

    :param area: Area (in square unit of the variogram ranges)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).

    :returns: Number of effective samples
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get the total sill from the sum of partial sills
    total_sill = np.nansum(params_variogram_model.psill)

    # Define the covariance sum function times the spatial lag, for later integration
    def hcov_sum(h: NDArrayf) -> NDArrayf:
        return h * covariance_from_variogram(params_variogram_model)(h)

    # Get a radius for which the circle as the defined area
    h_equiv = np.sqrt(area / np.pi)

    # Integrate the covariance function between the center and the radius
    full_int = _integrate_fun(hcov_sum, 0, h_equiv)

    # Get the standard error, and return the number of effective samples
    squared_se = 2 * np.pi * full_int / area

    # The number of effective sample is the fraction of total sill by squared standard error
    neff = total_sill / squared_se

    return neff


def neff_exact(
    coords: NDArrayf, errors: NDArrayf, params_variogram_model: pd.DataFrame, vectorized: bool = True
) -> float:
    """
     Exact number of effective samples derived from a double sum of covariance with euclidean coordinates based on
     the provided variogram parameters. This method works for any shape of area.

    :param coords: Center coordinates with size (N,2) for each spatial support (typically, pixel)
    :param errors: Errors at the coordinates with size (N,) for each spatial support (typically, pixel)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).
    :param vectorized: Perform the vectorized calculation (used for testing).

    :return: Number of effective samples
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get spatial correlation function from variogram parameters
    rho = correlation_from_variogram(params_variogram_model)

    # Get number of points and pairwise distance compacted matrix from scipy.pdist
    n = len(coords)
    pds = pdist(coords)

    # Now we compute the double covariance sum
    # Either using for-loop-version
    if not vectorized:
        var = 0.0
        for i in range(n):
            for j in range(n):

                # For index calculation of the pairwise distance,
                # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
                if i == j:
                    d = 0
                elif i < j:
                    ind = n * i + j - ((i + 2) * (i + 1)) // 2
                    d = pds[ind]
                else:
                    ind = n * j + i - ((j + 2) * (j + 1)) // 2
                    d = pds[ind]

                var += rho(d) * errors[i] * errors[j]  # type: ignore

    # Or vectorized version
    else:
        # Convert the compact pairwise distance into a square matrix
        pds_matrix = squareform(pds)
        # Vectorize calculation
        var = np.sum(
            errors.reshape((-1, 1)) @ errors.reshape((1, -1)) * rho(pds_matrix.flatten()).reshape(pds_matrix.shape)
        )

    # The number of effective sample is the fraction of total sill by squared standard error
    squared_se_dsc = var / n**2
    neff = float(np.mean(errors)) ** 2 / squared_se_dsc

    return neff


def neff_hugonnet_approx(
    coords: NDArrayf,
    errors: NDArrayf,
    params_variogram_model: pd.DataFrame,
    subsample: int = 1000,
    vectorized: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> float:
    """
    Approximated number of effective samples derived from a double sum of covariance subsetted on one of the two sums,
    based on euclidean coordinates with the provided variogram parameters. This method works for any shape of area.
    See Hugonnet et al. (2022), https://doi.org/10.1109/jstars.2022.3188922, in particular Supplementary Fig. S16.

    :param coords: Center coordinates with size (N,2) for each spatial support (typically, pixel)
    :param errors: Errors at the coordinates with size (N,) for each spatial support (typically, pixel)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).
    :param subsample: Number of samples to subset the calculation
    :param vectorized: Perform the vectorized calculation (used for testing).
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Number of effective samples
    """

    # Define random state
    rng = np.random.default_rng(random_state)

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get spatial correlation function from variogram parameters
    rho = correlation_from_variogram(params_variogram_model)

    # Get number of points and pairwise distance matrix from scipy.cdist
    n = len(coords)

    # At maximum, the number of subsamples has to be equal to number of points
    subsample = min(subsample, n)

    # Get random subset of points for one of the sums
    rand_points = rng.choice(n, size=subsample, replace=False)

    # Subsample coordinates in 1D before computing pairwise distances
    sub_coords = coords[rand_points, :]
    sub_errors = errors[rand_points]
    pds_matrix = cdist(coords, sub_coords, "euclidean")

    # Now we compute the double covariance sum
    # Either using for-loop-version
    if not vectorized:
        var = 0.0
        for i in range(pds_matrix.shape[0]):
            for j in range(pds_matrix.shape[1]):
                d = pds_matrix[i, j]
                var += rho(d) * errors[i] * errors[j]  # type: ignore

    # Or vectorized version
    else:
        # Vectorized calculation
        var = np.sum(
            errors.reshape((-1, 1)) @ sub_errors.reshape((1, -1)) * rho(pds_matrix.flatten()).reshape(pds_matrix.shape)
        )

    # The number of effective sample is the fraction of total sill by squared standard error
    squared_se_dsc = var / (n * subsample)
    neff = float(np.mean(errors)) ** 2 / squared_se_dsc

    return neff


def number_effective_samples(
    area: float | int | VectorType | gpd.GeoDataFrame,
    params_variogram_model: pd.DataFrame,
    rasterize_resolution: RasterType | float = None,
    **kwargs: Any,
) -> float:
    """
    Compute the number of effective samples, i.e. the number of uncorrelated samples, in an area accounting for spatial
    correlations described by a sum of variogram models.

    This function wraps two methods:

    - A discretized integration method that provides the exact estimate for any shape of area using a double sum of
        covariance. By default, this method is approximated using Equation 18 of Hugonnet et al. (2022),
        https://doi.org/10.1109/jstars.2022.3188922 to decrease computing times while preserving a good approximation.

    - A continuous integration method that provides a conservative (i.e., slightly overestimated) value for a disk
        area shape, based on a generalization of the approach of Rolstad et al. (2009),
        http://dx.doi.org/10.3189/002214309789470950.

    By default, if a numeric value is passed for an area, the continuous method is used considering a disk shape. If a
    vector is passed, the discretized method is computed on that shape. If the discretized method is used, a resolution
    for rasterization is generally expected, otherwise is arbitrarily chosen as a fifth of the shortest correlation
    range to ensure a sufficiently fine grid for propagation of the shortest range.

    :param area: Area of interest either as a numeric value of surface in the same unit as the variogram ranges (will
        assume a circular shape), or as a vector (shapefile) of the area
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).
    :param rasterize_resolution: Resolution to rasterize the area if passed as a vector. Can be a float value or a
        Raster.
    :param kwargs: Keyword argument to pass to the `neff_hugonnet_approx` function.

    :return: Number of effective samples
    """

    # Check input for variogram parameters
    _check_validity_params_variogram(params_variogram_model=params_variogram_model)

    # If area is numeric, run the continuous circular approximation
    if isinstance(area, (float, int)):
        neff = neff_circular_approx_numerical(area=area, params_variogram_model=params_variogram_model)

    # Otherwise, run the discrete sum of covariance
    elif isinstance(area, (Vector, gpd.GeoDataFrame)):

        # If the input is a geopandas dataframe, put into a Vector object
        if isinstance(area, gpd.GeoDataFrame):
            V = Vector(area)
        else:
            V = area

        if rasterize_resolution is None:
            rasterize_resolution = np.min(params_variogram_model["range"].values) / 5.0
            warnings.warn(
                "Resolution for vector rasterization is not defined and thus set at 20% of the shortest "
                "correlation range, which might result in large memory usage."
            )

        # Rasterize with numeric resolution or Raster metadata
        if isinstance(rasterize_resolution, (float, int, np.floating, np.integer)):

            # We only need relative mask and coordinates, not absolute
            mask = V.create_mask(xres=rasterize_resolution, as_array=True)
            x = rasterize_resolution * np.arange(0, mask.shape[0])
            y = rasterize_resolution * np.arange(0, mask.shape[1])
            coords = np.array(np.meshgrid(y, x))
            coords_on_mask = coords[:, mask].T

        elif isinstance(rasterize_resolution, Raster):

            # With a Raster we can get the coordinates directly
            mask = V.create_mask(raster=rasterize_resolution, as_array=True).squeeze()
            coords = np.array(rasterize_resolution.coords())
            coords_on_mask = coords[:, mask].T

        else:
            raise ValueError("The rasterize resolution must be a float, integer or Raster subclass.")

        # Here we don't care about heteroscedasticity, so all errors are standardized to one
        errors_on_mask = np.ones(len(coords_on_mask))

        neff = neff_hugonnet_approx(
            coords=coords_on_mask, errors=errors_on_mask, params_variogram_model=params_variogram_model, **kwargs
        )

    else:
        raise ValueError("Area must be a float, integer, Vector subclass or geopandas dataframe.")

    return neff


def spatial_error_propagation(
    areas: list[float | VectorType | gpd.GeoDataFrame],
    errors: RasterType,
    params_variogram_model: pd.Dataframe,
    **kwargs: Any,
) -> list[float]:
    """
    Spatial propagation of elevation errors to an area using the estimated heteroscedasticity and spatial correlations.

    This function is based on the `number_effective_samples` function to estimate uncorrelated samples. If given a
    vector area, it uses Equation 18 of Hugonnet et al. (2022), https://doi.org/10.1109/jstars.2022.3188922. If given
    a numeric area, it uses a generalization of Rolstad et al. (2009), http://dx.doi.org/10.3189/002214309789470950.

    The standard error SE (1-sigma) is then computed as SE = mean(SD) / Neff, where mean(SD) is the mean of errors in
    the area of interest which accounts for heteroscedasticity, and Neff is the number of effective samples.

    :param areas: Area of interest either as a numeric value of surface in the same unit as the variogram ranges (will
        assume a circular shape), or as a vector (shapefile) of the area.
    :param errors: Errors from heteroscedasticity estimation and modelling, as an array or Raster.
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the
        model types (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for
        the partial sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model
        (e.g., [None, 0.2]).
    :param kwargs: Keyword argument to pass to the `neff_hugonnet_approx` function.

    :return: List of standard errors (1-sigma) for the input areas
    """

    standard_errors = []
    errors_arr = get_array_and_mask(errors)[0]
    for area in areas:
        # We estimate the number of effective samples in the area
        neff = number_effective_samples(
            area=area, params_variogram_model=params_variogram_model, rasterize_resolution=errors, **kwargs
        )

        # We compute the average error in this area
        # If the area is only a value, take the average error over the entire Raster
        if isinstance(area, float):
            average_spread = np.nanmean(errors_arr)
        else:
            if isinstance(area, gpd.GeoDataFrame):
                area_vector = Vector(area)
            else:
                area_vector = area
            area_mask = area_vector.create_mask(errors, as_array=True).squeeze()

            average_spread = np.nanmean(errors_arr[area_mask])

        # Compute the standard error from those two values
        standard_error = average_spread / np.sqrt(neff)
        standard_errors.append(standard_error)

    return standard_errors


def _std_err_finite(std: float, neff_tot: float, neff: float) -> float:
    """
    Standard error formula for a subsample of a finite ensemble.

    :param std: standard deviation
    :param neff_tot: maximum number of effective samples
    :param neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / neff_tot * (neff_tot - neff) / neff_tot)


def _std_err(std: float, neff: float) -> float:
    """
    Standard error formula.

    :param std: standard deviation
    :param neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / neff)


def _distance_latlon(tup1: tuple[float, float], tup2: tuple[float, float], earth_rad: float = 6373000) -> float:
    """
    Distance between two lat/lon coordinates projected on a spheroid
    ref: https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude
    :param tup1: lon/lat coordinates of first point
    :param tup2: lon/lat coordinates of second point
    :param earth_rad: radius of the earth in meters

    :return: distance
    """
    lat1 = m.radians(abs(tup1[1]))
    lon1 = m.radians(abs(tup1[0]))
    lat2 = m.radians(abs(tup2[1]))
    lon2 = m.radians(abs(tup2[0]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = m.sin(dlat / 2) ** 2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon / 2) ** 2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    distance = earth_rad * c

    return distance


def _scipy_convolution(imgs: NDArrayf, filters: NDArrayf, output: NDArrayf) -> None:
    """
    Scipy convolution on a number n_N of 2D images of size N1 x N2 using a number of kernels n_M of sizes M1 x M2.

    :param imgs: Input array of size (n_N, N1, N2) with n_N images of size N1 x N2
    :param filters: Input array of filters of size (n_M, M1, M2) with n_M filters of size M1 x M2
    :param output: Initialized output array of size (n_N, n_M, N1, N2)
    """

    for i_N in np.arange(imgs.shape[0]):
        for i_M in np.arange(filters.shape[0]):
            output[i_N, i_M, :, :] = scipy.ndimage.convolve(
                imgs[i_N, :, :], filters[i_M, :, :], mode="constant", cval=np.nan
            )


@numba.njit(parallel=True)  # type: ignore
def _numba_convolution(imgs: NDArrayf, filters: NDArrayf, output: NDArrayf) -> NDArrayf:
    """
    Numba convolution on a number n_N of 2D images of size N1 x N2 using a number of kernels n_M of sizes M1 x M2.

    :param imgs: Input array of size (n_N, N1, N2) with n_N images of size N1 x N2
    :param filters: Input array of filters of size (n_M, M1, M2) with n_M filters of size M1 x M2
    :param output: Initialized output array of size (n_N, n_M, N1, N2)
    """
    # Shapes
    n_N, N1, N2 = imgs.shape
    n_M, M1, M2 = filters.shape

    # Range
    row_range = N1 - M1 + 1
    col_range = N2 - M2 + 1

    for ii in range(n_N):
        for rr in prange(row_range):
            for cc in prange(col_range):
                for m1 in range(M1):
                    for m2 in range(M2):
                        for ff in range(n_M):
                            imgval = imgs[ii, rr + m1, cc + m2]
                            filterval = filters[ff, m1, m2]
                            output[ii, ff, rr, cc] += imgval * filterval

    return output


def convolution(imgs: NDArrayf, filters: NDArrayf, method: str = "scipy") -> NDArrayf:
    """
    Convolution on a number n_N of 2D images of size N1 x N2 using a number of kernels n_M of sizes M1 x M2, using
    either scipy.signal.fftconvolve or accelerated numba loops.
    Note that the indexes on n_M and n_N correspond to first axes on the array to speed up computations (prefetching).
    Inspired by: https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

    :param imgs: Input array of size (n_N, N1, N2) with n_N images of size N1 x N2
    :param filters: Input array of filters of size (n_M, M1, M2) with n_M filters of size M1 x M2
    :param method: Method to perform the convolution: "scipy" or "numba"

    :return: Filled array of outputs of size (n_N, n_M, N1, N2)
    """

    # Initialize output array according to input shapes
    n_N, N1, N2 = imgs.shape
    n_M, M1, M2 = filters.shape
    output = np.zeros((n_N, n_M, N1, N2))

    if method.lower() == "scipy":
        _scipy_convolution(imgs=imgs, filters=filters, output=output)
    elif "numba" in method.lower():
        half_M1 = int((M1 - 1) / 2)
        half_M2 = int((M2 - 1) / 2)
        imgs_pad = np.pad(imgs, pad_width=((0, 0), (half_M1, half_M1), (half_M2, half_M2)), constant_values=np.nan)
        output = _numba_convolution(
            imgs=imgs_pad,
            filters=filters,
            output=output,
        )
    else:
        raise ValueError('Method must be "scipy" or "numba".')

    return output


def mean_filter_nan(
    img: NDArrayf, kernel_size: int, kernel_shape: str = "circular", method: str = "scipy"
) -> tuple[NDArrayf, NDArrayf, int]:
    """
    Apply a mean filter to an image with a square or circular kernel of size p and with NaN values ignored.

    :param img: Input array of size (N1, N2)
    :param kernel_size: Size M of kernel, which will be a symmetrical (M, M) kernel
    :param kernel_shape: Shape of kernel, either "square" or "circular"
    :param method: Method to perform the convolution: "scipy" or "numba"

    :return: Array of size (N1, N2) with mean values, Array of size (N1, N2) with number of valid pixels, Number of
        pixels in the kernel
    """

    # Simplify kernel size notation
    p = kernel_size

    # Copy the array and replace NaNs by zeros before summing them in the convolution
    img_zeroed = img.copy()
    img_zeroed[~np.isfinite(img_zeroed)] = 0

    # Define square kernel
    if kernel_shape.lower() == "square":
        kernel = np.ones((p, p), dtype="uint8")

    # Circle kernel
    elif kernel_shape.lower() == "circular":
        kernel = _create_circular_mask((p, p)).astype("uint8")
    else:
        raise ValueError('Kernel shape should be "square" or "circular".')

    # Run convolution to compute the sum of img values
    summed_img = convolution(
        imgs=img_zeroed.reshape((1, img_zeroed.shape[0], img_zeroed.shape[1])),
        filters=kernel.reshape((1, kernel.shape[0], kernel.shape[1])),
        method=method,
    ).squeeze()

    # Construct a boolean array for nodatas
    nodata_img = np.ones(np.shape(img), dtype=np.int8)
    nodata_img[~np.isfinite(img)] = 0

    # Count the number of valid pixels in the kernel with a convolution
    nb_valid_img = convolution(
        imgs=nodata_img.reshape((1, nodata_img.shape[0], nodata_img.shape[1])),  # type: ignore
        filters=kernel.reshape((1, kernel.shape[0], kernel.shape[1])),
        method=method,
    ).squeeze()

    # Compute the final mean filter which accounts for no data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero encountered in *divide")
        mean_img = summed_img / nb_valid_img

    # Compute the number of pixel per kernel
    nb_pixel_per_kernel = np.count_nonzero(kernel)

    return mean_img, nb_valid_img, nb_pixel_per_kernel


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

    :param values: Values as array of shape (N1, N2) with NaN for masked values
    :param gsd: Ground sampling distance
    :param area: Size of integration area (squared unit of ground sampling distance)
    :param perc_min_valid: Minimum valid area in the patch
    :param patch_shape: Shape of patch, either "circular" or "square"
    :param method: Method to perform the convolution, "scipy" or "numba"
    :param statistic_between_patches: Statistic to compute between all patches, typically a measure of spread, applied
        to the first in-patch statistic, which is typically the mean
    :param return_in_patch_statistics: Whether to return the dataframe of statistics for all patches and areas


    :return: Statistic between patches, Number of patches, Exact discretized area, (Optional) Dataframe of per-patch
        statistics
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
    mean_img, nb_valid_img, nb_pixel_per_kernel = mean_filter_nan(
        img=values, kernel_size=kernel_size, kernel_shape=patch_shape, method=method
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
    Patches method for empirical estimation of the standard error over an integration area


    :param values: Values as array of shape (N1, N2) with NaN for masked values
    :param gsd: Ground sampling distance
    :param area: Size of integration area (squared unit of ground sampling distance)
    :param perc_min_valid: Minimum valid area in the patch
    :param statistics_in_patch: List of statistics to compute in each patch (count is computed by default; only the
    first statistic is used by statistic_between_patches)
    :param statistic_between_patches: Statistic to compute between all patches, typically a measure of spread, applied
        to the first in-patch statistic, which is typically the mean
    :param patch_shape: Shape of patch, either "circular" or "square".
    :param n_patches: Maximum number of patches to sample
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)
    :param return_in_patch_statistics: Whether to return the dataframe of statistics for all patches and areas

    :return: Statistic between patches, Number of patches, Exact discretized area, Dataframe of per-patch statistics
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

    # Get values with NaNs on unstable terrain, preserving the shape by default
    values_arr, gsd = _preprocess_values_with_mask_to_array(
        values=values, include_mask=stable_mask, exclude_mask=unstable_mask, gsd=gsd
    )

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


def plot_variogram(
    df: pd.DataFrame,
    list_fit_fun: list[Callable[[NDArrayf], NDArrayf]] = None,
    list_fit_fun_label: list[str] = None,
    ax: matplotlib.axes.Axes = None,
    xscale: str = "linear",
    xscale_range_split: list[float] = None,
    xlabel: str = None,
    ylabel: str = None,
    xlim: str = None,
    ylim: str = None,
    out_fname: str = None,
) -> None:
    """
    Plot empirical variogram, and optionally also plot one or several model fits.
    Input dataframe is expected to be the output of xdem.spatialstats.sample_empirical_variogram.
    Input function model is expected to be the output of xdem.spatialstats.fit_sum_model_variogram.

    :param df: Empirical variogram, formatted as a dataframe with count (pairwise sample count), lags
        (upper bound of spatial lag bin), exp (experimental variance), and err_exp (error on experimental variance)
    :param list_fit_fun: List of model function fits
    :param list_fit_fun_label: List of model function fits labels
    :param ax: Plotting ax to use, creates a new one by default
    :param xscale: Scale of X-axis
    :param xscale_range_split: List of ranges at which to split the figure
    :param xlabel: Label of X-axis
    :param ylabel: Label of Y-axis
    :param xlim: Limits of X-axis
    :param ylim: Limits of Y-axis
    :param out_fname: File to save the variogram plot to
    :return:
    """

    # Create axes if they are not passed
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None")

    # Check format of input dataframe
    expected_values = ["exp", "lags", "count"]
    for val in expected_values:
        if val not in df.columns.values:
            raise ValueError(f'The expected variable "{val}" is not part of the provided dataframe column names.')

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    if ylabel is None:
        ylabel = r"Variance [$\mu$ $\pm \sigma$]"
    if xlabel is None:
        xlabel = "Spatial lag (m)"

    init_gridsize = [10, 10]
    # Create parameters to split x axis into different linear scales
    # If there is no split, get parameters for a single subplot
    if xscale_range_split is None:
        nb_subpanels = 1
        if xscale == "log":
            xmin = [np.min(df.lags) / 2]
        else:
            xmin = [0]
        xmax = [np.max(df.lags)]
        xgridmin = [0]
        xgridmax = [init_gridsize[0]]
        gridsize = init_gridsize
    # Otherwise, derive a list for each subplot
    else:
        # Add initial zero if not in input
        if xscale_range_split[0] != 0:
            if xscale == "log":
                first_xmin = np.min(df.lags) / 2
            else:
                first_xmin = 0
            xscale_range_split = [first_xmin] + xscale_range_split
        # Add maximum distance if not in input
        if xscale_range_split[-1] != np.max(df.lags):
            xscale_range_split.append(np.max(df.lags))

        # Scale grid size by the number of subpanels
        nb_subpanels = len(xscale_range_split) - 1
        gridsize = init_gridsize.copy()
        gridsize[0] *= nb_subpanels
        # Create list of parameters to pass to ax/grid objects of subpanels
        xmin = []
        xmax = []
        xgridmin = []
        xgridmax = []
        for i in range(nb_subpanels):
            xmin.append(xscale_range_split[i])
            xmax.append(xscale_range_split[i + 1])
            xgridmin.append(init_gridsize[0] * i)
            xgridmax.append(init_gridsize[0] * (i + 1))

    # Need a grid plot to show the sample count and the statistic
    grid = plt.GridSpec(gridsize[1], gridsize[0], wspace=0.5, hspace=0.5)

    # Loop over each subpanel
    for k in range(nb_subpanels):
        # First, an axis to plot the sample histogram
        ax0 = ax.inset_axes(grid[:3, xgridmin[k] : xgridmax[k]].get_position(fig).bounds)
        ax0.set_xscale(xscale)
        ax0.set_xticks([])

        # Plot the histogram manually with fill_between
        interval_var = [0] + list(df.lags)
        for i in range(len(df)):
            count = df["count"].values[i]
            ax0.fill_between(
                [interval_var[i], interval_var[i + 1]],
                [0] * 2,
                [count] * 2,
                facecolor=plt.cm.Greys(0.75),
                alpha=1,
                edgecolor="white",
                linewidth=0.5,
            )
        if k == 0:
            ax0.set_ylabel("Sample count")
            # Scientific format to avoid undesired additional space on the label side
            ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            ax0.set_yticks([])
        # Ignore warnings for log scales
        ax0.set_xlim((xmin[k], xmax[k]))

        # Now, plot the statistic of the data
        ax1 = ax.inset_axes(grid[3:, xgridmin[k] : xgridmax[k]].get_position(fig).bounds)

        # Get the lags bin centers
        bins_center = np.subtract(df.lags, np.diff([0] + df.lags.tolist()) / 2)

        # If all the estimated errors are all NaN (single run), simply plot the empirical variogram
        if np.all(np.isnan(df.err_exp)):
            ax1.scatter(bins_center, df.exp, label="Empirical variogram", color="blue", marker="x")
        # Otherwise, plot the error estimates through multiple runs
        else:
            ax1.errorbar(bins_center, df.exp, yerr=df.err_exp, label="Empirical variogram (1-sigma std error)", fmt="x")

        # If a list of functions is passed, plot the modelled variograms
        if list_fit_fun is not None:
            for i, fit_fun in enumerate(list_fit_fun):
                x = np.linspace(xmin[k], xmax[k], 1000)
                y = fit_fun(x)

                if list_fit_fun_label is not None:
                    ax1.plot(x, y, linestyle="dashed", label=list_fit_fun_label[i], zorder=30)
                else:
                    ax1.plot(x, y, linestyle="dashed", color="black", zorder=30)

            if list_fit_fun_label is None:
                ax1.plot([], [], linestyle="dashed", color="black", label="Model fit")

        ax1.set_xscale(xscale)
        if nb_subpanels > 1 and k == (nb_subpanels - 1):
            ax1.xaxis.set_ticks(np.linspace(xmin[k], xmax[k], 3))
        elif nb_subpanels > 1:
            ax1.xaxis.set_ticks(np.linspace(xmin[k], xmax[k], 3)[:-1])

        if xlim is None:
            ax1.set_xlim((xmin[k], xmax[k]))
        else:
            ax1.set_xlim(xlim)

        if ylim is not None:
            ax1.set_ylim(ylim)
        else:
            if np.all(np.isnan(df.err_exp)):
                ax1.set_ylim((0, 1.05 * np.nanmax(df.exp)))
            else:
                ax1.set_ylim((0, np.nanmax(df.exp) + np.nanmean(df.err_exp)))

        if k == int(nb_subpanels / 2):
            ax1.set_xlabel(xlabel)
        if k == nb_subpanels - 1:
            ax1.legend(loc="lower right")
        if k == 0:
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_yticks([])

    if out_fname is not None:
        plt.savefig(out_fname)


def plot_1d_binning(
    df: pd.DataFrame,
    var_name: str,
    statistic_name: str,
    label_var: str | None = None,
    label_statistic: str | None = None,
    min_count: int = 30,
    ax: matplotlib.axes.Axes | None = None,
    out_fname: str = None,
) -> None:
    """
    Plot a statistic and its count along a single binning variable.
    Input is expected to be formatted as the output of the xdem.spatialstats.nd_binning function.

    :param df: Output dataframe of nd_binning
    :param var_name: Name of binning variable to plot
    :param statistic_name: Name of statistic of interest to plot
    :param label_var: Label of binning variable
    :param label_statistic: Label of statistic of interest
    :param min_count: Removes statistic values computed with a count inferior to this minimum value
    :param ax: Plotting ax to use, creates a new one by default
    :param out_fname: File to save the variogram plot to
    """

    # Create axes
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None.")

    if var_name not in df.columns.values:
        raise ValueError(f'The variable "{var_name}" is not part of the provided dataframe column names.')

    if statistic_name not in df.columns.values:
        raise ValueError(f'The statistic "{statistic_name}" is not part of the provided dataframe column names.')

    # Re-format pandas interval if read from CSV as string
    if any(isinstance(x, pd.Interval) for x in df[var_name].values):
        pass
    # Check for any unformatted interval (saving and reading a pd.DataFrame without MultiIndexing transforms
    # pd.Interval into strings)
    elif any(isinstance(_pandas_str_to_interval(x), pd.Interval) for x in df[var_name].values):
        intervalindex_vals = [_pandas_str_to_interval(x) for x in df[var_name].values]
        df[var_name] = pd.IntervalIndex(intervalindex_vals)
    else:
        raise ValueError("The variable columns must be provided as string or pd.Interval values.")

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    if label_var is None:
        label_var = var_name
    if label_statistic is None:
        label_statistic = statistic_name

    # Subsample to 1D and for the variable of interest
    df_sub = df[np.logical_and(df.nd == 1, np.isfinite(pd.IntervalIndex(df[var_name]).mid))].copy()
    # Remove statistic calculated in bins with too low count
    df_sub.loc[df_sub["count"] < min_count, statistic_name] = np.nan

    # Need a grid plot to show the sample count and the statistic
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)

    # First, an axis to plot the sample histogram
    ax0 = ax.inset_axes(grid[:3, :].get_position(fig).bounds)
    ax0.set_xticks([])

    # Plot the histogram manually with fill_between
    interval_var = pd.IntervalIndex(df_sub[var_name])
    for i in range(len(df_sub)):
        count = df_sub["count"].values[i]
        ax0.fill_between(
            [interval_var[i].left, interval_var[i].right],
            [0] * 2,
            [count] * 2,
            facecolor=plt.cm.Greys(0.75),
            alpha=1,
            edgecolor="white",
            linewidth=0.5,
        )
    ax0.set_ylabel("Sample count")
    # Scientific format to avoid undesired additional space on the label side
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Try to identify if the count is always the same
    # (np.quantile can have a couple undesired effet, so leave an error margin of 2 wrong bins and 5 count difference)
    if np.sum(~(np.abs(df_sub["count"].values[0] - df_sub["count"].values) < 5)) <= 2:
        ax0.text(
            0.5,
            0.5,
            "Fixed number of\n samples: " + "{:,}".format(int(df_sub["count"].values[0])),
            ha="center",
            va="center",
            fontweight="bold",
            transform=ax0.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    ax0.set_ylim((0, 1.1 * np.max(df_sub["count"].values)))
    ax0.set_xlim((np.min(interval_var.left), np.max(interval_var.right)))

    # Now, plot the statistic of the data
    ax1 = ax.inset_axes(grid[3:, :].get_position(fig).bounds)
    ax1.scatter(interval_var.mid, df_sub[statistic_name], marker="x")
    ax1.set_xlabel(label_var)
    ax1.set_ylabel(label_statistic)
    ax1.set_xlim((np.min(interval_var.left), np.max(interval_var.right)))

    if out_fname is not None:
        plt.savefig(out_fname)


def plot_2d_binning(
    df: pd.DataFrame,
    var_name_1: str,
    var_name_2: str,
    statistic_name: str,
    label_var_name_1: str | None = None,
    label_var_name_2: str | None = None,
    label_statistic: str | None = None,
    cmap: matplotlib.colors.Colormap = plt.cm.Reds,
    min_count: int = 30,
    scale_var_1: str = "linear",
    scale_var_2: str = "linear",
    vmin: np.floating[Any] = None,
    vmax: np.floating[Any] = None,
    nodata_color: str | tuple[float, float, float, float] = "yellow",
    ax: matplotlib.axes.Axes | None = None,
    out_fname: str = None,
) -> None:
    """
    Plot one statistic and its count along two binning variables.
    Input is expected to be formatted as the output of the xdem.spatialstats.nd_binning function.

    :param df: Output dataframe of nd_binning
    :param var_name_1: Name of first binning variable to plot
    :param var_name_2: Name of second binning variable to plot
    :param statistic_name: Name of statistic of interest to plot
    :param label_var_name_1: Label of first binning variable
    :param label_var_name_2: Label of second binning variable
    :param label_statistic: Label of statistic of interest
    :param cmap: Colormap
    :param min_count: Removes statistic values computed with a count inferior to this minimum value
    :param scale_var_1: Scale along the axis of the first variable
    :param scale_var_2: Scale along the axis of the second variable
    :param vmin: Minimum statistic value in colormap range
    :param vmax: Maximum statistic value in colormap range
    :param nodata_color: Color for no data bins
    :param ax: Plotting ax to use, creates a new one by default
    :param out_fname: File to save the variogram plot to
    """

    # Create axes
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None.")

    if var_name_1 not in df.columns.values:
        raise ValueError(f'The variable "{var_name_1}" is not part of the provided dataframe column names.')
    elif var_name_2 not in df.columns.values:
        raise ValueError(f'The variable "{var_name_2}" is not part of the provided dataframe column names.')

    if statistic_name not in df.columns.values:
        raise ValueError(f'The statistic "{statistic_name}" is not part of the provided dataframe column names.')

    # Re-format pandas interval if read from CSV as string
    for var in [var_name_1, var_name_2]:
        if any(isinstance(x, pd.Interval) for x in df[var].values):
            pass
        # Check for any unformatted interval (saving and reading a pd.DataFrame without MultiIndexing transforms
        # pd.Interval into strings)
        elif any(isinstance(_pandas_str_to_interval(x), pd.Interval) for x in df[var].values):
            intervalindex_vals = [_pandas_str_to_interval(x) for x in df[var].values]
            df[var] = pd.IntervalIndex(intervalindex_vals)
        else:
            raise ValueError("The variable columns must be provided as string or pd.Interval values.")

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    # Subsample to 2D and for the variables of interest
    df_sub = df[
        np.logical_and.reduce(
            (
                df.nd == 2,
                np.isfinite(pd.IntervalIndex(df[var_name_1]).mid),
                np.isfinite(pd.IntervalIndex(df[var_name_2]).mid),
            )
        )
    ].copy()
    # Remove statistic calculated in bins with too low count
    df_sub.loc[df_sub["count"] < min_count, statistic_name] = np.nan

    # Let's do a 4 panel figure:
    # two histograms for the binning variables
    # + a colored grid to display the statistic calculated on the value of interest
    # + a legend panel with statistic colormap and nodata color

    # For some reason the scientific notation displays weirdly for default figure size
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)

    # First, an horizontal axis on top to plot the sample histogram of the first variable
    ax0 = ax.inset_axes(grid[:3, :-3].get_position(fig).bounds)
    ax0.set_xscale(scale_var_1)
    ax0.set_xticklabels([])

    # Plot the histogram manually with fill_between
    interval_var_1 = pd.IntervalIndex(df_sub[var_name_1])
    df_sub["var1_mid"] = interval_var_1.mid.values
    unique_var_1 = np.unique(df_sub.var1_mid)
    list_counts = []
    for i in range(len(unique_var_1)):
        df_var1 = df_sub[df_sub.var1_mid == unique_var_1[i]]
        count = np.nansum(df_var1["count"].values)
        list_counts.append(count)
        ax0.fill_between(
            [df_var1[var_name_1].values[0].left, df_var1[var_name_1].values[0].right],
            [0] * 2,
            [count] * 2,
            facecolor=plt.cm.Greys(0.75),
            alpha=1,
            edgecolor="white",
            linewidth=0.5,
        )
    ax0.set_ylabel("Sample count")
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax0.set_ylim((0, 1.1 * np.max(list_counts)))
        ax0.set_xlim((np.min(interval_var_1.left), np.max(interval_var_1.right)))
    ax0.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    # Try to identify if the count is always the same
    if np.sum(~(np.abs(list_counts[0] - np.array(list_counts)) < 5)) <= 2:
        ax0.text(
            0.5,
            0.5,
            "Fixed number of\nsamples: " + f"{int(list_counts[0]):,}",
            ha="center",
            va="center",
            fontweight="bold",
            transform=ax0.transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # Second, a vertical axis on the right to plot the sample histogram of the second variable
    ax1 = ax.inset_axes(grid[3:, -3:].get_position(fig).bounds)
    ax1.set_yscale(scale_var_2)
    ax1.set_yticklabels([])

    # Plot the histogram manually with fill_between
    interval_var_2 = pd.IntervalIndex(df_sub[var_name_2])
    df_sub["var2_mid"] = interval_var_2.mid.values
    unique_var_2 = np.unique(df_sub.var2_mid)
    list_counts = []
    for i in range(len(unique_var_2)):
        df_var2 = df_sub[df_sub.var2_mid == unique_var_2[i]]
        count = np.nansum(df_var2["count"].values)
        list_counts.append(count)
        ax1.fill_between(
            [0, count],
            [df_var2[var_name_2].values[0].left] * 2,
            [df_var2[var_name_2].values[0].right] * 2,
            facecolor=plt.cm.Greys(0.75),
            alpha=1,
            edgecolor="white",
            linewidth=0.5,
        )
    ax1.set_xlabel("Sample count")
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax1.set_xlim((0, 1.1 * np.max(list_counts)))
        ax1.set_ylim((np.min(interval_var_2.left), np.max(interval_var_2.right)))
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # Try to identify if the count is always the same
    if np.sum(~(np.abs(list_counts[0] - np.array(list_counts)) < 5)) <= 2:
        ax1.text(
            0.5,
            0.5,
            "Fixed number of\nsamples: " + f"{int(list_counts[0]):,}",
            ha="center",
            va="center",
            fontweight="bold",
            transform=ax1.transAxes,
            rotation=90,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    # Third, an axis to plot the data as a colored grid
    ax2 = ax.inset_axes(grid[3:, :-3].get_position(fig).bounds)

    # Define limits of colormap is none are provided, robust max and min using percentiles
    if vmin is None and vmax is None:
        vmax = np.nanpercentile(df_sub[statistic_name].values, 99)
        vmin = np.nanpercentile(df_sub[statistic_name].values, 1)

    # Create custom colormap
    col_bounds = np.array([vmin, np.mean(np.asarray([vmin, vmax])), vmax])
    cb = []
    cb_val = np.linspace(0, 1, len(col_bounds))
    for j in range(len(cb_val)):
        cb.append(cmap(cb_val[j]))
    cmap_cus = colors.LinearSegmentedColormap.from_list(
        "my_cb", list(zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000
    )

    # Plot a 2D colored grid using fill_between
    for i in range(len(unique_var_1)):
        for j in range(len(unique_var_2)):
            df_both = df_sub[np.logical_and(df_sub.var1_mid == unique_var_1[i], df_sub.var2_mid == unique_var_2[j])]

            stat = df_both[statistic_name].values[0]
            if np.isfinite(stat):
                stat_col = max(0.0001, min(0.9999, (stat - min(col_bounds)) / (max(col_bounds) - min(col_bounds))))
                col = cmap_cus(stat_col)
            else:
                col = nodata_color

            ax2.fill_between(
                [df_both[var_name_1].values[0].left, df_both[var_name_1].values[0].right],
                [df_both[var_name_2].values[0].left] * 2,
                [df_both[var_name_2].values[0].right] * 2,
                facecolor=col,
                alpha=1,
                edgecolor="white",
            )

    ax2.set_xlabel(label_var_name_1)
    ax2.set_ylabel(label_var_name_2)
    ax2.set_xscale(scale_var_1)
    ax2.set_yscale(scale_var_2)
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax2.set_xlim((np.min(interval_var_1.left), np.max(interval_var_1.right)))
        ax2.set_ylim((np.min(interval_var_2.left), np.max(interval_var_2.right)))

    # Fourth and finally, add a colormap and nodata color to the legend
    axcmap = ax.inset_axes(grid[:3, -3:].get_position(fig).bounds)

    # Remove ticks, labels, frame
    axcmap.set_xticks([])
    axcmap.set_yticks([])
    axcmap.spines["top"].set_visible(False)
    axcmap.spines["left"].set_visible(False)
    axcmap.spines["right"].set_visible(False)
    axcmap.spines["bottom"].set_visible(False)

    # Create an inset axis to manage the scale of the colormap
    cbaxes = axcmap.inset_axes([0, 0.75, 1, 0.2], label="cmap")

    # Create colormap object and plot
    norm = colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
    sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbaxes, orientation="horizontal", extend="both", shrink=0.8)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label(label_statistic)

    # Create an inset axis to manage the scale of the nodata legend
    nodata = axcmap.inset_axes([0.4, 0.1, 0.2, 0.2], label="nodata")

    # Plot a nodata legend
    nodata.fill_between([0, 1], [0, 0], [1, 1], facecolor=nodata_color)
    nodata.set_xlim((0, 1))
    nodata.set_ylim((0, 1))
    nodata.set_xticks([])
    nodata.set_yticks([])
    nodata.text(0.5, -0.25, "No data", ha="center", va="top")

    if out_fname is not None:
        plt.savefig(out_fname)

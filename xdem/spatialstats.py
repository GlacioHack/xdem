"""Spatial statistical tools to estimate uncertainties related to DEMs"""
from __future__ import annotations

import inspect
import math as m
import multiprocessing as mp
import os
import warnings
from functools import partial

from typing import Callable, Union, Iterable, Optional, Sequence, Any, overload

import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numba import njit
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import integrate
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
from skimage.draw import disk
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, griddata
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
from geoutils.spatial_tools import subsample_raster, get_array_and_mask
from geoutils.georaster import RasterType, Raster
from geoutils.geovector import VectorType, Vector

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import skgstat as skg
    from skgstat import models

def nmad(data: np.ndarray, nfact: float = 1.4826) -> float:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.
    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. Höhle and Höhle (2009), http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    :param data: Input data
    :param nfact: Normalization factor for the data

    :returns nmad: (normalized) median absolute deviation of data.
    """
    if isinstance(data, np.ma.masked_array):
        data_arr = get_array_and_mask(data, check_shape=False)[0]
    else:
        data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))


def nd_binning(values: np.ndarray, list_var: Iterable[np.ndarray], list_var_names=Iterable[str], list_var_bins: Optional[Union[int,Iterable[Iterable]]] = None,
                     statistics: Iterable[Union[str, Callable, None]] = ['count', np.nanmedian, nmad], list_ranges : Optional[Iterable[Sequence]] = None) \
        -> pd.DataFrame:
    """
    N-dimensional binning of values according to one or several explanatory variables with computed statistics in
    each bin. By default, the sample count, the median and the normalized absolute median deviation (NMAD). The count
    is always computed, no matter user input.
    Values input is a (N,) array and variable input is a L-sized list of flattened arrays of similar dimensions (N,).
    For more details on the format of input variables, see documentation of scipy.stats.binned_statistic_dd.

    :param values: Values array of size (N,)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param list_var_names: List of size (L) of names of the explanatory variables
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory variables; defaults to 10 bins
    :param statistics: List of size (X) of statistics to be computed; defaults to count, median and nmad
    :param list_ranges: List of size (L) of minimum and maximum ranges to bin the explanatory variables; defaults to min/max of the data
    :return:
    """

    # We separate 1d, 2d and nd binning, because propagating statistics between different dimensional binning is not always feasible
    # using scipy because it allows for several dimensional binning, while it's not straightforward in pandas
    if list_var_bins is None:
        list_var_bins = (10,) * len(list_var_names)
    elif isinstance(list_var_bins, (int, np.integer)):
        list_var_bins = (list_var_bins,) * len(list_var_names)

    # Flatten the arrays if this has not been done by the user
    values = values.ravel()
    list_var = [var.ravel() for var in list_var]

    # Remove no data values
    valid_data = np.logical_and.reduce([np.isfinite(values)]+[np.isfinite(var) for var in list_var])
    values = values[valid_data]
    list_var = [var[valid_data] for var in list_var]

    statistics = list(statistics)
    # In case the statistics are user-defined, and they forget count, we add it for later calculation or plotting
    if 'count' not in statistics:
        statistics = ['count'] + statistics

    statistics_name = [f if isinstance(f, str) else f.__name__ for f in statistics]

    # Get binned statistics in 1d: a simple loop is sufficient
    list_df_1d = []
    for i, var in enumerate(list_var):
        df_stats_1d = pd.DataFrame()
        # Get statistics
        for j, statistic in enumerate(statistics):
            stats_binned_1d, bedges_1d = binned_statistic(var,values,statistic=statistic,bins=list_var_bins[i],range=list_ranges)[:2]
            # Save in a dataframe
            df_stats_1d[statistics_name[j]] = stats_binned_1d
        # We need to get the middle of the bins from the edges, to get the same dimension length
        df_stats_1d[list_var_names[i]] = pd.IntervalIndex.from_breaks(bedges_1d,closed='left')
        # Report number of dimensions used
        df_stats_1d.insert(0, 'nd', 1)

        list_df_1d.append(df_stats_1d)

    # Get binned statistics in 2d: all possible 2d combinations
    list_df_2d = []
    if len(list_var)>1:
        combs = list(itertools.combinations(list_var_names, 2))
        for i, comb in enumerate(combs):
            var1_name, var2_name = comb
            # Corresponding variables indexes
            i1, i2 = list_var_names.index(var1_name), list_var_names.index(var2_name)
            df_stats_2d = pd.DataFrame()
            for j, statistic in enumerate(statistics):
                stats_binned_2d, bedges_var1, bedges_var2 = binned_statistic_2d(list_var[i1],list_var[i2],values,statistic=statistic
                                                             ,bins=[list_var_bins[i1],list_var_bins[i2]]
                                                             ,range=list_ranges)[:3]
                # Get statistics
                df_stats_2d[statistics_name[j]] = stats_binned_2d.flatten()
            # Derive interval indexes and convert bins into 2d indexes
            ii1 = pd.IntervalIndex.from_breaks(bedges_var1,closed='left')
            ii2 = pd.IntervalIndex.from_breaks(bedges_var2,closed='left')
            df_stats_2d[var1_name] = [i1 for i1 in ii1 for i2 in ii2]
            df_stats_2d[var2_name] = [i2 for i1 in ii1 for i2 in ii2]
            # Report number of dimensions used
            df_stats_2d.insert(0, 'nd', 2)

            list_df_2d.append(df_stats_2d)


    # Get binned statistics in nd, without redoing the same stats
    df_stats_nd = pd.DataFrame()
    if len(list_var)>2:
        for j, statistic in enumerate(statistics):
            stats_binned_2d, list_bedges = binned_statistic_dd(list_var,values,statistic=statistic,bins=list_var_bins,range=list_ranges)[0:2]
            df_stats_nd[statistics_name[j]] = stats_binned_2d.flatten()
        list_ii = []
        # Loop through the bin edges and create IntervalIndexes from them (to get both
        for bedges in list_bedges:
            list_ii.append(pd.IntervalIndex.from_breaks(bedges,closed='left'))

        # Create nd indexes in nd-array and flatten for each variable
        iind = np.meshgrid(*list_ii)
        for i, var_name in enumerate(list_var_names):
            df_stats_nd[var_name] = iind[i].flatten()

        # Report number of dimensions used
        df_stats_nd.insert(0, 'nd', len(list_var_names))

    # Concatenate everything
    list_all_dfs = list_df_1d + list_df_2d + [df_stats_nd]
    df_concat = pd.concat(list_all_dfs)
    # commenting for now: pd.MultiIndex can be hard to use
    # df_concat = df_concat.set_index(list_var_names)

    return df_concat


def interp_nd_binning(df: pd.DataFrame, list_var_names: Union[str, Iterable[str]], statistic : Union[str, Callable[[np.ndarray],float]] = nmad,
                      min_count: Optional[int] = 100) -> Callable[[tuple[np.ndarray, ...]], np.ndarray]:
    """
    Estimate an interpolant function for an N-dimensional binning. Preferably based on the output of nd_binning.
    For more details on the input dataframe, and associated list of variable name and statistic, see nd_binning.

    If the variable pd.DataSeries corresponds to an interval (as the output of nd_binning), uses the middle of the interval.
    Otherwise, uses the variable as such.

    Workflow of the function:
    Fills the no-data present on the regular N-D binning grid with nearest neighbour from scipy.griddata, then provides an
    interpolant function that linearly interpolates/extrapolates using scipy.RegularGridInterpolator.

    :param df: Dataframe with statistic of binned values according to explanatory variables (preferably output of nd_binning)
    :param list_var_names: Explanatory variable data series to select from the dataframe (containing interval or float dtype)
    :param statistic: Statistic to interpolate, stored as a data series in the dataframe
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata)
    :return: N-dimensional interpolant function

    :examples
    # Using a dataframe created from scratch
    >>> df = pd.DataFrame({"var1": [1, 2, 3, 1, 2, 3, 1, 2, 3], "var2": [1, 1, 1, 2, 2, 2, 3, 3, 3], "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9]})

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

    # Extrapolated linearly outside the 2D frame.
    >>> fun((-1, 1))
    array(-1.)
    """
    # If list of variable input is simply a string
    if isinstance(list_var_names, str):
        list_var_names = [list_var_names]

    # Check that the dataframe contains what we need
    for var in list_var_names:
        if var not in df.columns:
            raise ValueError('Variable "'+var+'" does not exist in the provided dataframe.')
    statistic_name = statistic if isinstance(statistic, str) else statistic.__name__
    if statistic_name not in df.columns:
        raise ValueError('Statistic "' + statistic_name + '" does not exist in the provided dataframe.')
    if min_count is not None and 'count' not in df.columns:
        raise ValueError('Statistic "count" is not in the provided dataframe, necessary to use the min_count argument.')
    if df.empty:
        raise ValueError('Dataframe is empty.')

    df_sub = df.copy()

    # If the dataframe is an output of nd_binning, keep only the dimension of interest
    if 'nd' in df_sub.columns:
        df_sub = df_sub[df_sub.nd == len(list_var_names)]

    # Compute the middle values instead of bin interval if the variable is a pandas interval type
    for var in list_var_names:
        check_any_interval = [isinstance(x, pd.Interval) for x in df_sub[var].values]
        if any(check_any_interval):
            df_sub[var] = pd.IntervalIndex(df_sub[var]).mid.values
        # Otherwise, leave as is

    # Check that explanatory variables have valid binning values which coincide along the dataframe
    df_sub = df_sub[np.logical_and.reduce([np.isfinite(df_sub[var].values) for var in list_var_names])]
    if df_sub.empty:
        raise ValueError('Dataframe does not contain a nd binning with the variables corresponding to the list of variables.')
    # Check that the statistic data series contain valid data
    if all(~np.isfinite(df_sub[statistic_name].values)):
        raise ValueError('Dataframe does not contain any valid statistic values.')

    # Remove statistic values calculated with a sample count under the minimum count
    if min_count is not None:
        df_sub.loc[df_sub['count'] < min_count, statistic_name] = np.nan

    values = df_sub[statistic_name].values
    ind_valid = np.isfinite(values)

    # Re-check that the statistic data series contain valid data after filtering with min_count
    if all(~ind_valid):
        raise ValueError("Dataframe does not contain any valid statistic values after filtering with min_count = "+str(min_count)+".")

    # Get a list of middle values for the binning coordinates, to define a nd grid
    list_bmid = []
    shape = []
    for var in list_var_names:
        bmid = sorted(np.unique(df_sub[var][ind_valid]))
        list_bmid.append(bmid)
        shape.append(len(bmid))

    # Use griddata first to perform nearest interpolation with NaNs (irregular grid)
    # Valid values
    values = values[ind_valid]
    # coordinates of valid values
    points_valid = tuple([df_sub[var].values[ind_valid] for var in list_var_names])
    # Grid coordinates
    bmid_grid = np.meshgrid(*list_bmid, indexing='ij')
    points_grid = tuple([bmid_grid[i].flatten() for i in range(len(list_var_names))])
    # Fill grid no data with nearest neighbour
    values_grid = griddata(points_valid, values, points_grid, method='nearest')
    values_grid = values_grid.reshape(shape)

    # RegularGridInterpolator to perform linear interpolation/extrapolation on the grid
    # (will extrapolate only outside of boundaries not filled with the nearest of griddata as fill_value = None)
    interp_fun = RegularGridInterpolator(tuple(list_bmid), values_grid, method='linear', bounds_error=False, fill_value=None)

    return interp_fun


def two_step_standardization(dvalues: np.ndarray, list_var: Iterable[np.ndarray],
                unscaled_error_fun: Callable[[tuple[np.ndarray, ...]], np.ndarray],
                spread_statistic: Callable = nmad,
                fac_spread_outliers: float | None = 7
                ) -> tuple[np.ndarray, Callable[[tuple[np.ndarray, ...]], np.ndarray]]:
    """
    Standardize the proxy differenced values using the modelled heteroscedasticity, re-scaled to the spread statistic,
    and generate the final standardization function.

    :param dvalues: Proxy values as array of size (N,) (i.e., differenced values where signal should be zero such as elevation differences on stable terrain)
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
    def error_fun(*args):
        return zscore_nmad * unscaled_error_fun(*args)

    return zscores, error_fun


def estimate_model_heteroscedasticity(dvalues: np.ndarray, list_var: Iterable[np.ndarray], list_var_names: Iterable[str],
                                      spread_statistic: Callable = nmad,
                                      list_var_bins: Optional[Union[int,Iterable[Iterable]]] = None,
                                      min_count: Optional[int] = 100,
                                      fac_spread_outliers: float | None = 7
                                      ) -> tuple[pd.DataFrame, Callable[[tuple[np.ndarray, ...]], np.ndarray]]:
    """
    Estimate and model the heteroscedasticity (i.e., variability in error) according to a list of explanatory variables
    from a proxy of differenced values (e.g., elevation differences), if possible compared to a source of higher
    precision.

    This function performs N-D data binning with the list of explanatory variable for a spread statistic, then
    performs N-D interpolation on this statistic, scales the output with a two-step standardization to return an error
    function of the explanatory variables.

    The functions used are `nd_binning`, `interp_nd_binning` and `two_step_standardization`.

    :param dvalues: Proxy values as array of size (N,) (i.e., differenced values where signal should be zero such as elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables array of size (N,)
    :param list_var_names: List of size (L) of names of the explanatory variables
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory variables; defaults to 10 bins
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata)
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Dataframe of binned spread statistic with explanatory variables, Error function with explanatory variables
    """

    # Perform N-D binning with the differenced values computing the spread statistic
    df = nd_binning(values=dvalues, list_var=list_var, list_var_names=list_var_names, statistics=[spread_statistic],
                    list_var_bins=list_var_bins)

    # Perform N-D linear interpolation for the spread statistic
    fun = interp_nd_binning(df, list_var_names=list_var_names, statistic=spread_statistic.__name__, min_count=min_count)

    # Get the final function based on a two-step standardization
    final_fun = two_step_standardization(dvalues=dvalues, list_var=list_var, unscaled_error_fun=fun,
                                         spread_statistic=spread_statistic, fac_spread_outliers=fac_spread_outliers)[1]

    return df, final_fun


@overload
def infer_heteroscedasticity_from_stable(dvalues: np.ndarray, list_var: list[np.ndarray | RasterType],
                                         stable_mask: np.ndarray | VectorType | gpd.GeoDataFrame,
                                         unstable_mask: np.ndarray | VectorType | gpd.GeoDataFrame,
                                         list_var_names: Iterable[str],
                                         spread_statistic: Callable,
                                         list_var_bins: Optional[Union[int,Iterable[Iterable]]],
                                         min_count: Optional[int],
                                         factor_spread_exclude_outliers: float | None,
                                         ) -> tuple[np.ndarray,
                                            pd.DataFrame,
                                            Callable[[tuple[np.ndarray, ...]], np.ndarray]]: ...

@overload
def infer_heteroscedasticity_from_stable(dvalues: RasterType, list_var: list[np.ndarray | RasterType],
                                         stable_mask: np.ndarray | VectorType | gpd.GeoDataFrame,
                                         unstable_mask: np.ndarray | VectorType | gpd.GeoDataFrame,
                                         list_var_names: Iterable[str],
                                         spread_statistic: Callable,
                                         list_var_bins: Optional[Union[int,Iterable[Iterable]]],
                                         min_count: Optional[int],
                                         factor_spread_exclude_outliers: float | None,
                                         ) -> tuple[RasterType,
                                            pd.DataFrame,
                                            Callable[[tuple[np.ndarray, ...]], np.ndarray]]: ...

def infer_heteroscedasticity_from_stable(dvalues: np.ndarray | RasterType, list_var: list[np.ndarray | RasterType],
                                         stable_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                         unstable_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                         list_var_names: Iterable[str] = None,
                                         spread_statistic: Callable = nmad,
                                         list_var_bins: Optional[Union[int,Iterable[Iterable]]] = None,
                                         min_count: Optional[int] = 100,
                                         fac_spread_outliers: float | None = 7,
                                         ) -> tuple[np.ndarray | RasterType,
                                            pd.DataFrame,
                                            Callable[[tuple[np.ndarray, ...]], np.ndarray]]:
    """
    Infer heteroscedasticity from differenced values on stable terrain and a list of explanatory variables.

    This function returns an error map, a dataframe of spread values and the error function with explanatory variables.
    It is a convenience wrapper for `estimate_model_heteroscedasticity` to work on either Raster or array, compute the
    stable mask and return an error map.

    If no stable or unstable mask is provided to mask in or out the values, all terrain is used.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as elevation differences on stable terrain)
    :param list_var: List of size (L) of explanatory variables as array or Raster of same shape as dvalues
    :param stable_mask: Vector shapefile of stable terrain (if dvalues is Raster), or boolean array of same shape as dvalues
    :param unstable_mask: Vector shapefile of unstable terrain (if dvalues is Raster), or boolean array of same shape as dvalues
    :param list_var_names: List of size (L) of names of the explanatory variables, otherwise named var1, var2, etc.
    :param spread_statistic: Statistic to be computed for the spread; defaults to nmad
    :param list_var_bins: Count of size (1), or list of size (L) of counts or custom bin edges for the explanatory variables; defaults to 10 bins
    :param min_count: Minimum number of samples to be used as a valid statistic (replaced by nodata)
    :param fac_spread_outliers: Exclude outliers outside this spread after standardizing; pass None to ignore.

    :return: Inferred error map (array or Raster, same as input proxy values),
        Dataframe of binned spread statistic with explanatory variables,
        Error function with explanatory variables
    """

    # Check inputs
    if not isinstance(dvalues, (Raster, np.ndarray)):
        raise ValueError('The dvalues must be a Raster or NumPy array.')
    if stable_mask is not None and not isinstance(stable_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The stable mask must be a Vector, GeoDataFrame or NumPy array.')
    if unstable_mask is not None and not isinstance(unstable_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The unstable mask must be a Vector, GeoDataFrame or NumPy array.')

    # Check that input stable mask can only be a georeferenced vector if the proxy values are a Raster to project onto
    if not isinstance(dvalues, Raster) and (isinstance(stable_mask, (Vector, gpd.GeoDataFrame)) or isinstance(unstable_mask,  (Vector, gpd.GeoDataFrame))):
        raise ValueError('The stable mask can only passed as a Vector or GeoDataFrame if the input dvalues is a Raster.')

    # Create placeholder variables names if those don't exist
    if list_var_names is None:
        list_var_names = ['var'+str(i+1) for i in range(len(list_var))]

    # Get the arrays for proxy values and explanatory variables
    if isinstance(dvalues, Raster):
        dvalues_arr = get_array_and_mask(dvalues)[0]
    else:
        dvalues_arr = dvalues
    list_var_arr = [get_array_and_mask(var)[0] if isinstance(var, Raster) else var
                    for var in list_var if isinstance(var, Raster)]

    # If the stable mask is not an array, create it
    if stable_mask is None:
        stable_mask_arr = np.ones(np.shape(dvalues_arr), dtype=bool)
    elif not isinstance(stable_mask, np.ndarray):

        # If the stable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(stable_mask, gpd.GeoDataFrame):
            stable_vector = Vector(stable_mask)
        else:
            stable_vector = stable_mask

        # Create the mask
        stable_mask_arr = stable_vector.create_mask(dvalues)
    # If the mask is already an array, just pass it
    else:
        stable_mask_arr = stable_mask

    # If the unstable mask is not an array, create it
    if unstable_mask is None:
        unstable_mask_arr = np.zeros(np.shape(dvalues_arr), dtype=bool)
    elif not isinstance(unstable_mask, np.ndarray):

        # If the unstable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(unstable_mask, gpd.GeoDataFrame):
            unstable_vector = Vector(unstable_mask)
        else:
            unstable_vector = unstable_mask

        # Create the mask
        unstable_mask_arr = unstable_vector.create_mask(dvalues)
    # If the mask is already an array, just pass it
    else:
        unstable_mask_arr = unstable_mask

    stable_mask_arr = np.logical_and(stable_mask_arr, ~unstable_mask_arr).squeeze()

    # Get the subsets on stable terrain
    dvalues_stable_arr = dvalues_arr[stable_mask_arr]
    list_var_stable_arr = [var_arr[stable_mask_arr] for var_arr in list_var_arr]

    # Estimate and model the heteroscedasticity using only stable terrain
    df, fun = estimate_model_heteroscedasticity(dvalues=dvalues_stable_arr, list_var=list_var_stable_arr,
                                                list_var_names=list_var_names, spread_statistic=spread_statistic,
                                                list_var_bins=list_var_bins, min_count=min_count,
                                                fac_spread_outliers=fac_spread_outliers)

    # Use the standardization function to get the error array for the entire input array (not only stable)
    error = fun(tuple(list_var_arr))

    # Return the right type, depending on dvalues input
    if isinstance(dvalues, Raster):
        return dvalues.copy(new_array=error), df, fun
    else:
        return error, df, fun


def _create_circular_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None,
                         radius: Optional[float] = None) -> np.ndarray:
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

    # Skimage disk is not inclusive (correspond to distance_from_center < radius and not <= radius)
    mask = np.zeros(shape, dtype=bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
        rr, cc = disk(center=center,radius=radius,shape=shape)
    mask[rr, cc] = True

    # manual solution
    # Y, X = np.ogrid[:h, :w]
    # dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    # mask = dist_from_center < radius

    return mask

def _create_ring_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None, in_radius: float = 0.,
                     out_radius: Optional[float] = None) -> np.ndarray:
    """
    Create ring mask on a raster, defaults to the center of the array and a circle mask of half width of the array

    :param shape: shape of array
    :param center: center
    :param in_radius: inside radius
    :param out_radius: outside radius
    :return:
    """

    w, h = shape

    if out_radius is None:
        center = (int(w / 2), int(h / 2))
        out_radius = min(center[0], center[1], w - center[0], h - center[1])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in true_divide")
        mask_inside = _create_circular_mask((w,h),center=center,radius=in_radius)
        mask_outside = _create_circular_mask((w,h),center=center,radius=out_radius)

    mask_ring = np.logical_and(~mask_inside,mask_outside)

    return mask_ring


def _subsample_wrapper(values: np.ndarray, coords: np.ndarray, shape: tuple[int,int] = None, subsample: int = 10000,
                       subsample_method: str = 'pdist_ring', inside_radius = None, outside_radius = None,
                       random_state: None | np.random.RandomState | np.random.Generator | int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    (Not used by default)
    Wrapper for subsampling pdist methods
    """
    nx, ny = shape

    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Subsample spatially for disk/ring methods
    if subsample_method in ['pdist_disk', 'pdist_ring']:
        # Select random center coordinates
        center_x = rnd.choice(nx, 1)[0]
        center_y = rnd.choice(ny, 1)[0]
        if subsample_method == 'pdist_ring':
            subindex = _create_ring_mask((nx, ny), center=[center_x, center_y], in_radius=inside_radius,
                                          out_radius=outside_radius)
        else:
            subindex = _create_circular_mask((nx, ny), center=[center_x, center_y], radius=inside_radius)

        index = subindex.flatten()
        values_sp = values[index]
        coords_sp = coords[index, :]

    else:
        values_sp = values
        coords_sp = coords

    index = subsample_raster(values_sp, subsample=subsample, return_indices=True, random_state=rnd)
    values_sub = values_sp[index[0]]
    coords_sub = coords_sp[index[0], :]

    return values_sub, coords_sub

def _aggregate_pdist_empirical_variogram(values: np.ndarray, coords: np.ndarray, subsample: int, shape: tuple,
                                         subsample_method: str, gsd: float,
                                         pdist_multi_ranges: Optional[list[float]] = None, **kwargs) -> pd.DataFrame:
    """
    (Not used by default)
    Aggregating subfunction of sample_empirical_variogram for pdist methods.
    The pairwise differences are calculated within each subsample.
    """

    # If no multi_ranges are provided, define a logical default behaviour with the pixel size and grid size
    if subsample_method in ['pdist_disk', 'pdist_ring']:

        if pdist_multi_ranges is None:

            # Define list of ranges as exponent 2 of the resolution until the maximum range
            pdist_multi_ranges = []
            # We start at 10 times the ground sampling distance
            new_range = gsd * 10
            while new_range < kwargs.get('maxlag') / 2:
                pdist_multi_ranges.append(new_range)
                new_range *= 2
            pdist_multi_ranges.append(kwargs.get('maxlag'))


        # Define subsampling parameters
        list_inside_radius, list_outside_radius = ([] for i in range(2))
        binned_ranges = [0] + pdist_multi_ranges
        for i in range(len(binned_ranges) - 1):

            # Radiuses need to be passed as pixel sizes, dividing by ground sampling distance
            outside_radius = binned_ranges[i + 1]/gsd
            if subsample_method == 'pdist_ring':
                inside_radius = binned_ranges[i]/gsd
            else:
                inside_radius = None

            list_outside_radius.append(outside_radius)
            list_inside_radius.append(inside_radius)
    else:
        # For random point selection, no need for multi-range parameters
        pdist_multi_ranges = [kwargs.get('maxlag')]
        list_outside_radius = [None]
        list_inside_radius = [None]

    # Estimate variogram with specific subsampling at multiple ranges
    list_df_range = []
    for j in range(len(pdist_multi_ranges)):

        values_sub, coords_sub = _subsample_wrapper(values, coords, shape = shape, subsample = subsample,
                                                    subsample_method = subsample_method,
                                                    inside_radius = list_inside_radius[j],
                                                    outside_radius = list_outside_radius[j],
                                                    random_state= kwargs.get('random_state'))
        if len(values_sub) == 0:
            continue
        df_range = _get_pdist_empirical_variogram(values=values_sub, coords=coords_sub, **kwargs)

        # Aggregate runs
        list_df_range.append(df_range)

    df = pd.concat(list_df_range)

    return df


def _get_pdist_empirical_variogram(values: np.ndarray, coords: np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object calculating pairwise distances within the sample

    :param values: Values
    :param coords: Coordinates
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """

    # Remove random_state keyword argument that is not used
    kwargs.pop('random_state')

    # Get arguments of Variogram class init function
    variogram_args = skg.Variogram.__init__.__code__.co_varnames[:skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in variogram_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: '+','.join(list(remaining_kwargs.keys()))+ ' were not used.')
    # Filter corresponding arguments before passing
    filtered_kwargs =  {k:kwargs[k] for k in variogram_args if k in kwargs}

    # Derive variogram with default MetricSpace (equivalent to scipy.pdist)
    V = skg.Variogram(coordinates=coords, values=values, normalize=False, fit_method=None, **filtered_kwargs)

    # Get bins, empirical variogram values, and bin count
    bins, exp = V.get_empirical()
    count = V.bin_count

    # Write to dataframe
    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df

def _choose_cdist_equidistant_sampling_parameters(**kwargs):
    """
    Add a little calculation to partition the "subsample" argument automatically into the "run" and "samples"
    arguments of RasterEquidistantMetricSpace, to have a similar number of points than with a classic pdist method.

    The number of pairwise samples for a classic pdist is N0(N0-1)/2 with N0 the number of samples of the ensemble. For
    the cdist equidistant calculation it is M*N*R where N are the subsamples in the center disk, M is the number of
    samples in the rings which amounts to X*N where X is the number of rings in the grid extent, as each ring draws N
    samples. And R is the number of runs with a different random center point.
    X is fixed by the extent and ratio_subsample parameters, and so N0**2/(2X) = N**2*R, and we want at least 30 runs
    with 10 subsamples.
    """

    # First, we extract the extent, shape and subsample values from the keyword arguments
    extent = kwargs['extent']
    shape = kwargs['shape']
    subsample = kwargs['subsample']
    # We derive the maximum distance and resolution automatically derived in skgstat.RasterEquidistantMetricSpace
    maxdist = np.sqrt((extent[1] - extent[0]) ** 2 + (extent[3] - extent[2]) ** 2)
    res = np.mean([(extent[1] - extent[0]) / (shape[0] - 1), (extent[3] - extent[2]) / (shape[1] - 1)])
    # Then, we compute the radius from the center ensemble with the default value of subsample ratio in the function
    # skgstat.RasterEquidistantMetricSpace
    ratio_subsample = 0.2
    center_radius = np.sqrt(1. / ratio_subsample * subsample / np.pi) * res
    # Now, we can derive the number of successive disks that are going to be sampled in the grid
    equidistant_radii = [0.]
    increasing_rad = center_radius
    while increasing_rad < maxdist:
        equidistant_radii.append(increasing_rad)
        increasing_rad *= np.sqrt(2)
    nb_disk_samples = len(equidistant_radii)

    # We divide the number of samples by the number of disks
    pairwise_comp_per_disk = np.ceil(subsample**2 / (2*nb_disk_samples))

    # Using the equation in the function description, we compute the number of runs (minimum 30)
    runs = int(max(np.ceil(pairwise_comp_per_disk**(1/3)), 30))
    # Then we deduce the number of samples per disk (and per ring)
    subsample_per_disk_per_run = int(np.ceil(np.sqrt(pairwise_comp_per_disk/runs)))

    final_pairwise_comparisons = runs*subsample_per_disk_per_run**2*nb_disk_samples

    if kwargs['verbose']:
        print('Equidistant circular sampling will be performed for {} runs (random center points) with pairwise '
              'comparison between {} samples (points) of the central disk and again {} samples times {} independent '
              'rings centered on the same center point. This results in approximately {} pairwise comparisons (duplicate'
              ' pairwise points randomly selected will be removed).'.format(runs, subsample_per_disk_per_run,
                                                                            subsample_per_disk_per_run, nb_disk_samples,
                                                                            final_pairwise_comparisons))

    return runs, subsample_per_disk_per_run

def _get_cdist_empirical_variogram(values: np.ndarray, coords: np.ndarray, subsample_method: str,
                                   **kwargs) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object calculating pairwise distances between two sample collections
    of a MetricSpace (see scikit-gstat documentation for more details)

    :param values: Values
    :param coords: Coordinates
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """

    if subsample_method == 'cdist_equidistant' and 'runs' not in kwargs.keys() and 'samples' not in kwargs.keys():

        # We define subparameters for the equidistant technique to match the number of pairwise comparison
        # that would have a classic "subsample" with pdist, except if those parameters are already user-defined
        runs, samples = _choose_cdist_equidistant_sampling_parameters(**kwargs)

        kwargs['runs'] = runs
        # The "samples" argument is used by skgstat Metric subclasses (and not "subsample")
        kwargs['samples'] = samples
        kwargs.pop('subsample')

    elif subsample_method == 'cdist_point':

        # We set the samples to match the subsample argument if the method is random points
        kwargs['samples'] = kwargs.pop('subsample')

    # Rename the "random_state" argument into "rnd", also used by skgstat Metric subclasses
    kwargs['rnd'] = kwargs.pop('random_state')

    # Define MetricSpace function to be used, fetch possible keywords arguments
    if subsample_method == 'cdist_point':
        # List keyword arguments of the Probabilistic class init function
        ms_args = skg.ProbabalisticMetricSpace.__init__.__code__.co_varnames[:skg.ProbabalisticMetricSpace.__init__.__code__.co_argcount]
        ms = skg.ProbabalisticMetricSpace
    else:
        # List keyword arguments of the RasterEquidistant class init function
        ms_args = skg.RasterEquidistantMetricSpace.__init__.__code__.co_varnames[:skg.RasterEquidistantMetricSpace.__init__.__code__.co_argcount]
        ms = skg.RasterEquidistantMetricSpace

    # Get arguments of Variogram class init function
    variogram_args = skg.Variogram.__init__.__code__.co_varnames[:skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed, accounting for MetricSpace arguments
    remaining_kwargs = kwargs.copy()
    for arg in variogram_args + ms_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: ' + ', '.join(list(remaining_kwargs.keys())) + ' were not used.')

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


def _wrapper_get_empirical_variogram(argdict: dict) -> pd.DataFrame:
    """
    Multiprocessing wrapper for get_pdist_empirical_variogram and get_cdist_empirical variogram

    :param argdict: Keyword argument to pass to get_pdist/cdist_empirical_variogram
    :return: Empirical variogram (variance, upper bound of lag bin, counts)

    """
    if argdict['verbose']:
        print('Working on run '+str(argdict['i']) + ' out of '+str(argdict['imax']))
    argdict.pop('i')
    argdict.pop('imax')

    if argdict['subsample_method'] in ['cdist_equidistant', 'cdist_point']:
        # Simple wrapper for the skgstat Variogram function for cdist methods
        get_variogram = _get_cdist_empirical_variogram
    else:
        # Aggregating several skgstat Variogram after iterative subsampling of specific points in the Raster
        get_variogram = _aggregate_pdist_empirical_variogram

    return get_variogram(**argdict)


def sample_empirical_variogram(values: Union[np.ndarray, RasterType], gsd: float = None, coords: np.ndarray = None,
                               subsample: int = 1000, subsample_method: str = 'cdist_equidistant',
                               n_variograms: int = 1, n_jobs: int = 1, verbose = False,
                               random_state: None | np.random.RandomState | np.random.Generator | int = None,
                               **kwargs) -> pd.DataFrame:
    """
    Sample empirical variograms with binning adaptable to multiple ranges and spatial subsampling adapted for raster data.
    Returns an empirical variogram (empirical variance, upper bound of spatial lag bin, count of pairwise samples).

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
    automatically to get close to N*(N-1)/2 pairwise samples. But those can be more finely adjusted by passing the
    argument "runs", "samples" and "ratio_subsample" to kwargs. Further details can be found in the description of
    skgstat.MetricSpace.RasterEquidistantMetricSpace.

    If values are provided as a Raster subclass, nothing else is required.
    If values are provided as a 2D array (M,N), a ground sampling distance is sufficient to derive the pairwise distances.
    If values are provided as a 1D array (N), an array of coordinates (N,2) or (2,N) is expected. If the coordinates
    do not correspond to points of a grid, a ground sampling distance is needed to correctly get the grid size.

    Spatial subsampling method argument subsample_method can be one of "cdist_equidistant", "cdist_point", "pdist_point",
    "pdist_disk" and "pdist_ring".
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
    :param verbose: Print statements during processing
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Empirical variogram (variance, upper bound of lag bin, counts)
    """
    # First, check all that the values provided are OK
    if isinstance(values, Raster):
        gsd = values.res[0]
        values, mask = get_array_and_mask(values.data)
    elif isinstance(values, (np.ndarray, np.ma.masked_array)):
        values, mask = get_array_and_mask(values)
    else:
        raise ValueError('Values must be of type np.ndarray, np.ma.masked_array or Raster subclass.')
    values = values.squeeze()

    # Then, check if the logic between values, coords and gsd is respected
    if (gsd is not None or subsample_method in ['cdist_equidistant', 'pdist_disk','pdist_ring']) and values.ndim == 1:
        raise ValueError('Values array must be 2D when using any of the "cdist_equidistant", "pdist_disk" and '
                        '"pdist_ring" methods, or providing a ground sampling distance instead of coordinates.')
    elif coords is not None and values.ndim != 1:
        raise ValueError('Values array must be 1D when providing coordinates.')
    elif coords is not None and (coords.shape[0] != 2 and coords.shape[1] != 2):
        raise ValueError('The coordinates array must have one dimension with length equal to 2')
    elif values.ndim == 2 and gsd is None:
        raise ValueError('The ground sampling distance must be defined when passing a 2D values array.')

    # Check the subsample method provided exists, otherwise list options
    if subsample_method not in ['cdist_equidistant','cdist_point','pdist_point','pdist_disk','pdist_ring']:
        raise TypeError('The subsampling method must be one of "cdist_equidistant, "cdist_point", "pdist_point", '
                        '"pdist_disk" or "pdist_ring".')
    # Check that, for several runs, the binning function is an Iterable, otherwise skgstat might provide variogram
    # values over slightly different binnings due to randomly changing subsample maximum lags
    if n_variograms > 1 and 'bin_func' in kwargs.keys() and not isinstance(kwargs.get('bin_func'), Iterable):
        warnings.warn('Using a named binning function of scikit-gstat might provide different binnings for each '
                      'independent run. To remediate that issue, pass bin_func as an Iterable of right bin edges, '
                      '(or use default bin_func).')

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
    if 'maxlag' not in kwargs.keys():
        # We define maximum lag as the maximum distance between coordinates (needed to provide custom bins, otherwise
        # skgstat rewrites the maxlag with the subsample of coordinates provided)
        maxlag = np.sqrt((np.max(coords[:, 0])-np.min(coords[:, 1]))**2
                         + (np.max(coords[:, 1]) - np.min(coords[:, 1]))**2)
        kwargs.update({'maxlag': maxlag})

    # Keep only valid data for cdist methods, remove later for pdist methods
    if 'cdist' in subsample_method:
        ind_valid = np.isfinite(values)
        values = values[ind_valid]
        coords = coords[ind_valid, :]

    if 'bin_func' not in kwargs.keys():
        # If no bin_func is provided, we provide an Iterable to provide a custom binning function to skgstat,
        # because otherwise bins might be unconsistent across runs
        bin_func = []
        right_bin_edge = np.sqrt(2) * gsd
        while right_bin_edge < kwargs.get('maxlag'):
            bin_func.append(right_bin_edge)
            # We use the default exponential increasing factor of RasterEquidistantMetricSpace, adapted for grids
            right_bin_edge *= np.sqrt(2)
        bin_func.append(kwargs.get('maxlag'))
        kwargs.update({'bin_func': bin_func})

    # Prepare necessary arguments to pass to variogram subfunctions
    args = {'values': values, 'coords': coords, 'subsample_method': subsample_method, 'subsample': subsample,
            'verbose': verbose}
    if subsample_method in ['cdist_equidistant','pdist_ring','pdist_disk', 'pdist_point']:
        # The shape is needed for those three methods
        args.update({'shape': (nx, ny)})
        if subsample_method == 'cdist_equidistant':
            # The coordinate extent is needed for this method
            args.update({'extent':extent})
        else:
            args.update({'gsd': gsd})

    # If a random_state is passed, each run needs to be passed an independent child random state, otherwise they will
    # provide exactly the same sampling and results
    if random_state is not None:
        # Define the random state if only a seed is provided
        if isinstance(random_state, (np.random.RandomState, np.random.Generator)):
            rnd = random_state
        else:
            rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

        # Create a list of child random states
        if n_variograms == 1:
            # No issue if there is only one variogram run
            list_random_state = [rnd]
        else:
            # Otherwise, pass a list of seeds
            list_random_state = list(rnd.choice(n_variograms, n_variograms, replace=False))
    else:
        list_random_state = [None for i in range(n_variograms)]

    # Derive the variogram
    # Differentiate between 1 core and several cores for multiple runs
    # All variogram runs have random sampling inherent to their subfunctions, so we provide the same input arguments
    if n_jobs == 1:
        if verbose:
            print('Using 1 core...')

        list_df_run = []
        for i in range(n_variograms):

            argdict = {'i': i, 'imax': n_variograms, 'random_state': list_random_state[i], **args, **kwargs}
            df_run = _wrapper_get_empirical_variogram(argdict=argdict)

            list_df_run.append(df_run)
    else:
        if verbose:
            print('Using ' + str(n_jobs) + ' cores...')

        pool = mp.Pool(n_jobs, maxtasksperchild=1)
        argdict = [{'i': i, 'imax': n_variograms, 'random_state': list_random_state[i], **args, **kwargs} for i in range(n_variograms)]
        list_df_run = pool.map(_wrapper_get_empirical_variogram, argdict, chunksize=1)
        pool.close()
        pool.join()

    # Aggregate multiple ranges subsampling
    df = pd.concat(list_df_run)

    # For a single run, no multi-run sigma estimated
    if n_variograms == 1:
        df = df.rename(columns={'bins': 'lags'})
        df['err_exp'] = np.nan
    # For several runs, group results, use mean as empirical variogram, estimate sigma, and sum the counts
    else:
        df_grouped = df.groupby('bins', dropna=False)
        df_mean = df_grouped[['exp']].mean()
        df_std = df_grouped[['exp']].std()
        df_count = df_grouped[['count']].sum()
        df_mean['lags'] = df_mean.index.values
        df_mean['err_exp'] = df_std['exp'] / np.sqrt(n_variograms)
        df_mean['count'] = df_count['count']
        df = df_mean

    # Remove the last spatial lag bin which is always undersampled
    # TODO: Solve this problem at the root: how the spatial lag binning is defined, probably?
    df.drop(df.tail(1).index, inplace=True)

    return df

def _get_skgstat_variogram_model_name(model: str | Callable) -> str:
    """Fonction to identify a SciKit-GStat variogram model from a string or a function"""

    list_supported_models = ['spherical', 'gaussian', 'exponential', 'cubic', 'stable', 'matern']

    if callable(model):
        if inspect.getmodule(model).__name__ == 'skgstat.models':
            model_name = model.__name__
        else:
            raise ValueError('Variogram models can only be passed as functions of the skgstat.models package.')

    elif isinstance(model, str):
        model_name = None
        for supp_model in list_supported_models:
            if model.lower() in [supp_model[0:3], supp_model]:
                model_name = supp_model.lower()
        if model_name is None:
            raise ValueError('Variogram model name {} not recognized. Supported models are: '.format(model)+
                             ', '.join(list_supported_models)+'.')

    else:
        raise ValueError('Variogram models can be passed as strings or skgstat.models function. '
                         'Supported models are: '+', '.join(list_supported_models)+'.')

    return model_name

def get_variogram_model_func(params_variogram_model: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct the sum of spatial variogram function from a dataframe of variogram parameters.

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).

    :return: Function of sum of variogram with spatial lags.
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Define the function of sum of variogram models of h (spatial lag) to return
    def sum_model(h: np.ndarray) -> np.ndarray:

        fn = np.zeros(np.shape(h))

        for i in range(len(params_variogram_model)):
            # Get scikit-gstat model from name or Callable
            model_name = _get_skgstat_variogram_model_name(params_variogram_model['model'].values[i])
            model_function = getattr(skg.models, model_name)
            r = params_variogram_model['range'].values[i]
            p = params_variogram_model['psill'].values[i]
            # For models that expect 2 parameters
            if model_name in ['spherical', 'gaussian', 'exponential', 'cubic']:
                fn += model_function(h, r, p)
            # For models that expect 3 parameters
            elif model_name in ['stable', 'matern']:
                s = params_variogram_model['smooth'].values[i]
                fn += model_function(h, r, p, s)
        return fn

    return sum_model

def covariance_from_variogram(params_variogram_model: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
    """
    Construct the spatial covariance function from a dataframe of variogram parameters.
    The covariance function is the sum of partial sills "PS" minus the sum of associated variograms "gamma":
    C = PS - gamma

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).

    :return: Covariance function with spatial lags
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get total sill
    total_sill = np.sum(params_variogram_model['psill'])

    # Get function from sum of variogram
    sum_variogram = get_variogram_model_func(params_variogram_model)

    def cov(h):
        return total_sill - sum_variogram(h)

    return cov

def correlation_from_variogram(params_variogram_model: pd.DataFrame)-> Callable[[np.ndarray], np.ndarray]:
    """
    Construct the spatial correlation function from a dataframe of variogram parameters.
    The correlation function is the covariance function "C" divided by the sum of partial sills "PS": rho = C / PS

    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).

    :return: Correlation function with spatial lags
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get total sill
    total_sill = np.sum(params_variogram_model['psill'].values)

    # Get covariance from sum of variogram
    cov = covariance_from_variogram(params_variogram_model)

    def rho(h):
        return cov(h)/total_sill

    return rho


def fit_sum_model_variogram(list_models: list[str | Callable], empirical_variogram: pd.DataFrame,
                            bounds: list[tuple[float, float]] = None,
                            p0: list[float] = None) -> tuple[Callable[[np.ndarray], np.ndarray], pd.DataFrame]:
    """
    Fit a sum of variogram models to an empirical variogram, with weighted least-squares based on sampling errors. To
    use preferably with the empirical variogram dataframe returned by the `sample_empirical_variogram` function.

    :param list_models: List of K variogram models to sum for the fit in order from short to long ranges. Can either be
        a 3-letter string, full string of the variogram name or SciKit-GStat model function (e.g., for a
        spherical model "Sph", "Spherical" or skgstat.models.spherical).
    :param empirical_variogram: Empirical variogram, formatted as a dataframe with count (pairwise sample count), lags
        (upper bound of spatial lag bin), exp (experimental variance), and err_exp (error on experimental variance).
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper, sill lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).

    :return: Function of sum of variogram, Dataframe of optimized coefficients.
    """

    # Define a function of a sum of variogram model forms, with undetermined arguments
    def variogram_sum(h, *args):
        fn = 0
        i = 0
        for model in list_models:
            # Get the model name and convert to SciKit-GStat function
            model_name = _get_skgstat_variogram_model_name(model)
            model_function = getattr(skg.models, model_name)
            # For models that expect 2 parameters
            if model_name in ['spherical', 'gaussian', 'exponential', 'cubic']:
                fn += model_function(h, args[i], args[i+1])
                i += 2
            # For models that expect 3 parameters
            elif model_name in ['stable', 'matern']:
                fn += model_function(h, args[i], args[i+1], args[i+2])
                i += 3

        return fn

    # First, filter outliers
    empirical_variogram = empirical_variogram[np.isfinite(empirical_variogram.exp.values)]

    # Use shape of empirical variogram to assess rough boundaries/first estimates
    n_average = np.ceil(len(empirical_variogram.exp.values) / 10)
    exp_movaverage = np.convolve(empirical_variogram.exp.values, np.ones(int(n_average)) / n_average, mode='valid')
    grad = np.gradient(exp_movaverage, 2)
    # Maximum variance of the process
    max_var = np.max(exp_movaverage)

    # Simplify things for scipy: let's provide boundaries and first guesses
    if bounds is None:
        bounds = []
        for i in range(len(list_models)):

            # Use largest boundaries possible for our problem
            psill_bound = [0, max_var]
            range_bound = [0, empirical_variogram.lags.values[-1]]

            # Add bounds and guesses with same order as function arguments
            bounds.append(range_bound)
            bounds.append(psill_bound)
    if p0 is None:
        p0 = []
        for i in range(len(list_models)):
            # Use psill evenly distributed
            psill_p0 = ((i+1)/len(list_models))*max_var

            # Use corresponding ranges
            # !! This fails when no empirical value crosses this (too wide binning/nugget)
            # ind = np.array(np.abs(exp_movaverage-psill_p0)).argmin()
            # range_p0 = empirical_variogram.lags.values[ind]
            range_p0 = ((i+1)/len(list_models)) * empirical_variogram.lags.values[-1]

            p0.append(range_p0)
            p0.append(psill_p0)

    bounds = np.transpose(np.array(bounds))

    # If the error provided is all NaNs (single variogram run), or all zeros (two variogram runs), run without weights
    if np.all(np.isnan(empirical_variogram.err_exp.values)) or np.all(empirical_variogram.err_exp.values == 0):
        cof, cov = curve_fit(variogram_sum, empirical_variogram.lags.values, empirical_variogram.exp.values, method='trf',
                             p0=p0, bounds=bounds)
    # Otherwise, use a weighted fit
    else:
        # We need to filter for possible no data in the error
        valid = np.isfinite(empirical_variogram.err_exp.values)
        cof, cov = curve_fit(variogram_sum, empirical_variogram.lags.values[valid], empirical_variogram.exp.values[valid],
                             method='trf', p0=p0, bounds=bounds, sigma=empirical_variogram.err_exp.values[valid])

    # Store optimized parameters
    list_df = []
    i = 0
    for model in list_models:
        model_name = _get_skgstat_variogram_model_name(model)
        # For models that expect 2 parameters
        if model_name in ['spherical', 'gaussian', 'exponential', 'cubic']:
            df = pd.DataFrame()
            df = df.assign(model=[model_name], range=[cof[i]], psill=[cof[i+1]])
            i += 2
        # For models that expect 3 parameters
        elif model_name in ['stable', 'matern']:
            df = pd.DataFrame()
            df = df.assign(model=[model_name], range=[cof[i]], psill=[cof[i + 1]], smooth=[cof[i+2]])
            i += 3
        list_df.append(df)
    df_params = pd.concat(list_df)

    # Also pass the function of sum of variogram
    variogram_sum_fit = get_variogram_model_func(df_params)

    return variogram_sum_fit, df_params

def estimate_model_spatial_correlation(dvalues: Union[np.ndarray, RasterType], list_models: list[str | Callable],
                                       estimator = 'dowd', gsd: float = None, coords: np.ndarray = None, subsample: int = 1000,
                                       subsample_method: str = 'cdist_equidistant', n_variograms: int = 1,
                                       n_jobs: int = 1, verbose = False,
                                       random_state: None | np.random.RandomState | np.random.Generator | int = None,
                                       bounds: list[tuple[float, float]] = None, p0: list[float] = None,
                                       **kwargs) -> tuple[pd.DataFrame, pd.DataFrame, Callable[[np.ndarray], np.ndarray]]:

    """
    Estimate and model the spatial correlation of the input variable by empirical variogram sampling and fitting of a
    sum of variogram model.

    The spatial correlation is returned as a function of spatial lags (in units of the input coordinates) which gives a
    correlation value between 0 and 1.

    This function samples an empirical variogram using skgstat.Variogram, then optimizes by weighted least-squares the
    sum of a defined number of models, using the functions `sample_empirical_variogram` and `fit_sum_model_variogram`.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as elevation differences on stable terrain)
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
    :param verbose: Print statements during processing
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper, sill lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).

    :return: Dataframe of empirical variogram, Dataframe of optimized model parameters, Function of spatial correlation (0 to 1) with spatial lags
    """

    empirical_variogram = sample_empirical_variogram(values=dvalues, estimator=estimator, gsd=gsd, coords=coords,
                                                     subsample=subsample, subsample_method=subsample_method,
                                                     n_variograms=n_variograms, n_jobs=n_jobs, verbose=verbose,
                                                     random_state=random_state, **kwargs)

    params_variogram_model = fit_sum_model_variogram(list_models=list_models, empirical_variogram=empirical_variogram,
                                             bounds=bounds, p0=p0)[1]

    spatial_correlation_func = correlation_from_variogram(params_variogram_model=params_variogram_model)

    return empirical_variogram, params_variogram_model, spatial_correlation_func

def infer_spatial_correlation_from_stable(dvalues: np.ndarray | RasterType,
                                          list_models: list[str | Callable],
                                          stable_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                          unstable_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                          errors: np.ndarray | RasterType = None,
                                          estimator = 'dowd', gsd: float = None, coords: np.ndarray = None,
                                          subsample: int = 1000, subsample_method: str = 'cdist_equidistant',
                                          n_variograms: int = 1, n_jobs: int = 1, verbose = False,
                                          bounds: list[tuple[float, float]] = None, p0: list[float] = None,
                                          random_state: None | np.random.RandomState | np.random.Generator | int = None,
                                          **kwargs
                                          ) -> tuple[pd.DataFrame, pd.DataFrame, Callable[[np.ndarray], np.ndarray]]:
    """
    Infer spatial correlation of errors from differenced values on stable terrain and a list of variogram model to fit
    as a sum.

    This function returns a dataframe of the empirical variogram, a dataframe of optimized model parameters, and a
    spatial correlation function. The spatial correlation is returned as a function of spatial lags
    (in units of the input coordinates) which gives a correlation value between 0 and 1.
    It is a convenience wrapper for `estimate_model_spatial_correlation` to work on either Raster or array and compute
    the stable mask.

    If no stable or unstable mask is provided to mask in or out the values, all terrain is used.

    :param dvalues: Proxy values as array or Raster (i.e., differenced values where signal should be zero such as elevation differences on stable terrain)
    :param list_models: List of K variogram models to sum for the fit in order from short to long ranges. Can either be
        a 3-letter string, full string of the variogram name or SciKit-GStat model function (e.g., for a
        spherical model "Sph", "Spherical" or skgstat.models.spherical).
    :param stable_mask: Vector shapefile of stable terrain (if dvalues is Raster), or boolean array of same shape as dvalues
    :param unstable_mask: Vector shapefile of unstable terrain (if dvalues is Raster), or boolean array of same shape as dvalues
    :param errors: Error values to account for heteroscedasticity (ignored if None).
    :param estimator: Estimator for the empirical variogram; default to Dowd's variogram (see skgstat.Variogram for
        the list of available estimators).
    :param gsd: Ground sampling distance
    :param coords: Coordinates
    :param subsample: Number of samples to randomly draw from the values
    :param subsample_method: Spatial subsampling method
    :param n_variograms: Number of independent empirical variogram estimations (to estimate empirical variogram spread)
    :param n_jobs: Number of processing cores
    :param verbose: Print statements during processing
    :param bounds: Bounds of range and sill parameters for each model (shape K x 4 = K x range lower, range upper, sill lower, sill upper).
    :param p0: Initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess).
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Dataframe of empirical variogram, Dataframe of optimized model parameters, Function of spatial correlation (0 to 1) with spatial lags
    """

    # Check inputs
    if not isinstance(dvalues, (Raster, np.ndarray)):
        raise ValueError('The dvalues must be a Raster or NumPy array.')
    if stable_mask is not None and not isinstance(stable_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The stable mask must be a Vector, GeoDataFrame or NumPy array.')
    if unstable_mask is not None and not isinstance(unstable_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The unstable mask must be a Vector, GeoDataFrame or NumPy array.')

    # Check that input stable mask can only be a georeferenced vector if the proxy values are a Raster to project onto
    if not isinstance(dvalues, Raster) and isinstance(stable_mask, (Vector, gpd.GeoDataFrame)):
        raise ValueError(
            'The stable mask can only passed as a Vector or GeoDataFrame if the input dvalues is a Raster.')

    # Get array if input is a Raster
    if isinstance(dvalues, Raster):
        dvalues_arr = get_array_and_mask(dvalues)[0]
        gsd = dvalues.res[0]
    else:
        dvalues_arr = dvalues

    # If the stable mask is not an array, create it
    if stable_mask is None:
        stable_mask_arr = np.ones(np.shape(dvalues_arr), dtype=bool)
    elif not isinstance(stable_mask, np.ndarray):

        # If the stable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(stable_mask, gpd.GeoDataFrame):
            stable_vector = Vector(stable_mask)
        else:
            stable_vector = stable_mask

        # Create the mask
        stable_mask_arr = stable_vector.create_mask(dvalues)
    # If the mask is already an array, just pass it
    else:
        stable_mask_arr = stable_mask

    # If the unstable mask is not an array, create it
    if unstable_mask is None:
        unstable_mask_arr = np.zeros(np.shape(dvalues_arr), dtype=bool)
    elif not isinstance(unstable_mask, np.ndarray):

        # If the unstable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(unstable_mask, gpd.GeoDataFrame):
            unstable_vector = Vector(unstable_mask)
        else:
            unstable_vector = unstable_mask

        # Create the mask
        unstable_mask_arr = unstable_vector.create_mask(dvalues)
    # If the mask is already an array, just pass it
    else:
        unstable_mask_arr = unstable_mask

    stable_mask_arr = np.logical_and(stable_mask_arr, ~unstable_mask_arr).squeeze()

    # Need to preserve the shape, so setting as NaNs all points not on stable terrain
    dvalues_stable_arr = dvalues_arr.copy()
    dvalues_stable_arr[~stable_mask_arr] = np.nan

    # Perform standardization if error array is provided
    if errors is not None:
        if isinstance(errors, Raster):
            errors_arr = get_array_and_mask(errors)[0]
        else:
            errors_arr = errors

        # Standardize
        dvalues_stable_arr /= errors_arr

    # Estimate and model spatial correlations
    empirical_variogram, params_variogram_model, spatial_correlation_func = estimate_model_spatial_correlation(
        dvalues=dvalues_stable_arr, list_models=list_models, estimator=estimator, gsd=gsd, coords=coords,
        subsample=subsample, subsample_method=subsample_method, n_variograms=n_variograms, n_jobs=n_jobs,
        verbose=verbose, random_state=random_state, bounds=bounds, p0=p0, **kwargs)

    return empirical_variogram, params_variogram_model, spatial_correlation_func


def _check_validity_params_variogram(params_variogram_model: pd.DataFrame):
    """Check the validity of the modelled variogram parameters dataframe (mostly in the case it is passed manually)."""

    # Check that expected columns exists
    expected_columns = ['model', 'range', 'psill']
    if not all([c in params_variogram_model for c in expected_columns]):
        raise ValueError('The dataframe with variogram parameters must contain the columns "model", "range" and "psill".')

    # Check that the format of variogram models are correct
    for m in params_variogram_model['model'].values:
        _get_skgstat_variogram_model_name(m)

    # Check that the format of ranges, sills are correct
    for r in params_variogram_model['range'].values:
        if not isinstance(r, (float, np.floating, int, np.integer)):
            raise ValueError('The variogram ranges must be float or integer.')
        if r <= 0:
            raise ValueError('The variogram ranges must have non-zero, positive values.')

    # Check that the format of ranges, sills are correct
    for p in params_variogram_model['psill'].values:
        if not isinstance(p, (float, np.floating, int, np.integer)):
            raise ValueError('The variogram partial sills must be float or integer.')
        if p <= 0:
            raise ValueError('The variogram partial sills must have non-zero, positive values.')

    # Check that the mattern smoothness factor exist and is rightly formatted
    if ['stable'] in params_variogram_model['model'].values or ['matern'] in params_variogram_model['model'].values:
        if 'smooth' not in params_variogram_model:
            raise ValueError('The dataframe with variogram parameters must contain the column "smooth" for '
                             'the smoothness factor when using Matern or Stable models.')
        for i in range(len(params_variogram_model)):
            if params_variogram_model['model'].values[i] in ['stable', 'matern']:
                s = params_variogram_model['smooth'].values[i]
                if not isinstance(s, (float, np.floating, int, np.integer)):
                    raise ValueError('The variogram smoothness parameter must be float or integer.')
                if s <= 0:
                    raise ValueError('The variogram smoothness parameter must have non-zero, positive values.')


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
    two models: if SE is the standard error, SD the standard deviation and N_eff the number of effective samples, we have:
    SE = SD / sqrt(N_eff) => N_eff = SD^2 / SE^2 => N_eff = (PS1 + PS2)/SE^2 where PS1 and PS2 are the partial sills
    estimated from the variogram models, and SE is estimated by integrating the variogram models with parameters PS1/PS2
    and R1/R2 where R1/R2 are the correlation ranges.

    :param area: Area (in square unit of the variogram ranges)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).

    :return: Number of effective samples
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Lag l equal to the radius needed for a disk of area A
    l = np.sqrt(area / np.pi)

    # Below, we list exact integral functions over an area A assumed a disk integrated radially from the center

    # Formulas of h * covariance = h * ( psill - variogram ) for each form, then its integral for each form to yield
    # the standard error SE. a1 = range and c1 = partial sill.

    # Spherical: h * covariance = c1 * h * ( 1 - 3/2 * h / a1 + 1/2 * (h/a1)**3 ) = c1 * (h - 3/2 * h**2 / a1 + 1/2 * h**4 / a1**3)
    # Spherical: radial integral of above from 0 to L: SE**2 = 2 / (L**2) * c1 * (L**2 / 2 - 3/2 * L**3 / 3 / a1 + 1/2 * 1/5 * L**5 / a1**3)
    # which leads to SE**2 =  c1 * (1 - L / a1 + 1/5 * (L/a1)**3 )
    # If spherical model is above the spherical range a1: SE**2 = c1 /5 * (a1/L)**2

    def spherical_exact_integral(a1, c1, L):
        if l <= a1:
            squared_se = c1 * (1 - L/a1 + 1/5 * (L/a1)**3)
        else:
            squared_se = c1 / 5 * (a1/L)**2
        return squared_se

    # Exponential: h * covariance = c1 * h * exp(-h/a); a = a1/3
    # Exponential: radial integral of above from 0 to L: SE**2 =  2 / (L**2) * c1 * a * (a - exp(-L/a) * (a + L))

    def exponential_exact_integral(a1, c1, L):
        a = a1 / 3
        squared_se = 2 * c1 * (a/L)**2 * (1 - np.exp(-L/a) * (1 + L/a))
        return squared_se

    # Gaussian: h * covariance = c1 * h * exp(-h**2/a**2) ; a = a1/2
    # Gaussian: radial integral of above from 0 to L: SE**2 = 2 / (L**2) * c1 * 1/2 * a**2 * (1 - exp(-L**2/a**2))

    def gaussian_exact_integral(a1, c1, L):
        a = a1 / 2
        squared_se = c1 * (a/L)**2 * (1 - np.exp(-L**2 / a**2))
        return squared_se

    # Cubic: h * covariance = c1 * h * (1 - (7 * (h**2 / a**2)) + ((35 / 4) * (h**3 / a**3)) -
    #                          ((7 / 2) * (h**5 / a**5)) + ((3 / 4) * (h**7 / a**7)))
    # Cubic: radial integral of above from 0 to L: SE**2 = c1 * (6*a**7 -21*a**5*L**2 + 21*a**4*L**3 - 6*a**2*L**5 + L**7) / (6*a**7)

    def cubic_exact_integral(a1, c1, L):
        if l <= a1:
            squared_se = c1 * (6*a1**7 -21*a1**5*L**2 + 21*a1**4*L**3 - 6*a1**2*L**5 + L**7) / (6*a1**7)
        else:
            squared_se = 1/6 * c1 * a1**2 / L**2
        return squared_se

    squared_se = 0
    valid_models = ['spherical', 'exponential', 'gaussian', 'cubic']
    exact_integrals = [spherical_exact_integral, exponential_exact_integral, gaussian_exact_integral, cubic_exact_integral]
    for i in np.arange((len(params_variogram_model))):
        model_name = _get_skgstat_variogram_model_name(params_variogram_model['model'].values[i])
        r = params_variogram_model['range'].values[i]
        p = params_variogram_model['psill'].values[i]
        if model_name in valid_models:
            exact_integral = exact_integrals[valid_models.index(model_name)]
            squared_se += exact_integral(r, p, l)

    # We sum all partial sill to get the total sill
    total_sill = np.nansum(params_variogram_model.psill)
    # The number of effective sample is the fraction of total sill by squared standard error
    neff = total_sill/squared_se

    return neff

def _integrate_fun(fun: Callable, low_b: float, upp_b: float) -> float:
    """
    Numerically integrate function between an upper and lower bounds
    :param fun: Function to integrate
    :param low_b: Lower bound
    :param upp_b: Upper bound

    :return: Integral between lower and upper bound
    """
    return integrate.quad(fun, low_b, upp_b)[0]

def neff_circular_approx_numerical(area: float, params_variogram_model: pd.DataFrame) -> float:
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
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).

    :returns: Number of effective samples
    """

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get the total sill from the sum of partial sills
    total_sill = np.nansum(params_variogram_model.psill)

    # Define the covariance sum function times the spatial lag, for later integration
    def hcov_sum(h):
        return h * covariance_from_variogram(params_variogram_model)(h)

    # Get a radius for which the circle as the defined area
    h_equiv = np.sqrt(area / np.pi)

    # Integrate the covariance function between the center and the radius
    full_int = _integrate_fun(hcov_sum, 0, h_equiv)

    # Get the standard error, and return the number of effective samples
    squared_se = 2*np.pi*full_int / area

    # The number of effective sample is the fraction of total sill by squared standard error
    neff = total_sill/squared_se

    return neff


def neff_exact(coords: np.ndarray, errors: np.ndarray, params_variogram_model: pd.DataFrame, vectorized: bool = True) -> float:
    """
     Exact number of effective samples derived from a double sum of covariance with euclidean coordinates based on
     the provided variogram parameters. This method works for any shape of area.

    :param coords: Center coordinates with size (N,2) for each spatial support (typically, pixel)
    :param errors: Errors at the coordinates with size (N,) for each spatial support (typically, pixel)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).
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
        var = 0
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

                var += rho(d) * errors[i] * errors[j]

    # Or vectorized version
    else:
        # Convert the compact pairwise distance into a square matrix
        pds_matrix = squareform(pds)
        # Vectorize calculation
        var = np.sum(errors.reshape((-1, 1)) @ errors.reshape((1, -1)) * rho(pds_matrix.flatten()).reshape(pds_matrix.shape))

    # The number of effective sample is the fraction of total sill by squared standard error
    squared_se_dsc = var / n ** 2
    neff = np.mean(errors)**2/squared_se_dsc

    return neff

def neff_hugonnet_approx(coords: np.ndarray, errors: np.ndarray, params_variogram_model: pd.DataFrame, subsample: int = 1000,
                         vectorized: bool = True, random_state: None | np.random.RandomState | np.random.Generator | int = None) -> float:
    """
    Approximated number of effective samples derived from a double sum of covariance subsetted on one of the two sums,
    based on euclidean coordinates with the provided variogram parameters. This method works for any shape of area.
    See Hugonnet et al. (2022), https://doi.org/10.1109/jstars.2022.3188922, in particular Supplementary Fig. S16.

    :param coords: Center coordinates with size (N,2) for each spatial support (typically, pixel)
    :param errors: Errors at the coordinates with size (N,) for each spatial support (typically, pixel)
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).
    :param subsample: Number of samples to subset the calculation
    :param vectorized: Perform the vectorized calculation (used for testing).
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: Number of effective samples
    """

    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Check input dataframe
    _check_validity_params_variogram(params_variogram_model)

    # Get spatial correlation function from variogram parameters
    rho = correlation_from_variogram(params_variogram_model)

    # Get number of points and pairwise distance compacted matrix from scipy.pdist
    n = len(coords)
    pds = pdist(coords)

    # At maximum, the number of subsamples has to be equal to number of points
    subsample = min(subsample, n)

    # Get random subset of points for one of the sums
    rand_points = rnd.choice(n, size=subsample, replace=False)

    # Now we compute the double covariance sum
    # Either using for-loop-version
    if not vectorized:
        var = 0
        for ind_sub in range(subsample):
            for j in range(n):

                i = rand_points[ind_sub]
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

                var += rho(d) * errors[i] * errors[j]

    # Or vectorized version
    else:
        # We subset the points used in one dimension, for errors and pairwise distances computed
        errors_sub = errors[rand_points]
        pds_matrix = squareform(pds)
        pds_matrix_sub = pds_matrix[:, rand_points]
        # Vectorized calculation
        var = np.sum(errors.reshape((-1, 1)) @ errors_sub.reshape((1, -1)) * rho(pds_matrix_sub.flatten()).reshape(pds_matrix_sub.shape))

    # The number of effective sample is the fraction of total sill by squared standard error
    squared_se_dsc = var / (n * subsample)
    neff = np.mean(errors)**2 / squared_se_dsc

    return neff

def number_effective_samples(area: float | int | VectorType | gpd.GeoDataFrame, params_variogram_model: pd.DataFrame,
                             rasterize_resolution: RasterType | float = None, **kwargs) -> float:
    """
    Compute the number of effective samples, i.e. the number of uncorrelated samples, in an area accounting for spatial
    correlations described by a sum of variogram models.

    This function wraps two methods:

    - A discretized integration method that provides the exact estimate for any shape of area using a double sum of
        covariance. By default, this method is approximated using Equation 18 of Hugonnet et al. (2022), https://doi.org/10.1109/jstars.2022.3188922
        to decrease computing times while preserving a good approximation.

    - A continuous integration method that provides a conservative (i.e., slightly overestimated) value for a disk
        area shape, based on a generalization of the approach of Rolstad et al. (2009), http://dx.doi.org/10.3189/002214309789470950.

    By default, if a numeric value is passed for an area, the continuous method is used considering a disk shape. If a
    vector is passed, the discretized method is computed on that shape. If the discretized method is used, a resolution
    for rasterization is generally expected, otherwise is arbitrarily chosen as a fifth of the shortest correlation
    range to ensure a sufficiently fine grid for propagation of the shortest range.

    :param area: Area of interest either as a numeric value of surface in the same unit as the variogram ranges (will
        assume a circular shape), or as a vector (shapefile) of the area
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).
    :param rasterize_resolution: Resolution to rasterize the area if passed as a vector. Can be a float value or a Raster.
    :param kwargs: Keyword argument to pass to the `neff_hugonnet_approx` function.

    :return: Number of effective samples
    """

    # Check input for variogram parameters
    _check_validity_params_variogram(params_variogram_model=params_variogram_model)

    # If area is numeric, run the continuous circular approximation
    if isinstance(area, (float, int, np.floating, np.integer)):
        neff = neff_circular_approx_numerical(area=area, params_variogram_model=params_variogram_model)

    # Otherwise, run the discrete sum of covariance
    elif isinstance(area, (Vector, gpd.GeoDataFrame)):

        # If the input is a geopandas dataframe, put into a Vector object
        if isinstance(area, gpd.GeoDataFrame):
            V = Vector(area)
        else:
            V = area

        if rasterize_resolution is None:
            rasterize_resolution =  np.min(params_variogram_model['range'].values)/5.
            warnings.warn('Resolution for vector rasterization is not defined and thus set at 20% of the shortest '
                'correlation range, which might result in large memory usage.')

        # Rasterize with numeric resolution or Raster metadata
        if isinstance(rasterize_resolution, (float, int, np.floating, np.integer)):

            # We only need relative mask and coordinates, not absolute
            mask = V.create_mask(xres=rasterize_resolution)
            x = rasterize_resolution * np.arange(0, mask.shape[0])
            y = rasterize_resolution * np.arange(0, mask.shape[1])
            coords = np.array(np.meshgrid(y, x))
            coords_on_mask = coords[:, mask].T

        elif isinstance(rasterize_resolution, Raster):

            # With a Raster we can get the coordinates directly
            mask = V.create_mask(rst=rasterize_resolution).squeeze()
            coords = np.array(rasterize_resolution.coords())
            coords_on_mask = coords[:, mask].T

        else:
            raise ValueError('The rasterize resolution must be a float, integer or Raster subclass.')

        # Here we don't care about heteroscedasticity, so all errors are standardized to one
        errors_on_mask = np.ones(len(coords_on_mask))

        neff = neff_hugonnet_approx(coords=coords_on_mask, errors=errors_on_mask,
                                    params_variogram_model=params_variogram_model, **kwargs)

    else:
        raise ValueError('Area must be a float, integer, Vector subclass or geopandas dataframe.')

    return neff

def spatial_error_propagation(areas: list[float | VectorType | gpd.GeoDataFrame],
                              errors: RasterType,
                              params_variogram_model: pd.Dataframe,
                              **kwargs) -> list[float]:
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
    :param params_variogram_model: Dataframe of variogram models to sum with three to four columns, "model" for the model types
        (e.g., ["spherical", "matern"]), "range" for the correlation ranges (e.g., [2, 100]), "psill" for the partial
        sills (e.g., [0.8, 0.2]) and "smooth" for the smoothness parameter if it exists for this model (e.g.,
        [None, 0.2]).
    :param kwargs: Keyword argument to pass to the `neff_hugonnet_approx` function.

    :return: List of standard errors (1-sigma) for the input areas
    """

    standard_errors = []
    errors_arr = get_array_and_mask(errors)[0]
    for area in areas:
        # We estimate the number of effective samples in the area
        neff = number_effective_samples(area=area, params_variogram_model=params_variogram_model,
                                        rasterize_resolution=errors, **kwargs)

        # We compute the average error in this area
        # If the area is only a value, take the average error over the entire Raster
        if isinstance(area, float):
            average_spread = np.nanmean(errors_arr)
        else:
            if isinstance(area, gpd.GeoDataFrame):
                area_vector = Vector(area)
            else:
                area_vector = area
            area_mask = area_vector.create_mask(errors).squeeze()

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


def _distance_latlon(tup1: tuple, tup2: tuple, earth_rad: float = 6373000) -> float:
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

    a = m.sin(dlat / 2)**2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon / 2)**2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    distance = earth_rad * c

    return distance

def patches_method(values: np.ndarray, gsd: float, area: float, mask: Optional[np.ndarray] = None,
                   perc_min_valid: float = 80., statistics: Iterable[Union[str, Callable, None]] = ['count', np.nanmedian ,nmad],
                   patch_shape: str = 'circular', n_patches: int = 1000, verbose: bool = False,
                   random_state: None | int | np.random.RandomState | np.random.Generator = None) -> pd.DataFrame:

    """
    Patches method for empirical estimation of the standard error over an integration area

    :param values: Values
    :param gsd: Ground sampling distance
    :param mask: Mask of sampled terrain
    :param area: Size of integration area
    :param perc_min_valid: Minimum valid area in the patch
    :param statistics: List of statistics to compute in the patch
    :param patch_shape: Shape of patch ['circular' or 'rectangular']
    :param n_patches: Maximum number of patches to sample
    :param verbose: Print statement to console
    :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

    :return: tile, mean, median, std and count of each patch
    """

    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    statistics_name = [f if isinstance(f, str) else f.__name__ for f in statistics]

    values, mask_values = get_array_and_mask(values)

    values = values.squeeze()

    # Use all grid if no mask is provided
    if mask is None:
        mask = np.ones(np.shape(values),dtype=bool)

    # First, remove non sampled area (but we need to keep the 2D shape of raster for patch sampling)
    valid_mask = np.logical_and(~mask_values, mask)
    values[~valid_mask] = np.nan

    # Divide raster in cadrants where we can sample
    nx, ny = np.shape(values)
    valid_count = len(values[~np.isnan(values)])
    count = nx * ny
    if verbose:
        print('Number of valid pixels: ' + str(count))
    nb_cadrant = int(np.floor(np.sqrt((count * gsd ** 2) / area) + 1))
    # For rectangular quadrants
    nx_sub = int(np.floor((nx - 1) / nb_cadrant))
    ny_sub = int(np.floor((ny - 1) / nb_cadrant))
    # For circular patches
    rad = np.sqrt(area/np.pi) / gsd

    # Create list of all possible cadrants
    list_cadrant = [[i, j] for i in range(nb_cadrant) for j in range(nb_cadrant)]
    u = 0
    # Keep sampling while there is cadrants left and below maximum number of patch to sample
    remaining_nsamp = n_patches
    list_df = []
    while len(list_cadrant) > 0 and u < n_patches:

        # Draw a random coordinate from the list of cadrants, select more than enough random points to avoid drawing
        # randomly and differencing lists several times
        list_idx_cadrant = rnd.choice(len(list_cadrant), size=min(len(list_cadrant), 10*remaining_nsamp))

        for idx_cadrant in list_idx_cadrant:

            if verbose:
                print('Working on a new cadrant')

            # Select center coordinates
            i = list_cadrant[idx_cadrant][0]
            j = list_cadrant[idx_cadrant][1]

            if patch_shape == 'rectangular':
                patch = values[nx_sub * i:nx_sub * (i + 1), ny_sub * j:ny_sub * (j + 1)].flatten()
            elif patch_shape == 'circular':
                center_x = np.floor(nx_sub*(i+1/2))
                center_y = np.floor(ny_sub*(j+1/2))
                mask = _create_circular_mask((nx, ny), center=[center_x, center_y], radius=rad)
                patch = values[mask]
            else:
                raise ValueError('Patch method must be rectangular or circular.')

            nb_pixel_total = len(patch)
            nb_pixel_valid = len(patch[np.isfinite(patch)])
            if nb_pixel_valid >= np.ceil(perc_min_valid / 100. * nb_pixel_total):
                u=u+1
                if u > n_patches:
                    break
                if verbose:
                    print('Found valid cadrant ' + str(u) + ' (maximum: ' + str(n_patches) + ')')

                df = pd.DataFrame()
                df = df.assign(tile=[str(i) + '_' + str(j)])
                for j, statistic in enumerate(statistics):
                    if isinstance(statistic, str):
                        if statistic == 'count':
                            df[statistic] = [nb_pixel_valid]
                        else:
                            raise ValueError('No other string than "count" are supported for named statistics.')
                    else:
                        df[statistics_name[j]] = [statistic(patch)]

                list_df.append(df)

        # Get remaining samples to draw
        remaining_nsamp = n_patches - u
        # Remove cadrants already sampled from list
        list_cadrant = [c for j, c in enumerate(list_cadrant) if j not in list_idx_cadrant]

    if len(list_df)>0:
        df_all = pd.concat(list_df)
    else:
        warnings.warn('No valid patch found covering this area: returning dataframe containing only nans' )
        df_all = pd.DataFrame()
        for j, statistic in enumerate(statistics):
            df_all[statistics_name[j]] = [np.nan]

    return df_all


def plot_variogram(df: pd.DataFrame, list_fit_fun: Optional[list[Callable[[np.ndarray], np.ndarray]]] = None,
                   list_fit_fun_label: Optional[list[str]] = None, ax: matplotlib.axes.Axes | None = None,
                   xscale='linear', xscale_range_split: Optional[list] = None,
                   xlabel = None, ylabel = None, xlim = None, ylim = None):
    """
    Plot empirical variogram, and optionally also plot one or several model fits.
    Input dataframe is expected to be the output of xdem.spatialstats.sample_empirical_variogram.
    Input function model is expected to be the output of xdem.spatialstats.fit_sum_model_variogram.

    :param df: Empirical variogram, formatted as a dataframe with count (pairwise sample count), lags
        (upper bound of spatial lag bin), exp (experimental variance), and err_exp (error on experimental variance).
    :param list_fit_fun: List of model function fits
    :param list_fit_fun_label: List of model function fits labels
    :param ax: Plotting ax to use, creates a new one by default
    :param xscale: Scale of X-axis
    :param xscale_range_split: List of ranges at which to split the figure
    :param xlabel: Label of X-axis
    :param ylabel: Label of Y-axis
    :param xlim: Limits of X-axis
    :param ylim: Limits of Y-axis
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
    expected_values = ['exp', 'lags', 'count']
    for val in expected_values:
        if val not in df.columns.values:
            raise ValueError('The expected variable "{}" is not part of the provided dataframe column names.'.format(val))

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    if ylabel is None:
        ylabel = r'Variance [$\mu$ $\pm \sigma$]'
    if xlabel is None:
        xlabel = 'Spatial lag (m)'

    init_gridsize = [10, 10]
    # Create parameters to split x axis into different linear scales
    # If there is no split, get parameters for a single subplot
    if xscale_range_split is None:
        nb_subpanels=1
        if xscale == 'log':
            xmin = [np.min(df.lags)/2]
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
            if xscale == 'log':
                first_xmin = np.min(df.lags)/2
            else:
                first_xmin = 0
            xscale_range_split = [first_xmin] + xscale_range_split
        # Add maximum distance if not in input
        if xscale_range_split[-1] != np.max(df.lags):
            xscale_range_split.append(np.max(df.lags))

        # Scale grid size by the number of subpanels
        nb_subpanels = len(xscale_range_split)-1
        gridsize = init_gridsize.copy()
        gridsize[0] *= nb_subpanels
        # Create list of parameters to pass to ax/grid objects of subpanels
        xmin, xmax, xgridmin, xgridmax = ([] for i in range(4))
        for i in range(nb_subpanels):
            xmin.append(xscale_range_split[i])
            xmax.append(xscale_range_split[i+1])
            xgridmin.append(init_gridsize[0]*i)
            xgridmax.append(init_gridsize[0]*(i+1))

    # Need a grid plot to show the sample count and the statistic
    grid = plt.GridSpec(gridsize[1], gridsize[0], wspace=0.5, hspace=0.5)

    # Loop over each subpanel
    for k in range(nb_subpanels):
        # First, an axis to plot the sample histogram
        ax0 = ax.inset_axes(grid[:3, xgridmin[k]:xgridmax[k]].get_position(fig).bounds)
        ax0.set_xscale(xscale)
        ax0.set_xticks([])

        # Plot the histogram manually with fill_between
        interval_var = [0] + list(df.lags)
        for i in range(len(df)):
            count = df['count'].values[i]
            ax0.fill_between([interval_var[i], interval_var[i+1]], [0] * 2, [count] * 2,
                             facecolor=plt.cm.Greys(0.75), alpha=1,
                             edgecolor='white', linewidth=0.5)
        if k == 0:
            ax0.set_ylabel('Sample count')
        # Scientific format to avoid undesired additional space on the label side
            ax0.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        else:
            ax0.set_yticks([])
        # Ignore warnings for log scales
        ax0.set_xlim((xmin[k], xmax[k]))

        # Now, plot the statistic of the data
        ax1 = ax.inset_axes(grid[3:, xgridmin[k]:xgridmax[k]].get_position(fig).bounds)

        # Get the lags bin centers
        bins_center = np.subtract(df.lags, np.diff([0] + df.lags.tolist()) / 2)

        # If all the estimated errors are all NaN (single run), simply plot the empirical variogram
        if np.all(np.isnan(df.err_exp)):
            ax1.scatter(bins_center, df.exp, label='Empirical variogram', color='blue', marker='x')
        # Otherwise, plot the error estimates through multiple runs
        else:
            ax1.errorbar(bins_center, df.exp, yerr=df.err_exp, label='Empirical variogram (1-sigma s.d)', fmt='x')

        # If a list of functions is passed, plot the modelled variograms
        if list_fit_fun is not None:
            for i, fit_fun in enumerate(list_fit_fun):
                x = np.linspace(xmin[k], xmax[k], 1000)
                y = fit_fun(x)

                if list_fit_fun_label is not None:
                    ax1.plot(x, y, linestyle='dashed', label=list_fit_fun_label[i], zorder=30)
                else:
                    ax1.plot(x, y, linestyle='dashed', color='black', zorder=30)

            if list_fit_fun_label is None:
                ax1.plot([],[],linestyle='dashed',color='black',label='Model fit')

        ax1.set_xscale(xscale)
        if nb_subpanels>1 and k == (nb_subpanels-1):
            ax1.xaxis.set_ticks(np.linspace(xmin[k], xmax[k], 3))
        elif nb_subpanels>1:
            ax1.xaxis.set_ticks(np.linspace(xmin[k],xmax[k],3)[:-1])

        if xlim is None:
            ax1.set_xlim((xmin[k], xmax[k]))
        else:
            ax1.set_xlim(xlim)

        if ylim is not None:
            ax1.set_ylim(ylim)
        else:
            if np.all(np.isnan(df.err_exp)):
                ax1.set_ylim((0, 1.05*np.nanmax(df.exp)))
            else:
                ax1.set_ylim((0, np.nanmax(df.exp)+np.nanmean(df.err_exp)))

        if k == int(nb_subpanels/2):
            ax1.set_xlabel(xlabel)
        if k == nb_subpanels - 1:
            ax1.legend(loc='lower right')
        if k == 0:
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_yticks([])


def plot_1d_binning(df: pd.DataFrame, var_name: str, statistic_name: str, label_var: Optional[str] = None,
                    label_statistic: Optional[str] = None, min_count: int = 30, ax: matplotlib.axes.Axes | None = None):
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
        raise ValueError('The variable "{}" is not part of the provided dataframe column names.'.format(var_name))

    if statistic_name not in df.columns.values:
        raise ValueError('The statistic "{}" is not part of the provided dataframe column names.'.format(statistic_name))

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    if label_var is None:
        label_var = var_name
    if label_statistic is None:
        label_statistic = statistic_name

    # Subsample to 1D and for the variable of interest
    df_sub = df[np.logical_and(df.nd == 1, np.isfinite(pd.IntervalIndex(df[var_name]).mid))].copy()
    # Remove statistic calculated in bins with too low count
    df_sub.loc[df_sub['count']<min_count, statistic_name] = np.nan

    # Need a grid plot to show the sample count and the statistic
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)

    # First, an axis to plot the sample histogram
    ax0 = ax.inset_axes(grid[:3, :].get_position(fig).bounds)
    ax0.set_xticks([])

    # Plot the histogram manually with fill_between
    interval_var = pd.IntervalIndex(df_sub[var_name])
    for i in range(len(df_sub) ):
        count = df_sub['count'].values[i]
        ax0.fill_between([interval_var[i].left, interval_var[i].right], [0] * 2, [count] * 2, facecolor=plt.cm.Greys(0.75), alpha=1,
                         edgecolor='white',linewidth=0.5)
    ax0.set_ylabel('Sample count')
    # Scientific format to avoid undesired additional space on the label side
    ax0.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    # Try to identify if the count is always the same
    # (np.quantile can have a couple undesired effet, so leave an error margin of 2 wrong bins and 5 count difference)
    if np.sum(~(np.abs(df_sub['count'].values[0] - df_sub['count'].values) < 5)) <= 2:
        ax0.text(0.5, 0.5, "Fixed number of\n samples: "+'{:,}'.format(int(df_sub['count'].values[0])), ha='center', va='center',
                 fontweight='bold', transform=ax0.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    ax0.set_ylim((0,1.1*np.max(df_sub['count'].values)))
    ax0.set_xlim((np.min(interval_var.left),np.max(interval_var.right)))

    # Now, plot the statistic of the data
    ax1 = ax.inset_axes(grid[3:, :].get_position(fig).bounds)
    ax1.scatter(interval_var.mid, df_sub[statistic_name],marker='x')
    ax1.set_xlabel(label_var)
    ax1.set_ylabel(label_statistic)
    ax1.set_xlim((np.min(interval_var.left),np.max(interval_var.right)))


def plot_2d_binning(df: pd.DataFrame, var_name_1: str, var_name_2: str, statistic_name: str,
                    label_var_name_1: Optional[str] = None, label_var_name_2: Optional[str] = None,
                    label_statistic: Optional[str] = None, cmap: matplotlib.colors.Colormap = plt.cm.Reds, min_count: int = 30,
                    scale_var_1: str = 'linear', scale_var_2: str = 'linear', vmin: float = None, vmax: float = None,
                    nodata_color: Union[str,tuple[float,float,float,float]] = 'yellow', ax: matplotlib.axes.Axes | None = None):
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
    """

    # Create axes
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None.")

    if var_name_1 not in df.columns.values:
        raise ValueError('The variable "{}" is not part of the provided dataframe column names.'.format(var_name_1))
    elif var_name_2 not in df.columns.values:
        raise ValueError('The variable "{}" is not part of the provided dataframe column names.'.format(var_name_2))

    if statistic_name not in df.columns.values:
        raise ValueError('The statistic "{}" is not part of the provided dataframe column names.'.format(statistic_name))

    # Hide axes for the main subplot (which will be subdivded)
    ax.axis("off")

    # Subsample to 2D and for the variables of interest
    df_sub = df[np.logical_and.reduce((df.nd == 2, np.isfinite(pd.IntervalIndex(df[var_name_1]).mid),
                                       np.isfinite(pd.IntervalIndex(df[var_name_2]).mid)))].copy()
    # Remove statistic calculated in bins with too low count
    df_sub.loc[df_sub['count']<min_count, statistic_name] = np.nan

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
    df_sub['var1_mid'] = interval_var_1.mid.values
    unique_var_1 = np.unique(df_sub.var1_mid)
    list_counts = []
    for i in range(len(unique_var_1)):
        df_var1 = df_sub[df_sub.var1_mid == unique_var_1[i]]
        count = np.nansum(df_var1['count'].values)
        list_counts.append(count)
        ax0.fill_between([df_var1[var_name_1].values[0].left, df_var1[var_name_1].values[0].right], [0] * 2, [count] * 2, facecolor=plt.cm.Greys(0.75), alpha=1,
                         edgecolor='white', linewidth=0.5)
    ax0.set_ylabel('Sample count')
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax0.set_ylim((0,1.1*np.max(list_counts)))
        ax0.set_xlim((np.min(interval_var_1.left),np.max(interval_var_1.right)))
    ax0.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    # Try to identify if the count is always the same
    if np.sum(~(np.abs(list_counts[0] - np.array(list_counts)) < 5)) <= 2:
        ax0.text(0.5, 0.5, "Fixed number of\nsamples: " + '{:,}'.format(int(list_counts[0])), ha='center', va='center',
                 fontweight='bold', transform=ax0.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Second, a vertical axis on the right to plot the sample histogram of the second variable
    ax1 = ax.inset_axes(grid[3:, -3:].get_position(fig).bounds)
    ax1.set_yscale(scale_var_2)
    ax1.set_yticklabels([])

    # Plot the histogram manually with fill_between
    interval_var_2 = pd.IntervalIndex(df_sub[var_name_2])
    df_sub['var2_mid'] = interval_var_2.mid.values
    unique_var_2 = np.unique(df_sub.var2_mid)
    list_counts = []
    for i in range(len(unique_var_2)):
        df_var2 = df_sub[df_sub.var2_mid == unique_var_2[i]]
        count = np.nansum(df_var2['count'].values)
        list_counts.append(count)
        ax1.fill_between([0, count], [df_var2[var_name_2].values[0].left] * 2, [df_var2[var_name_2].values[0].right] * 2, facecolor=plt.cm.Greys(0.75),
                         alpha=1, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Sample count')
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax1.set_xlim((0,1.1*np.max(list_counts)))
        ax1.set_ylim((np.min(interval_var_2.left),np.max(interval_var_2.right)))
    ax1.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Try to identify if the count is always the same
    if np.sum(~(np.abs(list_counts[0] - np.array(list_counts)) < 5)) <= 2:
        ax1.text(0.5, 0.5, "Fixed number of\nsamples: " + '{:,}'.format(int(list_counts[0])), ha='center', va='center',
                 fontweight='bold', transform=ax1.transAxes, rotation=90, bbox=dict(facecolor='white', alpha=0.8))

    # Third, an axis to plot the data as a colored grid
    ax2 = ax.inset_axes(grid[3:, :-3].get_position(fig).bounds)

    # Define limits of colormap is none are provided, robust max and min using percentiles
    if vmin is None and vmax is None:
        vmax = np.nanpercentile(df_sub[statistic_name].values, 99)
        vmin = np.nanpercentile(df_sub[statistic_name].values, 1)

    # Create custom colormap
    col_bounds = np.array([vmin, np.mean([vmin,vmax]), vmax])
    cb = []
    cb_val = np.linspace(0, 1, len(col_bounds))
    for j in range(len(cb_val)):
        cb.append(cmap(cb_val[j]))
    cmap_cus = colors.LinearSegmentedColormap.from_list('my_cb', list(
        zip((col_bounds - min(col_bounds)) / (max(col_bounds - min(col_bounds))), cb)), N=1000)

    # Plot a 2D colored grid using fill_between
    for i in range(len(unique_var_1)):
        for j in range(len(unique_var_2)):
            df_both = df_sub[np.logical_and(df_sub.var1_mid == unique_var_1[i], df_sub.var2_mid == unique_var_2[j])]

            stat = df_both[statistic_name].values[0]
            if np.isfinite(stat):
                stat_col = max(0.0001,min(0.9999,(stat - min(col_bounds))/(max(col_bounds)-min(col_bounds))))
                col = cmap_cus(stat_col)
            else:
                col = nodata_color

            ax2.fill_between([df_both[var_name_1].values[0].left, df_both[var_name_1].values[0].right], [df_both[var_name_2].values[0].left] * 2,
                            [df_both[var_name_2].values[0].right] * 2, facecolor=col, alpha=1, edgecolor='white')

    ax2.set_xlabel(label_var_name_1)
    ax2.set_ylabel(label_var_name_2)
    ax2.set_xscale(scale_var_1)
    ax2.set_yscale(scale_var_2)
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax2.set_xlim((np.min(interval_var_1.left),np.max(interval_var_1.right)))
        ax2.set_ylim((np.min(interval_var_2.left),np.max(interval_var_2.right)))

    # Fourth and finally, add a colormap and nodata color to the legend
    axcmap = ax.inset_axes(grid[:3, -3:].get_position(fig).bounds)

    # Remove ticks, labels, frame
    axcmap.set_xticks([])
    axcmap.set_yticks([])
    axcmap.spines['top'].set_visible(False)
    axcmap.spines['left'].set_visible(False)
    axcmap.spines['right'].set_visible(False)
    axcmap.spines['bottom'].set_visible(False)

    # Create an inset axis to manage the scale of the colormap
    cbaxes = axcmap.inset_axes([0, 0.75, 1, 0.2], label='cmap')

    # Create colormap object and plot
    norm = colors.Normalize(vmin=min(col_bounds), vmax=max(col_bounds))
    sm = plt.cm.ScalarMappable(cmap=cmap_cus, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbaxes, orientation='horizontal', extend='both', shrink=0.8)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label(label_statistic)

    # Create an inset axis to manage the scale of the nodata legend
    nodata = axcmap.inset_axes([0.4, 0.1, 0.2, 0.2], label='nodata')

    # Plot a nodata legend
    nodata.fill_between([0, 1], [0, 0], [1, 1], facecolor=nodata_color)
    nodata.set_xlim((0, 1))
    nodata.set_ylim((0, 1))
    nodata.set_xticks([])
    nodata.set_yticks([])
    nodata.text(0.5, -0.25, 'No data', ha='center',va='top')

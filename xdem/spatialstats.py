"""Spatial statistical tools to estimate uncertainties related to DEMs"""
from __future__ import annotations

import math as m
import multiprocessing as mp
import os
import warnings
from functools import partial

from typing import Callable, Union, Iterable, Optional, Sequence, Any

import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numba import njit
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import curve_fit
from skimage.draw import disk
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, griddata
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
from geoutils.spatial_tools import subsample_raster, get_array_and_mask
from geoutils.georaster import RasterType, Raster

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import skgstat as skg
    from skgstat import models

def nmad(data: np.ndarray, nfact: float = 1.4826) -> float:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.
    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    :param data: input data
    :param nfact: normalization factor for the data

    :returns nmad: (normalized) median absolute deviation of data.
    """
    if isinstance(data, np.ma.masked_array):
        data_arr = get_array_and_mask(data, check_shape=False)[0]
    else:
        data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))

def interp_nd_binning(df: pd.DataFrame, list_var_names: Union[str,list[str]], statistic : Union[str, Callable[[np.ndarray],float]] = nmad,
                      min_count: Optional[int] = 100) -> Callable[[tuple[np.ndarray, ...]], np.ndarray]:
    """
    Estimate an interpolant function for an N-dimensional binning. Preferably based on the output of nd_binning.
    For more details on the input dataframe, and associated list of variable name and statistic, see nd_binning.

    If the variable pd.DataSeries corresponds to an interval (as the output of nd_binning), uses the middle of the interval.
    Otherwise, uses the variable as such.

    Workflow of the function:
    Fills the no-data present on the regular N-D binning grid with nearest neighbour from scipy.griddata, then provides an
    interpolant function that linearly interpolates/extrapolates using scipy.RegularGridInterpolator.

    :param df: dataframe with statistic of binned values according to explanatory variables (preferably output of nd_binning)
    :param list_var_names: explanatory variable data series to select from the dataframe (containing interval or float dtype)
    :param statistic: statistic to interpolate, stored as a data series in the dataframe
    :param min_count: minimum number of samples to be used as a valid statistic (replaced by nodata)
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
    # if list of variable input is simply a string
    if isinstance(list_var_names,str):
        list_var_names = [list_var_names]

    # check that the dataframe contains what we need
    for var in list_var_names:
        if var not in df.columns:
            raise ValueError('Variable "'+var+'" does not exist in the provided dataframe.')
    statistic_name = statistic if isinstance(statistic,str) else statistic.__name__
    if statistic_name not in df.columns:
        raise ValueError('Statistic "' + statistic_name + '" does not exist in the provided dataframe.')
    if min_count is not None and 'count' not in df.columns:
        raise ValueError('Statistic "count" is not in the provided dataframe, necessary to use the min_count argument.')
    if df.empty:
        raise ValueError('Dataframe is empty.')

    df_sub = df.copy()

    # if the dataframe is an output of nd_binning, keep only the dimension of interest
    if 'nd' in df_sub.columns:
        df_sub = df_sub[df_sub.nd == len(list_var_names)]

    # compute the middle values instead of bin interval if the variable is a pandas interval type
    for var in list_var_names:
        check_any_interval = [isinstance(x, pd.Interval) for x in df_sub[var].values]
        if any(check_any_interval):
            df_sub[var] = pd.IntervalIndex(df_sub[var]).mid.values
        # otherwise, leave as is

    # check that explanatory variables have valid binning values which coincide along the dataframe
    df_sub = df_sub[np.logical_and.reduce([np.isfinite(df_sub[var].values) for var in list_var_names])]
    if df_sub.empty:
        raise ValueError('Dataframe does not contain a nd binning with the variables corresponding to the list of variables.')
    # check that the statistic data series contain valid data
    if all(~np.isfinite(df_sub[statistic_name].values)):
        raise ValueError('Dataframe does not contain any valid statistic values.')

    # remove statistic values calculated with a sample count under the minimum count
    if min_count is not None:
        df_sub.loc[df_sub['count'] < min_count,statistic_name] = np.nan

    values = df_sub[statistic_name].values
    ind_valid = np.isfinite(values)

    # re-check that the statistic data series contain valid data after filtering with min_count
    if all(~ind_valid):
        raise ValueError("Dataframe does not contain any valid statistic values after filtering with min_count = "+str(min_count)+".")

    # get a list of middle values for the binning coordinates, to define a nd grid
    list_bmid = []
    shape = []
    for var in list_var_names:
        bmid = sorted(np.unique(df_sub[var][ind_valid]))
        list_bmid.append(bmid)
        shape.append(len(bmid))

    # griddata first to perform nearest interpolation with NaNs (irregular grid)
    # valid values
    values = values[ind_valid]
    # coordinates of valid values
    points_valid = tuple([df_sub[var].values[ind_valid] for var in list_var_names])
    # grid coordinates
    bmid_grid = np.meshgrid(*list_bmid, indexing='ij')
    points_grid = tuple([bmid_grid[i].flatten() for i in range(len(list_var_names))])
    # fill grid no data with nearest neighbour
    values_grid = griddata(points_valid, values, points_grid, method='nearest')
    values_grid = values_grid.reshape(shape)

    # RegularGridInterpolator to perform linear interpolation/extrapolation on the grid
    # (will extrapolate only outside of boundaries not filled with the nearest of griddata as fill_value = None)
    interp_fun = RegularGridInterpolator(tuple(list_bmid), values_grid, method='linear', bounds_error=False, fill_value=None)

    return interp_fun


def nd_binning(values: np.ndarray, list_var: Iterable[np.ndarray], list_var_names=Iterable[str], list_var_bins: Optional[Union[int,Iterable[Iterable]]] = None,
                     statistics: Iterable[Union[str, Callable, None]] = ['count', np.nanmedian ,nmad], list_ranges : Optional[Iterable[Sequence]] = None) \
        -> pd.DataFrame:
    """
    N-dimensional binning of values according to one or several explanatory variables.
    Values input is a (N,) array and variable input is a list of flattened arrays of similar dimensions (N,).
    For more details on the format of input variables, see documentation of scipy.stats.binned_statistic_dd.

    :param values: values array (N,)
    :param list_var: list (L) of explanatory variables array (N,)
    :param list_var_names: list (L) of names of the explanatory variables
    :param list_var_bins: count, or list (L) of counts or custom bin edges for the explanatory variables; defaults to 10 bins
    :param statistics: list (X) of statistics to be computed; defaults to count, median and nmad
    :param list_ranges: list (L) of minimum and maximum ranges to bin the explanatory variables; defaults to min/max of the data
    :return:
    """

    # we separate 1d, 2d and nd binning, because propagating statistics between different dimensional binning is not always feasible
    # using scipy because it allows for several dimensional binning, while it's not straightforward in pandas
    if list_var_bins is None:
        list_var_bins = (10,) * len(list_var_names)
    elif isinstance(list_var_bins,int):
        list_var_bins = (list_var_bins,) * len(list_var_names)

    # flatten the arrays if this has not been done by the user
    values = values.ravel()
    list_var = [var.ravel() for var in list_var]

    # remove no data values
    valid_data = np.logical_and.reduce([np.isfinite(values)]+[np.isfinite(var) for var in list_var])
    values = values[valid_data]
    list_var = [var[valid_data] for var in list_var]

    statistics_name = [f if isinstance(f,str) else f.__name__ for f in statistics]

    # get binned statistics in 1d: a simple loop is sufficient
    list_df_1d = []
    for i, var in enumerate(list_var):
        df_stats_1d = pd.DataFrame()
        # get statistics
        for j, statistic in enumerate(statistics):
            stats_binned_1d, bedges_1d = binned_statistic(var,values,statistic=statistic,bins=list_var_bins[i],range=list_ranges)[:2]
            # save in a dataframe
            df_stats_1d[statistics_name[j]] = stats_binned_1d
        # we need to get the middle of the bins from the edges, to get the same dimension length
        df_stats_1d[list_var_names[i]] = pd.IntervalIndex.from_breaks(bedges_1d,closed='left')
        # report number of dimensions used
        df_stats_1d['nd'] = 1

        list_df_1d.append(df_stats_1d)

    # get binned statistics in 2d: all possible 2d combinations
    list_df_2d = []
    if len(list_var)>1:
        combs = list(itertools.combinations(list_var_names, 2))
        for i, comb in enumerate(combs):
            var1_name, var2_name = comb
            # corresponding variables indexes
            i1, i2 = list_var_names.index(var1_name), list_var_names.index(var2_name)
            df_stats_2d = pd.DataFrame()
            for j, statistic in enumerate(statistics):
                stats_binned_2d, bedges_var1, bedges_var2 = binned_statistic_2d(list_var[i1],list_var[i2],values,statistic=statistic
                                                             ,bins=[list_var_bins[i1],list_var_bins[i2]]
                                                             ,range=list_ranges)[:3]
                # get statistics
                df_stats_2d[statistics_name[j]] = stats_binned_2d.flatten()
            # derive interval indexes and convert bins into 2d indexes
            ii1 = pd.IntervalIndex.from_breaks(bedges_var1,closed='left')
            ii2 = pd.IntervalIndex.from_breaks(bedges_var2,closed='left')
            df_stats_2d[var1_name] = [i1 for i1 in ii1 for i2 in ii2]
            df_stats_2d[var2_name] = [i2 for i1 in ii1 for i2 in ii2]
            # report number of dimensions used
            df_stats_2d['nd'] = 2

            list_df_2d.append(df_stats_2d)


    # get binned statistics in nd, without redoing the same stats
    df_stats_nd = pd.DataFrame()
    if len(list_var)>2:
        for j, statistic in enumerate(statistics):
            stats_binned_2d, list_bedges = binned_statistic_dd(list_var,values,statistic=statistic,bins=list_var_bins,range=list_ranges)[0:2]
            df_stats_nd[statistics_name[j]] = stats_binned_2d.flatten()
        list_ii = []
        # loop through the bin edges and create IntervalIndexes from them (to get both
        for bedges in list_bedges:
            list_ii.append(pd.IntervalIndex.from_breaks(bedges,closed='left'))

        # create nd indexes in nd-array and flatten for each variable
        iind = np.meshgrid(*list_ii)
        for i, var_name in enumerate(list_var_names):
            df_stats_nd[var_name] = iind[i].flatten()

        # report number of dimensions used
        df_stats_nd['nd'] = len(list_var_names)

    # concatenate everything
    list_all_dfs = list_df_1d + list_df_2d + [df_stats_nd]
    df_concat = pd.concat(list_all_dfs)
    # commenting for now: pd.MultiIndex can be hard to use
    # df_concat = df_concat.set_index(list_var_names)

    return df_concat

def create_circular_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None,
                         radius: Optional[float] = None) -> np.ndarray:
    """
    Create circular mask on a raster, defaults to the center of the array and it's half width

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

    # skimage disk is not inclusive (correspond to distance_from_center < radius and not <= radius)
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

def create_ring_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None, in_radius: float = 0.,
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
        mask_inside = create_circular_mask((w,h),center=center,radius=in_radius)
        mask_outside = create_circular_mask((w,h),center=center,radius=out_radius)

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
            subindex = create_ring_mask((nx, ny), center=[center_x, center_y], in_radius=inside_radius,
                                          out_radius=outside_radius)
        else:
            subindex = create_circular_mask((nx, ny), center=[center_x, center_y], radius=inside_radius)

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

    :param values: values
    :param coords: coordinates
    :return: empirical variogram (variance, lags, counts)

    """

    # Remove random_state keyword argument that is not used
    kwargs.pop('random_state')

    # Get arguments of Variogram class init function
    vgm_args = skg.Variogram.__init__.__code__.co_varnames[:skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in vgm_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: '+','.join(list(remaining_kwargs.keys()))+ ' were not used.')
    # Filter corresponding arguments before passing
    filtered_kwargs =  {k:kwargs[k] for k in vgm_args if k in kwargs}

    # Derive variogram with default MetricSpace (equivalent to scipy.pdist)
    V = skg.Variogram(coordinates=coords, values=values, normalize=False, fit_method=None, **filtered_kwargs)

    # Get bins, empirical variogram values, and bin count
    bins, exp = V.get_empirical()
    count = V.bin_count

    # Write to dataframe
    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df


def _get_cdist_empirical_variogram(values: np.ndarray, coords: np.ndarray, subsample_method: str,
                                   **kwargs) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object calculating pairwise distances between two sample collections
    of a MetricSpace (see scikit-gstat documentation for more details)

    :param values: values
    :param coords: coordinates
    :return: empirical variogram (variance, lags, counts)

    """
    # Rename the "subsample" argument into "samples", which is used by skgstat Metric subclasses
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
    vgm_args = skg.Variogram.__init__.__code__.co_varnames[:skg.Variogram.__init__.__code__.co_argcount]
    # Check no other argument is left to be passed, accounting for MetricSpace arguments
    remaining_kwargs = kwargs.copy()
    for arg in vgm_args + ms_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: ' + ', '.join(list(remaining_kwargs.keys())) + ' were not used.')

    # Filter corresponding arguments before passing to MetricSpace function
    filtered_ms_kwargs = {k: kwargs[k] for k in ms_args if k in kwargs}
    M = ms(coords=coords, **filtered_ms_kwargs)

    # Filter corresponding arguments before passing to Variogram function
    filtered_var_kwargs = {k: kwargs[k] for k in vgm_args if k in kwargs}
    V = skg.Variogram(M, values=values, normalize=False, fit_method=None, **filtered_var_kwargs)

    # Get bins, empirical variogram values, and bin count
    bins, exp = V.get_empirical()
    count = V.bin_count

    # Write to dataframe
    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df


def _wrapper_get_empirical_variogram(argdict: dict) -> pd.DataFrame:
    """
    Multiprocessing wrapper for get_pdist_empirical_variogram and get_cdist_empirical variogram

    :param argdict: Keyword argument to pass to get_pdist/cdist_empirical_variogram
    :return: empirical variogram (variance, lags, counts)

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
                               subsample: int = 10000, subsample_method: str = 'cdist_equidistant',
                               n_variograms: int = 1, n_jobs: int = 1, verbose=False,
                               random_state: None | np.random.RandomState | np.random.Generator | int = None,
                               **kwargs) -> pd.DataFrame:
    """
    Sample empirical variograms with binning adaptable to multiple ranges and spatial subsampling adapted for raster data.
    By default, subsampling is based on RasterEquidistantMetricSpace implemented in scikit-gstat. This method samples more
    effectively large grid data by isolating pairs of spatially equidistant ensembles for distributed pairwise comparison.
    In practice, two subsamples are drawn for pairwise comparison: one from a disk of certain radius within the grid, and
    another one from rings of larger radii that increase steadily between the pixel size and the extent of the raster.
    Those disk and rings are sampled several times across the grid using random centers.

    If values are provided as a Raster subclass, nothing else is required.
    If values are provided as a 2D array (M,N), a ground sampling distance is sufficient to derive the pairwise distances.
    If values are provided as a 1D array (N), an array of coordinates (N,2) or (2,N) is expected. If the coordinates
    do not correspond to all points of the grid, a ground sampling distance is needed to correctly get the grid size.

    Spatial subsampling method argument subsample_method can be one of "cdist_equidistant", "cdist_point", "pdist_point",
     "pdist_disk" and "pdist_ring".
    The cdist methods use MetricSpace classes of scikit-gstat and do pairwise comparison of two ensembles as in
    scipy.spatial.cdist.
    The pdist methods use methods to subsample the Raster points directly and do pairwise comparison within a single
    ensemble as in scipy.spatial.pdist.

    For the cdist methods, the variogram is estimated in a single run from the MetricSpace.

    For the pdist methods, an iterative process is required: a list of ranges subsampled independently is used.
    Variograms are derived independently for several runs and ranges using each pairwise sample, and later aggregated.
    If the subsampling method selected is "random_point", the multi-range argument is ignored as range has no effect on
    this subsampling method.

    For pdist methods, keyword arguments are passed to skgstat.Variogram.
    For cdist methods, keyword arguments are passed to both skgstat.Variogram and skgstat.MetricSpace.

    :param values: values
    :param gsd: ground sampling distance
    :param coords: coordinates
    :param subsample: number of samples to randomly draw from the values
    :param subsample_method: spatial subsampling method
    :param n_variograms: number of independent empirical variogram estimations
    :param n_jobs: number of processing cores
    :param verbose: print statements during processing
    :param random_state: random state or seed number to use for calculations (to fix random sampling during testing)

    :return: empirical variogram (variance, lags, counts)
    """
    # First, check all that the values provided are OK
    if isinstance(values, Raster):
        gsd = values.res[0]
        values, mask = get_array_and_mask(values.data)
    elif isinstance(values, (np.ndarray, np.ma.masked_array)):
        values, mask = get_array_and_mask(values)
    else:
        raise TypeError('Values must be of type np.ndarray, np.ma.masked_array or Raster subclass.')
    values = values.squeeze()

    # Then, check if the logic between values, coords and gsd is respected
    if (gsd is not None or subsample_method in ['cdist_equidistant', 'pdist_disk','pdist_ring']) and values.ndim == 1:
        raise TypeError('Values array must be 2D when using any of the "cdist_equidistant", "pdist_disk" and '
                        '"pdist_ring" methods, or providing a ground sampling distance instead of coordinates.')
    elif coords is not None and values.ndim != 1:
        raise TypeError('Values array must be 1D when providing coordinates.')
    elif coords is not None and (coords.shape[0] != 2 and coords.shape[1] != 2):
        raise TypeError('The coordinates array must have one dimension with length equal to 2')

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
        df['err_exp'] = np.nan
    # For several runs, group results, use mean as empirical variogram, estimate sigma, and sum the counts
    else:
        df_grouped = df.groupby('bins', dropna=False)
        df_mean = df_grouped[['exp']].mean()
        df_std = df_grouped[['exp']].std()
        df_count = df_grouped[['count']].sum()
        df_mean['bins'] = df_mean.index.values
        df_mean['err_exp'] = df_std['exp']
        df_mean['count'] = df_count['count']
        df = df_mean

    return df

def fit_sum_model_variogram(list_model: list[str], empirical_variogram: pd.DataFrame,
                            bounds: list[tuple[float, float]] = None,
                            p0: list[float] = None) -> tuple[Callable, list[float]]:
    """
    Fit a multi-range variogram model to an empirical variogram, weighted least-squares based on sampling errors

    :param list_model: list of K variogram models to sum for the fit: from short-range to long-ranges
    :param empirical_variogram: empirical variogram
    :param bounds: bounds of ranges and sills for each model (shape K x 4 = K x range lower, range upper, sill lower, sill upper)
    :param p0: initial guess of ranges and sills each model (shape K x 2 = K x range first guess, sill first guess)

    :return: modelled variogram function, coefficients
    """
    # TODO: expand to other models than spherical, exponential, gaussian (more than 2 arguments)

    # Define a sum of variogram function
    def vgm_sum(h, *args):
        fn = 0
        i = 0
        for model in list_model:
            if model == 'Sph':
                fn += skg.models.spherical(h, args[i], args[i+1])
            elif model == 'Gau':
                fn += skg.models.gaussian(h, args[i], args[i+1])
            elif model == 'Exp':
                fn += skg.models.exponential(h, args[i], args[i+1])
            i += 2

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
        for i in range(len(list_model)):

            # Use largest boundaries possible for our problem
            psill_bound = [0, max_var]
            range_bound = [0, empirical_variogram.bins.values[-1]]

            # Add bounds and guesses with same order as function arguments
            bounds.append(range_bound)
            bounds.append(psill_bound)
    if p0 is None:
        p0 = []
        for i in range(len(list_model)):
            # Use psill evenly distributed
            psill_p0 = ((i+1)/len(list_model))*max_var

            # Use corresponding ranges
            # !! This fails when no empirical value crosses this (too wide binning/nugget)
            # ind = np.array(np.abs(exp_movaverage-psill_p0)).argmin()
            # range_p0 = empirical_variogram.bins.values[ind]
            range_p0 = ((i+1)/len(list_model)) * empirical_variogram.bins.values[-1]

            p0.append(range_p0)
            p0.append(psill_p0)

    bounds = np.transpose(np.array(bounds))

    # If the error provided is all NaNs (single variogram run), or all zeros (two variogram runs), run without weights
    if np.all(np.isnan(empirical_variogram.err_exp.values)) or np.all(empirical_variogram.err_exp.values == 0):
        cof, cov = curve_fit(vgm_sum, empirical_variogram.bins.values, empirical_variogram.exp.values, method='trf',
                             p0=p0, bounds=bounds)
    # Otherwise, use a weighted fit
    else:
        # We need to filter for possible no data in the error
        valid = np.isfinite(empirical_variogram.err_exp.values)
        cof, cov = curve_fit(vgm_sum, empirical_variogram.bins.values[valid], empirical_variogram.exp.values[valid],
                             method='trf', p0=p0, bounds=bounds, sigma=empirical_variogram.err_exp.values[valid])

    # Provide the output function (couldn't find a way to pass this through functool.partial as arguments are unordered)
    def vgm_sum_fit(h):
        fn = 0
        i = 0
        for model in list_model:
            if model == 'Sph':
                fn += skg.models.spherical(h, cof[i], cof[i + 1])
            elif model == 'Gau':
                fn += skg.models.gaussian(h, cof[i], cof[i + 1])
            elif model == 'Exp':
                fn += skg.models.exponential(h, cof[i], cof[ i + 1])
            # fn += skg.models.spherical(h, cof[i], cof[i+1])
            i += 2

        return fn

    return vgm_sum_fit, cof


def exact_neff_sphsum_circular(area: float, crange1: float, psill1: float, crange2: float, psill2: float) -> float:
    """
    Number of effective samples derived from exact integration of sum of 2 spherical variogram models over a circular area.
    The number of effective samples serves to convert between standard deviation/partial sills and standard error
    over the area.
    If SE is the standard error, SD the standard deviation and N_eff the number of effective samples, we have:
    SE = SD / sqrt(N_eff) => N_eff = SD^2 / SE^2 => N_eff = (PS1 + PS2)/SE^2 where PS1 and PS2 are the partial sills
    estimated from the variogram models, and SE is estimated by integrating the variogram models with parameters PS1/PS2
    and R1/R2 where R1/R2 are the correlation ranges.
    Source: Rolstad et al. (2009), appendix: http://dx.doi.org/10.3189/002214309789470950

    :param area: circular area
    :param crange1: range of short-range variogram model
    :param psill1: partial sill of short-range variogram model
    :param crange2: range of long-range variogram model
    :param psill2: partial sill of long-range variogram model

    :return: number of effective samples
    """
    # short range variogram
    c1 = psill1  # partial sill
    a1 = crange1  # short correlation range

    # long range variogram
    c1_2 = psill2
    a1_2 = crange2  # long correlation range

    h_equiv = np.sqrt(area / np.pi)

    # hypothesis of a circular shape to integrate variogram model
    if h_equiv > a1_2:
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * a1_2 ** 2 / (5 * h_equiv ** 2))
    elif (h_equiv < a1_2) and (h_equiv > a1):
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))
    else:
        std_err = np.sqrt(c1 * (1-h_equiv / a1+1 / 5 * (h_equiv / a1) ** 3) +
                          c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))

    return (psill1 + psill2)/std_err**2


def neff_circ(area: float, list_vgm: list[tuple[float, str, float]]) -> float:
    """
    Number of effective samples derived from numerical integration for any sum of variogram models a circular area
    (generalization of Rolstad et al. (2009): http://dx.doi.org/10.3189/002214309789470950)
    The number of effective samples N_eff serves to convert between standard deviation/partial sills and standard error
    over the area: SE = SD / sqrt(N_eff) if SE is the standard error, SD the standard deviation.

    :param area: area
    :param list_vgm: variogram functions to sum (range, model name, partial sill)

    :returns: number of effective samples
    """
    psill_tot = 0
    for vario in list_vgm:
        psill_tot += vario[2]

    def hcov_sum(h):
        fn = 0
        for vario in list_vgm:
            crange, model, psill = vario
            fn += h*(cov(h, crange, model=model, psill=psill))

        return fn

    h_equiv = np.sqrt(area / np.pi)

    full_int = integrate_fun(hcov_sum, 0, h_equiv)
    std_err = np.sqrt(2*np.pi*full_int / area)

    return psill_tot/std_err**2


def neff_rect(area: float, width: float, crange1: float, psill1: float, model1: str = 'Sph', crange2: float = None,
              psill2: float = None, model2: str = None) -> float:
    """
    Number of effective samples derived from numerical integration for a sum of 2 variogram functions over a rectangular area

    :param area: area
    :param width: width of rectangular area
    :param crange1: correlation range of first variogram
    :param psill1: partial sill of first variogram
    :param model1: model of first variogram
    :param crange2: correlation range of second variogram
    :param psill2: partial sill of second variogram
    :param model2: model of second variogram

    :returns: number of effective samples
    """
    def hcov_sum(h, crange1=crange1, psill1=psill1, model1=model1, crange2=crange2, psill2=psill2, model2=model2):

        if crange2 is None or psill2 is None or model2 is None:
            return h*(cov(h, crange1, model=model1, psill=psill1))
        else:
            return h*(cov(h, crange1, model=model1, psill=psill1)+cov(h, crange2, model=model2, psill=psill2))

    width = min(width, area/width)

    full_int = integrate_fun(hcov_sum, 0, width/2)
    bin_int = np.linspace(width/2, area/width, 100)
    for i in range(len(bin_int)-1):
        low = bin_int[i]
        upp = bin_int[i+1]
        mid = bin_int[i] + (bin_int[i+1] - bin_int[i])/2
        piec_int = integrate_fun(hcov_sum, low, upp)
        full_int += piec_int * 2/np.pi*np.arctan(width/(2*mid))

    std_err = np.sqrt(2*np.pi*full_int / area)

    if crange2 is None or psill2 is None or model2 is None:
        return psill1 / std_err ** 2
    else:
        return (psill1 + psill2) / std_err ** 2


def integrate_fun(fun: Callable, low_b: float, upp_b: float) -> float:
    """
    Numerically integrate function between upper and lower bounds
    :param fun: function
    :param low_b: lower bound
    :param upp_b: upper bound

    :return: integral
    """
    return integrate.quad(fun, low_b, upp_b)[0]


def cov(h: float, crange: float, model: str = 'Sph', psill: float = 1., kappa: float = 1/2, nugget: float = 0) -> Callable:
    """
    Covariance function based on variogram function (COV = STD - VGM)

    :param h: spatial lag
    :param crange: correlation range
    :param model: model
    :param psill: partial sill
    :param kappa: smoothing parameter for Exp Class
    :param nugget: nugget

    :returns: covariance function
    """
    return (nugget + psill) - vgm(h, crange, model=model, psill=psill, kappa=kappa)


def vgm(h: float, crange: float, model: str = 'Sph', psill: float = 1., kappa: float = 1/2, nugget: float = 0):
    """
    Compute variogram model function (Spherical, Exponential, Gaussian or Exponential Class)

    :param h: spatial lag
    :param crange: correlation range
    :param model: model
    :param psill: partial sill
    :param kappa: smoothing parameter for Exp Class
    :param nugget: nugget

    :returns: variogram function
    """
    c0 = nugget  # nugget
    c1 = psill  # partial sill
    a1 = crange  # correlation range
    s = kappa  # smoothness parameter for Matern class

    if model == 'Sph':  # spherical model
        if h < a1:
            vgm = c0 + c1 * (3 / 2 * h / a1-1 / 2 * (h / a1) ** 3)
        else:
            vgm = c0 + c1
    elif model == 'Exp':  # exponential model
        vgm = c0 + c1 * (1-np.exp(-h / a1))
    elif model == 'Gau':  # gaussian model
        vgm = c0 + c1 * (1-np.exp(- (h / a1) ** 2))
    elif model == 'Exc':  # stable exponential model
        vgm = c0 + c1 * (1-np.exp(-(h / a1)**s))

    return vgm


def std_err_finite(std: float, neff_tot: float, neff: float) -> float:
    """
    Standard error of subsample of a finite ensemble

    :param std: standard deviation
    :param neff_tot: maximum number of effective samples
    :param neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / neff_tot * (neff_tot - neff) / neff_tot)


def std_err(std: float, neff: float) -> float:
    """
    Standard error

    :param std: standard deviation
    :param neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / neff)


def distance_latlon(tup1: tuple, tup2: tuple, earth_rad: float = 6373000) -> float:
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


def kernel_sph(xi: float, x0: float, a1: float) -> float:
    # TODO: homogenize kernel/variogram use
    """
    Spherical kernel
    :param xi: position of first point
    :param x0: position of second point
    :param a1: range of kernel

    :return: covariance between the two points
    """
    if np.abs(xi - x0) > a1:
        return 0
    else:
        return 1 - 3 / 2 * np.abs(xi-x0) / a1 + 1 / 2 * (np.abs(xi-x0) / a1) ** 3


def part_covar_sum(argsin: tuple) -> float:
    """
    Multiprocessing wrapper for covariance summing
    :param argsin: Tupled argument for covariance calculation

    :return: covariance sum
    """
    list_tuple_errs, corr_ranges, list_area_tot, list_lat, list_lon, i_range = argsin

    n = len(list_tuple_errs)
    part_var_err = 0
    for i in i_range:
        for j in range(n):
            d = distance_latlon((list_lon[i], list_lat[i]), (list_lon[j], list_lat[j]))
            for k in range(len(corr_ranges)):
                part_var_err += kernel_sph(0, d, corr_ranges[k]) * list_tuple_errs[i][k] * list_tuple_errs[j][k] * \
                    list_area_tot[i] * list_area_tot[j]

    return part_var_err


def double_sum_covar(list_tuple_errs: list[float], corr_ranges: list[float], list_area_tot: list[float],
                     list_lat: list[float], list_lon: list[float], nproc: int = 1) -> float:
    """
    Double sum of covariances for propagating multi-range correlated errors between disconnected spatial ensembles

    :param list_tuple_errs: list of tuples of correlated errors by range, by ensemble
    :param corr_ranges: list of correlation ranges
    :param list_area_tot: list of areas of ensembles
    :param list_lat: list of center latitude of ensembles
    :param list_lon: list of center longitude of ensembles
    :param nproc: number of cores to use for multiprocessing

    :returns: sum of covariances
    """
    n = len(list_tuple_errs)

    if nproc == 1:
        print('Deriving double covariance sum with 1 core...')
        var_err = 0
        for i in range(n):
            for j in range(n):
                d = distance_latlon((list_lon[i], list_lat[i]), (list_lon[j], list_lat[j]))
                for k in range(len(corr_ranges)):
                    var_err += kernel_sph(0, d, corr_ranges[k]) * list_tuple_errs[i][k] * list_tuple_errs[j][k] * \
                        list_area_tot[i] * list_area_tot[j]
    else:
        print('Deriving double covariance sum with '+str(nproc)+' cores...')
        pack_size = int(np.ceil(n/nproc))
        argsin = [(list_tuple_errs, corr_ranges, list_area_tot, list_lon, list_lat, np.arange(
            i, min(i+pack_size, n))) for k, i in enumerate(np.arange(0, n, pack_size))]
        pool = mp.Pool(nproc, maxtasksperchild=1)
        outputs = pool.map(part_covar_sum, argsin, chunksize=1)
        pool.close()
        pool.join()

        var_err = np.sum(np.array(outputs))

    area_tot = 0
    for j in range(len(list_area_tot)):
        area_tot += list_area_tot[j]

    var_err /= np.nansum(area_tot) ** 2

    return np.sqrt(var_err)


def patches_method(values: np.ndarray, gsd: float, area: float, mask: Optional[np.ndarray] = None,
                   perc_min_valid: float = 80., statistics: Iterable[Union[str, Callable, None]] = ['count', np.nanmedian ,nmad],
                   patch_shape: str = 'circular', n_patches: int = 1000, verbose: bool = False,
                   random_state: None | int | np.random.RandomState | np.random.Generator = None) -> pd.DataFrame:

    """
    Patches method for empirical estimation of the standard error over an integration area

    :param values: values
    :param gsd: ground sampling distance
    :param mask: mask of sampled terrain
    :param area: size of integration area
    :param perc_min_valid: minimum valid area in the patch
    :param statistics: list of statistics to compute in the patch
    :param patch_shape: shape of patch ['circular' or 'rectangular']
    :param n_patches: maximum number of patches to sample
    :param verbose: print statement to console
    :param random_state: random state or seed number to use for calculations (to fix random sampling during testing)

    :return: tile, mean, median, std and count of each patch
    """

    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    statistics_name = [f if isinstance(f,str) else f.__name__ for f in statistics]

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
                mask = create_circular_mask((nx, ny), center=[center_x, center_y], radius=rad)
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


def plot_vgm(df: pd.DataFrame, list_fit_fun: Optional[list[Callable[[float],float]]] = None,
             list_fit_fun_label: Optional[list[str]] = None, ax: matplotlib.axes.Axes | None = None,
             xscale='linear', xscale_range_split: Optional[list] = None,
             xlabel = None, ylabel = None, xlim = None, ylim = None):
    """
    Plot empirical variogram, and optionally also plot one or several model fits.
    Input dataframe is expected to be the output of xdem.spatialstats.sample_empirical_variogram.
    Input function model is expected to be the output of xdem.spatialstats.fit_sum_model_variogram.

    :param df: dataframe of empirical variogram
    :param list_fit_fun: list of model function fits
    :param list_fit_fun_label: list of model function fits labels
    :param ax: plotting ax to use, creates a new one by default
    :param xscale: scale of x axis
    :param xscale_range_split: list of ranges at which to split the figure
    :param xlabel: label of x axis
    :param ylabel: label of y axis
    :param xlim: limits of x axis
    :param ylim: limits of y axis
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
            xmin = [np.min(df.bins)/2]
        else:
            xmin = [0]
        xmax = [np.max(df.bins)]
        xgridmin = [0]
        xgridmax = [init_gridsize[0]]
        gridsize = init_gridsize
    # Otherwise, derive a list for each subplot
    else:
        # Add initial zero if not in input
        if xscale_range_split[0] != 0:
            if xscale == 'log':
                first_xmin = np.min(df.bins)/2
            else:
                first_xmin = 0
            xscale_range_split = [first_xmin] + xscale_range_split
        # Add maximum distance if not in input
        if xscale_range_split[-1] != np.max(df.bins):
            xscale_range_split.append(np.max(df.bins))

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
        interval_var = [0] + list(df.bins)
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

        # Get the bins center
        bins_center = np.subtract(df.bins, np.diff([0] + df.bins.tolist()) / 2)

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
            ax1.set_ylim((0, np.nanmax(df.exp)+np.nanmean(df.err_exp)))

        if k == int(nb_subpanels/2):
            ax1.set_xlabel(xlabel)
        if k == nb_subpanels - 1:
            ax1.legend(loc='best')
        if k == 0:
            ax1.set_ylabel(ylabel)
        else:
            ax1.set_yticks([])


def plot_1d_binning(df: pd.DataFrame, var_name: str, statistic_name: str, label_var: Optional[str] = None,
                    label_statistic: Optional[str] = None, min_count: int = 30, ax: matplotlib.axes.Axes | None = None):
    """
    Plot a statistic and its count along a single binning variable.
    Input is expected to be formatted as the output of the xdem.spatialstats.nd_binning function.

    :param df: output dataframe of nd_binning
    :param var_name: name of binning variable to plot
    :param statistic_name: name of statistic of interest to plot
    :param label_var: label of binning variable
    :param label_statistic: label of statistic of interest
    :param min_count: removes statistic values computed with a count inferior to this minimum value
    :param ax: plotting ax to use, creates a new one by default
    """

    # Create axes
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None")

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

    :param df: output dataframe of nd_binning
    :param var_name_1: name of first binning variable to plot
    :param var_name_2: name of second binning variable to plot
    :param statistic_name: name of statistic of interest to plot
    :param label_var_name_1: label of first binning variable
    :param label_var_name_2: label of second binning variable
    :param label_statistic: label of statistic of interest
    :param cmap: colormap
    :param min_count: removes statistic values computed with a count inferior to this minimum value
    :param scale_var_1: scale along the axis of the first variable
    :param scale_var_2: scale along the axis of the second variable
    :param vmin: minimum statistic value in colormap range
    :param vmax: maximum statistic value in colormap range
    :param nodata_color: color for no data bins
    :param ax: plotting ax to use, creates a new one by default
    """

    # Create axes
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot(111)
    elif isinstance(ax, matplotlib.axes.Axes):
        fig = ax.figure
    else:
        raise ValueError("ax must be a matplotlib.axes.Axes instance or None")

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

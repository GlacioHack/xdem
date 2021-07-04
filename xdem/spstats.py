"""Spatial statistical tools to estimate uncertainties related to DEMs"""
from __future__ import annotations

import math as m
import multiprocessing as mp
import os
import random
import warnings
from functools import partial

from typing import Callable, Union, Iterable, Optional, Sequence, Any

import itertools
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
from xdem.spatial_tools import nmad

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import skgstat as skg
    from skgstat import models

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
    >>> df = pd.DataFrame({"var1": [1, 1, 1, 2, 2, 2, 3, 3, 3], "var2": [1, 2, 3, 1, 2, 3, 1, 2, 3], "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9]})

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
        if isinstance(df_sub[var].values[0],pd.Interval):
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

    vals = df_sub[statistic_name].values
    ind_valid = np.isfinite(vals)

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
    vals = vals[ind_valid]
    # coordinates of valid values
    points_valid = tuple([df_sub[var].values[ind_valid] for var in list_var_names])
    # grid coordinates
    bmid_grid = np.meshgrid(*list_bmid)
    points_grid = tuple([bmid_grid[i].flatten() for i in range(len(list_var_names))])
    # fill grid no data with nearest neighbour
    vals_grid = griddata(points_valid, vals, points_grid, method='nearest')
    vals_grid = vals_grid.reshape(tuple(shape))

    # RegularGridInterpolator to perform linear interpolation/extrapolation on the grid
    # (will extrapolate only outside of boundaries not filled with the nearest of griddata as fill_value = None)
    interp_fun = RegularGridInterpolator(tuple(list_bmid), vals_grid, method='linear', bounds_error=False, fill_value=None)

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

def get_empirical_variogram(dh: np.ndarray, coords: np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Get empirical variogram from skgstat.Variogram object

    :param dh: elevation differences
    :param coords: coordinates
    :return: empirical variogram (variance, lags, counts)

    """
    # deriving empirical variogram variance, bin, and count
    try:
        V = skg.Variogram(coordinates=coords, values=dh, normalize=False, **kwargs)
        # return V.to_DataFrame()

        exp = V.experimental
        bins = V.bins
        count = np.zeros(V.n_lags)
        tmp_count = np.fromiter((g.size for g in V.lag_classes()), dtype=int)
        count[0:len(tmp_count)] = tmp_count

    # there are still some exceptions not well handled by skgstat
    except ZeroDivisionError:
        n_lags = kwargs.get('n_lags') or 10
        exp, bins, count = (np.zeros(n_lags)*np.nan for i in range(3))

    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, count=count)

    return df


def wrapper_get_empirical_variogram(argdict: dict, **kwargs) -> pd.DataFrame:
    """
    Multiprocessing wrapper for get_empirical_variogram

    :param argdict: Keyword argument to pass to get_empirical_variogram()

    :return: empirical variogram (variance, lags, counts)

    """
    print('Working on subsample '+str(argdict['i']) + ' out of '+str(argdict['max_i']))

    return get_empirical_variogram(dh=argdict['dh'], coords=argdict['coords'], **kwargs)


def random_subset(dh: np.ndarray, coords: np.ndarray, nsamp: int) -> tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:

    """
    Subsampling of elevation differences with random coordinates

    :param dh: elevation differences
    :param coords: coordinates
    :param nsamp: number of sammples for subsampling

    :return: subsets of dh and coords
    """
    if len(coords) > nsamp:
        # TODO: maybe we can also introduce something to sample without replacement between all samples?
        subset = np.random.choice(len(coords), nsamp, replace=False)
        coords_sub = coords[subset]
        dh_sub = dh[subset]
    else:
        coords_sub = coords
        dh_sub = dh

    return dh_sub, coords_sub

def create_circular_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None, radius: Optional[float] = None) -> np.ndarray:
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
    rr, cc = disk(center=center,radius=radius,shape=shape)
    mask[rr, cc] = True

    # manual solution
    # Y, X = np.ogrid[:h, :w]
    # dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    # mask = dist_from_center < radius

    return mask

def create_ring_mask(shape: Union[int, Sequence[int]], center: Optional[list[float]] = None, in_radius: float = 0., out_radius: Optional[float] = None) -> np.ndarray:
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

    mask_inside = create_circular_mask((w,h),center=center,radius=in_radius)
    mask_outside = create_circular_mask((w,h),center=center,radius=out_radius)

    mask_ring = np.logical_and(~mask_inside,mask_outside)

    return mask_ring


def ring_subset(dh: np.ndarray, coords: np.ndarray, inside_radius: float = 0, outside_radius: float = 0) -> tuple[Union[np.ndarray, Any], Union[np.ndarray, Any]]:
    """
    Subsampling of elevation differences within a ring/disk (to sample points at similar pairwise distances)

    :param dh: elevation differences
    :param coords: coordinates
    :param inside_radius: radius of inside ring disk in pixels
    :param outside_radius: radius of outside ring disk in pixels

    :return: subsets of dh and coords
    """

    # select random center coordinates
    nx, ny = np.shape(dh)
    center_x = np.random.choice(nx, 1)
    center_y = np.random.choice(ny, 1)

    mask_ring = create_ring_mask((nx,ny),center=(center_x,center_y),in_radius=inside_radius,out_radius=outside_radius)

    dh_ring = dh[mask_ring]
    coords_ring = coords[mask_ring]

    return dh_ring, coords_ring


def sample_multirange_empirical_variogram(dh: np.ndarray, gsd: float = None, coords: np.ndarray = None,
                                          nsamp: int = 10000, range_list: list = None, nrun: int = 1, nproc: int = 1,
                                          **kwargs) -> pd.DataFrame:
    """
    Wrapper to sample multi-range empirical variograms from the data.

    If no option is passed, a varying binning is used with adapted ranges and data subsampling

    :param dh: elevation differences
    :param gsd: ground sampling distance (if array is 2D on structured grid)
    :param coords: coordinates, to be used only with a flattened elevation differences array and passed as an array of \the pairs of coordinates: one dimension equal to two and the other to that of the flattened elevation differences
    :param range_list: successive ranges with even binning
    :param nsamp: number of samples to randomly draw from the elevation differences
    :param nrun: number of samplings
    :param nproc: number of processing cores

    :return: empirical variogram (variance, lags, counts)
    """
    # checks
    dh = dh.squeeze()
    if coords is None and gsd is None:
        raise TypeError('Must provide either coordinates or ground sampling distance.')
    elif gsd is not None and dh.ndim == 1:
        raise TypeError('Array must be 2-dimensional when providing only ground sampling distance')
    elif coords is not None and dh.ndim != 1:
        raise TypeError('Coordinate array must be provided with 1-dimensional input array')
    elif coords is not None and (coords.shape[0] != 2 and coords.shape[1] != 2):
        raise TypeError('One dimension of the coordinates array must be of length equal to 2')

    # defaulting to xx and yy if those are provided
    if coords is not None:
        if coords.shape[0] == 2 and coords.shape[1] != 2:
            coords = np.transpose(coords)
    else:
        x, y = np.meshgrid(np.arange(0, dh.shape[0] * gsd, gsd), np.arange(0, dh.shape[1] * gsd, gsd))
        coords = np.dstack((x.flatten(), y.flatten())).squeeze()
        dh = dh.flatten()

    # COMMENTING: custom binning is not supported by skgstat yet...
    # if no range list is specified, define a default one based on the spatial extent of the data and its resolution
    # if 'bin_func' not in kwargs.keys():
    #     if range_list is None:
    #
    #         # define max range as half the maximum distance between coordinates
    #         max_range = np.sqrt((np.max(coords[:,0])-np.min(coords[:,0]))**2+(np.max(coords[:,1])-np.min(coords[:,1]))**2)/2
    #
    #         # get the ground sampling distance
    #         if gsd is None:
    #             est_gsd = np.abs(coords[0,0] - coords[0,1])
    #         else:
    #             est_gsd = gsd
    #
    #         # define ranges as multiple of the resolution until they get close to the maximum range
    #         range_list = []
    #         new_range = gsd
    #         while new_range < max_range/10:
    #             range_list.append(new_range)
    #             new_range *= 10
    #         range_list.append(max_range)
    #
    # else:
    #     if range_list is not None:
    #         print('Both range_list and bin_func are defined for binning: defaulting to bin_func')

    # default value we want to use (kmeans is failing)
    if 'bin_func' not in kwargs.keys():
        kwargs.update({'bin_func': 'even'})
    if 'n_lags' not in kwargs.keys():
        kwargs.update({'n_lags': 100})

    # estimate variogram
    if nrun == 1:
        # subsetting
        dh_sub, coords_sub = random_subset(dh, coords, nsamp)
        # getting empirical variogram
        print(dh_sub.shape)
        print(coords_sub.shape)
        df = get_empirical_variogram(dh=dh_sub, coords=coords_sub, **kwargs)
        df['exp_sigma'] = np.nan

    else:

        # multiple run only work for an even binning function for now (would need a customized binning not supported by skgstat)
        if kwargs.get('bin_func') is None:
            raise ValueError('Binning function must be "even" when doing multiple runs.')

        # define max range as half the maximum distance between coordinates
        max_range = np.sqrt((np.max(coords[:, 0])-np.min(coords[:, 0]))**2 +
                            (np.max(coords[:, 1])-np.min(coords[:, 1]))**2)/2
        # also need a cutoff value to get the exact same bins
        if 'maxlag' not in kwargs.keys():
            kwargs.update({'maxlag': max_range})

        # TODO: somewhere here we could think of adding random sampling without replacement
        if nproc == 1:
            print('Using 1 core...')
            list_df_nb = []
            for i in range(nrun):
                dh_sub, coords_sub = random_subset(dh, coords, nsamp)
                df = get_empirical_variogram(dh=dh_sub, coords=coords_sub, **kwargs)
                df['run'] = i
                list_df_nb.append(df)
        else:
            print('Using '+str(nproc) + ' cores...')
            list_dh_sub = []
            list_coords_sub = []
            for i in range(nrun):
                dh_sub, coords_sub = random_subset(dh, coords, nsamp)
                list_dh_sub.append(dh_sub)
                list_coords_sub.append(coords_sub)

            pool = mp.Pool(nproc, maxtasksperchild=1)
            argsin = [{'dh': list_dh_sub[i], 'coords': list_coords_sub[i], 'i':i, 'max_i':nrun} for i in range(nrun)]
            list_df = pool.map(partial(wrapper_get_empirical_variogram, **kwargs), argsin, chunksize=1)
            pool.close()
            pool.join()

            list_df_nb = []
            for i in range(10):
                df_nb = list_df[i]
                df_nb['run'] = i
                list_df_nb.append(df_nb)

        df = pd.concat(list_df_nb)

        # group results, use mean as empirical variogram, estimate sigma, and sum the counts
        df_grouped = df.groupby('bins', dropna=False)
        df_mean = df_grouped[['exp']].mean()
        df_sig = df_grouped[['exp']].std()
        df_count = df_grouped[['count']].sum()
        df_mean['bins'] = df_mean.index.values
        df_mean['exp_sigma'] = df_sig['exp']
        df_mean['count'] = df_count['count']
        df = df_mean

    return df



def fit_model_sum_vgm(list_model: list[str], emp_vgm_df: pd.DataFrame) -> tuple[Callable, list[float]]:
    """
    Fit a multi-range variogram model to an empirical variogram, weighted based on sampling and elevation errors

    :param list_model: list of variogram models to sum for the fit: from short-range to long-ranges
    :param emp_vgm_df: empirical variogram

    :return: modelled variogram function, coefficients
    """
    # TODO: expand to other models than spherical, exponential, gaussian (more than 2 arguments)
    def vgm_sum(h, *args):
        fn = 0
        i = 0
        for model in list_model:
            fn += skg.models.spherical(h, args[i], args[i+1])
            # fn += vgm(h, model=model,crange=args[i],psill=args[i+1])
            i += 2

        return fn

    # use shape of empirical variogram to assess rough boundaries/first estimates
    n_average = np.ceil(len(emp_vgm_df.exp.values) / 10)
    exp_movaverage = np.convolve(emp_vgm_df.exp.values, np.ones(int(n_average))/n_average, mode='valid')
    grad = np.gradient(exp_movaverage, 2)
    # maximum variance
    max_var = np.max(exp_movaverage)

    # to simplify things for scipy, let's provide boundaries and first guesses
    p0 = []
    bounds = []
    for i in range(len(list_model)):

        # use largest boundaries possible for our problem
        psill_bound = [0, max_var]
        range_bound = [0, emp_vgm_df.bins.values[-1]]

        # use psill evenly distributed
        psill_p0 = ((i+1)/len(list_model))*max_var
        # use corresponding ranges

        # this fails when no empirical value crosses this (too wide binning/nugget)
        # ind = np.array(np.abs(exp_movaverage-psill_p0)).argmin()
        # range_p0 = emp_vgm_df.bins.values[ind]
        range_p0 = ((i+1)/len(list_model))*emp_vgm_df.bins.values[-1]

        # TODO: if adding other variogram models, add condition here

        # add bounds and guesses with same order as function arguments
        bounds.append(range_bound)
        bounds.append(psill_bound)

        p0.append(range_p0)
        p0.append(psill_p0)

    bounds = np.transpose(np.array(bounds))

    if np.all(np.isnan(emp_vgm_df.exp_sigma.values)):
        valid = ~np.isnan(emp_vgm_df.exp.values)
        cof, cov = curve_fit(vgm_sum, emp_vgm_df.bins.values[valid],
                             emp_vgm_df.exp.values[valid], method='trf', p0=p0, bounds=bounds)
    else:
        valid = np.logical_and(~np.isnan(emp_vgm_df.exp.values), ~np.isnan(emp_vgm_df.exp_sigma.values))
        cof, cov = curve_fit(vgm_sum, emp_vgm_df.bins.values[valid], emp_vgm_df.exp.values[valid],
                             method='trf', p0=p0, bounds=bounds, sigma=emp_vgm_df.exp_sigma.values[valid])

    # rewriting the output function: couldn't find a way to pass this with functool.partial because arguments are unordered
    def vgm_sum_fit(h):
        fn = 0
        i = 0
        for model in list_model:
            fn += skg.models.spherical(h, cof[i], cof[i+1])
            # fn += vgm(h, model=model,crange=args[i],psill=args[i+1])
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


def patches_method(dh : np.ndarray, mask: np.ndarray[bool], gsd : float, area_size : float, perc_min_valid: float = 80.,
                   patch_shape: str = 'circular',nmax : int = 1000, verbose: bool = False) -> pd.DataFrame:

    """
    Patches method for empirical estimation of the standard error over an integration area

    :param dh: elevation differences
    :param mask: mask of sampled terrain
    :param gsd: ground sampling distance
    :param area_size: size of integration area
    :param perc_min_valid: minimum valid area in the patch
    :param patch_shape: shape of patch ['circular' or 'rectangular']
    :param nmax: maximum number of patch to sample

    #TODO: add overlap option?

    :return: tile, mean, median, std and count of each patch
    """

    # first, remove non sampled area (but we need to keep the 2D shape of raster for patch sampling)
    dh = dh.squeeze()
    valid_mask = np.logical_and(np.isfinite(dh), mask)
    dh[~valid_mask] = np.nan

    # divide raster in cadrants where we can sample
    nx, ny = np.shape(dh)
    count = len(dh[~np.isnan(dh)])
    print('Number of valid pixels: ' + str(count))
    nb_cadrant = int(np.floor(np.sqrt((count * gsd ** 2) / area_size) + 1))
    # rectangular
    nx_sub = int(np.floor((nx - 1) / nb_cadrant))
    ny_sub = int(np.floor((ny - 1) / nb_cadrant))
    # radius size for a circular patch
    rad = int(np.floor(np.sqrt(area_size/np.pi * gsd ** 2)))

    tile, mean_patch, med_patch, std_patch, nb_patch = ([] for i in range(5))

    # create list of all possible cadrants
    list_cadrant = [[i, j] for i in range(nb_cadrant) for j in range(nb_cadrant)]
    u = 0
    # keep sampling while there is cadrants left and below maximum number of patch to sample
    while len(list_cadrant) > 0 and u < nmax:

        check = 0
        while check == 0:
            rand_cadrant = random.randint(0, len(list_cadrant)-1)

            i = list_cadrant[rand_cadrant][0]
            j = list_cadrant[rand_cadrant][1]

            check_x = int(np.floor(nx_sub*(i+1/2)))
            check_y = int(np.floor(ny_sub*(j+1/2)))
            if mask[check_x, check_y]:
                check = 1

        list_cadrant.remove(list_cadrant[rand_cadrant])

        tile.append(str(i) + '_' + str(j))
        if patch_shape == 'rectangular':
            patch = dh[nx_sub * i:nx_sub * (i + 1), ny_sub * j:ny_sub * (j + 1)].flatten()
        elif patch_shape == 'circular':
            center_x = np.floor(nx_sub*(i+1/2))
            center_y = np.floor(ny_sub*(j+1/2))
            mask = create_circular_mask((nx, ny), center=(center_x, center_y), radius=rad)
            patch = dh[mask]
        else:
            raise ValueError('Patch method must be rectangular or circular.')

        nb_pixel_total = len(patch)
        nb_pixel_valid = len(patch[np.isfinite(patch)])
        if nb_pixel_valid > np.ceil(perc_min_valid / 100. * nb_pixel_total):
            u=u+1
            if verbose:
                print('Found valid cadrant ' + str(u)+ ' (maximum: '+str(nmax)+')')

            mean_patch.append(np.nanmean(patch))
            med_patch.append(np.nanmedian(patch.filled(np.nan) if isinstance(patch, np.ma.masked_array) else patch))
            std_patch.append(np.nanstd(patch))
            nb_patch.append(nb_pixel_valid)

    df = pd.DataFrame()
    df = df.assign(tile=tile, mean=mean_patch, med=med_patch, std=std_patch, count=nb_patch)

    return df


def plot_vgm(df: pd.DataFrame, list_fit_fun: Optional[list[Callable[[float],float]]] = None,
             list_fit_fun_label: Optional[list[str]] = None):
    """
    Plot empirical variogram, with optionally one or several model fits
    :param df: dataframe of empirical variogram
    :param list_fit_fun: list of function fits
    :param list_fit_fun_label: list of function fits labels
    :param
    :return:
    """
    fig, ax = plt.subplots(1)
    if np.all(np.isnan(df.exp_sigma)):
        ax.scatter(df.bins, df.exp, label='Empirical variogram', color='blue')
    else:
        ax.errorbar(df.bins, df.exp, yerr=df.exp_sigma, label='Empirical variogram (1-sigma s.d)')

    if list_fit_fun is not None:
        for i, fit_fun in enumerate(list_fit_fun):
            x = np.linspace(0, np.max(df.bins), 10000)
            y = fit_fun(x)

            if list_fit_fun_label is not None:
                ax.plot(x, y, linestyle='dashed', label=list_fit_fun_label[i], zorder=30)
            else:
                ax.plot(x, y, linestyle='dashed', color='black', zorder=30)

        if list_fit_fun_label is None:
            ax.plot([],[],linestyle='dashed',color='black',label='Model fit')

    ax.set_xlabel('Lag (m)')
    ax.set_ylabel(r'Variance [$\mu$ $\pm \sigma$]')
    ax.legend(loc='best')

    return ax

def plot_1d_binning(df: pd.DataFrame, var_name: str, statistic_name: str, label_var: Optional[str] = None,
                    label_statistic: Optional[str] = None, min_count: int = 30):
    """
    Plot one statistic and its count along a single binning variable.
    Input is expected to be formatted as the output of the nd_binning function.

    :param df: output dataframe of nd_binning
    :param var_name: name of binning variable to plot
    :param statistic_name: name of statistic of interest to plot
    :param label_var: label of binning variable
    :param label_statistic: label of statistic of interest
    :param min_count: removes statistic values computed with a count inferior to this minimum value
    """

    if label_var is None:
        label_var = var_name
    if label_statistic is None:
        label_statistic = statistic_name

    # Subsample to 1D and for the variable of interest
    df_sub = df[np.logical_and(df.nd == 1, np.isfinite(pd.IntervalIndex(df[var_name]).mid))].copy()
    # Remove statistic calculated in bins with too low count
    df_sub.loc[df_sub['count']<min_count, statistic_name] = np.nan

    # Need a grid plot to show the sample count and the statistic
    fig = plt.figure()
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)

    # First, an axis to plot the sample histogram
    ax0 = fig.add_subplot(grid[:3, :])
    ax0.set_xticks([])

    # Plot the histogram manually with fill_between
    interval_var = pd.IntervalIndex(df_sub[var_name])
    for i in range(len(df_sub) ):
        count = df_sub['count'].values[i]
        ax0.fill_between([interval_var[i].left, interval_var[i].right], [0] * 2, [count] * 2, facecolor=plt.cm.Greys(0.75), alpha=1,
                         edgecolor='white',linewidth=0.1)
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
    ax = fig.add_subplot(grid[3:, :])

    ax.scatter(interval_var.mid, df_sub[statistic_name],marker='x')
    ax.set_xlabel(label_var)
    ax.set_ylabel(label_statistic)
    ax.set_xlim((np.min(interval_var.left),np.max(interval_var.right)))


def plot_2d_binning(df: pd.DataFrame, var_name_1: str, var_name_2: str, statistic_name: str,
                    label_var_name_1: Optional[str] = None, label_var_name_2: Optional[str] = None,
                    label_statistic: Optional[str] = None, cmap: colors.LinearSegmentedColormap = plt.cm.Reds, min_count: int = 30,
                    scale_var_1: str = 'linear', scale_var_2: str = 'linear', vmin: float = None, vmax: float = None,
                    nodata_color: Union[str,tuple[float,float,float,float]] ='yellow'):
    """
    Plot one statistic and its count along two binning variables.
    Input is expected to be formatted as the output of the nd_binning function.

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
    """


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
    fig = plt.figure(figsize=(10,7.5))
    grid = plt.GridSpec(10, 10, wspace=0.5, hspace=0.5)

    # First, an horizontal axis on top to plot the sample histogram of the first variable
    ax0 = fig.add_subplot(grid[:3, :-3])
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
                         edgecolor='white')
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
    ax1 = fig.add_subplot(grid[3:, -3:])
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
                         alpha=1, edgecolor='white')
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
    ax = fig.add_subplot(grid[3:, :-3])

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

            ax.fill_between([df_both[var_name_1].values[0].left, df_both[var_name_1].values[0].right], [df_both[var_name_2].values[0].left] * 2,
                            [df_both[var_name_2].values[0].right] * 2, facecolor=col, alpha=1, edgecolor='white')

    ax.set_xlabel(label_var_name_1)
    ax.set_ylabel(label_var_name_2)
    ax.set_xscale(scale_var_1)
    ax.set_yscale(scale_var_2)
    # In case the axis value does not agree with the scale (e.g., 0 for log scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.set_xlim((np.min(interval_var_1.left),np.max(interval_var_1.right)))
        ax.set_ylim((np.min(interval_var_2.left),np.max(interval_var_2.right)))

    # Fourth and finally, add a colormap and nodata color to the legend
    axcmap = fig.add_subplot(grid[:3, -3:])

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

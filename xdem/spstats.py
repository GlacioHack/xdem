"""
xdem.spstats provides tools to use spatial statistics for elevation change data
"""
from __future__ import annotations

import math as m
import multiprocessing as mp
import os
import random
import warnings
from functools import partial
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.optimize import curve_fit

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import skgstat as skg
    from skgstat import models


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


def random_subset(dh: np.ndarray, coords: np.ndarray, nsamp: int):

    # TODO: add methods that might be more relevant with the multi-distance sampling?
    """
    Subsampling of elevation differences

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

    :return: modelled variogram
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
    Number of effective samples derived from exact integration of sum of spherical variogram models over a circular area.
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


def neff_circ(area: float, list_vgm: list[Union[float, str, float]]) -> float:
    """
    Number of effective samples derived from numerical integration for any sum of variogram models a circular area
    (generalization of Rolstad et al. (2009): http://dx.doi.org/10.3189/002214309789470950)
    The number of effective samples N_eff serves to convert between standard deviation/partial sills and standard error
    over the area: SE = SD / sqrt(N_eff) if SE is the standard error, SD the standard deviation.

    :param area: area
    :param list_vgm: variogram functions to sum

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


def patches_method(dh: np.ndarray, mask: np.ndarray[bool], gsd: float, area_size: float, perc_min_valid: float = 80.,
                   patch_shape: str = 'circular', nmax: int = 1000) -> pd.DataFrame:
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
    def create_circular_mask(h, w, center=None, radius=None):
        """
        Create circular mask on a raster

        :param h: height position
        :param w: width position
        :param center: center
        :param radius: radius
        :return:
        """

        if center is None:  # use the middle of the image
            center = [int(w / 2), int(h / 2)]
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius

        return mask

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
            mask = create_circular_mask(nx, ny, center=[center_x, center_y], radius=rad)
            patch = dh[mask]
        else:
            raise ValueError('Patch method must be rectangular or circular.')

        nb_pixel_total = len(patch)
        nb_pixel_valid = len(patch[np.isfinite(patch)])
        if nb_pixel_valid > np.ceil(perc_min_valid / 100. * nb_pixel_total):
            u = u+1
            print('Found valid cadrant ' + str(u) + ' (maximum: '+str(nmax)+')')
            mean_patch.append(np.nanmean(patch))
            med_patch.append(np.nanmedian(patch.filled(np.nan) if isinstance(patch, np.ma.masked_array) else patch))
            std_patch.append(np.nanstd(patch))
            nb_patch.append(nb_pixel_valid)

    df = pd.DataFrame()
    df = df.assign(tile=tile, mean=mean_patch, med=med_patch, std=std_patch, count=nb_patch)

    return df


def plot_vgm(df: pd.DataFrame, fit_fun: Callable = None):

    fig, ax = plt.subplots(1)
    if np.all(np.isnan(df.exp_sigma)):
        ax.scatter(df.bins, df.exp, label='Empirical variogram', color='blue')
    else:
        ax.errorbar(df.bins, df.exp, yerr=df.exp_sigma, label='Empirical variogram (1-sigma s.d)')

    if fit_fun is not None:
        x = np.linspace(0, np.max(df.bins), 10000)
        y = fit_fun(x)

        ax.plot(x, y, linestyle='dashed', color='black', label='Model fit', zorder=30)

    ax.set_xlabel('Lag (m)')
    ax.set_ylabel(r'Variance [$\mu$ $\pm \sigma$]')
    ax.legend(loc='best')
    ax.grid()
    plt.show()

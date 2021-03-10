"""
xdem.spstats provides tools to use spatial statistics for elevation change data
"""
from __future__ import annotations
import os
import numpy as np
import math as m
import skgstat as skg
from scipy import integrate
import multiprocessing as mp
import pandas as pd
import ogr
from functools import partial

def get_empirical_variogram(dh: np.ndarray,coords: np.ndarray, **kwargs) -> pd.DataFrame:
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
    except:
        if 'n_lags' in kwargs.keys():
            n_lags = kwargs.get('n_lags')
        else:
            n_lags = 10
        exp, bins, count = (np.zeros(n_lags)*np.nan for i in range(3))

    df = pd.DataFrame()
    df = df.assign(exp=exp, bins=bins, n=count)

    return df

def wrapper_get_empirical_variogram(argdict: dict,**kwargs) -> pd.DataFrame:

    """
    Multiprocessing wrapper for get_empirical_variogram

    :param argdict: Keyword argument to pass to get_empirical_variogram()

    :return: empirical variogram (variance, lags, counts)

    """
    print('Working on subsample '+str(argdict['i'])+ ' out of '+str(argdict['max_i']))

    return get_empirical_variogram(dh=argdict['dh'],coords=argdict['coords'],**kwargs)

def random_subset(dh,coords,nsamp,method='random_index'):

    #TODO: add methods that might be more relevant with the multi-distance sampling?
    """
    Subsampling of elevation differences

    :param dh: elevation differences
    :param coords: coordinates
    :param method:

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
                                          nsamp: int = 10000, range_list: list= None, nrun=1, nproc=1, **kwargs) -> pd.DataFrame:

    """
    Wrapper to sample multi-range empirical variograms from the data.
    If no option is passed, a varying binning is used with adapted ranges and data subsampling

    :param dh: elevation differences
    :param gsd: ground sampling distance (if array is 2D on structured grid)
    :param coords: coordinates (if array is 1D)
    :param range_list: successive ranges with even binning
    :param nsamp: number of samples to randomly draw from the elevation differences
    :param nrun: number of samplings

    :return: empirical variogram (variance, lags, counts)
    """

    # checks
    if coords is None and gsd is None:
        raise TypeError('Must provide either coordinates or ground sampling distance.')
    elif gsd is not None and len(dh.shape) == 1:
        raise TypeError('Array must be 2-dimensional when providing only ground sampling distance')
    elif coords is not None and len(dh.shape) != 1:
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

    # default value we want to use
    if 'bin_func' not in kwargs.keys():
        kwargs.update({'bin_func':'kmeans'})
    if 'n_lags' not in kwargs.keys():
        kwargs.update({'n_lags':100})

    # estimate variogram
    if nrun == 1:
        # subsetting
        dh_sub, coords_sub = random_subset(dh, coords, nsamp)
        # getting empirical variogram
        df = get_empirical_variogram(dh=dh_sub, coords=coords_sub, **kwargs)

    else:
        # TODO: somewhere here we could think of adding random sampling without replacement
        if nproc == 1:
            print('Using 1 core...')
            list_df_nb = []
            for i in range(nrun):
                dh_sub, coords_sub = random_subset(dh,coords,nsamp)
                df = get_empirical_variogram(dh=dh_sub, coords=coords_sub, **kwargs)
                df['run'] = i
                list_df_nb.append(df)
        else:
            print('Using '+str(nproc)+ 'cores...')
            list_dh_sub = []
            list_coords_sub = []
            for i in range(nrun):
                dh_sub, coords_sub = random_subset(dh, coords, nsamp)
                list_dh_sub.append(dh_sub)
                list_coords_sub.append(coords_sub)

            pool = mp.Pool(nproc, maxtasksperchild=1)
            argsin = [{'dh': list_dh_sub[i], 'coords_sub': list_coords_sub[i],'i':i,'max_i':nrun} for i in range(nrun)]
            list_df = pool.map(partial(wrapper_get_empirical_variogram,**kwargs), argsin, chunksize=1)
            pool.close()
            pool.join()

            list_df_nb = []
            for i in range(10):
                df_nb = list_df[i]
                df_nb['run'] = i
                list_df_nb.append(df_nb)
        df = pd.concat(list_df_nb)

    return df


def exact_neff_sphsum_circular(area: float,crange1: float,psill1: float,crange2: float,psill2: float) -> float:
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

    #short range variogram
    c1 = psill1 # partial sill
    a1 = crange1  # short correlation range

    #long range variogram
    c1_2 = psill2
    a1_2 = crange2 # long correlation range

    h_equiv = np.sqrt(area / np.pi)

    #hypothesis of a circular shape to integrate variogram model
    if h_equiv > a1_2:
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * a1_2 ** 2 / (5 * h_equiv ** 2))
    elif (h_equiv < a1_2) and (h_equiv > a1):
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))
    else:
        std_err = np.sqrt(c1 * (1-h_equiv / a1+1 / 5 * (h_equiv / a1) ** 3) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))

    return (psill1 + psill2)/std_err**2

def neff_circ(area: float,list_vgm: list) -> float:
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
            fn += h*(cov(h,crange,model=model,psill=psill))

        return fn

    h_equiv = np.sqrt(area / np.pi)

    full_int = integrate_fun(hcov_sum,0,h_equiv)
    std_err = np.sqrt(2*np.pi*full_int / area)

    return psill_tot/std_err**2


def neff_rect(area: float,width: float,crange1: float,psill1: float,model1: str ='Sph',crange2: float = None,
              psill2: float = None,model2: str = None) -> float:
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

    def hcov_sum(h,crange1=crange1,psill1=psill1,model1=model1,crange2=crange2,psill2=psill2,model2=model2):

        if crange2 is None or psill2 is None or model2 is None:
            return h*(cov(h,crange1,model=model1,psill=psill1))
        else:
            return h*(cov(h,crange1,model=model1,psill=psill1)+cov(h,crange2,model=model2,psill=psill2))

    width = min(width,area/width)

    full_int = integrate_fun(hcov_sum,0,width/2)
    bin_int = np.linspace(width/2,area/width,100)
    for i in range(len(bin_int)-1):
        low = bin_int[i]
        upp = bin_int[i+1]
        mid = bin_int[i] + (bin_int[i+1]- bin_int[i])/2
        piec_int = integrate_fun(hcov_sum, low, upp)
        full_int += piec_int * 2/np.pi*np.arctan(width/(2*mid))

    std_err = np.sqrt(2*np.pi*full_int / area)

    if crange2 is None or psill2 is None or model2 is None:
        return psill1 / std_err ** 2
    else:
        return (psill1 + psill2) / std_err ** 2


def integrate_fun(fun: Union[function,LowLevelCallable],low_b: float ,upp_b: float) -> float:
    """
    Numerically integrate function between upper and lower bounds
    :param fun: function
    :param low_b: lower bound
    :param upp_b: upper bound

    :return: integral
    """

    return integrate.quad(fun,low_b,upp_b)[0]

def cov(h: float,crange: float,model: str='Sph',psill: float=1.,kappa: float=1/2,nugget: float=0) -> function:
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

    return (nugget + psill) - vgm(h,crange,model=model,psill=psill,kappa=kappa)

def vgm(h: float ,crange: float,model: str='Sph',psill: float=1.,kappa:float=1/2,nugget:float=0):
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


    c0 = nugget #nugget
    c1 = psill #partial sill
    a1 = crange #correlation range
    s = kappa #smoothness parameter for Matern class

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
        vgm = c0 + c1 * (1-np.exp(-(h/ a1)**s))

    return vgm

def std_err_finite(std: float, Neff: float, neff: float) -> float:
    """
    Standard error of subsample of a finite ensemble

    :param std: standard deviation
    :param Neff: maximum number of effective samples
    :param neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / Neff * (Neff - neff) / Neff)

def std_err(std: float, Neff: float) -> float:
    """
    Standard error

    :param std: standard deviation
    :param Neff: number of effective samples

    :return: standard error
    """
    return std * np.sqrt(1 / Neff)

def distance_latlon(tup1: tuple,tup2: tuple) -> float:
    """
    Distance between two lat/lon coordinates projected on a spheroid
    :param tup1: lon/lat coordinates of first point
    :param tup2: lon/lat coordinates of second point

    :return: distance
    """

    # approximate radius of earth in km
    R = 6373000

    lat1 = m.radians(abs(tup1[1]))
    lon1 = m.radians(abs(tup1[0]))
    lat2 = m.radians(abs(tup2[1]))
    lon2 = m.radians(abs(tup2[0]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = m.sin(dlat / 2)**2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon / 2)**2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    distance = R * c

    return distance

def kernel_sph(xi: float,x0: float,a1: float)-> float:
    #TODO: homogenize kernel/variogram use
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


def part_covar_sum(argsin:tuple) -> float:
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

def double_sum_covar(list_tuple_errs: list, corr_ranges: list, list_area_tot: list, list_lat: list, list_lon: list,
                     nproc: int=1):

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

    if nproc==1:
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
        argsin = [(list_tuple_errs,corr_ranges,list_area_tot,list_lon,list_lat,np.arange(i,min(i+pack_size,n))) for k, i in enumerate(np.arange(0,n,pack_size))]
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



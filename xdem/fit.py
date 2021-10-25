"""
Functions to perform normal, weighted and robust fitting.
"""
from __future__ import annotations

import inspect
from typing import Callable, Union, Sized, Optional
import warnings

import numpy as np
import pandas as pd
import scipy.optimize

from xdem.spatialstats import nd_binning
from geoutils.spatial_tools import subsample_raster

try:
    from sklearn.metrics import mean_squared_error, median_absolute_error
    from sklearn.linear_model import (
        LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    _has_sklearn = True
except ImportError:
    _has_sklearn = False

def rmse(z: np.ndarray) -> float:
    """
    Return root mean square error
    :param z: Residuals between predicted and true value
    :return: Root Mean Square Error
    """
    return np.sqrt(np.nanmean(np.square(z)))

def huber_loss(z: np.ndarray) -> float:
    """
    Huber loss cost (reduces the weight of outliers)
    :param z: Residuals between predicted and true values
    :return: Huber cost
    """
    out = np.where(z > 1, 2 * np.sqrt(z[np.where(z > 1)]) - 1, np.square(z))

    return out.sum()

def soft_loss(z: np.ndarray, scale = 0.5) -> float:
    """
    Soft loss cost (reduces the weight of outliers)
    :param z: Residuals between predicted and true values
    :param scale: Scale factor
    :return: Soft loss cost
    """
    return np.sum(np.square(scale) * 2 * (np.sqrt(1 + np.square(z/scale)) - 1))

def _costfun_sumofsin(p, x, y, cost_func):
    """
    Calculate robust cost function for sum of sinusoids
    """
    z = y - _sumofsinval(x, p)
    return cost_func(z)


def _choice_best_order(cost: np.ndarray, margin_improvement : float = 20., verbose: bool = False) -> int:
    """
    Choice of the best order (polynomial, sum of sinusoids) with a margin of improvement. The best cost value does
    not necessarily mean the best predictive fit because high-degree polynomials tend to overfit, and sum of sinusoids
    as well. To mitigate this issue, we should choose the lesser order from which improvement becomes negligible.
    :param cost: cost function residuals to the polynomial
    :param margin_improvement: improvement margin (percentage) below which the lesser degree polynomial is kept
    :param verbose: if text should be printed

    :return: degree: degree for the best-fit polynomial
    """

    # get percentage of spread from the minimal cost
    ind_min = cost.argmin()
    min_cost = cost[ind_min]
    perc_cost_improv = (cost - min_cost) / min_cost

    # costs below threshold and lesser degrees
    below_margin = np.logical_and(perc_cost_improv < margin_improvement / 100., np.arange(len(cost))<=ind_min)
    costs_below_thresh = cost[below_margin]
    # minimal costs
    subindex = costs_below_thresh.argmin()
    # corresponding index (degree)
    ind = np.arange(len(cost))[below_margin][subindex]

    if verbose:
        print('Order '+str(ind_min+1)+ ' has the minimum cost value of '+str(min_cost))
        print('Order '+str(ind+1)+ ' is selected within a '+str(margin_improvement)+' % margin of'
            ' the minimum cost, with a cost value of '+str(min_cost))

    return ind

def _wrapper_scipy_leastsquares(residual_func, p0, x, y, verbose, **kwargs):
    """
    Wrapper function for scipy.optimize.least_squares: passes down keyword, extracts cost and final parameters, print
    statements in the console

    :param residual_func: Residual function to fit
    :param p0: Initial guess
    :param x: X vector
    :param y: Y vector
    :param verbose: Whether to print out statements
    :return:
    """

    # Get arguments of scipy.optimize
    fun_args = scipy.optimize.least_squares.__code__.co_varnames[:scipy.optimize.least_squares.__code__.co_argcount]
    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in fun_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: ' + ','.join(list(remaining_kwargs.keys())) + ' were not used.')
    # Filter corresponding arguments before passing
    filtered_kwargs = {k: kwargs[k] for k in fun_args if k in kwargs}

    # Run function with associated keyword arguments
    myresults = scipy.optimize.least_squares(residual_func, p0, args=(x, y), **filtered_kwargs)
    if verbose:
        print("Initial Parameters: ", p0)
        print("Status: ", myresults.success, " - ", myresults.status)
        print(myresults.message)
        print("Lowest cost:", myresults.cost)
        print("Parameters:", myresults.x)
    cost = myresults.cost
    coefs = myresults.x

    return cost, coefs

def _wrapper_sklearn_robustlinear(model, estimator_name, cost_func, x, y, **kwargs):
    """
    Wrapper function of sklearn.linear_models: passes down keyword, extracts cost and final parameters, sets random
    states, scales input and de-scales output data, prints out statements

    :param model: Function model to fit (e.g., Polynomial features)
    :param estimator_name: Linear estimator to use (one of "Linear", "Theil-Sen", "RANSAC" and "Huber")
    :param cost_func: Cost function to use for optimization
    :param x: X vector
    :param y: Y vector
    :return:
    """
    # Select sklearn estimator
    dict_estimators = {'Linear': LinearRegression, 'Theil-Sen': TheilSenRegressor,
                       'RANSAC': RANSACRegressor, 'Huber': HuberRegressor}

    est = dict_estimators[estimator_name]

    # Get existing arguments of the sklearn estimator and model
    estimator_args = list(inspect.signature(est.__init__).parameters.keys())

    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in estimator_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn('Keyword arguments: ' + ','.join(list(remaining_kwargs.keys())) + ' were not used.')
    # Filter corresponding arguments before passing
    filtered_kwargs = {k: kwargs[k] for k in estimator_args if k in kwargs}

    # TODO: Find out how to re-scale polynomial coefficient + doc on what is the best scaling for polynomials
    # # Scale output data (important for ML algorithms):
    # robust_scaler = RobustScaler().fit(x.reshape(-1,1))
    # x_scaled = robust_scaler.transform(x.reshape(-1,1))
    # # Fit scaled data
    # model.fit(x_scaled, y)
    # y_pred = model.predict(x_scaled)

    # Initialize estimator with arguments
    init_estimator = est(**filtered_kwargs)

    # Create pipeline
    pipeline = make_pipeline(model, init_estimator)

    # Run with data
    pipeline.fit(x.reshape(-1, 1), y)
    y_pred = pipeline.predict(x.reshape(-1, 1))

    # Calculate cost
    cost = cost_func(y_pred, y)

    # Get polynomial coefficients estimated with the estimators Linear, Theil-Sen and Huber
    if estimator_name in ['Linear','Theil-Sen','Huber']:
        coefs = init_estimator.coef_
    # For some reason RANSAC doesn't store coef at the same place
    else:
        coefs = init_estimator.estimator_.coef_

    return cost, coefs


def robust_polynomial_fit(x: np.ndarray, y: np.ndarray, max_order: int = 6, estimator_name: str  = 'Theil-Sen',
                          cost_func: Callable = median_absolute_error, margin_improvement : float = 20.,
                          subsample: Union[float,int] = 25000, linear_pkg = 'sklearn', verbose: bool = False,
                          random_state: None | np.random.RandomState | np.random.Generator | int = None, **kwargs) -> tuple[np.ndarray,int]:
    """
    Given 1D vectors x and y, compute a robust polynomial fit to the data. Order is chosen automatically by comparing
    residuals for multiple fit orders of a given estimator.
    Any keyword argument will be passed down to scipy.optimize.least_squares and sklearn linear estimators.

    :param x: input x data (N,)
    :param y: input y data (N,)
    :param max_order: maximum polynomial order tried for the fit
    :param estimator_name: robust estimator to use, one of 'Linear', 'Theil-Sen', 'RANSAC' or 'Huber'
    :param cost_func: cost function taking as input two vectors y (true y), y' (predicted y) of same length
    :param margin_improvement: improvement margin (percentage) below which the lesser degree polynomial is kept
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param linear_pkg: package to use for Linear estimator, one of 'scipy' and 'sklearn'
    :param random_state: random seed for testing purposes
    :param verbose: if text should be printed

    :returns coefs, degree: polynomial coefficients and degree for the best-fit polynomial
    """
    if not isinstance(estimator_name, str) or estimator_name not in ['Linear','Theil-Sen','RANSAC','Huber']:
        raise ValueError('Attribute estimator must be one of "Linear", "Theil-Sen", "RANSAC" or "Huber".')
    if not isinstance(linear_pkg, str) or linear_pkg not in ['sklearn','scipy']:
        raise ValueError('Attribute linear_pkg must be one of "scipy" or "sklearn".')

    # Remove NaNs
    valid_data = np.logical_and(np.isfinite(y), np.isfinite(x))
    x = x[valid_data]
    y = y[valid_data]

    # Subsample data
    subsamp = subsample_raster(x, subsample=subsample, return_indices=True, random_state=random_state)
    x = x[subsamp]
    y = y[subsamp]

    # Initialize cost function and output coefficients
    list_costs = np.empty(max_order)
    list_coeffs = np.zeros((max_order, max_order + 1))
    # Loop on polynomial degrees
    for deg in np.arange(1, max_order + 1):
        # If method is linear and package scipy
        if estimator_name == 'Linear' and linear_pkg == 'scipy':

            # Define the residual function to optimize with scipy
            def fitfun_polynomial(xx, params):
                return sum([p * (xx ** i) for i, p in enumerate(params)])
            def residual_func(p, xx, yy):
                return fitfun_polynomial(xx, p) - yy
            # Define the initial guess
            p0 = np.polyfit(x, y, deg)

            # Run the linear method with scipy
            cost, coef = _wrapper_scipy_leastsquares(residual_func, p0, x, y, verbose=verbose, **kwargs)

        else:
            # Otherwise, we use sklearn
            if not _has_sklearn:
                raise ValueError("Optional dependency needed. Install 'scikit-learn'")

            # Define the polynomial model to insert in the pipeline
            model = PolynomialFeatures(degree=deg)

            # Run the linear method with sklearn
            cost, coef = _wrapper_sklearn_robustlinear(model, estimator_name=estimator_name, cost_func=cost_func,
                                                       x=x, y=y, **kwargs)

        list_costs[deg - 1] = cost
        list_coeffs[deg - 1, 0:coef.size] = coef

    # Choose the best polynomial with a margin of improvement on the cost
    final_index = _choice_best_order(cost=list_costs, margin_improvement=margin_improvement, verbose=verbose)

    # The degree of the best polynomial corresponds to the index plus one
    return np.trim_zeros(list_coeffs[final_index], 'b'), final_index + 1


def _sumofsinval(x: np.array, params: np.ndarray) -> np.ndarray:
    """
    Function for a sum of N frequency sinusoids
    :param x: array of coordinates (N,)
    :param p: list of tuples with amplitude, frequency and phase parameters
    """
    aix = np.arange(0, params.size, 3)
    bix = np.arange(1, params.size, 3)
    cix = np.arange(2, params.size, 3)

    val = np.sum(params[aix] * np.sin(2 * np.pi / params[bix] * x[:, np.newaxis] + params[cix]), axis=1)

    return val

def robust_sumsin_fit(x: np.ndarray, y: np.ndarray, nb_frequency_max: int = 3,
                      bounds_amp_freq_phase: Optional[list[tuple[float,float], tuple[float,float], tuple[float,float]]] = None,
                      cost_func: Callable = soft_loss, subsample: Union[float,int] = 25000, hop_length : Optional[float] = None,
                      random_state: None | np.random.RandomState | np.random.Generator | int = None, verbose: bool = False) -> tuple[np.ndarray,int]:
    """
    Given 1D vectors x and y, compute a robust sum of sinusoid fit to the data. The number of frequency is chosen
    automatically by comparing residuals for multiple fit orders of a given estimator.
    Any keyword argument will be passed down to scipy.optimize.basinhopping.

    :param x: input x data (N,)
    :param y: input y data (N,)
    :param nb_frequency_max: maximum number of phases
    :param bounds_amp_freq_phase: bounds for amplitude, frequency and phase (L, 3, 2) and
    with mean value used for initialization
    :param hop_length: jump in function values to optimize basinhopping algorithm search (for best results, should be
    comparable to the separation (in function value) between local minima)
    :param cost_func: cost function taking as input two vectors y (true y), y' (predicted y) of same length
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param random_state: random seed for testing purposes
    :param verbose: if text should be printed

    :returns coefs, degree: polynomial coefficients and degree for the best-fit polynomial
    """

    def wrapper_costfun_sumofsin(p, x, y):
        return _costfun_sumofsin(p, x, y, cost_func=cost_func)

    # First, remove NaNs
    valid_data = np.logical_and(np.isfinite(y), np.isfinite(x))
    x = x[valid_data]
    y = y[valid_data]

    # If no significant resolution is provided, assume that it is the mean difference between sampled X values
    if hop_length is None:
        x_sorted = np.sort(x)
        hop_length = np.mean(np.diff(x_sorted))

    # Use binned statistics for first guess
    nb_bin = int((x.max() - x.min()) / (5 * hop_length))
    df = nd_binning(y, [x], ['var'], list_var_bins=nb_bin, statistics=[np.nanmedian])
    # Compute first guess for x and y
    x_fg = pd.IntervalIndex(df['var']).mid.values
    y_fg = df['nanmedian']
    valid_fg = np.logical_and(np.isfinite(x_fg),np.isfinite(y_fg))
    x_fg = x_fg[valid_fg]
    y_fg = y_fg[valid_fg]

    # Loop on all frequencies
    costs = np.empty(nb_frequency_max)
    amp_freq_phase = np.zeros((nb_frequency_max, 3*nb_frequency_max)) * np.nan

    for nb_freq in np.arange(1, nb_frequency_max+1):

        b = bounds_amp_freq_phase
        # If bounds are not provided, define as the largest possible bounds
        if b is None:
            lb_amp = 0
            ub_amp = (y_fg.max() - y_fg.min()) / 2
            # Define for phase
            lb_phase = 0
            ub_phase = 2 * np.pi
            # Define for the frequency, we need at least 5 points to see any kind of periodic signal
            lb_frequency = 1 / (5 * (x.max() - x.min()))
            ub_frequency = 1 / (5 * hop_length)

            b = []
            for i in range(nb_freq):
                b += [(lb_amp,ub_amp),(lb_frequency,ub_frequency),(lb_phase,ub_phase)]

        # Format lower and upper bounds for scipy
        lb = np.asarray(([b[i][0] for i in range(3*nb_freq)]))
        ub = np.asarray(([b[i][1] for i in range(3*nb_freq)]))
        # Insert in a scipy bounds object
        scipy_bounds = scipy.optimize.Bounds(lb, ub)
        # First guess for the mean parameters
        p0 = np.divide(lb + ub, 2)

        # Initialize with the first guess
        init_args = dict(args=(x_fg, y_fg), method="L-BFGS-B",
                         bounds=scipy_bounds, options={"ftol": 1E-6})
        init_results = scipy.optimize.basinhopping(wrapper_costfun_sumofsin, p0, disp=verbose,
                                                   T=hop_length, minimizer_kwargs=init_args, seed=random_state)
        init_results = init_results.lowest_optimization_result

        # Subsample the final raster
        subsamp = subsample_raster(x, subsample=subsample, return_indices=True, random_state=random_state)
        x = x[subsamp]
        y = y[subsamp]

        # Minimize the globalization with a larger number of points
        minimizer_kwargs = dict(args=(x, y),
                                method="L-BFGS-B",
                                bounds=scipy_bounds,
                                options={"ftol": 1E-6})
        myresults = scipy.optimize.basinhopping(wrapper_costfun_sumofsin, init_results.x, disp=verbose,
                                                T=5 * hop_length, niter_success=40,
                                                minimizer_kwargs=minimizer_kwargs, seed=random_state)
        myresults = myresults.lowest_optimization_result
        # Write results for this number of frequency
        costs[nb_freq-1] = wrapper_costfun_sumofsin(myresults.x,x,y)
        amp_freq_phase[nb_freq -1, 0:3*nb_freq] = myresults.x

    final_index = _choice_best_order(cost=costs)

    final_coefs =  amp_freq_phase[final_index][~np.isnan(amp_freq_phase[final_index])]

    # If an amplitude coefficient is almost zero, remove the coefs of that frequency and lower the degree
    final_degree = final_index + 1
    for i in range(final_index+1):
        if np.abs(final_coefs[3*i]) < (np.nanpercentile(x, 90) - np.nanpercentile(x, 10))/1000:
            final_coefs = np.delete(final_coefs, slice(3*i,3*i+3))
            final_degree -= 1
            break

    # The number of frequencies corresponds to the final index plus one
    return final_coefs, final_degree


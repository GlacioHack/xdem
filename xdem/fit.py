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

"""
Functions to perform normal, weighted and robust fitting.
"""
from __future__ import annotations

import inspect
import logging
import warnings
from typing import Any, Callable

import numpy as np
import scipy
from geoutils.raster import subsample_array
from numpy.polynomial.polynomial import polyval, polyval2d

from xdem._typing import NDArrayf

try:
    from sklearn.linear_model import (
        HuberRegressor,
        LinearRegression,
        RANSACRegressor,
        TheilSenRegressor,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    _has_sklearn = True
except ImportError:
    _has_sklearn = False


def rmse(ytrue: NDArrayf, ypred: NDArrayf) -> float:
    """
    Return root mean square error

    :param ytrue: True values
    :param ypred: Predicted values

    :return: Root mean square error
    """
    return np.sqrt(np.nanmean(np.square(ytrue - ypred)))


def huber_loss(ytrue: NDArrayf, ypred: NDArrayf) -> float:
    """
    Huber loss cost (reduces the weight of outliers)

    :param ytrue: True values
    :param ypred: Predicted values

    :return: Huber cost
    """
    z = ytrue - ypred
    out = np.where(z > 1, 2 * np.sqrt(z[np.where(z > 1)]) - 1, np.square(z))

    return out.sum()


def soft_loss(ytrue: NDArrayf, ypred: NDArrayf, scale: float = 0.5) -> float:
    """
    Soft loss cost (reduces the weight of outliers)

    :param ytrue: True values
    :param ypred: Predicted values
    :param scale: Scale factor

    :return: Soft loss cost
    """
    return np.sum(np.square(scale) * 2 * (np.sqrt(1 + np.square((ytrue - ypred) / scale)) - 1))


######################################################
# Most common functions for 1- or 2-D bias corrections
######################################################


def sumsin_1d(xx: NDArrayf, *params: NDArrayf) -> NDArrayf:
    """
    Sum of N sinusoids in 1D.

    :param xx: Array of coordinates.
    :param params: 3 x N parameters in order of amplitude (Y unit), wavelength (X unit) and phase (radians).
    """

    # Squeeze input in case it is a 1-D tuple or such
    xx = np.array(xx).squeeze()

    # Convert parameters to array
    p = np.array(params).copy()

    # Indexes of amplitude, frequencies and phases
    aix = np.arange(0, len(p), 3)
    bix = np.arange(1, len(p), 3)
    cix = np.arange(2, len(p), 3)

    # Expand array to the same size as data.dim + 1, and move params to axis 0 for sum (ndmin moves it to last axis)
    p = np.moveaxis(np.array(p, ndmin=xx.ndim + 1), source=xx.ndim, destination=0)

    # Perform the sum of sinusoid
    val = np.sum(p[aix, :] * np.sin(2 * np.pi / p[bix, :] * np.expand_dims(xx, axis=0) + p[cix, :]), axis=0)

    return val


def polynomial_1d(xx: NDArrayf, *params: NDArrayf) -> NDArrayf:
    """
    N-order 1D polynomial.

    :param xx: 1D array of values.
    :param params: N polynomial parameters.

    :return: Output value.
    """
    return polyval(x=xx, c=params)


def polynomial_2d(xx: tuple[NDArrayf, NDArrayf], *params: NDArrayf) -> NDArrayf:
    """
    N-order 2D polynomial.

    :param xx: The two 1D array of values.
    :param params: The N parameters (a, b, c, etc.) of the polynomial.

    :returns: Output value.
    """

    # The number of parameters of np.polyval2d is order^2, so a square array needs to be passed
    poly_order = np.sqrt(len(params))

    if not poly_order.is_integer():
        raise ValueError(
            "The parameters of the 2D polynomial should have a length equal to order^2, "
            "see np.polyval2d for more details."
        )

    # We reshape the parameter into the N x N shape expected by NumPy
    c = np.array(params).reshape((int(poly_order), int(poly_order)))

    return polyval2d(x=xx[0], y=xx[1], c=c)


#######################################################################
# Convenience wrappers for robust N-order polynomial or sum of sin fits
#######################################################################


def _choice_best_order(cost: NDArrayf, margin_improvement: float = 20.0) -> int:
    """
    Choice of the best order (polynomial, sum of sinusoids) with a margin of improvement. The best cost value does
    not necessarily mean the best predictive fit because high-degree polynomials tend to overfit, and sum of sinusoids
    as well. To mitigate this issue, we should choose the lesser order from which improvement becomes negligible.

    :param cost: cost function residuals to the polynomial
    :param margin_improvement: improvement margin (percentage) below which the lesser degree polynomial is kept

    :return: degree: degree for the best-fit polynomial
    """

    # Get percentage of spread from the minimal cost
    ind_min = cost.argmin()
    min_cost = cost[ind_min]
    perc_cost_improv = (cost - min_cost) / min_cost

    # Keep only good-performance costs that are within 20% of the minimal cost
    below_margin = np.logical_and(perc_cost_improv < margin_improvement / 100.0, np.arange(len(cost)) <= ind_min)

    # Choose the good-performance cost with lowest degree
    ind = next((i for i, j in enumerate(below_margin) if j))

    logging.debug("Order " + str(ind_min + 1) + " has the minimum cost value of " + str(min_cost))
    logging.debug(
        "Order " + str(ind + 1) + " is selected as its cost is within a " + str(margin_improvement) + "% margin of"
        " the minimum cost"
    )

    return ind


def _wrapper_scipy_leastsquares(
    f: Callable[..., NDArrayf],
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None = None,
    p0: NDArrayf = None,
    **kwargs: Any,
) -> tuple[float, NDArrayf]:
    """
    Wrapper function for scipy.optimize.least_squares: passes down keyword, extracts cost and final parameters, print
    statements in the console

    :param f: Function to fit.
    :param p0: Initial guess.
    :param x: X vector.
    :param y: Y vector.
    :return:
    """

    # Get arguments of scipy.optimize.curve_fit and subfunction least_squares
    fun_args = scipy.optimize.curve_fit.__code__.co_varnames[: scipy.optimize.curve_fit.__code__.co_argcount]
    ls_args = scipy.optimize.least_squares.__code__.co_varnames[: scipy.optimize.least_squares.__code__.co_argcount]

    all_args = list(fun_args) + list(ls_args)

    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in all_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn("Keyword arguments: " + ",".join(list(remaining_kwargs.keys())) + " were not used.")
    # Filter corresponding arguments before passing
    filtered_kwargs = {k: kwargs[k] for k in all_args if k in kwargs}

    # Run function with associated keyword arguments
    coefs = scipy.optimize.curve_fit(
        f=f,
        xdata=xdata,
        ydata=ydata,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        **filtered_kwargs,
    )[0]

    # Round results above the tolerance to get fixed results on different OS
    coefs = np.array([np.round(coef, 5) for coef in coefs])

    # If a specific loss function was passed, construct it to get the cost
    if "loss" in kwargs.keys():
        loss = kwargs["loss"]
        if "f_scale" in kwargs.keys():
            f_scale = kwargs["f_scale"]
        else:
            f_scale = 1.0
        from scipy.optimize._lsq.least_squares import construct_loss_function

        loss_func = construct_loss_function(m=ydata.size, loss=loss, f_scale=f_scale)
        cost = 0.5 * sum(np.atleast_1d(loss_func((f(xdata, *coefs) - ydata) ** 2, cost_only=True)))
    # Default is linear loss
    else:
        cost = 0.5 * sum((f(xdata, *coefs) - ydata) ** 2)

    return cost, coefs


def _wrapper_sklearn_robustlinear(
    model: PolynomialFeatures,
    cost_func: Callable[[NDArrayf, NDArrayf], float],
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None = None,
    estimator_name: str = "Linear",
    **kwargs: Any,
) -> tuple[float, NDArrayf]:
    """
    Wrapper function of sklearn.linear_models: passes down keyword, extracts cost and final parameters, sets random
    states, scales input and de-scales output data, prints out statements

    :param model: Function model to fit (e.g., Polynomial features)
    :param cost_func: Cost function taking as input two vectors y (true y), y' (predicted y) of same length.
    :param xdata: X vector
    :param ydata: Y vector
    :param estimator_name: Linear estimator to use (one of "Linear", "Theil-Sen", "RANSAC" and "Huber")
    :return:
    """
    # Select sklearn estimator
    dict_estimators = {
        "Linear": LinearRegression,
        "Theil-Sen": TheilSenRegressor,
        "RANSAC": RANSACRegressor,
        "Huber": HuberRegressor,
    }

    est = dict_estimators[estimator_name]

    # Get existing arguments of the sklearn estimator and model
    estimator_args = list(inspect.signature(est.__init__).parameters.keys())

    # Check no other argument is left to be passed
    remaining_kwargs = kwargs.copy()
    for arg in estimator_args:
        remaining_kwargs.pop(arg, None)
    if len(remaining_kwargs) != 0:
        warnings.warn("Keyword arguments: " + ",".join(list(remaining_kwargs.keys())) + " were not used.")
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
    # The sample weight can only be passed if it exists in the estimator call
    if sigma is not None and "sample_weight" in inspect.signature(est.fit).parameters.keys():
        # The weight is the inverse of the squared standard error
        sample_weight = 1 / sigma**2
        # The argument name to pass it through a pipeline is "estimatorname__sample_weight"
        args = {est.__name__.lower() + "__sample_weight": sample_weight}
        pipeline.fit(xdata.reshape(-1, 1), ydata, *args)
    else:
        pipeline.fit(xdata.reshape(-1, 1), ydata)

    y_pred = pipeline.predict(xdata.reshape(-1, 1))

    # Calculate cost
    cost = cost_func(ydata, y_pred)

    # Get polynomial coefficients estimated with the estimators Linear, Theil-Sen and Huber
    if estimator_name in ["Linear", "Theil-Sen", "Huber"]:
        coefs = init_estimator.coef_
    # For some reason RANSAC doesn't store coef at the same place
    else:
        coefs = init_estimator.estimator_.coef_

    return cost, coefs


def robust_norder_polynomial_fit(
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None = None,
    max_order: int = 6,
    estimator_name: str = "Theil-Sen",
    cost_func: Callable[[NDArrayf, NDArrayf], float] = soft_loss,
    margin_improvement: float = 20.0,
    subsample: float | int = 1,
    linear_pkg: str = "scipy",
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> tuple[NDArrayf, int]:
    """
    Given 1D vectors x and y, compute a robust polynomial fit to the data. Order is chosen automatically by comparing
    residuals for multiple fit orders of a given estimator.

    Any keyword argument will be passed down to scipy.optimize.least_squares and sklearn linear estimators.

    :param xdata: Input x data (N,).
    :param ydata: Input y data (N,).
    :param sigma: Standard error of y data (N,).
    :param max_order: Maximum polynomial order tried for the fit.
    :param estimator_name: Robust estimator to use, one of 'Linear', 'Theil-Sen', 'RANSAC' or 'Huber'.
    :param cost_func: Cost function taking as input two vectors y (true y), y' (predicted y) of same length.
    :param margin_improvement: improvement margin (percentage) below which the lesser degree polynomial is kept.
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of pixels to extract.
    :param linear_pkg: package to use for Linear estimator, one of 'scipy' and 'sklearn'.
    :param random_state: Random seed.

    :returns coefs, degree: Polynomial coefficients and degree for the best-fit polynomial
    """
    # Remove "f" and "absolute sigma" arguments passed, as both are fixed here
    if "f" in kwargs.keys():
        kwargs.pop("f")
    if "absolute_sigma" in kwargs.keys():
        kwargs.pop("absolute_sigma")

    # Raise errors for input string parameters
    if not isinstance(estimator_name, str) or estimator_name not in ["Linear", "Theil-Sen", "RANSAC", "Huber"]:
        raise ValueError('Attribute `estimator` must be one of "Linear", "Theil-Sen", "RANSAC" or "Huber".')
    if not isinstance(linear_pkg, str) or linear_pkg not in ["sklearn", "scipy"]:
        raise ValueError('Attribute `linear_pkg` must be one of "scipy" or "sklearn".')

    # Extract xdata from iterable
    if len(xdata) == 1:
        xdata = xdata[0]

    # Remove NaNs
    valid_data = np.logical_and(np.isfinite(ydata), np.isfinite(xdata))
    x = xdata[valid_data]
    y = ydata[valid_data]

    # Subsample data
    if subsample != 1:
        subsamp = subsample_array(x, subsample=subsample, return_indices=True, random_state=random_state)
        x = x[subsamp]
        y = y[subsamp]

    # Initialize cost function and output coefficients
    list_costs = np.empty(max_order)
    list_coeffs = np.zeros((max_order, max_order + 1))

    # Loop on polynomial degrees
    for deg in np.arange(1, max_order + 1):
        # If method is linear and package scipy
        if estimator_name == "Linear" and linear_pkg == "scipy":

            # Define the initial guess
            p0 = np.polyfit(x, y, deg)

            # Run the linear method with scipy
            try:
                cost, coef = _wrapper_scipy_leastsquares(
                    f=polynomial_1d, xdata=x, ydata=y, p0=p0, sigma=sigma, **kwargs
                )
            except RuntimeError:
                cost = np.inf
                coef = np.array([np.nan for i in range(len(p0))])

        else:
            # Otherwise, we use sklearn
            if not _has_sklearn:
                raise ValueError("Optional dependency needed. Install 'scikit-learn'.")

            # Define the polynomial model to insert in the pipeline
            model = PolynomialFeatures(degree=deg)

            # Run the linear method with sklearn
            cost, coef = _wrapper_sklearn_robustlinear(
                model, estimator_name=estimator_name, cost_func=cost_func, xdata=x, ydata=y, sigma=sigma, **kwargs
            )

        list_costs[deg - 1] = cost
        list_coeffs[deg - 1, 0 : coef.size] = coef

    # Choose the best polynomial with a margin of improvement on the cost
    final_index = _choice_best_order(cost=list_costs, margin_improvement=margin_improvement)

    # The degree of the best polynomial corresponds to the index plus one
    return np.trim_zeros(list_coeffs[final_index], "b"), final_index + 1


def _cost_sumofsin(
    x: NDArrayf,
    y: NDArrayf,
    cost_func: Callable[[NDArrayf, NDArrayf], float],
    *p: NDArrayf,
) -> float:
    """
    Calculate robust cost function for sum of sinusoids
    """
    return cost_func(y, sumsin_1d(x, *p))


def robust_nfreq_sumsin_fit(
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None = None,
    max_nb_frequency: int = 3,
    bounds_amp_wave_phase: list[tuple[float, float]] | None = None,
    cost_func: Callable[[NDArrayf, NDArrayf], float] = soft_loss,
    subsample: float | int = 1,
    hop_length: float | None = None,
    random_state: int | np.random.Generator | None = None,
    **kwargs: Any,
) -> tuple[NDArrayf, int]:
    """
    Given 1D vectors x and y, compute a robust sum of sinusoid fit to the data. The number of frequency is chosen
    automatically by comparing residuals for multiple fit orders of a given estimator.

    Any keyword argument will be passed down to scipy.optimize.basinhopping.

    :param xdata: Input x data (N,).
    :param ydata: Input y data (N,).
    :param sigma: Standard error of y data (N,).
    :param max_nb_frequency: Maximum number of sinusoid of different frequencies.
    :param bounds_amp_wave_phase: Bounds for amplitude, wavelength and phase (L, 3, 2) and
        with mean value used for initialization.
    :param hop_length: Jump in function values to optimize basinhopping algorithm search (for best results, should be
    comparable to the separation in function value between local minima).
    :param cost_func: Cost function taking as input two vectors y (true y), y' (predicted y) of same length.
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of pixels to extract.
    :param random_state: Random seed.
    :param kwargs: Keyword arguments to pass to scipy.optimize.basinhopping

    :returns coefs, degree: sinusoid coefficients (amplitude, frequency, phase) x N, Number N of summed sinusoids
    """

    # Remove "f" and "absolute sigma" arguments passed, as both are fixed here
    if "f" in kwargs.keys():
        kwargs.pop("f")
    if "absolute_sigma" in kwargs.keys():
        kwargs.pop("absolute_sigma")

    # Extract xdata from iterable
    if len(xdata) == 1:
        xdata = xdata[0]

    # Check if there is a number of iterations to stop the run if the global minimum candidate remains the same.
    if "niter_success" not in kwargs.keys():
        # Check if there is a number of basin-hopping iterations passed down to the function.
        if "niter" not in kwargs.keys():
            niter_success = 40
        else:
            niter_success = min(40, kwargs.get("niter"))  # type: ignore

        kwargs.update({"niter_success": niter_success})

    def wrapper_cost_sumofsin(p: NDArrayf, x: NDArrayf, y: NDArrayf) -> float:
        return _cost_sumofsin(x, y, cost_func, *p)

    # If no significant resolution is provided, assume that it is the mean difference between sampled X values
    x_res = np.mean(np.diff(np.sort(xdata)))

    # The hop length will condition jump in function values, needs of magnitude slightly lower than the signal
    if hop_length is None:
        hop = float(np.percentile(ydata, 90) - np.percentile(ydata, 10))
    else:
        hop = hop_length

    # Loop on all frequencies
    costs = np.empty(max_nb_frequency)
    amp_freq_phase = np.zeros((max_nb_frequency, 3 * max_nb_frequency)) * np.nan

    for nb_freq in np.arange(1, max_nb_frequency + 1):

        logging.info("Fitting with %d frequency", nb_freq)

        b = bounds_amp_wave_phase
        # If bounds are not provided, define as the largest possible bounds
        if b is None:
            # For the amplitude, from Y values
            lb_amp = 0
            ub_amp = ydata.max() - ydata.min()
            # For phase: all possible values for a sinusoid
            lb_phase = 0
            ub_phase = 2 * np.pi
            # For the wavelength: from the resolution and coordinate extent
            # (we don't want the lower bound to be zero, to avoid divisions by zero)
            lb_wavelength = x_res / 5
            ub_wavelength = xdata.max() - xdata.min()

            b = []
            for _i in range(nb_freq):
                b += [(lb_amp, ub_amp), (lb_wavelength, ub_wavelength), (lb_phase, ub_phase)]

        # Format lower and upper bounds for scipy
        lb = np.asarray([b[i][0] for i in range(3 * nb_freq)])
        ub = np.asarray([b[i][1] for i in range(3 * nb_freq)])
        # Insert in a scipy bounds object
        scipy_bounds = scipy.optimize.Bounds(lb, ub)
        # First guess for the mean parameters
        p0 = (np.abs((lb + ub) / 2)).squeeze()

        logging.debug("Bounds")
        logging.debug(lb)
        logging.debug(ub)

        # Minimize the globalization with a larger number of points
        minimizer_kwargs = dict(args=(xdata, ydata), bounds=scipy_bounds)
        myresults = scipy.optimize.basinhopping(
            wrapper_cost_sumofsin,
            p0,
            disp=logging.getLogger().getEffectiveLevel() < logging.WARNING,
            T=hop,
            minimizer_kwargs=minimizer_kwargs,
            seed=random_state,
            **kwargs,
        )
        myresults = myresults.lowest_optimization_result
        myresults_x = np.array([np.round(myres, 5) for myres in myresults.x])

        logging.info("Final result")
        logging.info(myresults_x)

        # Write results for this number of frequency
        costs[nb_freq - 1] = wrapper_cost_sumofsin(myresults_x, xdata, ydata)
        amp_freq_phase[nb_freq - 1, 0 : 3 * nb_freq] = myresults_x

    # Replace NaN cost by infinity
    costs[np.isnan(costs)] = np.inf

    logging.info("Costs")
    logging.info(costs)

    final_index = _choice_best_order(cost=costs)

    final_coefs = amp_freq_phase[final_index][~np.isnan(amp_freq_phase[final_index])]

    logging.info("Selecting best performing number of frequencies:")
    logging.info(final_coefs)

    # If an amplitude coefficient is almost zero, remove the coefs of that frequency and lower the degree
    final_degree = final_index + 1
    for i in range(final_index + 1):
        # If an amplitude has an estimated value of less than 0.1% the signal bounds (percentiles for robustness)
        # And if the degree is higher than 2 (need at least degree 1 return)
        if (
            np.abs(final_coefs[3 * i]) < (np.nanpercentile(ydata, 90) - np.nanpercentile(ydata, 10)) / 1000
            and len(final_coefs) > 3
        ):
            final_coefs = np.delete(final_coefs, slice(3 * i, 3 * i + 3))
            final_degree -= 1
            break

    # Re-order frequencies by highest amplitude
    amplitudes = final_coefs[0::3]
    indices = np.flip(np.argsort(amplitudes))
    new_amplitudes = amplitudes[indices]
    new_wavelengths = final_coefs[1::3][indices]
    new_phases = final_coefs[2::3][indices]

    final_coefs = np.array(
        [(new_amplitudes[i], new_wavelengths[i], new_phases[i]) for i in range(final_degree)]
    ).flatten()

    # The number of frequencies corresponds to the final index plus one
    return final_coefs, final_degree

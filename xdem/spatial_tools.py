"""Basic operations to be run on 2D arrays and DEMs"""
from __future__ import annotations

from typing import Callable, Union, Sized, Optional
import warnings

import itertools
import geoutils as gu
from geoutils.georaster import RasterType
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
import rasterio as rio
import rasterio.warp
from tqdm import tqdm
import numba
import skimage.transform

from xdem.misc import deprecate
from xdem.spatialstats import nd_binning

try:
    from sklearn.metrics import mean_squared_error, median_absolute_error
    from sklearn.linear_model import (
        LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    _has_sklearn = True
except ImportError:
    _has_sklearn = False

def get_mask(array: Union[np.ndarray, np.ma.masked_array]) -> np.ndarray:
    """
    Return the mask of invalid values, whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.

    :returns invalid_mask: boolean array, True where array is masked or Nan.
    """
    mask = (array.mask | ~np.isfinite(array.data)) if isinstance(array, np.ma.masked_array) else ~np.isfinite(array)
    return mask.squeeze()


def get_array_and_mask(
        array: np.ndarray | np.ma.masked_array | RasterType,
        check_shape: bool = True,
        copy: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return array with masked values set to NaN and the associated mask.
    Works whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.
    :param check_shape: Validate that the array is either a 1D array, a 2D array or a 3D array of shape (1, rows, cols).
    :param copy: Return a copy of 'array'. If False, a view will be attempted (and warn if not possible)

    :returns array_data, invalid_mask: a tuple of ndarrays. First is array with invalid pixels converted to NaN, \
    second is mask of invalid pixels (True if invalid).
    """
    if isinstance(array, gu.Raster):
        array = array.data

    if check_shape:
        if len(array.shape) > 2 and array.shape[0] > 1:
            raise ValueError(
                    f"Invalid array shape given: {array.shape}."
                    "Expected 2D array or 3D array where arr.shape[0] == 1"
            )

    # If an occupied mask exists and a view was requested, trigger a warning.
    if not copy and np.any(getattr(array, "mask", False)):
        warnings.warn("Copying is required to respect the mask. Returning copy. Set 'copy=True' to hide this message.")
        copy = True

    # If array is of type integer and has a mask, it needs to be converted to float (to assign nans)
    if np.any(getattr(array, "mask", False)) and np.issubdtype(array.dtype, np.integer):
        array = array.astype('float32')

    # Convert into a regular ndarray (a view or copy depending on the 'copy' argument)
    array_data = np.array(array).squeeze() if copy else np.asarray(array).squeeze()

    # Get the mask of invalid pixels and set nans if it is occupied.
    invalid_mask = get_mask(array)
    if np.any(invalid_mask):
        array_data[invalid_mask] = np.nan

    return array_data, invalid_mask


def get_valid_extent(array: Union[np.ndarray, np.ma.masked_array]) -> tuple:
    """
    Return (rowmin, rowmax, colmin, colmax), the first/last row/column of array with valid pixels
    """
    if not array.dtype == 'bool':
        valid_mask = ~get_mask(array)
    else:
        valid_mask = array
    cols_nonzero = np.where(np.count_nonzero(valid_mask, axis=0) > 0)[0]
    rows_nonzero = np.where(np.count_nonzero(valid_mask, axis=1) > 0)[0]
    return rows_nonzero[0], rows_nonzero[-1], cols_nonzero[0], cols_nonzero[-1]


def nmad(data: np.ndarray, nfact: float = 1.4826) -> float:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.

    :param data: input data
    :param nfact: normalization factor for the data; default is 1.4826

    :returns nmad: (normalized) median absolute deviation of data.
    """
    if isinstance(data, np.ma.masked_array):
        data_arr = get_array_and_mask(data, check_shape=False)[0]
    else:
        data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))

def resampling_method_from_str(method_str: str) -> rio.warp.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.warp.Resampling:
        if str(method).replace("Resampling.", "") == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.warp.Resampling method. "
            f"Valid methods: {[str(method).replace('Resampling.', '') for method in rio.warp.Resampling]}"
        )
    return resampling_method


@deprecate(
        removal_version="0.0.6",
        details=(
            "This function is redundant after the '-' operator for rasters was introduced."
            " Use 'dem1 - dem2.reproject(dem1, resampling_method='cubic_spline')' instead."
        )
)
def subtract_rasters(first_raster: Union[str, gu.georaster.Raster], second_raster: Union[str, gu.georaster.Raster],
                     reference: str = "first",
                     resampling_method: Union[str, rio.warp.Resampling] = "cubic_spline") -> gu.georaster.Raster:
    """
    Subtract one raster with another.

    difference = first_raster - reprojected_second_raster,
    OR
    difference = reprojected_first_raster - second_raster,

    depending on which raster is acting "reference".

    :param first_raster: The first raster in the equation.
    :param second_raster: The second raster in the equation.
    :param reference: Which raster to provide the reference bounds, CRS and resolution (can be "first" or "second").

    :raises: ValueError: If any of the given arguments are invalid.

    :returns: A raster of the difference between first_raster and second_raster.
    """
    # If the arguments are filepaths, load them as GeoUtils rasters.
    if isinstance(first_raster, str):
        first_raster = gu.georaster.Raster(first_raster)
    if isinstance(second_raster, str):
        second_raster = gu.georaster.Raster(second_raster)

    # Make sure that the reference string is valid
    if reference not in ["first", "second"]:
        raise ValueError(f"Invalid reference string: '{reference}', must be either 'first' or 'second'")
    # Parse the resampling method if given as a string.
    if isinstance(resampling_method, str):
        resampling_method = resampling_method_from_str(resampling_method)

    # Reproject the non-reference and subtract the two rasters.
    difference = \
        first_raster.data - second_raster.reproject(first_raster, resampling=resampling_method, silent=True).data if \
        reference == "first" else \
        first_raster.reproject(second_raster, resampling=resampling_method, silent=True).data - second_raster.data

    # Generate a GeoUtils raster from the difference array
    difference_raster = gu.georaster.Raster.from_array(
        difference,
        transform=first_raster.transform if reference == "first" else second_raster.transform,
        crs=first_raster.crs if reference == "first" else second_raster.crs,
        nodata=first_raster.nodata if reference == "first" else second_raster.nodata
    )

    return difference_raster


def merge_bounding_boxes(bounds: list[rio.coords.BoundingBox], resolution: float) -> rio.coords.BoundingBox:
    max_bounds = dict(zip(["left", "right", "top", "bottom"], [np.nan] * 4))
    for bound in bounds:
        for key in "right", "top":
            max_bounds[key] = np.nanmax([max_bounds[key], bound.__getattribute__(key)])
        for key in "bottom", "left":
            max_bounds[key] = np.nanmin([max_bounds[key], bound.__getattribute__(key)])

    # Make sure that extent is a multiple of resolution
    for key1, key2 in zip(("left", "bottom"), ("right", "top")):
        modulo = (max_bounds[key2] - max_bounds[key1]) % resolution
        max_bounds[key2] += modulo

    return rio.coords.BoundingBox(**max_bounds)


def stack_rasters(rasters: list[gu.georaster.Raster], reference: Union[int, gu.Raster] = 0,
                  resampling_method: Union[str, rio.warp.Resampling] = "bilinear", use_ref_bounds: bool = False,
                  diff: bool = False, progress: bool = True) -> gu.georaster.Raster:
    """
    Stack a list of rasters into a common grid as a 3D np array with nodata set to Nan.

    If use_ref_bounds is True, output will have the shape (N, height, width) where N is len(rasters) and \
height and width is equal to reference's shape.
    If use_ref_bounds is False, output will have the shape (N, height2, width2) where N is len(rasters) and \
height2 and width2 are set based on reference's resolution and the maximum extent of all rasters.

    Use diff=True to return directly the difference to the reference raster.

    Note that currently all rasters will be loaded once in memory. However, if rasters data is not loaded prior to \
    merge_rasters it will be loaded for reprojection and deleted, therefore avoiding duplication and \
    optimizing memory usage.

    :param rasters: A list of geoutils Raster objects to be stacked.
    :param reference: The reference index, in case the reference is to be stacked, or a separate Raster object \
 in case the reference should not be stacked. Defaults to the first raster in the list.
    :param resampling_method: The resampling method for the raster reprojections.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.
    :param diff: If True, will return the difference to the reference, rather than the DEMs.
    :param progress: If True, will display a progress bar. Default is True.

    :returns: The stacked raster with the same parameters (optionally bounds) as the reference.
    """
    # Check resampling method
    if isinstance(resampling_method, str):
        resampling_method = resampling_method_from_str(resampling_method)

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Set output bounds
    if use_ref_bounds:
        dst_bounds = reference_raster.bounds
    else:
        dst_bounds = merge_bounding_boxes([raster.bounds for raster in rasters], resolution=reference_raster.res[0])

    # Make a data list and add all of the reprojected rasters into it.
    data: list[np.ndarray] = []

    for raster in tqdm(rasters, disable=~progress):

        # Check that data is loaded, otherwise temporarily load it
        if not raster.is_loaded:
            raster.load()
            raster.is_loaded = False

        # Reproject to reference grid
        reprojected_raster = raster.reproject(
            dst_bounds=dst_bounds,
            dst_res=reference_raster.res,
            dst_crs=reference_raster.crs,
            dtype=reference_raster.data.dtype,
            nodata=reference_raster.nodata
        )

        # Optionally calculate difference
        if diff:
            ddem = (reference_raster.data - reprojected_raster.data).squeeze()
            ddem, _ = get_array_and_mask(ddem)
            data.append(ddem)
        else:
            dem, _ = get_array_and_mask(reprojected_raster.data.squeeze())
            data.append(dem)

        # Remove unloaded rasters
        if not raster.is_loaded:
            raster._data = None

    # Convert to numpy array
    data = np.asarray(data)

    # Save as gu.Raster
    raster = gu.georaster.Raster.from_array(
        data=data,
        transform=rio.transform.from_bounds(
            *dst_bounds, width=data[0].shape[1], height=data[0].shape[0]
        ),
        crs=reference_raster.crs,
        nodata=reference_raster.nodata
    )

    return raster


def merge_rasters(rasters: list[gu.georaster.Raster], reference: Union[int, gu.Raster] = 0,
                  merge_algorithm: Union[Callable, list[Callable]] = np.nanmean,
                  resampling_method: Union[str, rio.warp.Resampling] = "bilinear",
                  use_ref_bounds = False) -> gu.georaster.Raster:
    """
    Merge a list of rasters into one larger raster.

    Reprojects the rasters to the reference raster CRS and resolution.
    Note that currently all rasters will be loaded once in memory. However, if rasters data is not loaded prior to \
    merge_rasters it will be loaded for reprojection and deleted, therefore avoiding duplication and \
    optimizing memory usage.

    :param rasters: A list of geoutils Raster objects to be merged.
    :param reference: The reference index, in case the reference is to be merged, or a separate Raster object \
 in case the reference should not be merged. Defaults to the first raster in the list.
    :param merge_algorithm: The algorithm, or list of algorithms, to merge the rasters with. Defaults to the mean.\
If several algorithms are provided, each result is returned as a separate band.
    :param resampling_method: The resampling method for the raster reprojections.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.

    :returns: The merged raster with the same parameters (excl. bounds) as the reference.
    """
    # Make sure merge_algorithm is a list
    if not isinstance(merge_algorithm, (list, tuple)):
        merge_algorithm = [merge_algorithm,]

    # Try to run the merge_algorithm with an arbitrary list. Raise an error if the algorithm is incompatible.
    for algo in merge_algorithm:
        try:
            algo([1, 2])
        except TypeError as exception:
            raise TypeError(f"merge_algorithm must be able to take a list as its first argument.\n\n{exception}")

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Reproject and stack all rasters
    raster_stack = stack_rasters(rasters, reference=reference, resampling_method=resampling_method,
                                   use_ref_bounds=use_ref_bounds)

    # Try to use the keyword axis=0 for the merging algorithm (if it's a numpy ufunc).
    merged_data = []
    for algo in merge_algorithm:
        try:
            merged_data.append(algo(raster_stack.data, axis=0))
        # If that doesn't work, use the slower np.apply_along_axis approach.
        except TypeError as exception:
            if "'axis' is an invalid keyword" not in str(exception):
                raise exception
            merged_data.append(np.apply_along_axis(algo, axis=0, arr=raster_stack.data))

    # Save as gu.Raster
    merged_raster = gu.georaster.Raster.from_array(
        data=np.reshape(merged_data, (len(merged_data),) + merged_data[0].shape),
        transform=rio.transform.from_bounds(
            *raster_stack.bounds, width=merged_data[0].shape[1], height=merged_data[0].shape[0]
        ),
        crs=reference_raster.crs,
        nodata=reference_raster.nodata
    )

    return merged_raster


def _get_closest_rectangle(size: int) -> tuple[int, int]:
    """
    Given a 1D array size, return a rectangular shape that is closest to a cube which the size fits in.

    If 'size' does not have an integer root, a rectangle is returned that is slightly larger than 'size'.
    
    :examples:
        >>> _get_closest_rectangle(4)  # size will be 4
        (2, 2)
        >>> _get_closest_rectangle(9)  # size will be 9
        (3, 3)
        >>> _get_closest_rectangle(3)  # size will be 4; needs padding afterward.
        (2, 2)
        >>> _get_closest_rectangle(55) # size will be 56; needs padding afterward.
        (7, 8)
        >>> _get_closest_rectangle(24)  # size will be 25; needs padding afterward
        (5, 5)
        >>> _get_closest_rectangle(85620)  # size will be 85849; needs padding afterward
        (293, 293)
        >>> _get_closest_rectangle(52011)  # size will be 52212; needs padding afterward
        (228, 229)
    """
    close_cube = int(np.sqrt(size))

    # If size has an integer root, return the respective cube.
    if close_cube ** 2 == size:
        return (close_cube, close_cube)
    
    # One of these rectangles/cubes will cover all cells, so return the first that does.
    potential_rectangles = [
        (close_cube, close_cube + 1),
        (close_cube + 1, close_cube + 1)
    ]

    for rectangle in potential_rectangles:
        if np.prod(rectangle) >= size:
            return rectangle

    raise NotImplementedError(f"Function criteria not met for rectangle of size: {size}")



def subdivide_array(shape: tuple[int, ...], count: int) -> np.ndarray:
    """
    Create indices for subdivison of an array in a number of blocks.

    If 'count' is divisible by the product of 'shape', the amount of cells in each block will be equal.
    If 'count' is not divisible, the amount of cells in each block will be very close to equal.

    :param shape: The shape of a array to be subdivided.
    :param count: The amount of subdivisions to make.

    :examples:
        >>> subdivide_array((4, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((6, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((5, 4), 3)
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2]])

    :raises ValueError: If the 'shape' size (`np.prod(shape)`) is smallern than 'count'
                        If the shape is not a 2D shape.

    :returns: An array of shape 'shape' with 'count' unique indices.
    """
    if count > np.prod(shape):
        raise ValueError(f"Shape '{shape}' size ({np.prod(shape)}) is smaller than 'count' ({count}).")

    if len(shape) != 2:
        raise ValueError(f"Expected a 2D shape, got {len(shape)}D shape: {shape}")

    # Generate a small grid of indices, with the same unique count as 'count'
    rect = _get_closest_rectangle(count)
    small_indices = np.pad(np.arange(count), np.prod(rect) - count, mode="edge")[:np.prod(rect)].reshape(rect)

    # Upscale the grid to fit the output shape using nearest neighbour scaling.
    indices = skimage.transform.resize(small_indices, shape, order=0, preserve_range=True).astype(int)

    return indices.reshape(shape)

def get_xy_rotated(raster: gu.georaster.Raster, along_track_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate x, y axes of image to get along- and cross-track distances.
    :param raster: raster to get x,y positions from.
    :param along_track_angle: angle by which to rotate axes (degrees)
    :returns xxr, yyr: arrays corresponding to along (x) and cross (y) track distances.
    """

    myang = np.deg2rad(along_track_angle)

    # get grid coordinates
    xx, yy = raster.coords(grid=True)
    xx -= np.min(xx)
    yy -= np.min(yy)

    # get rotated coordinates

    # for along-track
    xxr = np.multiply(xx, np.cos(myang)) + np.multiply(-1 * yy, np.sin(along_track_angle))
    # for cross-track
    yyr = np.multiply(xx, np.sin(myang)) + np.multiply(yy, np.cos(along_track_angle))

    # re-initialize coordinate at zero
    xxr -= np.nanmin(xxr)
    yyr -= np.nanmin(yyr)

    return xxr, yyr

def rmse(z: np.ndarray) -> float:
    """
    Return root mean square error
    :param z: Residuals between predicted and true value
    :return: Root Mean Square Error
    """
    return np.sqrt(np.nanmean(np.square(np.asarray(z))))

def huber_loss(z: np.ndarray) -> float:
    """
    Huber loss cost (reduces the weight of outliers)
    :param z: Residuals between predicted and true values
    :return: Huber cost
    """
    out = np.asarray(np.square(z) * 1.000)
    out[np.where(z > 1)] = 2 * np.sqrt(z[np.where(z > 1)]) - 1
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
    z = y -_fitfun_sumofsin(x, p)
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


def robust_polynomial_fit(x: np.ndarray, y: np.ndarray, max_order: int = 6, estimator: str  = 'Theil-Sen',
                          cost_func: Callable = median_absolute_error, margin_improvement : float = 20.,
                          subsample: Union[float,int] = 25000, linear_pkg = 'sklearn', verbose: bool = False,
                          random_state = None, **kwargs) -> tuple[np.ndarray,int]:
    """
    Given 1D data x, y, compute a robust polynomial fit to the data. Order is chosen automatically by comparing
    residuals for multiple fit orders of a given estimator.
    :param x: input x data (N,)
    :param y: input y data (N,)
    :param max_order: maximum polynomial order tried for the fit
    :param estimator: robust estimator to use, one of 'Linear', 'Theil-Sen', 'RANSAC' or 'Huber'
    :param cost_func: cost function taking as input two vectors y (true y), y' (predicted y) of same length
    :param margin_improvement: improvement margin (percentage) below which the lesser degree polynomial is kept
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param linear_pkg: package to use for Linear estimator, one of 'scipy' and 'sklearn'
    :param random_state: random seed for testing purposes
    :param verbose: if text should be printed

    :returns coefs, degree: polynomial coefficients and degree for the best-fit polynomial
    """
    if not isinstance(estimator, str) or estimator not in ['Linear','Theil-Sen','RANSAC','Huber']:
        raise ValueError('Attribute estimator must be one of "Linear", "Theil-Sen", "RANSAC" or "Huber".')
    if not isinstance(linear_pkg, str) or linear_pkg not in ['sklearn','scipy']:
        raise ValueError('Attribute linear_pkg must be one of "scipy" or "sklearn".')

    # select sklearn estimator
    dict_estimators = {'Linear': LinearRegression(), 'Theil-Sen':TheilSenRegressor(random_state=random_state)
        , 'RANSAC': RANSACRegressor(random_state=random_state), 'Huber': HuberRegressor()}
    est = dict_estimators[estimator]

    # remove NaNs
    valid_data = np.logical_and(np.isfinite(y), np.isfinite(x))
    x = x[valid_data]
    y = y[valid_data]

    # subsample
    subsamp = subsample_raster(x, subsample=subsample, return_indices=True)
    x = x[subsamp]
    y = y[subsamp]

    # initialize cost function and output coefficients
    costs = np.empty(max_order)
    coeffs = np.zeros((max_order, max_order + 1))
    # loop on polynomial degrees
    for deg in np.arange(1, max_order + 1):
        # if method is linear, and package is scipy
        if estimator == 'Linear' and linear_pkg == 'scipy':
            # define the residual function to optimize
            def fitfun_polynomial(xx, params):
                return sum([p * (xx ** i) for i, p in enumerate(params)])
            def errfun(p, xx, yy):
                return fitfun_polynomial(xx, p) - yy
            p0 = np.polyfit(x, y, deg)
            myresults = scipy.optimize.least_squares(errfun, p0, args=(x, y), **kwargs)
            if verbose:
                print("Initial Parameters: ", p0)
                print("Polynomial degree - ", deg, " --> Status: ", myresults.success, " - ", myresults.status)
                print(myresults.message)
                print("Lowest cost:", myresults.cost)
                print("Parameters:", myresults.x)
            costs[deg - 1] = myresults.cost
            coeffs[deg - 1, 0:myresults.x.size] = myresults.x
        # otherwise, it's from sklearn
        else:
            if not _has_sklearn:
                raise ValueError("Optional dependency needed. Install 'scikit-learn'")

            # create polynomial + linear estimator pipeline
            p = PolynomialFeatures(degree=deg)
            model = make_pipeline(p, est)

            # TODO: find out how to re-scale polynomial coefficient + doc on what is the best scaling for polynomials
            # # scale output data (important for ML algorithms):
            # robust_scaler = RobustScaler().fit(x.reshape(-1,1))
            # x_scaled = robust_scaler.transform(x.reshape(-1,1))
            # # fit scaled data
            # model.fit(x_scaled, y)
            # y_pred = model.predict(x_scaled)

            # fit scaled data
            model.fit(x.reshape(-1,1), y)
            y_pred = model.predict(x.reshape(-1,1))

            # calculate cost
            cost = cost_func(y_pred, y)
            costs[deg - 1] = cost
            # get polynomial estimated with the estimator
            if estimator in ['Linear','Theil-Sen','Huber']:
                c = est.coef_
            # for some reason RANSAC doesn't store coef at the same place
            elif estimator == 'RANSAC':
                c = est.estimator_.coef_
            coeffs[deg - 1, 0:deg+1] = c

    # selecting the minimum (not robust)
    # final_index = mycost.argmin()
    # choosing the best polynomial with a margin of improvement on the cost
    final_index = _choice_best_order(cost=costs, margin_improvement=margin_improvement, verbose=verbose)

    # the degree of the polynom correspond to the index plus one
    return np.trim_zeros(coeffs[final_index], 'b'), final_index + 1


def _fitfun_sumofsin(x: np.array, params: np.ndarray) -> np.ndarray:
    """
    Function for a sum of N frequency sinusoids
    :param x: array of coordinates (N,)
    :param p: list of tuples with amplitude, frequency and phase parameters
    """
    aix = np.arange(0, params.size, 3)
    bix = np.arange(1, params.size, 3)
    cix = np.arange(2, params.size, 3)

    val = np.sum(params[aix] * np.sin(np.divide(2 * np.pi, params[bix]) * x[:, np.newaxis] + params[cix]), axis=1)

    return val

def robust_sumsin_fit(x: np.ndarray, y: np.ndarray, nb_frequency_max: int = 3,
                      bounds_amp_freq_phase: Optional[list[tuple[float,float], tuple[float,float], tuple[float,float]]] = None,
                      significant_res : Optional[float] = None, cost_func: Callable = soft_loss, subsample: Union[float,int] = 25000,
                      random_state: Optional[Union[int,np.random.Generator,np.random.RandomState]] = None, verbose: bool = False) -> tuple[np.ndarray,int]:
    """
    Given 1D data x, y, compute a robust sum of sinusoid fit to the data. The number of frequency is chosen
    automatically by comparing residuals for multiple fit orders of a given estimator.
    :param x: input x data (N,)
    :param y: input y data (N,)
    :param nb_frequency_max: maximum number of phases
    :param bounds_amp_freq_phase: bounds for amplitude, frequency and phase (L, 3, 2) and
    with mean value used for initialization
    :param significant_res: significant resolution of X data to optimize algorithm search
    :param cost_func: cost function taking as input two vectors y (true y), y' (predicted y) of same length
    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param random_state: random seed for testing purposes
    :param verbose: if text should be printed

    :returns coefs, degree: polynomial coefficients and degree for the best-fit polynomial
    """

    def wrapper_costfun_sumofsin(p,x,y):
        return _costfun_sumofsin(p,x,y,cost_func=cost_func)

    # remove NaNs
    valid_data = np.logical_and(np.isfinite(y), np.isfinite(x))
    x = x[valid_data]
    y = y[valid_data]

    # if no significant resolution is provided, assume that it is the mean difference between sampled X values
    if significant_res is None:
        x_sorted = np.sort(x)
        significant_res = np.mean(np.diff(x_sorted))

    # binned statistics for first guess
    nb_bin = int((x.max() - x.min())/(5*significant_res))
    df = nd_binning(y, [x], ['var'], list_var_bins=nb_bin, statistics=[np.nanmedian])
    # first guess for x and y
    x_fg = pd.IntervalIndex(df['var']).mid.values
    y_fg = df['nanmedian']
    valid_fg = np.logical_and(np.isfinite(x_fg),np.isfinite(y_fg))
    x_fg = x_fg[valid_fg]
    y_fg = y_fg[valid_fg]

    # loop on all frequencies
    costs = np.empty(nb_frequency_max)
    amp_freq_phase = np.zeros((nb_frequency_max, 3*nb_frequency_max))*np.nan

    for nb_freq in np.arange(1,nb_frequency_max+1):

        b = bounds_amp_freq_phase
        # if bounds are not provided, define as the largest possible bounds
        if b is None:
            lb_amp = 0
            ub_amp = (y_fg.max() - y_fg.min()) / 2
            # for the phase
            lb_phase = 0
            ub_phase = 2 * np.pi
            # for the frequency, we need at least 5 points to see any kind of periodic signal
            lb_frequency = 1 / (5 * (x.max() - x.min()))
            ub_frequency = 1 / (5 * significant_res)

            b = []
            for i in range(nb_freq):
                b += [(lb_amp,ub_amp),(lb_frequency,ub_frequency),(lb_phase,ub_phase)]

        # format lower bounds for scipy
        lb = np.asarray(([b[i][0] for i in range(3*nb_freq)]))
        # format upper bounds
        ub = np.asarray(([b[i][1] for i in range(3*nb_freq)]))
        # final bounds
        scipy_bounds = scipy.optimize.Bounds(lb, ub)
        # first guess for the mean parameters
        p0 = np.divide(lb + ub, 2)

        # initialize with a first guess
        init_args = dict(args=(x_fg, y_fg), method="L-BFGS-B",
                         bounds=scipy_bounds, options={"ftol": 1E-6})
        init_results = scipy.optimize.basinhopping(wrapper_costfun_sumofsin, p0, disp=verbose,
                                             T=significant_res,  minimizer_kwargs=init_args, seed = random_state)
        init_results = init_results.lowest_optimization_result

        # subsample
        subsamp = subsample_raster(x, subsample=subsample, return_indices=True)
        x = x[subsamp]
        y = y[subsamp]

        # minimize the globalization with a larger number of points
        minimizer_kwargs = dict(args=(x, y),
                                method="L-BFGS-B",
                                bounds=scipy_bounds,
                                options={"ftol": 1E-6})
        myresults = scipy.optimize.basinhopping(wrapper_costfun_sumofsin, init_results.x, disp=verbose,
                                          T=5*significant_res, niter_success=40,
                                          minimizer_kwargs=minimizer_kwargs, seed=random_state)
        myresults = myresults.lowest_optimization_result
        # write results for this number of frequency
        costs[nb_freq-1] = wrapper_costfun_sumofsin(myresults.x,x,y)
        amp_freq_phase[nb_freq -1, 0:3*nb_freq] = myresults.x

    # final_index = costs.argmin()
    final_index = _choice_best_order(cost=costs)

    final_coefs =  amp_freq_phase[final_index][~np.isnan(amp_freq_phase[final_index])]

    # check if an amplitude coefficient is almost 0: remove the coefs of that frequency and lower the degree
    final_degree = final_index + 1
    for i in range(final_index+1):
        if final_coefs[3*i] < (y_fg.max() - y_fg.min())/1000:
            final_coefs = np.delete(final_coefs,slice(3*i,3*i+2))
            final_degree -= 1

    # the number of frequency corresponds to the index plus one
    return final_coefs, final_degree


def subsample_raster(
    array: Union[np.ndarray, np.ma.masked_array], subsample: Union[float, int], return_indices: bool = False,
    random_state : None | np.random.RandomState | np.random.Generator | int = None) -> np.ndarray:
    """
    Randomly subsample a 1D or 2D array by a subsampling factor, taking only non NaN/masked values.

    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations (for testing)

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array)
    """
    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Get number of points to extract
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * np.size(array))
    elif subsample > 1:
        npoints = int(subsample)
    else:
        raise ValueError("`subsample` must be > 0")

    # Remove invalid values and flatten array
    mask = get_mask(array)  # -> need to remove .squeeze in get_mask
    valids = np.argwhere(~mask.flatten()).squeeze()

    # Checks that array and npoints are correct
    assert np.ndim(valids) == 1, "Something is wrong with array dimension, check input data and shape"
    if npoints > np.size(valids):
        npoints = np.size(valids)

    # Randomly extract npoints without replacement
    indices = rnd.choice(valids, npoints, replace=False)
    unraveled_indices = np.unravel_index(indices, array.shape)

    if return_indices:
        return unraveled_indices

    else:
        return array[unraveled_indices]
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

"""Filters to remove outliers and reduce noise in DEMs."""
from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import scipy

from xdem._typing import Any, NDArrayf


def gaussian_filter_scipy(array: NDArrayf, sigma: float) -> NDArrayf:
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using scipy's implementation.

    N.B: kernel_size is set automatically based on sigma.

    :param array: the input array to be filtered.
    :param sigma: the sigma of the Gaussian kernel

    :returns: the filtered array (same shape as input)
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    # In case array does not contain NaNs, use scipy's gaussian filter directly
    if np.count_nonzero(np.isnan(array)) == 0:
        return scipy.ndimage.gaussian_filter(array, sigma=sigma)

    # If array contain NaNs, need a more sophisticated approach
    # Inspired by https://stackoverflow.com/a/36307291
    else:

        # Run filter on a copy with NaNs set to 0
        array_no_nan = array.copy()
        array_no_nan[np.isnan(array)] = 0
        gauss_no_nan = scipy.ndimage.gaussian_filter(array_no_nan, sigma=sigma)
        del array_no_nan

        # Mask of NaN values
        nan_mask = 0 * array.copy() + 1
        nan_mask[np.isnan(array)] = 0
        gauss_mask = scipy.ndimage.gaussian_filter(nan_mask, sigma=sigma)
        del nan_mask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            gauss = gauss_no_nan / gauss_mask

        return gauss


def median_filter_scipy(array: NDArrayf, **kwargs: dict[Any, Any]) -> NDArrayf:
    """
    Apply a median filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, np.inf, array)
    array_nans_replaced_f = scipy.ndimage.median_filter(array_nans_replaced, **kwargs)
    # In the end we want the filtered array without infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def mean_filter(array: NDArrayf, kernel_size: int) -> NDArrayf:
    """
    Apply a mean filter to a raster that may contain NaNs.

    :param array: the input array to be filtered.
    :param kernel_size: the size of the kernel.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2
    if np.ndim(array) not in [2]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D array.")

    kernel = np.ones((kernel_size, kernel_size)) / kernel_size**2
    nans = np.isnan(array)
    # We replace temporarily NaNs by zeros during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, 0, array)
    array_nans_replaced_f = scipy.ndimage.convolve(array_nans_replaced, kernel)
    # In the end we want the filtered array without the introduced zeros, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def min_filter_scipy(array: NDArrayf, **kwargs: dict[Any, Any]) -> NDArrayf:
    """
    Apply a minimum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, np.inf, array)
    array_nans_replaced_f = scipy.ndimage.minimum_filter(array_nans_replaced, **kwargs)
    # In the end we want the filtered array without infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def max_filter_scipy(array: NDArrayf, **kwargs: dict[Any, Any]) -> NDArrayf:
    """
    Apply a maximum filter to a raster that may contain NaNs, using scipy's implementation.

    :param array: the input array to be filtered.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    nans = np.isnan(array)
    # We replace temporarily NaNs by negative infinite values during filtering to avoid spreading NaNs
    array_nans_replaced = np.where(nans, -np.inf, array)
    array_nans_replaced_f = scipy.ndimage.maximum_filter(array_nans_replaced, **kwargs)
    # In the end we want the filtered array without negative infinite values, so we put back NaNs
    return np.where(nans, array, array_nans_replaced_f)


def distance_filter(array: NDArrayf, radius: float, outlier_threshold: float) -> NDArrayf:
    """
    Filter out pixels whose value is distant more than a set threshold from the average value of all neighbor \
pixels within a given radius.
    Filtered pixels are set to NaN.

    TO DO: Add an option on how the "average" value should be calculated, i.e. using a Gaussian, median etc filter.

    :param array: the input array to be filtered.
    :param radius: the radius in which the average value is calculated (for Gaussian filter, this is sigma).
    :param outlier_threshold: the minimum difference abs(array - mean) for a pixel to be considered an outlier.

    :returns: the filtered array (same shape as input)
    """
    # Calculate the average value within the radius
    smooth = gaussian_filter_scipy(array, sigma=radius)

    # Filter outliers
    outliers = (np.abs(array - smooth)) > outlier_threshold
    out_array = np.copy(array)
    out_array[outliers] = np.nan

    return out_array


def generic_filter(array: NDArrayf, filter_function: Callable[..., NDArrayf], **kwargs: dict[Any, Any]) -> NDArrayf:
    """
    Apply a filter from a function.

    :param array: the input array to be filtered.
    :param filter_function: the function of the filter.

    :returns: the filtered array (same shape as input).
    """
    # Check that array dimension is 2 or 3
    if np.ndim(array) not in [2, 3]:
        raise ValueError(f"Invalid array shape given: {array.shape}. Expected 2D or 3D array.")

    return scipy.ndimage.generic_filter(array, filter_function, **kwargs)

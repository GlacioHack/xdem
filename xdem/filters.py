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

import numpy as np
import scipy

from xdem._typing import NDArrayf


def gaussian_filter_scipy(array: NDArrayf, sigma: float) -> NDArrayf:
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using scipy's implementation.
    gaussian_filter_cv is recommended as it is usually faster, but this depends on the value of sigma.

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

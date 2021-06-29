"""Filters to remove outliers and reduce noise in DEMs."""
from __future__ import annotations

import cv2 as cv
import numpy as np
import scipy
import warnings


# Gaussian filters

def gaussian_filter_scipy(array: np.ndarray, sigma: float) -> np.ndarray:
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
        raise ValueError(
            f"Invalid array shape given: {array.shape}. Expected 2D or 3D array"
        )

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


def gaussian_filter_cv(array: np.ndarray, sigma) -> np.ndarray:
    """
    Apply a Gaussian filter to a raster that may contain NaNs, using OpenCV's implementation.
    Arguments are for now hard-coded to be identical to scipy.

    N.B: kernel_size is set automatically based on sigma

    :param array: the input array to be filtered.
    :param sigma: the sigma of the Gaussian kernel

    :returns: the filtered array (same shape as input)
    """
    # Check that array dimension is 2, or can be squeezed to 2D
    orig_shape = array.shape
    if len(orig_shape) == 2:
        pass
    elif len(orig_shape) == 3:
        if orig_shape[0] == 1:
            array = array.squeeze()
        else:
            raise NotImplementedError("Case of array of dimension 3 not implemented")
    else:
        raise ValueError(
            f"Invalid array shape given: {orig_shape}. Expected 2D or 3D array"
        )

    # In case array does not contain NaNs, use OpenCV's gaussian filter directly
    # With kernel size (0, 0), i.e. set to default, and borderType=BORDER_REFLECT, the output is equivalent to scipy
    if np.count_nonzero(np.isnan(array)) == 0:
        gauss = cv.GaussianBlur(array, (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)

    # If array contain NaNs, need a more sophisticated approach
    # Inspired by https://stackoverflow.com/a/36307291
    else:

        # Run filter on a copy with NaNs set to 0
        array_no_nan = array.copy()
        array_no_nan[np.isnan(array)] = 0
        gauss_no_nan = cv.GaussianBlur(array_no_nan, (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)
        del array_no_nan

        # Mask of NaN values
        nan_mask = 0 * array.copy() + 1
        nan_mask[np.isnan(array)] = 0
        gauss_mask = cv.GaussianBlur(nan_mask, (0, 0), sigmaX=sigma, borderType=cv.BORDER_REFLECT)
        del nan_mask

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            gauss = gauss_no_nan / gauss_mask

    return gauss.reshape(orig_shape)


# Median filters

# To be added

# Min/max filters

# To be added


def distance_filter(array: np.ndarray, radius: float, outlier_threshold: float) -> np.ndarray:
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
    smooth = gaussian_filter_cv(array, sigma=radius)

    # Filter outliers
    outliers = (np.abs(array - smooth)) > outlier_threshold
    out_array = np.copy(array)
    out_array[outliers] = np.nan

    return out_array

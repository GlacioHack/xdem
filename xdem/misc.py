"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

import geopandas as gpd
import numpy as np

from geoutils.georaster import RasterType, Raster
from geoutils.geovector import VectorType, Vector
from geoutils.spatial_tools import get_array_and_mask

import xdem.version

def _preprocess_values_with_mask_to_array(values: np.ndarray | RasterType | list[np.ndarray | RasterType],
                                          include_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                          exclude_mask: np.ndarray | VectorType | gpd.GeoDataFrame = None,
                                          gsd: float = None, preserve_shape: bool = True) -> tuple[np.ndarray, float]:
    """
    Preprocess input values provided as Raster or ndarray with a stable and/or unstable mask provided as Vector or
    ndarray into an array of stable values.

    By default, the shape is preserved and the masked values converted to NaNs.

    :param values: Values or list of values as a Raster, array or a list of Raster/arrays
    :param include_mask: Vector shapefile of mask to include (if values is Raster), or boolean array of same shape as
        values
    :param exclude_mask: Vector shapefile of mask to exclude (if values is Raster), or boolean array of same shape
        as values
    :param gsd: Ground sampling distance, if all the input values are provided as array
    :param preserve_shape: If True, masks unstable values with NaN. If False, returns a 1D array of stable values.

    :return: Array of stable terrain values, Ground sampling distance
    """

    # Check inputs
    if not isinstance(values, (Raster, np.ndarray)):
        raise ValueError('The values must be a Raster or NumPy array.')
    if include_mask is not None and not isinstance(include_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The stable mask must be a Vector, GeoDataFrame or NumPy array.')
    if exclude_mask is not None and not isinstance(exclude_mask, (np.ndarray, Vector, gpd.GeoDataFrame)):
        raise ValueError('The unstable mask must be a Vector, GeoDataFrame or NumPy array.')

    # Check that input stable mask can only be a georeferenced vector if the proxy values are a Raster to project onto
    any_raster = any([isinstance(val, Raster) for val in values])
    if not any_raster and isinstance(include_mask, (Vector, gpd.GeoDataFrame)):
        raise ValueError(
            'The stable mask can only passed as a Vector or GeoDataFrame if the input values contain a Raster.')

    # If there is only one array or Raster, put alone in a list
    if not isinstance(values, list):
        values = [values]

    # Get the arrays
    values_arr = [get_array_and_mask(val)[0] if isinstance(val, Raster) else val for val in values]

    # Get the ground sampling distance from the first Raster if there is one
    if any_raster:
        indexes_raster = [i for i, x in enumerate(values) if x]
        first_raster = values[indexes_raster[0]]
        gsd = first_raster.res[0]
    else:
        gsd = gsd

    # If the stable mask is not an array, create it
    if include_mask is None:
        include_mask_arr = np.ones(np.shape(values_arr[0]), dtype=bool)
    elif isinstance(include_mask, (Vector, gpd.GeoDataFrame)):

        # If the stable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(include_mask, gpd.GeoDataFrame):
            stable_vector = Vector(include_mask)
        else:
            stable_vector = include_mask

        # Create the mask
        include_mask_arr = stable_vector.create_mask(first_raster)
    # If the mask is already an array, just pass it
    else:
        include_mask_arr = include_mask

    # If the unstable mask is not an array, create it
    if exclude_mask is None:
        exclude_mask_arr = np.zeros(np.shape(values_arr[0]), dtype=bool)
    elif isinstance(exclude_mask, (Vector, gpd.GeoDataFrame)):

        # If the unstable mask is a geopandas dataframe, wrap it in a Vector object
        if isinstance(exclude_mask, gpd.GeoDataFrame):
            unstable_vector = Vector(exclude_mask)
        else:
            unstable_vector = exclude_mask

        # Create the mask
        exclude_mask_arr = unstable_vector.create_mask(first_raster)
    # If the mask is already an array, just pass it
    else:
        exclude_mask_arr = exclude_mask

    include_mask_arr = np.logical_and(include_mask_arr, ~exclude_mask_arr).squeeze()

    if preserve_shape:
        # Need to preserve the shape, so setting as NaNs all points not on stable terrain
        values_stable_arr = []
        for val in values_arr:
            val_stable = val.copy()
            val_stable[include_mask_arr] = np.nan
            values_stable_arr.append(val_stable)
    else:
        values_stable_arr = [val_arr[include_mask_arr] for val_arr in values_arr]

    # If input was a list, give a list. If it was a single array, give a single array.
    if not isinstance(values, list):
        values_stable_arr  = values_stable_arr[0]

    return values_stable_arr, gsd


def generate_random_field(shape: tuple[int, int], corr_size: int) -> np.ndarray:
    """
    Generate a semi-random gaussian field (to simulate a DEM or DEM error)

    :param shape: The output shape of the field.
    :param corr_size: The correlation size of the field.

    :examples:
        >>> np.random.seed(1)
        >>> generate_random_field((4, 5), corr_size=2).round(2)
        array([[0.47, 0.5 , 0.56, 0.63, 0.65],
               [0.49, 0.51, 0.56, 0.62, 0.64],
               [0.56, 0.56, 0.57, 0.59, 0.59],
               [0.57, 0.57, 0.57, 0.58, 0.58]])

    :returns: A numpy array of semi-random values from 0 to 1
    """

    if not _has_cv2:
        raise ValueError("Optional dependency needed. Install 'opencv'")

    field = cv2.resize(
        cv2.GaussianBlur(
            np.repeat(
                np.repeat(
                    np.random.randint(0, 255, (shape[0] // corr_size, shape[1] // corr_size), dtype="uint8"),
                    corr_size,
                    axis=0,
                ),
                corr_size,
                axis=1,
            ),
            ksize=(2 * corr_size + 1, 2 * corr_size + 1),
            sigmaX=corr_size,
        )
        / 255,
        dsize=(shape[1], shape[0]),
    )
    return field


def deprecate(removal_version: str | None = None, details: str | None = None):
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed. 
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """
    def deprecator_func(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or removal_version > xdem.version.version

            # Add text depending on the given arguments and 'should_warn'.
            text = (
                f"Call to deprecated function '{func.__name__}'."
                if should_warn
                else f"Deprecated function '{func.__name__}' was removed in {removal_version}."
            )

            # Add the details explanation if it was given, and make sure the sentence is ended.
            if details is not None:
                details_frm = details.strip()
                if details_frm[0].islower():
                    details_frm = details_frm[0].upper() + details_frm[1:]

                text += " " + details_frm

                if not any(text.endswith(c) for c in ".!?"):
                    text += "."

            if should_warn and removal_version is not None:
                text += f" This functionality will be removed in version {removal_version}."
            elif not should_warn:
                text += f" Current version: {xdem.version.version}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)
            
            return func(*args, **kwargs)

        return new_func

    return deprecator_func

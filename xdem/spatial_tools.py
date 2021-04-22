"""

"""
from __future__ import annotations

from typing import Callable, Union

import geoutils as gu
import numpy as np
import rasterio as rio
import rasterio.warp


def get_mask(array: Union[np.ndarray, np.ma.masked_array]) -> np.ndarray:
    """
    Return the mask of invalid values, whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.

    :returns invalid_mask: boolean array, True where array is masked or Nan.
    """
    return (array.mask | ~np.isfinite(array.data)) if isinstance(array, np.ma.masked_array) else ~np.isfinite(array)


def get_array_and_mask(array: Union[np.ndarray, np.ma.masked_array]) -> (np.ndarray, np.ndarray):
    """
    Return array with masked values set to NaN and the associated mask.
    Works whether array is a ndarray with NaNs or a np.ma.masked_array.
    WARNING, if array is of dtype float, will return a view only, if integer dtype, will return a copy.

    :param array: Input array.

    :returns array_data, invalid_mask: a tuple of ndarrays. First is array with invalid pixels converted to NaN, \
    second is mask of invalid pixels (True if invalid).
    """
    # Get mask of invalid pixels
    invalid_mask = get_mask(array)

    # If array is of type integer, need to be converted to float, forcing not duplicate
    if np.issubdtype(array.dtype, np.integer):
        array = array.astype('float32')

    # Convert into a regular ndarray and convert invalid values to NaN
    array_data = np.asarray(array)
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
    m = np.nanmedian(data)
    return nfact * np.nanmedian(np.abs(data - m))


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
            f"'{resampling_method}' is not a valid rasterio.warp.Resampling method. "
            f"Valid methods: {[str(method).replace('Resampling.', '') for method in rio.warp.Resampling]}"
        )
    return resampling_method


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

    for key in max_bounds:
        modulo = max_bounds[key] % resolution
        max_bounds[key] -= modulo

        if key in ["right", "top"] and modulo > 0:
            max_bounds[key] += resolution

    return rio.coords.BoundingBox(**max_bounds)


def merge_rasters(rasters: list[gu.georaster.Raster], reference: int = 0, merge_algorithm: Callable = np.nanmean,
                  resampling_method: Union[str, rio.warp.Resampling] = "nearest") -> gu.georaster.Raster:
    """
    Merge a list of rasters into one larger raster.

    Reprojects the rasters to the reference raster CRS and resolution.

    :param rasters: A list of geoutils Raster objects.
    :param reference: The reference index (defaults to the first raster in the list).
    :param merge_algorithm: The algorithm to merge the rasters with. Defaults to the mean.
    :param resampling_method: The resampling method for the raster reprojections.

    :returns: The merged raster with the same parameters (excl. bounds) as the reference.
    """
    # Try to run the merge_algorithm with an arbitrary list. Raise an error if the algorithm is incompatible.
    try:
        merge_algorithm([1, 2])
    except TypeError as exception:
        raise TypeError(f"merge_algorithm must be able to take a list as its first argument.\n\n{exception}")
    if isinstance(resampling_method, str):
        resampling_method = resampling_method_from_str(resampling_method)

    reference_raster = rasters[reference]
    # Find the maximum covering bounding box
    max_bounds = merge_bounding_boxes([raster.bounds for raster in rasters], resolution=reference_raster.res[0])

    # Make a data list and add all of the reprojected rasters into it.
    data: list[np.ndarray] = []
    for raster in rasters:
        reprojected_raster = raster.reproject(
            dst_bounds=max_bounds,
            dst_crs=reference_raster.crs,
            dtype=reference_raster.data.dtype,
            nodata=reference_raster.nodata
        )
        data.append(reprojected_raster.data.squeeze())

    # Try to use the keyword axis=0 for the merging algorithm (if it's a numpy ufunc).
    try:
        merged_data = merge_algorithm(data, axis=0)
    # If that doesn't work, use the slower np.apply_along_axis approach.
    except TypeError as exception:
        if "'axis' is an invalid keyword" not in str(exception):
            raise exception
        merged_data = np.apply_along_axis(merge_algorithm, axis=0, arr=data)

    merged_raster = gu.georaster.Raster.from_array(
        data=merged_data.reshape((1,) + merged_data.shape),
        transform=rio.transform.from_bounds(*max_bounds, width=merged_data.shape[1], height=merged_data.shape[0]),
        crs=reference_raster.crs,
        nodata=reference_raster.nodata
    )

    return merged_raster


def hillshade(dem: Union[np.ndarray, np.ma.masked_array], resolution: Union[float, tuple[float, float]],
              azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0) -> np.ndarray:
    """
    Generate a hillshade from the given DEM.

    :param dem: The input DEM to calculate the hillshade from.
    :param resolution: One or two values specifying the resolution of the DEM.
    :param azimuth: The azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The altitude in degrees (0-90°). 90° is straight from above.
    :param z_factor: Vertical exaggeration factor.

    :raises AssertionError: If the given DEM is not a 2D array.
    :raises ValueError: If invalid argument types or ranges were given.

    :returns: A hillshade with the dtype "float32" with value ranges of 0-255.
    """
    # Extract the DEM and mask
    dem_values, mask = get_array_and_mask(dem.squeeze())
    # The above is not guaranteed to copy the data, so this needs to be done first.
    demc = dem_values.copy()

    # Validate the inputs.
    assert len(demc.shape) == 2, f"Expected a 2D array. Got shape: {dem.shape}"
    if (azimuth < 0.0) or (azimuth > 360.0):
        raise ValueError(f"Azimuth must be a value between 0 and 360 degrees (given value: {azimuth})")
    if (altitude < 0.0) or (altitude > 90):
        raise ValueError("Altitude must be a value between 0 and 90 degress (given value: {altitude})")
    if (z_factor < 0.0) or not np.isfinite(z_factor):
        raise ValueError(f"z_factor must be a non-negative finite value (given value: {z_factor})")

    # Fill the nonfinite values with the median (or maybe just 0?) to not interfere with the gradient analysis.
    demc[~np.isfinite(demc)] = np.nanmedian(demc)

    # Multiply the DEM with the z_factor to increase the apparent height.
    demc *= z_factor

    # Parse the resolution argument. If it's subscriptable, it's assumed to be [X, Y] resolution.
    try:
        resolution[0]  # type: ignore
    # If that fails, it's assumed to be the X&Y resolution.
    except TypeError as exception:
        if "not subscriptable" not in str(exception):
            raise exception
        resolution = (resolution,) * 2  # type: ignore

    # Calculate the gradient of each pixel.
    x_gradient, y_gradient = np.gradient(demc)
    # Normalize by the radius of the resolution to make it resolution variant.
    x_gradient /= resolution[0] * 0.5  # type: ignore
    y_gradient /= resolution[1] * 0.5  # type: ignore

    azimuth_rad = np.deg2rad(360 - azimuth)
    altitude_rad = np.deg2rad(altitude)

    # Calculate slope and aspect maps.
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x_gradient ** 2 + y_gradient ** 2))
    aspect = np.arctan2(-x_gradient, y_gradient)

    # Create a hillshade from these products.
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * \
        np.cos(slope) * np.cos((azimuth_rad - np.pi / 2.0) - aspect)

    # Set (potential) masked out values to nan
    shaded[mask] = np.nan

    # Return the hillshade, scaled to uint8 ranges.
    return np.clip(255 * (shaded + 0.5) / 2, 0, 255).astype("float32")

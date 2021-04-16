"""

"""
from __future__ import annotations

from typing import Callable, Union

import geoutils as gu
import numpy as np
import rasterio as rio
import rasterio.warp


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


def merge_rasters(rasters: list[gu.georaster.Raster], reference: int = 0,
                  merge_algorithm: Union[Callable, list(Callable)] = np.nanmean,
                  resampling_method: Union[str, rio.warp.Resampling] = "bilinear", include_ref = True,
                  use_ref_bounds = False) -> gu.georaster.Raster:
    """
    Merge a list of rasters into one larger raster.

    Reprojects the rasters to the reference raster CRS and resolution.
    Note that currently all rasters will be loaded once in memory. However, if rasters data is not loaded prior to \
    merge_rasters it will be loaded for reprojection and deleted, therefore avoiding duplication and \
    optimizing memory usage.

    :param rasters: A list of geoutils Raster objects.
    :param reference: The reference index (defaults to the first raster in the list).
    :param merge_algorithm: The algorithm, or list of algorithms, to merge the rasters with. Defaults to the mean.\
If several algorithms are provided, each result is returned as a separate band.
    :param resampling_method: The resampling method for the raster reprojections.
    :param include_ref: If False, will not merge the reference with the others.
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

    # Check resampling method
    if isinstance(resampling_method, str):
        resampling_method = resampling_method_from_str(resampling_method)

    # Select reference raster
    reference_raster = rasters[reference]

    # Set output bounds
    if use_ref_bounds:
        max_bounds = reference_raster.bounds
    else:
        max_bounds = merge_bounding_boxes([raster.bounds for raster in rasters], resolution=reference_raster.res[0])

    # Make a data list and add all of the reprojected rasters into it.
    data: list[np.ndarray] = []

    rasters_to_merge = rasters if include_ref else rasters[1:]
    for raster in rasters_to_merge:

        # Check that data is loaded, otherwise temporarily load it
        if not raster.is_loaded:
            raster.load()
            raster.is_loaded = False

        # Reproject to reference grid
        reprojected_raster = raster.reproject(
            dst_bounds=max_bounds,
            dst_res=reference_raster.res,
            dst_crs=reference_raster.crs,
            dtype=reference_raster.data.dtype,
            nodata=np.nan  # quick fix, should use get_mask when other PR is merged (this could fail if dtype is int)
        )
        data.append(reprojected_raster.data.squeeze())

        # Remove unloaded rasters
        if not raster.is_loaded:
            raster._data = None

    # Try to use the keyword axis=0 for the merging algorithm (if it's a numpy ufunc).
    merged_data = []
    for algo in merge_algorithm:
        try:
            merged_data.append(algo(data, axis=0))
        # If that doesn't work, use the slower np.apply_along_axis approach.
        except TypeError as exception:
            if "'axis' is an invalid keyword" not in str(exception):
                raise exception
            merged_data.append(np.apply_along_axis(algo, axis=0, arr=data))

    merged_raster = gu.georaster.Raster.from_array(
        data=np.reshape(merged_data, (len(merged_data),) + merged_data[0].shape),
        transform=rio.transform.from_bounds(
            *max_bounds, width=merged_data[0].shape[1], height=merged_data[0].shape[0]
        ),
        crs=reference_raster.crs,
        nodata=reference_raster.nodata
    )

    return merged_raster

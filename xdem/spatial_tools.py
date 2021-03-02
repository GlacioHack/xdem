"""

"""
from typing import Union

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
        # Try to match the string version of the resampling method with a rio Resampling enum name
        for method in rio.warp.Resampling:
            if str(method).replace("Resampling.", "") == resampling_method:
                resampling_method = method
                break
        # If no match was found, raise an error.
        else:
            raise ValueError(
                f"'{resampling_method}' is not a valid rasterio.warp.Resampling method. "
                f"Valid methods: {[str(method).replace('Resampling.', '') for method in rio.warp.Resampling]}"
            )

    # Reproject the non-reference and subtract the two rasters.
    difference = first_raster.data - second_raster.reproject(second_raster, resampling=resampling_method).data if \
        reference == "first" else \
        first_raster.reproject(second_raster, resampling=resampling_method).data - second_raster.data

    # Generate a GeoUtils raster from the difference array
    difference_raster = gu.georaster.Raster.from_array(
        difference,
        transform=first_raster.transform if reference == "first" else second_raster.transform,
        crs=first_raster.crs if reference == "first" else second_raster.crs
    )

    return difference_raster

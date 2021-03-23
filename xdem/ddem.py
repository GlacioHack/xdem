from __future__ import annotations

import warnings
from typing import Any, Optional

import geoutils as gu
import numpy as np
import rasterio.fill

import xdem


class dDEM(xdem.dem.DEM):   # pylint: disable=invalid-name
    """A difference-DEM object."""

    def __init__(self, raster: gu.Raster, start_time: np.datetime64, end_time: np.datetime64,
                 error: Optional[Any] = None):
        """
        Create a dDEM object from a Raster.

        :param raster: A georeferenced Raster object.
        :param start_time: The starting time of the dDEM.
        :param end_time: The end time of the dDEM.
        :param error: An error measure for the dDEM (UNUSED).

        :returns: A new dDEM instance.
        """
        super().__init__(raster)

        #self.__dict__ = raster.__dict__
        self.start_time = start_time
        self.end_time = end_time
        self.error = error
        self._filled_data: Optional[np.ndarray] = None

    def __str__(self) -> str:
        """Return a summary of the dDEM."""
        return f"dDEM from {self.start_time} to {self.end_time}.\n\n{super().__str__()}"

    @property
    def filled_data(self) -> Optional[np.ndarray]:
        """
        Get the filled data array if it exists, or else the original data if it has no nans.

        Returns None if the filled_data array does not exist, and the original data has nans.

        :returns: An array or None
        """
        if self._filled_data is not None:
            return self._filled_data
        if (isinstance(self.data, np.ma.masked_array) and np.any(self.data.mask)) or np.any(np.isnan(self.data)):
            return None

        return self.data

    @filled_data.setter
    def filled_data(self, array: np.ndarray):
        """Set the filled_data attribute and make sure that it is valid."""

        assert self.data.shape == array.shape, f"Array shape '{array.shape}' differs from the data shape '{self.data.shape}'"

        if (isinstance(array, np.ma.masked_array) and np.any(array.mask)) or np.any(np.isnan(array)):
            raise ValueError("Data contains NaNs")

        self._filled_data = array

    @property
    def time(self) -> np.timedelta64:
        """Get the time duration."""
        return self.end_time - self.start_time

    def from_array(data: np.ndarray, transform, crs, start_time, end_time, error=None, nodata=None) -> dDEM:
        """
        Create a new dDEM object from an array.

        :param data: The dDEM data array.
        :param transform: A geometric transform.
        :param crs: The coordinate reference system of the dDEM.
        :param start_time: The starting time of the dDEM.
        :param end_time: The end time of the dDEM.
        :param error: An error measure for the dDEM.
        :param nodata: The nodata value.

        :returns: A new dDEM instance.
        """
        return dDEM(
            gu.georaster.Raster.from_array(
                data=data,
                transform=transform,
                crs=crs,
                nodata=nodata
            ),
            start_time=start_time,
            end_time=end_time,
            error=error,
        )

    def interpolate(self, method: str = "linear"):
        """
        Interpolate the dDEM using the given method.

        :param method: The method to use for interpolation.
        """
        if method == "linear":
            coords = self.coords(offset="center")
            # Create a mask for where nans exist
            nan_mask = self.data.mask | np.isnan(self.data.data) if isinstance(
                self.data, np.ma.masked_array) else np.isnan(self.data)

            interpolated_ddem = rasterio.fill.fillnodata(self.data, mask=~nan_mask.astype("uint8"))

            # Fill the nans (values outside of the value boundaries) with the median value
            # This triggers a warning with np.masked_array's because it ignores the mask
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interpolated_ddem[np.isnan(interpolated_ddem)] = np.nanmedian(self.data)

            self.filled_data = interpolated_ddem.reshape(self.data.shape)

        else:
            raise NotImplementedError

        return self.filled_data

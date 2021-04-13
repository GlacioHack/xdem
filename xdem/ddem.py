from __future__ import annotations

import copy
import warnings
from typing import Any, Optional, Union

import geoutils as gu
import numpy as np
import shapely

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
        # super().__init__(raster)

        self.__dict__ = raster.__dict__
        self.start_time = start_time
        self.end_time = end_time
        self.error = error
        self._filled_data: Optional[np.ndarray] = None
        self._fill_method = ""

    def __str__(self) -> str:
        """Return a summary of the dDEM."""
        return f"dDEM from {self.start_time} to {self.end_time}.\n\n{super().__str__()}"

    def copy(self) -> dDEM:
        """Return a copy of the DEM."""
        new_ddem = dDEM.from_array(self.data.copy(), self.transform, self.crs, self.start_time, self.end_time)
        return new_ddem

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

        return np.asarray(self.data)

    @filled_data.setter
    def filled_data(self, array: np.ndarray):
        """Set the filled_data attribute and make sure that it is valid."""

        assert self.data.shape == array.shape, f"Array shape '{array.shape}' differs from the data shape '{self.data.shape}'"

        self._filled_data = np.asarray(array)

    @property
    def fill_method(self) -> str:
        """Return the fill method used for the filled_data."""
        return self._fill_method

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

    def interpolate(self, method: str = "linear",
                    reference_elevation: Optional[Union[np.ndarray, np.ma.masked_array, xdem.DEM]] = None,
                    mask: Optional[Union[np.ndarray, xdem.DEM, gu.Vector]] = None):
        """
        Interpolate the dDEM using the given method.

        :param method: The method to use for interpolation.
        :param reference_elevation: Reference DEM. Only required for hypsometric approaches.
        """
        if reference_elevation is not None:
            try:
                reference_elevation = reference_elevation.reproject(self, silent=True)  # type: ignore
            except AttributeError as exception:
                if "object has no attribute 'reproject'" not in str(exception):
                    raise exception

            if isinstance(reference_elevation, np.ndarray):
                reference_elevation = np.ma.masked_array(reference_elevation, mask=np.isnan(reference_elevation))

            assert reference_elevation.data.shape == self.data.shape, (
                f"'reference_elevation' shape ({reference_elevation.data.shape})"
                f" different from 'self' ({self.data.shape})"
            )

        if method == "linear":
            self.filled_data = xdem.volume.linear_interpolation(self.data)
        elif method == "local_hypsometric":
            assert reference_elevation is not None
            assert mask is not None

            if not isinstance(mask, gu.Vector):
                mask = gu.Vector(mask)

            nans = np.isnan(np.asarray(self.data)) | (self.data.mask
                                                      if isinstance(self.data, np.ma.masked_array)
                                                      else False)
            interpolated_ddem = np.where(nans, np.nan, np.asarray(self.data))
            entries = mask.ds[mask.ds.intersects(shapely.geometry.box(*self.bounds))]

            ddem_mask = nans.copy()
            for i in entries.index:
                feature_mask = (gu.Vector(entries.loc[entries.index == i]).create_mask(
                    self)).reshape(self.data.shape)
                if np.count_nonzero(feature_mask) == 0:
                    continue
                try:
                    interpolated_ddem = np.asarray(
                        xdem.volume.hypsometric_interpolation(
                            interpolated_ddem,
                            reference_elevation.data,
                            mask=feature_mask
                        )
                    )
                except ValueError as exception:
                    # Skip the feature if too few glacier values exist.
                    if "x and y arrays must have at least 2 entries" in str(exception):
                        continue
                    raise exception
                # Set the validity flag of all values within the feature to be valid
                ddem_mask[feature_mask] = False

                # All values that were nan in the start and are without the updated validity mask should now be nan
                # The above interpolates values outside of the dDEM, so this is necessary.
                interpolated_ddem[ddem_mask] = np.nan

            diff = abs(np.nanmean(interpolated_ddem - self.data))
            assert diff < 0.01, (diff, self.data.mean())

            self.filled_data = xdem.volume.linear_interpolation(interpolated_ddem)

        elif method == "regional_hypsometric":
            assert reference_elevation is not None
            assert mask is not None

            mask_array = xdem.coreg.mask_as_array(self, mask).reshape(self.data.shape)

            self.filled_data = xdem.volume.hypsometric_interpolation(
                self.data,
                reference_elevation.data,
                mask=mask_array
            ).data

        else:
            raise NotImplementedError(f"Interpolation method '{method}' not supported")

        return self.filled_data

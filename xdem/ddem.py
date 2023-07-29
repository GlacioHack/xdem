"""Difference of DEMs classes and functions."""
from __future__ import annotations

import warnings
from typing import Any

import geoutils as gu
import numpy as np
import shapely
from geoutils.raster import Raster, RasterType, get_array_and_mask
from rasterio.crs import CRS
from rasterio.warp import Affine

import xdem
from xdem._typing import MArrayf, NDArrayf


class dDEM(Raster):  # type: ignore
    """A difference-DEM object."""

    def __init__(self, raster: gu.Raster, start_time: np.datetime64, end_time: np.datetime64, error: Any | None = None):
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
        self._filled_data: NDArrayf | None = None
        self._fill_method = ""
        self.nodata: int | float | None = None

    def __str__(self) -> str:
        """Return a summary of the dDEM."""
        return f"dDEM from {self.start_time} to {self.end_time}.\n\n{super().__str__()}"

    def copy(self, new_array: NDArrayf = None) -> dDEM:
        """Return a copy of the DEM."""

        if new_array is None:
            new_array = self.data.copy()

        new_ddem = dDEM.from_array(new_array, self.transform, self.crs, self.start_time, self.end_time)
        return new_ddem

    @property
    def filled_data(self) -> NDArrayf | None:
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
    def filled_data(self, array: NDArrayf) -> None:
        """Set the filled_data attribute and make sure that it is valid."""

        assert (
            self.data.size == array.size
        ), f"Array shape '{array.shape}' differs from the data shape '{self.data.shape}'"

        self._filled_data = np.asarray(array).reshape(self.data.shape)

    @property
    def fill_method(self) -> str:
        """Return the fill method used for the filled_data."""
        return self._fill_method

    @property
    def time(self) -> np.timedelta64:
        """Get the time duration."""
        return self.end_time - self.start_time

    @classmethod
    def from_array(
        cls: type[RasterType],
        data: NDArrayf | MArrayf,
        transform: tuple[float, ...] | Affine,
        crs: CRS | int | None,
        start_time: np.datetime64,
        end_time: np.datetime64,
        nodata: int | float | None = None,
        error: float = None,
    ) -> dDEM:  # type: ignore
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
        return cls(
            gu.Raster.from_array(data=data, transform=transform, crs=crs, nodata=nodata),
            start_time=start_time,
            end_time=end_time,
            error=error,
        )

    def interpolate(
        self,
        method: str = "linear",
        reference_elevation: NDArrayf | np.ma.masked_array[Any, np.dtype[np.floating[Any]]] | xdem.DEM = None,
        mask: NDArrayf | xdem.DEM | gu.Vector = None,
    ) -> NDArrayf | None:
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

            interpolated_ddem, nans = get_array_and_mask(self.data.copy())
            entries = mask.ds[mask.ds.intersects(shapely.geometry.box(*self.bounds))]

            ddem_mask = nans.copy().squeeze()
            for i in entries.index:
                feature_mask = (gu.Vector(entries.loc[entries.index == i]).create_mask(self, as_array=True)).reshape(
                    interpolated_ddem.shape
                )
                if np.count_nonzero(feature_mask) == 0:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "Not enough valid bins")
                        warnings.filterwarnings("ignore", "invalid value encountered in divide")
                        interpolated_ddem = np.asarray(
                            xdem.volume.hypsometric_interpolation(
                                interpolated_ddem, reference_elevation.data, mask=feature_mask
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

            mask_array = xdem.coreg.base._mask_as_array(self, mask).reshape(self.data.shape)

            self.filled_data = xdem.volume.hypsometric_interpolation(
                self.data, reference_elevation.data, mask=mask_array
            ).data

        else:
            raise NotImplementedError(f"Interpolation method '{method}' not supported")

        return self.filled_data

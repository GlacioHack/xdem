from __future__ import annotations

import datetime
import warnings
from typing import Any, Optional, Union

import geoutils as gu
import numpy as np
import pandas as pd
import rasterio.fill
import scipy.interpolate

import xdem


class dDEM(xdem.dem.DEM):   # pylint: disable=invalid-name
    """A difference-DEM object."""

    def __init__(self, raster: gu.georaster.Raster, start_time: np.datetime64, end_time: np.datetime64,
                 error: Optional[Any] = None):
        """
        Create a dDEM object from a Raster.

        :param raster: A georeferenced Raster object.
        :param start_time: The starting time of the dDEM.
        :param end_time: The end time of the dDEM.
        :param error: An error measure for the dDEM (UNUSED).

        :returns: A new dDEM instance.
        """

        self.__dict__ = raster.__dict__
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


class DEMCollection:
    """A temporal collection of DEMs."""

    def __init__(self, dems: Union[list[gu.georaster.Raster], list[xdem.dem.DEM]],
                 timestamps: Optional[list[datetime.datetime]] = None,
                 outlines: Optional[Union[gu.geovector.Vector, dict[datetime.datetime, gu.geovector.Vector]]] = None,
                 reference_dem: Union[int, gu.georaster.Raster] = 0):
        """
        Create a new temporal DEM collection.

        :param dems: A list of DEMs.
        :param timestamps: A list of DEM timestamps.
        :param outlines: Polygons to separate the changing area of interest. Could for example be glacier outlines.
        :param reference_dem: An instance or index of which DEM in the 'dems' list is the reference.

        :returns: A new DEMCollection instance.
        """
        # If timestamps is not given, try to parse it from the (potential) 'datetime' attribute of each DEM.
        if timestamps is None:
            timestamp_attributes = [dem.datetime for dem in dems]
            print(timestamp_attributes)
            if any([stamp is None for stamp in timestamp_attributes]):
                raise ValueError("'timestamps' not provided and the given DEMs do not all have datetime attributes")

            timestamps = timestamp_attributes

        if not all(isinstance(dem, xdem.dem.DEM) for dem in dems):
            dems = [xdem.dem.DEM.from_array(dem.data, dem.transform, dem.crs, dem.nodata) for dem in dems]

        assert len(dems) == len(timestamps), "The 'dem' and 'timestamps' len differ."

        # Convert the timestamps to datetime64
        self.timestamps = np.array(timestamps).astype("datetime64[ns]")

        # Find the sort indices from the timestamps
        indices = np.argsort(self.timestamps.astype("int64"))
        self.dems = np.asarray(dems)[indices]
        self.ddems: list[dDEM] = []
        # The reference index changes place when sorted
        if isinstance(reference_dem, int):
            self.reference_index = np.argwhere(indices == reference_dem)[0][0]
        elif isinstance(reference_dem, gu.georaster.Raster):
            self.reference_index = np.argwhere(self.dems == reference_dem)[0][0]

        if outlines is None:
            self.outlines: dict[np.datetime64, gu.geovector.Vector] = {}
        elif isinstance(outlines, gu.geovector.Vector):
            self.outlines = {self.timestamps[self.reference_index]: outlines}
        elif all(isinstance(value, gu.geovector.Vector) for value in outlines.values()):
            self.outlines = dict(zip(np.array(list(outlines.keys())).astype("datetime64[ns]"), outlines.values()))
        else:
            raise ValueError(f"Invalid format on 'outlines': {type(outlines)},"
                             " expected one of ['gu.geovector.Vector', 'dict[datetime.datetime, gu.geovector.Vector']")

    @property
    def reference_dem(self) -> gu.georaster.Raster:
        """Get the DEM acting reference."""
        return self.dems[self.reference_index]

    @property
    def reference_timestamp(self) -> np.datetime64:
        """Get the reference DEM timestamp."""
        return self.timestamps[self.reference_index]

    def subtract_dems(self, resampling_method: str = "cubic_spline") -> list[dDEM]:
        """
        Generate dDEMs by subtracting all DEMs to the reference.

        :param resampling_method: The resampling method to use if reprojection is needed.

        :returns: A list of dDEM objects.
        """
        ddems: list[dDEM] = []

        # Subtract every DEM that is available.
        for i, dem in enumerate(self.dems):
            # If the reference DEM is encountered, make a dDEM where dH == 0 (to keep length consistency).
            if dem == self.reference_dem:
                ddem_raster = self.reference_dem.copy()
                ddem_raster.data[:] = 0.0
                ddem = dDEM(
                    ddem_raster,
                    start_time=self.reference_timestamp,
                    end_time=self.reference_timestamp,
                    error=0,
                )
            else:
                ddem = dDEM(
                    raster=xdem.spatial_tools.subtract_rasters(
                        dem,
                        self.reference_dem,
                        reference="second",
                        resampling_method=resampling_method
                    ),
                    start_time=min(self.reference_timestamp, self.timestamps[i]),
                    end_time=max(self.reference_timestamp, self.timestamps[i]),
                    error=None
                )
            ddems.append(ddem)

        self.ddems = ddems
        return self.ddems

    def interpolate_ddems(self, method="linear"):
        """
        Interpolate all the dDEMs in the DEMCollection object using the chosen interpolation method.

        :param method: The chosen interpolation method.
        """
        # TODO: Change is loop to run concurrently
        for ddem in self.ddems:
            ddem.interpolate(method=method)

        return [ddem.filled_data for ddem in self.ddems]

    def get_ddem_mask(self, ddem: dDEM) -> np.ndarray:
        """
        Get a fitting dDEM mask for a provided dDEM.

        The mask is created by evaluating these factors, in order:

        If self.outlines do not exist, a full True boolean mask is returned.
        If self.outlines have keys for the start and end time, their union is returned.
        If self.outlines only have contain the start_time, its mask is returned.
        If len(self.outlines) == 1, the mask of that outline is returned.

        :returns: A mask from the above conditions.
        """
        if not any(ddem is ddem_in_list for ddem_in_list in self.ddems):
            raise ValueError("Given dDEM must be a part of the DEMCollection object.")

        # If both the start and end time outlines exist, a mask is created from their union.
        if ddem.start_time in self.outlines and ddem.end_time in self.outlines:
            mask = np.logical_or(
                self.outlines[ddem.start_time].create_mask(ddem) == 255,
                self.outlines[ddem.end_time].create_mask(ddem) == 255
            )
        # If only start time outlines exist, these should be used as a mask
        elif ddem.start_time in self.outlines:
            mask = self.outlines[ddem.start_time].create_mask(ddem) == 255
        # If only one outlines file exist, use that as a mask.
        elif len(self.outlines) == 1:
            mask = list(self.outlines.values())[0].create_mask(ddem) == 255
        # If no fitting outlines were found, make a full true boolean mask in its stead.
        else:
            mask = np.ones(shape=ddem.data.shape, dtype=bool)
        return mask.reshape(ddem.data.shape)

    def get_dh_series(self, mask: Optional[np.ndarray] = None, nans_ok: bool = False) -> pd.DataFrame:
        """
        Return a dataframe of mean dDEM values and respective areas for every timestamp.

        The values are always compared to the reference DEM timestamp.

        :param mask: Optional. A mask for areas of interest. Overrides potential outlines of the same date.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A dataframe of dH values and respective areas with an Interval[Timestamp] index.
        """
        if len(self.ddems) == 0:
            raise ValueError("dDEMs have not yet been calculated")

        dh_values = pd.DataFrame(columns=["dh", "area"], dtype=float)
        for i, ddem in enumerate(self.ddems):
            # Skip if the dDEM is a self-comparison
            if float(ddem.time) == 0:
                continue

            # Use the provided mask unless it's None, otherwise make a dDEM mask.
            ddem_mask = mask if mask is not None else self.get_ddem_mask(ddem)

            # Warn if the dDEM contains nans and that's not okay
            if ddem.filled_data is None and not nans_ok:
                warnings.warn(f"NaNs found in dDEM ({ddem.start_time} - {ddem.end_time}).")

            data = ddem.data[ddem_mask] if ddem.filled_data is None else ddem.filled_data[ddem_mask]

            mean_dh = np.nanmean(data)
            area = np.count_nonzero(ddem_mask) * self.reference_dem.res[0] * self.reference_dem.res[1]

            dh_values.loc[pd.Interval(pd.Timestamp(ddem.start_time), pd.Timestamp(ddem.end_time))] = mean_dh, area

        return dh_values

    def get_dv_series(self, mask: Optional[np.ndarray] = None, nans_ok: bool = False) -> pd.Series:
        """
        Return a series of mean volume change (dV) for every timestamp.

        The values are always compared to the reference DEM timestamp.

        :param mask: Optional. A mask for areas of interest. Overrides potential outlines of the same date.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of dV values with an Interval[Timestamp] index.
        """
        dh_values = self.get_dh_series(mask=mask, nans_ok=nans_ok)

        return dh_values["area"] * dh_values["dh"]

    def get_cumulative_series(self, kind: str = "dh", mask: Optional[np.ndarray] = None,
                              nans_ok: bool = False) -> pd.Series:
        """
        Get the cumulative dH (elevation) or dV (volume) since the first timestamp.

        :param kind: The kind of series. Can be dh or dv.
        :param mask: Optional. A mask for areas of interest.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of cumulative dH/dV with a Timestamp index.
        """
        if kind.lower() == "dh":
            # Get the dH series (where all indices are: "year to reference_year")
            d_series = self.get_dh_series(mask=mask, nans_ok=nans_ok)["dh"]
        elif kind.lower() == "dv":
            # Get the dV series (where all indices are: "year to reference_year")
            d_series = self.get_dv_series(mask=mask, nans_ok=nans_ok)
        else:
            raise ValueError("Invalid argument: '{dh=}'. Choices: ['dh', 'dv']")

        # Simplify the index to just "year" (implictly still the same as above)
        cumulative_dh = pd.Series(dtype=d_series.dtype)
        cumulative_dh[self.reference_timestamp] = 0.0
        for i, value in zip(d_series.index, d_series.values):
            non_reference_year = [date for date in [i.left, i.right] if date != self.reference_timestamp][0]
            cumulative_dh.loc[non_reference_year] = value

        # Sort the dates (just to be sure. It should already be sorted)
        cumulative_dh.sort_index(inplace=True)
        # Subtract the entire series by the first value to
        cumulative_dh -= cumulative_dh.iloc[0]

        return cumulative_dh

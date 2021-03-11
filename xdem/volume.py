from __future__ import annotations

import datetime
from typing import Any, Optional, Union

import geoutils as gu
import numpy as np
import pandas as pd

import xdem


class dDEM(gu.georaster.Raster):   # pylint: disable=invalid-name
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

    def __str__(self) -> str:
        """Return a summary of the dDEM."""
        return f"dDEM from {self.start_time} to {self.end_time}.\n\n{super().__str__()}"

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


class tDEM:
    """A temporal collection of DEMs."""

    def __init__(self, dems: Union[list[gu.georaster.Raster], list[xdem.dem.DEM]],
                 timestamps: Optional[list[datetime.datetime]] = None,
                 reference_dem: Union[int, gu.georaster.Raster] = 0):
        """
        Create a new temporal DEM collection.

        :param dems: A list of DEMs.
        :param timestamps: A list of DEM timestamps.
        :param reference_dem: An instance or index of which DEM in the 'dems' list is the reference.

        :returns: A new tDEM instance.
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
        self.ddems: Optional[list[dDEM]] = None
        # The reference index changes place when sorted
        if isinstance(reference_dem, int):
            self.reference_index = np.argwhere(indices == reference_dem)[0][0]
        elif isinstance(reference_dem, gu.georaster.Raster):
            self.reference_index = np.argwhere(self.dems == reference_dem)[0][0]

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

    def get_dh_series(self, mask: Optional[np.ndarray] = None) -> pd.Series:
        """
        Return a series of mean dDEM values for every timestamp.

        The values are centered around the reference DEM timestamp.

        :param mask: Optional. A mask for areas of interest.

        :returns: A series of dH values with an Interval[Timestamp] index.
        """
        if self.ddems is None:
            raise ValueError("dDEMs have not yet been calculated")

        dh_values = pd.DataFrame(columns=["dh"], dtype=float)
        for ddem in self.ddems:
            if float(ddem.time) == 0:
                continue
            data = ddem.data if mask is None else ddem.data[mask]
            mean_dh = np.nanmean(data)

            dh_values.loc[pd.Interval(pd.Timestamp(ddem.start_time), pd.Timestamp(ddem.end_time))] = mean_dh

        return dh_values

    def get_cumulative_dh(self, mask: Optional[np.ndarray] = None) -> pd.Series:
        """
        Get the cumulative dH since the first timestamp.

        :param mask: Optional. A mask for areas of interest.

        :returns: A series of cumulative dH with a Timestamp index.
        """
        dh_series = self.get_dh_series(mask=mask)

        dh_interim = pd.Series(dtype=float)
        dh_interim[self.reference_timestamp] = 0.0

        for i, value in zip(dh_series.index, dh_series.values):
            non_reference_year = [date for date in [i.left, i.right] if date != self.reference_timestamp][0]
            dh_interim.loc[non_reference_year] = value[0]

        dh_interim.sort_index(inplace=True)
        dh_interim -= dh_interim.iloc[0]

        cumulative_dh = dh_interim.cumsum()

        return cumulative_dh


from __future__ import annotations

import datetime
import warnings
from typing import Optional, Union

import geoutils as gu
import numpy as np
import pandas as pd

import xdem


class DEMCollection:
    """A temporal collection of DEMs."""

    def __init__(self, dems: Union[list[gu.georaster.Raster], list[xdem.DEM]],
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
            if any([stamp is None for stamp in timestamp_attributes]):
                raise ValueError("'timestamps' not provided and the given DEMs do not all have datetime attributes")

            timestamps = timestamp_attributes

        if not all(isinstance(dem, xdem.DEM) for dem in dems):
            dems = [xdem.DEM.from_array(dem.data, dem.transform, dem.crs, dem.nodata) for dem in dems]

        assert len(dems) == len(timestamps), "The 'dem' and 'timestamps' len differ."

        # Convert the timestamps to datetime64
        self.timestamps = np.array(timestamps).astype("datetime64[ns]")

        # Find the sort indices from the timestamps
        indices = np.argsort(self.timestamps.astype("int64"))
        self.dems = np.asarray(dems)[indices]
        self.ddems: list[xdem.dDEM] = []
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

    def subtract_dems(self, resampling_method: str = "cubic_spline") -> list[xdem.dDEM]:
        """
        Generate dDEMs by subtracting all DEMs to the reference.

        :param resampling_method: The resampling method to use if reprojection is needed.

        :returns: A list of dDEM objects.
        """
        ddems: list[xdem.dDEM] = []

        # Subtract every DEM that is available.
        for i, dem in enumerate(self.dems):
            # If the reference DEM is encountered, make a dDEM where dH == 0 (to keep length consistency).
            if dem == self.reference_dem:
                ddem_raster = self.reference_dem.copy()
                ddem_raster.data[:] = 0.0
                ddem = xdem.dDEM(
                    ddem_raster,
                    start_time=self.reference_timestamp,
                    end_time=self.reference_timestamp,
                    error=0,
                )
            else:
                ddem = xdem.dDEM(
                    raster=xdem.spatial_tools.subtract_rasters(
                        self.reference_dem,
                        dem,
                        reference="first",
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
            ddem.interpolate(method=method, reference_elevation=self.reference_dem, mask=self.get_ddem_mask(ddem))

        return [ddem.filled_data for ddem in self.ddems]

    def get_ddem_mask(self, ddem: xdem.dDEM, outlines_filter: Optional[str] = None) -> np.ndarray:
        """
        Get a fitting dDEM mask for a provided dDEM.

        The mask is created by evaluating these factors, in order:

        If self.outlines do not exist, a full True boolean mask is returned.
        If self.outlines have keys for the start and end time, their union is returned.
        If self.outlines only have contain the start_time, its mask is returned.
        If len(self.outlines) == 1, the mask of that outline is returned.

        :param ddem: The dDEM to create a mask for.
        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".

        :returns: A mask from the above conditions.
        """
        if not any(ddem is ddem_in_list for ddem_in_list in self.ddems):
            raise ValueError("Given dDEM must be a part of the DEMCollection object.")

        if outlines_filter is None:
            outlines = self.outlines
        else:
            outlines = {key: gu.Vector(outline.ds.copy()) for key, outline in self.outlines.items()}
            for key in outlines:
                outlines[key].ds = outlines[key].ds.query(outlines_filter)

        # If both the start and end time outlines exist, a mask is created from their union.
        if ddem.start_time in outlines and ddem.end_time in outlines:
            mask = np.logical_or(
                outlines[ddem.start_time].create_mask(ddem),
                outlines[ddem.end_time].create_mask(ddem)
            )
        # If only start time outlines exist, these should be used as a mask
        elif ddem.start_time in outlines:
            mask = outlines[ddem.start_time].create_mask(ddem)
        # If only one outlines file exist, use that as a mask.
        elif len(outlines) == 1:
            mask = list(outlines.values())[0].create_mask(ddem)
        # If no fitting outlines were found, make a full true boolean mask in its stead.
        else:
            mask = np.ones(shape=ddem.data.shape, dtype=bool)
        return mask.reshape(ddem.data.shape)

    def get_dh_series(self, outlines_filter: Optional[str] = None, mask: Optional[np.ndarray] = None,
                      nans_ok: bool = False) -> pd.DataFrame:
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
            ddem_mask = mask if mask is not None else self.get_ddem_mask(ddem, outlines_filter=outlines_filter)

            # Warn if the dDEM contains nans and that's not okay
            if ddem.filled_data is None and not nans_ok:
                warnings.warn(f"NaNs found in dDEM ({ddem.start_time} - {ddem.end_time}).")

            data = ddem.data[ddem_mask] if ddem.filled_data is None else ddem.filled_data[ddem_mask]

            mean_dh = np.nanmean(data)
            area = np.count_nonzero(ddem_mask) * self.reference_dem.res[0] * self.reference_dem.res[1]

            dh_values.loc[pd.Interval(pd.Timestamp(ddem.start_time), pd.Timestamp(ddem.end_time))] = mean_dh, area

        return dh_values

    def get_dv_series(self, outlines_filter: Optional[str] = None,
                      mask: Optional[np.ndarray] = None, nans_ok: bool = False) -> pd.Series:
        """
        Return a series of mean volume change (dV) for every timestamp.

        The values are always compared to the reference DEM timestamp.

        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".
        :param mask: Optional. A mask for areas of interest. Overrides potential outlines of the same date.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of dV values with an Interval[Timestamp] index.
        """
        dh_values = self.get_dh_series(outlines_filter=outlines_filter, mask=mask, nans_ok=nans_ok)

        return dh_values["area"] * dh_values["dh"]

    def get_cumulative_series(self, kind: str = "dh", outlines_filter: Optional[str] = None,
                              mask: Optional[np.ndarray] = None,
                              nans_ok: bool = False) -> pd.Series:
        """
        Get the cumulative dH (elevation) or dV (volume) since the first timestamp.

        :param kind: The kind of series. Can be dh or dv.
        :param outlines_filter: A query to filter the outline vectors. Example: "name_column == 'specific glacier'".
        :param mask: Optional. A mask for areas of interest.
        :param nans_ok: Warn if NaNs are encountered in a dDEM (it should have been gap-filled).

        :returns: A series of cumulative dH/dV with a Timestamp index.
        """
        if kind.lower() == "dh":
            # Get the dH series (where all indices are: "year to reference_year")
            d_series = self.get_dh_series(mask=mask, outlines_filter=outlines_filter, nans_ok=nans_ok)["dh"]
        elif kind.lower() == "dv":
            # Get the dV series (where all indices are: "year to reference_year")
            d_series = self.get_dv_series(mask=mask, outlines_filter=outlines_filter, nans_ok=nans_ok)
        else:
            raise ValueError("Invalid argument: '{dh=}'. Choices: ['dh', 'dv']")

        # Simplify the index to just "year" (implictly still the same as above)
        cumulative_dh = pd.Series(dtype=d_series.dtype)
        cumulative_dh[self.reference_timestamp] = 0.0
        for i, value in zip(d_series.index, d_series.values):
            non_reference_year = [date for date in [i.left, i.right] if date != self.reference_timestamp][0]
            cumulative_dh.loc[non_reference_year] = -value

        # Sort the dates (just to be sure. It should already be sorted)
        cumulative_dh.sort_index(inplace=True)
        # Subtract the entire series by the first value to
        cumulative_dh -= cumulative_dh.iloc[0]

        return cumulative_dh

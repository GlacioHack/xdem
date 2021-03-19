"""
dem.py provides a class for working with digital elevation models (DEMs)
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import warnings
from typing import Any, Optional, Union

import geoutils as gu
import numpy as np
import pandas as pd
import pyproj
import rasterio.fill
from geoutils.georaster import Raster
from geoutils.satimg import SatelliteImage
from pyproj import Transformer

import xdem


def parse_vref_from_product(product):
    """

    :param product: Product name (typically from satimg.parse_metadata_from_fn)
    :type product: str

    :return: vref_name: Vertical reference name
    :rtype: vref_name: str
    """
    # sources for defining vertical references:
    # AW3D30: https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v11_format_e.pdf
    # SRTMGL1: https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf
    # SRTMv4.1: http://www.cgiar-csi.org/data/srtm-90m-digital-elevation-database-v4-1
    # ASTGTM2/ASTGTM3: https://lpdaac.usgs.gov/documents/434/ASTGTM_User_Guide_V3.pdf
    # NASADEM: https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf !! HGTS is ellipsoid, HGT is EGM96 geoid !!
    # ArcticDEM (mosaic and strips): https://www.pgc.umn.edu/data/arcticdem/
    # REMA (mosaic and strips): https://www.pgc.umn.edu/data/rema/
    # TanDEM-X 90m global: https://geoservice.dlr.de/web/dataguide/tdm90/
    # COPERNICUS DEM: https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198

    if product in ['ArcticDEM/REMA', 'TDM1', 'NASADEM-HGTS']:
        vref_name = 'WGS84'
    elif product in ['AW3D30', 'SRTMv4.1', 'SRTMGL1', 'ASTGTM2', 'NASADEM-HGT']:
        vref_name = 'EGM96'
    elif product in ['COPDEM']:
        vref_name = 'EGM08'
    else:
        vref_name = None

    return vref_name


dem_attrs = ['vref', 'vref_grid', 'ccrs']


class DEM(SatelliteImage):

    def __init__(self, filename_or_dataset, vref_name=None, vref_grid=None, silent=False, **kwargs):
        """
        Load digital elevation model data through the Raster class, parse additional attributes from filename or metadata
        trougth the SatelliteImage class, and then parse vertical reference from DEM product name.
        For manual input, only one of "vref", "vref_grid" or "ccrs" is necessary to set the vertical reference.

        :param filename_or_dataset: The filename of the dataset.
        :type filename_or_dataset: str, DEM, SatelliteImage, Raster, rio.io.Dataset, rio.io.MemoryFile
        :param vref_name: Vertical reference name
        :type vref_name: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_grid: str
        :param silent: Whether to display vertical reference setting
        :param silent: boolean
        """

        # If DEM is passed, simply point back to DEM
        if isinstance(filename_or_dataset, DEM):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            super().__init__(filename_or_dataset, silent=silent, **kwargs)

        if self.nbands > 1:
            raise ValueError('DEM rasters should be composed of one band only')

        # user input
        self.vref = vref_name
        self.vref_grid = vref_grid
        self.ccrs = None

        # trying to get vref from product name (priority to user input)
        self.__parse_vref_from_fn(silent=silent)

    def copy(self, new_array=None):

        new_dem = super().copy()
        # those attributes are immutable, including pyproj.CRS
        # dem_attrs = ['vref','vref_grid','ccrs'] #taken outside of class
        for attrs in dem_attrs:
            setattr(new_dem, attrs, getattr(self, attrs))

        return new_dem

    def __parse_vref_from_fn(self, silent=False):
        """
        Attempts to pull vertical reference from product name identified by SatImg
        """

        if self.product is not None:
            vref = parse_vref_from_product(self.product)
            if vref is not None and self.vref is None:
                if not silent:
                    print('From product name "' + str(self.product)+'": setting vertical reference as ' + str(vref))
                self.vref = vref
            elif vref is not None and self.vref is not None:
                if not silent:
                    print('Leaving user input of ' + str(self.vref) + ' for vertical reference despite reading ' + str(
                        vref) + ' from product name')
            else:
                if not silent:
                    print('Could not find a vertical reference based on product name: "'+str(self.product)+'"')

    def set_vref(self, vref_name=None, vref_grid=None, compute_ccrs=False):
        """
        Set vertical reference with a name or with a grid

        :param vref_name: Vertical reference name
        :type vref_name: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_grid: str
        :param compute_ccrs: Whether to compute the ccrs (read pyproj-data grid file)
        :type compute_ccrs: boolean

        :return:
        """

        # temporary fix for some CRS with proj < 7.2
        def get_crs(filepath: str) -> pyproj.CRS:
            """Get the CRS of a raster with the given filepath."""
            info = subprocess.run(
                ["gdalinfo", "-json", filepath],
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8"
            ).stdout

            wkt_string = json.loads(info)["coordinateSystem"]["wkt"]

            return pyproj.CRS.from_wkt(wkt_string)

        # for names, we only look for WGS84 ellipsoid or the EGM96/EGM08 geoids: those are used 99% of the time
        if isinstance(vref_grid, str):

            if isinstance(vref_name, str):
                print('Both a vertical reference name and vertical grid are provided: defaulting to using grid only.')

            if vref_grid == 'us_nga_egm08_25.tif':
                self.vref = 'EGM08'
                self.vref_grid = vref_grid
            elif vref_grid == 'us_nga_egm96_15.tif':
                self.vref = 'EGM96'
                self.vref_grid = vref_grid
            else:
                if os.path.exists(os.path.join(pyproj.datadir.get_data_dir(), vref_grid)):
                    self.vref = 'Unknown vertical reference name from: '+vref_grid
                    self.vref_grid = vref_grid
                else:
                    raise ValueError('Grid not found in '+str(pyproj.datadir.get_data_dir())+': check if proj-data is '
                                     'installed via conda-forge, the pyproj.datadir, and that you are using a grid available at '
                                     'https://github.com/OSGeo/PROJ-data')
        elif isinstance(vref_name, str):
            if vref_name == 'WGS84':
                self.vref_grid = None
                self.vref = 'WGS84'  # WGS84 ellipsoid
            elif vref_name == 'EGM08':
                self.vref_grid = 'us_nga_egm08_25.tif'  # EGM2008 at 2.5 minute resolution
                self.vref = 'EGM08'
            elif vref_name == 'EGM96':
                self.vref_grid = 'us_nga_egm96_15.tif'  # EGM1996 at 15 minute resolution
                self.vref = 'EGM96'
            else:
                raise ValueError(
                    'Vertical reference name must be either "WGS84", "EGM96" or "EGM08". Otherwise, provide'
                    ' a geoid grid from PROJ DATA: https://github.com/OSGeo/PROJ-data')
        else:
            raise ValueError('Vertical reference name or vertical grid must be a string')

        # temporary fix to get all types of CRS
        if pyproj.proj_version_str >= "7.2.0":
            crs = self.crs
        else:
            crs = get_crs(self.filename)

        # no deriving the ccrs until those are used in a reprojection (requires pyproj-data grids = ~500Mo)
        if compute_ccrs:
            if self.vref == 'WGS84':
                # the WGS84 ellipsoid essentially corresponds to no vertical reference in pyproj
                self.ccrs = pyproj.CRS(crs)
            else:
                # for other vrefs, keep same horizontal projection and add geoid grid (the "dirty" way: because init is so
                # practical and still going to be used for a while)
                # see https://gis.stackexchange.com/questions/352277/including-geoidgrids-when-initializing-projection-via-epsg/352300#352300
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module="pyproj")
                    self.ccrs = pyproj.Proj(init="EPSG:" + str(int(crs.to_epsg())), geoidgrids=self.vref_grid).crs

    def to_vref(self, vref_name='EGM96', vref_grid=None):
        """
        Convert between vertical references: ellipsoidal heights or geoid grids

        :param vref_name: Vertical reference name
        :type vref_grid: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_name: str

        :return:
        """

        # all transformations grids file are described here: https://github.com/OSGeo/PROJ-data
        if self.vref is None and self.vref_grid is None:
            raise ValueError('The current DEM has not vertical reference: need to set one before attempting a conversion '
                             'towards another vertical reference.')
        elif isinstance(self.vref, str) and self.vref_grid is None:
            # to set the vref grid names automatically EGM96/08 for geoids + compute the ccrs
            self.set_vref(vref_name=self.vref, compute_ccrs=True)

        # inital ccrs
        ccrs_init = self.ccrs

        # destination crs

        # set the new reference (before calculation doesn't change anything, we need to update the data manually anyway)
        self.set_vref(vref_name=vref_name, vref_grid=vref_grid, compute_ccrs=True)
        ccrs_dest = self.ccrs

        # transform matrix
        transformer = Transformer.from_crs(ccrs_init, ccrs_dest)
        meta = self.ds.meta
        zz = self.data
        xx, yy = self.coords(offset='center')
        zz_trans = transformer.transform(xx, yy, zz[0, :])[2]
        zz[0, :] = zz_trans

        # update raster
        self._update(metadata=meta, imgdata=zz)


class dDEM(DEM):   # pylint: disable=invalid-name
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

    def __init__(self, dems: Union[list[gu.georaster.Raster], list[DEM]],
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

        if not all(isinstance(dem, DEM) for dem in dems):
            dems = [DEM.from_array(dem.data, dem.transform, dem.crs, dem.nodata) for dem in dems]

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

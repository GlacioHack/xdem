"""Xarray accessor 'dem' for digital elevation models."""
from __future__ import annotations

import xarray as xr

from geoutils.raster.xr_accessor import RasterAccessor, open_raster

from xdem.dem.base import DEMBase

def open_dem(filename: str, **kwargs):

    # Use open raster
    ds = open_raster(filename, **kwargs)

    # TODO: Force the DEM to be of type float, and get the vertical CRS if it exists

    return ds

@xr.register_dataarray_accessor("dem")
class DEMAccessor(DEMBase, RasterAccessor):
    """
    This class defines the Xarray accessor 'dem' for digital elevation models.

    Most attributes and functionalities are inherited from the DEMBase class (also parent of the DEM class) and
    RasterAccessor class defining the 'rst' Xarray accessor for rasters.
    Only methods specific to the functioning of the 'dem' Xarray accessor live in this class: mostly initialization,
    I/O or copying.
    """
    def __init__(self, xarray_obj: xr.DataArray):

        super().__init__()

        self._obj = xarray_obj
        self._area_or_point = self._obj.attrs.get("AREA_OR_POINT", None)
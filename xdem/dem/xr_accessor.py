"""Xarray accessor 'dem' for digital elevation models."""

from __future__ import annotations

from typing import Literal

import xarray as xr
from geoutils.raster.xr_accessor import RasterAccessor, open_raster
from pyproj.crs import VerticalCRS

from xdem.dem.base import DEMBase
from xdem.vcrs import _check_vcrs_input


def open_dem(
    filename: str, vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | int | None = None, **kwargs
):
    """Wrapper around open_raster with vertical CRS input support."""

    # Use open raster
    ds = open_raster(filename, **kwargs)

    # Cast CRS with user-input vertical CRS (returns 2D or 3D) and re-set
    new_crs = _check_vcrs_input(vcrs, ds.rst.crs)
    ds.rst.set_crs(new_crs)

    return ds


@xr.register_dataarray_accessor("dem")
class DEMAccessor(RasterAccessor, DEMBase):
    """
    This class defines the Xarray accessor 'dem' for digital elevation models.

    Most attributes and functionalities are inherited from the DEMBase class (also parent of the DEM class) and
    RasterAccessor class defining the 'rst' Xarray accessor for rasters.
    Only methods specific to the functioning of the 'dem' Xarray accessor live in this class: mostly initialization,
    I/O or copying.
    """

    def __init__(self, xarray_obj: xr.DataArray):

        super().__init__(xarray_obj=xarray_obj)

        self._obj = xarray_obj

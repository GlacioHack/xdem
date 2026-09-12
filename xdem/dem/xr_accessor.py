"""Xarray accessor ``dem`` and file opening for digital elevation models."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import xarray as xr
from geoutils.raster.xr_accessor import RasterAccessor, open_raster
from pyproj.crs import VerticalCRS

from xdem.dem.base import DEMBase
from xdem.dem.dem import DEM
from xdem.vcrs import _check_vcrs_input, _parse_vcrs_name_from_product


def open_dem(
    filename: str | pathlib.Path,
    vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Open a single-band DEM with optional vertical referencing and Dask chunks.

    :param filename: Path to the elevation raster.
    :param vcrs: Vertical CRS name, EPSG code, PyProj CRS or grid path. Overrides file metadata when provided.
    :param kwargs: Keyword arguments passed to :func:`geoutils.open_raster`, including ``chunks``.
    :returns: A DataArray exposing the ``dem`` and ``rst`` accessors, with data loaded on demand.
    """

    # Open the raster and validate its band count without reading elevation values
    ds = open_raster(filename, **kwargs)
    accessor = ds.dem
    if vcrs is None and accessor.vcrs is None and "product" in accessor.tags:
        vcrs = _parse_vcrs_name_from_product(accessor.tags["product"])

    # Store horizontal and vertical references together so copies and file output retain them
    new_crs = _check_vcrs_input(vcrs, accessor.crs)
    if new_crs != accessor.crs:
        accessor.set_crs(new_crs)
    return ds


@xr.register_dataarray_accessor("dem")
class DEMAccessor(DEMBase, RasterAccessor):  # type: ignore[misc]
    """
    Xarray accessor ``dem`` sharing elevation methods with DEM and raster methods with ``rst``.

    Only DataArray validation and conversion to the native DEM class live here. Calculation and metadata behavior
    are inherited from DEMBase and RasterAccessor.
    """

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Validate a single-band elevation raster and initialize its raster accessor."""

        super().__init__(xarray_obj)
        if self.count != 1:
            raise ValueError("DEM rasters should be composed of one band only. Select a single band first.")

    def to_xdem(self) -> DEM:
        """
        Convert the DataArray to an in-memory DEM, preserving its georeferencing.

        :returns: A DEM with the same values and metadata. A Dask source remains lazy after conversion.
        """

        # Compute a separate result so conversion never replaces a lazy source with an eager array
        ds = self._obj.compute() if self._chunks is not None else self._obj
        return DEM.from_array(
            data=ds.data,
            transform=self.transform,
            crs=self.crs,
            nodata=self.nodata,
            area_or_point=self.area_or_point,
            tags=self.tags.copy(),
        )

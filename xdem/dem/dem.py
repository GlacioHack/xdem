# Copyright (c) 2026 xDEM developers
#
# This file is part of xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEM class and functions."""
from __future__ import annotations

import pathlib
import warnings
from typing import Any, Literal

import rasterio as rio
from affine import Affine
from geoutils.raster import RasterType, Raster
from pyproj import CRS
from pyproj.crs import VerticalCRS

from xdem._typing import NDArrayf
from xdem.vcrs import (
    _parse_vcrs_name_from_product,
    _check_vcrs_input
)
from xdem.dem.base import DEMBase

class DEM(Raster, DEMBase):  # type: ignore
    """
    The digital elevation model.

    The DEM has a single main attribute in addition to that inherited from :class:`geoutils.Raster`:
        vcrs: :class:`pyproj.VerticalCRS`
            Vertical coordinate reference system of the DEM.

    Other derivative attributes are:
        vcrs_name: :class:`str`
            Name of vertical CRS of the DEM.
        vcrs_grid: :class:`str`
            Grid path to the vertical CRS of the DEM.
        ccrs: :class:`pyproj.CompoundCRS`
            Compound vertical and horizontal CRS of the DEM.

    The attributes inherited from :class:`geoutils.Raster` are:
        data: :class:`np.ndarray`
            Data array of the DEM, with dimensions corresponding to (count, height, width).
        transform: :class:`affine.Affine`
            Geotransform of the DEM.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the DEM.
        nodata: :class:`int` or :class:`float`
            Nodata value of the DEM.

    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    def __init__(
        self,
        filename_or_dataset: str | RasterType | rio.io.DatasetReader | rio.io.MemoryFile,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
        load_data: bool = False,
        parse_sensor_metadata: bool = False,
        silent: bool = True,
        downsample: int = 1,
        force_nodata: int | float | None = None,
    ) -> None:
        """
        Instantiate a digital elevation model.

        The vertical reference of the DEM can be defined by passing the `vcrs` argument.
        Otherwise, a vertical reference is tentatively parsed from the DEM product name.

        Inherits all attributes from the :class:`geoutils.Raster` class.

        :param filename_or_dataset: The filename of the dataset.
        :param vcrs: Vertical coordinate reference system either as a name ("WGS84", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        :param load_data: Whether to load the array during instantiation. Default is False.
        :param parse_sensor_metadata: Whether to parse sensor metadata from filename and similarly-named metadata files.
        :param silent: Whether to display vertical reference parsing.
        :param downsample: Downsample the array once loaded by a round factor. Default is no downsampling.
        :param force_nodata: Force nodata value to be used (overwrites the metadata). Default reads from metadata.
        """

        self.data: NDArrayf
        self._vcrs: VerticalCRS | Literal["Ellipsoid"] | None = None

        # If DEM is passed, simply point back to DEM
        if isinstance(filename_or_dataset, DEM):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Parse metadata from file not implemented")
                super().__init__(
                    filename_or_dataset,
                    load_data=load_data,
                    parse_sensor_metadata=parse_sensor_metadata,
                    silent=silent,
                    downsample=downsample,
                    force_nodata=force_nodata,
                )

        # Ensure DEM has only one band: self.bands can be None when data is not loaded through the Raster class
        if self.bands is not None and len(self.bands) > 1:
            raise ValueError(
                "DEM rasters should be composed of one band only. Either use argument `bands` to specify "
                "a single band on opening, or use .split_bands() on an opened raster."
            )

        # If no vertical CRS was provided by the user or defined in the CRS
        if vcrs is None and "product" in self.tags:
            vcrs = _parse_vcrs_name_from_product(self.tags["product"])

        # Cast CRS with vertical CRS (returns 2D or 3D) and re-set
        new_crs = _check_vcrs_input(vcrs, self.crs)
        self.set_crs(new_crs)
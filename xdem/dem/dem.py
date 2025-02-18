# Copyright (c) 2024 xDEM developers
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

from xdem._typing import MArrayf, NDArrayf
from xdem.vcrs import (
    _parse_vcrs_name_from_product,
    _vcrs_from_crs,
    _vcrs_from_user_input,
)
from xdem.dem.base import DEMBase

dem_attrs = ["_vcrs", "_vcrs_name", "_vcrs_grid"]


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
        self._vcrs_name: str | None = None
        self._vcrs_grid: str | None = None

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

        # If the CRS in the raster metadata has a 3rd dimension, could set it as a vertical reference
        vcrs_from_crs = _vcrs_from_crs(CRS(self.crs))
        if vcrs_from_crs is not None:
            # If something was also provided by the user, user takes precedence
            # (we leave vcrs as it was for input)
            if vcrs is not None:
                # Raise a warning if the two are not the same
                vcrs_user = _vcrs_from_user_input(vcrs)
                if not vcrs_from_crs == vcrs_user:
                    warnings.warn(
                        "The CRS in the raster metadata already has a vertical component, "
                        "the user-input '{}' will override it.".format(vcrs)
                    )
            # Otherwise, use the one from the raster 3D CRS
            else:
                vcrs = vcrs_from_crs

        # If no vertical CRS was provided by the user or defined in the CRS
        if vcrs is None and "product" in self.tags:
            vcrs = _parse_vcrs_name_from_product(self.tags["product"])

        # If a vertical reference was parsed or provided by user
        if vcrs is not None:
            self.set_vcrs(vcrs)

    def copy(self, new_array: NDArrayf | None = None) -> DEM:
        """
        Copy the DEM, possibly updating the data array.

        :param new_array: New data array.

        :return: Copied DEM.
        """

        new_dem = super().copy(new_array=new_array)  # type: ignore
        # The rest of attributes are immutable, including pyproj.CRS
        for attrs in dem_attrs:
            setattr(new_dem, attrs, getattr(self, attrs))

        return new_dem  # type: ignore

    @classmethod
    def from_array(
            cls: type[DEM],
            data: NDArrayf | MArrayf,
            transform: tuple[float, ...] | Affine,
            crs: CRS | int | None,
            nodata: int | float | None = None,
            area_or_point: Literal["Area", "Point"] | None = None,
            tags: dict[str, Any] = None,
            cast_nodata: bool = True,
            vcrs: (
                    Literal["Ellipsoid"] | Literal["EGM08"] | Literal[
                "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
            ) = None,
    ) -> DEM:
        """Create a DEM from a numpy array and the georeferencing information.

        :param data: Input array.
        :param transform: Affine 2D transform. Either a tuple(x_res, 0.0, top_left_x,
            0.0, y_res, top_left_y) or an affine.Affine object.
        :param crs: Coordinate reference system. Either a rasterio CRS, or an EPSG integer.
        :param nodata: Nodata value.
        :param area_or_point: Pixel interpretation of the raster, will be stored in AREA_OR_POINT metadata.
        :param tags: Metadata stored in a dictionary.
        :param cast_nodata: Automatically cast nodata value to the default nodata for the new array type if not
            compatible. If False, will raise an error when incompatible.
        :param vcrs: Vertical coordinate reference system.

        :returns: DEM created from the provided array and georeferencing.
        """
        # We first apply the from_array of the parent class
        rast = Raster.from_array(
            data=data,
            transform=transform,
            crs=crs,
            nodata=nodata,
            area_or_point=area_or_point,
            tags=tags,
            cast_nodata=cast_nodata,
        )
        # Then add the vcrs to the class call (that builds on top of the parent class)
        return cls(filename_or_dataset=rast, vcrs=vcrs)
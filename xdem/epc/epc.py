# Copyright (c) 2025 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for the EPC class for elevation point clouds."""
from __future__ import annotations

import pathlib
import warnings
from typing import Literal, overload

import geopandas as gpd
import numpy as np
from geoutils import PointCloud
from geoutils.raster import Raster, RasterType
from pyproj.crs import CRS, CompoundCRS, VerticalCRS
from shapely.geometry.base import BaseGeometry

import xdem
from xdem import coreg
from xdem._typing import NDArrayb, NDArrayf, MArrayf
from xdem.vcrs import (
    _build_ccrs_from_crs_and_vcrs,
    _grid_from_user_input,
    _transform_zz,
    _vcrs_from_crs,
    _vcrs_from_user_input,
)

epc_attrs = ["_vcrs", "_vcrs_name", "_vcrs_grid"]


class EPC(PointCloud):  # type: ignore
    """
    The georeferenced elevation point cloud.

    An elevation point cloud is a vector of either 3D point geometries or 2D point geometries associated to
    an elevation value from a main data column, optionally with auxiliary data columns.

     Main attributes:
        ds: :class:`geopandas.GeoDataFrame`
            Geodataframe of the point cloud.
        data_column: str
            Name of point cloud data column.
        crs: :class:`pyproj.crs.CRS`
            Coordinate reference system of the point cloud.
        bounds: :class:`rio.coords.BoundingBox`
            Coordinate bounds of the point cloud.


    All other attributes are derivatives of those attributes, or read from the file on disk.
    See the API for more details.
    """

    def __init__(
        self,
        filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry,
        data_column: str | None = None,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
    ):
        """
        Instantiate an elevation point cloud from either a z column name and a vector (filename, GeoPandas
        dataframe or series, or a Shapely geometry), or only with a point cloud file type.

        :param filename_or_dataset: Path to point cloud file, or GeoPandas dataframe or series, or Shapely geometry.
        :param data_column: If the point geometries are only 2D, the data column to define the Z coordinate of the
            elevation point cloud.
        """

        self._vcrs: VerticalCRS | Literal["Ellipsoid"] | None = None
        self._vcrs_name: str | None = None
        self._vcrs_grid: str | None = None

        super().__init__(filename_or_dataset=filename_or_dataset, data_column=data_column)

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
                        "The CRS in the point cloud metadata already has a vertical component, "
                        "the user-input '{}' will override it.".format(vcrs)
                    )
            # Otherwise, use the one from the raster 3D CRS
            else:
                vcrs = vcrs_from_crs

        # If a vertical reference was parsed or provided by user
        if vcrs is not None:
            self.set_vcrs(vcrs)

    @property
    def _has_z(self) -> bool:
        """Whether the point geometries all have a Z coordinate or not."""

        return all(p.has_z for p in self.ds.geometry)

    @property
    def data(self) -> NDArrayf:
        """
        Data of the elevation point cloud.

        Points to either the Z axis of the point geometries, or the associated data column of the geodataframe.
        """
        # Triggers the loading mechanism through self.ds
        if not self._has_z:
            return self.ds[self.data_column].values
        else:
            return self.geometry.z.values

    @data.setter
    def data(self, new_data: NDArrayf) -> None:
        """Set new data for the point cloud."""

        if not self._has_z:
            self.ds[self.data_column] = new_data
        else:
            self.ds.geometry = gpd.points_from_xy(x=self.geometry.x, y=self.geometry.y, z=new_data, crs=self.crs)

    def copy(self, new_array: NDArrayf | NDArrayb | None = None) -> EPC:
        """
        Copy the elevation point cloud, possibly updating the data array.

        :param new_array: New data array.

        :return: Copied elevation point cloud.
        """

        new_epc = super().copy(new_array=new_array)  # type: ignore
        # The rest of attributes are immutable, including pyproj.CRS
        for attrs in epc_attrs:
            setattr(new_epc, attrs, getattr(self, attrs))

        return new_epc  # type: ignore

    @property
    def vcrs(self) -> VerticalCRS | Literal["Ellipsoid"] | None:
        """Vertical coordinate reference system of the elevation point cloud."""

        return self._vcrs

    @property
    def vcrs_grid(self) -> str | None:
        """Grid path of vertical coordinate reference system of the DEM."""

        return self._vcrs_grid

    @property
    def vcrs_name(self) -> str | None:
        """Name of vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            # If it is the ellipsoid
            if isinstance(self.vcrs, str):
                # Need to call CRS() here to make it work with rasterio.CRS...
                vcrs_name = f"Ellipsoid (No vertical CRS). Datum: {CRS(self.crs).ellipsoid.name}."
            # Otherwise, return the vertical reference name
            else:
                vcrs_name = self.vcrs.name
        else:
            vcrs_name = None

        return vcrs_name

    @property
    def ccrs(self) -> CompoundCRS | CRS | None:
        """Compound horizontal and vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            ccrs = _build_ccrs_from_crs_and_vcrs(crs=self.crs, vcrs=self.vcrs)
            return ccrs
        else:
            return None

    def set_vcrs(
        self,
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int,
    ) -> None:
        """
        Set the vertical coordinate reference system of the elevation point cloud.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Get vertical CRS and set it and the grid
        self._vcrs = _vcrs_from_user_input(vcrs_input=new_vcrs)
        self._vcrs_grid = _grid_from_user_input(vcrs_input=new_vcrs)

    @overload
    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        *,
        inplace: Literal[False] = False,
    ) -> EPC: ...

    @overload
    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        *,
        inplace: Literal[True],
    ) -> None: ...

    @overload
    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        *,
        inplace: bool = False,
    ) -> EPC | None: ...

    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        inplace: bool = False,
    ) -> EPC | None:
        """
        Convert the elevation point cloud to another vertical coordinate reference system.

        :param vcrs: Destination vertical CRS. Either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param force_source_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `vcrs`.
        :param inplace: Whether to return a new EPC (default) or the same EPC updated in-place.

        :return: Elevation point cloud with vertical reference transformed, or None.
        """

        if self.vcrs is None and force_source_vcrs is None:
            raise ValueError(
                "The current EPC has no vertical reference, define one with .set_vref() "
                "or by passing `src_vcrs` to perform a conversion."
            )

        # Initial Compound CRS (only exists if vertical CRS is not None, as checked above)
        if force_source_vcrs is not None:
            # Warn if a vertical CRS already existed for that EPC
            if self.vcrs is not None:
                warnings.warn(
                    category=UserWarning,
                    message="Overriding the vertical CRS of the EPC with the one provided in `src_vcrs`.",
                )
            src_ccrs = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=force_source_vcrs)
        else:
            src_ccrs = self.ccrs

        # New destination Compound CRS
        dst_ccrs = _build_ccrs_from_crs_and_vcrs(self.crs, vcrs=_vcrs_from_user_input(vcrs_input=vcrs))

        # If both compound CCRS are equal, do not run any transform
        if src_ccrs.equals(dst_ccrs):
            warnings.warn(
                message="Source and destination vertical CRS are the same, skipping vertical transformation.",
                category=UserWarning,
            )
            return None

        # Transform elevation with new vertical CRS
        zz = self.data
        xx, yy = self.geometry.x.values, self.geometry.y.values
        zz_trans = _transform_zz(crs_from=src_ccrs, crs_to=dst_ccrs, xx=xx, yy=yy, zz=zz)
        new_data = zz_trans.astype(self.data.dtype)  # type: ignore

        # If inplace, update EPC and vcrs
        if inplace:
            self.data = new_data
            self.set_vcrs(new_vcrs=vcrs)
            return None
        # Otherwise, return new EPC
        else:
            epc = self.copy(new_array=new_data)
            epc.set_vcrs(new_vcrs=vcrs)
            return epc

    def coregister_3d(  # type: ignore
        self,
        reference_elev: xdem.DEM | gpd.GeoDataFrame | EPC,
        coreg_method: coreg.Coreg,
        inlier_mask: Raster | NDArrayb = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] = None,
        random_state: int | np.random.Generator | None = None,
        **kwargs,
    ) -> EPC:
        """
        Coregister elevation point cloud to a reference elevation data in three dimensions.

        Any coregistration method or pipeline from xdem.Coreg can be passed. Default is only horizontal and vertical
        shifts of Nuth and Kääb (2011).

        :param reference_elev: Reference elevation, DEM or elevation point cloud, for the alignment.
        :param coreg_method: Coregistration method or pipeline.
        :param inlier_mask: Optional. 2D boolean array or mask of areas to include in the analysis (inliers=True).
        :param bias_vars: Optional, only for some bias correction methods. 2D array or rasters of bias variables used.
        :param random_state: Random state or seed number to use for subsampling and optimizer.

        :param kwargs: Keyword arguments passed to Coreg.fit().

        :return: Coregistered DEM
        """

        src_epc = self.copy()

        # Check inputs
        if not isinstance(coreg_method, coreg.Coreg):
            raise ValueError("Argument `coreg_method` must be an xdem.coreg instance (e.g. xdem.coreg.NuthKaab()).")

        aligned_epc = coreg_method.fit_and_apply(
            reference_elev,
            src_epc,
            inlier_mask=inlier_mask,
            random_state=random_state,
            bias_vars=bias_vars,
            **kwargs,
        )

        return aligned_epc

    # def estimate_uncertainty(self):

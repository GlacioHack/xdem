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

"""EPC class for elevation point clouds."""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import geopandas as gpd
from geoutils import PointCloud
from geoutils.multiproc import MultiprocConfig
from pyproj import CRS
from pyproj.crs import VerticalCRS
from shapely.geometry.base import BaseGeometry

from xdem.epc.base import EPCBase
from xdem.vcrs import _check_vcrs_input


class EPC(EPCBase, PointCloud):  # type: ignore[misc]
    """
    A georeferenced elevation point cloud with optional auxiliary data columns.

    Elevation comes from a selected column or from 3D point geometry. The CRS stores both horizontal and vertical
    referencing. File opening reads metadata first; point values are loaded on demand through GeoUtils.
    """

    def __init__(
        self,
        filename_or_dataset: str | pathlib.Path | gpd.GeoDataFrame | gpd.GeoSeries | BaseGeometry | PointCloud,
        data_column: str | None = None,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
    ) -> None:
        """
        Open elevation points or wrap an existing geospatial dataset.

        :param filename_or_dataset: File path, point cloud, GeoDataFrame, GeoSeries or point geometry.
        :param data_column: Column containing elevations. LAS/LAZ defaults to ``Z``; 3D geometries need no column.
        :param vcrs: Vertical CRS name, EPSG code, PyProj CRS or grid path, overriding file metadata when provided.
        """

        # Preserve GeoUtils' loading and optional LAS/LAZ dependency handling
        super().__init__(filename_or_dataset=filename_or_dataset, data_column=data_column)
        new_crs = _check_vcrs_input(vcrs, self.crs)
        if new_crs != self.crs:
            self._set_vcrs_crs(new_crs)

    def _set_vcrs_crs(self, new_crs: CRS) -> None:
        """Update CRS metadata immediately and loaded geometry only when it already exists."""

        self._crs = new_crs
        if self.is_loaded:
            # Copy the frame so setting a vertical reference does not mutate a constructor input
            self._ds = self.ds.set_crs(new_crs, allow_override=True)

    def load(
        self,
        columns: Literal["all", "main"] | list[str] = "main",
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Load point values while retaining a vertical reference assigned before loading.

        :param columns: LAS dimensions to read, using ``main``, ``all`` or a list of dimension names.
        :param mp_config: Optional multiprocessing configuration for loading LAS/LAZ partitions.
        :param kwargs: Additional arguments passed to the GeoUtils point cloud loader.
        """

        # File readers restore the original file CRS, so preserve the current metadata across loading
        crs = self.crs
        super().load(columns=columns, mp_config=mp_config, **kwargs)
        if crs != self.ds.crs:
            self.ds.set_crs(crs, allow_override=True, inplace=True)

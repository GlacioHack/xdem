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

from geoutils import PointCloud

class EPC(PointCloud):
    """
    The georeferenced elevation point cloud.

    An elevation point cloud is a vector of either 3D point geometries, or 2D point geometries associated to
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

    def __init__(filename_or_dataset, z_column = None, vcrs = None):
    
        self._vcrs: VerticalCRS | Literal["Ellipsoid"] | None = None
        self._vcrs_name: str | None = None
        self._vcrs_grid: str | None = None

        super().__init__()

    @property
    def vcrs(self):
        """Vertical coordinate reference system of the elevation point cloud."""

        return self._vcrs

    def set_vcrs(self):

    def to_vcrs(self):

    def coregister_3d(self):

    def estimate_uncertainty(self):


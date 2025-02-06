"""Module for the ElevationPointCloud class."""
from __future__ import annotations

from geoutils import PointCloud

class EPC(PointCloud):
    """
    The georeferenced elevation point cloud.

    An elevation point cloud is a vector of 3D point geometries, or a vector of 2D point geometries associated to
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

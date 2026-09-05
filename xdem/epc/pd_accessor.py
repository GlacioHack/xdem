# Copyright (c) 2026 xDEM developers
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

"""Pandas accessor ``epc`` and file opening for elevation point clouds."""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from geoutils.pointcloud.base import _get_dataframe_attrs, _set_dataframe_attrs
from geoutils.pointcloud.pd_accessor import PointCloudAccessor, open_pointcloud
from pyproj import CRS
from pyproj.crs import VerticalCRS

from xdem._misc import import_optional
from xdem.epc.base import EPCBase
from xdem.epc.epc import EPC
from xdem.vcrs import _check_vcrs_input

_DASK_ACCESSOR_REGISTERED = False


def _register_dask_epc_accessor() -> None:
    """Register ``epc`` in Dask's separate accessor registry when a lazy point cloud is first requested."""

    global _DASK_ACCESSOR_REGISTERED

    # Dask warns if the same accessor is registered more than once
    if _DASK_ACCESSOR_REGISTERED:
        return

    # Register on Dask separately from Pandas while keeping the dependency optional
    # https://docs.dask.org/en/stable/dataframe-extend.html#accessors
    import_optional("dask")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
        warnings.filterwarnings("ignore", message="registration of accessor.*", category=UserWarning)
        from dask.dataframe.extensions import register_dataframe_accessor

        register_dataframe_accessor("epc")(EPCAccessor)
    _DASK_ACCESSOR_REGISTERED = True


def open_epc(
    filename: str | pathlib.Path,
    data_column: str | None = None,
    vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | VerticalCRS | str | pathlib.Path | int | None = None,
    columns: Literal["all", "main"] | list[str] = "main",
    chunks: int | None = None,
) -> gpd.GeoDataFrame | Any:
    """
    Open elevation points as a GeoDataFrame, optionally partitioned with Dask.

    :param filename: Point cloud file path, including LAS, LAZ and supported vector formats.
    :param data_column: Elevation column, defaulting to native ``Z`` for LAS/LAZ files.
    :param vcrs: Vertical CRS name, EPSG code, PyProj CRS or grid path, overriding file metadata when provided.
    :param columns: LAS dimensions to read: ``main``, ``all`` or a list of names.
    :param chunks: Points per Dask partition. None loads the GeoDataFrame eagerly.
    :returns: An eager or lazy GeoDataFrame with both ``epc`` and GeoUtils ``pc`` behavior.
    """

    # Reuse GeoUtils' readers and register Dask only for a partitioned result
    ds = open_pointcloud(filename, data_column=data_column, columns=columns, chunks=chunks)
    if chunks is not None:
        _register_dask_epc_accessor()

    # Assign the combined CRS through the accessor without reading lazy partitions
    new_crs = _check_vcrs_input(vcrs, ds.epc.crs)
    if new_crs != ds.epc.crs:
        ds.epc._set_vcrs_crs(new_crs)
    return ds


@pd.api.extensions.register_dataframe_accessor("epc")
class EPCAccessor(EPCBase, PointCloudAccessor):  # type: ignore[misc]
    """Expose EPC elevation methods alongside the GeoUtils ``pc`` Pandas accessor."""

    def _set_vcrs_crs(self, new_crs: CRS) -> None:
        """Set geometry and cached CRS metadata without computing Dask partitions."""

        # Both GeoPandas and Dask-GeoPandas support metadata assignment through their CRS property
        attrs = _get_dataframe_attrs(self.ds)
        if self._is_dask:
            self.ds.crs = new_crs
        else:
            self.ds.set_crs(new_crs, allow_override=True, inplace=True)
        attrs["crs"] = new_crs
        _set_dataframe_attrs(self.ds, attrs)

    def to_xdem(self) -> EPC:
        """
        Convert the dataframe to an in-memory elevation point cloud.

        :returns: An EPC with identical elevations, auxiliary columns and CRS. Lazy sources remain lazy.
        """

        # Computing returns a separate dataframe and never replaces the source's partitions
        ds = self.load() if self._is_dask else self.ds.copy()
        return EPC(ds, data_column=self.data_column)

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

"""Module of EPCBase class, parent of EPC object and the ``epc`` Pandas accessor."""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, Literal

import geopandas as gpd
import numpy as np
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.base import PointCloudBase
from pyproj.crs import VerticalCRS

from xdem.coreg import Coreg
from xdem.vcrs import _to_vcrs_1d, _VerticalReference


class EPCBase(PointCloudBase, _VerticalReference):  # type: ignore[misc]
    """Share elevation methods between EPC and the 'epc' Pandas accessor.

    This is an internal base class built on PointCloudBase. It inherits _VerticalReference so that DEMBase and EPCBase
    reuse the same VCRS metadata and manipulation without code duplication. ``vcrs`` exposes the vertical part,
    while the inherited ``crs`` property contains the complete 3D CRS.
    """

    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        mp_config: MultiprocConfig | None = None,
        *,
        inplace: bool = False,
    ) -> Any:
        """
        Convert point elevations to another vertical coordinate reference system.

        :param vcrs: Destination vertical CRS name, EPSG code, PyProj CRS or grid path.
        :param force_source_vcrs: Override the source vertical CRS for this transformation.
        :param mp_config: Optional configuration for transforming point partitions in workers.
        :param inplace: Update this object instead of returning a copy. Deprecated; use the returned result.
        :returns: An EPC or GeoDataFrame matching the input interface. Dask inputs produce a lazy GeoDataFrame.
        """

        if inplace:
            warnings.warn(
                "Argument 'inplace' is deprecated and will be removed in future versions. "
                "Use epc = epc.to_vcrs() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self._is_dask:
                raise ValueError("In-place vertical transformation is not supported for Dask point clouds.")

        # Transform in the selected backend, retaining auxiliary columns and row order
        result = _to_vcrs_1d(self, vcrs, force_source_vcrs=force_source_vcrs, mp_config=mp_config)
        if inplace:
            self.ds = result if self._is_pd else result.ds
            self._set_vcrs_crs(result.crs)
            return None
        return result

    def coregister_3d(
        self,
        reference_elev: Any,
        coreg_method: Coreg,
        inlier_mask: Any = None,
        bias_vars: dict[str, Any] | None = None,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Align an elevation point cloud to reference elevation data in three dimensions.

        :param reference_elev: Reference DEM or elevation point cloud, including eager accessors.
        :param coreg_method: Coregistration method or pipeline to fit and apply.
        :param inlier_mask: Boolean raster or array selecting stable terrain.
        :param bias_vars: Auxiliary raster or array variables required by the chosen correction.
        :param random_state: Seed or generator for reproducible subsampling and fitting.
        :param kwargs: Additional arguments passed to :meth:`xdem.coreg.Coreg.fit_and_apply`.
        :returns: An aligned EPC or eager GeoDataFrame matching the input interface.
        """

        from xdem.coreg.base import _as_eager_elevation

        # Validate before loading or copying the source; Dask coregistration is a separate feature
        if not isinstance(coreg_method, Coreg):
            raise ValueError("Argument `coreg_method` must be an xdem.coreg instance (e.g. xdem.coreg.NuthKaab()).")
        source = _as_eager_elevation(self)
        reference = _as_eager_elevation(reference_elev)
        mask = _as_eager_elevation(inlier_mask)
        variables = None if bias_vars is None else {name: _as_eager_elevation(var) for name, var in bias_vars.items()}

        # Use the native point cloud interface to preserve the chosen elevation column through fitting
        aligned = coreg_method.fit_and_apply(
            reference,
            source.copy(),
            inlier_mask=mask,
            bias_vars=variables,
            random_state=random_state,
            **kwargs,
        )

        # Restore 3D geometry when the source stored elevations in its point coordinates
        result = aligned.ds
        if self.data_column is None:
            result = self.ds.copy()
            result.geometry = gpd.points_from_xy(
                aligned.geometry.x, aligned.geometry.y, z=aligned.data, crs=aligned.crs
            )
            result.attrs["data_column"] = None
        return self._cast_pointcloud_output(result)

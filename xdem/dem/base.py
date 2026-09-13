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

"""Module of DEMBase class, parent of DEM class and ``dem`` accessor."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, Union

import geopandas as gpd
import geoutils as gu
import numpy as np
import xarray as xr
from geoutils import profiler
from geoutils._dispatch import get_geo_attr
from geoutils.multiproc import MultiprocConfig
from geoutils.raster import Raster, RasterType
from geoutils.raster.base import RasterBase
from geoutils.stats import nmad
from pyproj import CRS
from pyproj.crs import VerticalCRS

import xdem
from xdem import coreg, terrain
from xdem._misc import copy_doc, deprecate
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg import Coreg
from xdem.coreg.base import _as_eager_elevation
from xdem.vcrs import (
    _to_vcrs_2d,
    _VerticalReference,
)

# Input/output is a RasterType (= Raster or RasterAccessor subclass)
DEMType = TypeVar("DEMType", bound="DEMBase")
# For inputs, we also accept a xr.DataArray
DEMLike = Union["DEMBase", xr.DataArray]


class DEMBase(RasterBase, _VerticalReference):  # type: ignore[misc]
    """Share elevation methods between DEM and the 'dem' Xarray accessor.

    This is an internal base class built on RasterBase. It inherits _VerticalReference so that DEMBase and EPCBase
    reuse the same VCRS metadata and manipulation without code duplication. ``vcrs`` exposes the vertical part,
    while the inherited ``crs`` property contains the complete 3D CRS.
    """

    if TYPE_CHECKING:
        data: Any

    def _set_vcrs_crs(self, new_crs: CRS) -> None:
        """Store the vertical reference in the raster's CRS metadata."""

        self.set_crs(new_crs)

    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | VerticalCRS | int,
        force_source_vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | VerticalCRS | int | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> DEMLike | None:
        """
        Convert the DEM to another vertical coordinate reference system.

        :param vcrs: Destination vertical CRS. Either as a name ("WGS84", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param force_source_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `vcrs`.
        :param mp_config: Multiprocessing configuration.

        :param kwargs: Deprecated ``inplace`` option for updating the source instead of returning a copy.
        :returns: DEM or DataArray with transformed elevations, or None when updating in place.
        """

        # Raise deprecation warning for old in-place behaviour
        if "inplace" in kwargs and kwargs["inplace"]:
            warnings.warn(
                "Argument 'inplace' is deprecated and will be removed in future versions. "
                "Use dem = dem.to_vcrs() instead.",
                category=DeprecationWarning,
            )
            inplace = True
        else:
            inplace = False

        # Apply transformation
        new_dem = _to_vcrs_2d(dem=self, dst_vcrs=vcrs, force_source_vcrs=force_source_vcrs, mp_config=mp_config)

        # Keep logic below until we deprecate 'inplace'
        # If early exit because no transformation was required
        if new_dem is None:
            if inplace:
                return None
            if not self._is_xr and not self.is_loaded:
                return self.__class__(self)
            return self.copy(deep=False)

        # If inplace, update DEM and vcrs
        if inplace:
            self.data = new_dem.data
            self.set_crs(new_crs=get_geo_attr(new_dem, "crs"))
            return None
        return new_dem

    @copy_doc(terrain, remove_dem_res_params=True)
    def slope(
        self,
        method: Literal["Horn", "ZevenbergThorne"] = None,
        surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
        degrees: bool = True,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        # Deprecating method
        if method is not None:
            warnings.warn(
                "'method' is deprecated, use 'surface_fit' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            surface_fit = method  # override
            method = None

        return terrain.slope(self, surface_fit=surface_fit, degrees=degrees, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def aspect(
        self,
        method: Literal["Horn", "ZevenbergThorne"] = None,
        surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
        degrees: bool = True,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        # Deprecating method
        if method is not None:
            warnings.warn(
                "'method' is deprecated, use 'surface_fit' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            surface_fit = method  # override
            method = None

        return terrain.aspect(self, surface_fit=surface_fit, degrees=degrees, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def hillshade(
        self,
        method: Literal["Horn", "ZevenbergThorne"] = None,
        surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
        azimuth: float = 315.0,
        altitude: float = 45.0,
        z_factor: float = 1.0,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        # Deprecating method
        if method is not None:
            warnings.warn(
                "'method' is deprecated, use 'surface_fit' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            surface_fit = method  # override
            method = None

        return terrain.hillshade(
            self,
            surface_fit=surface_fit,
            azimuth=azimuth,
            altitude=altitude,
            z_factor=z_factor,
            mp_config=mp_config,
        )

    @copy_doc(terrain, remove_dem_res_params=True)
    def curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.curvature(self, surface_fit=surface_fit, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def profile_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.profile_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def tangential_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.tangential_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def planform_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.planform_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def flowline_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.flowline_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def max_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.max_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def min_curvature(
        self,
        surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
        curv_method: Literal["geometric", "directional"] = "geometric",
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.min_curvature(self, surface_fit=surface_fit, curv_method=curv_method, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def topographic_position_index(
        self,
        window_size: int = 3,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.topographic_position_index(self, window_size=window_size, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def terrain_ruggedness_index(
        self,
        method: Literal["Riley", "Wilson"] = "Riley",
        window_size: int = 3,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.terrain_ruggedness_index(self, method=method, window_size=window_size, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def roughness(self, window_size: int = 3, mp_config: MultiprocConfig | None = None) -> RasterType:

        return terrain.roughness(self, window_size=window_size, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def rugosity(self, mp_config: MultiprocConfig | None = None) -> RasterType:

        return terrain.rugosity(self, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def fractal_roughness(self, window_size_fractal: int = 13, mp_config: MultiprocConfig | None = None) -> RasterType:

        return terrain.fractal_roughness(self, window_size_fractal=window_size_fractal, mp_config=mp_config)

    @copy_doc(terrain, remove_dem_res_params=True)
    def texture_shading(
        self,
        alpha: float = 0.8,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType:

        return terrain.texture_shading(
            self,
            alpha=alpha,
            mp_config=mp_config,
        )

    @copy_doc(terrain, remove_dem_res_params=True)
    def get_terrain_attribute(self, attribute: str | list[str], **kwargs: Any) -> RasterType | list[RasterType]:
        return terrain.get_terrain_attribute(self, attribute=attribute, **kwargs)

    @profiler.profile("xdem.dem.coregister_3d", memprof=True)
    def coregister_3d(  # type: ignore
        self,
        reference_elev: DEMLike | gpd.GeoDataFrame | xdem.EPC,
        coreg_method: coreg.Coreg,
        inlier_mask: Raster | NDArrayb = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] = None,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> DEMLike:
        """
        Coregister DEM to a reference DEM in three dimensions.

        Any coregistration method or pipeline from xdem.Coreg can be passed. Default is only horizontal and vertical
        shifts of Nuth and Kääb (2011).

        :param reference_elev: Reference elevation, DEM or elevation point cloud, for the alignment.
        :param coreg_method: Coregistration method or pipeline.
        :param inlier_mask: Optional. 2D boolean array or mask of areas to include in the analysis (inliers=True).
        :param bias_vars: Optional, only for some bias correction methods. 2D array or rasters of bias variables used.
        :param random_state: Random state or seed number to use for subsampling and optimizer.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, only \
            the array/transform will be updated (if possible) and no resampling is done. \
            Useful to avoid spreading data gaps.
        :param kwargs: Keyword arguments passed to Coreg.fit().

        :return: Coregistered DEM
        """

        # Check inputs before loading elevation values
        if not isinstance(coreg_method, Coreg):
            raise ValueError("Argument `coreg_method` must be an xdem.coreg instance (e.g. xdem.coreg.NuthKaab()).")

        # Reuse eager coregistration with equivalent GeoUtils objects for every accessor input
        source = _as_eager_elevation(self)
        reference = _as_eager_elevation(reference_elev)
        mask = _as_eager_elevation(inlier_mask)
        variables = None if bias_vars is None else {name: _as_eager_elevation(var) for name, var in bias_vars.items()}
        aligned_dem = coreg_method.fit_and_apply(
            reference,
            source.copy(),
            inlier_mask=mask,
            random_state=random_state,
            bias_vars=variables,
            **kwargs,
        )

        return self._cast_raster_output(aligned_dem)

    def estimate_error_structure(
        self,
        other_elev: DEMLike | gpd.GeoDataFrame | xdem.EPC,
        *,
        stable_terrain: Raster | gu.Vector | NDArrayb | gpd.GeoDataFrame | None = None,
        predictors: Mapping[str, Any] | tuple[Any, ...] | None = None,
        components: Mapping[str, Mapping[str, Any]] | None = None,
        other_error: Literal["negligible", "same"] = "negligible",
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> xdem.ErrorStructure:
        """Estimate named error components from another elevation dataset on stable terrain.

        The default model combines predictor dependent short range errors with a constant long range component.
        Terrain slope and maximum curvature are used as predictors unless ``predictors`` is given explicitly.

        :param other_elev: Comparison DEM or elevation point cloud.
        :param stable_terrain: Spatial or Boolean mask where elevation differences represent error.
        :param predictors: Named magnitude predictors or an ordered tuple using terrain attribute names where possible.
        :param components: Ordered component specifications defining magnitude and correlation forms.
        :param other_error: Whether comparison errors are negligible or have the same structure as this DEM.
        :param z_name: Elevation column selected from a plain GeoDataFrame.
        :param random_state: Random generator or seed used throughout estimation.
        :param kwargs: Additional options passed to :meth:`xdem.ErrorStructure.estimate`.
        :returns: Fitted error structure with compact grouped and variogram diagnostics.
        """

        return xdem.uncertainty.estimate_error_structure(
            self,
            other_elev,
            stable_terrain=stable_terrain,
            predictors=predictors,
            components=components,
            other_error=other_error,
            z_name=z_name,
            random_state=random_state,
            **kwargs,
        )

    @deprecate(
        details=(
            "Use DEM.estimate_error_structure. "
            "Migration: https://xdem.readthedocs.io/en/stable/uncertainty_migration.html"
        )
    )
    def estimate_uncertainty(
        self,
        other_elev: DEMLike | gpd.GeoDataFrame,
        stable_terrain: Raster | NDArrayb = None,
        approach: Literal["H2022", "R2009", "Basic"] = "H2022",
        precision_of_other: Literal["finer"] | Literal["same"] = "finer",
        spread_estimator: Callable[[NDArrayf], np.floating[Any]] = nmad,
        variogram_estimator: Literal["matheron", "cressie", "genton", "dowd"] = "dowd",
        list_vars: tuple[RasterType | str, ...] = ("slope", "max_curvature"),
        list_vario_models: str | tuple[str, ...] = ("gaussian", "spherical"),
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[RasterType, Callable[[NDArrayf], NDArrayf]]:
        """
        Estimate the uncertainty of DEM.

        Derives either a map of variable errors (based on slope and curvature by default) and a function describing the
        spatial correlation of error (between 0 and 1) with spatial lag (distance between observations).

        Uses stable terrain as an error proxy and assumes a higher or similar-precision DEM is used as reference.

        See Hugonnet et al. (2022) for methodological details.

        :param other_elev: Other elevation dataset to use for estimation, either of finer or similar precision for
            reliable estimates.
        :param stable_terrain: Raster of stable terrain to use as error proxy.
        :param approach: Whether to use Hugonnet et al., 2022 (variable errors, multiple ranges of error correlation),
            or Rolstad et al., 2009 (constant error, multiple ranges of error correlation), or a basic approach
            (constant error, single range of error correlation). Note that all approaches use robust estimators of
            variance (NMAD) and variograms (Dowd) by default, despite not being used in Rolstad et al., 2009. These
            estimators can be tuned separately.
        :param precision_of_other: Whether finer precision (3 times more precise = 95% of estimated error will come from
            this DEM) or similar precision (for instance another acquisition of the same DEM).
        :param spread_estimator: Estimator for statistical dispersion (e.g., standard deviation), defaults to the
            normalized median absolute deviation (NMAD) for robustness.
        :param variogram_estimator: Estimator for empirical variogram, defaults to Dowd for robustness and consistency
            with the NMAD estimator for the spread.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param random_state: Random state or seed number to use for subsampling and optimizer.
        :param list_vars: Variables to use to predict error variability (= elevation heteroscedasticity). Either rasters
            or names of a terrain attributes. Defaults to slope and maximum curvature of the DEM.
        :param list_vario_models: Variogram forms to model the spatial correlation of error. A list translates into
            a sum of models. Uses three by default for a method allowing multiple correlation range, otherwise one.
        :param random_state: State or seed to use for randomization.

        :return: Raster of spread of random errors (1-sigma), Spatial variogram of error correlation.
        """

        # Summarize approach steps
        from xdem.uncertainty import estimate_error_structure

        # Translate the published approach shorthand into explicit error components
        if approach not in {"H2022", "R2009", "Basic"}:
            raise ValueError("approach must be 'H2022', 'R2009' or 'Basic'.")
        models = [list_vario_models] if isinstance(list_vario_models, str) else list(list_vario_models)
        if approach == "Basic":
            models = models[:1]
        variables = (
            {
                var if isinstance(var, str) else f"var{i + 1}": (
                    getattr(terrain, var)(self) if isinstance(var, str) else var
                )
                for i, var in enumerate(list_vars)
            }
            if approach == "H2022"
            else {}
        )
        components = {
            f"component_{i + 1}": {
                "magnitude": "heteroscedastic" if i == 0 and variables else "constant",
                "correlation": model,
            }
            for i, model in enumerate(models)
        }
        if precision_of_other not in {"finer", "same"}:
            raise ValueError("precision_of_other must be 'finer' or 'same'.")
        structure = estimate_error_structure(
            self,
            other_elev,
            stable_terrain=stable_terrain,
            predictors=variables,
            components=components,
            other_error="negligible" if precision_of_other == "finer" else "same",
            z_name=z_name,
            spread_estimator=spread_estimator,
            variogram_estimator=variogram_estimator,
            random_state=random_state,
        )
        return structure.predict_magnitude(variables, like=self), lambda distance: np.asarray(
            structure.predict_correlation(distance)
        )

    def _cast_pointcloud_output(self, pointcloud: Any) -> Any:
        """Preserve EPC behavior for point outputs from DEMs and their accessors."""

        # GeoUtils chooses the native class or dataframe representation and attaches the data column
        output = super()._cast_pointcloud_output(pointcloud)
        if isinstance(output, gu.PointCloud):
            return xdem.EPC(output)

        # Register lazily because raster-to-point conversion may create the first Dask dataframe
        from geoutils._dispatch import is_dask_dataframe

        if is_dask_dataframe(output):
            from xdem.epc.pd_accessor import _register_dask_epc_accessor

            _register_dask_epc_accessor()
        return output

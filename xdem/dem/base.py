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

"""Module of DEMBase class, parent of DEM class and 'dem' accessor."""

from __future__ import annotations

import pathlib
import warnings
import re
from typing import Any, Callable, Literal, overload, TypeVar, Union

import geopandas as gpd
import geoutils as gu
import numpy as np
from affine import Affine
from geoutils import profiler
from geoutils._typing import NDArrayNum
from geoutils.raster import Raster, RasterType
from geoutils.raster.base import RasterBase
from geoutils.multiproc import MultiprocConfig
from geoutils.stats import nmad
from pyproj import CRS
import xarray as xr
from pyproj.crs import CompoundCRS, VerticalCRS

import xdem
from xdem import coreg, terrain
from xdem._misc import copy_doc
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.coreg import Coreg
from xdem.spatialstats import (
    infer_heteroscedasticity_from_stable,
    infer_spatial_correlation_from_stable,
)
from xdem.vcrs import (
    _build_ccrs_from_crs_and_vcrs,
    _to_vcrs,
    _vcrs_from_crs,
    _vcrs_from_user_input,
)

# Input/output is a RasterType (= Raster or RasterAccessor subclass)
DEMType = TypeVar("DEMType", bound="DEMBase")
# For inputs, we also accept a xr.DataArray
DEMLike = Union["DEMBase", xr.DataArray]

class DEMBase(RasterBase):
    """
    This class is non-public and made to be subclassed.

    It is built on top of the RasterBase class. It implements all the functions shared by the DEM class and the
    'dem' Xarray accessor.
    """

    def __init__(self):
        """
        Initialize additional DEM metadata as None, for it to be overridden in sublasses.
        """

        super().__init__()
        self._vcrs: VerticalCRS | Literal["Ellipsoid"] | None = None
        self._data: NDArrayf

    @property
    def vcrs(self) -> VerticalCRS | Literal["Ellipsoid"] | None:
        """
        Vertical coordinate reference system of the DEM.
        """
        return _vcrs_from_crs(self.crs)

    @property
    def _vcrs_name(self) -> str | None:
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
    def _vcrs_grid(self) -> str | None:
        """Human-readable vertical grid description of the DEM."""

        if self.vcrs is None or isinstance(self.vcrs, str):
            return None

        vcrs = CRS(self.vcrs)

        # 1/ Try structured JSON first
        try:
            crs_json = vcrs.to_json_dict()
        except Exception:
            crs_json = None

        if isinstance(crs_json, dict):

            def _find_grid(obj: Any) -> str | None:
                """Recursively search CRS JSON for a geoid/grid filename."""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        key_low = str(key).lower()

                        # Common cases for grid references
                        if key_low in {"grids", "grid", "geoidgrids", "geoid_grid", "filename", "file"}:
                            if isinstance(value, str):
                                return value.split("@")[0]
                            if isinstance(value, list):
                                for item in value:
                                    if isinstance(item, str):
                                        return item.split("@")[0]
                                    if isinstance(item, dict):
                                        for subkey in ("name", "filename", "file", "value"):
                                            subval = item.get(subkey)
                                            if isinstance(subval, str):
                                                return subval.split("@")[0]
                        found = _find_grid(value)
                        if found is not None:
                            return found

                elif isinstance(obj, list):
                    for item in obj:
                        found = _find_grid(item)
                        if found is not None:
                            return found

                return None

            grid_name = _find_grid(crs_json)
            if grid_name is not None:
                return grid_name

        # 2/ Try PROJ string
        try:
            proj4 = vcrs.to_proj4()
        except Exception:
            proj4 = ""

        match = re.search(r"(?:^|\s)\+?geoidgrids=([^\s]+)", proj4)
        if match is not None:
            return match.group(1).split(",")[0].split("@")[0]

        match = re.search(r"(?:^|\s)\+?grids=([^\s]+)", proj4)
        if match is not None:
            return match.group(1).split(",")[0].split("@")[0]

        # 3/ Fallback to CRS name, e.g. "unknown using geoidgrids=us_nga_egm08_25.tif"
        name = vcrs.name or ""

        match = re.search(r"geoidgrids=([^,\s]+)", name)
        if match is not None:
            return match.group(1).split("@")[0]

        match = re.search(r"grids=([^,\s]+)", name)
        if match is not None:
            return match.group(1).split("@")[0]

    def set_vcrs(
        self,
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int,
    ) -> None:
        """
        Set the vertical coordinate reference system of the DEM.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Get vertical CRS and re-set the CRS
        new_vcrs = _vcrs_from_user_input(vcrs_input=new_vcrs)
        new_crs = _build_ccrs_from_crs_and_vcrs(crs=self.crs, vcrs=new_vcrs)
        self.set_crs(new_crs)

    @overload
    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        *,
        inplace: Literal[False] = False,
    ) -> DEMLike: ...

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
    ) -> DEMLike | None: ...

    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        inplace: bool = False,
    ) -> DEMLike | None:
        """
        Convert the DEM to another vertical coordinate reference system.

        :param vcrs: Destination vertical CRS. Either as a name ("WGS84", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param force_source_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `vcrs`.
        :param inplace: Whether to return a new DEM (default) or the same DEM updated in-place.

        :return: DEM with vertical reference transformed, or None.
        """
        # Apply transformation
        new_data, new_crs = _to_vcrs(data=self.data,
                            transform=self.transform,
                            crs=self.crs,
                            dst_vcrs=vcrs,
                            force_source_vcrs=force_source_vcrs)
        # If early exit because no transformation was required
        if new_data is None:
            if inplace:
                return None
            else:
                return self.copy(deep=False)

        # If inplace, update DEM and vcrs
        if inplace:
            self._data = new_data
            self.set_crs(new_crs=new_crs)
            return None
        # Otherwise, return new DEM
        else:
            return self.from_array(
                data=new_data,
                transform=self.transform,
                crs=new_crs,
                nodata=self.nodata,
                area_or_point=self.area_or_point,
                tags=self.tags,
                cast_nodata=False,
            )

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
    def fractal_roughness(self, window_size: int = 13, mp_config: MultiprocConfig | None = None) -> RasterType:

        return terrain.fractal_roughness(self, window_size=window_size, mp_config=mp_config)

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
        **kwargs,
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

        src_dem = self.copy()

        # Check inputs
        if not isinstance(coreg_method, Coreg):
            raise ValueError("Argument `coreg_method` must be an xdem.coreg instance (e.g. xdem.coreg.NuthKaab()).")

        aligned_dem = coreg_method.fit_and_apply(
            reference_elev,
            src_dem,
            inlier_mask=inlier_mask,
            random_state=random_state,
            bias_vars=bias_vars,
            **kwargs,
        )

        return aligned_dem

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

        :return: Uncertainty raster, Variogram of uncertainty correlation.
        """

        # Summarize approach steps
        approach_dict = {
            "H2022": {"heterosc": True, "multi_range": True},
            "R2009": {"heterosc": False, "multi_range": True},
            "Basic": {"heterosc": False, "multi_range": False},
        }

        # Elevation change with the other DEM or elevation point cloud
        if isinstance(other_elev, DEM):
            dh = other_elev.reproject(self, silent=True) - self
        elif isinstance(other_elev, gpd.GeoDataFrame):
            other_elev = other_elev.to_crs(self.crs)
            points = (other_elev.geometry.x.values, other_elev.geometry.y.values)
            dh = other_elev[z_name].values - self.interp_points(points)
            stable_terrain = stable_terrain
        else:
            raise TypeError("Other elevation should be a DEM or elevation point cloud object.")

        # If the precision of the other DEM is the same, divide the dh values by sqrt(2)
        # See Equation 7 and 8 of Hugonnet et al. (2022)
        if precision_of_other == "same":
            dh /= np.sqrt(2)

        # If the approach allows heteroscedasticity, derive a map of errors
        if approach_dict[approach]["heterosc"]:
            # Derive terrain attributes of DEM if string is passed in the list of variables
            list_var_rast = []
            for var in list_vars:
                if isinstance(var, str):
                    list_var_rast.append(getattr(terrain, var)(self))
                else:
                    list_var_rast.append(var)

            # Estimate variable error from these variables
            sig_dh = infer_heteroscedasticity_from_stable(
                dvalues=dh,
                list_var=list_var_rast,
                spread_statistic=spread_estimator,
                stable_mask=stable_terrain,
            )[0]
        # Otherwise, return a constant error raster
        else:
            sig_dh = self.copy(new_array=spread_estimator(dh[stable_terrain]) * np.ones(self.shape))

        # If the approach does not allow multiple ranges of spatial correlation
        if not approach_dict[approach]["multi_range"]:
            if not isinstance(list_vario_models, str) and len(list_vario_models) > 1:
                warnings.warn(
                    "Several variogram models passed but this approach uses a single range,"
                    "keeping only the first model.",
                    category=UserWarning,
                )
                list_vario_models = list_vario_models[0]

        # Otherwise keep all ranges
        corr_sig = infer_spatial_correlation_from_stable(
            dvalues=dh,
            list_models=list(list_vario_models),
            stable_mask=stable_terrain,
            errors=sig_dh,
            estimator=variogram_estimator,
            random_state=random_state,
        )[2]

        return sig_dh, corr_sig

    def to_pointcloud(
        self,
        data_column_name: str = "b1",
        data_band: int = 1,
        auxiliary_data_bands: list[int] | None = None,
        auxiliary_column_names: list[str] | None = None,
        subsample: float | int = 1,
        skip_nodata: bool = True,
        as_array: bool = False,
        random_state: int | np.random.Generator | None = None,
        force_pixel_offset: Literal["center", "ul", "ur", "ll", "lr"] = "ul",
    ) -> NDArrayNum | xdem.EPC:

        pc = super().to_pointcloud(
            data_column_name=data_column_name,
            data_band=data_band,
            auxiliary_data_bands=auxiliary_data_bands,
            auxiliary_column_names=auxiliary_column_names,
            subsample=subsample,
            skip_nodata=skip_nodata,
            as_array=as_array,
            random_state=random_state,
            force_pixel_offset=force_pixel_offset,
        )

        if isinstance(pc, gu.PointCloud):
            return xdem.EPC(pc)
        else:
            return pc
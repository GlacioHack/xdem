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

"""This module defines the DEM class."""
from __future__ import annotations

import pathlib
import warnings
from typing import Any, Callable, Literal, overload

import geopandas as gpd
import numpy as np
import rasterio as rio
from affine import Affine
from geoutils import Raster
from geoutils.raster import Mask, RasterType
from pyproj import CRS
from pyproj.crs import CompoundCRS, VerticalCRS
from skgstat import Variogram

from xdem import coreg, terrain
from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.misc import copy_doc
from xdem.spatialstats import (
    infer_heteroscedasticity_from_stable,
    infer_spatial_correlation_from_stable,
    nmad,
)
from xdem.vcrs import (
    _build_ccrs_from_crs_and_vcrs,
    _grid_from_user_input,
    _parse_vcrs_name_from_product,
    _transform_zz,
    _vcrs_from_crs,
    _vcrs_from_user_input,
)

dem_attrs = ["_vcrs", "_vcrs_name", "_vcrs_grid"]


class DEM(Raster):  # type: ignore
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
        nodata: int | float | None = None,
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
        :param nodata: Nodata value to be used (overwrites the metadata). Default reads from metadata.
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
                    nodata=nodata,
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
            Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int | None
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

    @property
    def vcrs(self) -> VerticalCRS | Literal["Ellipsoid"] | None:
        """Vertical coordinate reference system of the DEM."""

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

    def set_vcrs(
        self,
        new_vcrs: Literal["Ellipsoid"] | Literal["EGM08"] | Literal["EGM96"] | str | pathlib.Path | VerticalCRS | int,
    ) -> None:
        """
        Set the vertical coordinate reference system of the DEM.

        :param new_vcrs: Vertical coordinate reference system either as a name ("Ellipsoid", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data).
        """

        # Get vertical CRS and set it and the grid
        self._vcrs = _vcrs_from_user_input(vcrs_input=new_vcrs)
        self._vcrs_grid = _grid_from_user_input(vcrs_input=new_vcrs)

    @property
    def ccrs(self) -> CompoundCRS | CRS | None:
        """Compound horizontal and vertical coordinate reference system of the DEM."""

        if self.vcrs is not None:
            ccrs = _build_ccrs_from_crs_and_vcrs(crs=self.crs, vcrs=self.vcrs)
            return ccrs
        else:
            return None

    @overload
    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        *,
        inplace: Literal[False] = False,
    ) -> DEM: ...

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
    ) -> DEM | None: ...

    def to_vcrs(
        self,
        vcrs: Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int,
        force_source_vcrs: (
            Literal["Ellipsoid", "EGM08", "EGM96"] | str | pathlib.Path | VerticalCRS | int | None
        ) = None,
        inplace: bool = False,
    ) -> DEM | None:
        """
        Convert the DEM to another vertical coordinate reference system.

        :param vcrs: Destination vertical CRS. Either as a name ("WGS84", "EGM08", "EGM96"),
            an EPSG code or pyproj.crs.VerticalCRS, or a path to a PROJ grid file (https://github.com/OSGeo/PROJ-data)
        :param force_source_vcrs: Force a source vertical CRS (uses metadata by default). Same formats as for `vcrs`.
        :param inplace: Whether to return a new DEM (default) or the same DEM updated in-place.

        :return: DEM with vertical reference transformed, or None.
        """

        if self.vcrs is None and force_source_vcrs is None:
            raise ValueError(
                "The current DEM has no vertical reference, define one with .set_vref() "
                "or by passing `src_vcrs` to perform a conversion."
            )

        # Initial Compound CRS (only exists if vertical CRS is not None, as checked above)
        if force_source_vcrs is not None:
            # Warn if a vertical CRS already existed for that DEM
            if self.vcrs is not None:
                warnings.warn(
                    category=UserWarning,
                    message="Overriding the vertical CRS of the DEM with the one provided in `src_vcrs`.",
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
        xx, yy = self.coords()
        zz_trans = _transform_zz(crs_from=src_ccrs, crs_to=dst_ccrs, xx=xx, yy=yy, zz=zz)
        new_data = zz_trans.astype(self.dtype)  # type: ignore

        # If inplace, update DEM and vcrs
        if inplace:
            self._data = new_data
            self.set_vcrs(new_vcrs=vcrs)
            return None
        # Otherwise, return new DEM
        else:
            return DEM.from_array(
                data=new_data,
                transform=self.transform,
                crs=self.crs,
                nodata=self.nodata,
                area_or_point=self.area_or_point,
                tags=self.tags,
                vcrs=vcrs,
                cast_nodata=False,
            )

    @copy_doc(terrain, remove_dem_res_params=True)
    def slope(self, method: str = "Horn", degrees: bool = True) -> RasterType:
        return terrain.slope(self, method=method, degrees=degrees)

    @copy_doc(terrain, remove_dem_res_params=True)
    def aspect(
        self,
        method: str = "Horn",
        degrees: bool = True,
    ) -> RasterType:

        return terrain.aspect(self, method=method, degrees=degrees)

    @copy_doc(terrain, remove_dem_res_params=True)
    def hillshade(
        self, method: str = "Horn", azimuth: float = 315.0, altitude: float = 45.0, z_factor: float = 1.0
    ) -> RasterType:

        return terrain.hillshade(self, method=method, azimuth=azimuth, altitude=altitude, z_factor=z_factor)

    @copy_doc(terrain, remove_dem_res_params=True)
    def curvature(self) -> RasterType:

        return terrain.curvature(self)

    @copy_doc(terrain, remove_dem_res_params=True)
    def planform_curvature(self) -> RasterType:

        return terrain.planform_curvature(self)

    @copy_doc(terrain, remove_dem_res_params=True)
    def profile_curvature(self) -> RasterType:

        return terrain.profile_curvature(self)

    @copy_doc(terrain, remove_dem_res_params=True)
    def maximum_curvature(self) -> RasterType:

        return terrain.maximum_curvature(self)

    @copy_doc(terrain, remove_dem_res_params=True)
    def topographic_position_index(self, window_size: int = 3) -> RasterType:

        return terrain.topographic_position_index(self, window_size=window_size)

    @copy_doc(terrain, remove_dem_res_params=True)
    def terrain_ruggedness_index(self, method: str = "Riley", window_size: int = 3) -> RasterType:

        return terrain.terrain_ruggedness_index(self, method=method, window_size=window_size)

    @copy_doc(terrain, remove_dem_res_params=True)
    def roughness(self, window_size: int = 3) -> RasterType:

        return terrain.roughness(self, window_size=window_size)

    @copy_doc(terrain, remove_dem_res_params=True)
    def rugosity(self) -> RasterType:

        return terrain.rugosity(self)

    @copy_doc(terrain, remove_dem_res_params=True)
    def fractal_roughness(self, window_size: int = 13) -> RasterType:

        return terrain.fractal_roughness(self, window_size=window_size)

    @copy_doc(terrain, remove_dem_res_params=True)
    def get_terrain_attribute(self, attribute: str | list[str], **kwargs: Any) -> RasterType | list[RasterType]:
        return terrain.get_terrain_attribute(self, attribute=attribute, **kwargs)

    def coregister_3d(
        self,
        reference_elev: DEM | gpd.GeoDataFrame,
        coreg_method: coreg.Coreg = None,
        inlier_mask: Mask | NDArrayb = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] = None,
        **kwargs: Any,
    ) -> DEM:
        """
        Coregister DEM to a reference DEM in three dimensions.

        Any coregistration method or pipeline from xdem.Coreg can be passed. Default is only horizontal and vertical
        shifts of Nuth and Kääb (2011).

        :param reference_elev: Reference elevation, DEM or elevation point cloud, for the alignment.
        :param coreg_method: Coregistration method or pipeline.
        :param inlier_mask: Optional. 2D boolean array or mask of areas to include in the analysis (inliers=True).
        :param bias_vars: Optional, only for some bias correction methods. 2D array or rasters of bias variables used.
        :param kwargs: Keyword arguments passed to Coreg.fit().

        :return: Coregistered DEM.
        """

        if coreg_method is None:
            coreg_method = coreg.NuthKaab()

        coreg_method.fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=self,
            inlier_mask=inlier_mask,
            bias_vars=bias_vars,
            **kwargs,
        )
        return coreg_method.apply(self)  # type: ignore

    def estimate_uncertainty(
        self,
        other_elev: DEM | gpd.GeoDataFrame,
        stable_terrain: Mask | NDArrayb = None,
        approach: Literal["H2022", "R2009", "Basic"] = "H2022",
        precision_of_other: Literal["finer"] | Literal["same"] = "finer",
        spread_estimator: Callable[[NDArrayf], np.floating[Any]] = nmad,
        variogram_estimator: Literal["matheron", "cressie", "genton", "dowd"] = "dowd",
        list_vars: tuple[RasterType | str, ...] = ("slope", "maximum_curvature"),
        list_vario_models: str | tuple[str, ...] = ("gaussian", "spherical"),
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
    ) -> tuple[RasterType, Variogram]:
        """
        Estimate uncertainty of DEM.

        Derives either a map of variable errors (based on slope and curvature by default) and a function describing the
        spatial correlation of error (between 0 and 1) with spatial lag (distance between observations).

        Uses stable terrain as an error proxy and assumes a higher or similar-precision DEM is used as reference.

        See Hugonnet et al. (2022) for methodological details.

        :param other_elev: Other elevation dataset to use for estimation, either of finer or similar precision for
            reliable estimates.
        :param stable_terrain: Mask of stable terrain to use as error proxy.
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
        :param list_vars: Variables to use to predict error variability (= elevation heteroscedasticity). Either rasters
            or names of a terrain attributes. Defaults to slope and maximum curvature of the DEM.
        :param list_vario_models: Variogram forms to model the spatial correlation of error. A list translates into
            a sum of models. Uses three by default for a method allowing multiple correlation range, otherwise one.

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

        # If approach allows heteroscedasticity, derive a map of errors
        if approach_dict[approach]["heterosc"]:
            # Derive terrain attributes of DEM if string are passed in the list of variables
            list_var_rast = []
            for var in list_vars:
                if isinstance(var, str):
                    list_var_rast.append(getattr(terrain, var)(self))
                else:
                    list_var_rast.append(var)

            # Estimate variable error from these variables
            sig_dh = infer_heteroscedasticity_from_stable(
                dvalues=dh, list_var=list_var_rast, spread_statistic=spread_estimator, stable_mask=stable_terrain
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

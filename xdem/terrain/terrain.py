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

"""Terrain attribute core module, for surface fits (slope, aspect) and windowed indexes (TPI, roughness)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sized, overload

import numpy as np
import xarray as xr
from geoutils import profiler
from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array
from geoutils.multiproc import (
    MultiprocConfig,
    map_overlap,
)
from geoutils.raster import Raster, RasterType
from geoutils.raster.referencing import _res

from xdem._misc import import_optional
from xdem._typing import DTypeLike, NDArrayf
from xdem.terrain.freq import _texture_shading_fft
from xdem.terrain.surfit import _get_surface_attributes
from xdem.terrain.window import _get_windowed_indexes

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike

    from xdem.dem.base import DEMLike

# List available attributes
available_attributes = [
    "slope",
    "aspect",
    "hillshade",
    "profile_curvature",
    "tangential_curvature",
    "planform_curvature",
    "flowline_curvature",
    "max_curvature",
    "min_curvature",
    "topographic_position_index",
    "terrain_ruggedness_index",
    "roughness",
    "rugosity",
    "fractal_roughness",
    "texture_shading",
]
# Attributes per category
# 1/ Requiring surface fit
list_requiring_surface_fit = [
    "slope",
    "aspect",
    "hillshade",
    "curvature",
    "profile_curvature",
    "tangential_curvature",
    "planform_curvature",
    "flowline_curvature",
    "max_curvature",
    "min_curvature",
]
# 2/ Requiring windowed index
list_requiring_windowed_index = [
    "terrain_ruggedness_index",
    "topographic_position_index",
    "roughness",
    "rugosity",
]

# 3/ Requiring windowed fractal index
list_requiring_windowed_fractal_index = ["fractal_roughness"]

# 4/ Requiring fractal domain
list_requiring_frequency_domain = ["texture_shading"]

# Helpers for chunked execution
###############################


def _terrain_attribute_array_block(array: NDArrayf, attribute: str, **kwargs: Any) -> NDArrayf:
    """Calculate one terrain attribute on an array block, including its overlap."""

    return _get_terrain_attribute_base(array, attribute=[attribute], **kwargs)[0]


def _terrain_attribute_raster_block(raster: Raster, attribute: str, **kwargs: Any) -> Raster:
    """Calculate one terrain attribute on a georeferenced multiprocessing tile."""

    values = _terrain_attribute_array_block(raster.data, attribute=attribute, **kwargs)
    return raster.copy(new_array=values)


@overload
def get_terrain_attribute(
    dem: NDArrayf,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def get_terrain_attribute(
    dem: NDArrayf,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> list[NDArrayf]: ...


@overload
def get_terrain_attribute(
    dem: DEMLike,
    attribute: list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> list[RasterType]: ...


@overload
def get_terrain_attribute(
    dem: DEMLike,
    attribute: str,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> RasterLike: ...


@profiler.profile("xdem.terrain.get_terrain_attribute", memprof=True)
def get_terrain_attribute(
    dem: NDArrayf | DEMLike,
    attribute: str | list[str],
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | list[NDArrayf] | RasterType | list[RasterType]:
    """

    Derive one or multiple terrain attributes from a DEM.
    The attributes are based on:

    - Slope, aspect, hillshade (first method) from Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918,
    - Slope, aspect, hillshade (second method), and terrain curvatures from Zevenbergen and Thorne (1987),
        http://dx.doi.org/10.1002/esp.3290120107, with curvature expanded in Moore et al. (1991),
    - Curvatures (profile, tangential, planform, flowline, max, min) following the methods outlined
        in Minár et al. (2020), https://doi.org/10.1016/j.earscirev.2020.103414,
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Terrain Ruggedness Index (topography) from Riley et al. (1999),
        http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf.
    - Terrain Ruggedness Index (bathymetry) from Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962.
    - Roughness from Dartnell (2000), thesis referenced in Wilson et al. (2007) above.
    - Rugosity from Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    Aspect and hillshade are derived using the slope, and thus depend on the same method.
    More details on the equations in the functions get_quadric_coefficients() and get_windowed_indexes().

    This function can be run out-of-memory in multiprocessing by passing a Multiproc config argument.

    Attributes:

    * 'slope': The slope in degrees or radians (degs: 0=flat, 90=vertical). Default method: "Horn".
    * 'aspect': The slope aspect in degrees or radians (degs: 0=N, 90=E, 180=S, 270=W).
    * 'hillshade': The shaded slope in relation to its aspect.
    * 'curvature': The second derivative of elevation (the rate of slope change per pixel), multiplied by 100.
    * 'profile_curvature': The curvature of a normal section having a common tangent line with a steepest slope,
        multiplied by 100.
    * 'tangential_curvature': The curvature perpendicular to the profile curvature, multiplied by 100.
    * 'planform_curvature': The curvature of a projected contour line, multiplied by 100.
    * 'flowline_curvature': The curvature of a projected slope line, multiplied by 100.
    * 'max_curvature': The maximal (geometric) or maximum (directional derivative) curvature at a point in any
        direction, multiplied by 100.
    * 'min_curvature': The minimal (geometric) or minimum (directional derivative) curvature at a point in any
        direction, multiplied by 100.
    * 'surface_fit': A quadric surface fit for each individual pixel.
    * 'topographic_position_index': The topographic position index defined by a difference to the average of
        neighbouring pixels.
    * 'terrain_ruggedness_index': The terrain ruggedness index. For topography, defined by the squareroot of squared
        differences to neighbouring pixels. For bathymetry, defined by the mean absolute difference to neighbouring
        pixels. Default method: "Riley" (topography).
    * 'roughness': The roughness, i.e. maximum difference between neighbouring pixels.
    * 'rugosity': The rugosity, i.e. difference between real and planimetric surface area.
    * 'fractal_roughness': The roughness based on a volume box-counting estimate of the fractal dimension.
    * 'texture_shading': Texture shaded relief using fractional Laplacian operator to enhance terrain texture and
        fine-scale topographic features.


    :param dem: Input DEM.
    :param attribute: Terrain attribute(s) to calculate.
    :param resolution: Resolution of the DEM.
    :param degrees: Whether to convert radians to degrees.
    :param hillshade_altitude: Shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: Shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.
    :param slope_method: Deprecated. Use surface_fit instead. Accepts "Horn" or "ZevenbergThorne".
    :param surface_fit: Surface fit method to use for slope, aspect, hillshade and curvatures: "Horn",
        "ZevenbergThorne" or "Florinsky".
    :param curv_method: Method to calculate the curvatures: "geometric" or "directional".
    :param tri_method: Method to calculate the Terrain Ruggedness Index: "Riley" (topography) or "Wilson" (bathymetry).
    :param window_size: Window size for windowed attributes (TPI, TRI, roughness, rugosity).
    :param window_size_fractal: Window size for windowed fractal attributes (fractal roughness).
    :param engine: Engine to use for computing the attributes, windowed and surface fit attributes all support
        "scipy" or "numba".
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.

    :raises ValueError: If the inputs are poorly formatted or are invalid.

    :examples:
        >>> dem = np.repeat(np.arange(3), 3)[::-1].reshape(3, 3)
        >>> dem
        array([[2, 2, 2],
               [1, 1, 1],
               [0, 0, 0]])
        >>> slope, aspect = get_terrain_attribute(dem, ["slope", "aspect"], resolution=1, surface_fit="ZevenbergThorne")
        >>> slope[1, 1]
        np.float32(45.0)
        >>> aspect[1, 1]
        np.float32(180.0)

    :returns: One or multiple arrays of the requested attribute(s)
    """

    # Normalize DataArrays to the shared raster interface without reading their values
    if isinstance(dem, xr.DataArray):
        dem = dem.rst

    # 0/ Deprecating slope method
    if slope_method is not None:
        warnings.warn(
            "'slope_method' is deprecated, use 'surface_fit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        surface_fit = slope_method  # override
        slope_method = None

    # 1/ Input check

    # Check that we're not using Horn for curvatures
    if surface_fit == "Horn":

        curvature_list = [
            "curvature",
            "profile_curvature",
            "tangential_curvature",
            "planform_curvature",
            "flowline_curvature",
            "max_curvature",
            "min_curvature",
        ]

        if isinstance(attribute, str):
            found = attribute in curvature_list
        else:
            found = any(item in curvature_list for item in attribute)

        if found:
            raise ValueError(
                "'Horn' surface fit method cannot be used for to calculate curvatures. "
                "Use 'ZevenbergThorne' or 'Florinsky' instead."
            )

    # Derive pixel spacing from raster metadata when available
    if has_geo_attr(dem, "transform"):
        transform = get_geo_attr(dem, "transform")
        resolution = _res(transform)

    # Validate and format the inputs
    if isinstance(attribute, str):
        attribute = [attribute]

    # If output dtype is None, used that of input DEM
    if out_dtype is None:
        if np.issubdtype(dem.dtype, np.integer):
            out_dtype = np.float32
        else:
            out_dtype = np.dtype(dem.dtype)

    # Get list of attributes for each type
    attributes_requiring_surface_fit = [attr for attr in attribute if attr in list_requiring_surface_fit]

    # Warn if default window size for fractal roughness
    if "fractal_roughness" in attribute:
        if window_size_fractal < 5:
            warnings.warn(
                category=UserWarning,
                stacklevel=2,
                message="Fractal roughness can only be computed on window sizes larger or equal to 5.",
            )
        elif window_size_fractal < 13:
            warnings.warn(
                category=UserWarning,
                stacklevel=2,
                message="Fractal roughness results with window size of less than 13 can be inaccurate.",
            )

    attributes_requiring_resolution = attributes_requiring_surface_fit + (
        ["rugosity"] if "rugosity" in attribute else []
    )
    if len(attributes_requiring_resolution) > 0:
        if resolution is None:
            raise ValueError(
                f"'resolution' must be provided as an argument for attributes: {attributes_requiring_resolution}"
            )

        if not isinstance(resolution, Sized):
            resolution = (float(resolution), float(resolution))  # type: ignore
        if resolution[0] != resolution[1]:
            raise ValueError(
                f"Surface fit and rugosity require the same X and Y resolution ({resolution} was given). "
                f"This was required by: {attributes_requiring_resolution}."
            )
    # If resolution is still None and there was no error, use a placeholder
    if resolution is None:
        resolution = 1
    # If sized, used the first
    elif isinstance(resolution, Sized):
        resolution = resolution[0]

    choices = (
        list_requiring_surface_fit
        + list_requiring_windowed_index
        + list_requiring_windowed_fractal_index
        + list_requiring_frequency_domain
    )
    for attr in attribute:
        if attr not in choices:
            raise ValueError(f"Attribute '{attr}' is not supported. Choices: {choices}")

    # list_slope_methods = ["Horn", "ZevenbergThorne"]
    list_surface_fit = ["Horn", "ZevenbergThorne", "Florinsky"]
    if surface_fit.lower() not in [sm.lower() for sm in list_surface_fit]:
        raise ValueError(f"Surface fit '{surface_fit}' is not supported. Must be one of: {list_surface_fit}")
    list_curv_methods = ["geometric", "directional"]
    if curv_method.lower() not in [cm.lower() for cm in list_curv_methods]:
        raise ValueError(f"Curvature method '{curv_method}' is not supported. Must be one of: {list_curv_methods}")
    list_tri_methods = ["Riley", "Wilson"]
    if tri_method.lower() not in [tm.lower() for tm in list_tri_methods]:
        raise ValueError(f"TRI method '{tri_method}' is not supported. Must be one of: {list_tri_methods}")
    if (hillshade_azimuth < 0.0) or (hillshade_azimuth > 360.0):
        raise ValueError(f"Azimuth must be a value between 0 and 360 degrees (given value: {hillshade_azimuth})")
    if (hillshade_altitude < 0.0) or (hillshade_altitude > 90):
        raise ValueError("Altitude must be a value between 0 and 90 degrees (given value: {altitude})")
    if (hillshade_z_factor < 0.0) or not np.isfinite(hillshade_z_factor):
        raise ValueError(f"z_factor must be a non-negative finite value (given value: {hillshade_z_factor})")

    # Raise warning if CRS is not projected and using a surface fit attribute
    crs = get_geo_attr(dem, "crs") if has_geo_attr(dem, "crs") else None
    if crs is not None and not crs.is_projected and attributes_requiring_surface_fit:
        warnings.warn(
            category=UserWarning,
            message=f"DEM is not in a projected CRS, the following surface fit attributes might be "
            f"wrong: {list_requiring_surface_fit}."
            f"Use DEM.reproject(crs=DEM.get_metric_crs()) to reproject in a projected CRS.",
        )

    # 2/ Select the required neighborhood and execution backend

    # Include the largest requested window so each tile has all neighboring elevations
    window_depth = 0
    if any(attr in list_requiring_windowed_index for attr in attribute):
        window_depth = window_size // 2
    if "fractal_roughness" in attribute:
        window_depth = max(window_depth, window_size_fractal // 2)
    surface_fit_depth = 0
    if attributes_requiring_surface_fit:
        surface_fit_depth = 2 if surface_fit.lower() == "florinsky" else 1
    depth = max(window_depth, surface_fit_depth)

    # Inspect chunk metadata without triggering implicit loading of file-backed rasters
    raster: Any = dem if has_geo_attr(dem, "transform") else None
    dask_backend = raster._chunks is not None if raster is not None else is_dask_array(dem)
    if mp_config is not None and dask_backend:
        raise ValueError("Cannot use multiprocessing and Dask simultaneously. Remove mp_config for Dask inputs.")
    if mp_config is not None and not isinstance(dem, Raster):
        raise TypeError("The DEM must be a Raster to use multiprocessing.")

    # Pass identical numerical options to eager calculations and worker callbacks
    options: dict[str, Any] = {
        "resolution": resolution,
        "degrees": degrees,
        "hillshade_altitude": hillshade_altitude,
        "hillshade_azimuth": hillshade_azimuth,
        "hillshade_z_factor": hillshade_z_factor,
        "surface_fit": surface_fit,
        "curv_method": curv_method,
        "tri_method": tri_method,
        "window_size": window_size,
        "window_size_fractal": window_size_fractal,
        "engine": engine,
        "texture_alpha": texture_alpha,
        "out_dtype": out_dtype,
    }

    # 3/ Calculate attributes and preserve the caller's raster or array interface
    if mp_config is not None:
        outputs = []
        for attr in attribute:
            config = mp_config.copy()
            if len(attribute) > 1:
                outfile = Path(config.outfile)
                config.outfile = str(outfile.with_name(outfile.stem + "_" + attr + outfile.suffix))

            # Texture shading uses a global FFT: independent tiles would change its result
            block_depth = depth
            if attr in list_requiring_frequency_domain:
                config.chunks = dem.shape
                block_depth = 0
            result = map_overlap(
                _terrain_attribute_raster_block, dem, config, attribute=attr, depth=block_depth, **options
            )
            outputs.append(result)
    elif dask_backend:
        import_optional("dask")
        import dask.array as da

        array: Any = raster.data if raster is not None else dem
        has_band_axis = array.ndim == 3 and array.shape[0] == 1
        if has_band_axis:
            array = array[0]
        outputs = []
        for attr in attribute:
            # Defer the global FFT as one task while preserving the source's original chunks
            if attr in list_requiring_frequency_domain:
                result = (
                    array.rechunk(array.shape)
                    .map_blocks(
                        _terrain_attribute_array_block,
                        attribute=attr,
                        dtype=out_dtype,
                        meta=np.empty((0, 0), dtype=out_dtype),
                        **options,
                    )
                    .rechunk(array.chunks)
                )
            else:
                result = da.map_overlap(
                    _terrain_attribute_array_block,
                    array,
                    attribute=attr,
                    depth=depth,
                    boundary="none",
                    trim=True,
                    dtype=out_dtype,
                    meta=np.empty((0, 0), dtype=out_dtype),
                    **options,
                )
            if has_band_axis:
                result = result[None, ...]
            outputs.append(raster.copy(new_array=result) if raster is not None else result)
    else:
        array = raster.data if raster is not None else dem
        has_band_axis = array.ndim == 3 and array.shape[0] == 1
        if has_band_axis:
            array = array[0]
        arrays = _get_terrain_attribute_base(array, attribute=attribute, **options)
        if has_band_axis:
            arrays = [array[None, ...] for array in arrays]
        outputs = [raster.copy(new_array=array) for array in arrays] if raster is not None else arrays

    # Return the sole attribute directly, preserving the requested order for multiple attributes
    return outputs[0] if len(outputs) == 1 else outputs


def _get_terrain_attribute_base(
    dem: NDArrayf,
    attribute: list[str],
    resolution: float,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    window_size_fractal: int = 13,
    engine: Literal["scipy", "numba"] = "scipy",
    texture_alpha: float = 0.8,
    out_dtype: DTypeLike | None = None,
) -> list[NDArrayf]:
    """Calculate terrain attributes by algorithm family and return them in the requested order."""

    # Create list of required for each type
    attributes_requiring_surface_fit = [attr for attr in attribute if attr in list_requiring_surface_fit]
    attributes_requiring_windowed_index = [attr for attr in attribute if attr in list_requiring_windowed_index]
    attributes_requiring_windowed_fractal_index = [
        attr for attr in attribute if attr in list_requiring_windowed_fractal_index
    ]
    attributes_requiring_frequency_domain = [attr for attr in attribute if attr in list_requiring_frequency_domain]

    # Get array of DEM, we need to be able to use NaNs to propagate invalid values in attributes
    if np.issubdtype(dem.dtype, np.integer):
        dem_arr = dem.astype(np.float32)
    else:
        dem_arr = dem
    if np.ma.isMaskedArray(dem):
        dem_arr = dem_arr.filled(np.nan)

    # Process surface attributes
    if len(attributes_requiring_surface_fit) > 0:

        # Keyword arguments
        surface_kwargs = {
            "hillshade_azimuth": hillshade_azimuth,
            "hillshade_altitude": hillshade_altitude,
            "hillshade_z_factor": hillshade_z_factor,
        }

        # Get attributes
        surface_attributes = _get_surface_attributes(
            dem=dem_arr,
            resolution=resolution,
            surface_attributes=attributes_requiring_surface_fit,
            out_dtype=out_dtype,
            surface_fit=surface_fit,
            curv_method=curv_method,
            engine=engine,
            **surface_kwargs,
        )

        # Convert the unit if wanted
        if degrees:
            for attr in ["slope", "aspect"]:
                if attr not in attributes_requiring_surface_fit:
                    continue
                idx_attr = attributes_requiring_surface_fit.index(attr)
                surface_attributes[idx_attr] = np.rad2deg(surface_attributes[idx_attr])

        # Clip values for hillshade
        if "hillshade" in attributes_requiring_surface_fit:
            idx_hs = attributes_requiring_surface_fit.index("hillshade")
            surface_attributes[idx_hs] = np.clip(surface_attributes[idx_hs], 0, 255)

        # Convert to list in-place to save memory
        surface_attributes = [surface_attributes[i] for i in range(surface_attributes.shape[0])]  # type: ignore
    else:
        surface_attributes = []  # type: ignore

    # Process windowed attributes
    if len(attributes_requiring_windowed_index) > 0:

        windowed_indexes = _get_windowed_indexes(
            dem=dem_arr,
            windowed_indexes=attributes_requiring_windowed_index,
            window_size=window_size,
            resolution=resolution,
            out_dtype=out_dtype,
            tri_method=tri_method,
            engine=engine,
        )
        windowed_indexes = [windowed_indexes[i] for i in range(windowed_indexes.shape[0])]  # type: ignore
    else:
        windowed_indexes = []  # type: ignore

    # Process windowed fractal attributes
    if len(attributes_requiring_windowed_fractal_index) > 0:

        windowed_fractal_indexes = _get_windowed_indexes(
            dem=dem_arr,
            windowed_indexes=attributes_requiring_windowed_fractal_index,
            window_size=window_size_fractal,
            resolution=resolution,
            out_dtype=out_dtype,
            tri_method=tri_method,
            engine=engine,
        )
        windowed_fractal_indexes = [
            windowed_fractal_indexes[i] for i in range(windowed_fractal_indexes.shape[0])
        ]  # type: ignore
    else:
        windowed_fractal_indexes = []  # type: ignore

    # Process frequency domain attributes
    if len(attributes_requiring_frequency_domain) > 0:
        frequency_attributes = []
        for attr in attributes_requiring_frequency_domain:
            if attr == "texture_shading":
                # Use texture_alpha parameter for texture shading
                result = _texture_shading_fft(dem_arr, alpha=texture_alpha)
                frequency_attributes.append(result.astype(out_dtype))
    else:
        frequency_attributes = []  # type: ignore

    # Convert 3D array output to list of 2D arrays
    output_attributes = surface_attributes + windowed_indexes + windowed_fractal_indexes + frequency_attributes

    calculated_attributes = (
        attributes_requiring_surface_fit
        + attributes_requiring_windowed_index
        + attributes_requiring_windowed_fractal_index
        + attributes_requiring_frequency_domain
    )
    output_attributes = [output_attributes[calculated_attributes.index(attr)] for attr in attribute]

    return output_attributes


@overload
def slope(
    dem: NDArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def slope(
    dem: DEMLike,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> Raster: ...


@profiler.profile("xdem.terrain.slope", memprof=True)
def slope(
    dem: NDArrayf | DEMLike,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | Raster:
    """
    Generate a slope map for a DEM, returned in degrees by default.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to generate a slope map for.
    :param method: Deprecated; use `surface_fit` instead.
    :param surface_fit: Surface fit method to use for slope: "Horn", "ZevenbergThorne" or "Florinsky".
    :param degrees: Whether to use degrees or radians (False means radians).
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :examples:
        >>> dem = np.repeat(np.arange(3), 3).reshape(3, 3)
        >>> dem
        array([[0, 0, 0],
               [1, 1, 1],
               [2, 2, 2]])
        >>> slope(dem, surface_fit="ZevenbergThorne", resolution=1, degrees=True)[1, 1] # Slope in degrees
        np.float32(45.0)

    :returns: A slope map of the same shape as 'dem' in degrees or radians.
    """

    # Deprecating slope method
    if method is not None:
        warnings.warn(
            "'method' is deprecated, use 'surface_fit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        surface_fit = method  # override
        method = None

    return get_terrain_attribute(
        dem,
        attribute="slope",
        surface_fit=surface_fit,
        resolution=resolution,
        degrees=degrees,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def aspect(
    dem: NDArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def aspect(
    dem: DEMLike,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.aspect", memprof=True)
def aspect(
    dem: NDArrayf | DEMLike,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    degrees: bool = True,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | Raster:
    """
    Calculate the aspect of each cell in a DEM, returned in degrees by default. The aspect of flat slopes is 180° by
    default (as in GDAL).

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918 and on Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107.

    0=N, 90=E, 180=S, 270=W.

    Note that aspect, representing only the orientation of the slope, is independent of the grid resolution.

    :param dem: The DEM to calculate the aspect from.
    :param method: Deprecated; use `surface_fit` instead.
    :param surface_fit: Surface fit method to use for aspect: "Horn", "ZevenbergThorne" or "Florinsky".
    :param degrees: Whether to use degrees or radians (False means radians).
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :examples:
        >>> dem = np.tile(np.arange(3), (3,1))
        >>> dem
        array([[0, 1, 2],
               [0, 1, 2],
               [0, 1, 2]])
        >>> aspect(dem, surface_fit="ZevenbergThorne", degrees=True)[1, 1]
        np.float32(270.0)
        >>> dem2 = np.repeat(np.arange(3), 3)[::-1].reshape(3, 3)
        >>> dem2
        array([[2, 2, 2],
               [1, 1, 1],
               [0, 0, 0]])
        >>> aspect(dem2, surface_fit="ZevenbergThorne", degrees=True)[1, 1]
        np.float32(180.0)

    """

    # Deprecating slope method
    if method is not None:
        warnings.warn(
            "'method' is deprecated, use 'surface_fit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        surface_fit = method  # override
        method = None

    return get_terrain_attribute(
        dem,
        attribute="aspect",
        surface_fit=surface_fit,
        resolution=1.0,
        degrees=degrees,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def hillshade(
    dem: NDArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def hillshade(
    dem: DEMLike,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.hillshade", memprof=True)
def hillshade(
    dem: NDArrayf,
    method: Literal["Horn", "ZevenbergThorne"] = None,
    surface_fit: Literal["Horn", "ZevenbergThorne", "Florinsky"] = "Florinsky",
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Generate a hillshade from the given DEM. The value 0 is used for nodata, and 1 to 255 for hillshading.

    Based on Horn (1981), http://dx.doi.org/10.1109/PROC.1981.11918.

    :param dem: The input DEM to calculate the hillshade from.
    :param method: Deprecated; use `surface_fit` instead.
    :param surface_fit: Surface fit method to use for slope and aspect: "Horn", "ZevenbergThorne" or "Florinsky".
    :param azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param z_factor: Vertical exaggeration factor.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises AssertionError: If the given DEM is not a 2D array.
    :raises ValueError: If invalid argument types or ranges were given.

    :returns: A hillshade with the dtype "float32" with value ranges of 0-255.
    """

    # Deprecating slope method
    if method is not None:
        warnings.warn(
            "'method' is deprecated, use 'surface_fit' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        surface_fit = method  # override
        method = None

    return get_terrain_attribute(
        dem,
        attribute="hillshade",
        resolution=resolution,
        # slope_method=method,
        surface_fit=surface_fit,
        hillshade_azimuth=azimuth,
        hillshade_altitude=altitude,
        hillshade_z_factor=z_factor,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.curvature", memprof=True)
def curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    THIS FUNCTION IS DEPRECATED - REFER TO DOCS FOR SPECIFIC CURVATURE RECOMMENDATIONS

    Calculate the terrain curvature (second derivative of elevation) in m-1 multiplied by 100.

    Based on Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    Information:
       * Curvature is positive on convex surfaces and negative on concave surfaces.
       * Per convention, it is multiplied by 100 to obtain more reasonable numbers. \
               For analytic purposes, dividing by 100 is needed.
       * The unit is the second derivative of elevation (times 100), so '100m²/m' or '100/m' (assuming the unit is m).
       * It is created from the second derivative of a quadric surface fit for each pixel. \
               See xdem.terrain.get_quadric_coefficients() for more information.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The curvature array of the DEM.
    """

    # Warn that this approach is deprecated and will be removed in a future version
    warnings.warn(
        "The curvature attribute is deprecated, refer to docs for specific curvature functions.",
        DeprecationWarning,
        stacklevel=2,
    )

    return get_terrain_attribute(
        dem=dem,
        attribute="curvature",
        surface_fit=surface_fit,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def profile_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def profile_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.profile_curvature", memprof=True)
def profile_curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """

    Calculates profile curvature in units of m-1 multiplied by 100. Defined as the curvature of a normal section of
    slope that is tangential to the slope line (steepest slope). Also known as vertical curvature.

    Geometric (default) method follows Krcho (1973) and Evans (1979) as outlined in in Minár et al. (2020),
    https://doi.org/10.1016/j.earscirev.2020.103414

    Directional derivative method follows Zevenbergen and Thorne (1987), http://dx.doi.org/10.1002/esp.3290120107.

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param curv_method: The method to use to calculate the curvature, either 'geometric' or 'directional'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> profile_curvature(dem, surface_fit="ZevenbergThorne", curv_method="directional", resolution=1.0)[1, 1]
        np.float32(-100.0)
        >>> dem = np.array([[1, 2, 3],
        ...                 [1, 2, 3],
        ...                 [1, 2, 3]], dtype="float32")
        >>> profile_curvature(dem, surface_fit="ZevenbergThorne", curv_method="directional", resolution=1.0)[1, 1]
        np.float32(-0.0)

    :returns: The profile curvature array of the DEM.
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="profile_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def tangential_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def tangential_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.tangential_curvature", memprof=True)
def tangential_curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """

    Calculates tangential curvature in units of m-1 multiplied by 100. Defined as the curvature of a normal section of
    slope that is tangential to the contour line. Sometimes known as the horizontal curvature, although this
    terminology has been shared with planform curvature.

    Geometric (default) tangential curvature (normal contour curvature) follows Krcho, 1983 in Minár et al. (2020),
    https://doi.org/10.1016/j.earscirev.2020.103414

    Directional derivative tangential curvature follows 'plan curvature' of Zevenbergen and Thorne (1987),
    http://dx.doi.org/10.1002/esp.3290120107

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The tangential curvature array of the DEM.

      :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> tangential_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-0.0)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> tangential_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-221.88008)
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="tangential_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def planform_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def planform_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.planform_curvature", memprof=True)
def planform_curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """

    Calculates planform (or plan) curvature in units of m-1 multiplied by 100., defined as the curvature of a
    projection of the contour line onto a horizontal plane. Sometimes known as the horizontal curvature, although this
    terminology has been shared with tangential curvature.

    Geometric and directional derivatives are identical, following method based on Sobolevsky (1932) in
    Minár et al. (2020), https://doi.org/10.1016/j.earscirev.2020.103414

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param curv_method: The method to use to calculate the curvature, either 'geometric' or 'directional'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> planform_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-0.0)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> planform_curvature(dem, surface_fit="ZevenbergThorne", curv_method="directional", resolution=1.0)[1, 1]
        np.float32(-266.66666)

    :returns: The planform curvature array of the DEM.
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="planform_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def flowline_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def flowline_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.flowline_curvature", memprof=True)
def flowline_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf:
    """
    Calculates flow line curvature in units of m-1 multiplied by 100. Defined as the curvature of a projection of the
    slope line onto a horizontal plane. Sometimes known as the rotor or steam line curvature.

    Geometric (default) flowline curvature follows the contour torsion described by Minár et al. (2020),
    https://doi.org/10.1016/j.earscirev.2020.103414

    Directional derivative flowline curvature follows Shary, 1991 in Minár et al. (2020),
    https://doi.org/10.1016/j.earscirev.2020.103414

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param curv_method: The method to use to calculate the curvature, either 'geometric' or 'directional'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The flowline curvature array of the DEM.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> flowline_curvature(dem, surface_fit="ZevenbergThorne", curv_method="directional", resolution=1.0)[1, 1]
        np.float32(-0.0)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> flowline_curvature(dem, surface_fit="ZevenbergThorne", curv_method="directional", resolution=1.0)[1, 1]
        np.float32(0.0)
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="flowline_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def max_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def max_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.max_curvature", memprof=True)
def max_curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculate the maximal (geometric) or maximum (directional derivative) curvature in units of m-1 multiplied by 100.
    Defined as curvature of the normal section of slope with the greatest curvature value.

    Geometric (default) maximal curvature is calculated following Shary (1995, https://doi.org/10.1007/BF02084608)
    and is equal to the minimal curvature of Euler (1760).

    Directional derivative maximum curvature is the minimum second derivative following Wood (1996),
    https://lra.le.ac.uk/handle/2381/34503

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param curv_method: The method to use to calculate the curvature, either 'geometric' or 'directional'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The maximal or maximum curvature array of the DEM.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> max_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(0.0)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> max_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-17.067698)
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="max_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def min_curvature(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def min_curvature(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.min_curvature", memprof=True)
def min_curvature(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    surface_fit: Literal["ZevenbergThorne", "Florinsky"] = "Florinsky",
    curv_method: Literal["geometric", "directional"] = "geometric",
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculate the minimal (geometric) or minimum (directional derivative) curvature in units of m-1 multiplied by 100.
    Defined as curvature of the normal section of slope with the smallest curvature value.

    Geometric (default) minimal curvature is calculated following Shary (1995, https://doi.org/10.1007/BF02084608)
    and is equal to the maximal curvature of Euler (1760).

    Directional derivative minimum curvature is the maximum second derivative following Wood (1996),
    https://lra.le.ac.uk/handle/2381/34503

    :param dem: The DEM to calculate the curvature from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param surface_fit: The surface fit to use, either 'ZevenbergThorne' or 'Florinsky'.
    :param curv_method: The method to use to calculate the curvature, either 'geometric' or 'directional'.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :returns: The minimal or minimum curvature array of the DEM.

    :examples:
        >>> dem = np.array([[1, 2, 4],
        ...                 [1, 2, 4],
        ...                 [1, 2, 4]], dtype="float32")
        >>> min_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-17.067698)
        >>> dem = np.array([[1, 4, 8],
        ...                 [1, 2, 4],
        ...                 [1, 4, 8]], dtype="float32")
        >>> min_curvature(dem, surface_fit="ZevenbergThorne", resolution=1.0)[1, 1]
        np.float32(-221.88008)
    """

    return get_terrain_attribute(
        dem=dem,
        attribute="min_curvature",
        surface_fit=surface_fit,
        curv_method=curv_method,
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def topographic_position_index(
    dem: NDArrayf,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def topographic_position_index(
    dem: DEMLike,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.topographic_position_index", memprof=True)
def topographic_position_index(
    dem: NDArrayf | DEMLike,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculates the Topographic Position Index, the difference to the average of neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based on: Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.

    :param dem: The DEM to calculate the topographic position index from.
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        np.float32(1.0)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> topographic_position_index(dem)[1, 1]
        np.float32(0.0)

    :returns: The topographic position index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="topographic_position_index",
        window_size=window_size,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def terrain_ruggedness_index(
    dem: NDArrayf,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def terrain_ruggedness_index(
    dem: DEMLike,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.terrain_ruggedness_index", memprof=True)
def terrain_ruggedness_index(
    dem: NDArrayf | DEMLike,
    method: Literal["Riley", "Wilson"] = "Riley",
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculates the Terrain Ruggedness Index, the cumulated differences to neighbouring pixels. Output is in the
    unit of the DEM (typically meters).

    Based either on:

    * Riley et al. (1999), http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf that derives
        the squareroot of squared differences to neighbouring pixels, preferred for topography.
    * Wilson et al. (2007), http://dx.doi.org/10.1080/01490410701295962 that derives the mean absolute difference to
        neighbouring pixels, preferred for bathymetry.

    :param dem: The DEM to calculate the terrain ruggedness index from.
    :param method: The algorithm used ("Riley" for topography or "Wilson" for bathymetry).
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        np.float32(2.828427)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> terrain_ruggedness_index(dem)[1, 1]
        np.float32(0.0)

    :returns: The terrain ruggedness index array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="terrain_ruggedness_index",
        tri_method=method,
        window_size=window_size,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def roughness(
    dem: NDArrayf,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def roughness(
    dem: DEMLike,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.roughness", memprof=True)
def roughness(
    dem: NDArrayf | DEMLike,
    window_size: int = 3,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculates the roughness, the maximum difference between neighbouring pixels, for any window size. Output is in the
    unit of the DEM (typically meters).

    Based on: Dartnell (2000), https://environment.sfsu.edu/node/11292.

    :param dem: The DEM to calculate the roughness from.
    :param window_size: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        np.float32(1.0)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> roughness(dem)[1, 1]
        np.float32(0.0)

    :returns: The roughness array of the DEM (unit of the DEM).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="roughness",
        window_size=window_size,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def rugosity(
    dem: NDArrayf,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def rugosity(
    dem: DEMLike,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.rugosity", memprof=True)
def rugosity(
    dem: NDArrayf | DEMLike,
    resolution: float | tuple[float, float] | None = None,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculates the rugosity, the ratio between real area and planimetric area. Only available for a 3x3 window. The
    output is unitless.

    Based on: Jenness (2004), https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.

    :param dem: The DEM to calculate the rugosity from.
    :param resolution: The X/Y resolution of the DEM, only if passed as an array.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 2, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> rugosity(dem, resolution=1.)[1, 1]
        np.float32(1.4142131)
        >>> dem = np.array([[1, 1, 1],
        ...                 [1, 1, 1],
        ...                 [1, 1, 1]], dtype="float32")
        >>> np.round(rugosity(dem, resolution=1.)[1, 1], 5)
        np.float32(1.0)

    :returns: The rugosity array of the DEM (unitless).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="rugosity",
        resolution=resolution,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def fractal_roughness(
    dem: NDArrayf,
    window_size_fractal: int = 13,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf: ...


@overload
def fractal_roughness(
    dem: DEMLike,
    window_size_fractal: int = 13,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> RasterLike: ...


@profiler.profile("xdem.terrain.fractal_roughness", memprof=True)
def fractal_roughness(
    dem: NDArrayf | DEMLike,
    window_size_fractal: int = 13,
    mp_config: MultiprocConfig | None = None,
    engine: Literal["scipy", "numba"] = "scipy",
) -> NDArrayf | RasterLike:
    """
    Calculates the fractal roughness, the local 3D fractal dimension. Can only be computed on window sizes larger or
    equal to 5x5, defaults to 13x13. Output unit is a fractal dimension between 1 and 3.

    Based on: Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    :param dem: The DEM to calculate the roughness from.
    :param window_size_fractal: The size of the window for deriving the metric.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None.
    :param engine: Engine to use for computing the attribute, "scipy" or "numba".

    :raises ValueError: If the inputs are poorly formatted.

    :examples:
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[1, 1] = 6.5
        >>> np.round(fractal_roughness(dem)[6, 6], 3) # The fractal dimension of a line is 1
        np.float32(1.0)
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, 1] = 13
        >>> np.round(fractal_roughness(dem)[6, 6], 3) # The fractal dimension of plane is 2
        np.float32(2.0)
        >>> dem = np.zeros((13, 13), dtype='float32')
        >>> dem[:, :6] = 13
        >>> np.round(fractal_roughness(dem)[6, 6], 3) # The fractal dimension of cube is 3
        np.float32(3.0)

    :returns: The fractal roughness array of the DEM in fractal dimension (between 1 and 3).
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="fractal_roughness",
        window_size_fractal=window_size_fractal,
        mp_config=mp_config,
        engine=engine,
    )


@overload
def texture_shading(
    dem: NDArrayf,
    alpha: float = 0.8,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf: ...


@overload
def texture_shading(
    dem: DEMLike,
    alpha: float = 0.8,
    mp_config: MultiprocConfig | None = None,
) -> RasterLike: ...


@profiler.profile("xdem.terrain.texture_shading", memprof=True)
def texture_shading(
    dem: NDArrayf | DEMLike,
    alpha: float = 0.8,
    mp_config: MultiprocConfig | None = None,
) -> NDArrayf | RasterLike:
    """
    Generate a texture shaded relief map using fractional Laplacian operator.

    This technique, developed by Leland Brown, applies a fractional Laplacian operator
    in the frequency domain to enhance terrain texture and fine-scale topographic features.
    It's particularly effective for visualizing subtle terrain variations that may not
    be apparent in traditional hillshading.

    The fractional Laplacian operator is controlled by the alpha parameter:
    - alpha = 0: Preserves all frequencies with mean removed (~zero-centered DEM)
    - alpha = 1: Standard Laplacian operator (edge detection)
    - alpha = 2: Enhanced high-frequency features

    Based on: Brown, L. (2010). Texture Shading: A New Technique for Depicting Terrain Relief.
    Workshop on Mountain Cartography, Banff, Canada.

    Also described in: Allmendinger and Karabinos (2023), https://doi.org/10.1130/GES02531.1

    Adapted from the Python implementation at https://github.com/fasiha/texshade-py

    :param dem: Input DEM array or Raster object
    :param alpha: Fractional exponent for Laplacian operator (0-2, default 0.8).
        Higher values enhance fine details, lower values provide smoother results.
    :param mp_config: Multiprocessing configuration, run the function in multiprocessing if not None

    :raises ValueError: If alpha is not between 0 and 2

    :examples:
        >>> import numpy as np
        >>> # Create a simple test DEM with a ridge
        >>> dem = np.zeros((50, 50))
        >>> dem[20:30, :] = np.sin(np.linspace(0, np.pi, 50)) * 10
        >>> textured = texture_shading(dem, alpha=0.8)
        >>> textured.shape == dem.shape
        True
        >>> # Flat surface returns no texture (all 0)
        >>> dem_flat = np.ones((32, 32), dtype=float)
        >>> dem_flat_ts = texture_shading(dem_flat, alpha=0.8)
        >>> np.allclose(dem_flat_ts, 0.0)
        True
        >>> # Higher alpha enhances fine details
        >>> textured_enhanced = texture_shading(dem, alpha=1.5)

    :returns: Texture shaded array with same shape as input DEM
    """
    return get_terrain_attribute(
        dem=dem,
        attribute="texture_shading",
        texture_alpha=alpha,
        mp_config=mp_config,
    )

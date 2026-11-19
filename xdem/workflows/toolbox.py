# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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
"""
Workflow class
"""

import csv
import logging
import math
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import geoutils as gu
import numpy as np
import pyogrio.errors
import rasterio.errors
from geoutils import Raster
from geoutils.raster import RasterType
from pyproj import CRS
from rasterio.warp import calculate_default_transform

import xdem
from xdem import DEM
from xdem._misc import import_optional
from xdem.coreg.base import InputCoregDict, OutputCoregDict
from xdem.dem.xr_accessor import DEMAccessor
from xdem.examples import _FILEPATHS_ALL
from xdem.vcrs import vertical_unit_symbol
from xdem.workflows.schemas import validate_configuration

# Inheritance of optional dependencies
try:
    from yaml.dumper import SafeDumper  # type: ignore

    _HAS_YAML = True
except ImportError:
    SafeDumper = object
    _HAS_YAML = False

try:
    import plutoprint

    _HAS_PLUTOPRINT = True
except ImportError:
    _HAS_PLUTOPRINT = False


def return_info(dem: RasterType) -> dict[str, Any]:
    """
    xxxx

    :param dem:
    :return
    """
    return {
        "Driver": dem.driver,
        "Filename": dem.name,
        "Number of band": dem.bands,
        "Data types": str(dem.dtype),
        "Nodata Value": (float(dem.nodata) if dem.nodata is not None else None),
        "Pixel interpretation": dem.area_or_point,
        "Pixel size": dem.res,
        "Width": dem.width,
        "Height": dem.height,
        "Transform": dem.transform,
        "Bounds": dem.bounds,
    }


def floats_process(dict_with_floats: Dict[str, Any] | InputCoregDict | OutputCoregDict | Any) -> Dict[str, Any]:  # type: ignore
    """
    Allows rounding all floats present in a dictionary to two decimal places
    :param dict_with_floats: Dictionary with float
    :return: Dictionary with floats
    """
    if isinstance(dict_with_floats, dict):
        return {k: floats_process(v) for k, v in dict_with_floats.items()}
    elif isinstance(dict_with_floats, list):
        return [floats_process(elem) for elem in dict_with_floats]  # type: ignore
    elif isinstance(dict_with_floats, tuple):
        return tuple(floats_process(elem) for elem in dict_with_floats)  # type: ignore
    elif isinstance(dict_with_floats, (float, np.floating)):
        return round(float(dict_with_floats), 2)  # type: ignore
    else:
        return dict_with_floats


def generate_plot(
    dem: RasterType,
    title: str,
    file_path: str,
    dem_right: RasterType = None,
    title_dem_right: str = None,
    **kwargs: Any,
) -> None:
    """
    Generate plot from a DEM.

    :param dem: Input digital elevation model (left)
    :param title: Title of dem plot (left)
    :param filename: Filename of figure.
    :param dem_right: Input digital elevation model (right)
    :param title_dem_right: Title of dem_right plot (right)
    :param mask_path: Path to mask file.
    :return: None
    """
    import_optional("matplotlib")
    import matplotlib.pyplot as plt

    size_font = 6
    plt.rc("font", size=size_font)
    plt.rc("axes", titlesize=size_font)
    plt.rc("axes", labelsize=size_font)
    plt.rc("xtick", labelsize=size_font)
    plt.rc("ytick", labelsize=size_font)
    plt.rc("legend", fontsize=size_font)
    plt.rc("figure", titlesize=size_font)

    # Apply default cmap if not given in inputs
    if "cmap" in kwargs:
        cmap = plt.get_cmap(name=kwargs["cmap"])
    else:
        cmap = plt.get_cmap(name="terrain")
    cmap.set_bad(color="k", alpha=None)
    kwargs["cmap"] = cmap

    # Add colormap
    vunit = vertical_unit_symbol(dem.crs)
    kwargs.setdefault(
        "cbar_kwargs", {"label": f"Elevation differences ({vunit})" if vunit else "Elevation differences"}
    )
    # Force figsize with the good ratio to prevent larger right axe if not filled
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6.4, 2.4])

    # Add the first image to the figure (left position)
    if isinstance(dem, DEMAccessor):
        dem = dem._obj

    dem.plot(ax=ax1)  # , **kwargs)
    ax1.set_title(title)

    # If exists, add the second image to the figure
    if dem_right is not None:
        if isinstance(dem_right, DEMAccessor):
            dem_right = dem_right._obj
        dem_right.plot(ax=ax2)  # , **kwargs)
        ax2.set_title(title_dem_right)
    else:
        ax2.set_axis_off()

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_plot_with_profiles(
    dem: RasterType,
    title: str,
    file_path: str,
    **kwargs: Any,
) -> None:
    """
    Generate plot from a DEM with profiles.

    :param dem: Input digital elevation model
    :param title: Title of dem plot
    :param filename: Filename of figure
    """
    print("generate")
    import_optional("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Add the first image to the figure (left position)

    logging.info(dem)
    vunit = vertical_unit_symbol(dem.crs)
    unit_label = f" ({vunit})" if vunit else ""
    logging.info(unit_label)

    # Raster data
    data = dem.data
    ny, nx = data.shape
    logging.info(nx)
    logging.info(ny)

    # Initial min/max for mean profiles
    profile_cols = np.nanmean(data, axis=0)
    profile_cols_stats = [profile_cols.min(), profile_cols.max()]
    profile_rows = np.nanmean(data, axis=1)

    profile_rows_stats = [profile_rows.min(), profile_rows.max()]

    # Keep profiles with at least more than 50% of valid values
    nb_valid_rows = np.sum(~np.isnan(data), axis=1)  # todo
    nb_valid_cols = np.sum(~np.isnan(data), axis=0)
    # min_valid_rows, min_valid_cols = data.shape[1] / 2.0, data.shape[0] / 2.0

    # Update profiles values according to valid values
    profile_rows = np.ma.masked_where(
        nb_valid_rows < data.shape[1] / 2,
        profile_rows,
    )
    profile_cols = np.ma.masked_where(
        nb_valid_cols < data.shape[0] / 2,
        profile_cols,
    )

    # Force figsize with the same size as generate_plot function
    size_font = 6
    plt.rc("font", size=size_font)
    plt.rc("axes", titlesize=size_font)
    plt.rc("axes", labelsize=size_font)
    plt.rc("xtick", labelsize=size_font)
    plt.rc("ytick", labelsize=size_font)
    plt.rc("legend", fontsize=size_font)
    plt.rc("figure", titlesize=size_font)

    gs = GridSpec(2, 3, width_ratios=[1.2, 4, 0.3], height_ratios=[4, 1.2])  # 1 for the colobar

    fig = plt.figure()
    ax_left = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])
    ax_bottom = fig.add_subplot(gs[1, 1])

    # Apply default cmap if not given in inputs
    if "cmap" in kwargs:
        cmap = plt.get_cmap(name=kwargs["cmap"])
    else:
        cmap = plt.get_cmap(name="terrain")
    cmap.set_bad(color="k", alpha=None)
    kwargs["cmap"] = cmap

    # Plot DEM with colorbar
    im = ax_map.imshow(data, aspect="auto", **kwargs)
    ax_map.text(0.5, 1.12, title, transform=ax_map.transAxes, ha="center", va="top")
    fig.colorbar(im, cax=cax).set_label(f"Elevation differences{unit_label}")

    # Lines profiles
    y = np.arange(ny)
    ax_left.plot(profile_rows, y, color="black")
    ax_left.set_ylim(ax_map.get_ylim())
    ax_left.invert_xaxis()
    ax_left.yaxis.tick_left()
    ax_left.yaxis.set_label_position("left")
    ax_left.set_xlabel(
        f"Mean along lines{unit_label}\n"
        f"Min: {np.round(profile_rows_stats[0], 2)} / Max: {np.round(profile_rows_stats[1], 2)}"
    )

    # Columns profiles
    x = np.arange(nx)
    ax_bottom.plot(x, profile_cols, color="black")
    ax_bottom.set_xlim(ax_map.get_xlim())
    ax_bottom.yaxis.tick_left()
    ax_bottom.xaxis.set_label_position("bottom")
    ax_bottom.set_xlabel(
        f"Mean along columns{unit_label}\n"
        f"Min: {np.round(profile_cols_stats[0], 2)} / Max: {np.round(profile_cols_stats[1], 2)}"
    )

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_terrain_attributes(
    dem: RasterType,
    list_attributes: list[str],
    outputs_folder,
    config_attributes,
    proj_crs,
    proj_crs_value,
    export_tif: bool = False,
) -> None:
    """
    Generates an image png containing the plots of the terrain attributes requested by the user.

    :param dem:
    :param export_tif: export tif for each terrain attributes
    :return: None
    """

    proj_crs = None

    if dem.crs.is_geographic:
        if proj_crs:
            proj_crs = dem.get_metric_crs()

            logging.info(f"Reprojection in default projected CRS ({proj_crs})")

        elif not proj_crs_value:
            warnings.warn(
                "As the input dem is not in a projected CRS, the following surface fit attributes might be wrong."
                "Please use a projected CRS or let it empty to reproject in default projected CRS.",
                UserWarning,
            )

        else:
            proj_crs = proj_crs_value
            logging.info(f"Reprojection with crs = {proj_crs}")

            if CRS.from_user_input(proj_crs_value).is_geographic:
                warnings.warn(
                    'As the input dem is not in a projected CRS and the "reproject/crs" either,'
                    "the following surface fit attributes might be wrong.",
                    UserWarning,
                )

    if proj_crs is not None:
        # Terrain derivatives require square pixels; request GDAL's suggested spacing explicitly
        target_transform, _, _ = calculate_default_transform(dem.crs, proj_crs, dem.width, dem.height, *dem.bounds)
        dem = dem.reproject(crs=proj_crs, res=abs(target_transform.a))
        if export_tif:
            dem.to_file(outputs_folder / "rasters" / "elev_reprojected.tif")
    elif dem.res[0] != dem.res[1]:
        # Retain the chosen CRS while regularizing a rectangular grid at its finer pixel spacing
        dem = dem.reproject(res=min(dem.res))

    attribute_extra = {}
    from_str_to_fun = {
        "slope": lambda: dem.slope(**attribute_extra),
        "aspect": lambda: dem.aspect(**attribute_extra),
        "hillshade": lambda: dem.hillshade(**attribute_extra),
        "profile_curvature": lambda: dem.profile_curvature(**attribute_extra),
        "tangential_curvature": lambda: dem.tangential_curvature(**attribute_extra),
        "planform_curvature": lambda: dem.planform_curvature(**attribute_extra),
        "flowline_curvature": lambda: dem.flowline_curvature(**attribute_extra),
        "max_curvature": lambda: dem.max_curvature(**attribute_extra),
        "min_curvature": lambda: dem.min_curvature(**attribute_extra),
        "topographic_position_index": lambda: dem.topographic_position_index(**attribute_extra),
        "terrain_ruggedness_index": lambda: dem.terrain_ruggedness_index(**attribute_extra),
        "roughness": lambda: dem.roughness(**attribute_extra),
        "rugosity": lambda: dem.rugosity(**attribute_extra),
        "texture_shading": lambda: dem.texture_shading(**attribute_extra),
        "fractal_roughness": lambda: dem.fractal_roughness(**attribute_extra),
    }

    logging.info(f"Computing attributes : {list_attributes}")
    attributes: list[Any] = []

    if isinstance(config_attributes, list):
        results_attributes = xdem.terrain.get_terrain_attribute(
            dem,
            attribute=list_attributes,
        )

        # if only one attribute, put it in a list
        if len(list_attributes) == 1:
            attributes = [results_attributes]  # type: ignore
        else:
            attributes = list(results_attributes)
    else:
        for attr in list_attributes:
            attribute_extra = config_attributes.get(attr) or {}  # type: ignore
            attributes.append(from_str_to_fun[attr]().compute())

    # Generate terrain attributes png
    generate_terrain_attributes_png(dem, list_attributes, outputs_folder, attributes)

    # Generate terrain attributes tif
    if export_tif:
        for k, attr in enumerate(list_attributes):
            logging.info(f"Saving {attr} as a raster file (rasters/{attr}.tif)")
            if isinstance(dem, Raster):
                attributes[k].to_file(outputs_folder / "rasters" / f"{attr}.tif")
            else:
                attributes[k].rst.to_file(outputs_folder / "rasters" / f"{attr}.tif")

    logging.info("fin ok generate_terrain_attributes")


def generate_terrain_attributes_png(
    dem: RasterType, list_attribues: list[str], outputs_folder, attributes: list[RasterType]
) -> None:
    """
    Generate terrain attributes png
    :param dem:
    :param attributes:
    """

    n = len(attributes)
    ncols = 3 if n > 6 else 2
    nrows = math.ceil(n / ncols)
    unit = vertical_unit_symbol(dem.crs)
    attribute_params: dict[str, dict[str, Any]] = {
        "hillshade": {"label": "Hillshade", "cmap": "Greys_r", "vlim": (0, 255)},
        "texture_shading": {"label": "Texture shading", "cmap": "Greys_r", "vlim": (-20, 20)},
        "slope": {"label": "Slope (°)", "cmap": "Reds", "vlim": (0, 90)},
        "aspect": {"label": "Aspect (°)", "cmap": "twilight", "vlim": (0, 360)},
        "profile_curvature": {"label": f"Profile curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "tangential_curvature": {"label": f"Tangential curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "planform_curvature": {"label": f"Planform curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "flowline_curvature": {"label": f"Flowline curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "max_curvature": {"label": f"Max. curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "min_curvature": {"label": f"Min. curvature (100/{unit})", "cmap": "RdGy_r", "vlim": (-2, 2)},
        "terrain_ruggedness_index": {"label": "Terrain Ruggedness Index", "cmap": "Purples", "vlim": (None, None)},
        "rugosity": {"label": "Rugosity", "cmap": "YlOrRd", "vlim": (None, None)},
        "topographic_position_index": {
            "label": f"Topographic position index ({unit})",
            "cmap": "Spectral",
            "vlim": (None, None),
        },
        "roughness": {"label": f"Roughness ({unit})", "cmap": "Oranges", "vlim": (None, None)},
        "fractal_roughness": {"label": "Fractal roughness (dimensions)", "cmap": "Reds", "vlim": (None, None)},
    }

    import_optional("matplotlib")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols)
    size_font = 6
    plt.rc("font", size=size_font)
    plt.rc("axes", titlesize=size_font)
    plt.rc("axes", labelsize=size_font)
    plt.rc("xtick", labelsize=size_font)
    plt.rc("ytick", labelsize=size_font)
    plt.rc("legend", fontsize=size_font)
    plt.rc("figure", titlesize=size_font)

    axes = np.atleast_1d(axes).flatten()

    for i, attr in enumerate(list_attribues):
        ax = axes[i]
        params = attribute_params[attr]
        label = params["label"]
        vmin, vmax = params["vlim"]
        kwargs = {"cmap": params["cmap"]}
        kwargs.setdefault("cbar_kwargs", {"label": f"{label}"})
        attributes[i].plot(ax=ax, vmin=vmin, vmax=vmax)  # , **kwargs)
        ax.set_title(None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    plt.tight_layout()
    plt.savefig(outputs_folder / "plots" / "terrain_attributes_map.png", dpi=300)
    plt.close()

# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES)
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
Toolbox for workflows.
"""

import csv
import logging
import math
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, ParamSpec, TypeVar, Union

import geoutils as gu
import numpy as np
from geoutils import Raster
from geoutils.raster import RasterType
from matplotlib.axes import Axes
from pyproj import CRS
from rasterio.warp import calculate_default_transform

import xdem
from xdem import DEM
from xdem._misc import import_optional
from xdem.dem.base import DEMBase
from xdem.dem.xr_accessor import DEMAccessor
from xdem.vcrs import vertical_unit_symbol

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


_P = ParamSpec("_P")
_T = TypeVar("_T")


def _with_plot_style(func: Callable[_P, _T]) -> Callable[_P, _T]:
    """Apply the plotting style in Matplotlib settings."""

    @wraps(func)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        with plt.rc_context(
            {
                "font.size": 6,
                "axes.titlesize": 6,
                "axes.labelsize": 6,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 6,
                "figure.titlesize": 6,
            }
        ):
            return func(*args, **kwargs)

    return wrapped


def _plot_raster(dem: DEMBase, ax: Axes, label: str, **kwargs: Any) -> None:
    """Plot a DEM or a DEM accessor with options."""
    if isinstance(dem, DEM):
        dem.plot(ax=ax, cbar_title=label, **kwargs)
    else:
        data = dem._obj if isinstance(dem, DEMAccessor) else dem
        data.plot(ax=ax, cbar_kwargs={"label": label}, **kwargs)
        ax.set_xlabel("")
        ax.set_ylabel("")


def save_dem(dem: DEMBase, path: str | Path) -> None:
    """Save a DEM to a file."""
    dem.to_file(path) if isinstance(dem, xdem.DEM) else dem.dem.to_file(path)


def save_raster(raster: RasterType, path: str | Path) -> None:
    """Save a Raster to a file."""
    raster.to_file(path) if isinstance(raster, gu.Raster) else raster.rst.to_file(path)


def return_info(dem: DEMBase) -> dict[str, Any]:
    """
    Get information about a DEM to save them in a file.

    :param dem:
    :return dict: Dictionary with information about the DEM
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


@_with_plot_style
def generate_plot(
    dem: DEMBase,
    title: str,
    file_path: str | Path,
    dem_right: DEMBase | None = None,
    title_dem_right: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Generate plot from one or two DEM(s).

    :param dem: Input digital elevation model (left)
    :param title: Title of dem plot (left)
    :param file_path: Path of the figure.
    :param dem_right: Input digital elevation model (right)
    :param title_dem_right: Title of dem_right plot (right)
    :return: None
    """
    import_optional("matplotlib")
    import matplotlib.pyplot as plt

    # Apply default cmap if not given in inputs
    if "cmap" in kwargs:
        cmap = plt.get_cmap(name=kwargs["cmap"])
    else:
        cmap = plt.get_cmap(name="terrain")
    cmap.set_bad(color="k", alpha=None)
    kwargs["cmap"] = cmap

    # Add colormap
    vunit = vertical_unit_symbol(dem.crs)
    label = f"Elevation differences ({vunit})" if vunit else "Elevation differences"
    # Force figsize with the good ratio to prevent larger right axe if not filled
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[6.4, 2.4])

    # Add the first image to the figure (left position)
    _plot_raster(dem, ax1, label, **kwargs)
    ax1.set_title(title)

    # If exists, add the second image to the figure
    if dem_right is not None:
        _plot_raster(dem_right, ax2, label, **kwargs)
        ax2.set_title(title_dem_right)
    else:
        ax2.set_axis_off()

    fig.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@_with_plot_style
def generate_plot_with_profiles(
    dem: DEMBase,
    title: str,
    file_path: str | Path,
    **kwargs: Any,
) -> None:
    """
    Generate plot from a DEM with profiles.

    :param dem: Input digital elevation model
    :param title: Title of dem plot
    :param file_path: Path of the figure
    """
    import_optional("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Add the first image to the figure (left position)

    vunit = vertical_unit_symbol(dem.crs)
    unit_label = f" ({vunit})" if vunit else ""

    # Raster data
    data = dem.data
    ny, nx = data.shape

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

    if isinstance(dem, gu.Raster):
        row_min, row_max = profile_rows_stats
        col_min, col_max = profile_cols_stats
    else:
        import dask

        row_min, row_max = dask.compute(*profile_rows_stats)
        col_min, col_max = dask.compute(*profile_cols_stats)

    ax_left.set_xlabel(
        f"Mean along lines{unit_label}\nMin: {format_values(row_min, 'min')} / Max: {format_values(row_max, 'max')}"
    )

    # Columns profiles
    x = np.arange(nx)
    ax_bottom.plot(x, profile_cols, color="black")
    ax_bottom.set_xlim(ax_map.get_xlim())
    ax_bottom.yaxis.tick_left()
    ax_bottom.xaxis.set_label_position("bottom")
    ax_bottom.set_xlabel(
        f"Mean along columns{unit_label}\nMin: {format_values(col_min, 'min')} / Max: {format_values(col_max, 'max')}"
    )

    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_terrain_attributes(
    dem: DEMBase,
    list_attributes: list[str] | dict[str, Any],
    outputs_folder: Path,
    config_attributes: dict[str, Any],
    proj_crs: bool = True,
    proj_crs_value: str | None = None,
    export_tif: bool = False,
) -> None:
    """Generates the terrain attributes

    :param dem: input digital elevation model
    :param outputs_folder: path of the output folder
    :param list_attributes: list of terrain attributes to generate
    :param config_attributes: configuration for the terrain attributes
    :param proj_crs: whether to project to a CRS
    :param proj_crs_value: the CRS value to project to
    :param export_tif: export tif for each terrain attributes
    :return: None
    """

    target_crs = None

    if dem.crs.is_geographic:
        if proj_crs:
            target_crs = dem.get_metric_crs()

            logging.info(f"Reprojection in default projected CRS ({target_crs})")

        elif not proj_crs_value:
            warnings.warn(
                "As the input dem is not in a projected CRS, the following surface fit attributes might be wrong."
                "Please use a projected CRS or let it empty to reproject in default projected CRS.",
                UserWarning,
            )

        else:
            target_crs = proj_crs_value
            logging.info(f"Reprojection with crs = {target_crs}")

            if CRS.from_user_input(proj_crs_value).is_geographic:
                warnings.warn(
                    'As the input dem is not in a projected CRS and the "reproject/crs" either,'
                    "the following surface fit attributes might be wrong.",
                    UserWarning,
                )

    if target_crs is not None:
        target_transform, _, _ = calculate_default_transform(dem.crs, target_crs, dem.width, dem.height, *dem.bounds)
        reprojected = dem.reproject(crs=target_crs, res=abs(target_transform.a))
        dem = reprojected if isinstance(reprojected, Raster) else reprojected.dem
        if export_tif:
            save_dem(dem, outputs_folder / "elev_reprojected.tif")
    elif dem.res[0] != dem.res[1]:
        reprojected = dem.reproject(res=min(dem.res))
        dem = reprojected if isinstance(reprojected, Raster) else reprojected.dem

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
            terrain = from_str_to_fun[attr]()
            if isinstance(dem, DEM):
                attributes.append(terrain)
            else:
                attributes.append(terrain.compute())

    # Generate terrain attributes png
    generate_terrain_attributes_png(dem, list_attributes, outputs_folder, attributes)

    # Generate terrain attributes tif
    if export_tif:
        for k, attr in enumerate(list_attributes):
            logging.info(f"Saving {attr} as a raster file (rasters/{attr}.tif)")
            save_raster(attributes[k], outputs_folder / "rasters" / f"{attr}.tif")

    logging.info("Terrain attributes generated")


@_with_plot_style
def generate_terrain_attributes_png(
    dem: RasterType,
    list_attributes: list[str] | dict[str, Any],
    outputs_folder: Path,
    attributes: list[RasterType],
) -> None:
    """
    Generate terrain attributes png
    :param dem: input digital elevation model
    :param list_attributes: list of terrain attributes name
    :param outputs_folder: path of the output folder
    :param attributes: terrain attributes
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
    axes = np.atleast_1d(axes).flatten()

    print("ici", list_attributes, attributes)
    for i, attr in enumerate(list_attributes):
        ax = axes[i]
        params = attribute_params[attr]
        label = params["label"]
        vmin, vmax = params["vlim"]
        _plot_raster(attributes[i], ax, label, cmap=params["cmap"], vmin=vmin, vmax=vmax)
        ax.set_title(None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(None)
        ax.set_ylabel(None)

    [fig.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    plt.tight_layout()
    plt.savefig(outputs_folder / "plots" / "terrain_attributes_map.png", dpi=300)
    plt.close(fig)


def recursive_format_values(value: Any) -> Any:
    """Format floats recursively in dictionaries, lists and tuples."""
    if isinstance(value, dict):
        return {k: recursive_format_values(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursive_format_values(elem) for elem in value]
    elif isinstance(value, tuple):
        return tuple(recursive_format_values(elem) for elem in value)
    elif isinstance(value, (float, np.floating)):
        return format_values(float(value))
    else:
        return value


def format_values(val: Union[float, int], key: str = None) -> str:
    """Format values (according to the statistics)."""
    if isinstance(val, str):
        return val
    if key is not None and "count" in key.lower():
        return str(int(val))
    if key is not None and "percentage" in key.lower():
        return f"{val:.2f}" + "%"
    elif abs(val) > 10e4:
        return np.format_float_scientific(val, precision=3)
    elif abs(val) < 10e-4:
        return np.format_float_scientific(val, precision=3)
    else:
        return f"{val:.3f}"


def generate_pdf(html_path: Path, pdf_path: Path) -> None:
    """
    Create PDF report page from HTML page.

    :param html_path: Path to the HTML file
    :param pdf_path: Path to the PDF file
    :return: None
    """
    if _HAS_PLUTOPRINT:
        book = plutoprint.Book(plutoprint.PAGE_SIZE_A4, plutoprint.PAGE_MARGINS_NARROW)
        book.load_url(html_path.resolve().as_uri())

        # Export the entire document to PDF
        book.write_to_pdf(pdf_path)
        logging.info("Report generated in " + str(pdf_path))


def save_stat_as_csv(data: dict[str, float], file_path: Path) -> None:
    """
    Save the statistics into a CSV file.

    :param data: Statistics dictionary
    :param file_path: Path to the CSV file
    """
    cleaned_data = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in data.items()}

    fieldnames = list(cleaned_data.keys())
    print(file_path, type(file_path))
    with file_path.open(mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(cleaned_data)


@_with_plot_style
def compute_histogram(
    stats_before: Dict[str, Any],
    stats_after: Dict[str, Any],
    diff_before: RasterType,
    diff_after: RasterType,
    path: str | Path,
) -> None:
    """
    Compute altitudes difference histogram.

    :param stats_before: Statistics before coregistration
    :param stats_after: Statistics after coregistration
    :param diff_before: Altitude difference before coregistration
    :param diff_after: Altitude difference after coregistration
    :param path: Path to save the histogram plot
    """

    import_optional("matplotlib")
    import matplotlib.pyplot as plt

    logging.info("Computing histogram on altitude difference")
    vunit = (
        vertical_unit_symbol(diff_before.crs)
        if isinstance(diff_before, gu.Raster)
        else vertical_unit_symbol(diff_before.dem.crs)
    )
    unit_label = f" ({vunit})" if vunit else ""

    # Force figsize with the same size as generate_plot function
    fig = plt.figure(figsize=[6.4, 2.34])

    bins = np.linspace(stats_before["min"], stats_before["max"], 300)
    std_before = np.std(diff_before)
    plt.xlim((-4 * std_before, 4 * std_before))

    if isinstance(diff_after, gu.Raster):
        plt.hist(diff_before.data.flatten(), bins=bins, color="g", alpha=0.5, label="Before coregistration")
        plt.hist(diff_after.data.flatten().data, bins=bins, color="b", alpha=0.5, label="After coregistration")

    else:
        import dask
        import dask.array as da

        hist_before, _ = da.histogram(diff_before.data, bins=bins)
        hist_after, _ = da.histogram(diff_after.data, bins=bins)

        hist_before, hist_after = dask.compute(hist_before, hist_after)

        plt.stairs(hist_before, bins, fill=True, color="g", alpha=0.5, label="Before coregistration")
        plt.stairs(hist_after, bins, fill=True, color="b", alpha=0.5, label="After coregistration")
    ax = plt.gca()
    plt.text(
        0.2,
        0.5,
        f"Before:\nmedian = {format_values(stats_before['median'], 'median')}\n"
        f"nmad = {format_values(stats_before['nmad'], 'nmad')}",
        color="g",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    plt.text(
        0.8,
        0.5,
        f"After:\nmedian = {format_values(stats_after['median'], 'median')}\n"
        f"nmad = {format_values(stats_after['nmad'], 'nmad')}",
        color="b",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    plt.title("Histogram of elevation differences\nbefore and after coregistration")
    plt.xlabel(f"Elevation differences{unit_label}")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

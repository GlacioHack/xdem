# Copyright (c) 2024 xDEM developers
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
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

"""Terrain attribute calculations using multiprocessing."""
from pathlib import Path

import numpy as np
import rasterio as rio
from geoutils._typing import NDArrayNum
from geoutils.raster import Raster, RasterType, compute_tiling
from geoutils.raster.distributed_computing import AbstractCluster, ClusterGenerator

from xdem.terrain import get_terrain_attribute


def load_raster_tile(raster_unload: RasterType, tile: NDArrayNum) -> RasterType:
    """
    Extracts a specific tile (spatial subset) from the raster based on the provided tile coordinates.

    :param raster_unload: The input raster from which the tile is to be extracted.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :return: The extracted raster tile.
    """
    xmin, xmax, ymin, ymax = tile
    raster_tile = raster_unload.icrop(bbox=(xmin, ymin, xmax, ymax))
    return raster_tile


def remove_tile_padding(dem: RasterType, raster_tile: RasterType, tile: NDArrayNum, padding: int) -> None:
    """
    Removes the padding added around tiles during terrain attribute computation to prevent edge effects.

    :param dem: The full DEM object from which tiles are extracted.
    :param raster_tile: The raster tile with possible padding that needs removal.
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :param padding: The padding size to be removed from each side of the tile.
    """
    # Remove padding if the tile is not on the edge of the DEM
    xmin, xmax, ymin, ymax = 0, raster_tile.height, 0, raster_tile.width
    if tile[0] != 0:
        tile[0] += padding
        xmin += padding
    if tile[1] != dem.height:
        tile[1] -= padding
        xmax -= padding
    if tile[2] != 0:
        tile[2] += padding
        ymin += padding
    if tile[3] != dem.width:
        tile[3] -= padding
        ymax -= padding
    raster_tile.icrop(bbox=(xmin, ymin, xmax, ymax), inplace=True)


def get_terrain_attribute_multiproc(
    dem: str | RasterType,
    attribute: str | list[str],
    tile_size: int,
    out_dir: str | None = None,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: str = "Horn",
    tri_method: str = "Riley",
    fill_method: str = "none",
    edge_method: str = "none",
    window_size: int = 3,
    cluster: AbstractCluster | None = None,
) -> None:
    """
    Compute terrain attributes (e.g., slope, aspect, hillshade) in parallel using tiling and multiprocessing.
    Saves the terrain attribute raster(s) as a GeoTIFF in out_dir or next to the original DEM by default.
    See description of :func:`xdem.DEM.get_terrain_attribute` for more information.

    :param dem: The DEM to analyze.
    :param attribute: The terrain attribute(s) to calculate.
    :param tile_size: Size of each tile in pixels (square tiles).
    :param out_dir: Saving directory for the terrain attribute rasters(s).
    :param resolution: The X/Y or (X, Y) resolution of the DEM.
    :param degrees: Convert radians to degrees?
    :param hillshade_altitude: The shading altitude in degrees (0-90°). 90° is straight from above.
    :param hillshade_azimuth: The shading azimuth in degrees (0-360°) going clockwise, starting from north.
    :param hillshade_z_factor: Vertical exaggeration factor.
    :param slope_method: Method to calculate the slope, aspect and hillshade: "Horn" or "ZevenbergThorne".
    :param tri_method: Method to calculate the Terrain Ruggedness Index: "Riley" (topography) or "Wilson" (bathymetry).
    :param fill_method: See the 'get_quadric_coefficients()' docstring for information.
    :param edge_method: See the 'get_quadric_coefficients()' docstring for information.
    :param window_size: The window size for windowed ruggedness and roughness indexes.
    :param cluster: A `geoutils.AbstractCluster` object that handles multiprocessing. This object is responsible for
            distributing the tasks across multiple processes and retrieving the results. If `None`, the function will
            execute without parallelism.

    :raises ValueError: If the inputs are poorly formatted or are invalid.
    """
    if cluster is None:
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    # load dem metadata if not already loaded
    if isinstance(dem, str):
        dem = Raster(dem)

    # Generate tiling grid
    # Adapt the overlap if needed
    overlap = 1
    tilind_grid = compute_tiling(tile_size, dem.shape, dem.shape, overlap=overlap)

    # loop over each attribute
    if isinstance(attribute, str):
        attribute = [attribute]
    for attr in attribute:
        dem_path = Path(dem.filename)
        filename = dem_path.stem + "_" + attr + ".tif"
        # The attr will be saved in out_dir or next to the DEM under the name "dem_path_attr.tif"
        if out_dir is None:
            outfile = dem_path.parent.joinpath(filename)
        else:
            outfile = Path(out_dir).joinpath(filename)

        # Create tasks for multiprocessing
        tasks = []

        for row in range(tilind_grid.shape[0]):
            for col in range(tilind_grid.shape[1]):
                tile = tilind_grid[row, col]
                tasks.append(
                    cluster.launch_task(
                        fun=terrain_attribute_by_block,
                        args=[
                            dem,
                            attr,
                            tile,
                            overlap,
                            resolution,
                            degrees,
                            hillshade_altitude,
                            hillshade_azimuth,
                            hillshade_z_factor,
                            slope_method,
                            tri_method,
                            fill_method,
                            edge_method,
                            window_size,
                        ],
                    )
                )

        # get first tile to retrieve dtype and nodata
        attr_tile1, _ = cluster.get_res(tasks[1])

        with rio.open(
            outfile,
            "w",
            driver="GTiff",
            height=dem.height,
            width=dem.width,
            count=dem.count,
            dtype=attr_tile1.dtype,
            crs=dem.crs,
            transform=dem.transform,
            nodata=attr_tile1.nodata,
        ) as dst:
            try:
                # Retrieve terrain attribute computation results
                for results in tasks:
                    attr_tile, dst_tile = cluster.get_res(results)

                    dst_window = rio.windows.Window(
                        col_off=dst_tile[2],
                        row_off=dst_tile[0],
                        width=dst_tile[3] - dst_tile[2],
                        height=dst_tile[1] - dst_tile[0],
                    )

                    # Cast to 3D before saving if single band
                    if attr_tile.count == 1:
                        data = attr_tile[np.newaxis, :, :]
                    else:
                        data = attr_tile.data

                    # Write the tile to the correct location in the full raster
                    dst.write(data, window=dst_window)
            except Exception as e:
                raise RuntimeError(f"Error retrieving terrain attribute from multiprocessing tasks: {e}")


def terrain_attribute_by_block(
    dem: RasterType,
    attribute: str,
    tile: NDArrayNum,
    overlap: int,
    resolution: tuple[float, float] | float | None = None,
    degrees: bool = True,
    hillshade_altitude: float = 45.0,
    hillshade_azimuth: float = 315.0,
    hillshade_z_factor: float = 1.0,
    slope_method: str = "Horn",
    tri_method: str = "Riley",
    fill_method: str = "none",
    edge_method: str = "none",
    window_size: int = 3,
) -> tuple[RasterType, NDArrayNum]:
    """
    Calculate a terrain attribute for a specific tile (spatial subset) of the DEM.

    :param dem: The input DEM as a raster object.
    :param attribute: The terrain attribute to compute (e.g., slope, aspect, hillshade).
    :param tile: The bounding box of the tile as [xmin, xmax, ymin, ymax].
    :param overlap: The overlap (in pixels) to add to tiles to prevent edge artifacts.
    :param resolution: The spatial resolution of the DEM or None for the default resolution.
    :param degrees: Whether to convert angle-based attributes to degrees (True) or keep in radians.
    :param hillshade_altitude: The sun altitude angle for hillshade (used if hillshade is the attribute).
    :param hillshade_azimuth: The sun azimuth angle for hillshade.
    :param hillshade_z_factor: Vertical exaggeration factor for hillshade calculations.
    :param slope_method: Method for slope calculation. Options: "Horn", "ZevenbergThorne".
    :param tri_method: Method for Terrain Ruggedness Index (TRI) calculation: "Riley" or "Wilson".
    :param fill_method: Edge/corner filling method for missing data.
    :param edge_method: Handling method for edges during computation.
    :param window_size: The size of the moving window for windowed terrain attributes.

    :return: A tuple containing the raster object with the computed attribute and the tile coordinates.
    """
    # Load tile
    dem_tile = load_raster_tile(dem, tile)

    # Compute terrain attribute
    dem_tile_attr = get_terrain_attribute(
        dem_tile,
        attribute,
        resolution,
        degrees,
        hillshade_altitude,
        hillshade_azimuth,
        hillshade_z_factor,
        slope_method,
        tri_method,
        fill_method,
        edge_method,
        window_size,
    )

    # Remove padding
    remove_tile_padding(dem, dem_tile_attr, tile, overlap)
    return dem_tile_attr, tile

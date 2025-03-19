import numpy as np
import rasterio as rio
from geoutils._typing import NDArrayNum
from geoutils.raster import RasterType, compute_tiling
from geoutils.raster.distributed_computing.cluster import (
    AbstractCluster,
    ClusterGenerator,
)

from xdem import DEM
from xdem.terrain import get_terrain_attribute


def get_raster_tile(raster_unload: RasterType, tile: NDArrayNum) -> RasterType:
    xmin, xmax, ymin, ymax = tile
    raster_tile = raster_unload.icrop(bbox=(xmin, ymin, xmax, ymax))
    return raster_tile


def remove_tile_padding(dem: DEM, raster_tile: RasterType, tile: NDArrayNum, padding: int) -> None:
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
    dem_path: str,
    attribute: str | list[str],
    tile_size: int,
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
    See description of terrain.get_terrain_attribute.

    :param dem_path: The DEM to analyze.
    :param attribute: The terrain attribute(s) to calculate.
    :param tile_size: Size of each tile in pixels (square tiles).
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

    :returns: One or multiple arrays of the requested attribute(s)
    """
    if cluster is None:
        cluster = ClusterGenerator("basic")  # type: ignore
    assert cluster is not None  # for mypy

    # load dem metadata
    dem = DEM(dem_path)

    # Generate tiling grid
    # Adapt the overlap if needed
    overlap = 1
    tilind_grid = compute_tiling(tile_size, dem.shape, dem.shape, overlap=overlap)

    # loop over each attribute
    if isinstance(attribute, str):
        attribute = [attribute]
    for attr in attribute:
        # The attr will be saved next to the DEM under the name "dem_attr"
        outfile = dem_path.removesuffix(".tif") + "_" + attr + ".tif"

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
                            attribute,
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

        with rio.open(
            outfile,
            "w",
            driver="GTiff",
            height=dem.height,
            width=dem.width,
            count=dem.count,
            dtype=dem.dtype,
            crs=dem.crs,
            transform=dem.transform,
            nodata=dem.nodata,
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
    # Load tile
    dem_tile = get_raster_tile(dem, tile)

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

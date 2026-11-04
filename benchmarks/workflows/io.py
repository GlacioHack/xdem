"""Open and write the inputs and outputs (I/O) for benchmark tests."""

from __future__ import annotations

import os
import pathlib
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio
from geoutils._misc import import_optional
from shapely.geometry import box

from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import ExecutionMode

############################
# Shared size and I/O helpers
############################


def logical_raster_size_mb(config: RuntimeConfig) -> float:
    """Return a float32 raster size in MBs."""

    return config.shape[0] * config.shape[1] * np.dtype("float32").itemsize / 1_000_000


def memory_limit_mb(memory_limit: str) -> float:
    """Convert the memory formats of workers (used by the benchmark suite) to MBs."""

    # Match the common decimal and binary units accepted by Dask
    value = memory_limit.strip().lower()
    if value.endswith("gib"):
        return float(value[:-3]) * 1024**3 / 1_000_000
    if value.endswith("gb"):
        return float(value[:-2]) * 1000
    if value.endswith("mib"):
        return float(value[:-3]) * 1024**2 / 1_000_000
    if value.endswith("mb"):
        return float(value[:-2])
    return float(value) / 1_000_000


def tiff_block_size(size: int, requested: int) -> int:
    """Return a valid GeoTIFF block size to write with."""

    # The GeoTIFF tile must be divisible by 16 (to allow factors of 2), and smaller than raster size
    block_size = min(size, requested, 512)
    return max(16, block_size // 16 * 16)


def read_raster_center(filename: str) -> float:
    """Read one central output pixel without loading the complete raster, to validate outputs values without loading."""

    with rio.open(filename) as dataset:
        row = dataset.height // 2
        col = dataset.width // 2
        return float(dataset.read(1, window=rio.windows.Window(col, row, 1, 1))[0, 0])


def read_point_file_sample(filename: str, column: str) -> tuple[int, float]:
    """Read the feature count and one value without loading a point file, again for validation."""

    if pathlib.Path(filename).suffix.lower() in (".las", ".laz"):
        laspy = import_optional("laspy")
        with laspy.open(filename) as reader:
            count = int(reader.header.point_count)
            sample = reader.read_points(1)
        return count, float(sample.z[0])

    import pyogrio

    # Use file metadata for the complete row count and read only one feature for the value check
    info = pyogrio.read_info(filename, force_feature_count=True)
    sample = pyogrio.read_dataframe(filename, columns=[column], max_features=1)
    return int(info["features"]), float(sample[column].iloc[0])


def open_raster_input(filename: str, backend: ExecutionMode, config: RuntimeConfig) -> Any:
    """Open one raster through the interface used by an execution backend."""

    if backend == "dask":
        from geoutils.raster.xr_accessor import open_raster

        # Raster chunks remain lazy until the output values are computed
        return open_raster(filename, chunks={"y": config.chunks[0], "x": config.chunks[1]})

    from geoutils import Raster

    # In-memory comparisons load the complete input while multiprocessing reads windows
    return Raster(filename, load_data=backend == "inmem")


def prepare_output_file(
    directory: str,
    backend: ExecutionMode,
    operation: str,
    suffix: str = ".tif",
) -> str:
    """Return an output path that can be reused for one benchmark operation."""

    filename = os.path.join(directory, f"output-{backend}-{operation}{suffix}")
    if os.path.isfile(filename):
        os.remove(filename)
    return filename


def materialize_raster_output(
    raster: Any,
    backend: ExecutionMode,
    filename: str | None,
    config: RuntimeConfig,
) -> str:
    """Write a complete raster output and return its file path."""

    if backend == "multiprocessing":
        # Multiprocessing operations already wrote their returned Raster to disk
        return str(raster.name)
    if filename is None:
        raise ValueError("A raster output path is required outside multiprocessing")

    # Match the tiled, uncompressed output used by the external commands
    block_y = tiff_block_size(config.shape[0], config.chunks[0])
    block_x = tiff_block_size(config.shape[1], config.chunks[1])
    if backend == "dask":
        if raster.chunks is None:
            raise ValueError("Dask benchmark output must remain chunked before writing")

        # GeoTIFF stores boolean masks as bytes
        dtype = "uint8" if np.issubdtype(raster.dtype, np.bool_) else raster.dtype
        raster.rst.to_file(
            filename,
            driver="GTiff",
            dtype=dtype,
            compute=True,
            tiled=True,
            blockysize=block_y,
            blockxsize=block_x,
            compress="NONE",
            BIGTIFF="IF_NEEDED",
        )
        if raster.rst.is_loaded:
            raise AssertionError("Writing a Dask raster must leave its source array lazy.")
        return filename

    # In-memory results use the same tiled output contract
    raster.to_file(
        filename,
        co_opts={
            "TILED": "YES",
            "BLOCKYSIZE": str(block_y),
            "BLOCKXSIZE": str(block_x),
            "COMPRESS": "NONE",
        },
    )
    return filename


##############################
# Deterministic source files
##############################


def write_constant_raster(filename: str, config: RuntimeConfig) -> None:
    """Write a deterministic constant raster one storage block at a time."""

    if os.path.exists(filename):
        return

    # Use a real WGS84 extent so reprojection exercises a coordinate transform
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = tiff_block_size(height, config.chunks[0])
    block_x = tiff_block_size(width, config.chunks[1])

    # Compression keeps the deterministic constant fixture compact on disk
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-99999,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        compress="DEFLATE",
        BIGTIFF="IF_NEEDED",
    ) as destination:
        # Allocate only the current storage block instead of the complete raster
        for _, window in destination.block_windows(1):
            block = np.full((int(window.height), int(window.width)), config.raster_value, dtype=np.float32)
            destination.write(block, indexes=1, window=window)


def write_vector_source(filename: str, features_per_axis: int = 1) -> None:
    """Write regularly spaced polygons used by rasterization and mask scenarios."""

    if os.path.exists(filename):
        return
    if features_per_axis < 1:
        raise ValueError("Vector features per axis must be strictly positive")

    # Leave a regular gap around every feature while retaining one central feature
    x_edges = np.linspace(7.05, 7.95, features_per_axis + 1)
    y_edges = np.linspace(45.05, 45.95, features_per_axis + 1)
    geometries = []
    for x_start, x_stop in zip(x_edges[:-1], x_edges[1:]):
        for y_start, y_stop in zip(y_edges[:-1], y_edges[1:]):
            x_margin = (x_stop - x_start) * 0.2
            y_margin = (y_stop - y_start) * 0.2
            geometries.append(box(x_start + x_margin, y_start + y_margin, x_stop - x_margin, y_stop - y_margin))

    # Constant burn values give every engine the same binary output
    vector = gpd.GeoDataFrame({"value": np.ones(len(geometries), dtype=np.uint8)}, geometry=geometries, crs=4326)
    vector.to_file(filename, driver="GPKG")

"""Define polygonization benchmarks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio

from benchmarks.workflows.config import (
    DEM_SIZES,
    RuntimeConfig,
    dem_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
    reference_case,
    strategy_cases,
)
from benchmarks.workflows.io import tiff_block_size

ORDER = 60
STRATEGIES = ("label_union", "label_stitch", "geometry_stitch")
STRATEGY_LABELS = {
    "label_union": "Label union",
    "label_stitch": "Label stitch",
    "geometry_stitch": "Geometry stitch",
}
POLYGON_OPTIONS = {"polygon_regions_per_axis": 21}

###########################################
# Define setup for polygonization operation
###########################################


def write_polygon_dem(filename: str, config: RuntimeConfig) -> None:
    """Prepare a dem of regularly spaced regions."""

    if os.path.exists(filename):
        return
    regions = int(config.value("polygon_regions_per_axis", 1))
    if regions < 1:
        raise ValueError("Polygon regions per axis must be strictly positive")

    # Separate rectangles with value of 1 with nodata so every rectangle is one region
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = tiff_block_size(height, config.chunks[0])
    block_x = tiff_block_size(width, config.chunks[1])

    # Write per block
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
        for _, window in destination.block_windows(1):
            # Pixel phases locate the inner half of every regular grid cell
            row_start = int(window.row_off)
            col_start = int(window.col_off)
            rows = np.arange(row_start, row_start + int(window.height))
            cols = np.arange(col_start, col_start + int(window.width))
            row_phase = ((rows + 0.5) * regions / height) % 1
            col_phase = ((cols + 0.5) * regions / width) % 1
            inside_rows = (row_phase >= 0.25) & (row_phase <= 0.75)
            inside_cols = (col_phase >= 0.25) & (col_phase <= 0.75)
            inside = inside_rows[:, None] & inside_cols[None, :]

            # Nodata gaps keep neighboring rectangles disconnected for both engines
            block = np.full(inside.shape, -99999, dtype=np.float32)
            block[inside] = config.dem_value
            destination.write(block, indexes=1, window=window)


def polygonize_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define polygonization options."""

    options: dict[str, Any] = {"target_values": 1}
    if case.strategy is not None:
        options["strategy"] = case.strategy
    return options


def prepare_polygonize(runner: Any, case: Case) -> None:
    """Prepare the dem to be polygonized."""

    write_polygon_dem(runner.path("source-polygonize.tif"), runner.config)


def run_polygonize(runner: Any, case: Case) -> float:
    """Run polygonization and write output (Dask/MP)."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    dem = runner.make_dem(runner.path("source-polygonize.tif"))

    # The selected chunk strategy reconciles polygons that cross output tiles
    mp_config = runner._multiproc_config(suffix=".gpkg") if runner.backend == "multiprocessing" else None
    options = polygonize_options(case, runner.config)
    polygons = (
        dem.dem.polygonize(**options) if runner.backend == "dask" else dem.polygonize(**options, mp_config=mp_config)
    )
    runner._last_output_file = runner._output_path(suffix=".gpkg")
    polygon_data = polygons if isinstance(polygons, gpd.GeoDataFrame) else polygons.ds
    polygon_data.to_file(runner._last_output_file)
    return float(len(polygon_data))


POLYGONIZE = Operation(
    "polygonize",
    prepare_polygonize,
    run_polygonize,
    polygonize_options,
    ("strategy",),
    label="Polygonization",
    order=11,
    large_data_cases=tuple(
        Case(engine="rasterio", execution=execution, strategy="label_stitch")
        for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (POLYGONIZE,)


def strategy_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define dem size with chunks crossed by every strategy."""

    assert parameter is not None
    size = int(parameter)
    return {"shape": (size, size), "chunks": (500, 500)}


#####################################
# Define benchmarks and comparisons
#####################################


EXECUTION_CASES = execution_cases(
    None,
    "rasterio",
    strategy="label_stitch",
    options=POLYGON_OPTIONS,
    labels={"engine": "Rasterio/GDAL", "strategy": STRATEGY_LABELS["label_stitch"]},
)
STRATEGY_CASES = strategy_cases(
    None,
    "rasterio",
    STRATEGIES,
    execution="dask",
    options=POLYGON_OPTIONS,
    strategy_labels=STRATEGY_LABELS,
    variant="strategy",
)
REFERENCE = reference_case(EXECUTION_CASES, implementation="gdal")


def polygon_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the dem, chunks and region grid."""

    config = configs[0]
    regions = int(config.value("polygon_regions_per_axis", 1))
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} dem; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {regions:,} × {regions:,} dem regions"
    )


# Measure the execution cases with their GDAL reference and the strategy cases over the same dem-size axis
BENCHMARKS = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        POLYGONIZE,
        (*EXECUTION_CASES, REFERENCE),
        dem_size_config,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=polygon_workload,
    ),
    parameter_config(
        "dem_size",
        DEM_SIZES,
        POLYGONIZE,
        STRATEGY_CASES,
        strategy_size,
        name="polygonize-strategy",
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=polygon_workload,
    ),
)

# Comparisons select saved series for report plots: GeoUtils with GDAL, then the GeoUtils strategies by themselves
COMPARISONS = (
    comparison(BENCHMARKS[0], by="execution"),
    comparison(BENCHMARKS[1], by="strategy", documentation=False),
)

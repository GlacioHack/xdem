"""Build, execute and validate GDAL CLI commands."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from typing import Any, Literal

import geopandas as gpd

from benchmarks.comparisons.subprocess import execute_command
from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import Case
from benchmarks.workflows.io import read_raster_center

# Only these GeoUtils operations have an equivalent GDAL CLI command for the external comparison
ComparisonOperation = Literal["clip", "reproject", "polygonize", "rasterize", "grid"]
COMPARISON_OPERATIONS: tuple[ComparisonOperation, ...] = ("clip", "reproject", "polygonize", "rasterize", "grid")

# We list the GDAL gridding algorithms that the command builder can use for matching GeoUtils methods
GdalGridAlgorithm = Literal[
    "nearest",
    "linear",
    "invdist",
    "average",
    "minimum",
    "maximum",
    "range",
    "count",
    "average_distance",
    "average_distance_pts",
]


@dataclass(frozen=True)
class GdalCommand:
    """Store one GDAL command and the file it must create."""

    command: list[str]
    output_file: str


def _warp_memory_limit_mb(config: RuntimeConfig) -> int:
    """Return the GDAL warp memory closest to one GeoUtils execution chunk."""

    # GDAL holds one Float32 source and destination buffer plus their 1-bit nodata masks
    chunk_height = min(config.shape[0], config.chunks[0])
    chunk_width = min(config.shape[1], config.chunks[1])
    working_bits = chunk_height * chunk_width * 2 * (32 + 1)
    return max(1, math.ceil(working_bits / 8 / 1024**2))


def _require_command(name: str) -> str:
    """Return an installed GDAL or raise a clear error."""

    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required GDAL command is not installed: {name}")
    return executable


def build_gdal_grid_command(
    point_file: str,
    output_file: str,
    *,
    algorithm: GdalGridAlgorithm,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    radius: tuple[float, float],
    layer: str = "source-points",
    zfield: str = "z",
    min_points: int = 1,
    distance_power: float = 2.0,
    nodata: float = -9999,
    gdal_cachemax_mb: int | None = None,
    output_crs: str | None = None,
    creation_options: list[str] | None = None,
) -> GdalCommand:
    """Build one GDAL gridding command for a defined raster grid."""

    # Linear interpolation has one fallback radius while other methods use an ellipse
    if algorithm == "linear":
        if radius[0] != radius[1]:
            raise ValueError("GDAL linear gridding requires an equal X and Y fallback radius")
        algorithm_options = [f"radius={radius[0]}", f"nodata={nodata}"]
    else:
        algorithm_options = [f"radius1={radius[0]}", f"radius2={radius[1]}", "angle=0"]

        # GDAL gives IDW a additional power parameter
        if algorithm == "invdist":
            algorithm_options.insert(0, f"power={distance_power}")
            algorithm_options.append("smoothing=0")
            algorithm_options.append("max_points=0")
        if algorithm != "nearest":
            algorithm_options.append(f"min_points={min_points}")
        algorithm_options.append(f"nodata={nodata}")
    algorithm_definition = ":".join((algorithm, *algorithm_options))

    # Bounds describe raster edges while GeoUtils grid coordinates describe cell centers
    left, bottom, right, top = bounds
    height, width = shape
    command = [
        _require_command("gdal_grid"),
        # Fix internal parallelism so every comparison uses one calculation thread
        "--config",
        "GDAL_NUM_THREADS",
        "1",
    ]
    if gdal_cachemax_mb is not None:
        command.extend(("--config", "GDAL_CACHEMAX", str(gdal_cachemax_mb)))
    command.extend(
        [
            "-zfield",
            zfield,
            "-a",
            algorithm_definition,
            "-txe",
            str(left),
            str(right),
            "-tye",
            str(bottom),
            str(top),
            "-outsize",
            str(width),
            str(height),
            "-ot",
            "Float64",
        ]
    )
    if output_crs is not None:
        command.extend(("-a_srs", output_crs))
    if creation_options is not None:
        command.extend(creation_options)
    command.extend(
        [
            "-l",
            layer,
            point_file,
            output_file,
        ]
    )
    return GdalCommand(command=command, output_file=output_file)


def build_gdal_command(
    operation: ComparisonOperation,
    case: Case,
    config: RuntimeConfig,
    raster_file: str,
    vector_file: str,
    point_file: str,
) -> GdalCommand:
    """Build one GDAL command with the same input and output grid as GeoUtils."""

    if config.directory is None:
        raise ValueError("GDAL comparison commands require an explicit output directory")
    operation_method = case.method

    # Match output storage tiles and the GDAL block cache used by GeoUtils
    # These settings control file access, not GDAL's internal processing chunks
    height, width = config.shape
    common_config = ["--config", "GDAL_CACHEMAX", str(config.gdal_cachemax_mb)]
    common_creation = ["-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512"]

    if operation == "clip":
        # Keep the source extent and mask cells outside the cutline, matching Raster.clip()
        output_file = os.path.join(config.directory, "output-gdal-clip.tif")
        command = [
            _require_command("gdalwarp"),
            *common_config,
            "-overwrite",
            "-cutline",
            vector_file,
            "-cl",
            "source-vector",
            "-te",
            "7",
            "45",
            "8",
            "46",
            "-te_srs",
            "EPSG:4326",
            "-ts",
            str(width),
            str(height),
            "-r",
            "near",
            "-ot",
            "Float32",
            "-dstnodata",
            "-99999",
            "-et",
            "0",
            "-wm",
            str(_warp_memory_limit_mb(config)),
            "-wo",
            "NUM_THREADS=1",
            "-wo",
            "XSCALE=1",
            "-wo",
            "YSCALE=1",
            *common_creation,
            raster_file,
            output_file,
        ]
        return GdalCommand(command, output_file)

    if operation == "reproject":
        # Match the GeoUtils WGS84 to UTM zone 32N grid and selected resampling method
        if case.method is None:
            raise ValueError("A GDAL reprojection comparison requires a resampling method")
        resampling = {"nearest": "near", "cubic_spline": "cubicspline"}.get(case.method, case.method)
        output_file = os.path.join(config.directory, "output-gdal-reproject.tif")
        command = [
            _require_command("gdalwarp"),
            *common_config,
            "-overwrite",
            "-t_srs",
            "EPSG:32632",
            "-ts",
            str(width),
            str(height),
            "-r",
            resampling,
            "-ot",
            "Float32",
            "-dstnodata",
            "-99999",
            "-et",
            "0",
            "-wm",
            str(_warp_memory_limit_mb(config)),
            "-wo",
            "NUM_THREADS=1",
            "-wo",
            "XSCALE=1",
            "-wo",
            "YSCALE=1",
            *common_creation,
            raster_file,
            output_file,
        ]
        return GdalCommand(command, output_file)

    if operation == "polygonize":
        # The patterned raster lets feature count validate the complete vector output
        output_file = os.path.join(config.directory, "output-gdal-polygonize.gpkg")
        command = [
            _require_command("gdal_polygonize.py"),
            *common_config,
            raster_file,
            output_file,
            "polygons",
            "value",
            "-f",
            "GPKG",
            "-b",
            "1",
        ]
        return GdalCommand(command, output_file)

    if operation == "rasterize":
        # Byte burn values match the GeoUtils binary rasterization output
        output_file = os.path.join(config.directory, "output-gdal-rasterize.tif")
        command = [
            _require_command("gdal_rasterize"),
            *common_config,
            "-burn",
            "1",
            "-init",
            "0",
            "-ot",
            "Byte",
            "-te",
            "7",
            "45",
            "8",
            "46",
            "-ts",
            str(width),
            str(height),
            "-a_srs",
            "EPSG:4326",
            "-l",
            "source-vector",
            *common_creation,
            vector_file,
            output_file,
        ]
        return GdalCommand(command, output_file)

    if operation == "grid":
        # Select the equivalent GDAL algorithm for each representative GeoUtils method
        output_file = os.path.join(config.directory, "output-gdal-grid.tif")
        algorithms: dict[str, GdalGridAlgorithm] = {
            "nearest": "nearest",
            "linear": "linear",
            "idw": "invdist",
            "mean": "average",
            "average": "average",
            "minimum": "minimum",
            "min": "minimum",
            "maximum": "maximum",
            "max": "maximum",
            "range": "range",
            "count": "count",
            "average_distance": "average_distance",
            "average_distance_pts": "average_distance_pts",
        }
        if operation_method not in algorithms:
            raise ValueError(f"GDAL has no matching gridding method for {operation_method!r}")

        # A zero radius means unlimited nearest support and disables linear fallback outside the triangulation
        if operation_method in ("nearest", "linear"):
            radius = (0.0, 0.0)
        else:
            pixel_width = 1 / width
            pixel_height = 1 / height
            radius = (
                config.value("grid_dist_nodata_pixel", float("inf")) * pixel_width,
                config.value("grid_dist_nodata_pixel", float("inf")) * pixel_height,
            )
        return build_gdal_grid_command(
            point_file,
            output_file,
            algorithm=algorithms[operation_method],
            bounds=(7.0, 45.0, 8.0, 46.0),
            shape=config.shape,
            radius=radius,
            min_points=1,
            distance_power=2,
            gdal_cachemax_mb=config.gdal_cachemax_mb,
            output_crs="EPSG:4326",
            creation_options=common_creation,
        )

    raise ValueError(f"Unsupported GDAL comparison operation: {operation}")


def execute_gdal(runner: Any, case: Case) -> float:
    """Run and validate the GDAL command matching the selected operation."""

    operation = runner.operation.name
    raster_file = runner.path("source-polygonize.tif" if operation == "polygonize" else "source-raster.tif")
    command = build_gdal_command(
        operation,
        case,
        runner.config,
        raster_file=raster_file,
        vector_file=runner.path("source-vector.gpkg"),
        point_file=runner.path("source-points.gpkg"),
    )
    execute_command("GDAL", command.command, command.output_file)
    runner._last_output_file = command.output_file

    # Feature count validates vector output, raster workflows use one deterministic central pixel
    if operation == "polygonize":
        value = float(len(gpd.read_file(command.output_file)))
        expected = int(runner.config.value("polygon_regions_per_axis", 1)) ** 2
    else:
        value = read_raster_center(command.output_file)
        expected = runner.config.raster_value
    if operation == "reproject" and case.method == "sum":
        valid_value = math.isfinite(value)
    else:
        valid_value = value == expected
    if not valid_value:
        raise RuntimeError(f"Unexpected GDAL {operation} validation value: {value}")
    return value

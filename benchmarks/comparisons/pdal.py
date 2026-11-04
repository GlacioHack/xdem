"""Build, execute and validate PDAL pipelines."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import pyogrio

from benchmarks.comparisons.subprocess import execute_command
from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import Case
from benchmarks.workflows.io import read_point_file_sample

# Only these GeoUtils operations have an equivalent PDAL pipeline
PdalComparisonOperation = Literal["subsample", "to_pointcloud"]
PDAL_COMPARISON_OPERATIONS: tuple[PdalComparisonOperation, ...] = ("subsample", "to_pointcloud")


@dataclass(frozen=True)
class PdalCommand:
    """Store one PDAL command, its pipeline definition and the file it must create."""

    command: list[str]
    pipeline_file: str
    output_file: str


def _require_command(name: str) -> str:
    """Return an installed PDAL executable or raise a clear environment error."""

    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required PDAL command is not installed: {name}")
    return executable


def build_pdal_command(
    operation: PdalComparisonOperation,
    case: Case,
    config: RuntimeConfig,
    raster_file: str,
) -> PdalCommand:
    """Build one PDAL raster-to-point pipeline and write its JSON definition."""

    if config.directory is None:
        raise ValueError("PDAL comparison commands require an explicit output directory")

    # Read every raster cell as a point with pixel-center coordinates and its band value as an attribute
    stages: list[dict[str, Any]] = [
        {
            "type": "readers.gdal",
            "filename": raster_file,
            "gdalopts": [f"GDAL_CACHEMAX={config.gdal_cachemax_mb}"],
        }
    ]
    if case.output_driver in ("LAS", "LAZ"):
        # Store the first raster band in the native elevation dimension used by the GeoUtils output
        stages.append({"type": "filters.ferry", "dimensions": "band_1=>Z"})
    if operation == "subsample":
        # Randomize the complete point view before keeping the requested fixed-size sample
        stages.extend(
            (
                {"type": "filters.randomize", "seed": 42},
                {"type": "filters.head", "count": config.value("subsample_size", 2_048)},
            )
        )

    # Write using the same point format used by GeoUtils multiprocessing
    suffix = case.output_driver.lower()
    output_file = os.path.join(config.directory, f"output-pdal-{operation}.{suffix}")
    if case.output_driver == "GPKG":
        stages.append(
            {
                "type": "writers.ogr",
                "filename": output_file,
                "ogrdriver": "GPKG",
                "attr_dims": "all",
            }
        )
    else:
        stages.append(
            {
                "type": "writers.las",
                "filename": output_file,
                "compression": case.output_driver == "LAZ",
                "extra_dims": "all",
            }
        )

    pipeline_file = os.path.join(config.directory, f"pipeline-pdal-{operation}.json")
    with open(pipeline_file, "w", encoding="utf-8") as pipeline_stream:
        json.dump({"pipeline": stages}, pipeline_stream, indent=2)

    command = [_require_command("pdal"), "pipeline", pipeline_file]
    return PdalCommand(command=command, pipeline_file=pipeline_file, output_file=output_file)


def _read_value(runner: Any, output_file: str) -> float:
    """Validate one complete PDAL point output and read one band value."""

    operation = runner.operation.name
    expected_count = (
        int(runner.config.value("subsample_size", 2_048))
        if operation == "subsample"
        else runner.config.shape[0] * runner.config.shape[1]
    )
    if pathlib.Path(output_file).suffix.lower() in (".las", ".laz"):
        count, value = read_point_file_sample(output_file, "Z")
        if count != expected_count:
            raise RuntimeError(f"Unexpected PDAL {operation} output count: {count}")
        return value

    # Feature metadata verifies the complete output without loading every point back into memory
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Measured \(M\) geometry types are not supported.*")
        info = pyogrio.read_info(output_file, force_feature_count=True)
    if info["features"] != expected_count:
        raise RuntimeError(f"Unexpected PDAL {operation} output count: {info['features']}")

    # The constant raster makes any band value sufficient regardless of randomized point order
    fields = list(info.get("fields", []))
    if not fields:
        raise RuntimeError(f"PDAL {operation} output contains no raster band attribute")
    field = fields[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Measured \(M\) geometry types are not supported.*")
        sample = pyogrio.read_dataframe(output_file, columns=[field], read_geometry=False, max_features=1)
    return float(sample[field].iloc[0])


def execute_pdal(runner: Any, case: Case) -> float:
    """Run and validate the PDAL pipeline matching the selected operation."""

    command = build_pdal_command(
        runner.operation.name,
        case,
        runner.config,
        raster_file=runner.path("source-raster.tif"),
    )
    execute_command("PDAL", command.command, command.output_file)
    runner._last_output_file = command.output_file
    value = _read_value(runner, command.output_file)
    if value != runner.config.raster_value:
        raise RuntimeError(f"Unexpected PDAL {runner.operation.name} validation value: {value}")
    return value

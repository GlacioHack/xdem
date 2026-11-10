"""Define vector rasterization and mask benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.workflows.config import (
    DEM_SIZES,
    VECTOR_COMPARISON_OPTIONS,
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
)
from benchmarks.workflows.io import write_vector_source

ORDER = 70

############################################
# Define setup for rasterization operations
############################################


def rasterize_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define rasterization or mask options."""

    options: dict[str, Any] = {
        "shape": config.shape,
        "bounds": (7.0, 45.0, 8.0, 46.0),
        "crs": 4326,
        "chunksizes": config.chunks,
    }
    if config.value("operation") == "rasterize":
        options.update({"in_value": 1, "out_value": 0, "out_dtype": np.uint8})
    return options


def prepare_rasterize(runner: Any, case: Case) -> None:
    """Prepare polygons for rasterization or masking."""

    write_vector_source(
        runner.path("source-vector.gpkg"),
        int(runner.config.value("vector_features_per_axis", 1)),
    )


def run_rasterize(runner: Any, case: Case) -> float:
    """Run rasterization or masking and ensure output computes (Dask/MP)."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    from geoutils import Vector

    # Vector input is small while the produced dem is larger than memory
    vector = Vector(runner.path("source-vector.gpkg"))
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    options = rasterize_options(case, runner.config)
    output = getattr(vector, runner.operation.name)(
        dask=runner.backend == "dask",
        mp_config=mp_config,
        **options,
    )
    return runner._compute_dem(output)


RASTERIZE = Operation(
    "rasterize",
    prepare_rasterize,
    run_rasterize,
    rasterize_options,
    label="Rasterization",
    order=13,
    large_data_cases=tuple(
        Case(engine="rasterio", execution=execution, options={"operation": "rasterize"})
        for execution in ("dask", "multiprocessing")
    ),
)
CREATE_MASK = Operation(
    "create_mask",
    prepare_rasterize,
    run_rasterize,
    rasterize_options,
    label="Mask creation",
    order=14,
    large_data_cases=tuple(
        Case(execution=execution, options={"operation": "create_mask"}) for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (RASTERIZE, CREATE_MASK)


#####################################
# Define benchmarks and comparisons
#####################################


CASES = execution_cases(
    None,
    "rasterio",
    options={**VECTOR_COMPARISON_OPTIONS, "operation": "rasterize"},
    labels={"engine": "Rasterio/GDAL"},
)
REFERENCE = reference_case(CASES, implementation="gdal")


def rasterize_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the output dem, chunks and input polygon grid."""

    config = configs[0]
    features = int(config.value("vector_features_per_axis", 1))
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} dem; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {features:,} × {features:,} vector features"
    )


BENCHMARKS = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        RASTERIZE,
        (*CASES, REFERENCE),
        dem_size_config,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=rasterize_workload,
    ),
)
COMPARISONS = (comparison(BENCHMARKS[0], by="execution"),)

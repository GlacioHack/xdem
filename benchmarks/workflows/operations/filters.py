"""Define filtering benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    DEM_CHUNK_SIZES,
    RuntimeConfig,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
)
from benchmarks.workflows.io import write_constant_dem

ORDER = 50

#######################################
# Define setup for filtering operation
#######################################


def filter_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define filter options."""

    return {"method": case.method, "engine": case.engine, "size": 5}


def prepare_filter(runner: Any, case: Case) -> None:
    """Prepare the dem to be filtered."""

    write_constant_dem(runner.path("source-raster.tif"), runner.config)


def run_filter(runner: Any, case: Case) -> float:
    """Run the filter and ensure output computes (Dask/MP)."""

    dem = runner.make_dem()
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    options = filter_options(case, runner.config)
    output = dem.dem.filter(**options) if runner.backend == "dask" else dem.filter(**options, mp_config=mp_config)
    return runner._compute_dem(output)


FILTER = Operation(
    "filter",
    prepare_filter,
    run_filter,
    filter_options,
    ("method", "engine"),
    label="Filtering",
    order=4,
    large_data_cases=tuple(
        Case(method="mean", engine="scipy", execution=execution) for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (FILTER,)

#####################################
# Define benchmarks and comparisons
#####################################


def chunk_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    assert parameter is not None
    size = int(parameter)
    return {"chunks": (size, size)}


CASES = execution_cases(
    "mean",
    "scipy",
    executions=("dask", "multiprocessing"),
    labels={"method": "Circular mean", "engine": "SciPy"},
)
BENCHMARKS = (
    parameter_config(
        "chunk_size",
        DEM_CHUNK_SIZES,
        FILTER,
        CASES,
        chunk_size,
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
    ),
)
COMPARISONS = (comparison(BENCHMARKS[0], by="execution"),)

"""Define dem point-interpolation benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.workflows.config import (
    DEFAULT_POINT_COUNT,
    INTERPOLATED_POINT_COUNTS,
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

ORDER = 20

##########################################
# Define setup for interpolation operation
##########################################


def interpolation_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define interpolation options."""

    return {"method": case.method, "as_array": True}


def prepare_interpolation(runner: Any, case: Case) -> None:
    """Prepare the dem to be interpolated."""

    write_constant_dem(runner.path("source-raster.tif"), runner.config)


def run_interpolation(runner: Any, case: Case) -> float:
    """Run interpolation and ensure output computes (Dask/MP)."""

    dem = runner.make_dem()

    # We use a uniform distribution to touch many chunks
    rng = np.random.default_rng(42)
    points = (
        rng.uniform(7.01, 7.99, size=runner.config.value("ninterp", DEFAULT_POINT_COUNT)),
        rng.uniform(45.01, 45.99, size=runner.config.value("ninterp", DEFAULT_POINT_COUNT)),
    )
    options = interpolation_options(case, runner.config)
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    values = (
        dem.dem.interp_points(points, **options)
        if runner.backend == "dask"
        else dem.interp_points(points, **options, mp_config=mp_config)
    )
    if hasattr(values, "compute"):
        values = values.compute()
    return float(np.nanmean(values))


INTERPOLATION = Operation(
    "interp_points",
    prepare_interpolation,
    run_interpolation,
    interpolation_options,
    ("method",),
    label="Point interpolation",
    order=10,
    large_data_cases=tuple(
        Case(method="linear", engine="scipy", execution=execution, options={"ninterp": DEFAULT_POINT_COUNT})
        for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (INTERPOLATION,)


def point_count(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define the number of interpolation points."""

    assert parameter is not None
    return {"ninterp": int(parameter)}


def interpolation_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the dem and interpolation point count."""

    config = configs[0]
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} dem; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {int(parameter):,} interpolated points"
    )


#####################################
# Define benchmarks and comparisons
#####################################


CASES = execution_cases(
    "linear",
    "scipy",
    labels={"method": "Linear (Delaunay)", "engine": "SciPy"},
)
BENCHMARKS = (
    parameter_config(
        "interpolated_points",
        INTERPOLATED_POINT_COUNTS,
        INTERPOLATION,
        CASES,
        point_count,
        parameter_label="Number of interpolated points",
        parameter_title="point count",
        describe_workload=interpolation_workload,
    ),
)

# The comparison selects the saved execution-mode series for one report plot; it runs no additional benchmark
COMPARISONS = (comparison(BENCHMARKS[0], by="execution", logarithmic_x=True),)

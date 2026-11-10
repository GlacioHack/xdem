"""Define random field benchmarks for GeoUtils backends and native GSTools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geoutils as gu
import numpy as np
from geoutils._misc import import_optional
from geoutils.stats.variography import VariogramModel

from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import (
    Benchmark,
    Case,
    Operation,
    comparison,
    execution_cases,
)
from benchmarks.workflows.io import write_constant_dem

ORDER = 95
FIELD_SIZES = (128, 256, 512, 1024)
FIELD_CHUNK_SIZE = 256
FIELD_EFFECTIVE_RANGE = 0.1
FIELD_SEED = 42
GPYTORCH_INDUCING_POINTS = 256


def random_field_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Describe the covariance library and shared approximation size."""

    options: dict[str, Any] = {"backend": case.engine}
    if case.engine == "gpytorch":
        options["gpytorch_inducing_points"] = GPYTORCH_INDUCING_POINTS
    return options


def prepare_random_field(runner: Any, case: Case) -> None:
    """Write one dem and prepare the same Gaussian covariance for every case."""

    write_constant_dem(runner.path("source-dem.tif"), runner.config)
    correlation = VariogramModel("gaussian", effective_range=FIELD_EFFECTIVE_RANGE, partial_sill=1)
    runner.random_field_structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])


def run_random_field(runner: Any, case: Case) -> float:
    """Draw a complete field and write it through the shared benchmark output path."""

    dem = runner.make_dem()
    if case.implementation == "native_gstools":
        # Match GeoUtils's model, component seed and pixel spacing in the native draw
        gstools = import_optional("gstools")
        model = gstools.Gaussian(dim=2, var=1, len_scale=FIELD_EFFECTIVE_RANGE, rescale=2)
        seed = int(np.random.default_rng(FIELD_SEED).integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        columns = np.arange(dem.shape[1]) * abs(float(dem.transform.a))
        rows = np.arange(dem.shape[0]) * abs(float(dem.transform.e))
        values = gstools.SRF(model, seed=seed)((columns, rows), mesh_type="structured").T
        field = gu.dem.from_array(values, dem.transform, dem.crs, nodata=dem.nodata)
    else:
        # Use the public random_field() API with identical options in every mode
        options = random_field_options(case, runner.config)
        mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
        field = gu.uncertainty.random_field(
            runner.random_field_structure,
            like=dem,
            random_state=FIELD_SEED,
            chunksizes=runner.config.chunks if runner.backend == "dask" else None,
            mp_config=mp_config,
            **options,
        )
    return runner._compute_dem(field)


RANDOM_FIELD = Operation(
    "random_field",
    prepare_random_field,
    run_random_field,
    random_field_options,
    ("backend",),
    call_name="uncertainty.random_field",
    label="Random field",
    order=95,
)
OPERATIONS = (RANDOM_FIELD,)

GSTOOLS_CASES = execution_cases("gaussian", "gstools")
GPYTORCH_CASES = execution_cases("gaussian", "gpytorch")
DIRECT_GSTOOLS = Case(
    method="gaussian",
    engine="gstools",
    execution="inmem",
    implementation="native_gstools",
    labels={"execution": "Native GSTools"},
)


def random_field_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary dem size while using the same chunk size and covariance."""

    assert parameter is not None
    size = int(parameter)
    # One worker calculates and writes the tiles through to_file()
    return {
        "shape": (size, size),
        "chunks": (FIELD_CHUNK_SIZE, FIELD_CHUNK_SIZE),
        "n_workers": 1,
        "threads_per_worker": 1,
    }


def describe_random_field_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Show the dem dimensions, tile size and worker count beside timing plots."""

    config = configs[0]
    size = int(parameter)
    return (
        f"{size:,} × {size:,} dem; {config.chunks[0]:,} × {config.chunks[1]:,} chunks; "
        f"{config.n_workers} worker, {config.threads_per_worker} thread"
    )


DEM_SIZE_BENCHMARK = Benchmark(
    RANDOM_FIELD,
    (*GSTOOLS_CASES, *GPYTORCH_CASES, DIRECT_GSTOOLS),
    random_field_size,
    parameter_name="dem_size",
    values=FIELD_SIZES,
    parameter_label="Size of dem (pixels per side)",
    parameter_title="dem size",
    describe_workload=describe_random_field_workload,
)
BENCHMARKS = (DEM_SIZE_BENCHMARK,)
COMPARISONS = (
    comparison(DEM_SIZE_BENCHMARK, by="execution", cases=(*GSTOOLS_CASES, DIRECT_GSTOOLS), slug="random-field-gstools"),
    comparison(DEM_SIZE_BENCHMARK, by="execution", cases=GPYTORCH_CASES, slug="random-field-gpytorch"),
)

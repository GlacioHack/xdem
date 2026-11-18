"""Define topo benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    DEM_CHUNK_SIZES,
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
)
from benchmarks.workflows.io import write_constant_dem

ORDER = 40

COMPLETE_CONFIG_TOPO = {
    "inputs": {
        "reference_elev": {
            "path_to_elev": "",
            "force_source_nodata": None,
            "force_vcrs": None,
            "path_to_mask": None,
            "downsample": 1,
        },
    },
    "reprojection": {"crs": None},
    "outputs": {"level": 1, "path": "outputs", "generate_pdf": True},
    "statistics": ["mean"],
    "terrain_attributes": ["slope", "aspect", "max_curvature"],
}


#########################################
# Define setup for topoion operation
#########################################


def topo_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define topoion options."""

    return {}


def prepare_topo(runner: Any, case: Case) -> None:
    """Prepare the dem to be topoed."""

    write_constant_dem(runner.path("source-dem.tif"), runner.config)
    import yaml

    print("\n", "prepare_topo", runner.config.chunks)

    config = COMPLETE_CONFIG_TOPO
    config["inputs"]["reference_elev"] = runner.path("source-dem.tif")
    with open(
        "/home/mbouchet/Documents/xDem_project/new_xdem/xdem/bm/runner_backend_" + runner.backend + ".yml", "w"
    ) as yaml_file:
        yaml.dump(COMPLETE_CONFIG_TOPO, yaml_file, sort_keys=False, allow_unicode=True)


def run_topo(runner: Any, case: Case) -> float:
    """Run topoion and ensure output computes (Dask/MP)."""

    dem = runner.make_dem()
    print(dem.shape)

    options = topo_options(case, runner.config)

    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    output = dem.dem.reproject(**options) if runner.backend == "dask" else dem.reproject(**options, mp_config=mp_config)
    return runner._compute_dem(output)


TOPO = Operation(
    "topo",
    prepare_topo,
    run_topo,
    topo_options,
    label="Topo",
    order=5,
    large_data_cases=tuple(Case(engine="rasterio", execution=execution) for execution in ("dask", "multiprocessing")),
)
OPERATIONS = (TOPO,)


#####################################
# Define benchmarks and comparisons
#####################################


CASES = execution_cases(
    None,
    "scipy",
    executions=("inmem", "dask", "multiprocessing"),
    labels={"engine": "SciPy"},
)

CASES_CHUNKS = execution_cases(
    None,
    "scipy",
    executions=("dask", "multiprocessing"),
    labels={"engine": "SciPy"},
)


def chunk_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    assert parameter is not None
    size = int(parameter)
    return {"chunks": (size, size)}


BENCHMARKS = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        TOPO,
        CASES,
        dem_size_config,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
    ),
    parameter_config(
        "chunk_size",
        DEM_CHUNK_SIZES,
        TOPO,
        CASES_CHUNKS,
        chunk_size,
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
    ),
)


COMPARISONS = (
    comparison(BENCHMARKS[0], by="execution"),
    comparison(BENCHMARKS[1], by="execution"),
)

"""Define raster reprojection benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib.util import find_spec
from typing import Any

import numpy as np
from geoutils.operators.interpolator import (
    Interpolator,
    Linear,
    Nearest,
    RasterConvolution,
)
from geoutils.operators.reducer import (
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Quantile,
    Reducer,
    RootMeanSquare,
    Sum,
)

from benchmarks.workflows.config import (
    RASTER_SIZES,
    RuntimeConfig,
    raster_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
    reference_case,
)
from benchmarks.workflows.io import write_constant_raster

ORDER = 40

# These operators implement the same statistic or interpolation kernel as the named GDAL warp method
GDAL_OPERATOR_METHODS: dict[str, Callable[[], Interpolator | Reducer]] = {
    "nearest": Nearest,
    "bilinear": Linear,
}
OPERATOR_CALLS = {
    "nearest": "Nearest()",
    "bilinear": "Linear()",
}


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

    options: dict[str, Any] = {
        "crs": 32632,
        "grid_size": config.shape[::-1],
        "nodata": -99999,
        "n_threads": 1,
        "memory_limit": 64,
    }
    if case.engine == "operator":
        assert case.method is not None
        operator = GDAL_OPERATOR_METHODS[case.method]()
        options["resampling"] = operator
        if isinstance(operator, Reducer):
            options["coverage"] = "fractional"
    else:
        options["resampling"] = case.method
    return options


def prepare_topo(runner: Any, case: Case) -> None:
    """Prepare the raster to be topoed."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)
    import yaml

    config = COMPLETE_CONFIG_TOPO
    config["inputs"]["reference_elev"] = runner.path("source-raster.tif")
    with open(
        "/home/mbouchet/Documents/xDem_project/new_xdem/xdem/bm/runner_backend_" + runner.backend + ".yml", "w"
    ) as yaml_file:
        yaml.dump(COMPLETE_CONFIG_TOPO, yaml_file, sort_keys=False, allow_unicode=True)


def run_topo(runner: Any, case: Case) -> float:
    """Run topoion and ensure output computes (Dask/MP)."""

    raster = runner.make_raster()
    options = topo_options(case, runner.config)

    # Fix the target size so GeoUtils and GDAL references write the same pixel count
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    output = (
        raster.rst.reproject(**options)
        if runner.backend == "dask"
        else raster.reproject(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output)


TOPO = Operation(
    "topo",
    prepare_topo,
    run_topo,
    topo_options,
    ("resampling",),
    label="Topo",
    order=5,
    large_data_cases=tuple(
        Case(method="nearest", engine="rasterio", execution=execution) for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (TOPO,)


#####################################
# Define benchmarks and comparisons
#####################################


CASES = execution_cases(
    "nearest",
    "rasterio",
    executions=("inmem", "dask", "multiprocessing"),
    labels={"method": "Nearest", "engine": "Rasterio/GDAL"},
)
REFERENCE = reference_case(CASES, implementation="geoutils")

OPERATOR_CASES = tuple(
    Case(
        method=method,
        engine="operator",
        execution="inmem",
        labels={
            "engine": "GeoUtils operator",
            "api_resampling": OPERATOR_CALLS[method],
        },
    )
    for method in GDAL_OPERATOR_METHODS
)
OPERATOR_REFERENCES = tuple(
    replace(
        reference_case(case, implementation="gdal"),
        variant="operator",
        labels={"engine": "GDAL CLI"},
    )
    for case in OPERATOR_CASES
)

BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        TOPO,
        (*CASES, REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
)
COMPARISONS = (comparison(BENCHMARKS[0], by="execution"),)

"""    comparison(BENCHMARKS[0], by="execution"),
*(
    comparison(
        BENCHMARKS[1],
        by="engine",
        cases=(operator, reference),
        slug=f"topo-operator-{operator.method}",
        documentation=False,
        summary=False,
    )
    for operator, reference in zip(OPERATOR_CASES, OPERATOR_REFERENCES)
),"""

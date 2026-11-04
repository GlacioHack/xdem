"""Define raster topo benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib.util import find_spec
from typing import Any

import numpy as np
import yaml

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
from geoutils.operators.interpolator import Interpolator, Linear, Nearest, RasterConvolution
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


ORDER = 40


#########################################
# Define setup for topo operation
#########################################


def topo_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define topo options."""

    options: dict[str, Any] = {
        "crs": 32632,
        "grid_size": config.shape[::-1],
        "nodata": -99999,
        "n_threads": 1,
        "memory_limit": 64,
    }
    options["resampling"] = "nearest"
    return options


def prepare_topo(runner: Any, case: Case) -> None:
    """Prepare the raster to be topoed."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)

def run_topo(runner: Any, case: Case) -> float:
    """Run topo and ensure output computes (Dask/MP)."""

    with open("/home/mbouchet/Documents/xDem_project/geoutils/runner_backend.yml", "w") as yaml_file:
        yaml.dump(COMPLETE_CONFIG_TOPO, yaml_file, sort_keys=False, allow_unicode=True)
        print ("ecriture ok")


    raster = runner.make_raster()
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None

    options: dict[str, Any] = {
        "crs": 32632,
        "nodata": -99999,
        "n_threads": 1,
        "memory_limit": 64,
    }

    output = (
        raster.rst.reproject(**options)
        if runner.backend == "dask"
        else raster.reproject(mp_config=mp_config)
    )
    return runner._compute_raster(output)



REPROJECT = Operation(
    "topo",
    prepare_topo,
    run_topo,
    topo_options,
    ("resampling",),
    label="topo",
    order=5,
    large_data_cases=tuple(
        Case(method="nearest", engine="rasterio", execution=execution) for execution in ("in-memory", "dask", "multiprocessing")
    ),
)
OPERATIONS = (REPROJECT,)


#####################################
# Define benchmarks and comparisons
#####################################

CASES = execution_cases(
    "nearest",
    "rasterio",
    labels={"method": "Nearest", "engine": "Rasterio/GDAL"},
)
REFERENCE = reference_case(CASES, implementation="gdal")
# These operators implement the same statistic or interpolation kernel as the named GDAL warp method
GDAL_OPERATOR_METHODS: dict[str, Callable[[], Interpolator | Reducer]] = {
    "nearest": Nearest
)
OPERATOR_CASES = tuple(
    Case(
        method=method,
        engine="operator",
        execution="inmem",
        labels={
            "engine": "GeoUtils operator",
            "api_resampling": "nearest!",
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
        REPROJECT,
        (*CASES, REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    )
)

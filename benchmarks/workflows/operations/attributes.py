"""Define terrain attributes benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import rasterio as rio

from benchmarks.workflows.config import DEM_SIZES, RuntimeConfig, dem_size_config
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
)
from benchmarks.workflows.io import write_constant_dem

ORDER = 40
TERRAIN_ATTRIBUTES = ("slope", "aspect")
SURFACE_FITS = ("Horn", "Florinsky")  #  "ZevenbergThorne",


def attributes_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define terrain attribute calculation options."""

    options = {"attribute": case.terrain_attribute, "engine": case.engine}
    if case.surface_fit is not None:
        options["surface_fit"] = case.surface_fit
    return options


def prepare_attributes(runner: Any, case: Case) -> None:
    """Prepare the DEM."""

    filename = runner.path("source-dem.tif")
    write_constant_dem(filename, runner.config)


def run_attributes(runner: Any, case: Case) -> float:
    """Compute terrain attribute(s) and write the complete output."""

    dem = runner.make_dem()
    options = attributes_options(case, runner.config)
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    output = (
        dem.dem.get_terrain_attribute(**options)
        if runner.backend == "dask"
        else dem.get_terrain_attribute(**options, mp_config=mp_config)
    )
    return runner._compute_dem(output)


ATTRIBUTES = Operation(
    "attributes",
    prepare_attributes,
    run_attributes,
    attributes_options,
    ("attribute", "engine", "surface_fit"),
    call_name=".get_terrain_attribute",
    label="Terrain attributes",
)
OPERATIONS = (ATTRIBUTES,)

CASES = tuple(
    case
    for attribute in TERRAIN_ATTRIBUTES
    for case in execution_cases(
        None,
        "scipy",
        terrain_attribute=attribute,
        labels={"terrain_attribute": attribute.title(), "engine": "SciPy"},
    )
)

SURFACE_FIT_CASES = tuple(
    case
    for attribute in TERRAIN_ATTRIBUTES
    for surface_fit in SURFACE_FITS
    for case in execution_cases(
        None,
        "scipy",
        terrain_attribute=attribute,
        surface_fit=surface_fit,
        labels={"terrain_attribute": attribute.title(), "engine": "SciPy", "surface_fit": surface_fit},
    )
)

NUMBA_CASES = tuple(
    case
    for attribute in TERRAIN_ATTRIBUTES
    for case in execution_cases(
        None,
        "numba",
        terrain_attribute=attribute,
        surface_fit="Florinsky",
        labels={"terrain_attribute": attribute.title(), "engine": "Numba", "surface_fit": "Florinsky"},
    )
)

BENCHMARKS = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        ATTRIBUTES,
        CASES,
        dem_size_config,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="DEM size",
    ),
    parameter_config(
        "dem_size",
        DEM_SIZES,
        ATTRIBUTES,
        (*SURFACE_FIT_CASES, *NUMBA_CASES),
        dem_size_config,
        name="attributes-surface-fit",
        parameter_label="Size of dem (pixels per side)",
        parameter_title="DEM size",
    ),
)

COMPARISONS = (
    *(
        comparison(
            BENCHMARKS[0],
            by="execution",
            cases=tuple(case for case in CASES if case.terrain_attribute == attribute),
            slug=f"attributes-{attribute}-dem-size",
        )
        for attribute in TERRAIN_ATTRIBUTES
    ),
    *(
        comparison(
            BENCHMARKS[0],
            by="terrain_attribute",
            cases=tuple(case for case in CASES if case.execution == execution),
            slug=f"attributes-{execution}-dem-size",
        )
        for execution in ("inmem", "dask", "multiprocessing")
    ),
    *(
        comparison(
            BENCHMARKS[1],
            by="surface_fit",
            cases=tuple(
                case
                for case in SURFACE_FIT_CASES
                if case.terrain_attribute == attribute and case.execution == execution
            ),
            slug=f"attributes-{attribute}-{execution}-surface-fit",
            summary=False,
        )
        for attribute in TERRAIN_ATTRIBUTES
        for execution in ("inmem", "dask", "multiprocessing")
    ),
    *(
        comparison(
            BENCHMARKS[1],
            by="engine",
            cases=tuple(
                case
                for case in (*SURFACE_FIT_CASES, *NUMBA_CASES)
                if case.terrain_attribute == attribute
                and case.surface_fit == "Florinsky"
                and case.execution == execution
            ),
            slug=f"attributes-{attribute}-florinsky-{execution}-engine",
            summary=False,
        )
        for attribute in TERRAIN_ATTRIBUTES
        for execution in ("inmem", "dask", "multiprocessing")
    ),
)

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
    "cubic": lambda: RasterConvolution("cubic"),
    "cubic_spline": lambda: RasterConvolution("cubic_spline"),
    "lanczos": lambda: RasterConvolution("lanczos"),
    "average": Mean,
    "sum": Sum,
    "min": Minimum,
    "max": Maximum,
    "rms": RootMeanSquare,
    "mode": lambda: Mode(weighted=False, tie_break="first_to_mode"),
    "med": lambda: Median(method="inverted_cdf"),
    "q1": lambda: Quantile(0.25, method="inverted_cdf"),
    "q3": lambda: Quantile(0.75, method="inverted_cdf"),
}
OPERATOR_CALLS = {
    "nearest": "Nearest()",
    "bilinear": "Linear()",
    "cubic": "RasterConvolution('cubic')",
    "cubic_spline": "RasterConvolution('cubic_spline')",
    "lanczos": "RasterConvolution('lanczos')",
    "average": "Mean()",
    "sum": "Sum()",
    "min": "Minimum()",
    "max": "Maximum()",
    "rms": "RootMeanSquare()",
    "mode": "Mode(weighted=False, tie_break='first_to_mode')",
    "med": "Median(method='inverted_cdf')",
    "q1": "Quantile(0.25, method='inverted_cdf')",
    "q3": "Quantile(0.75, method='inverted_cdf')",
}

#########################################
# Define setup for reprojection operation
#########################################


def reproject_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define reprojection options."""

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


def prepare_reproject(runner: Any, case: Case) -> None:
    """Prepare the raster to be reprojected."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)
    if case.engine == "operator" and find_spec("numba") is not None:
        if case.method in ("cubic", "cubic_spline", "lanczos"):
            # Compile the selected raster kernel outside the measured operation
            RasterConvolution(case.method)._interpolate_grid(np.ones((1, 1)), np.array([0.0]), np.array([0.0]), "gdal")
        elif case.method not in ("nearest", "bilinear"):
            # Compile polygon clipping outside the measured operation
            from geoutils.operators.overlap import _pixel_quadrilateral_intersections

            unit_columns = np.array([[0.0, 1.0, 1.0, 0.0]])
            unit_rows = np.array([[0.0, 0.0, 1.0, 1.0]])
            _pixel_quadrilateral_intersections(unit_columns, unit_rows, 1, 1)


def run_reproject(runner: Any, case: Case) -> float:
    """Run reprojection and ensure output computes (Dask/MP)."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    raster = runner.make_raster()
    options = reproject_options(case, runner.config)

    # Fix the target size so GeoUtils and GDAL references write the same pixel count
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    output = (
        raster.rst.reproject(**options)
        if runner.backend == "dask"
        else raster.reproject(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output)


REPROJECT = Operation(
    "reproject",
    prepare_reproject,
    run_reproject,
    reproject_options,
    ("resampling",),
    label="Reprojection",
    order=5,
    large_data_cases=tuple(
        Case(method="nearest", engine="rasterio", execution=execution) for execution in ("dask", "multiprocessing")
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
        REPROJECT,
        (*CASES, REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
    parameter_config(
        "raster_size",
        (128, 256),
        REPROJECT,
        (*OPERATOR_CASES, *OPERATOR_REFERENCES),
        raster_size_config,
        name="reproject-operator",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
)
COMPARISONS = (
    comparison(BENCHMARKS[0], by="execution"),
    *(
        comparison(
            BENCHMARKS[1],
            by="engine",
            cases=(operator, reference),
            slug=f"reproject-operator-{operator.method}",
            documentation=False,
            summary=False,
        )
        for operator, reference in zip(OPERATOR_CASES, OPERATOR_REFERENCES)
    ),
)

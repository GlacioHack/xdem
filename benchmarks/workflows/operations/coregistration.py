"""Benchmark DEM coregistration methods."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from benchmarks.workflows.config import DEM_SIZES, RuntimeConfig, dem_size_config
from benchmarks.workflows.core import Case, Operation, comparison, execution_cases, parameter_config
from benchmarks.workflows.io import tiff_block_size, write_constant_dem
from xdem.coreg import CPD, ICP, LZD, DhMinimize, NuthKaab, VerticalShift
from xdem.coreg.base import Coreg

ORDER = 60

COREG_METHODS: dict[str, type[Coreg]] = {
    "vertical_shift": VerticalShift,
    "nuth_kaab": NuthKaab,
    # "lzd": LZD,
    # "dh_minimize": DhMinimize,
    # "icp": ICP,
    # "cpd": CPD,
}
SAMPLING_STRATEGIES = (
    "same_xy",
    "nearest_xy",
    "independent",
)


def coregistration_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    if case.coreg_method not in COREG_METHODS:
        raise ValueError(f"Unknown coregistration method: {case.coreg_method!r}")
    return {"coreg_method": COREG_METHODS[case.coreg_method](), "random_state": 42}


def prepare_coregistration(runner: Any, case: Case) -> None:
    filename = runner.path("ref-dem.tif")
    write_constant_dem(filename, runner.config)
    filename = runner.path("tba-dem.tif")
    write_constant_dem(filename, runner.config)


def load_coregistration_inputs(runner: Any) -> tuple[Any, Any]:
    ref = runner.make_dem(
        "/home/mbouchet/Documents/xDem_project/new_xdem/xdem/xdem/example_data/Longyearbyen/data/DEM_2009_ref_test.tif"
    )
    tba = runner.make_dem(
        "/home/mbouchet/Documents/xDem_project/new_xdem/xdem/xdem/example_data/Longyearbyen/data/DEM_1990_test.tif"
    )
    return (ref.dem, tba.dem) if runner.backend == "dask" else (ref, tba)


def run_coreg_fit(runner: Any, case: Case) -> float:
    ref, tba = load_coregistration_inputs(runner)
    options = coregistration_options(case, runner.config)
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    coreg = options["coreg_method"]
    if runner.backend == "dask":
        fitted = coreg.fit(ref, tba, random_state=options["random_state"])
    else:
        fitted = coreg.fit(ref, tba, random_state=options["random_state"], mp_config=mp_config)
    return float(fitted.to_matrix()[2, 3])


def prepare_coreg_apply(runner: Any, case: Case) -> None:
    prepare_coregistration(runner, case)
    ref, tba = load_coregistration_inputs(runner)
    options = coregistration_options(case, runner.config)
    runner.coreg_fitted = options["coreg_method"].fit(ref, tba, random_state=options["random_state"])


def run_coreg_apply(runner: Any, case: Case) -> float:
    """Apply the prepared transform and materialize the resulting DEM."""
    _, tba = load_coregistration_inputs(runner)
    return runner._compute_dem(runner.coreg_fitted.apply(tba))


COREG_FIT = Operation(
    "coreg_fit",
    prepare_coregistration,
    run_coreg_fit,
    coregistration_options,
    ("coreg_method",),
    call_name=".fit",
    label="Coregistration fit",
)
COREG_APPLY = Operation(
    "coreg_apply",
    prepare_coreg_apply,
    run_coreg_apply,
    coregistration_options,
    ("coreg_method",),
    call_name=".apply",
    label="Coregistration apply",
)
OPERATIONS = (COREG_FIT, COREG_APPLY)

CASES = tuple(
    case
    for method, coreg_class in COREG_METHODS.items()
    for case in execution_cases(
        None,
        None,
        executions=("inmem", "dask"),
        coreg_method=method,
        labels={
            "coreg_method": coreg_class.__name__,
        },
    )
)

BENCHMARKS = (
    *(
        parameter_config(
            "dem_size",
            DEM_SIZES,
            operation,
            CASES,
            dem_size_config,
            parameter_label="Size of dem (pixels per side)",
            parameter_title="DEM size",
        )
        for operation in OPERATIONS
    ),
)

COMPARISONS = (
    *(
        comparison(
            benchmark,
            by="coreg_method",
            cases=tuple(case for case in CASES if case.execution == execution),
            slug=f"{benchmark.operation.name.replace('_', '-')}-{execution}-methods-dem-size",
        )
        for benchmark in BENCHMARKS
        for execution in ("inmem", "dask")
    ),
    *(
        comparison(
            benchmark,
            by="execution",
            cases=tuple(case for case in CASES if case.coreg_method == method),
            slug=f"{benchmark.operation.name.replace('_', '-')}-{method.replace('_', '-')}-dem-size",
        )
        for benchmark in BENCHMARKS
        for method in COREG_METHODS
    ),
)

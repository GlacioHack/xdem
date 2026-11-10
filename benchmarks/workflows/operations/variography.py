"""Define public variogram benchmarks and their deterministic inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import geoutils as gu
import numpy as np
import xarray as xr
from geoutils._misc import import_optional
from geoutils.stats import Variogram, variogram
from rasterio.transform import from_origin

from benchmarks.workflows.config import (
    DEM_SIZES,
    POINT_COUNTS,
    VARIOGRAM_DEM_CHUNK_SIZE,
    VARIOGRAM_DEM_SIZE,
    VARIOGRAM_LAG_COUNTS,
    VARIOGRAM_LAG_PAIRS,
    VARIOGRAM_N_LAGS,
    VARIOGRAM_PAIR_COUNTS,
    VARIOGRAM_POINT_PAIRS,
    VARIOGRAM_SAMPLE_PAIRS,
    RuntimeConfig,
)
from benchmarks.workflows.core import (
    Benchmark,
    Case,
    ExecutionMode,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
)
from xdem import DEM

ORDER = 100


###########################################
# Define setup for variography operations
###########################################


def prepare_variogram_pairs(n_pairs: int) -> xr.Dataset:
    """Prepare endpoint values at log-uniform distances."""

    # Spread observations across short and long distances without enumerating a spatial distance matrix
    rng = np.random.default_rng(42)
    distances = np.exp(rng.uniform(0, np.log(1024), n_pairs))
    first = rng.normal(size=n_pairs)
    differences = rng.normal(size=n_pairs) * np.sqrt(1 - np.exp(-distances / 100))

    # Keep only the public pair layout consumed by Variogram.from_pairs()
    return xr.Dataset(
        {
            "distance": ("pair", distances),
            "value": (("pair", "endpoint"), np.column_stack((first, first + differences))),
        },
        attrs={"min_distance": 1.0, "max_distance": 1024.0},
    )


def prepare_pair_dem(
    size: int,
    execution_mode: Literal["inmem", "dask"],
    chunks: tuple[int, int],
) -> Any:
    """Prepare a dem with missing values and optional Dask chunks."""

    # Vary values in both directions and remove a known fraction of cells to require finite endpoint checks
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    values = (np.sin(columns / 31) + np.cos(rows / 53)).astype(np.float32)
    values[(rows * size + columns) % 17 == 0] = np.nan
    transform = from_origin(0, size, 1, 1)

    # Expose the public object method in both modes while leaving lazy values uncomputed
    if execution_mode == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        array = da.from_array(values, chunks=chunks)
        return gu.DataArrayRasterAccessor.from_array(array, transform, 32633, nodata=-99999).rst
    return xdem.DEM.from_array(values, transform, 32633, nodata=-99999)


def prepare_pair_pointcloud(n_points: int) -> gu.PointCloud:
    """Prepare an irregular point cloud with smoothly varying values."""

    # Keep average point density constant so increasing point count increases the search extent
    rng = np.random.default_rng(42)
    coordinates = rng.uniform(0, np.sqrt(n_points), size=(n_points, 2))
    x, y = coordinates.T
    values = np.sin(x / 31) + np.cos(y / 53)
    return gu.PointCloud.from_xyz(x, y, values, crs=32633)


def prepare_estimator(estimator: str) -> None:
    """Prepare the optional variogram estimator."""

    # ASV records an unavailable optional package as a skipped case, and compilation stays outside timing
    try:
        import_optional("skgstat", package_name="scikit-gstat")
    except ImportError as exc:
        raise NotImplementedError("Install geoutils[opt] to measure variogram estimators") from exc
    Variogram.from_pairs(prepare_variogram_pairs(64), estimator=estimator, n_lags=4)


def variogram_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define variogram options."""

    options = {"estimator": case.method}
    for name in ("reduction_backend", "flox_engine", "method", "tdigest_compression"):
        if name in case.options:
            options[name] = case.options[name]
    return options


def pair_sampling_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define pair sampling options."""

    if case.method == "random_xy":
        return {"sampling": "random_xy", "strategy": "chunk_anchors"}
    options = {"sampling": "loglag", "strategy": case.method}
    if "read_strategy" in case.options:
        options["read_strategy"] = case.options["read_strategy"]
    return options


def prepare_dem_variogram(runner: Any, case: Case) -> None:
    """Prepare the dem and variogram estimator."""

    # Keep the same values and sampling request while changing the source size and loading mode
    assert case.method is not None
    runner.variography_source = prepare_pair_dem(runner.config.shape[0], runner.backend, runner.config.chunks)
    prepare_estimator(case.method)


def run_dem_variogram(runner: Any, case: Case) -> float:
    """Run the dem variogram."""

    assert case.method is not None
    result = variogram(
        runner.variography_source,
        **variogram_options(case, runner.config),
        output_chunks=int(runner.config.value("pair_chunk_size", 10_007)),
        n_lags=int(runner.config.value("n_lags", VARIOGRAM_N_LAGS)),
        n_pairs=int(runner.config.value("n_pairs", VARIOGRAM_SAMPLE_PAIRS)),
        sampling="loglag",
        strategy="chunk_anchors",
        min_distance=1,
        max_distance=runner.config.shape[0] / 2,
        batch_pairs=100_000,
        anchors_per_round=2_000,
        random_state=42,
    )
    return float(np.sum(result.counts))


def prepare_pair_reduction(runner: Any, case: Case) -> None:
    """Prepare variogram pairs and the estimator."""

    assert case.method is not None
    prepare_estimator(case.method)
    if case.options.get("reduction_backend") == "flox":
        try:
            import_optional("flox", extra_name="num")
            if case.method in {"dowd", "percentile"} and case.options.get("method") != "groupwise":
                import_optional("crick", extra_name="num")
        except ImportError as exc:
            raise NotImplementedError("Install geoutils[num] for this variogram comparison") from exc
    runner.variography_pairs = prepare_variogram_pairs(int(runner.config.value("n_pairs")))
    if runner.backend == "dask":
        # This measures reduction of identical prepared pairs, independently of spatial sampling
        runner.variography_pairs = runner.variography_pairs.chunk(
            {"pair": int(runner.config.value("pair_chunk_size", 10_007))}
        )

    # Warm the local engine outside the ASV operation timing
    Variogram.from_pairs(prepare_variogram_pairs(64), **variogram_options(case, runner.config), n_lags=4)


def run_pair_reduction(runner: Any, case: Case) -> float:
    """Run variogram reduction on prepared pairs."""

    assert case.method is not None
    result = Variogram.from_pairs(
        runner.variography_pairs,
        **variogram_options(case, runner.config),
        n_lags=int(runner.config.value("n_lags", VARIOGRAM_N_LAGS)),
    )
    return float(np.sum(result.counts))


def prepare_dem_pair_sampling(runner: Any, case: Case) -> None:
    """Prepare the dem for pair sampling."""

    # Keep the same source values and worker count for every sampling method
    runner.variography_source = prepare_pair_dem(runner.config.shape[0], runner.backend, runner.config.chunks)


def run_dem_pair_sampling(runner: Any, case: Case) -> float:
    """Run dem pair sampling and ensure output computes (Dask)."""

    options = pair_sampling_options(case, runner.config)

    # Bound candidate batches and reuse the same map-distance range and random seed
    pairs = runner.variography_source.pairsample(
        n_pairs=int(runner.config.value("n_pairs", VARIOGRAM_SAMPLE_PAIRS)),
        **options,
        min_distance=1,
        max_distance=runner.config.shape[0] / 2,
        batch_pairs=100_000,
        anchors_per_round=2_000,
        random_state=42,
    )
    pairs.load()
    return float(pairs.sizes["pair"])


def prepare_point_pair_sampling(runner: Any, case: Case) -> None:
    """Prepare the point cloud for pair sampling."""

    runner.variography_source = prepare_pair_pointcloud(int(runner.config.value("point_count")))


def run_point_pair_sampling(runner: Any, case: Case) -> float:
    """Run point cloud pair sampling."""

    assert case.method is not None
    n_points = int(runner.config.value("point_count"))
    pairs = runner.variography_source.pairsample(
        n_pairs=int(runner.config.value("n_pairs", VARIOGRAM_POINT_PAIRS)),
        strategy=case.method,
        min_distance=1,
        max_distance=n_points**0.5 / 2,
        anchors_per_round=2_000,
        nn_tolerance=0.5,
        random_state=42,
    )
    pairs.load()
    return float(pairs.sizes["pair"])


DEM_VARIOGRAM = Operation(
    "dem_variogram",
    prepare_dem_variogram,
    run_dem_variogram,
    variogram_options,
    ("estimator",),
    call_name="variogram",
    label="DEM variogram",
    benchmark_name="variogram",
)
PAIR_REDUCTION = Operation(
    "variogram_from_pairs",
    prepare_pair_reduction,
    run_pair_reduction,
    variogram_options,
    ("estimator",),
    call_name="Variogram.from_pairs",
    label="Variogram pair reduction",
    benchmark_name="variogrampairs",
)
DEM_PAIR_SAMPLING = Operation(
    "dem_pairsample",
    prepare_dem_pair_sampling,
    run_dem_pair_sampling,
    pair_sampling_options,
    ("sampling", "strategy", "read_strategy"),
    call_name=".pairsample",
    label="Dem pair sampling",
    benchmark_name="dempairsample",
)
POINT_PAIR_SAMPLING = Operation(
    "point_pairsample",
    prepare_point_pair_sampling,
    run_point_pair_sampling,
    pair_sampling_options,
    ("strategy",),
    call_name=".pairsample",
    label="Point pair sampling",
    benchmark_name="pointpairsample",
)
OPERATIONS = (DEM_VARIOGRAM, PAIR_REDUCTION, DEM_PAIR_SAMPLING, POINT_PAIR_SAMPLING)


###################
# Define benchmarks
###################


DEM_VARIOGRAM_CASES = execution_cases(
    "dowd",
    None,
    executions=("inmem", "dask"),
    labels={"method": "Dowd", "engine": "xdem"},
)
PAIR_EXECUTIONS: tuple[ExecutionMode, ...] = ("inmem", "dask")
PAIR_REDUCTION_CASES = (
    tuple(
        Case(
            method=estimator,
            execution=execution,
            options={"method": "groupwise" if estimator == "dowd" else "map-reduce"},
            labels={"method": estimator.title(), "engine": "xdem"},
        )
        for estimator in ("matheron", "cressie", "dowd")
        for execution in PAIR_EXECUTIONS
    )
    + tuple(
        Case(
            method=estimator,
            engine=engine,
            execution=execution,
            options={"reduction_backend": "flox", "flox_engine": "numba" if engine == "flox_numba" else None},
            labels={"method": estimator.title(), "engine": "Flox Numba" if engine == "flox_numba" else "Flox default"},
        )
        for estimator in ("matheron", "cressie")
        for engine in ("flox_default", "flox_numba")
        for execution in PAIR_EXECUTIONS
    )
    + tuple(
        Case(
            method="dowd",
            engine="tdigest",
            execution=execution,
            options={"reduction_backend": "flox", "method": "map-reduce", "tdigest_compression": 200},
            labels={"method": "Dowd", "engine": "Flox t-digest"},
        )
        for execution in PAIR_EXECUTIONS
    )
)
DEM_VARIOGRAM_CASES += tuple(case for case in PAIR_REDUCTION_CASES if case.method != "dowd" or case.engine == "tdigest")
DEM_PAIR_METHODS = ("independent", "anchors", "chunk_anchors", "anchor_batched", "random_xy")
DEM_PAIR_CASES = tuple(
    Case(method=method, execution=execution) for method in DEM_PAIR_METHODS for execution in PAIR_EXECUTIONS
)
DEM_PAIR_SIZE_CASES = execution_cases(
    "chunk_anchors",
    None,
    executions=("inmem", "dask"),
)
DEM_PAIR_READ_CASES = tuple(
    Case(
        method="chunk_anchors",
        engine=read_strategy,
        execution=execution,
        options={"read_strategy": read_strategy},
        labels={"engine": read_strategy.replace("_", " ").title()},
    )
    for read_strategy in ("reuse_values", "packed_validity")
    for execution in PAIR_EXECUTIONS
)
DEM_PAIR_CASES += DEM_PAIR_READ_CASES
DEM_PAIR_SIZE_CASES += DEM_PAIR_READ_CASES
POINT_PAIR_CASES = tuple(Case(method=strategy) for strategy in ("kdtree", "hashgrid", "nn_logvector"))


def dem_variogram_size_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary dem size while fixing chunks, sampled pairs and distance bins."""

    assert parameter is not None
    size = int(parameter)
    return {
        "shape": (size, size),
        "chunks": (VARIOGRAM_DEM_CHUNK_SIZE, VARIOGRAM_DEM_CHUNK_SIZE),
        "n_pairs": VARIOGRAM_SAMPLE_PAIRS,
        "n_lags": VARIOGRAM_N_LAGS,
    }


def variogram_pair_count_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary complete input pairs while fixing the number of distance bins."""

    assert parameter is not None
    return {"n_pairs": int(parameter), "n_lags": VARIOGRAM_N_LAGS}


def variogram_lag_count_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary distance bins while keeping the pair sample fixed."""

    assert parameter is not None
    return {"n_pairs": VARIOGRAM_LAG_PAIRS, "n_lags": int(parameter)}


def DEM_pair_count_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary sampled pairs on one fixed DEM layout."""

    assert parameter is not None
    return {
        "shape": (VARIOGRAM_DEM_SIZE, VARIOGRAM_DEM_SIZE),
        "chunks": (VARIOGRAM_DEM_CHUNK_SIZE, VARIOGRAM_DEM_CHUNK_SIZE),
        "n_pairs": int(parameter),
    }


def DEM_pair_size_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary DEM size while keeping the requested pair count fixed."""

    assert parameter is not None
    size = int(parameter)
    return {
        "shape": (size, size),
        "chunks": (VARIOGRAM_DEM_CHUNK_SIZE, VARIOGRAM_DEM_CHUNK_SIZE),
        "n_pairs": VARIOGRAM_SAMPLE_PAIRS,
    }


def point_pair_count_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary point count while keeping the requested pair count fixed."""

    assert parameter is not None
    return {"point_count": int(parameter), "n_pairs": VARIOGRAM_POINT_PAIRS}


BENCHMARKS: tuple[Benchmark, ...] = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        DEM_VARIOGRAM,
        DEM_VARIOGRAM_CASES,
        dem_variogram_size_config,
        parameter_label="Size of DEM (pixels per side)",
        parameter_title="DEM size",
    ),
    parameter_config(
        "n_pairs",
        VARIOGRAM_PAIR_COUNTS,
        PAIR_REDUCTION,
        PAIR_REDUCTION_CASES,
        variogram_pair_count_config,
        name="variogram-pair-count",
        parameter_label="Number of sampled pairs",
        parameter_title="pair count",
    ),
    parameter_config(
        "n_lags",
        VARIOGRAM_LAG_COUNTS,
        PAIR_REDUCTION,
        PAIR_REDUCTION_CASES,
        variogram_lag_count_config,
        name="variogram-lag-count",
        parameter_label="Number of distance bins",
        parameter_title="distance-bin count",
    ),
    parameter_config(
        "n_pairs",
        POINT_COUNTS,
        DEM_PAIR_SAMPLING,
        DEM_PAIR_CASES,
        DEM_pair_count_config,
        name="DEM-pair-count",
        parameter_label="Number of sampled pairs",
        parameter_title="pair count",
    ),
    parameter_config(
        "dem_size",
        DEM_SIZES,
        DEM_PAIR_SAMPLING,
        DEM_PAIR_SIZE_CASES,
        DEM_pair_size_config,
        name="DEM-pair-size",
        parameter_label="Size of DEM (pixels per side)",
        parameter_title="DEM size",
    ),
    parameter_config(
        "n_points",
        POINT_COUNTS,
        POINT_PAIR_SAMPLING,
        POINT_PAIR_CASES,
        point_pair_count_config,
        parameter_label="Number of input points",
        parameter_title="point count",
    ),
)


#############################
# Define report comparisons
#############################

COMPARISONS = tuple(
    comparison(
        benchmark,
        by="engine",
        cases=tuple(case for case in benchmark.cases if case.method == estimator and case.execution == execution),
        slug=f"{benchmark.id}-{estimator}-{execution}",
        documentation=False,
    )
    for benchmark in BENCHMARKS[:3]
    for estimator in ("matheron", "cressie", "dowd")
    for execution in ("inmem", "dask")
)
COMPARISONS += tuple(
    comparison(
        benchmark,
        by="engine",
        cases=tuple(case for case in benchmark.cases if case.method == "chunk_anchors" and case.execution == execution),
        slug=f"{benchmark.id}-reads-{execution}",
        documentation=False,
    )
    for benchmark in BENCHMARKS[3:5]
    for execution in PAIR_EXECUTIONS
)

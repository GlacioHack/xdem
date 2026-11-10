"""Define dem and grouped-statistics benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from geoutils._misc import import_optional
from geoutils.stats.reduction import (
    _normalize_statistics,
    _reduce_values,
)

from benchmarks.asv_suite import asv_parameter_values
from benchmarks.comparisons.dask import global_statistics
from benchmarks.comparisons.flox import compute_geoutils_grouped_stats
from benchmarks.comparisons.flox import grouped_stats as flox_grouped_stats
from benchmarks.comparisons.flox import prepare_grouped_inputs
from benchmarks.workflows.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEM_SIZE,
    DEM_CHUNK_SIZES,
    DEM_SIZES,
    GROUP_COUNTS,
    RuntimeConfig,
    dem_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    merge_cases,
    parameter_config,
    reference_case,
    strategy_cases,
)
from benchmarks.workflows.io import write_constant_dem

ORDER = 10
REDUCTION_OPTIONS = {
    "auto": {"method": "auto", "reindex": None},
    "dense": {"method": "map-reduce", "reindex": True},
    "sparse": {"method": "map-reduce", "reindex": False},
    "groupwise": {"method": "groupwise", "reindex": None},
}
STRATEGY_LABELS = {
    "auto": "Automatic",
    "dense": "Dense summaries",
    "sparse": "Sparse summaries",
    "groupwise": "Complete groups",
}

##########################################
# Define setup for statistics operations
##########################################


def statistics_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define statistics options."""

    if config.value("operation") == "statistics":
        return {"statistics": ["mean", "std", "valid count"]}
    statistics = ["mean", "std", "min", "max"] if case.method == "moments" else ["median", "nmad"]
    return {"statistics": statistics, **REDUCTION_OPTIONS[case.strategy or "auto"]}


def prepare_statistics(runner: Any, case: Case) -> None:
    """Prepare inputs for statistics."""

    if case.variant == "flox" or case.implementation == "flox":
        if case.implementation == "flox":
            try:
                import_optional("flox")
            except ImportError as exc:
                raise NotImplementedError("Install optional flox to run this comparison") from exc
        runner.grouped_inputs = prepare_grouped_inputs(
            runner.config.shape[0],
            int(runner.config.value("grouped_regions_per_axis", 8)),
            runner.config.value("grouped_layout", "local"),
            runner.backend,
            runner.config.chunks,
        )
        return
    write_constant_dem(runner.path("source-raster.tif"), runner.config)


def run_statistics(runner: Any, case: Case) -> float:
    """Run statistics and ensure output computes (Dask/MP)."""

    if case.variant == "flox" or case.implementation == "flox":
        if case.implementation == "flox":
            flox_grouped_stats(*runner.grouped_inputs, use_dask=runner.backend == "dask")
        else:
            if runner.backend == "multiprocessing":
                from geoutils.multiproc import MultiprocConfig

                mp_config = MultiprocConfig(chunks=runner.config.chunks, cluster=runner.mp_cluster)
            else:
                mp_config = None
            compute_geoutils_grouped_stats(*runner.grouped_inputs, mp_config=mp_config)
        return 1.0

    dem = runner.make_dem()
    if runner.operation.name == "statistics":
        if runner.backend != "dask":
            raise ValueError("DEM statistics are only registered for Dask")

        # Compute selected reductions without evaluating unrelated quantiles
        import_optional("dask", extra_name="benchmark")
        import dask

        statistics = dem.dem.stats(statistics_options(case, runner.config)["statistics"])
        mean, _, _ = dask.compute(*statistics.values())
        return float(mean)

    return grouped_statistics(runner, dem, case)


def grouped_statistics(runner: Any, dem: Any, case: Case) -> float:
    """Run grouped statistics on the prepared arrays."""

    # Generate coordinates with the same execution backend as the input dem
    height, width = runner.config.shape
    if runner.backend == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        rows = da.arange(height, chunks=runner.config.chunks[0])[:, None]
        columns = da.arange(width, chunks=runner.config.chunks[1])[None, :]
    else:
        rows = np.arange(height)[:, None]
        columns = np.arange(width)[None, :]
    regions = int(runner.config.value("grouped_regions_per_axis", 8))
    if regions < 1 or regions > min(height, width):
        raise ValueError("Grouped regions per axis must fit within the dem dimensions.")

    # Separate localized membership from groups repeated through every chunk
    if runner.config.value("grouped_layout", "local") == "local":
        groups = (rows * regions // height) * regions + columns * regions // width
    else:
        groups = (rows % regions) * regions + columns % regions
    positions = rows * width + columns
    base = dem.data.squeeze()
    signal = base + (rows % 97) * 0.125 + (columns % 53) * 0.25
    values = {
        "signal": np.where(positions % 17 != 0, signal, np.nan),
        "offset": np.where(positions % 29 != 0, 2 * signal + 10, np.nan),
    }

    # Measure the complete public calculation, including exact group gathering when requested
    from geoutils.stats import stats

    options = statistics_options(case, runner.config)
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    result = stats(
        values,
        by={"zone": groups},
        categories={"zone": range(regions**2)},
        **options,
        mp_config=mp_config,
    )

    # Every pixel belongs to a group; missing values follow independent, analytically known periods
    count = height * width
    for name, period in (("signal", 17), ("offset", 29)):
        expected = count - (count + period - 1) // period
        if result[(name, "count")].sum() != expected:
            raise AssertionError(f"Grouped benchmark lost finite observations in {name!r}.")
        if not np.isfinite(result[name].to_numpy()).all():
            raise AssertionError(f"Grouped benchmark returned invalid estimates in {name!r}.")
    return 1.0


def mergeable_statistics_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Select the public reduction backend while preserving the same spatial or stratified workload."""

    options: dict[str, Any] = {"statistics": ("sum", "mean", "valid count"), "backend": "geoutils"}
    if case.engine != "native":
        options.update(backend="flox", flox_engine="numba" if case.engine == "flox_numba" else None)
    if case.method == "stratified":
        options.update(subsample=128, subsample_per_group=True, random_state=42)
    elif case.method in {"fractional", "separate_bands"}:
        options["fractional"] = True
    return options


def prepare_mergeable_statistics(runner: Any, case: Case) -> None:
    """Prepare identical interleaved groups or sparse partial-pixel polygons for every reduction backend."""

    import geoutils as gu

    if case.engine != "native":
        try:
            import_optional("flox", extra_name="num")
        except ImportError as exc:
            raise NotImplementedError("Install geoutils[num] for this grouped comparison") from exc
    values, groups, mask, categories = prepare_grouped_inputs(
        runner.config.shape[0],
        8,
        "interleaved",
        runner.backend,
        runner.config.chunks,
    )
    if case.method == "stratified":
        runner.mergeable_inputs = {
            "source": {"first": values[0], "second": values[1]},
            "by": {"zone": groups},
            "categories": {"zone": categories},
            "mask": mask,
        }
    else:
        import geopandas as gpd
        from rasterio.transform import from_origin
        from shapely.geometry import box

        # Small overlapping polygons touch few chunks; repeated labels require union before cell intersections
        size = runner.config.shape[0]
        locations = np.arange(8, size - 8, 64)
        polygons = [
            box(x + offset + 0.25, y + 0.25, x + offset + 7.75, y + 9.75)
            for x in locations
            for y in locations
            for offset in (0, 4)
        ]
        labels = np.arange(len(polygons)) // 2
        vector = gu.Vector(gpd.GeoDataFrame({"zone": labels}, geometry=polygons, crs=32633))
        import xdem

        constructor = xdem.DEMAccessor if runner.backend == "dask" else xdem.DEM
        raster = constructor.from_array(values, from_origin(0, size, 1, 1), 32633, nodata=-99999)
        runner.mergeable_inputs = {
            "source": raster,
            "values": {"first": 1, "second": 2},
            "by": {"zone": (vector, "zone")},
        }

    # Compile the selected numerical engine outside the ASV operation timing
    if case.engine != "native":
        gu.stats.stats(
            np.arange(16.0),
            "mean",
            by={"zone": np.zeros(16, dtype=int)},
            categories={"zone": [0]},
            backend="flox",
            flox_engine="numba" if case.engine == "flox_numba" else None,
        )


def run_mergeable_statistics(runner: Any, case: Case) -> float:
    """Run unchanged selection or spatial intersections followed by the explicitly selected grouped reduction."""

    from geoutils.stats import stats

    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    inputs = dict(runner.mergeable_inputs)
    options = mergeable_statistics_options(case, runner.config)

    # Include label rasterization in the center and all-touched timings
    if case.method in {"center", "all_touched"}:
        vector, column = inputs["by"]["zone"]
        labels = vector.rasterize(
            ref=inputs["source"],
            in_value=(vector.gdf[column] + 1).tolist(),
            out_value=0,
            out_dtype=np.int32,
            all_touched=case.method == "all_touched",
            mp_config=mp_config,
        )
        inputs.update(by={"zone": labels}, categories={"zone": np.unique(vector.gdf[column]) + 1})

    # Compare shared intersections across bands with a complete call for each band
    selectors = [inputs.pop("values", None)]
    if case.method == "separate_bands":
        selectors = [{name: band} for name, band in selectors[0].items()]
    count = 0.0
    for values in selectors:
        table = stats(**inputs, values=values, **options, mp_config=mp_config)
        count += float(table.xs("count", axis=1, level="statistic").to_numpy().sum())
    return count


# Multiprocessing currently tiles arrays already resident in the client, so only Dask is out of core
STATISTICS = Operation(
    "statistics",
    prepare_statistics,
    run_statistics,
    statistics_options,
    call_name=".stats",
    label="Statistics",
    benchmark_name="stats",
    order=6,
    large_data_cases=(Case(execution="dask", options={"operation": "statistics"}),),
)
GROUPED_STATS = Operation(
    "grouped_stats",
    prepare_statistics,
    run_statistics,
    statistics_options,
    ("statistics", "method", "reindex"),
    call_name="stats",
    label="Grouped statistics",
    benchmark_name="stats",
    order=7,
    large_data_cases=(
        Case(
            method="moments",
            engine="numpy",
            execution="dask",
            strategy="auto",
            options={"operation": "grouped_stats"},
        ),
    ),
)
MERGEABLE_GROUPED_STATS = Operation(
    "mergeable_grouped_stats",
    prepare_mergeable_statistics,
    run_mergeable_statistics,
    mergeable_statistics_options,
    ("statistics", "backend", "fractional", "subsample"),
    call_name="stats",
    label="Mergeable grouped statistics",
    benchmark_name="mergeablestats",
    order=8,
)
OPERATIONS = (STATISTICS, GROUPED_STATS, MERGEABLE_GROUPED_STATS)


###################
# Define benchmarks
###################


EXECUTION_CASES = execution_cases(
    "moments",
    "numpy",
    strategy="dense",
    options={"operation": "grouped_stats"},
    labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["dense"]},
)
STRATEGY_CASES = {
    sweep_id: strategy_cases(
        "moments",
        "numpy",
        tuple(REDUCTION_OPTIONS),
        execution="dask",
        options={"operation": "grouped_stats"},
        strategy_labels=STRATEGY_LABELS,
        variant="interleaved" if sweep_id == "grouped-stats-interleaved-chunks" else None,
    )
    for sweep_id in (
        "grouped-stats-dem-size",
        "grouped-stats-chunk-size",
        "grouped-stats-interleaved-chunks",
        "grouped-stats-group-count",
    )
}
ROBUST_CASES = execution_cases(
    "robust",
    "numpy",
    strategy="groupwise",
    options={"operation": "grouped_stats"},
    labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["groupwise"]},
)

# Compare public GeoUtils stats() with direct Flox reductions on the same prepared arrays
FLOX_CASES = {
    sweep_id: execution_cases(
        "moments",
        "numpy",
        strategy="auto",
        options={"operation": "grouped_stats"},
        labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["auto"]},
        variant="flox",
    )
    for sweep_id in ("grouped-flox-dem-size", "grouped-flox-group-count")
}
FLOX_REFERENCES = {
    sweep_id: tuple(
        reference_case(
            FLOX_CASES[sweep_id],
            implementation="flox",
            execution=execution,
        )
        for execution in ("inmem", "dask")
    )
    for sweep_id in FLOX_CASES
}

MERGEABLE_CASES = tuple(
    Case(
        method=method,
        engine=engine,
        execution=execution,
        labels={"engine": {"native": "GeoUtils", "flox_default": "Flox default", "flox_numba": "Flox Numba"}[engine]},
    )
    for method in ("stratified", "fractional")
    for engine in ("native", "flox_default", "flox_numba")
    for execution in ("inmem", "dask", "multiprocessing")
    if engine == "native" or execution != "multiprocessing"
)
MERGEABLE_CASES += tuple(
    Case(method=method, engine="native", execution=execution, labels={"engine": "GeoUtils"})
    for method in ("center", "all_touched", "separate_bands")
    for execution in ("inmem", "dask", "multiprocessing")
)


def mergeable_dem_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Vary source size with uneven spatial chunks and a fixed per-group sample bound."""

    assert parameter is not None
    return {"shape": (int(parameter), int(parameter)), "chunks": (127, 131)}


def chunk_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define uneven chunks crossed by group boundaries."""

    assert parameter is not None
    chunk_size = int(parameter)
    return {"shape": (DEFAULT_DEM_SIZE, DEFAULT_DEM_SIZE), "chunks": (chunk_size, chunk_size)}


def interleaved_chunk_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define interleaved groups for each chunk size."""

    return {**chunk_size(parameter, case), "grouped_layout": "interleaved"}


def group_count(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define group count for sparse reduction."""

    assert parameter is not None
    return {
        "shape": (DEFAULT_DEM_SIZE, DEFAULT_DEM_SIZE),
        "chunks": (500, 500),
        "grouped_regions_per_axis": int(parameter),
    }


def flox_dem_size(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define dem size with fixed groups and chunks."""

    assert parameter is not None
    size = int(parameter)
    return {
        "shape": (size, size),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "grouped_regions_per_axis": 20,
    }


def flox_group_count(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Define interleaved group count on a fixed dem."""

    assert parameter is not None
    return {
        "shape": (DEFAULT_DEM_SIZE, DEFAULT_DEM_SIZE),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "grouped_regions_per_axis": int(parameter),
        "grouped_layout": "interleaved",
    }


def grouped_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the dem, chunks and grouped membership layout."""

    config = configs[0]
    regions = int(config.value("grouped_regions_per_axis", 8))
    layout = config.value("grouped_layout", "local")
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} dem; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; "
        f"{regions:,} × {regions:,} {layout} groups"
    )


# Measure the selected cases over each numeric axis, including matched Flox cases for the first two sweeps
BENCHMARKS = (
    parameter_config(
        "dem_size",
        DEM_SIZES,
        GROUPED_STATS,
        (*FLOX_CASES["grouped-flox-dem-size"], *FLOX_REFERENCES["grouped-flox-dem-size"]),
        flox_dem_size,
        name="grouped-flox",
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "groups_per_axis",
        GROUP_COUNTS,
        GROUPED_STATS,
        (*FLOX_CASES["grouped-flox-group-count"], *FLOX_REFERENCES["grouped-flox-group-count"]),
        flox_group_count,
        name="grouped-flox",
        parameter_label="Number of groups per axis",
        parameter_title="group count",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "dem_size",
        DEM_SIZES,
        GROUPED_STATS,
        merge_cases(EXECUTION_CASES, STRATEGY_CASES["grouped-stats-dem-size"]),
        dem_size_config,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "chunk_size",
        DEM_CHUNK_SIZES,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-chunk-size"],
        chunk_size,
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "chunk_size",
        DEM_CHUNK_SIZES,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-interleaved-chunks"],
        interleaved_chunk_size,
        name="grouped-stats-interleaved",
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "groups_per_axis",
        GROUP_COUNTS,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-group-count"],
        group_count,
        parameter_label="Number of groups per axis",
        parameter_title="group count",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "dem_size",
        DEM_SIZES,
        GROUPED_STATS,
        ROBUST_CASES,
        dem_size_config,
        name="grouped-stats-robust",
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
        describe_workload=grouped_workload,
    ),
)

BENCHMARKS += (
    parameter_config(
        "dem_size",
        (256, 512, 1024),
        MERGEABLE_GROUPED_STATS,
        MERGEABLE_CASES,
        mergeable_dem_size,
        parameter_label="Size of dem (pixels per side)",
        parameter_title="dem size",
    ),
)


#############################
# Define report comparisons
#############################

# Comparisons select saved GeoUtils and Flox series for external plots, then GeoUtils series for internal strategy plots
COMPARISONS = (
    *tuple(comparison(benchmark, by="execution", documentation=False, summary=False) for benchmark in BENCHMARKS[:2]),
    comparison(
        BENCHMARKS[2],
        by="execution",
        cases=EXECUTION_CASES,
        slug="grouped-stats-execution-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            benchmark,
            by="strategy",
            cases=STRATEGY_CASES[sweep_id],
            documentation=False,
        )
        for benchmark, sweep_id in zip(
            BENCHMARKS[2:6],
            (
                "grouped-stats-dem-size",
                "grouped-stats-chunk-size",
                "grouped-stats-interleaved-chunks",
                "grouped-stats-group-count",
            ),
            strict=True,
        )
    ),
    comparison(BENCHMARKS[6], by="execution", documentation=False),
    *tuple(
        comparison(
            BENCHMARKS[7],
            by="engine",
            cases=tuple(case for case in MERGEABLE_CASES if case.method == method and case.execution == execution),
            slug=f"mergeable-stats-{method}-{execution}",
            documentation=False,
        )
        for method in ("stratified", "fractional")
        for execution in ("inmem", "dask", "multiprocessing")
    ),
    *tuple(
        comparison(
            BENCHMARKS[7],
            by="method",
            cases=tuple(
                case
                for case in MERGEABLE_CASES
                if case.method != "stratified" and case.engine == "native" and case.execution == execution
            ),
            slug=f"zonal-stats-{execution}",
            documentation=False,
        )
        for execution in ("inmem", "dask", "multiprocessing")
    ),
)


###################################
# Internal reduction microbenchmark
###################################


class GlobalDaskReduction:
    """Measure the shared GeoUtils reducer and native Dask reductions on the same prepared array."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300
    param_names = ["implementation", "dem_size"]
    params = [["geoutils", "native_dask"], asv_parameter_values(DEM_SIZES)]

    def setup(self, implementation: Literal["geoutils", "native_dask"], dem_size: int) -> None:
        """Prepare one finite Dask dem and the same mergeable statistic request for both implementations."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        del implementation
        generator = np.random.default_rng(42)
        values = generator.normal(size=(dem_size, dem_size))
        values[::97, ::89] = np.nan
        self.values = da.from_array(values, chunks=(500, 500))
        self.aliases = {"mean", "min", "max", "sum", "sumofsquares", "rmse", "std"}
        self.statistics = _normalize_statistics(sorted(self.aliases), grouped=False)

    def time_reduction(self, implementation: Literal["geoutils", "native_dask"], dem_size: int) -> None:
        """Compute the requested global estimates with the selected Dask reduction implementation."""

        import dask

        del dem_size
        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils":
                _reduce_values([self.values], self.statistics)
            else:
                global_statistics(self.values)


# Keep the stored identifier while locating this internal comparison outside the public ASV modules
setattr(
    GlobalDaskReduction.time_reduction,
    "benchmark_name",
    "asv_suite.global_stats.GlobalDaskReduction.time_reduction",
)

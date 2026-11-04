"""Configuration for benchmarking: input defaults, worker settings, and varying parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from benchmarks.workflows.core import Case

#########################
# Benchmark input sizes
#########################

# Default values for fixed benchmark (not parameter varying), or for when we vary another input
DEFAULT_RASTER_SIZE = 2_000
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_MEMORY_LIMIT = "1GB"
DEFAULT_N_WORKERS = 1
DEFAULT_THREADS_PER_WORKER = 1
DEFAULT_GDAL_CACHEMAX_MB = 64
DEFAULT_PROFILE_INTERVAL = 0.05
DEFAULT_RASTER_VALUE = 1.0
DEFAULT_POINT_COUNT = 2_048

# Values for variable inputs
RASTER_SIZES = (2_000, 4_000)
RASTER_CHUNK_SIZES = (100, 500, 1_000)
GROUP_COUNTS = (10, 20, 50)
POINT_COUNTS = (1_000, 10_000, 100_000)
INTERPOLATED_POINT_COUNTS = POINT_COUNTS
SUBSAMPLE_SIZES = POINT_COUNTS
GRID_POINT_COUNTS = (10, 20, 40)
POINT_CHUNK_SIZE = 100
VECTOR_COMPARISON_OPTIONS = {"vector_features_per_axis": 51}
WORKER_RASTER_SIZES = (1_000,)
VARIOGRAM_PAIR_COUNTS = (10_000, 100_000, 1_000_000)
VARIOGRAM_LAG_COUNTS = (10, 100, 1_000)
VARIOGRAM_RASTER_SIZE = 1_024
VARIOGRAM_RASTER_CHUNK_SIZE = 256
VARIOGRAM_SAMPLE_PAIRS = 10_000
VARIOGRAM_LAG_PAIRS = 100_000
VARIOGRAM_POINT_PAIRS = 2_000
VARIOGRAM_N_LAGS = 24
DASK_CUTOFF_SIZES = (262_145, 524_288, 1_048_576)


#########################
# Runtime configuration
#########################


@dataclass
class RuntimeConfig:
    """Collect settings shared by worker management, I/O and profiling."""

    shape: tuple[int, int] = (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE)
    chunks: tuple[int, int] = (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    n_workers: int = DEFAULT_N_WORKERS
    threads_per_worker: int = DEFAULT_THREADS_PER_WORKER
    gdal_cachemax_mb: int = DEFAULT_GDAL_CACHEMAX_MB
    profile_interval: float = DEFAULT_PROFILE_INTERVAL
    raster_value: float = DEFAULT_RASTER_VALUE
    directory: str | None = None
    workload: Mapping[str, Any] = field(default_factory=dict)

    def value(self, name: str, default: Any = None) -> Any:
        """Return one benchmark-specific workload value."""

        return self.workload.get(name, default)


def runtime_config(values: Mapping[str, Any]) -> RuntimeConfig:
    """Separate shared runtime settings from operation-specific workload values."""

    runtime_names = {item.name for item in fields(RuntimeConfig)} - {"workload"}
    runtime = {name: value for name, value in values.items() if name in runtime_names}
    workload = {name: value for name, value in values.items() if name not in runtime_names}
    return RuntimeConfig(**runtime, workload=workload)


def fixed_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Return the common fixed workload used when no input parameter varies."""

    return {
        "shape": (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "subsample_size": DEFAULT_POINT_COUNT,
        "ninterp": DEFAULT_POINT_COUNT,
    }


def raster_size_config(parameter: int | float | None, case: Case) -> Mapping[str, Any]:
    """Set a square raster size around the shared scheduled chunk size."""

    assert parameter is not None
    size = int(parameter)
    return {"shape": (size, size), "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE)}


def with_directory(config: RuntimeConfig, directory: str) -> RuntimeConfig:
    """Return a runtime configuration that writes into one persistent directory."""

    return replace(config, directory=directory)

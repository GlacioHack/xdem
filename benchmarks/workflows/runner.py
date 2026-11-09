# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module implements the benchmark runner infrastructure, which includes the call/storage of profiling."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any

import rasterio as rio
from geoutils._misc import (
    _get_process_mem_mb,
    _prepare_benchmark_process,
    import_optional,
)
from geoutils.profiler import ProfileMetrics, profile_call

from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import Case, ExecutionMode, Operation
from benchmarks.workflows.io import (
    materialize_dem_output,
    open_dem_input,
    prepare_output_file,
    read_dem_center,
)

####################
# Measured results
####################


def process_tree_memory_increase_mb(metrics: ProfileMetrics) -> float:
    """Return peak process-tree memory above its first measured value."""

    if not metrics.process_tree_mem_mb:
        raise RuntimeError("Process-tree memory was not collected for this benchmark result")
    baseline = metrics.process_tree_mem_mb[0][1]
    peak = max(value for _, value in metrics.process_tree_mem_mb)
    return max(0.0, peak - baseline)


class ProfiledResult:
    """Store memory measurements for a benchmark implementation."""

    metrics: ProfileMetrics

    @property
    def peak_process_tree_mem_mb(self) -> float:
        """Return peak aggregate memory for the measured process and its children."""

        peak = self.metrics.peak_process_tree_mem_mb
        if peak is None:
            raise RuntimeError("Process-tree memory was not collected for this benchmark result")
        return peak

    @property
    def process_tree_mem_increase_mb(self) -> float:
        """Return peak memory above the initialized process-tree baseline."""

        return process_tree_memory_increase_mb(self.metrics)


@dataclass
class BenchmarkResult(ProfiledResult):
    """Store one computed result including memory and worker "health" measurements."""

    value: float
    metrics: ProfileMetrics
    worker_pids_before: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    worker_pids_after: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    dask_worker_baseline_mem_mb: float | None = None
    output_file: str | None = None

    @property
    def worker_restarted(self) -> bool:
        """Whether the backend replaced a worker during the operation."""

        return self.worker_pids_before != self.worker_pids_after


############################################
# Worker lifecycle and complete operations
############################################


# We prepare shared inputs, start an execution mode and ensure each operation produces a complete output
class BenchmarkRunner:
    """This class manage runtime resources while executing a benchmark implementation."""

    def __init__(self, operation: Operation, case: Case, config: RuntimeConfig | None = None) -> None:
        """Prepare runner state."""

        self.operation = operation
        self.case = case
        self.backend: ExecutionMode = case.execution or "inmem"
        selected_config = config or RuntimeConfig()
        self.config = replace(selected_config, workload={**case.options, **selected_config.workload})
        self.cluster: Any | None = None
        self.client: Any | None = None
        self.mp_cluster: Any | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._dask_config_context: Any | None = None
        self._directory: str | None = None
        self._last_output_file: str | None = None
        self._opened_inputs: list[Any] = []

    @property
    def directory(self) -> str:
        """Return the working directory containing sources and outputs."""

        if self._directory is None:
            raise RuntimeError("BenchmarkRunner has not been prepared")
        return self._directory

    def path(self, filename: str) -> str:
        """Return a path inside the benchmark directory."""
        return os.path.join(self.directory, filename)

    def __enter__(self) -> BenchmarkRunner:
        """Prepare sources and start the selected backend."""

        return self.start()

    def __exit__(self, *args: object) -> None:
        """Close workers and temporary files when leaving the context."""

        self.close()

    def prepare(self) -> BenchmarkRunner:
        """Create the working directory and operation inputs without starting workers."""

        # Use the caller directory when results must survive this runner
        if self._directory is None and self.config.directory is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-benchmark-")
            self._directory = self._tmpdir.name
        elif self._directory is None:
            assert self.config.directory is not None
            os.makedirs(self.config.directory, exist_ok=True)
            self._directory = self.config.directory

        # The operation owns the sources needed by its GeoUtils and implementation cases
        self.operation.prepare(self, self.case)
        return self

    def start(self) -> BenchmarkRunner:
        """Prepare source files and start workers when the implementation needs them."""

        self.prepare()
        if self.backend == "dask":
            self._start_dask()
        elif self.backend == "multiprocessing":
            self._start_multiprocessing()
        return self

    def _start_dask(self) -> None:
        """Start one local Dask cluster with early disk spilling."""

        # Import benchmark packages at runtime to keep Dask optional
        dask = import_optional("dask", extra_name="benchmark")
        distributed = import_optional("distributed", extra_name="benchmark")

        # Spill arrays early and let the large data tests enforce the measured memory bound
        self._dask_config_context = dask.config.set(
            {
                "temporary-directory": self.directory,
                "distributed.worker.memory.target": 0.45,
                "distributed.worker.memory.spill": 0.60,
                # Native imports and task metadata can prevent a paused worker from resuming
                "distributed.worker.memory.pause": False,
                # Peak memory and stable PIDs are asserted directly by the large data test
                "distributed.worker.memory.terminate": False,
            }
        )
        self._dask_config_context.__enter__()

        # Separate worker processes make their memory independent from the client
        self.cluster = distributed.LocalCluster(
            n_workers=self.config.n_workers,
            threads_per_worker=self.config.threads_per_worker,
            processes=True,
            memory_limit=self.config.memory_limit,
            dashboard_address=":0",
            scheduler_kwargs={"dashboard": False},
            local_directory=self.directory,
            env={"GDAL_CACHEMAX": str(self.config.gdal_cachemax_mb)},
        )
        self.client = distributed.Client(self.cluster)

        # Configure every worker's live GDAL library before measurement
        self.client.run(_prepare_benchmark_process, self.config.gdal_cachemax_mb)

    def _start_multiprocessing(self) -> None:
        """Start a real multiprocessing pool with its normal bounded task lifetime."""

        from geoutils.multiproc.cluster import MpCluster

        # Child processes inherit a bounded GDAL block cache before the pool starts
        previous_cachemax = os.environ.get("GDAL_CACHEMAX")
        os.environ["GDAL_CACHEMAX"] = str(self.config.gdal_cachemax_mb)
        try:
            # Rasterio has already initialized GDAL, so set its live cache while workers fork
            # Rasterio's integer option is measured in bytes, while worker environment strings are measured in MiB
            with rio.Env(GDAL_CACHEMAX=self.config.gdal_cachemax_mb * 1024**2):
                # Default recycling bounds allocator and native-library caches in long jobs
                self.mp_cluster = MpCluster(conf={"nb_workers": self.config.n_workers})

                # Forkserver and spawn workers import modules independently, so finish their warm-up before profiling
                warmup = self.mp_cluster.submit(_prepare_benchmark_process, self.config.gdal_cachemax_mb)
                self.mp_cluster.compute(warmup)
        finally:
            # Restore the caller environment after all workers have inherited it
            if previous_cachemax is None:
                os.environ.pop("GDAL_CACHEMAX", None)
            else:
                os.environ["GDAL_CACHEMAX"] = previous_cachemax

    def close(self) -> None:
        """Close workers and remove only temporary files owned by this runner."""

        # Release Xarray file managers while their supporting libraries are still initialized
        for source in self._opened_inputs:
            source.close()
        self._opened_inputs.clear()

        # Close every execution backend before removing its working directory
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None
        if self.mp_cluster is not None:
            self.mp_cluster.close()
            self.mp_cluster = None

        # Restore Dask configuration after all distributed processes have stopped
        if self._dask_config_context is not None:
            self._dask_config_context.__exit__(None, None, None)
            self._dask_config_context = None

        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._directory = None

    def worker_pids(self) -> tuple[int, ...] | dict[str, int]:
        """Return stable worker identifiers for the selected backend."""

        if self.backend == "dask":
            if self.client is None:
                return {}
            return {str(address): int(pid) for address, pid in self.client.run(os.getpid).items()}
        if self.backend == "inmem" or self.mp_cluster is None:
            return ()
        return self.mp_cluster.worker_pids()

    def make_dem(self, filename: str | None = None) -> Any:
        """Open one prepared dem through the selected GeoUtils interface."""

        # Most operations use the constant source while polygonization uses regions
        source_file = self.path("source-raster.tif") if filename is None else filename
        dem = open_dem_input(source_file, self.backend, self.config)
        if self.backend == "dask":
            self._opened_inputs.append(dem)
        return dem

    def run(self, *, profile: bool = True) -> BenchmarkResult:
        """Compute one operation while measuring its complete backend."""

        if self.backend == "dask" and self.client is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")
        if self.backend == "multiprocessing" and self.mp_cluster is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")
        if self.backend == "inmem" and self._directory is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")

        # Worker identities reveal failures hidden by automatic replacement
        worker_pids_before = self.worker_pids()
        if self.client is not None:
            # Query workers directly because Dask's first monitor sample can predate warm-up
            worker_baseline = sum(float(value) for value in self.client.run(_get_process_mem_mb).values())
        else:
            worker_baseline = None
        if profile:
            value, metrics = profile_call(
                self._execute,
                interval=self.config.profile_interval,
                client=self.client,
                dask=self.backend == "dask",
                # One process-tree measurement is comparable across both backends
                include_children=True,
            )
        else:
            value, metrics = profile_call(
                self._execute,
                interval=self.config.profile_interval,
                dask=False,
                include_children=False,
            )
        worker_pids_after = self.worker_pids()

        # Return one small value while leaving large output on disk
        return BenchmarkResult(
            value=float(value),
            metrics=metrics,
            worker_pids_before=worker_pids_before,
            worker_pids_after=worker_pids_after,
            dask_worker_baseline_mem_mb=worker_baseline,
            output_file=self._last_output_file,
        )

    def _output_path(self, suffix: str = ".tif") -> str:
        """Return one reusable output path for an operation."""

        return prepare_output_file(self.directory, self.backend, self.operation.name, suffix)

    def _multiproc_config(self, *, suffix: str = ".tif", driver: str | None = None) -> Any:
        """Build a multiprocessing configuration sharing this runner's worker pool."""

        from geoutils.multiproc import MultiprocConfig

        return MultiprocConfig(
            chunks=self.config.chunks,
            outfile=self._output_path(suffix=suffix),
            driver=driver,
            cluster=self.mp_cluster,
        )

    def _compute_dem(self, dem: Any) -> float:
        """Write or inspect the complete dem produced by one implementation."""

        output_file = None if self.backend == "multiprocessing" else self._output_path()
        self._last_output_file = materialize_dem_output(
            dem,
            self.backend,
            output_file,
            self.config,
        )
        return read_dem_center(self._last_output_file)

    def _execute(self) -> float:
        """Fully compute the selected operation and case."""

        self._last_output_file = None
        return self.operation.execute(self, self.case)

"""Render a custom HTML webpage + plots for user documentation from the ASV measurements."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from inspect import signature
from pathlib import Path
from typing import Any, Literal

from benchmarks.workflows.core import (
    IMPLEMENTATION_LABELS,
    Case,
    Comparison,
    ComparisonDimension,
    ExecutionMode,
)
from benchmarks.workflows.operations import (
    BENCHMARK_BY_CLASS,
    BENCHMARK_CASE_BY_CLASS,
    COMPARISONS,
    OPERATIONS,
)
from xdem.dem.base import DEMBase

GDAL_CLI_LABEL = IMPLEMENTATION_LABELS["gdal"]
PDAL_CLI_LABEL = IMPLEMENTATION_LABELS["pdal"]

# Name the files written by this renderer and linked from the benchmark website and documentation
COMPARISON_BENCHMARK_MODULE = "asv_suite.parameter_sweeps"
COMPARISON_REPORT_DIRECTORY = "comparisons"
SCALING_REPORT_PAGE = "scaling.html"
REFERENCE_REPORT_PAGE = "gdal-pdal.html"
PARAMETER_REPORT_PAGE = "parameters.html"
HISTORY_REPORT_PAGE = "history.html"
DOCUMENTATION_TIME_PLOT = "time_relative_to_gdal.svg"
DOCUMENTATION_MEMORY_PLOT = "memory_increase_by_operation.svg"
DOCUMENTATION_MEMORY_SCALING_PLOT = "memory_increase_by_dem_size.svg"
DOCUMENTATION_PDAL_PLOT = "time_relative_to_pdal.svg"
DOCUMENTATION_DATA = "benchmark_snapshot.json"
PERFORMANCE_CHANGE_REPORT = "performance-change.md"
PREVIEW_WEBSITE_DIRECTORY = Path("benchmarks/results/asv/preview")
SUMMARY_DEM_SIZE = 10_000
SUMMARY_POINT_COUNT = 100_000_000

# Colors stay consistent between the detailed ASV plots and the concise documentation snapshot
SERIES_COLORS = {
    "In memory": "#0072B2",
    "Dask": "#E69F00",
    "Multiprocessing": "#009E73",
    GDAL_CLI_LABEL: "#6C6C6C",
    PDAL_CLI_LABEL: "#CC79A7",
}

# Group plots by the xDEM choice represented by their separate lines; every plot still scales a numeric input
COMPARISON_SECTION_DETAILS: dict[ComparisonDimension, tuple[str, str]] = {
    "execution": (
        "Execution modes",
        "Compare in-memory, Dask and Multiprocessing with one worker, showing coordination costs without parallel "
        "speed-up.",
    ),
    "engine": (
        "Calculation engines",
        "Compare numerical libraries during in-memory execution, so scheduling remains fixed.",
    ),
    "method": (
        "Operation methods",
        "Compare operation-specific algorithms, such as nearest, linear/Delaunay and inverse-distance gridding.",
    ),
    "strategy": (
        "Chunk strategies",
        "Compare how chunked operations reconcile partial results at chunk boundaries.",
    ),
    "output_driver": (
        "Point output formats",
        "Compare file-to-file LAS and LAZ output from v multiprocessing and PDAL CLI.",
    ),
}

# Scaling sections answer which input grows, independently of the GeoUtils choices represented by plot lines
SCALING_SECTION_DETAILS = {
    "Size of dem (pixels per side)": (
        "Scaling with dem size",
        "DEM width and height vary together; other inputs remain fixed.",
    ),
    "Size of chunks (pixels per side)": (
        "Scaling with chunk size",
        "Chunk width and height vary while the complete dem remains fixed.",
    ),
    "Number of interpolated points": (
        "Scaling with the number of interpolated points",
        "The number of requested point locations varies while the source dem remains fixed.",
    ),
    "Number of source points per axis": (
        "Scaling with the number of source points",
        "The number of source points varies while the output dem remains fixed.",
    ),
    "Number of sampled values": (
        "Scaling with the number of sampled values",
        "The returned sample size varies while the source dem remains fixed.",
    ),
    "Number of output points": (
        "Scaling with the number of output points",
        "The point cloud size varies while the source dem remains fixed.",
    ),
    "Number of groups per axis": (
        "Scaling with the number of groups",
        "The number of rectangular groups varies while dem and chunk dimensions remain fixed.",
    ),
}

# Group operations by input and output type, using the same data-flow labels as the scalability documentation
OPERATION_GROUP_ORDER = (
    "DEM ⟶ DEM",
    "Point ⟶ Point",
    "DEM ⟶ Point",
    "DEM ⟶ Vector",
    "Point ⟶ DEM",
    "Vector ⟶ DEM",
    "DEM ⟶ Other",
    "Point ⟶ Other",
)
OPERATION_GROUPS: dict[str, str] = {
    "crop": "DEM ⟶ DEM",
    "clip": "DEM ⟶ DEM",
    "translate": "DEM ⟶ DEM",
    "copy": "DEM ⟶ DEM",
    "filter": "DEM ⟶ DEM",
    "reproject": "DEM ⟶ DEM",
    "random_field": "DEM ⟶ DEM",
    "subsample": "DEM ⟶ Point",
    "to_pointcloud": "DEM ⟶ Point",
    "interp_points": "DEM ⟶ Point",
    "polygonize": "DEM ⟶ Vector",
    "grid": "Point ⟶ DEM",
    "rasterize": "Vector ⟶ DEM",
    "create_mask": "Vector ⟶ DEM",
    "statistics": "DEM ⟶ Other",
    "grouped_stats": "DEM ⟶ Other",
    "mergeable_grouped_stats": "DEM ⟶ Other",
    "dem_variogram": "DEM ⟶ Other",
    "dem_pairsample": "DEM ⟶ Other",
    "variogram_from_pairs": "Point ⟶ Other",
    "write": "DEM ⟶ Other",
}


# Store one row per plot point, with operation, method, engine, strategy and execution mode in separate fields
@dataclass(frozen=True)
class ComparisonMeasurement:
    """Store one fully identified benchmark result at one numeric parameter value."""

    comparison: str
    series_label: str
    operation: str
    method: str | None
    calculation_engine: str | None
    strategy: str | None
    execution_mode: ExecutionMode | None
    external_reference: str | None
    output_driver: str
    series_dimension: str
    parameter: int
    operation_time_s: float
    end_to_end_time_s: float
    process_tree_mem_increase_mb: float


class _PreviewResult:
    """Provide complete deterministic ASV-like measurements for renderer development."""

    def __init__(self, commit_hash: str = "preview-current", xdem_scale: float = 1.0) -> None:
        """Create sample values matching each comparison's parameter count."""

        self.commit_hash = commit_hash
        self.date = 1_700_000_000_000
        self.params = {"machine": "preview-machine", "cpu": "preview-cpu"}
        self.started_at = {"benchmark": 1_700_000_001_000}
        self._env_name = "preview"
        self.values: dict[str, list[float]] = {}
        self.parameters: dict[str, list[list[str]]] = {}

        # Create all keys normally read from saved ASV results so every report section can render
        for comparison in COMPARISONS:
            parameter_values = comparison.benchmark.values
            if comparison.benchmark.parameter_name == "dem_size" and SUMMARY_DEM_SIZE not in parameter_values:
                parameter_values = (*parameter_values, SUMMARY_DEM_SIZE)
            parameter_count = len(parameter_values)

            for series_index, (_, class_name) in enumerate(comparison.series, start=1):
                prefix = f"{COMPARISON_BENCHMARK_MODULE}.{class_name}"
                if f"{prefix}.time_operation" in self.values:
                    continue

                # Scale xDEM values between fake revisions while leaving optional and CLI references stable
                benchmark_case = BENCHMARK_CASE_BY_CLASS[class_name]
                scale = xdem_scale if benchmark_case.implementation == "xdem" else 1.0
                parameters = [[repr(value) for value in parameter_values]]
                self.parameters[f"{prefix}.time_operation"] = parameters
                self.parameters[f"{prefix}.track_end_to_end_time_s"] = parameters
                self.parameters[f"{prefix}.track_process_tree_mem_increase_mb"] = parameters
                self.values[f"{prefix}.time_operation"] = [
                    scale * series_index * 0.05 * 2**index for index in range(parameter_count)
                ]
                self.values[f"{prefix}.track_end_to_end_time_s"] = [
                    scale * series_index * 0.10 * 2**index for index in range(parameter_count)
                ]
                self.values[f"{prefix}.track_process_tree_mem_increase_mb"] = [
                    scale * series_index * (100.0 + 40.0 * index + 20.0 * max(0, index - 1))
                    for index in range(parameter_count)
                ]

    def get_all_result_keys(self) -> Any:
        """Return every generated benchmark result key."""

        return self.values.keys()

    def get_result_params(self, key: str) -> list[list[str]]:
        """Return the numeric parameter axis stored for one benchmark."""

        return self.parameters[key]

    def get_result_value(self, key: str, params: list[list[str]]) -> list[float]:
        """Return measurements aligned with the requested parameter axis."""

        if params != self.parameters[key]:
            raise ValueError(f"Preview parameters do not match the generated values for {key}")
        return self.values[key]


def _benchmark_key(class_name: str, method_name: str) -> str:
    """Return the stable ASV result key for one comparison method."""

    # ASV names benchmarks relative to the configured benchmark directory
    return f"{COMPARISON_BENCHMARK_MODULE}.{class_name}.{method_name}"


def _latest_timestamp(result: Any) -> int:
    """Return the most recent measurement timestamp stored in one ASV result."""

    # Start times distinguish reruns of the same commit better than commit dates
    started_at = list(result.started_at.values())
    if started_at:
        return int(max(started_at))
    return int(result.date or 0)


def _required_benchmark_keys(comparisons: Iterable[Comparison] = COMPARISONS) -> set[str]:
    """Return every saved result needed to render the selected comparisons."""

    # A partially interrupted ASV run must not replace the latest complete documentation result
    methods = ("time_operation", "track_end_to_end_time_s", "track_process_tree_mem_increase_mb")
    return {
        _benchmark_key(class_name, method)
        for comparison in comparisons
        for _, class_name in comparison.series
        for method in methods
    }


def _select_complete_result(
    results: Iterable[Any],
    *,
    machine: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
    require_complete: bool = True,
) -> Any:
    """Select the newest matching result from an ASV result sequence."""

    required_keys = _required_benchmark_keys()
    candidates = []
    for result in results:
        # Machine and commit filters make local multi-machine histories predictable
        if machine is not None and result.params.get("machine") != machine:
            continue
        if commit is not None and not result.commit_hash.startswith(commit):
            continue

        # ASV does not expose the environment name as a public property
        result_environment = str(getattr(result, "_env_name", ""))
        if environment is not None and result_environment != environment:
            continue
        if require_complete and not required_keys.issubset(result.get_all_result_keys()):
            continue
        candidates.append(result)

    if not candidates:
        raise RuntimeError("No matching saved ASV result contains the requested comparisons")
    return max(candidates, key=_latest_timestamp)


def select_asv_result(
    results_directory: Path,
    *,
    machine: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
    require_complete: bool = True,
) -> Any:
    """Read ASV results and select the newest matching comparison result."""

    # Import ASV only when rendering so benchmark discovery remains lightweight
    from asv.results import iter_results

    return _select_complete_result(
        iter_results(str(results_directory)),
        machine=machine,
        commit=commit,
        environment=environment,
        require_complete=require_complete,
    )


def _decode_parameter(value: Any) -> int:
    """Convert the stable ASV parameter representation back to an integer."""

    # ASV stores parameters as repr strings even when benchmark inputs are numeric
    decoded = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(decoded, int):
        raise TypeError(f"Expected one integer ASV parameter, received {decoded!r}")
    return decoded


def _result_series(result: Any, key: str) -> tuple[list[int], list[float]]:
    """Return aligned numeric parameters and measurements for one ASV result key."""

    # Comparisons deliberately define exactly one parameter dimension
    params = result.get_result_params(key)
    if len(params) != 1:
        raise ValueError(f"Comparison must contain one parameter: {key}")
    parameters = [_decode_parameter(value) for value in params[0]]

    # Ask ASV to align values against the parameter order stored with this result
    values = result.get_result_value(key, params)
    if not isinstance(values, list):
        values = [values]
    if len(parameters) != len(values) or any(value is None for value in values):
        raise ValueError(f"Incomplete comparison result: {key}")
    return parameters, [float(value) for value in values]


def collect_comparison_measurements(
    result: Any,
    comparisons: Iterable[Comparison] = COMPARISONS,
) -> list[ComparisonMeasurement]:
    """Combine matching operation-only, elapsed-time and peak-memory results."""

    # Convert each saved ASV series into one record per numeric input value
    records = []
    available = set(result.get_all_result_keys())
    for comparison in comparisons:
        for series_label, class_name in comparison.series:
            selected_case = BENCHMARK_CASE_BY_CLASS[class_name]
            if BENCHMARK_BY_CLASS[class_name] != comparison.benchmark:
                raise ValueError(f"Comparison benchmark does not match registered case {class_name}")

            # The three ASV methods share the same numeric parameter values
            keys = {
                "operation": _benchmark_key(class_name, "time_operation"),
                "end_to_end": _benchmark_key(class_name, "track_end_to_end_time_s"),
                "memory": _benchmark_key(class_name, "track_process_tree_mem_increase_mb"),
            }
            missing = set(keys.values()) - available
            if missing:
                raise ValueError(f"Missing comparison results: {', '.join(sorted(missing))}")

            operation_params, operation_times = _result_series(result, keys["operation"])
            end_to_end_params, end_to_end_times = _result_series(result, keys["end_to_end"])
            memory_params, peak_memory = _result_series(result, keys["memory"])
            if operation_params != end_to_end_params or operation_params != memory_params:
                raise ValueError(f"Time and RAM parameters differ for {class_name}")

            # ASV stores skipped parameter combinations as NaN, which should leave no plotted metric
            records.extend(
                ComparisonMeasurement(
                    comparison=comparison.slug,
                    series_label=series_label,
                    operation=comparison.operation,
                    method=selected_case.method,
                    calculation_engine=selected_case.engine if selected_case.implementation == "xdem" else None,
                    strategy=selected_case.strategy if selected_case.implementation == "xdem" else None,
                    execution_mode=selected_case.execution,
                    external_reference={"gdal": "gdal_cli", "pdal": "pdal_cli", "flox": "flox"}.get(
                        selected_case.implementation
                    ),
                    output_driver=selected_case.output_driver,
                    series_dimension={
                        "engine": "calculation_engine",
                        "execution": "execution_mode",
                        "output_driver": "output_format",
                    }.get(comparison.by, comparison.by),
                    parameter=parameter,
                    operation_time_s=operation_time,
                    end_to_end_time_s=end_to_end_time,
                    process_tree_mem_increase_mb=memory,
                )
                for parameter, operation_time, end_to_end_time, memory in zip(
                    operation_params,
                    operation_times,
                    end_to_end_times,
                    peak_memory,
                )
                if all(math.isfinite(value) for value in (operation_time, end_to_end_time, memory))
            )
    return records


def _plot_comparison(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    output: Path,
    *,
    include_operation: bool = True,
) -> None:
    """Plot elapsed time and peak additional RAM, optionally adding operation-only time."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xdem-matplotlib-cache"))

    # Import plotting only for explicit rendering commands
    import matplotlib.pyplot as plt

    metrics = [
        ("end_to_end_time_s", "Elapsed time (s)"),
        ("process_tree_mem_increase_mb", "Peak additional RAM (MB)"),
    ]
    if include_operation:
        metrics.insert(0, ("operation_time_s", "Operation only (s)"))
    figure, axes = plt.subplots(len(metrics), 1, figsize=(9, 3 * len(metrics)), sharex=True, layout="constrained")
    for series_label, _ in comparison.series:
        # Stable parameter sorting keeps lines readable if ASV changes storage order
        selected = sorted(
            (
                record
                for record in records
                if record.comparison == comparison.slug and record.series_label == series_label
            ),
            key=lambda record: record.parameter,
        )
        parameters = [record.parameter for record in selected]
        for axis, (metric, _) in zip(axes, metrics):
            axis.plot(parameters, [getattr(record, metric) for record in selected], marker="o", label=series_label)

    # Separate panels keep initialized work, complete execution and memory unambiguous
    axes[0].set_title(comparison.title)
    axes[-1].set_xlabel(comparison.parameter_label)
    for axis, (_, label) in zip(axes, metrics):
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    if comparison.logarithmic_x:
        axes[-1].set_xscale("log", base=2)

    # SVG stays sharp in local reports and on the published benchmark website
    figure.savefig(output, format="svg")
    plt.close(figure)


def _gdal_comparisons() -> tuple[Comparison, ...]:
    """Return dem-size comparisons that provide a GDAL CLI reference line."""

    return tuple(
        comparison
        for comparison in COMPARISONS
        if comparison.documentation and any(label == GDAL_CLI_LABEL for label, _ in comparison.series)
    )


def _pdal_comparisons() -> tuple[Comparison, ...]:
    """Return documentation comparisons that provide a PDAL CLI reference line."""

    return tuple(
        comparison
        for comparison in COMPARISONS
        if comparison.documentation and any(label == PDAL_CLI_LABEL for label, _ in comparison.series)
    )


def _largest_shared_parameter(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
) -> int:
    """Return the largest parameter measured by every series in one comparison."""

    # Shared parameters keep summary bars based on exactly the same input size
    parameters_by_series = []
    for series_label, class_name in comparison.series:
        parameters = {
            record.parameter
            for record in records
            if record.comparison == comparison.slug and record.series_label == series_label
        }
        # Optional Flox references may be skipped while the xDEM measurements remain available
        case = BENCHMARK_CASE_BY_CLASS[class_name]
        if not parameters and case.implementation == "flox":
            continue
        parameters_by_series.append(parameters)
    shared_parameters = set.intersection(*parameters_by_series)
    if not shared_parameters:
        raise ValueError(f"No shared parameter found for {comparison.slug}")
    return max(shared_parameters)


def _measurement_at(
    records: list[ComparisonMeasurement],
    comparison: str,
    series_label: str,
    parameter: int,
) -> ComparisonMeasurement:
    """Return one uniquely identified plot-series measurement."""

    selected = [
        record
        for record in records
        if record.comparison == comparison and record.series_label == series_label and record.parameter == parameter
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {series_label} result for {comparison} at parameter {parameter}, found {len(selected)}"
        )
    return selected[0]


def _documentation_operation_name(comparison: Comparison) -> str:
    """Return the concise operation name used on documentation graphics."""

    # Convert registry names such as "reproject" to the titles shown in public plots and tables
    operation_labels = {
        "clip": "Clipping",
        "reproject": "Reprojection",
        "polygonize": "Polygonization",
        "rasterize": "Rasterization",
        "grid": "Nearest gridding",
        "subsample": "Subsampling",
        "to_pointcloud": "Point cloud conversion",
    }
    return operation_labels[comparison.operation]


def _shared_change_parameter(
    comparison: Comparison,
    baseline_records: list[ComparisonMeasurement],
    current_records: list[ComparisonMeasurement],
) -> int | None:
    """Return the largest parameter shared by every requested series in both results."""

    # Compare only input sizes measured for all three xDEM execution modes and the GDAL CLI on both revisions
    series_labels = ("In memory", "Dask", "Multiprocessing", GDAL_CLI_LABEL)
    parameters = []
    for records in (baseline_records, current_records):
        for series_label in series_labels:
            parameters.append(
                {
                    record.parameter
                    for record in records
                    if record.comparison == comparison.slug and record.series_label == series_label
                }
            )
    if any(not values for values in parameters):
        return None
    shared = set.intersection(*parameters)
    return max(shared) if shared else None


def _format_change(value: float) -> str:
    """Describe one normalized performance change with an explicit direction."""

    if value >= 1:
        return f"{value:.2f}× faster"
    return f"{1 / value:.2f}× slower"


def performance_change_markdown(baseline_result: Any, current_result: Any) -> str:
    """Compare execution-mode time before and after a change, normalized to the GDAL CLI."""

    baseline_machine = str(baseline_result.params.get("machine", "unknown"))
    current_machine = str(current_result.params.get("machine", "unknown"))
    if baseline_machine != current_machine:
        raise ValueError(f"Performance results use different machines: {baseline_machine} and {current_machine}")

    # Normalizing each revision to its own GDAL CLI result limits machine-wide timing drift
    baseline_available = set(baseline_result.get_all_result_keys())
    current_available = set(current_result.get_all_result_keys())

    # Build the report only from operations with all three measurements on both revisions
    shared_comparisons = []
    baseline_records = []
    current_records = []
    for comparison in _gdal_comparisons():
        if not _required_benchmark_keys((comparison,)).issubset(baseline_available & current_available):
            continue
        try:
            baseline_comparison = collect_comparison_measurements(baseline_result, (comparison,))
            current_comparison = collect_comparison_measurements(current_result, (comparison,))
        except ValueError:
            # Interrupted runs may contain keys whose parameter series still has missing values
            continue
        shared_comparisons.append(comparison)
        baseline_records.extend(baseline_comparison)
        current_records.extend(current_comparison)

    # Group GDAL-normalized timings by execution mode before summarizing across operations
    execution_modes = ("In memory", "Dask", "Multiprocessing")
    ratios: dict[str, list[tuple[Comparison, int, float, float]]] = {name: [] for name in execution_modes}
    for comparison in shared_comparisons:
        parameter = _shared_change_parameter(comparison, baseline_records, current_records)
        if parameter is None:
            continue

        baseline_gdal = _measurement_at(baseline_records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
        current_gdal = _measurement_at(current_records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
        for execution_mode in execution_modes:
            baseline_ratio = (
                _measurement_at(baseline_records, comparison.slug, execution_mode, parameter).end_to_end_time_s
                / baseline_gdal
            )
            current_ratio = (
                _measurement_at(current_records, comparison.slug, execution_mode, parameter).end_to_end_time_s
                / current_gdal
            )
            ratios[execution_mode].append((comparison, parameter, baseline_ratio, current_ratio))

    if not all(ratios.values()):
        raise ValueError("No GDAL CLI comparison is complete in both performance results")

    # Geometric means combine multiplicative time ratios without favoring slower operations
    summary_rows = []
    for execution_mode in execution_modes:
        execution_mode_ratios = ratios[execution_mode]
        count = len(execution_mode_ratios)
        baseline_mean = math.prod(row[2] for row in execution_mode_ratios) ** (1 / count)
        current_mean = math.prod(row[3] for row in execution_mode_ratios) ** (1 / count)
        summary_rows.append((execution_mode, baseline_mean, current_mean, baseline_mean / current_mean))

    shared_rows = ratios[execution_modes[0]]
    operation_names = ", ".join(_documentation_operation_name(row[0]) for row in shared_rows)
    lines = [
        "# Performance change relative to the GDAL CLI",
        "",
        f"Baseline `{baseline_result.commit_hash}` → current `{current_result.commit_hash}` on `{current_machine}`.",
        "",
        "Elapsed time is normalized to the GDAL CLI for the same revision and input. Lower ratios are better. "
        "The summary "
        f"is the geometric mean across {operation_names}, using the largest parameter shared by every execution mode.",
        "",
        "| xDEM execution mode | Before / GDAL CLI | After / GDAL CLI | Change |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {execution_mode} | {baseline:.2f}× | {current:.2f}× | {_format_change(change)} |"
        for execution_mode, baseline, current, change in summary_rows
    )
    lines.extend(
        [
            "",
            "## Per-operation values",
            "",
            "| Operation | Input | Execution mode | Before / GDAL CLI | After / GDAL CLI | Change |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for execution_mode in execution_modes:
        for comparison, parameter, baseline, current in ratios[execution_mode]:
            lines.append(
                f"| {_documentation_operation_name(comparison)} | {parameter} | {execution_mode} | "
                f"{baseline:.2f}× | {current:.2f}× | {_format_change(baseline / current)} |"
            )
    return "\n".join(lines) + "\n"


def _plot_time_relative_to_gdal(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare elapsed execution-mode time with the GDAL CLI on the largest shared dem."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    execution_modes = ("In memory", "Dask", "Multiprocessing")
    bar_width = 0.24
    positions = list(range(len(comparisons)))
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")

    for execution_mode_index, execution_mode in enumerate(execution_modes):
        ratios = []
        for comparison in comparisons:
            parameter = _largest_shared_parameter(comparison, records)
            gdal_time = _measurement_at(records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
            execution_mode_time = _measurement_at(
                records,
                comparison.slug,
                execution_mode,
                parameter,
            ).end_to_end_time_s
            ratios.append(execution_mode_time / gdal_time)

        # Group execution modes around each operation so their GDAL CLI ratio is easy to compare
        offset = (execution_mode_index - 1) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            ratios,
            width=bar_width,
            label=execution_mode,
            color=SERIES_COLORS[execution_mode],
        )
        axis.bar_label(bars, fmt="%.1f×", padding=3, fontsize=8)

    # The GDAL CLI equals one by definition and remains visible when every execution mode is slower
    axis.axhline(
        1.0,
        color=SERIES_COLORS[GDAL_CLI_LABEL],
        linestyle="--",
        linewidth=1.2,
        label=GDAL_CLI_LABEL,
    )
    axis.set_xticks(positions, [_documentation_operation_name(comparison) for comparison in comparisons])
    axis.set_ylabel("Elapsed time relative to GDAL CLI")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=4)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_memory_increase_by_operation(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare additional process-tree memory on the largest shared GDAL workload."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    series_labels = ("In memory", "Dask", "Multiprocessing", GDAL_CLI_LABEL)
    bar_width = 0.2
    positions = list(range(len(comparisons)))
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")

    for series_index, series_label in enumerate(series_labels):
        memory_increases = []
        for comparison in comparisons:
            parameter = _largest_shared_parameter(comparison, records)
            measurement = _measurement_at(records, comparison.slug, series_label, parameter)
            memory_increases.append(measurement.process_tree_mem_increase_mb)

        # Show every implementation directly because each value already excludes its initialized baseline
        offset = (series_index - (len(series_labels) - 1) / 2) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            memory_increases,
            width=bar_width,
            label=series_label,
            color=SERIES_COLORS[series_label],
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)

    axis.set_xticks(positions, [_documentation_operation_name(comparison) for comparison in comparisons])
    axis.set_ylabel("Additional combined memory (MB)")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=4)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_memory_increase_by_dem_size(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare additional process-tree memory as the dem size increases."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    column_count = 3
    row_count = math.ceil(len(comparisons) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.5, 3.4 * row_count),
        layout="constrained",
        squeeze=False,
    )
    for axis, comparison in zip(axes.flat, comparisons):
        for series_label, _ in comparison.series:
            # Sorting protects the displayed dependency from ASV storage order
            selected = sorted(
                (
                    record
                    for record in records
                    if record.comparison == comparison.slug and record.series_label == series_label
                ),
                key=lambda record: record.parameter,
            )
            axis.plot(
                [record.parameter for record in selected],
                [record.process_tree_mem_increase_mb for record in selected],
                marker="o",
                label=series_label,
                color=SERIES_COLORS[series_label],
            )

        axis.set_title(_documentation_operation_name(comparison))
        axis.set_xlabel("Size of dem (pixels per side)")
        axis.set_ylabel("Additional combined memory (MB)")
        axis.grid(alpha=0.3)

    # Hide unused panels when the number of documented operations does not fill the final row
    for axis in axes.flat[len(comparisons) :]:
        axis.set_visible(False)

    # One legend applies to every panel and leaves the operation curves uncluttered
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def _plot_time_relative_to_pdal(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare xDEM elapsed time with PDAL CLI on the largest shared point workload."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _pdal_comparisons()
    execution_modes = ("Dask", "Multiprocessing")
    bar_width = 0.32
    positions = list(range(len(comparisons)))
    figure, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")

    for execution_mode_index, execution_mode in enumerate(execution_modes):
        ratios = []
        for comparison in comparisons:
            parameter = _largest_shared_parameter(comparison, records)
            pdal_time = _measurement_at(records, comparison.slug, PDAL_CLI_LABEL, parameter).end_to_end_time_s
            xdem_time = _measurement_at(
                records,
                comparison.slug,
                execution_mode,
                parameter,
            ).end_to_end_time_s
            ratios.append(xdem_time / pdal_time)

        # Center both GeoUtils modes around each dem-to-point operation
        offset = (execution_mode_index - 0.5) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            ratios,
            width=bar_width,
            label=execution_mode,
            color=SERIES_COLORS[execution_mode],
        )
        axis.bar_label(bars, fmt="%.1f×", padding=3, fontsize=8)

    # PDAL equals one by definition and gives both xDEM ratios a common reference
    axis.axhline(
        1.0,
        color=SERIES_COLORS[PDAL_CLI_LABEL],
        linestyle="--",
        linewidth=1.2,
        label=PDAL_CLI_LABEL,
    )
    axis.set_xticks(positions, [_documentation_operation_name(comparison) for comparison in comparisons])
    axis.set_ylabel("Elapsed time relative to PDAL CLI")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=3)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _result_payload(result: Any, records: list[ComparisonMeasurement]) -> dict[str, Any]:
    """Return measurements and enough run metadata to interpret them later."""

    return {
        "metadata": {
            "commit": result.commit_hash,
            "date": result.date,
            "environment": str(getattr(result, "_env_name", "")),
            "machine": result.params,
            "operation_time": "wall-clock time after execution-mode initialization",
            "end_to_end_time": "elapsed wall-clock time from prepared inputs through completed output",
            "memory": "peak process-tree memory minus its initialized baseline",
        },
        "measurements": [asdict(record) for record in records],
    }


def render_documentation_snapshot(
    result: Any,
    output_directory: Path,
) -> list[ComparisonMeasurement]:
    """Write the documentation graphics and their complete numeric source."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "xdem-matplotlib-cache"))
    output_directory.mkdir(parents=True, exist_ok=True)
    records = collect_comparison_measurements(result)

    # Documentation uses concise summaries while the ASV site retains every detailed panel
    _plot_time_relative_to_gdal(records, output_directory / DOCUMENTATION_TIME_PLOT)
    _plot_memory_increase_by_operation(records, output_directory / DOCUMENTATION_MEMORY_PLOT)
    _plot_memory_increase_by_dem_size(records, output_directory / DOCUMENTATION_MEMORY_SCALING_PLOT)
    _plot_time_relative_to_pdal(records, output_directory / DOCUMENTATION_PDAL_PLOT)
    (output_directory / DOCUMENTATION_DATA).write_text(
        json.dumps(_result_payload(result, records), indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def _operation_and_method_label(comparison: Comparison) -> str:
    """Return a concise operation label including its fixed method or strategy."""

    label = comparison.benchmark.operation.display_name
    if comparison.method is not None:
        method = comparison.choice_label("method", comparison.method)
        label = f"{label} — {method}"
    if comparison.strategy is not None:
        label = f"{label} — {comparison.choice_label('strategy', comparison.strategy)}"
    return label


def _workload_label(comparison: Comparison, parameter: int) -> str:
    """Describe every input size that identifies one representative benchmark workload."""

    return comparison.workload(parameter)


def _workload_html(comparison: Comparison, parameter: int) -> str:
    """Place each input dimension on a separate line in an HTML workload cell."""

    parts = _workload_label(comparison, parameter).split("; ")
    return '<span class="workload-list">' + "".join(f"<span>{html.escape(part)}</span>" for part in parts) + "</span>"


def _largest_parameter_for_series(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    series_labels: Iterable[str],
) -> int:
    """Return the largest parameter measured for every selected plot series."""

    parameters = [
        {
            record.parameter
            for record in records
            if record.comparison == comparison.slug and record.series_label == series_label
        }
        for series_label in series_labels
    ]
    if not parameters or any(not values for values in parameters):
        raise ValueError(f"No complete parameter values found for {comparison.slug}")
    shared = set.intersection(*parameters)
    if not shared:
        raise ValueError(f"No shared parameter found for {comparison.slug}")
    return max(shared)


def _group_operation_rows(rows: list[tuple[Comparison, str]], *, column_count: int) -> str:
    """Group operation table rows by their input and output data types."""

    bodies = []
    rendered_rows = 0
    print("_group_operation_rows", len(rows), column_count)
    for comparison, row in rows:
        print(OPERATION_GROUPS.get(comparison.operation))
        print(comparison)
        print(row)

    for group in OPERATION_GROUP_ORDER:
        print("#", group)
        selected = [row for comparison, row in rows if OPERATION_GROUPS.get(comparison.operation) == group]
        if not selected:
            continue
        print("selected", selected)
        rendered_rows += len(selected)
        bodies.append(
            f'<tbody><tr class="operation-group"><th colspan="{column_count}" scope="rowgroup">{group}</th></tr>'
            f"{''.join(selected)}</tbody>"
        )

    if rendered_rows != len(rows):
        raise ValueError("Every benchmark operation must belong to one input/output group")
    return "".join(bodies)


def _format_measurement_cell(
    measurement: ComparisonMeasurement | None,
    *,
    reference: ComparisonMeasurement | None,
    reference_label: str,
    best_time: float | None,
    lowest_memory: float | None,
) -> str:
    """Render elapsed time and additional memory with optional relative values."""

    if measurement is None:
        return '<span class="unavailable" aria-label="Not supported">—</span>'
    time_class = " best" if best_time is not None and math.isclose(measurement.end_to_end_time_s, best_time) else ""
    memory_class = (
        " best"
        if lowest_memory is not None and math.isclose(measurement.process_tree_mem_increase_mb, lowest_memory)
        else ""
    )
    time_relative = ""
    memory_relative = ""
    if reference is not None:
        time_relative = (
            f'<span class="ratio">{measurement.end_to_end_time_s / reference.end_to_end_time_s:.2f}× '
            f"{reference_label}</span>"
        )
        if reference.process_tree_mem_increase_mb > 0:
            memory_relative = (
                f'<span class="ratio">'
                f"{measurement.process_tree_mem_increase_mb / reference.process_tree_mem_increase_mb:.2f}× "
                f"{reference_label}</span>"
            )
    return "".join(
        [
            f'<span class="metric{time_class}"><span class="metric-label">Time</span>'
            f"{measurement.end_to_end_time_s:.3g} s{time_relative}</span>",
            f'<span class="metric{memory_class}"><span class="metric-label">Memory</span>'
            f"{measurement.process_tree_mem_increase_mb:.3g} MB{memory_relative}</span>",
        ]
    )


def _inmem_implementation_measurements(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    parameter: int | None = None,
) -> tuple[int, dict[str, ComparisonMeasurement]]:
    """Return in-memory xDEM engines and any external GDAL result at one shared workload."""

    selected_labels = [
        label
        for label, _ in comparison.series
        if label == GDAL_CLI_LABEL or comparison.by == "engine" or label == "In memory"
    ]
    if parameter is None:
        parameter = _largest_parameter_for_series(comparison, records, selected_labels)
    selected = (
        record
        for record in records
        if record.comparison == comparison.slug
        and record.parameter == parameter
        and record.series_label in selected_labels
    )
    by_implementation: dict[str, ComparisonMeasurement] = {}
    for measurement in selected:
        implementation = measurement.external_reference or measurement.calculation_engine
        if implementation is not None:
            by_implementation[implementation] = measurement
    return parameter, by_implementation


def _execution_mode_measurements(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    parameter: int | None = None,
) -> tuple[int, dict[ExecutionMode, ComparisonMeasurement]]:
    """Return GeoUtils execution modes at the largest workload shared by every mode."""

    selected_labels = [
        label for label, class_name in comparison.series if BENCHMARK_CASE_BY_CLASS[class_name].implementation == "xdem"
    ]
    if parameter is None:
        parameter = _largest_parameter_for_series(comparison, records, selected_labels)
    measurements = (
        record
        for record in records
        if record.comparison == comparison.slug
        and record.parameter == parameter
        and record.external_reference is None
        and record.execution_mode is not None
    )
    by_mode: dict[ExecutionMode, ComparisonMeasurement] = {}
    for measurement in measurements:
        assert measurement.execution_mode is not None
        by_mode[measurement.execution_mode] = measurement
    return parameter, by_mode


def _engine_summary_comparisons() -> tuple[Comparison, ...]:
    """Select one representative input axis for every in-memory operation and method."""

    # Prefer a direct engine comparison, then fall back to an execution comparison containing in-memory execution
    selected: dict[tuple[str, str | None], tuple[int, Comparison]] = {}
    for comparison in COMPARISONS:
        if not comparison.summary:
            continue
        priority = 0
        if comparison.by == "engine":
            priority = 2 if comparison.parameter_label == "Size of dem (pixels per side)" else 1
        elif comparison.by == "execution" and any(label == "In memory" for label, _ in comparison.series):
            priority = 1
        if priority == 0:
            continue
        key = (comparison.operation, comparison.method)
        current = selected.get(key)
        if current is None or priority > current[0]:
            selected[key] = (priority, comparison)
    return tuple(item[1] for item in selected.values())


def _engine_summary_table(records: list[ComparisonMeasurement]) -> str:
    """Compare in-memory GeoUtils calculation engines with the external GDAL CLI."""

    rows: list[tuple[Comparison, str]] = []
    columns = ("rasterio", "scipy", "numba", "numpy", "gdal_cli")

    for comparison in _engine_summary_comparisons():
        print(comparison)
        parameter = _summary_reference_parameter(comparison, records)
        _, by_implementation = _inmem_implementation_measurements(comparison, records, parameter)
        gdal = by_implementation.get("gdal_cli")
        best_time = min(measurement.end_to_end_time_s for measurement in by_implementation.values())
        lowest_memory = min(measurement.process_tree_mem_increase_mb for measurement in by_implementation.values())
        cells = [
            _format_measurement_cell(
                by_implementation.get(column),
                reference=gdal if column != "gdal_cli" else None,
                reference_label="GDAL CLI",
                best_time=best_time,
                lowest_memory=lowest_memory,
            )
            for column in columns
        ]
        rows.append(
            (
                comparison,
                "<tr>"
                f'<th scope="row">{html.escape(_operation_and_method_label(comparison))}</th>'
                f"<td>{_workload_html(comparison, parameter)}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>",
            )
        )
    print(len(rows))
    return "".join(
        [
            '<div class="table-wrap"><table><thead>',
            '<tr><th class="row-heading" rowspan="2" scope="col">Operation and method</th>',
            '<th class="workload-heading" rowspan="2" scope="col">Reference workload</th>',
            '<th class="group-heading" colspan="4" scope="colgroup">GeoUtils calculation engine</th>',
            '<th class="external-heading group-heading" scope="colgroup">External reference</th></tr>',
            '<tr><th scope="col">Rasterio/GDAL</th><th scope="col">SciPy</th><th scope="col">Numba</th>',
            '<th scope="col">NumPy</th>',
            '<th class="external-heading" scope="col">GDAL CLI</th></tr></thead>',
            _group_operation_rows(rows, column_count=7),
            "</table></div>",
        ]
    )


def _execution_summary_comparisons() -> tuple[Comparison, ...]:
    """Return every plot that directly compares GeoUtils execution modes."""

    return tuple(comparison for comparison in COMPARISONS if comparison.summary and comparison.by == "execution")


def _execution_summary_table(records: list[ComparisonMeasurement]) -> str:
    """Compare in-memory, Dask and Multiprocessing with one fixed calculation engine."""

    rows: list[tuple[Comparison, str]] = []
    columns: tuple[ExecutionMode, ...] = ("inmem", "dask", "multiprocessing")
    for comparison in _execution_summary_comparisons():
        parameter = _summary_reference_parameter(comparison, records)
        _, by_mode = _execution_mode_measurements(comparison, records, parameter)
        inmem = by_mode.get("inmem")
        best_time = min(measurement.end_to_end_time_s for measurement in by_mode.values())
        lowest_memory = min(measurement.process_tree_mem_increase_mb for measurement in by_mode.values())
        cells = [
            _format_measurement_cell(
                by_mode.get(column),
                reference=inmem,
                reference_label="In memory",
                best_time=best_time,
                lowest_memory=lowest_memory,
            )
            for column in columns
        ]
        engine = (
            comparison.choice_label("engine", comparison.calculation_engine)
            if comparison.calculation_engine is not None
            else "No separate engine"
        )
        rows.append(
            (
                comparison,
                "<tr>"
                f'<th scope="row">{html.escape(_operation_and_method_label(comparison))}</th>'
                f"<td>{html.escape(engine)}</td>"
                f"<td>{_workload_html(comparison, parameter)}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>",
            )
        )
    return "".join(
        [
            '<div class="table-wrap"><table><thead>',
            '<tr><th class="row-heading" rowspan="2" scope="col">Operation and method</th>',
            '<th rowspan="2" scope="col">Engine</th>',
            '<th class="workload-heading" rowspan="2" scope="col">Reference workload</th>',
            '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils execution mode</th></tr>',
            '<tr><th scope="col">In memory</th><th scope="col">Dask</th><th scope="col">Multiprocessing</th>',
            "</tr></thead>",
            _group_operation_rows(rows, column_count=6),
            "</table></div>",
        ]
    )


def _summary_reference_parameter(comparison: Comparison, records: list[ComparisonMeasurement]) -> int:
    """Use the common dem size for summary rows and the largest value on other input axes."""

    if comparison.parameter_label == "Size of dem (pixels per side)" and all(
        any(
            record.comparison == comparison.slug and record.series_label == series_label and record.parameter == 2048
            for record in records
        )
        for series_label, _ in comparison.series
    ):
        return 2048
    return _largest_shared_parameter(comparison, records)


def _default_cases(comparison: Comparison, *, execution: ExecutionMode | None = None) -> tuple[Case, ...]:
    """Select GeoUtils cases using the default numerical backend and chunk strategy."""

    cases = []
    for case in comparison.cases:
        if case.implementation != "xdem" or (execution is not None and case.execution != execution):
            continue

        # These operations also benchmark choices that differ from their public defaults
        if comparison.operation == "random_field":
            backend = signature(DEMBase.random_field).parameters["backend"].default
            if case.engine not in (None, backend):
                continue
        if comparison.operation == "grouped_stats":
            if case.strategy not in (None, "auto"):
                continue
        cases.append(case)
    return tuple(cases)


def _default_comparisons() -> tuple[Comparison, ...]:
    """Choose one comparison per operation and method, preferring default Dask cases."""

    selected: dict[tuple[str, str | None], Comparison] = {}
    for comparison in COMPARISONS:
        if not comparison.summary or comparison.by not in ("execution", "strategy"):
            continue
        if not any(case.execution == "dask" and case.implementation == "xdem" for case in comparison.cases):
            continue
        key = (comparison.operation, comparison.method)
        current = selected.get(key)
        preferred_axis = (
            "Number of source points per axis" if comparison.operation == "grid" else "Size of dem (pixels per side)"
        )
        priority = (bool(_default_cases(comparison, execution="dask")), comparison.parameter_label == preferred_axis)
        if current is None:
            selected[key] = comparison
            continue

        # Prefer public defaults, then a comparison that varies the requested input size
        current_priority = (bool(_default_cases(current, execution="dask")), current.parameter_label == preferred_axis)
        if priority > current_priority:
            selected[key] = comparison
    return tuple(selected.values())


def _scaling_comparisons() -> tuple[Comparison, ...]:
    """Choose dem-size comparisons of execution modes with other settings at their defaults."""
    selected: dict[str, Comparison] = {}
    for comparison in COMPARISONS:
        if comparison.by != "execution" or comparison.benchmark.parameter_name != "dem_size":
            continue

        # Use the usual method for each operation, rather than repeating every alternative
        operation = comparison.benchmark.operation
        if operation.large_data_cases and comparison.method != operation.large_data_cases[0].method:
            continue
        cases = _default_cases(comparison)
        if not any(case.execution == "dask" for case in cases):
            continue

        # A comparison may include an external tool or force a non-default chunk strategy
        comparison = replace(comparison, cases=cases)
        current = selected.get(comparison.operation)
        if current is None or len(comparison.cases) > len(current.cases):
            selected[comparison.operation] = comparison
    return tuple(selected.values())


def _headline_summary_table(records: list[ComparisonMeasurement]) -> str:
    """Show exact-size Dask measurements without substituting smaller workloads."""

    # Include supported operations even when no parameter comparison has measured them yet
    comparisons = _default_comparisons()
    represented = {comparison.operation for comparison in comparisons}
    rows: list[tuple[int, str]] = []
    print("records", records)

    for operation in OPERATIONS:
        if operation.name not in represented and any(case.execution == "dask" for case in operation.large_data_cases):
            rows.append(
                (
                    operation.order,
                    f'<tr><th scope="row">{html.escape(operation.display_name)}</th><td>—</td><td>—</td></tr>',
                )
            )
    # Use explicit size parameters so older runs with different fixed defaults cannot be mislabelled
    for comparison in comparisons:
        operation = comparison.benchmark.operation
        label = operation.display_name
        if comparison.method is not None:
            label += f" — {comparison.choice_label('method', comparison.method)}"
        measurements = []
        default_cases = _default_cases(comparison, execution="dask")
        for series_label, class_name in comparison.series:
            case = BENCHMARK_CASE_BY_CLASS[class_name]
            if case not in default_cases:
                continue

            for record in records:

                if record.comparison != comparison.slug or record.series_label != series_label:
                    continue
                if operation.name == "grid":
                    matches = (
                        comparison.parameter_label == "Number of source points per axis"
                        and record.parameter**2 == SUMMARY_POINT_COUNT
                    )
                else:
                    matches = (
                        comparison.parameter_label == "Size of dem (pixels per side)"
                        and record.parameter == SUMMARY_DEM_SIZE
                    )
                    print(comparison.parameter_label, record.parameter, SUMMARY_DEM_SIZE, matches)
                if matches:
                    measurements.append(record)
        measurement = max(measurements, key=lambda item: item.parameter) if measurements else None
        memory = f"{measurement.process_tree_mem_increase_mb:,.3g}" if measurement is not None else "—"
        elapsed = f"{measurement.end_to_end_time_s:,.3g}" if measurement is not None else "—"
        rows.append(
            (operation.order, f'<tr><th scope="row">{html.escape(label)}</th><td>{memory}</td><td>{elapsed}</td></tr>')
        )
    rows.sort(key=lambda item: item[0])
    return "".join(
        [
            '<div class="table-wrap"><table class="summary-table"><thead><tr>',
            '<th scope="col" class="row-heading">Operation / method</th>',
            '<th scope="col">Peak additional RAM (MB)</th><th scope="col">Time (s)</th>',
            "</tr></thead><tbody>",
            *(row for _, row in rows),
            "</tbody></table></div>",
        ]
    )


def _choice_tables(
    dimension: Literal["method", "strategy", "output_driver"], records: list[ComparisonMeasurement]
) -> str:
    """Render representative time and memory for methods, strategies or point output formats."""

    articles = []
    choice_name = {"method": "method", "strategy": "chunk strategy", "output_driver": "output and writer"}[dimension]
    choice_title = choice_name.title() if dimension == "output_driver" else f"GeoUtils {choice_name}"
    for comparison in (item for item in COMPARISONS if item.by == dimension):
        parameter = _summary_reference_parameter(comparison, records)
        measurements = [
            _measurement_at(records, comparison.slug, series_label, parameter) for series_label, _ in comparison.series
        ]
        headers = "".join(f'<th scope="col">{html.escape(label)}</th>' for label, _ in comparison.series)
        cells = "".join(
            "<td>"
            + _format_measurement_cell(
                measurement,
                reference=None,
                reference_label="",
                best_time=None,
                lowest_memory=None,
            )
            + "</td>"
            for measurement in measurements
        )
        articles.append(
            '<article class="choice-card">'
            f"<h3>{html.escape(comparison.title)}</h3>"
            f'<p class="workload"><strong>Reference workload:</strong> '
            f"{_workload_html(comparison, parameter)}</p>"
            '<div class="table-wrap"><table class="choice-table"><thead>'
            f'<tr><th colspan="{len(measurements)}" scope="colgroup">{choice_title}</th></tr>'
            f"<tr>{headers}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>"
            f'<p><a href="{PARAMETER_REPORT_PAGE}#{html.escape(comparison.slug)}">View this comparison plot</a></p>'
            "</article>"
        )
    return "".join(articles)


def _markdown_report(result: Any, *, includes_performance_change: bool = False) -> str:
    """Return a compact Markdown index linking plots and numeric results."""

    machine = result.params.get("machine", "unknown")
    lines = [
        "# GeoUtils comparisons",
        "",
        f"Commit `{result.commit_hash}` on machine `{machine}`",
        "",
        "Operation-only time starts after execution-mode initialization. Elapsed time runs from prepared inputs "
        "through completed output and is the comparable boundary for GeoUtils versus external CLIs. "
        "Additional memory is the process-tree peak minus its initialized baseline.",
        "",
        "Raw values: [CSV](comparisons.csv) · [JSON](comparisons.json)",
    ]
    if includes_performance_change:
        lines.extend(["", f"Before/after summary: [{PERFORMANCE_CHANGE_REPORT}]({PERFORMANCE_CHANGE_REPORT})"])
    for dimension, (section_title, section_description) in COMPARISON_SECTION_DETAILS.items():
        comparisons = tuple(comparison for comparison in COMPARISONS if comparison.by == dimension)
        if not comparisons:
            continue
        lines.extend(["", f"## {section_title}", "", section_description])
        for comparison in comparisons:
            lines.extend(["", f"### {comparison.title}", ""])
            lines.append(f"![{comparison.title}]({comparison.slug}.svg)")
    return "\n".join(lines) + "\n"


_SITE_STYLE = """
:root { color-scheme: light; --ink: #222; --muted: #555; --line: #ddd; --soft: #f5f5f5; }
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: white; font: 16px/1.6 system-ui, sans-serif; }
a { color: #175ca2; text-underline-offset: .15em; }
.site-header { border-bottom: 1px solid var(--line); }
.header-inner { max-width: 1140px; margin: auto; padding: 1rem; }
.brand { color: var(--ink); font-weight: 700; text-decoration: none; }
.site-nav { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: .5rem; }
.site-nav a[aria-current="page"] { color: var(--ink); font-weight: 700; }
main { max-width: 1140px; margin: auto; padding: 1.5rem 1rem 3rem; }
h1 { font-size: 1.9rem; line-height: 1.25; }
h2 { margin-top: 2rem; font-size: 1.4rem; }
h3 { margin-top: 1.5rem; font-size: 1.1rem; }
li { margin: .3rem 0; }
img { display: block; max-width: 100%; height: auto; margin: 1rem 0; }
.run-meta, .section-intro, .comparison-note, .lede { color: var(--muted); }
.run-meta, .ratio, .workload { font-size: .9rem; }
.eyebrow { display: none; }
code { overflow-wrap: anywhere; }
.table-wrap { width: 100%; overflow-x: auto; margin: 1rem 0; }
table { width: 100%; border-collapse: collapse; font-size: .95rem; }
.table-wrap > table:not(.summary-table) { min-width: 780px; }
th, td { padding: .55rem .75rem; border-bottom: 1px solid var(--line); text-align: right; vertical-align: top; }
thead th { background: var(--soft); }
tbody th, .row-heading { text-align: left; }
.operation-group th { background: var(--soft); }
.summary-table { max-width: 900px; }
.summary-table td { white-space: nowrap; }
.metric { display: flex; flex-direction: column; align-items: flex-end; white-space: nowrap; }
.metric + .metric { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid var(--line); }
.metric-label { color: var(--muted); font-size: .75rem; font-weight: 400; }
.ratio { color: var(--muted); font-size: .75rem; font-weight: 400; }
.metric.best { font-weight: 700; }
.metric-guide { display: flex; flex-direction: column; gap: .3rem; margin: 1.5rem 0; color: var(--muted); }
.workload-list { display: flex; flex-direction: column; text-align: left; min-width: 160px; }
.unavailable { color: var(--muted); }
.local-links { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
.choice-card { margin: 1.5rem 0; }
.choice-table th, .choice-table td { text-align: center; }
.choice-table .metric { align-items: center; }
.scaling-layout { display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 2rem; align-items: start; }
.plot-toc { position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow: auto; }
.plot-toc ul { padding-left: 1.4rem; }
.plot-card { margin: 1rem 0; border-bottom: 1px solid var(--line); }
.plot-card summary { cursor: pointer; padding: .6rem 0; }
.plot-kind { margin-left: 1rem; color: var(--muted); font-size: .9rem; }
.history-frame { width: 100%; height: 80vh; min-height: 600px; border: 1px solid var(--line); }
.footer-links { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--line); }
@media (max-width: 900px) {
    .scaling-layout { grid-template-columns: 1fr; }
    .plot-toc { position: static; max-height: none; }
}
"""


def _site_navigation(active_page: str, relative_root: str, include_asv_history: bool) -> str:
    """Return consistent navigation ordered from summary to detailed ASV history."""

    links = [
        ("summary", f"{relative_root}index.html", "Summary"),
        ("references", f"{relative_root}comparisons/{REFERENCE_REPORT_PAGE}", "GDAL / PDAL"),
        ("scaling", f"{relative_root}comparisons/{SCALING_REPORT_PAGE}", "Scaling"),
        ("options", f"{relative_root}comparisons/index.html", "Options"),
    ]
    if include_asv_history:
        links.append(("history", f"{relative_root}{HISTORY_REPORT_PAGE}", "Performance history"))
    rendered_links = []
    for key, url, label in links:
        current = ' aria-current="page"' if key == active_page else ""
        rendered_links.append(f'<a href="{url}"{current}>{label}</a>')
    return "".join(rendered_links)


def _html_page(
    title: str,
    active_page: str,
    body: str,
    *,
    relative_root: str,
    include_asv_history: bool,
    script: str = "",
) -> str:
    """Wrap one generated view in the shared visual style and navigation."""

    navigation = _site_navigation(active_page, relative_root, include_asv_history)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title><style>{_SITE_STYLE}</style></head><body>",
            '<header class="site-header"><div class="header-inner">',
            f'<a class="brand" href="{relative_root}index.html">GeoUtils benchmarks</a>',
            f'<nav class="site-nav" aria-label="Benchmark views">{navigation}</nav>',
            "</div></header>",
            f"<main>{body}</main>",
            script,
            "</body></html>",
        ]
    )


def _run_metadata(result: Any) -> str:
    """Return concise commit and machine metadata for a generated page."""

    machine = html.escape(str(result.params.get("machine", "unknown")))
    commit = html.escape(str(result.commit_hash))
    return f'<p class="run-meta">Commit <code>{commit}</code> on machine <code>{machine}</code></p>'


def _metric_guide(*, include_operation: bool = False, include_ratios: bool = True) -> str:
    """Define the metrics used by fixed-workload tables and scaling plots."""

    items = []
    if include_operation:
        items.append(
            "<span><strong>Operation only</strong> Calculation after the execution mode has been initialized.</span>"
        )
    items.extend(
        [
            "<span><strong>Elapsed time</strong> Prepared inputs through completed output, including worker or CLI "
            "startup.</span>",
            "<span><strong>Additional memory</strong> Combined process-tree peak minus its initialized baseline."
            "</span>",
        ]
    )
    if include_ratios:
        items.append("<span><strong>Ratios</strong> Relative values for the same workload; lower is better.</span>")
    return f'<div class="metric-guide" aria-label="Metric definitions">{"".join(items)}</div>'


def _gdal_reference_note() -> str:
    """Explain the purpose and limits of the external GDAL comparison."""

    return (
        '<p class="comparison-note">Equivalent GDAL commands use matching inputs and output grids. '
        "Reprojection uses a memory budget based on the GeoUtils chunk size.</p>"
    )


def _pdal_reference_note() -> str:
    """Explain the purpose and limits of the external PDAL comparison."""

    return '<p class="comparison-note">PDAL pipelines perform the same dem sampling and point cloud conversion.</p>'


def _comparison_parameters() -> str:
    """List the settings and input characteristics varied by the benchmarks."""

    return "".join(
        [
            "<ul>",
            "<li><strong>Execution backend:</strong> in memory, Dask or Multiprocessing.</li>",
            "<li><strong>Numerical backend:</strong> Rasterio/GDAL, SciPy, Numba, NumPy, GSTools or GPyTorch.</li>",
            "<li><strong>Method:</strong> the algorithm used for each operation.</li>",
            "<li><strong>Chunk strategy:</strong> how results from separate chunks are combined.</li>",
            "<li><strong>Input size:</strong> dem dimensions and point counts.</li>",
            "<li><strong>Chunk size:</strong> pixels or points processed together.</li>",
            "<li><strong>Operation settings:</strong> sample size, group count, pair count and variogram lags.</li>",
            "<li><strong>External tools:</strong> equivalent GDAL and PDAL commands.</li>",
            "</ul>",
        ]
    )


def _reference_html(result: Any, *, include_asv_history: bool = True) -> str:
    """Show the GDAL and PDAL comparisons on their own page."""

    body = "".join(
        [
            "<h1>Comparison with GDAL and PDAL</h1>",
            "<p>These plots compare GeoUtils execution time and peak additional RAM with equivalent GDAL and PDAL "
            "commands, using one worker and one thread.</p>",
            _run_metadata(result),
            "<h2>GDAL</h2>",
            _gdal_reference_note(),
            f'<img src="../documentation/{DOCUMENTATION_TIME_PLOT}" alt="Execution time relative to GDAL">',
            f'<img src="../documentation/{DOCUMENTATION_MEMORY_PLOT}" alt="Peak additional RAM by operation">',
            "<h2>PDAL</h2>",
            _pdal_reference_note(),
            f'<img src="../documentation/{DOCUMENTATION_PDAL_PLOT}" alt="Execution time relative to PDAL">',
        ]
    )
    return _html_page(
        "GeoUtils compared with GDAL and PDAL",
        "references",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
    )


def _comparison_html(
    result: Any,
    records: list[ComparisonMeasurement],
    *,
    includes_performance_change: bool = False,
    include_asv_history: bool = True,
) -> str:
    """Return representative tables comparing each kind of GeoUtils option."""

    change_link = (
        f'<a href="{PERFORMANCE_CHANGE_REPORT}">View the before/after performance summary</a> · '
        if includes_performance_change
        else ""
    )
    body = "".join(
        [
            '<section class="hero"><span class="eyebrow">Options</span>',
            "<h1>Compare GeoUtils options</h1>",
            '<p class="lede">These tables compare execution modes, numerical backends, methods and chunk strategies '
            "at fixed workloads, using one worker and one thread.</p>",
            _run_metadata(result),
            f'<p><a href="{PARAMETER_REPORT_PAGE}">All option and input-size plots</a></p>',
            '<div class="local-links"><a href="#execution-modes">Execution modes</a>',
            '<a href="#calculation-engines">Calculation engines</a><a href="#operation-methods">Methods</a>',
            '<a href="#chunk-strategies">Chunk strategies</a><a href="#point-output-formats">Point output formats</a>',
            "</div></section>",
            _metric_guide(),
            '<section class="option-section" id="execution-modes">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['execution'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["execution"][1])}</p>',
            _execution_summary_table(records),
            "</section>",
            '<section class="option-section" id="calculation-engines">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['engine'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["engine"][1])} GDAL CLI '
            "is shown separately where an equivalent external command exists.</p>",
            _engine_summary_table(records),
            "</section>",
            '<section class="option-section" id="operation-methods">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['method'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["method"][1])} Methods can answer '
            "different numerical questions, so these costs are not a quality ranking.</p>",
            f'<div class="choice-grid">{_choice_tables("method", records)}</div></section>',
            '<section class="option-section" id="chunk-strategies">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['strategy'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["strategy"][1])}</p>',
            f'<div class="choice-grid">{_choice_tables("strategy", records)}</div></section>',
            '<section class="option-section" id="point-output-formats">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['output_driver'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["output_driver"][1])}</p>',
            f'<div class="choice-grid">{_choice_tables("output_driver", records)}</div></section>',
            '<p class="footer-links">',
            change_link,
            '<a href="comparisons.csv">Download CSV</a> · <a href="comparisons.json">Download JSON</a></p>',
        ]
    )

    return _html_page(
        "Compare GeoUtils options",
        "options",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
    )


def _scaling_html(result: Any, *, include_asv_history: bool = True) -> str:
    """Show collapsible dem-size plots comparing execution modes with default options."""

    toc = []
    plots = []
    for comparison in _scaling_comparisons():
        slug = html.escape(comparison.slug)
        label = html.escape(comparison.benchmark.operation.display_name)
        toc.append(f'<li><a href="#{slug}">{label}</a></li>')
        plots.append(
            f'<details class="plot-card" id="{slug}"><summary><strong>{label}</strong></summary>'
            f'<img loading="lazy" src="default-{slug}.svg" '
            f'alt="Time and peak additional RAM for {label} as dem size grows"></details>'
        )
    body = "".join(
        [
            "<h1>Scaling with dem size</h1>",
            "<p>These plots show how execution time and peak additional RAM grow with dem size, comparing "
            "in-memory, Dask and Multiprocessing execution with default options.</p>",
            _run_metadata(result),
            "<p>DEM width and height vary together; chunk size stays fixed. Select an operation to show its plot.</p>",
            '<div class="scaling-layout"><nav class="plot-toc" aria-label="Scaling operations">',
            "<strong>Operations</strong><ul>",
            *toc,
            "</ul></nav><div>",
            *plots,
            "</div></div>",
        ]
    )
    script = (
        "<script>function openLinkedPlot(){const id=location.hash.slice(1);const target=document.getElementById(id);"
        "if(target&&target.tagName==='DETAILS')target.open=true;}addEventListener('hashchange',openLinkedPlot);"
        "openLinkedPlot();</script>"
    )
    return _html_page(
        "GeoUtils scaling benchmarks",
        "scaling",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
        script=script,
    )


def _parameter_html(
    result: Any,
    *,
    include_asv_history: bool = True,
) -> str:
    """Return collapsible plots grouped by the input quantity that grows."""

    dimension_labels = {
        "execution": "Execution modes",
        "engine": "Calculation engines",
        "method": "Methods",
        "strategy": "Chunk strategies",
        "output_driver": "Point output formats",
    }
    toc = []
    sections = []
    for parameter_label, (section_title, section_description) in SCALING_SECTION_DETAILS.items():
        comparisons = tuple(comparison for comparison in COMPARISONS if comparison.parameter_label == parameter_label)
        if not comparisons:
            continue
        section_id = parameter_label.lower().replace(" ", "-").replace("(", "").replace(")", "")
        plot_links = "".join(
            f'<li><a href="#{html.escape(comparison.slug)}">{html.escape(comparison.title)}</a></li>'
            for comparison in comparisons
        )
        toc.append(f'<li><a href="#{section_id}">{html.escape(section_title)}</a><ul>{plot_links}</ul></li>')
        plots = []
        for comparison in comparisons:
            slug = html.escape(comparison.slug)
            title = html.escape(comparison.title)
            plots.append(
                f'<details class="plot-card" id="{slug}"><summary><span><strong>{title}</strong>'
                "</span>"
                f'<span class="chip plot-kind">{dimension_labels[comparison.by]}</span></summary>'
                f'<div class="plot-content"><img loading="lazy" src="{slug}.svg" alt="{title}"></div></details>'
            )
        sections.append(
            f'<section class="scaling-section" id="{section_id}"><h2>{html.escape(section_title)}</h2>'
            f'<p class="section-intro">{html.escape(section_description)}</p>{"".join(plots)}</section>'
        )
    body = "".join(
        [
            '<section class="hero"><span class="eyebrow">Options</span>',
            "<h1>Option and input-size plots</h1>",
            '<p class="lede">These plots compare GeoUtils options as input size, chunk size or operation settings '
            "vary. Select a comparison to show its plot.</p>",
            _run_metadata(result),
            "</section>",
            _metric_guide(include_operation=True, include_ratios=False),
            '<div class="scaling-layout"><nav class="plot-toc" aria-label="Scaling sections">',
            "<strong>Input varied</strong><ul>",
            *toc,
            "</ul></nav><div>",
            *sections,
            '<p class="footer-links"><a href="comparisons.csv">Download CSV</a> · '
            '<a href="comparisons.json">Download JSON</a></p></div></div>',
        ]
    )
    script = (
        "<script>function openLinkedPlot(){const id=location.hash.slice(1);const target=document.getElementById(id);"
        "if(target&&target.tagName==='DETAILS')target.open=true;}addEventListener('hashchange',openLinkedPlot);"
        "openLinkedPlot();</script>"
    )
    return _html_page(
        "GeoUtils option and parameter plots",
        "options",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
        script=script,
    )


def _history_html(result: Any) -> str:
    """Embed the native ASV report below the shared GeoUtils navigation."""

    body = "".join(
        [
            '<section class="history-intro"><span class="eyebrow">History</span>',
            "<h1>Performance history</h1>",
            '<p class="lede">This page tracks execution time and memory usage across commits using ASV.</p>',
            _run_metadata(result),
            "</section>",
            '<iframe class="history-frame" src="asv/index.html" title="ASV performance history"></iframe>',
            '<p class="footer-links"><a href="asv/index.html">Open the native ASV report</a></p>',
        ]
    )
    return _html_page(
        "GeoUtils performance history",
        "history",
        body,
        relative_root="",
        include_asv_history=True,
    )


def _site_index(
    result: Any,
    records: list[ComparisonMeasurement],
    *,
    includes_performance_change: bool = False,
    include_asv_history: bool = True,
) -> str:
    """Show the compared parameters and a table of default Dask measurements."""

    # Link detailed results in the same order as the page navigation
    links = [
        f'<li><a href="comparisons/{REFERENCE_REPORT_PAGE}">Comparison with GDAL and PDAL</a></li>',
        f'<li><a href="comparisons/{SCALING_REPORT_PAGE}">Scaling with dem size</a></li>',
        '<li><a href="comparisons/index.html">Comparison with options</a></li>',
    ]
    if include_asv_history:
        links.append(f'<li><a href="{HISTORY_REPORT_PAGE}">Performance history</a></li>')
    change_link = (
        f'<p><a href="comparisons/{PERFORMANCE_CHANGE_REPORT}">Before/after branch summary</a></p>'
        if includes_performance_change
        else ""
    )

    body = "".join(
        [
            "<h1>GeoUtils benchmarks</h1>",
            "<p>These benchmarks measure execution time and peak additional RAM for GeoUtils operations, "
            "with comparisons across settings, input sizes and external tools.</p>",
            (
                "<p><strong>Preview only: values are synthetic.</strong></p>"
                if getattr(result, "_env_name", "") == "preview"
                else ""
            ),
            _run_metadata(result),
            "<p>The benchmarks compare:</p>",
            _comparison_parameters(),
            '<h2 id="performance-summary">Summary</h2>',
            "<p>Dask execution with the default numerical backend and chunk strategy, on a "
            "10,000 × 10,000 dem or 100,000,000 input points. Substantially different methods have separate rows.</p>",
            "<p>Time includes completed output. Peak additional RAM is the combined process-tree peak minus its "
            "initialized baseline. A dash means that this workload has not been measured.</p>",
            _headline_summary_table(records),
            "<h2>Detailed comparisons</h2><ol>",
            *links,
            "</ol>",
            change_link,
            '<p class="footer-links"><a href="comparisons/comparisons.csv">Download CSV</a> · '
            '<a href="comparisons/comparisons.json">Download JSON</a></p>',
        ]
    )
    return _html_page(
        "GeoUtils benchmark results",
        "summary",
        body,
        relative_root="",
        include_asv_history=include_asv_history,
    )


def render_comparisons(
    result: Any,
    website_directory: Path,
    *,
    baseline_result: Any | None = None,
    include_asv_history: bool = True,
) -> list[ComparisonMeasurement]:
    """Write comparison plots and one landing page beside the native ASV site."""

    # Only expose the history shell when ASV has already published its native report
    include_asv_history = include_asv_history and (website_directory / "asv" / "index.html").is_file()

    # All generated material stays under the gitignored ASV website directory
    report_directory = website_directory / COMPARISON_REPORT_DIRECTORY
    report_directory.mkdir(parents=True, exist_ok=True)
    records = collect_comparison_measurements(result)
    for comparison in COMPARISONS:
        _plot_comparison(comparison, records, report_directory / f"{comparison.slug}.svg")

    # Scaling compares execution modes with the usual method, numerical backend and chunk strategy
    for comparison in _scaling_comparisons():
        _plot_comparison(
            comparison,
            records,
            report_directory / f"default-{comparison.slug}.svg",
            include_operation=False,
        )

    # Use the same reference graphics in the website and documentation
    render_documentation_snapshot(result, website_directory / "documentation")
    (report_directory / REFERENCE_REPORT_PAGE).write_text(
        _reference_html(result, include_asv_history=include_asv_history), encoding="utf-8"
    )
    (report_directory / PARAMETER_REPORT_PAGE).write_text(
        _parameter_html(result, include_asv_history=include_asv_history), encoding="utf-8"
    )

    # JSON retains run metadata while CSV remains convenient for independent analysis
    payload = _result_payload(result, records)
    (report_directory / "comparisons.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Write explicit columns so the stable schema does not depend on plotting code
    with open(report_directory / "comparisons.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    includes_performance_change = baseline_result is not None
    if baseline_result is not None:
        (report_directory / PERFORMANCE_CHANGE_REPORT).write_text(
            performance_change_markdown(baseline_result, result), encoding="utf-8"
        )
    (report_directory / "README.md").write_text(
        _markdown_report(result, includes_performance_change=includes_performance_change), encoding="utf-8"
    )
    (report_directory / "index.html").write_text(
        _comparison_html(
            result,
            records,
            includes_performance_change=includes_performance_change,
            include_asv_history=include_asv_history,
        ),
        encoding="utf-8",
    )
    (report_directory / SCALING_REPORT_PAGE).write_text(
        _scaling_html(result, include_asv_history=include_asv_history),
        encoding="utf-8",
    )

    # The custom summary stays separate from files generated internally by ASV
    website_directory.mkdir(parents=True, exist_ok=True)
    (website_directory / "index.html").write_text(
        _site_index(
            result,
            records,
            includes_performance_change=includes_performance_change,
            include_asv_history=include_asv_history,
        ),
        encoding="utf-8",
    )
    if include_asv_history:
        (website_directory / HISTORY_REPORT_PAGE).write_text(_history_html(result), encoding="utf-8")
    return records


def _render_preview(website_directory: Path) -> None:
    """Write a complete comparison website using deterministic fake measurements."""

    # Two fake revisions also exercise the generated before/after performance table
    current_result = _PreviewResult()
    baseline_result = _PreviewResult(commit_hash="preview-baseline", xdem_scale=1.25)
    render_comparisons(
        current_result,
        website_directory,
        baseline_result=baseline_result,
        include_asv_history=(website_directory / "asv" / "index.html").is_file(),
    )


def main() -> None:
    """Select one saved ASV result and render its requested report outputs."""

    # This command runs explicitly after asv publish and is not called during ASV measurement
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=Path("benchmarks/results/asv/results"))
    parser.add_argument("--website-directory", type=Path)
    parser.add_argument("--doc-dir", type=Path)
    parser.add_argument("--doc-only", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="render a complete website from deterministic fake data",
    )
    parser.add_argument("--machine")
    parser.add_argument("--commit")
    parser.add_argument("--baseline-commit")
    parser.add_argument("--environment")
    args = parser.parse_args()

    # Preview mode needs neither saved ASV results nor benchmark execution
    if args.preview:
        website_directory = args.website_directory or PREVIEW_WEBSITE_DIRECTORY
        _render_preview(website_directory)
        print(f"Benchmark preview written to {website_directory / 'index.html'}")
        return

    # Optional filters select a reproducible record from histories with several machines
    result = select_asv_result(
        args.results_directory,
        machine=args.machine,
        commit=args.commit,
        environment=args.environment,
    )

    baseline_result = None
    if args.baseline_commit is not None:
        baseline_result = select_asv_result(
            args.results_directory,
            machine=str(result.params.get("machine", "")),
            commit=args.baseline_commit,
            environment=str(getattr(result, "_env_name", "")),
            require_complete=False,
        )
    if not args.doc_only:
        website_directory = args.website_directory or Path("benchmarks/results/asv/html")
        render_comparisons(result, website_directory, baseline_result=baseline_result)
    if args.doc_dir is not None and (
        args.doc_only or args.doc_dir.resolve() != (website_directory / "documentation").resolve()
    ):
        render_documentation_snapshot(result, args.doc_dir)
    elif args.doc_only:
        parser.error("--doc-only requires --doc-dir")


if __name__ == "__main__":
    main()

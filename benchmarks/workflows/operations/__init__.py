"""Automatically collect benchmark operations defined in files within ``operations/``."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from benchmarks.workflows.config import RuntimeConfig, fixed_config
from benchmarks.workflows.core import Benchmark, Case, Comparison, Operation

############################
# Automatic discovery
############################


@dataclass(frozen=True)
class OperationCatalog:
    """Collect the operation declarations used by runners, ASV and reports."""

    modules: tuple[ModuleType, ...]
    operations: tuple[Operation, ...]
    benchmarks: tuple[Benchmark, ...]
    comparisons: tuple[Comparison, ...]
    cases: tuple[Case, ...]


def discover_operation_modules() -> tuple[ModuleType, ...]:
    """Import operation modules in a stable name order."""

    # Find every operation module
    module_names = sorted(module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_"))

    # Import the declarations, and apply their custom order
    modules = tuple(importlib.import_module(f"{__name__}.{module_name}") for module_name in module_names)
    return tuple(sorted(modules, key=lambda module: (getattr(module, "ORDER", 100), module.__name__)))


def unique_by(values: Iterable[Any], attribute: str, kind: str) -> tuple[Any, ...]:
    """Reject duplicate declaration IDs."""

    unique: dict[str, Any] = {}
    for value in values:
        identifier = getattr(value, attribute)
        if identifier in unique:
            raise ValueError(f"Duplicate {kind} ID: {identifier}")
        unique[identifier] = value
    return tuple(unique.values())


def unique_cases(benchmarks: tuple[Benchmark, ...]) -> tuple[Case, ...]:
    """Reject duplicate benchmark case IDs."""

    unique: dict[str, Case] = {}
    for benchmark in benchmarks:
        for case in benchmark.cases:
            class_name = benchmark.benchmark_class(case)
            if class_name in unique:
                raise ValueError(f"Duplicate benchmark case ID: {class_name}")
            unique[class_name] = case
    return tuple(unique.values())


def fixed_benchmarks(operations: tuple[Operation, ...], benchmarks: tuple[Benchmark, ...]) -> tuple[Benchmark, ...]:
    """Define fixed benchmarks for operations without parameter sweeps."""

    parameterized = {benchmark.operation.name for benchmark in benchmarks if benchmark.parameter_name is not None}
    return tuple(
        Benchmark(operation, operation.large_data_cases, fixed_config)
        for operation in operations
        if operation.name not in parameterized and operation.large_data_cases
    )


def collect_operation_modules(modules: Iterable[Any]) -> OperationCatalog:
    """Collect declarations from the given operation modules."""

    # Freeze the discovered order before collecting each kind of declaration
    ordered_modules = cast(tuple[ModuleType, ...], tuple(modules))
    operations = unique_by(
        (operation for module in ordered_modules for operation in getattr(module, "OPERATIONS", ())),
        "name",
        "operation",
    )
    declared = unique_by(
        (benchmark for module in ordered_modules for benchmark in getattr(module, "BENCHMARKS", ())),
        "id",
        "benchmark",
    )

    # Fixed operations use the same Benchmark model and generic ASV harness as parameter sweeps
    benchmarks = (*declared, *fixed_benchmarks(operations, declared))
    comparisons = unique_by(
        (comparison for module in ordered_modules for comparison in getattr(module, "COMPARISONS", ())),
        "slug",
        "comparison",
    )

    # Check that every benchmark refers to the discovered operation object
    operation_names = {operation.name for operation in operations}
    for benchmark in benchmarks:
        if benchmark.operation.name not in operation_names:
            raise ValueError(f"Benchmark {benchmark.id!r} has no operation handler")
    cases = unique_cases(benchmarks)
    return OperationCatalog(ordered_modules, operations, benchmarks, comparisons, cases)


# Discover once at import so ASV and the renderer receive the same benchmark declarations
CATALOG = collect_operation_modules(discover_operation_modules())
OPERATION_MODULES = CATALOG.modules
OPERATIONS = CATALOG.operations
BENCHMARKS = CATALOG.benchmarks
COMPARISONS = CATALOG.comparisons
BENCHMARK_CASES = CATALOG.cases

# Index declarations for tests, ASV, reports and large-data checks
OPERATION_BY_NAME = {operation.name: operation for operation in OPERATIONS}
BENCHMARK_BY_ID = {benchmark.id: benchmark for benchmark in BENCHMARKS}
BENCHMARK_BY_CLASS = {
    benchmark.benchmark_class(case): benchmark for benchmark in BENCHMARKS for case in benchmark.cases
}
BENCHMARK_CASE_BY_CLASS = {
    benchmark.benchmark_class(case): case for benchmark in BENCHMARKS for case in benchmark.cases
}
LARGE_DATA_CASES = {
    f"{case.execution}-{operation.name}": (operation, case)
    for operation in sorted(OPERATIONS, key=lambda item: item.order)
    for case in operation.large_data_cases
}


def format_api_label(operation: Operation, case: Case) -> str:
    """Format an operation call from its execution options."""

    # A declared call spells out operator arguments without constructing optional implementations during discovery
    if operation.label_options and all(f"api_{name}" in case.labels for name in operation.label_options):
        arguments = ", ".join(f"{name}={case.labels[f'api_{name}']}" for name in operation.label_options)
        return f"{operation.public_call_name}({arguments})"

    # Build the public options once so labels remain aligned with the measured call
    values = {**case.options, "operation": operation.name}
    options = operation.option_builder(case, RuntimeConfig(workload=values))

    # Keep only options chosen by the operation for concise report labels
    selected = ((name, options[name]) for name in operation.label_options if name in options)
    arguments = ", ".join(f"{name}={case.labels.get(f'api_{name}', repr(value))}" for name, value in selected)
    return f"{operation.public_call_name}({arguments})"

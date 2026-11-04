"""This module contains the core objects to facilitate defining benchmarks, shared by ASV and large data tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from benchmarks.workflows.config import RuntimeConfig, runtime_config

ExecutionMode = Literal["inmem", "dask", "multiprocessing"]

ComparisonDimension = Literal["method", "engine", "strategy", "execution", "output_driver"]

EXECUTION_MODE_LABELS: dict[ExecutionMode, str] = {
    "inmem": "In memory",
    "dask": "Dask",
    "multiprocessing": "Multiprocessing",
}
IMPLEMENTATION_LABELS = {"gdal": "GDAL CLI", "pdal": "PDAL CLI", "flox": "Flox"}


##############################
# Cases, benchmarks and reports
##############################


@dataclass(frozen=True)
class Case:
    """A small class to define one "case" (combination of parameters/inputs) of the operation being benchmarked."""

    method: str | None = None
    engine: str | None = None
    execution: ExecutionMode | None = "inmem"
    strategy: str | None = None
    output_driver: str = "GPKG"
    options: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)
    implementation: str = "geoutils"
    variant: str | None = None


OperationPrepare = Callable[[Any, Case], None]
OperationExecute = Callable[[Any, Case], float]
OptionBuilder = Callable[[Case, RuntimeConfig], Mapping[str, Any]]


@dataclass(frozen=True)
class Operation:
    """A small class to consistently apply the preparation and execution functions of an operation."""

    name: str
    prepare: OperationPrepare
    execute: OperationExecute
    option_builder: OptionBuilder
    label_options: tuple[str, ...] = ()
    call_name: str | None = None
    label: str | None = None
    benchmark_name: str | None = None
    order: int = 100
    large_data_cases: tuple[Case, ...] = ()
    large_data_dependencies: tuple[str, ...] = ()
    expected_value: float = 1

    @property
    def public_call_name(self) -> str:
        """Return the public call name shown beside benchmark results."""

        return f".{self.name}" if self.call_name is None else self.call_name

    @property
    def display_name(self) -> str:
        """Return the operation name used in report titles."""

        return self.label or self.name.replace("_", " ").title()


ConfigurationBuilder = Callable[[int | float | None, Case], Mapping[str, Any]]
WorkloadBuilder = Callable[[int | float, tuple[RuntimeConfig, ...]], str]


def _class_token(value: str) -> str:
    """Convert one benchmark field to a lowercase ASV ID."""

    return value.replace("-", "").replace("_", "").lower()


@dataclass(frozen=True)
class Benchmark:
    """
    A class to describe all tested cases for benchmarking one operation.
    """

    operation: Operation
    cases: tuple[Case, ...]
    configure: ConfigurationBuilder
    parameter_name: str | None = None
    values: tuple[int | float, ...] = ()
    name: str | None = None
    parameter_label: str | None = None
    parameter_title: str | None = None
    describe_workload: WorkloadBuilder | None = None

    def __post_init__(self) -> None:
        """Check that the benchmark has cases and complete parameter metadata."""

        if not self.cases:
            raise ValueError("A benchmark must contain at least one case")
        if self.parameter_name is None and self.values:
            raise ValueError("A fixed benchmark cannot define parameter values")
        if self.parameter_name is not None and not self.values:
            raise ValueError("A parameterized benchmark needs values")

    @property
    def id(self) -> str:
        """Return the operation and varied input used to identify this benchmark."""

        operation = self.operation.name.replace("_", "-")
        subject = operation if self.name is None else self.name
        if self.parameter_name is None:
            return subject
        parameter = self.parameter_name.replace("_", "-")
        operation_prefix = f"{operation}-"
        if parameter.startswith(operation_prefix):
            parameter = parameter.removeprefix(operation_prefix)
        return f"{subject}-{parameter}"

    def benchmark_class(self, case: Case) -> str:
        """Return the stable public ASV class name for one case."""

        output_driver = None if case.output_driver == "GPKG" else case.output_driver
        implementation: str | None = {"gdal": "gdal_cli", "pdal": "pdal_cli"}.get(
            case.implementation, case.implementation
        )
        implementation = case.engine if case.implementation == "geoutils" else implementation
        function = self.operation.benchmark_name or self.operation.name
        values = (function, case.method, implementation, case.strategy, output_driver, case.execution, case.variant)
        implementation_name = "_".join(_class_token(value) for value in values if value is not None)
        if self.parameter_name is None:
            return implementation_name
        return f"{implementation_name}__{_class_token(self.parameter_name)}"

    def make_config(self, parameter: int | float | None, case: Case) -> RuntimeConfig:
        """Build runtime and workload settings for one case."""

        values = {**case.options, **self.configure(parameter, case)}
        return runtime_config(values)

    def workload(self, parameter: int | float, cases: tuple[Case, ...] | None = None) -> str:
        """Describe the fixture values shared by selected report series."""

        selected = self.cases if cases is None else cases
        configs = tuple(self.make_config(parameter, case) for case in selected)
        if self.describe_workload is not None:
            return self.describe_workload(parameter, configs)

        # Raster dimensions and chunks are shared by most operation benchmarks
        shape = _common_config(configs, "shape")
        chunks = _common_config(configs, "chunks")
        parts = []
        if shape is not None:
            parts.append(f"{shape[0]:,} × {shape[1]:,} raster")
        if chunks is not None:
            parts.append(f"{chunks[0]:,} × {chunks[1]:,} chunks")
        return "; ".join(parts)


def _common_config(configs: tuple[RuntimeConfig, ...], name: str) -> Any | None:
    """Return one runtime or workload value shared by every configuration."""

    values = [getattr(config, name) if hasattr(config, name) else config.value(name) for config in configs]
    return values[0] if all(value == values[0] for value in values[1:]) else None


@dataclass(frozen=True)
class Comparison:
    """
    Choose which benchmark results appear as separate lines in one report plot.

    ``cases`` selects the results, and ``by`` names the :class:`Case` field used to label each line.
    """

    benchmark: Benchmark
    cases: tuple[Case, ...]
    by: ComparisonDimension
    slug: str
    logarithmic_x: bool = False
    documentation: bool = True
    summary: bool = True

    @property
    def operation(self) -> str:
        """Return the operation measured by the benchmark."""

        return self.benchmark.operation.name

    @property
    def method(self) -> str | None:
        """Return the method fixed across selected cases, if any."""

        return self._fixed("method")

    @property
    def calculation_engine(self) -> str | None:
        """Return the engine fixed across selected cases, if any."""

        return self._fixed("engine")

    @property
    def strategy(self) -> str | None:
        """Return the strategy fixed across selected cases, if any."""

        return self._fixed("strategy")

    @property
    def execution_mode(self) -> ExecutionMode | None:
        """Return the execution mode fixed across selected cases, if any."""

        return self._fixed("execution")

    def _fixed(self, attribute: str) -> Any | None:
        """Return one non-varied case value when all GeoUtils cases share it."""

        if self.by == attribute:
            return None
        values = {getattr(case, attribute) for case in self.cases if case.implementation == "geoutils"}
        values.discard(None)
        return next(iter(values)) if len(values) == 1 else None

    def choice_label(self, attribute: str, value: Any) -> str:
        """Return operation-local display text for one case choice."""

        for case in self.cases:
            if case.implementation == "geoutils" and getattr(case, attribute) == value:
                return case.labels.get(attribute, _display_value(value))
        return _display_value(value)

    @property
    def parameter_label(self) -> str:
        """Return the plot label for the varied numeric input."""

        if self.benchmark.parameter_label is None:
            raise ValueError(f"Benchmark {self.benchmark.id!r} has no parameter label")
        return self.benchmark.parameter_label

    @property
    def title(self) -> str:
        """Return the plot title from explicit benchmark and comparison metadata."""

        subject = self.benchmark.operation.display_name
        if self.method is not None:
            subject = f"{self.choice_label('method', self.method)} {subject.lower()}"
        choice = {
            "execution": "",
            "engine": " engines",
            "method": " methods",
            "strategy": " strategies",
            "output_driver": " formats",
        }[self.by]
        return f"{subject}{choice} by {self.benchmark.parameter_title}"

    @property
    def series(self) -> tuple[tuple[str, str], ...]:
        """Return plot labels and ASV classes for selected cases."""

        cases = self.cases
        if self.by == "output_driver":
            # Keep each GeoUtils format beside implementation cases writing the same format
            drivers = tuple(dict.fromkeys(case.output_driver for case in cases))
            cases = tuple(case for driver in drivers for case in cases if case.output_driver == driver)

        series = []
        for case in cases:
            label = IMPLEMENTATION_LABELS.get(case.implementation)
            if self.by == "output_driver" and case.implementation == "geoutils":
                label = case.labels.get("output_driver", f"GeoUtils {case.output_driver}")
            elif label is None:
                label = case.labels.get(self.by, _display_value(getattr(case, self.by)))
            elif self.by == "output_driver":
                label = f"{label.removesuffix(' CLI')} {case.output_driver}"
            elif case.execution is not None:
                label = f"{label} ({EXECUTION_MODE_LABELS[case.execution]})"
            series.append((label, self.benchmark.benchmark_class(case)))
        return tuple(series)

    def workload(self, parameter: int | float) -> str:
        """Describe the fixture values shared by every series at one parameter value."""

        return self.benchmark.workload(parameter, self.cases)


def _display_value(value: Any) -> str:
    """Format one case choice for a report legend."""

    if value in EXECUTION_MODE_LABELS:
        return EXECUTION_MODE_LABELS[value]
    return str(value).replace("_", " ").title()


###########################################
# Convenience functions for multiple cases
###########################################


def comparison(
    benchmark: Benchmark,
    *,
    by: ComparisonDimension,
    cases: tuple[Case, ...] | None = None,
    slug: str | None = None,
    logarithmic_x: bool = False,
    documentation: bool = True,
    summary: bool = True,
) -> Comparison:
    """Build concise reporting metadata from one benchmark and explicit comparison dimension."""

    return Comparison(
        benchmark=benchmark,
        cases=benchmark.cases if cases is None else cases,
        by=by,
        slug=benchmark.id if slug is None else slug,
        logarithmic_x=logarithmic_x,
        documentation=documentation,
        summary=summary,
    )


def execution_cases(
    method: str | None,
    engine: str | None,
    *,
    strategy: str | None = None,
    executions: tuple[ExecutionMode, ...] = ("inmem", "dask", "multiprocessing"),
    output_driver: str = "GPKG",
    options: Mapping[str, Any] | None = None,
    labels: Mapping[str, str] | None = None,
    variant: str | None = None,
) -> tuple[Case, ...]:
    """Generate an execution comparison around fixed numerical choices."""

    return tuple(
        Case(
            method=method,
            engine=engine,
            execution=execution,
            strategy=strategy if execution != "inmem" else None,
            output_driver=output_driver,
            options=options or {},
            labels=labels or {},
            variant=variant,
        )
        for execution in executions
    )


def strategy_cases(
    method: str | None,
    engine: str | None,
    strategies: tuple[str, ...],
    *,
    execution: Literal["dask", "multiprocessing"],
    options: Mapping[str, Any] | None = None,
    strategy_labels: Mapping[str, str] | None = None,
    variant: str | None = None,
) -> tuple[Case, ...]:
    """Generate cases for approaches that coordinate one chunked operation."""

    return tuple(
        Case(
            method=method,
            engine=engine,
            execution=execution,
            strategy=strategy,
            options=options or {},
            labels={"strategy": strategy_labels[strategy]} if strategy_labels is not None else {},
            variant=variant,
        )
        for strategy in strategies
    )


def reference_case(
    benchmark: Case | tuple[Case, ...],
    *,
    implementation: str,
    execution: ExecutionMode | None = None,
) -> Case:
    """Define one reference implementation equivalent to a GeoUtils case."""

    case = benchmark[0] if isinstance(benchmark, tuple) else benchmark
    if isinstance(benchmark, tuple):
        choices = {(item.method, item.output_driver, repr(item.options)) for item in benchmark}
        if len(choices) != 1:
            raise ValueError(f"Expected equivalent benchmark cases for one reference implementation, got: {choices}")
    return Case(
        method=case.method,
        execution=execution,
        output_driver=case.output_driver,
        options=case.options,
        labels=case.labels,
        implementation=implementation,
    )


def merge_cases(*groups: tuple[Case, ...]) -> tuple[Case, ...]:
    """Deduplicate cases reused by several plots."""

    cases: dict[str, Case] = {}
    for case in (case for group in groups for case in group):
        key = repr(replace(case, labels={}))
        existing = cases.get(key)
        if existing is not None:
            case = replace(existing, labels={**existing.labels, **case.labels})
        cases[key] = case
    return tuple(cases.values())


def parameter_config(
    parameter_name: str,
    values: tuple[int | float, ...],
    operation: Operation,
    cases: tuple[Case, ...],
    configure: ConfigurationBuilder,
    **kwargs: Any,
) -> Benchmark:
    """Build one parameterized benchmark without a separate axis object."""

    return Benchmark(
        operation=operation,
        cases=cases,
        configure=configure,
        parameter_name=parameter_name,
        values=values,
        **kwargs,
    )

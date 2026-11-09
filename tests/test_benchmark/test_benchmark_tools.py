"""Minimal tests for benchmark workflows, reports and external CLI commands."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from geoutils._misc import import_optional

from benchmarks.asv_suite import benchmarks as generated_benchmarks
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_MEMORY_SCALING_PLOT,
    DOCUMENTATION_PDAL_PLOT,
    DOCUMENTATION_TIME_PLOT,
    PARAMETER_REPORT_PAGE,
    PERFORMANCE_CHANGE_REPORT,
    REFERENCE_REPORT_PAGE,
    SCALING_REPORT_PAGE,
    _headline_summary_table,
    _PreviewResult,
    _render_preview,
    _scaling_comparisons,
    collect_comparison_measurements,
    render_documentation_snapshot,
)
from benchmarks.comparisons.flox import (
    compute_geoutils_grouped_stats,
)
from benchmarks.comparisons.flox import grouped_stats as flox_grouped_stats
from benchmarks.comparisons.flox import (
    prepare_grouped_inputs,
)
from benchmarks.comparisons.gdal import (
    COMPARISON_OPERATIONS,
    _warp_memory_limit_mb,
    build_gdal_command,
)
from benchmarks.comparisons.pdal import (
    PDAL_COMPARISON_OPERATIONS,
    PdalComparisonOperation,
    build_pdal_command,
)
from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import Case
from benchmarks.workflows.io import materialize_dem_output
from benchmarks.workflows.operations import (
    BENCHMARK_BY_CLASS,
    BENCHMARK_BY_ID,
    BENCHMARK_CASE_BY_CLASS,
    BENCHMARK_CASES,
    BENCHMARKS,
    COMPARISONS,
    OPERATION_BY_NAME,
    OPERATION_MODULES,
    collect_operation_modules,
    discover_operation_modules,
    format_api_label,
)
from benchmarks.workflows.runner import BenchmarkRunner


class TestComparisonReport:
    """
    Test module for benchmark registration and report generation.

    Those are minimal tests to ensure changes to ``benchmarks/`` don't break the routines.
    Even if ASV runs quick checks of the benchmarking setup through CI, it's easier to have a detailed traceback here
    through Pytest for some aspects.
    We especially tests our custom routines/rendering for the benchmark webpage, comparisons to external refs (e.g.
    GDAL CLI), and across variables (dem/point size, chunk size, etc) and categories (e.g. in-memory/Dask/MP, method,
    etc) of interest.
    """

    def test_benchmark_registry__generated_classes_exist(self) -> None:
        """Checks that every registered benchmark and plotted series has a generated ASV class."""

        # Gather the GeoUtils/external classes registered for ASV and the classes referenced by report plots
        registered = set(BENCHMARK_CASE_BY_CLASS)
        plotted = {class_name for comparison in COMPARISONS for _, class_name in comparison.series}

        # Importing benchmarks.py should create every class needed by ASV and the report
        assert BENCHMARK_CASES and COMPARISONS
        assert plotted <= registered
        assert all(hasattr(generated_benchmarks, class_name) for class_name in registered)

    def test_benchmark_registry__generated_class_names_stable(self) -> None:
        """Checks that generated ASV identifiers keep function fields separate from their varying input."""

        # Hash the sorted names to keep the identifier check compact and order independent
        class_names = sorted(
            benchmark.benchmark_class(case)
            for benchmark in BENCHMARKS
            if benchmark.parameter_name is not None
            if benchmark.id != "reproject-operator-dem-size"
            for case in benchmark.cases
            if "read_strategy" not in case.options
        )
        digest = hashlib.sha256("\n".join(class_names).encode()).hexdigest()

        # A changed name would split ASV history even when the underlying operation remained the same
        assert digest == "7bf237dba4a181ffa896576926f31dd969315c270849b41511636fbf327fef9c"

    def test_benchmark_registry__flox_execution_modes(self) -> None:
        """Checks that Flox benchmarks use supported execution modes."""

        # Collect Flox benchmark cases
        cases = [
            case
            for benchmark in BENCHMARKS
            for case in benchmark.cases
            if case.implementation == "flox"
            or (case.engine is not None and case.engine.startswith("flox"))
            or case.options.get("reduction_backend") == "flox"
        ]

        # Check supported execution modes
        assert cases
        assert {case.execution for case in cases} == {"inmem", "dask"}

    def test_operation_discovery__deterministic_modules(self) -> None:
        """Checks that operation discovery returns the same modules in their stable report order."""

        # Repeat package discovery rather than reusing the modules collected during the first import
        discovered = discover_operation_modules()

        # Every run should return the same local modules in the order used for sweeps and comparison plots
        assert tuple(module.__name__ for module in discovered) == tuple(module.__name__ for module in OPERATION_MODULES)
        assert [getattr(module, "ORDER", 100) for module in discovered] == sorted(
            getattr(module, "ORDER", 100) for module in discovered
        )

    def test_operation_discovery__error_duplicate_benchmark_id(self) -> None:
        """Checks that two operation modules cannot register the same benchmark identifier."""

        # Present one real sweep through two small module-like objects to isolate duplicate validation
        duplicate_modules = (SimpleNamespace(BENCHMARKS=(BENCHMARKS[0],)),) * 2

        # Reject the collision before generated classes or report mappings can silently replace each other
        with pytest.raises(ValueError, match=f"Duplicate benchmark ID: {BENCHMARKS[0].id}"):
            collect_operation_modules(duplicate_modules)

    def test_scaling__default_options(self) -> None:
        """Checks that scaling compares execution modes with the default method, backend and chunk strategy."""

        # Select scaling comparisons from the registry, which also includes alternative options and external tools
        comparisons = _scaling_comparisons()
        by_operation = {comparison.operation: comparison for comparison in comparisons}

        # Each operation appears once, using only GeoUtils execution modes as separate lines
        assert len(by_operation) == len(comparisons)
        assert all(comparison.by == "execution" for comparison in comparisons)
        assert all(case.implementation == "geoutils" for comparison in comparisons for case in comparison.cases)

        assert by_operation["grid"].method == "nearest"
        assert by_operation["random_field"].calculation_engine == "gpytorch"

        # Grouped statistics must use the automatic strategy for both chunked modes
        grouped = by_operation["grouped_stats"]
        assert {case.execution for case in grouped.cases} == {"inmem", "dask", "multiprocessing"}
        assert all(case.strategy == "auto" for case in grouped.cases if case.execution != "inmem")

    def test_render_preview__essential_files(self, tmp_path: Path) -> None:
        """Checks that preview rendering writes the main pages, data exports and plots."""

        pytest.importorskip("matplotlib")

        # Render the complete preview from the small fake ASV result bundled with the renderer
        print("tmp_path", ":", tmp_path)
        _render_preview(tmp_path)
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY
        print("report_directory", ":", report_directory)

        # Keep this list to files used directly by the published benchmark site or downloaded result data
        expected_files = (
            tmp_path / "index.html",
            report_directory / "index.html",
            report_directory / SCALING_REPORT_PAGE,
            report_directory / REFERENCE_REPORT_PAGE,
            report_directory / PARAMETER_REPORT_PAGE,
            report_directory / "comparisons.csv",
            report_directory / "comparisons.json",
            report_directory / PERFORMANCE_CHANGE_REPORT,
        )

        assert all(path.is_file() for path in expected_files)
        assert any(report_directory.glob("*.svg"))

        # The JSON export must contain measurements rather than only an empty report shell
        payload = json.loads((report_directory / "comparisons.json").read_text(encoding="utf-8"))
        assert payload["measurements"]

        # Check the simpler overview and the order of its detailed pages
        overview = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "concept-map" not in overview
        assert "<strong>Output format:</strong>" not in overview
        assert "Peak additional RAM (MB)" in overview
        assert overview.index(REFERENCE_REPORT_PAGE) < overview.index(SCALING_REPORT_PAGE)
        assert overview.index(SCALING_REPORT_PAGE) < overview.index("comparisons/index.html")

        # Scaling has a TOC and collapsible plots; other parameters have a separate page
        scaling = (report_directory / SCALING_REPORT_PAGE).read_text(encoding="utf-8")
        assert "default-reproject-dem-size.svg" in scaling
        assert 'src="filter-chunk-size.svg"' not in scaling
        assert '<a href="#reproject-dem-size">' in scaling
        assert '<details class="plot-card" id="reproject-dem-size">' in scaling

        # The simplified scaling plot includes all three execution modes and excludes the external GDAL reference
        plot = (report_directory / "default-reproject-dem-size.svg").read_text(encoding="utf-8")
        assert all(label in plot for label in ("In memory", "Dask", "Multiprocessing"))
        assert "GDAL CLI" not in plot

        # Option tables link to the detailed plots, since scaling now shows only default options
        options = (report_directory / "index.html").read_text(encoding="utf-8")
        assert f'href="{PARAMETER_REPORT_PAGE}#gridding-method-dem-size"' in options
        assert 'href="scaling.html#gridding-method-dem-size"' not in options

    def test_summary__exact_size_and_dask(self) -> None:
        """Checks that the summary uses the requested dem size and Dask execution only."""

        # Use a small-run result to check that smaller dems do not fill the standard-size table
        records = collect_comparison_measurements(_PreviewResult())
        records = [record for record in records if record.parameter != 10_000]
        empty_summary = _headline_summary_table(records)
        assert set(re.findall(r"<td>(.*?)</td>", empty_summary)) == {"—"}
        print("empty_summary", empty_summary)

        # Add distinct Dask and in-memory results at 10,000 × 10,000 so a wrong choice is easy to see
        source = next(
            record for record in records if record.comparison == "reproject-dem-size" and record.series_label == "Dask"
        )
        dask = replace(source, parameter=10_000, end_to_end_time_s=7.25, process_tree_mem_increase_mb=482.5)
        eager = replace(dask, series_label="In memory", execution_mode="inmem", end_to_end_time_s=999)
        summary = _headline_summary_table([*records, dask, eager])
        print("")
        print("summary", summary)
        # Check the displayed measurements and exclude the in-memory timing
        assert "<td>482</td><td>7.25</td>" in summary
        assert "<td>999</td>" not in summary

    @pytest.mark.parametrize(
        "default_comparison, default_series, other_comparison",
        [
            ("random-field-gpytorch", "Dask", "random-field-gstools"),
            ("grouped-stats-dem-size", "Automatic", "grouped-stats-execution-size"),
        ],
    )
    def test_summary__default_choices(
        self, default_comparison: str, default_series: str, other_comparison: str
    ) -> None:
        """Checks that the summary uses the default numerical backend and chunk strategy."""

        # Give the default and alternative choices distinct values so the selection is clear
        records = collect_comparison_measurements(_PreviewResult())
        default = next(
            record
            for record in records
            if record.comparison == default_comparison and record.series_label == default_series
        )
        other = next(
            record for record in records if record.comparison == other_comparison and record.series_label == "Dask"
        )
        default = replace(default, parameter=10_000, end_to_end_time_s=11, process_tree_mem_increase_mb=101)
        other = replace(other, parameter=10_000, end_to_end_time_s=777, process_tree_mem_increase_mb=888)

        # Render both choices and check that only the default appears in the table
        summary = _headline_summary_table([default, other])
        assert "<td>101</td><td>11</td>" in summary
        assert "<td>777</td>" not in summary

    def test_render_documentation_snapshot__essential_files(self, tmp_path: Path) -> None:
        """Checks that documentation rendering writes its summary plots and numeric data."""

        pytest.importorskip("matplotlib")

        # Render the documentation files from the same small result used by preview mode
        records = render_documentation_snapshot(_PreviewResult(), tmp_path)

        # All graphics and their JSON source are needed when benchmark results are updated in the docs
        assert records
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_SCALING_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_PDAL_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()


class TestBenchmarkScenarios:
    """Test module for scheduled inputs and benchmark configurations."""

    def test_sweeps__representative_configs(self) -> None:
        """Checks representative updates for strategy, grouped and point sweeps."""

        # Select cases whose configuration depends on their method, layout or point count
        gridding = BENCHMARK_BY_ID["grid-dem-size"]
        idw_case = next(case for case in gridding.cases if case.method == "idw" and case.execution == "inmem")
        interleaved = BENCHMARK_BY_ID["grouped-stats-interleaved-chunk-size"]
        grouped_case = interleaved.cases[0]
        pointcloud = BENCHMARK_BY_ID["to-pointcloud-dem-size"]
        point_case = pointcloud.cases[0]

        # These values cover method-specific support, uneven chunks, repeated groups and point inputs
        grid_config = gridding.make_config(8_000, idw_case)
        grouped_config = interleaved.make_config(500, grouped_case)
        point_config = pointcloud.make_config(1_000, point_case)
        assert (grid_config.shape, grid_config.chunks, grid_config.value("grid_dist_nodata_pixel")) == (
            (8_000, 8_000),
            (1_000, 1_000),
            16,
        )
        assert (grouped_config.shape, grouped_config.chunks, grouped_config.value("grouped_layout")) == (
            (2_000, 2_000),
            (500, 500),
            "interleaved",
        )
        assert (point_config.shape, point_config.chunks) == ((1_000, 1_000), (1_000, 1_000))

    def test_comparison__derived_identity_and_workload(self) -> None:
        """Checks that sweep cases provide report metadata and workload sizes without duplicate text."""

        # Select a comparison with dem, chunk and vector fixture dimensions
        comparison = next(item for item in COMPARISONS if item.slug == "clip-dem-size")

        # The operation and parameter axis name the sweep, while its config supplies every displayed workload value
        assert comparison.benchmark.id == "clip-dem-size"
        assert comparison.operation == "clip"
        assert comparison.by == "execution"
        assert comparison.workload(8_000) == ("8,000 × 8,000 dem; 1,000 × 1,000 chunks; 51 × 51 vector features")

    def test_comparison__output_formats_group_matching_implementations(self) -> None:
        """Checks that each GeoUtils point format stays beside the matching PDAL format in report series."""

        # Select the output-format comparison whose benchmark stores GeoUtils cases before implementation cases
        comparison = next(item for item in COMPARISONS if item.slug == "subsample-las-laz-size")

        # Keep the established report order while treating every reference implementation as a normal case
        assert tuple(label for label, _ in comparison.series) == (
            "GeoUtils LAS",
            "PDAL LAS",
            "GeoUtils LAZ",
            "PDAL LAZ",
        )

    @pytest.mark.parametrize(
        ("class_name", "expected_label"),
        (
            ("write_dask", ".to_file()"),
            ("reproject_nearest_rasterio_inmem__demsize", ".reproject(resampling='nearest')"),
            ("polygonize_rasterio_labelstitch_dask__demsize", ".polygonize(strategy='label_stitch')"),
            (
                "stats_moments_numpy_dense_dask__demsize",
                "stats(statistics=['mean', 'std', 'min', 'max'], method='map-reduce', reindex=True)",
            ),
        ),
    )
    def test_api_label__matches_execution_options(self, class_name: str, expected_label: str) -> None:
        """Checks that generated labels use the public argument names passed by operation handlers."""

        # Resolve the structured case instead of deriving options from its generated class name
        case = BENCHMARK_CASE_BY_CLASS[class_name]
        benchmark = BENCHMARK_BY_CLASS[class_name]

        # Execution mode stays separate from the public call while method, engine and strategy remain visible
        assert format_api_label(benchmark.operation, case) == expected_label
        assert getattr(generated_benchmarks, class_name).pretty_name == expected_label

    def test_comparison_references__use_normal_cases(self) -> None:
        """Checks that Flox, GDAL and PDAL references are normal benchmark cases."""

        # Flox measures prepared arrays directly; GDAL and PDAL execute through their command runners
        flox_benchmark = BENCHMARK_BY_ID["grouped-flox-dem-size"]
        gdal_benchmark = BENCHMARK_BY_ID["rasterize-dem-size"]
        pdal_benchmark = BENCHMARK_BY_ID["to-pointcloud-dem-size"]

        # Flox is a public stats() backend, while GDAL and PDAL remain external command references
        assert {case.implementation for case in flox_benchmark.cases} == {"geoutils", "flox"}
        assert {case.implementation for case in gdal_benchmark.cases} == {"geoutils", "gdal"}
        assert {case.implementation for case in pdal_benchmark.cases} == {"geoutils", "pdal"}


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestBenchmarkRunner:
    """Test module for bounded operation outputs produced by BenchmarkRunner."""

    @pytest.mark.parametrize("execution_mode", ["inmem", "dask", "multiprocessing"])
    def test_clip__masks_outside_fixture_polygons(
        self,
        execution_mode: Literal["inmem", "dask", "multiprocessing"],
        tmp_path: Path,
    ) -> None:
        """Checks that the clipping benchmark keeps polygon interiors and masks the surrounding dem."""

        if execution_mode == "dask":
            pytest.importorskip("distributed")

        # Use several polygons across multiple chunks so worker modes process inside and outside cells
        config = RuntimeConfig(
            shape=(64, 64),
            chunks=(32, 32),
            directory=str(tmp_path / execution_mode),
            workload={"vector_features_per_axis": 3, "operation": "clip"},
        )

        # Run the complete benchmark workflow and inspect one kept center cell and one clipped corner
        operation = OPERATION_BY_NAME["clip"]
        case = Case(execution=execution_mode, options={"operation": "clip"})
        with BenchmarkRunner(operation, case, config) as runner:
            result = runner.run(profile=False)
        assert result.output_file is not None
        with rio.open(result.output_file) as dataset:
            values = dataset.read(1)
            nodata = dataset.nodata

        # The odd polygon grid covers the dem center, while every fixture polygon stays away from the corners
        assert result.value == config.dem_value
        assert values[values.shape[0] // 2, values.shape[1] // 2] == config.dem_value
        assert not np.isfinite(values[0, 0]) or values[0, 0] == nodata


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestBenchmarkOutputsChunked:
    """Test module for complete dem and point files written by Dask benchmark workflows."""

    @pytest.mark.parametrize("dtype", ["float32", "bool"])
    def test_to_file__dem_chunks(self, dtype: str, tmp_path: Path) -> None:
        """Checks that Dask benchmark files match in-memory dem writes without loading the original arrays."""

        pytest.importorskip("distributed")
        da = pytest.importorskip("dask.array")

        # Uneven edge chunks reveal missing windows; two workers share the destination file
        config = RuntimeConfig(shape=(35, 49), chunks=(16, 32), n_workers=2, directory=str(tmp_path))
        values = np.arange(np.prod(config.shape)).reshape(config.shape).astype(dtype)
        nodata = None if dtype == "bool" else -99999
        if dtype == "bool":
            values = np.indices(config.shape).sum(axis=0) % 2 == 0
        else:
            values[0, 0] = nodata

        # Binary fractions give both writers exactly representable pixel sizes
        transform = rio.transform.from_origin(7, 46, 0.25, 0.25)
        chunks = da.from_array(values, chunks=config.chunks)
        lazy = gu.DataArrayRasterAccessor.from_array(chunks, transform, 4326, nodata=nodata)
        eager_values = values.astype("uint8") if dtype == "bool" else np.ma.masked_equal(values, nodata)
        eager = gu.Raster.from_array(eager_values, transform, 4326, nodata=nodata)

        # Write both results through the shared benchmark path; GeoTIFF represents boolean masks as bytes
        assert not lazy.rst.is_loaded and eager.is_loaded
        operation = OPERATION_BY_NAME["write"]
        with BenchmarkRunner(operation, Case(execution="dask", options={"operation": "write"}), config) as runner:
            runner._compute_dem(lazy)
            actual_file = runner._last_output_file
            expected_file = materialize_dem_output(eager, "inmem", str(tmp_path / "expected.tif"), config)

        # Compare every saved pixel and its grid, including the shorter final row and column chunks
        assert actual_file is not None
        with rio.open(actual_file) as actual, rio.open(expected_file) as expected:
            np.testing.assert_array_equal(actual.read(), expected.read())
            assert actual.transform == expected.transform
            assert actual.crs == expected.crs
            assert actual.nodata == expected.nodata
            assert actual.block_shapes == expected.block_shapes
            assert actual.compression is None
        assert not lazy.rst.is_loaded
        assert lazy.chunks == chunks.chunks

    @pytest.mark.parametrize("operation_name", ["subsample", "to_pointcloud"])
    @pytest.mark.parametrize("driver", ["GPKG", "LAS", "LAZ"])
    def test_to_file__point_outputs(self, operation_name: str, driver: str, tmp_path: Path) -> None:
        """Checks that Dask point benchmarks save every expected point and match in-memory file output."""

        pytest.importorskip("distributed")
        if driver in ("LAS", "LAZ"):
            laspy = import_optional("laspy")

        # A shorter final dem chunk exposes dropped rows in complete conversion and fixed-size sampling
        config = RuntimeConfig(shape=(35, 49), chunks=(16, 32), directory=str(tmp_path))
        case = Case(
            execution="dask",
            output_driver=driver,
            variant="xarray" if driver in ("LAS", "LAZ") else None,
            options={"operation": operation_name, "subsample_size": 37},
        )
        operation = OPERATION_BY_NAME[operation_name]
        with BenchmarkRunner(operation, case, config) as runner:
            source = runner.make_dem()
            assert not source.rst.is_loaded
            eager_source = source.compute()
            options = dict(operation.option_builder(case, runner.config))
            options["pointcloud_return_type"] = "dataarray" if case.variant == "xarray" else "geodataframe"
            eager_points = getattr(eager_source.rst, operation_name)(**options)

            # Run the benchmark and save the same public call on in-memory inputs for comparison
            assert eager_points.pc.is_loaded
            result = runner.run(profile=False)
            expected_file = str(tmp_path / f"expected.{driver.lower()}")
            eager_points.pc.to_file(expected_file)
            assert not source.rst.is_loaded

        # Compare complete coordinates and values, including LAS integer encoding and LAZ compression
        assert result.output_file is not None
        assert Path(result.output_file).suffix == f".{driver.lower()}"
        assert result.value == config.dem_value
        expected_count = 37 if operation_name == "subsample" else int(np.prod(config.shape))
        if driver == "GPKG":
            actual = gpd.read_file(result.output_file)
            expected = gpd.read_file(expected_file)
            assert len(actual) == expected_count
            assert actual.crs == expected.crs

            # GeoDataFrame partitions write in chunk order, so compare the same points sorted by Y/X
            actual_points = np.column_stack((actual.geometry.x, actual.geometry.y, actual["b1"]))
            expected_points = np.column_stack((expected.geometry.x, expected.geometry.y, expected["b1"]))
            actual_order = np.lexsort((actual_points[:, 0], actual_points[:, 1]))
            expected_order = np.lexsort((expected_points[:, 0], expected_points[:, 1]))
            np.testing.assert_array_equal(actual_points[actual_order], expected_points[expected_order])
        else:
            actual_las = laspy.read(result.output_file)
            expected_las = laspy.read(expected_file)
            assert actual_las.header.point_count == expected_count
            for dimension in ("X", "Y", "Z"):
                np.testing.assert_array_equal(actual_las[dimension], expected_las[dimension])
            np.testing.assert_array_equal(actual_las.header.scales, expected_las.header.scales)
            np.testing.assert_array_equal(actual_las.header.offsets, expected_las.header.offsets)


def test_grouped_reference__matches_geoutils() -> None:
    """Checks that direct Flox and public GeoUtils grouped statistics return the same in-memory table."""

    pytest.importorskip("flox")

    # Prepare one small shared input and run each implementation directly
    inputs = prepare_grouped_inputs(16, 2, "interleaved", "inmem")
    expected = compute_geoutils_grouped_stats(*inputs)
    result = flox_grouped_stats(*inputs, use_dask=False)

    # Matching labelled tables establish that the external reference measures the same calculation
    assert result.equals(expected)


class TestGdalCommands:
    """Test module for building GDAL commands used by external benchmark comparisons."""

    @pytest.mark.parametrize(
        "method",
        [
            "nearest",
            "bilinear",
            "cubic",
            "cubic_spline",
            "lanczos",
            "average",
            "sum",
            "min",
            "max",
            "rms",
            "mode",
            "med",
            "q1",
            "q3",
        ],
    )
    def test_reproject__operator_reference_uses_matching_gdal_method(
        self, method: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that every operator benchmark selects the corresponding GDAL warp method."""

        # Resolve the registered operator and its GDAL reference on the same benchmark grid
        monkeypatch.setattr("benchmarks.comparisons.gdal._require_command", lambda name: name)
        benchmark = BENCHMARK_BY_ID["reproject-operator-dem-size"]
        operator_case = next(case for case in benchmark.cases if case.method == method and case.engine == "operator")
        reference_case = next(
            case for case in benchmark.cases if case.method == method and case.implementation == "gdal"
        )
        config = benchmark.make_config(128, operator_case)
        config.directory = str(tmp_path)

        # The source size, target size, and GDAL -r choice must describe one comparable calculation
        options = benchmark.operation.option_builder(operator_case, config)

        command = build_gdal_command(
            "reproject",
            reference_case,
            config,
            dem_file="source-raster.tif",
            vector_file="source-vector.gpkg",
            point_file="source-points.gpkg",
        ).command
        assert type(options["resampling"]).__name__ in {
            "Nearest",
            "Linear",
            "RasterConvolution",
            "Mean",
            "Sum",
            "Minimum",
            "Maximum",
            "RootMeanSquare",
            "Mode",
            "Median",
            "Quantile",
        }
        expected_method = {"nearest": "near", "cubic_spline": "cubicspline"}.get(method, method)
        assert command[command.index("-r") + 1] == expected_method
        assert options["grid_size"] == (128, 128)
        assert format_api_label(benchmark.operation, operator_case).endswith(
            f"resampling={operator_case.labels['api_resampling']})"
        )

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command__essential_arguments(
        self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that every GDAL comparison names its executable, input, output and cache size."""

        # Return command names directly because this smoke test builds arguments without running GDAL
        monkeypatch.setattr("benchmarks.comparisons.gdal._require_command", lambda name: name)
        config = RuntimeConfig(shape=(64, 96), chunks=(32, 32), directory=str(tmp_path))

        # Build the command with one recognizable source path for each supported input type
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            Case(method="nearest" if operation in ("reproject", "grid") else None),
            config,
            dem_file="source-raster.tif",
            vector_file="source-vector.gpkg",
            point_file="source-points.gpkg",
        )
        command = comparison.command

        # These fields are enough to catch a command wired to the wrong tool, source, output or cache setting
        expected_executable = {
            "clip": "gdalwarp",
            "reproject": "gdalwarp",
            "polygonize": "gdal_polygonize.py",
            "rasterize": "gdal_rasterize",
            "grid": "gdal_grid",
        }[operation]
        expected_source = {
            "clip": "source-raster.tif",
            "reproject": "source-raster.tif",
            "polygonize": "source-raster.tif",
            "rasterize": "source-vector.gpkg",
            "grid": "source-points.gpkg",
        }[operation]
        cache_index = command.index("GDAL_CACHEMAX")
        assert command[0] == expected_executable
        assert expected_source in command
        if operation == "clip":
            assert "source-vector.gpkg" in command
        if operation in ("clip", "reproject"):
            warp_memory_index = command.index("-wm")
            assert command[warp_memory_index + 1] == str(_warp_memory_limit_mb(config))
        assert comparison.output_file in command
        assert command[cache_index + 1] == str(config.gdal_cachemax_mb)


class TestPdalCommands:
    """Test module for building PDAL pipelines used by external benchmark comparisons."""

    @pytest.mark.parametrize("operation", PDAL_COMPARISON_OPERATIONS)
    @pytest.mark.parametrize("driver", ["GPKG", "LAS", "LAZ"])
    def test_comparison_command__essential_stages(
        self,
        operation: PdalComparisonOperation,
        driver: Literal["GPKG", "LAS", "LAZ"],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checks that each PDAL comparison reads the dem and writes the expected point output."""

        # Return the command name directly because this test checks the pipeline without running PDAL
        monkeypatch.setattr("benchmarks.comparisons.pdal._require_command", lambda name: name)
        config = RuntimeConfig(
            shape=(64, 96),
            chunks=(32, 32),
            directory=str(tmp_path),
            workload={"subsample_size": 17},
        )

        # Build the pipeline and read the JSON passed to the PDAL command
        comparison = build_pdal_command(
            operation,
            Case(output_driver=driver),
            config,
            dem_file="source-raster.tif",
        )
        pipeline = json.loads(Path(comparison.pipeline_file).read_text(encoding="utf-8"))["pipeline"]
        stage_types = [stage["type"] for stage in pipeline]

        # Conversion keeps every point, while subsampling randomizes first and keeps the requested count
        writer = "writers.ogr" if driver == "GPKG" else "writers.las"
        point_stages = ["readers.gdal"]
        if driver in ("LAS", "LAZ"):
            point_stages.append("filters.ferry")
            assert pipeline[1]["dimensions"] == "band_1=>Z"
        expected_stages = [*point_stages, writer]
        if operation == "subsample":
            expected_stages = [*point_stages, "filters.randomize", "filters.head", writer]
            assert pipeline[-3]["seed"] == 42
            assert pipeline[-2]["count"] == config.value("subsample_size")
        assert stage_types == expected_stages

        # Both paths use the configured cache, source dem and point output
        assert comparison.command == ["pdal", "pipeline", comparison.pipeline_file]
        assert pipeline[0]["filename"] == "source-raster.tif"
        assert pipeline[0]["gdalopts"] == [f"GDAL_CACHEMAX={config.gdal_cachemax_mb}"]
        assert pipeline[-1]["filename"] == comparison.output_file
        assert Path(comparison.output_file).suffix == f".{driver.lower()}"
        if driver == "GPKG":
            assert pipeline[-1]["ogrdriver"] == "GPKG"
        else:
            assert pipeline[-1]["compression"] == (driver == "LAZ")
            assert pipeline[-1]["extra_dims"] == "all"

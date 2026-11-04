# GeoUtils benchmarks

This directory contains benchmarking tools to assess, record, compare and publish performance (compute time and memory usage) for GeoUtils main operations.

Mainly, it contains tools for running:
- Repeatable performance measurements (=benchmarking) that monitor and improve performance using [ASV](https://github.com/airspeed-velocity/asv),
  both to facilitate the check of a given functionality locally when developping, and to run a full benchmarking suite (~1h) that publishes
  detailed performance results to a GitHub page (for every PR merged into main),
- Large data tests for Pytest (pass or fail), which are less exhaustive but run faster (~10min) to quickly catch large regressions (runs every commit to a PR).

## Organization

- `comparisons/` contains GDAL commands, PDAL pipelines, and more, to perform comparisons of external operations equivalent to those in GeoUtils,
- `workflows/config.py` defines the configuration of input sizes and worker parameters shared by all benchmarks,
- `workflows/io.py` defines low-level input opening, output writing and fixtures shared by most operations,
- `workflows/operations/` contains individual files defining GeoUtils benchmarked **operations** (e.g. ``reproject()``, ``grid()``); its **cases**, which are combinations of
  methods (e.g. ``resampling="linear"``), calculation engines (e.g., SciPy, Numba), chunk strategies (e.g., "dense" or "sparse" for grouped stats),
  execution modes (in-memory, Dask, multiprocessing), and input data ranges (e.g., raster or point size); and finally defines **comparisons**,
  which selects chosen benchmark cases for generating comparison plots, including comparisons with external implementations (GDAL, PDAL, etc),
- `workflows/operations/__init__.py` automatically collects the operations, benchmarks and comparisons defined in the ``operations/`` directory,
- `workflows/core.py` contains the `Operation`, `Case`, `Benchmark` and `Comparison` objects that facilitate the definitions of benchmarks,
- `workflows/runner.py` contains the logic about worker execution (in-memory vs Dask vs multiprocessing) and links to GeoUtils profiling,
- `asv_suite/benchmarks.py` defines a small class to register required ASV methods on all benchmarks,
- `asv_suite/render_results.py` renders the raw measurements into HTML/graphics used by the GitHub pages and documentation,
- `test_large_data.py` is a Pytest module to verify that every supported Dask/Multiprocessing operation computes correctly without
  loading the complete raster into memory.

## Adding a benchmark

Benchmarks are defined in `workflows/operations/`, with one file per functionality (e.g., ``reprojection.py`` or ``filters.py``).
Each file describes which GeoUtils function to call (e.g., ``reproject()``), which cases to compare (e.g., Dask/Multiproc, varying raster
input size) and finally if/how to compare to other implementations (e.g., GDAL/PDAL).

The benchmarks rely on four objects:

- An `Operation` stores the functions that prepare and run the **operation**.
- A `Case` stores one fixed implementation **case** (method, engine, execution mode, etc) of its operation. Each case becomes one ASV benchmark.
- A `Benchmark` combines one **operation with all its possible cases**. It can also define one varied input parameter and its
  values; without that parameter, it is a fixed benchmark.
- A `Comparison`, returned by `comparison()`, selects the case series shown together in one report plot and the
  `Case` attribute used to label them.

To add a new benchmark, follow these steps (the code in `operations/filters.py` is an easy example that illustrates them):

1. Choose the file in `workflows/operations/` to add your new benchmark of a functionality, or create a new one.
2. Write the preparation and execution functions for your operation then register them with `Operation(...)`.
3. Define the cases to run with `Case(...)`, or use `execution_cases()` or `strategy_cases()` to generate common sets.
4. **With a varying input range**, such as raster size or point count, define the values to test and a function that
   builds the settings for each value. Use `parameter_config(...)` to combine the range, operation and cases, then add
   the returned `Benchmark` to `BENCHMARKS`.
5. **Without a varying input range**, pass the fixed cases to `Operation(..., large_data_cases=...)`. Discovery creates
   one fixed `Benchmark` for the operation automatically, so it does not need an entry in `BENCHMARKS`.
6. Add `Comparison` objects to `COMPARISONS` for the report plots you want. Use `comparison(..., by=...)` to select
   cases and their attribute that will be plotted in the Benchmark webpage.
7. To include an external implementation, define it under `comparisons/`, create its case with `reference_case(...)`,
   and include that case in both the benchmark and the relevant comparison.

Automatic discovery reads `OPERATIONS`, `BENCHMARKS` and `COMPARISONS` from each operation module. `CASES` is only a
local naming convention: cases are discovered through `BENCHMARKS`, or through `Operation.large_data_cases` when a
fixed benchmark is created automatically. `BENCHMARKS` and `COMPARISONS` may be omitted when the module has none.

## Performance benchmarks with ASV

### Local outputs

When running ASV, local outputs are generated under the gitignored `results/` directory:

```text
results/
├── asv/
│   ├── env/        # ASV environments
│   ├── results/    # Raw measurements
│   └── html/       # Combined website, starting at index.html
└── documentation/  # Optional local preview of the documentation graphics
```

### Working with our ASV benchmarks locally

Below a short summary on how to use our benchmarks locally, including both typical ASV commands and our custom routines.

To run a benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

For example, `<benchmark-regex>` can be `grid_idw_numba_inmem__rastersize.time_operation` to benchmark ``grid(resampling="idw", engine="numba")``.
We use `_` between function parameters, and `__` before a varied input size.

To compare the performance a new implementation with that of the `main` branch: commit current changes, then use:

```bash
asv continuous main HEAD -b 'grid_idw_numba_inmem__rastersize.time_operation'
```

To save the results and generate the HTML report locally:

```bash
asv run --show-stderr -E existing --bench 'grid_idw_numba_inmem__rastersize.time_operation'
asv publish
python -m benchmarks.asv_suite.render_results
```

Then open `results/asv/html/index.html`.

To visualize the rendering of the GitHub page without running benchmarks, generate fake results and render them locally with:

```bash
python -m benchmarks.asv_suite.render_results --preview
asv preview --browser --html-dir benchmarks/results/asv/preview
```

Pass `--baseline-commit <commit>` to the renderer to add `comparisons/performance-change.md`, an before/after
table for in-memory, Dask and Multiprocessing end-to-end time normalized to the GDAL CLI on the same revision.

To generate the benchmarking figures used both on the benchmarking webpage + linked in the RTD documentation, run:

```bash
python -m benchmarks.asv_suite.render_results --doc-only --doc-dir benchmarks/results/documentation
```

### How our ASV benchmarks run in CI

Three workflows are related to ASV benchmark in our continuous integration.

1. For every PR commit, `benchmark-asv-check` runs a quick check (~10min) of benchmark setups (it uses the `GEOUTILS_ASV_PR_CHECK=1`
environment variable to run only the smallest test case of every benchmark, and otherwise relies on ``asv check``).

2. For every PR merge into `main`, or on weekly schedule, `benchmark-asv` runs the full suite (1h+) and records
the performance results to an `asv-results` branch on the GeoUtils repository (relying on ``asv run``).

3. After a successful run of the previous `benchmark-asv`, `benchmark-publish` automatically rebuilds the
saved history from `asv-results`, re-renders the graphics, and deploys the benchmarking webpage to GitHub Pages.

Note that the last two workflows can be triggered manually. Additionally, the user documentation on ReadTheDocs contains direct
links to the latest graphics published on the Benchmarking page (which are therefore always updated to the latest benchmark build,
on all documentation versions).


## Large data tests

Normal Pytest tests (that run across all OSs and Python versions) skip large data tests by default.
In the CI, they run once on Ubuntu with Python 3.12 for every PR commit.

Locally, run the complete suite with:

```bash
python -m pytest --large-data -m large_data -ra
```

To save time while developing, select one specific test with `-k`  and/or add `--lf` to repeat only failed cases.
Other practical instructions and environment variables to set are documented at the top of `test_large_data.py`.

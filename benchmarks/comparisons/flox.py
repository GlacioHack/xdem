"""Prepare and calculate equivalent GeoUtils and Flox grouped statistics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig

from benchmarks.workflows.core import ExecutionMode


def prepare_grouped_inputs(
    size: int,
    groups_per_axis: int,
    layout: Literal["local", "interleaved"],
    execution_mode: ExecutionMode,
    chunks: tuple[int, int] = (1_000, 1_000),
) -> tuple[Any, Any, Any, NDArrayNum]:
    """Create equivalent in-memory or Dask values, groups, categories and selection for both implementations."""

    # Prepare two bounded signals with different missing observations
    positions = np.arange(size * size).reshape(size, size)
    first = (positions % 997).astype(np.float64) * 0.125
    values = np.stack((first, first * 2 + 10))
    values[0, positions % 17 == 0] = np.nan
    values[1, positions % 29 == 0] = np.nan

    # Keep group membership and the common selection independent of missing values
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    if layout == "local":
        groups = (rows * groups_per_axis // size) * groups_per_axis + columns * groups_per_axis // size
    else:
        groups = (rows % groups_per_axis) * groups_per_axis + columns % groups_per_axis
    groups = groups.astype(np.int32)
    mask = positions % 13 != 0
    categories = np.arange(groups_per_axis**2)

    # Give both implementations the same lazy arrays and spatial partitions
    if execution_mode == "dask":
        import dask.array as da

        values = da.from_array(values, chunks=(1, *chunks))
        groups = da.from_array(groups, chunks=chunks)
        mask = da.from_array(mask, chunks=chunks)
    return values, groups, mask, categories


def compute_geoutils_grouped_stats(
    values: Any,
    groups: Any,
    mask: Any,
    categories: NDArrayNum,
    *,
    mp_config: MultiprocConfig | None = None,
) -> pd.DataFrame:
    """Compute the comparison statistics through public GeoUtils stats()."""

    from geoutils.stats import stats

    return stats(
        {"first": values[0], "second": values[1]},
        ("mean", "std"),
        by={"zone": groups},
        categories={"zone": categories},
        mask=mask,
        observed=False,
        mp_config=mp_config,
    )


def grouped_stats(
    values: Any,
    groups: Any,
    mask: Any,
    categories: np.ndarray[Any, Any],
    *,
    use_dask: bool,
) -> pd.DataFrame:
    """Return finite count, mean and population standard deviation from direct Flox reductions."""

    try:
        import flox
    except ImportError as exc:
        raise NotImplementedError("Install optional flox to run this comparison") from exc

    # Apply the common selection and each value's missing data before reducing both arrays together
    if use_dask:
        import dask.array as da

        selected = da.where(mask[None, ...] & da.isfinite(values), values, np.nan)
    else:
        selected = np.where(mask[None, ...] & np.isfinite(values), values, np.nan)

    # Ask Flox for the three public results used by the equivalent GeoUtils stats() call
    axes = tuple(range(-groups.ndim, 0))
    method = "map-reduce" if use_dask else None
    reductions = []
    for function, fill_value, dtype, finalize_kwargs in (
        ("count", 0, np.int64, None),
        ("nanmean", np.nan, np.float64, None),
        ("nanstd", np.nan, np.float64, {"ddof": 0}),
    ):
        result = flox.groupby_reduce(
            selected,
            groups,
            func=function,
            expected_groups=categories,
            sort=False,
            axis=axes,
            fill_value=fill_value,
            dtype=dtype,
            method=method,
            finalize_kwargs=finalize_kwargs,
        )[0]
        reductions.append(result)
    if use_dask:
        import dask

        reductions = list(dask.compute(*reductions))

    # Match the complete typed table returned by GeoUtils without calling any GeoUtils formatter
    count, mean, std = reductions
    data = np.stack((count, mean, std), axis=-1).transpose(1, 0, 2).reshape(len(categories), -1)
    columns = pd.MultiIndex.from_product(
        (("first", "second"), ("count", "mean", "std")),
        names=("value", "statistic"),
    )
    table = pd.DataFrame(data, index=pd.Index(categories, name="zone"), columns=columns)
    for name in ("first", "second"):
        table[(name, "count")] = table[(name, "count")].astype(np.int64)
    return table

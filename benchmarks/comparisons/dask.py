"""
Prepare and calculate Dask equivalent to GeoUtils topk/reduction (required to support various sampling/stats method
while supporting them through the Multiprocessing backend).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def splitmix64(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Map unsigned integer cell numbers to deterministic SplitMix64 keys."""

    values = np.asarray(values, dtype=np.uint64)
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    mixed = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    mixed = (mixed ^ (mixed >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    mixed &= mask
    mixed = (mixed ^ (mixed >> 27)) * np.uint64(0x94D049BB133111EB)
    mixed &= mask
    mixed ^= mixed >> 31
    return mixed.astype(np.uint64, copy=False)


def global_statistics(values: Any) -> tuple[dict[str, Any], Any]:
    """Compute the benchmark finite count and global estimates through Dask reductions."""

    import dask
    import dask.array as da

    squared = da.square(values)
    estimates = {
        "mean": da.nanmean(values),
        "min": da.nanmin(values),
        "max": da.nanmax(values),
        "sum": da.nansum(values),
        "sumofsquares": da.nansum(squared),
        "rmse": da.sqrt(da.nanmean(squared)),
        "std": da.nanstd(values),
    }
    count = da.isfinite(values).sum()
    computed = dask.compute(estimates, count)
    return computed[0], computed[1]


def topk_indices(values: Any, sample_size: int) -> Any:
    """Select deterministic valid cell keys through Dask argtopk()."""

    import dask.array as da

    flat = values.reshape(-1)
    valid = da.isfinite(flat)
    count = min(sample_size, int(valid.sum().compute()))
    cell_numbers = da.arange(flat.size, chunks=flat.chunks, dtype=np.int64)
    key_input = np.uint64(42) ^ cell_numbers.astype(np.uint64)
    keys = key_input.map_blocks(splitmix64, dtype=np.uint64)
    eligible_keys = da.where(valid, keys, np.iinfo(np.uint64).max)
    return da.argtopk(eligible_keys, -count, split_every=8).compute()


def topk_keys(values: Any, sample_size: int) -> Any:
    """Select deterministic valid cell keys through Dask topk()."""

    import dask.array as da

    flat = values.reshape(-1)
    valid = da.isfinite(flat)
    cell_numbers = da.arange(flat.size, chunks=flat.chunks, dtype=np.int64)
    key_input = np.uint64(42) ^ cell_numbers.astype(np.uint64)
    keys = key_input.map_blocks(splitmix64, dtype=np.uint64)
    eligible_keys = da.where(valid, keys, np.iinfo(np.uint64).max)
    return da.topk(eligible_keys, -sample_size, split_every=8).compute()

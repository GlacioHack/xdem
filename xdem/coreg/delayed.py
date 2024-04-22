"""
Temporary module for delayed functions used in coregistration,
most should be probably be moved to GeoUtils/array and integrated into the GeoUtils Xarray accessor

See xDEM discussion thread: https://github.com/GlacioHack/xdem/discussions/501
"""
import warnings
from typing import Any, Literal

import numpy as np
import dask.delayed
import dask.array as da

from scipy.interpolate import interpn

# 1/ SUBSAMPLING
# Getting an exact subsample size only for valid values with small memory usage using dask.delayed
# This is not trivial because of ragged output considerations (varying output length)
# see https://blog.dask.org/2021/07/02/ragged-output
# The function dask.array.map_blocks creates larger RAM usage by having to drop an axis (re-chunk along 1D of the
# 2D array), so we use our own dask.delayed implementation

def _random_state_from_user_input(random_state: np.random.RandomState | int | None = None) -> np.random.RandomState:
    """Define random state based on varied user input."""

    # Define state for random sampling (to fix results during testing)
    # Not using the legacy random call is crucial for RAM usage: https://github.com/numpy/numpy/issues/14169
    if random_state is None:
        rnd: np.random.RandomState | np.random.Generator = np.random.default_rng()
    elif isinstance(random_state, np.random.RandomState):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    return rnd

def _get_subsample_size_from_user_input(subsample: int | float, total_valids: int) -> int:
    """Get subsample size based on a user input of either integer size or fraction of the number of valid points."""

    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * total_valids)
    elif subsample > 1:
        # Use the number of valid points if larger than subsample asked by user
        npoints = min(int(subsample), total_valids)
        if subsample > total_valids:
            warnings.warn(f"Subsample value of {subsample} is larger than the number of valid pixels of {total_valids},"
                          f"using all valid pixels as a subsample.", category=UserWarning)
    else:
        raise ValueError("Subsample must be > 0.")

    return npoints

def _get_indices_block_per_subsample(xxs: np.ndarray, num_chunks: tuple[int, int], list_valids: list[int]) -> list[list[int]]:
    """Get indices of block where each 1D subsample belongs."""

    # Apply a cumulative sum to valids to get their first 1D total index
    valids_cumsum = np.cumsum(list_valids)

    # We can write a faster algorithm by sorting
    xxs = np.sort(xxs)

    # We define a list of indices bucket per block.
    ind_buckets = [[] for _ in range((num_chunks[0] * num_chunks[1]))]
    k = 0
    for i, x in enumerate(xxs):

        # Move to the next block(s) where current index is, if not in this one
        while xxs > valids_cumsum[k]:
            k += 1

        # Add index ID to block
        ind_buckets[k].append(i)

    return ind_buckets

@dask.delayed
def _delayed_subsample_block(arr_chunk: np.ndarray, subsample_indices: np.ndarray) -> np.ndarray:
    """Subsample the values at the corresponding indices per block."""

    s_chunk = arr_chunk[np.isfinite(arr_chunk)][subsample_indices]

    return s_chunk

def delayed_subsample(darr: da.Array,
                      subsample: int | float = 1,
                      return_indices: bool = False,
                      random_state: np.random.RandomState | int | None = None):
    """
    Subsample a raster at valid values on out-of-memory chunks.

    First, the number of valid values in each chunk are identified.
    Then, the number of values to be sampled is partitioned randomly among the chunks, giving them a flattened index
    of where to sample among their valid values.

    Optionally, this function can return the index of the subsample.

    :param darr: Input dask array.
    :param subsample: Subsample size. If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of valid pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations.

    :return: Subsample of values from the array (optionally, their indexes).
    """

    # Get random state
    rnd = _random_state_from_user_input(random_state=random_state)

    # Compute number of valid points for each block out-of-memory
    def nb_valids_func(arr, axis, keepdims):
        return np.count_nonzero(np.isfinite(arr), axis, keepdims=keepdims)

    def keep_same(arr, axis, keepdims):
        return arr
    list_valids = da.reduction(darr, nb_valids_func, aggregate=keep_same, axis=None, dtype=np.int32).compute()

    # Sum to get total number of valid points
    total_valids = np.sum(list_valids)

    # Get subsample size (depending on user input)
    subsample_size = _get_subsample_size_from_user_input(subsample=subsample, total_valids=total_valids)

    # Get random 1D indexes for the subsample size
    indices_1d = rnd.choice(subsample_size, total_valids, replace=False)

    # Sort which indexes belong to which chunk
    ind_buckets = _get_indices_block_per_subsample(indices_1d, num_chunks=darr.numblocks, valids=list_valids)

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = darr.to_delayed().ravel()

    # Task a delayed subsample to be computed for each block, skipping blocks with no values to sample
    list_subsamples = [
        _delayed_subsample_block(b, ind)
        for i, (b, ind) in enumerate(zip(blocks, ind_buckets))
        if len(list_valids[i]) > 0
    ]

    # Cast output to the right expected dtype and length, then compute and concatenate
    list_subsamples_delayed = [da.from_delayed(s, shape=(list_valids[i]), dtype=darr.dtype)
                               for i, s in enumerate(list_subsamples)]
    subsamples = np.concatenate(dask.compute(*list_subsamples_delayed), axis=0)

    return subsamples


# 2/ POINT INTERPOLATION ON REGULAR OR EQUAL GRID
# This functionality is not covered efficiently by Dask/Xarray, because they need to support rectilinear grids, which
# is difficult when interpolating in the chunked dimensions, loading nearly all array memory when using .interp().

# Here we harness the fact that rasters are always on regular (or sometimes equal) grids to efficiently map
# the location of the blocks required for interpolation, which requires little memory usage.

# Code inspired by https://blog.dask.org/2021/07/02/ragged-output and the "block_id" in map_blocks

def _get_interp_indices_per_block(interp_x, interp_y, starts, num_chunks, chunksize, xres, yres):
    """Map blocks where each pair of interpolation coordinates will have to be computed."""

    # TODO 1: Check the robustness for chunksize different and X and Y

    # TODO 2: Check if computing block_i_id matricially + using an == comparison (possibly delayed) to get index
    #  per block is not more computationally efficient?
    #  (as it uses array instead of nested lists, and nested lists grow in RAM very fast)

    # We use one bucket per block, assuming a flattened blocks shape
    ind_buckets = [[] for _ in range((num_chunks[0] * num_chunks[1]))]
    for i, (x, y) in enumerate(zip(interp_x, interp_y)):
        # Because it is a regular grid, we know exactly in which block ID the coordinate will fall
        block_i_1d = int((x - starts[0][0]) / (xres * chunksize[0])) * num_chunks[1] + int((y - starts[1][0])/ (yres * chunksize[1]))
        ind_buckets[block_i_1d].append(i)

    return ind_buckets


@dask.delayed
def _delayed_interp_block(arr_chunk: np.ndarray, block_id: dict[str, Any], interp_coords: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Interpolate block in 2D out-of-memory for a regular or equal grid.
    """

    # Extract information out of block_id dictionary
    xs, ys, xres, yres = (block_id["xstart"], block_id["ystart"], block_id["xres"], block_id["yres"])

    # Reconstruct the coordinates from xi/yi/xres/yres (as it has to be a regular grid)
    x_coords = np.arange(xs, xs + xres*arr_chunk.shape[0], xres)
    y_coords = np.arange(ys, ys + yres*arr_chunk.shape[1], yres)

    # TODO: Use scipy.map_coordinates for an equal grid as in Raster.interp_points?

    # Interpolate to points
    interp_chunk = interpn(points=(x_coords, y_coords), values=arr_chunk, xi=interp_coords)

    # And return the interpolated array
    return interp_chunk

def delayed_interp_points(darr: da.Array,
                          points: tuple[list[float], list[float]],
                          resolution = tuple[float, float],
                          method: Literal["nearest", "linear", "cubic", "quintic"] = "linear"):
    """
    Interpolate raster at point coordinates on out-of-memory chunks.

    This function harnesses the fact that a raster is defined on a regular (or equal) grid, and it is therefore
    faster than Xarray.interpn (especially for small sample sizes) and uses only a fraction of the memory usage.

    :param darr: Input dask array.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be (N,2).
    :param resolution: Resolution of the raster (xres, yres).
    :param method: Interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more information,
            see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.

    :return: Array of raster value(s) for the given points.
    """

    # Map depth of overlap required for each interpolation method
    # TODO: Double-check this window somewhere in SciPy's documentation
    map_depth = {"nearest": 1, "linear": 2, "cubic": 3, "quintic": 5}

    # Expand dask array for overlapping computations
    chunksize = darr.chunksize
    expanded = da.overlap.overlap(darr, depth=map_depth[method], boundary=np.nan)

    # Get robust list of starts (using what is done in block_id of dask.array.map_blocks)
    # https://github.com/dask/dask/blob/24493f58660cb933855ba7629848881a6e2458c1/dask/array/core.py#L908
    from dask.utils import cached_cumsum
    starts = [cached_cumsum(c, initial_zero=True) for c in darr.chunks]
    num_chunks = expanded.numblocks

    # Get samples indices per blocks
    indices = _get_interp_indices_per_block(points[0], points[1], starts, num_chunks, chunksize, resolution[0], resolution[1])

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = expanded.to_delayed().ravel()

    # Build the block IDs by unravelling starting indexes for each block
    indexes_xi, indexes_yi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [{"xstart": starts[0][indexes_xi[i]] - map_depth[method],
                  "ystart": starts[1][indexes_yi[i]] - map_depth[method],
                  "xres": resolution[0],
                  "yres": resolution[1]}
                 for i in range(len(blocks))]

    # Compute values delayed
    list_interp = [_delayed_interp_block(blocks[i],
                                    block_ids[i],
                                    points[indices[i], :])
                                    for i, data_chunk in enumerate(blocks)
                                    if len(indices[i]) > 0
              ]

    # Use np.nan for unknown chunk sizes https://dask.pydata.org/en/latest/array-chunks.html#unknown-chunks
    list_interp_delayed = [da.from_delayed(p, shape=(1, len(indices[i])), dtype=darr.dtype) for i, p in enumerate(list_interp)]
    points = np.concatenate(dask.compute(*list_interp_delayed), axis=0)

    # Re-order per-block output points to match their original indices
    indices = np.concatenate(indices).astype(int)
    argsort = np.argsort(indices)
    points = np.array(points)[argsort]


# 3/ TERRAIN ATTRIBUTES AND APPLY_MATRIX
# Output array is same shape as input, so we can use directly map_overlap here!
# Copyright (c) 2025 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Terrain submodule on windowed attributes: independent calculations in a varying window size."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Callable, Literal

import numpy as np
import scipy
from packaging.version import Version

from xdem._misc import import_optional
from xdem._typing import DTypeLike, NDArrayf

_HAS_VECTORIZED_FILTER = Version(scipy.__version__) >= Version("1.16.0")

# Manage numba as an optional dependency
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake jit decorator if numba is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


#########################################################################
# WINDOWED ATTRIBUTES: INDEPENDENT OF EACH OTHER WITH VARYING WINDOW SIZE
#########################################################################

# Implementation are both in Scipy (non-vectorized and vectorized) and Numba (using JIT on non-vectorized function).

###################################################
# Terrain Ruggedness Index from Riley et al. (1999)
# #################################################
# Ref link: http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf


def _tri_riley_func(arr: NDArrayf) -> float:
    """
    Non-vectorized TRI from Riley, the squareroot of squared sum of differences between center and
    neighbouring pixels.
    """
    mid_ind = int(arr.shape[0] / 2)
    diff = np.abs(arr - arr[mid_ind])
    return np.sqrt(np.sum(diff**2))


# Numba wrapper for TRI from Riley
# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_tri_riley_func_numba = njit(inline="always", cache=True)(_tri_riley_func)


def _tri_riley_func_vectorized(input_block: NDArrayf, axis: tuple[int, ...] = (-2, -1)) -> None:
    """
    Vectorized TRI from Riley, with input sizes (N, w, w) and output size (N).

    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """
    w = input_block.shape[-1]
    mid = w // 2

    center = input_block[..., mid, mid]

    diff = input_block - center[..., None, None]
    output_block = np.sqrt(np.sum(diff * diff, axis=axis))

    return output_block


def _tri_riley_func_scipy(
    dem: NDArrayf, window_size: int, force_backend: Literal["generic", "vectorized"] | None = None
) -> NDArrayf:
    """SciPy wrapper for TRI Riley implementation, with option of forcing backend for tests."""

    # If vectorized is available, use it
    if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":
        tri = scipy.ndimage.vectorized_filter(
            dem,
            _tri_riley_func_vectorized,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )
    # Otherwise fallback on generic function
    else:
        tri = scipy.ndimage.generic_filter(dem, _tri_riley_func, mode="constant", size=window_size, cval=np.nan)

    return tri


####################################################
# Terrain Ruggedness Index from Wilson et al. (2007)
####################################################
# Ref link: http://dx.doi.org/10.1080/01490410701295962


def _tri_wilson_func(arr: NDArrayf, window_size: int) -> float:
    """Non-vectorized TRI from Wilson, the mean difference between center and neighbouring pixels."""
    mid_ind = int(arr.shape[0] / 2)
    diff = np.abs(arr - arr[mid_ind])
    return np.sum(diff) / (window_size**2 - 1)


# Numba wrapper for TRI from Wilson
# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_tri_wilson_func_numba = njit(inline="always", cache=True)(_tri_wilson_func)


def _tri_wilson_func_vectorized(input_block: NDArrayf, window_size: int, axis: tuple[int, ...] = (-2, -1)) -> NDArrayf:
    """
    Vectorized implementation of TRI from Wilson, with input sizes (N, w, w) and output size (N).

    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """
    w = window_size
    mid = w // 2

    center = input_block[..., mid, mid]
    diff = np.abs(input_block - center[..., None, None])

    # Subtract center contribution (which is zero anyway, but explicit is clearer)
    sum_diff = np.sum(diff, axis=axis)

    output_block = sum_diff / (w * w - 1)
    return output_block


def _tri_wilson_func_scipy(
    dem: NDArrayf, window_size: int, force_backend: Literal["generic", "vectorized"] | None = None
) -> NDArrayf:
    """SciPy wrapper for TRI Wilson implementation, with option of forcing backend for tests."""

    # If vectorized is available, use it
    if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":
        part_func = partial(_tri_wilson_func_vectorized, window_size=window_size)
        tri = scipy.ndimage.vectorized_filter(
            dem,
            part_func,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )
    # Otherwise fallback on generic function
    else:
        tri = scipy.ndimage.generic_filter(
            dem,
            _tri_wilson_func,
            mode="constant",
            size=window_size,
            cval=np.nan,
            extra_arguments=(window_size,),
        )

    return tri


##############################################
# Topographic Position Index from Weiss (2001)
##############################################
# Ref link: http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf


def _tpi_func(arr: NDArrayf, window_size: int) -> float:
    """Non-vectorized TPI, the difference between center and mean of neighbouring pixels."""

    mid_ind = int(arr.shape[0] / 2)
    return arr[mid_ind] - (np.sum(arr) - arr[mid_ind]) / (window_size**2 - 1)


# Numba wrapper for TPI
# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_tpi_func_numba = njit(inline="always", cache=True)(_tpi_func)


def _tpi_func_vectorized(input_block: NDArrayf, window_size: int, axis: tuple[int, ...] = (-2, -1)) -> NDArrayf:
    """
    Vectorized TPI, with input sizes (N, w, w) and output size (N).

    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """
    w = window_size
    mid = w // 2

    center = input_block[..., mid, mid]
    sum_all = np.sum(input_block, axis=axis)
    sum_neighbors = sum_all - center

    output_block = center - sum_neighbors / (w * w - 1)

    return output_block


def _tpi_func_scipy(
    dem: NDArrayf, window_size: int, force_backend: Literal["generic", "vectorized"] | None = None
) -> NDArrayf:
    """SciPy wrapper for TPI implementation, with option of forcing backend for tests."""

    # If vectorized is available, use it
    if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":
        _part_func = partial(_tpi_func_vectorized, window_size=window_size)
        tpi = scipy.ndimage.vectorized_filter(
            dem,
            _part_func,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )

    # Otherwise fallback on generic function
    else:
        tpi = scipy.ndimage.generic_filter(
            dem,
            _tpi_func,
            mode="constant",
            size=window_size,
            cval=np.nan,
            extra_arguments=(window_size,),
        )

    return tpi


################################
# Roughness from Dartnell (2000)
################################
# Ref link: https://environment.sfsu.edu/node/11292


def _roughness_func(arr: NDArrayf) -> float:
    """Non-vectorized roughness from Dartnell, the difference between maximum and minimum of the window."""
    if np.count_nonzero(np.isnan(arr)) > 0:
        return float("nan")  # This is somehow necessary for Numba to not ignore NaNs
    else:
        return float(np.max(arr) - np.min(arr))


# Numba wrapper for roughness
# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_roughness_func_numba = njit(inline="always", cache=True)(_roughness_func)


def _roughness_func_vectorized(input_block: NDArrayf, axis: tuple[int, ...] = (-2, -1)) -> NDArrayf:
    """
    Vectorized roughness with input sizes (N, w, w) and output size (N).

    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """
    has_nan = np.isnan(input_block).any(axis=axis)

    max_val = np.max(input_block, axis=axis)
    min_val = np.min(input_block, axis=axis)

    output_block = max_val - min_val
    output_block[has_nan] = np.nan

    return output_block


def _roughness_func_scipy(
    dem: NDArrayf, window_size: int, force_backend: Literal["generic", "vectorized"] | None = None
) -> NDArrayf:
    """SciPy wrapper for TPI implementation, with option of forcing backend for tests."""

    # If vectorized is available, use it
    if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":
        return scipy.ndimage.vectorized_filter(
            dem,
            _roughness_func_vectorized,
            size=window_size,
            mode="constant",
            cval=np.nan,
        )
    # Otherwise fallback on generic function
    else:
        return scipy.ndimage.generic_filter(dem, _roughness_func, mode="constant", size=window_size, cval=np.nan)


###############################################
# Fractal roughness from Taud and Parrot (2005)
###############################################
# Ref link: https://doi.org/10.4000/geomorphologie.622


def _fractal_roughness_func(arr: NDArrayf, window_size: int, out_dtype: DTypeLike = np.float32) -> float:
    """Non-vectorized fractal roughness according to the box-counting method of Taud and Parrot."""

    # First, we compute the number of voxels for each pixel of Equation 4
    mid_ind = int(np.floor(arr.shape[0] / 2))
    hw = int(np.floor(window_size / 2))
    mid_val = arr[mid_ind]

    count = 0
    V = np.empty((window_size, window_size), dtype=out_dtype)
    for j in range(window_size):
        for k in range(window_size):
            T = arr[window_size * j + k] - mid_val
            # The following is the equivalent of np.clip, written like this for numba
            if T < 0:
                V[j, k] = 0
            elif T > window_size:
                V[j, k] = window_size
            else:
                V[j, k] = T
            count += 1

    # Then, we compute the maximum number of voxels for varying box splitting of the cube of side the window
    # size, following Equation 5

    # Get all the divisors of the half window size
    list_box_sizes = np.zeros((hw,), dtype=np.uint8)
    for j in range(1, hw + 1):
        if hw % j == 0:
            list_box_sizes[j - 1] = j

    valids = list_box_sizes != 0
    sub_list_box_sizes = list_box_sizes[valids]

    Ns = np.empty((len(sub_list_box_sizes),), dtype=out_dtype)
    for l0 in range(0, len(sub_list_box_sizes)):
        # We loop over boxes of size q x q in the cube
        q = sub_list_box_sizes[l0]
        sumNs = 0
        for j in range(0, int((window_size - 1) / q)):
            for k in range(0, int((window_size - 1) / q)):
                sumNs += np.max(V[slice(j * q, (j + 1) * q), slice(k * q, (k + 1) * q)].flatten())
        Ns[l0] = sumNs / q

    # Finally, we calculate the slope of the logarithm of Ns with q
    # We do the linear regression manually, as np.polyfit is not supported by numba
    x = np.log(sub_list_box_sizes)
    y = np.log(Ns)
    # The number of observations
    n = len(x)
    # Mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)
    # Cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x
    # Calculating slope
    b_1 = SS_xy / SS_xx

    # The fractal dimension D is the opposite of the slope
    D = -b_1

    return D


# Numba wrapper for fractal roughness
# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_fractal_roughness_func_numba = njit(inline="always", cache=True)(_fractal_roughness_func)


def _fractal_precompute(window_size: int) -> tuple[NDArrayf, NDArrayf, float, float]:
    """Pre-compute scale-dependent constants for vectorized fractal roughness."""

    hw = window_size // 2
    qs = np.array([q for q in range(1, hw + 1) if hw % q == 0], dtype=np.int32)

    log_q = np.log(qs)
    n = len(qs)
    mx = log_q.mean()
    SS_xx = np.sum(log_q * log_q) - n * mx * mx

    return qs, log_q, mx, SS_xx  # type: ignore


def _fractal_roughness_func_vectorized(
    input_block: NDArrayf,
    window_size: int,
    qs: NDArrayf,
    log_q: NDArrayf,
    mx: float,
    SS_xx: float,
    axis: tuple[int, ...] = (-2, -1),
) -> NDArrayf:
    """
    Vectorized fractal roughness, with input sizes (N, w, w) and output size (N).

    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """

    w = window_size
    mid = w // 2
    # Get center value
    Zc = input_block[..., mid, mid]

    # Voxel count V of Equation 4
    V = np.clip(input_block - Zc[..., None, None], 0, w)

    # Then Equation 5
    Ns = []
    for q in qs:
        nq = (w - 1) // q
        # Reshape into blocks of q x q
        # Shape: (..., nq, q, nq, q)
        blocks = V[..., : nq * q, : nq * q]
        blocks = blocks.reshape(blocks.shape[:-2] + (nq, q, nq, q))
        # Max per block
        max_per_block = blocks.max(axis=(axis[0] - 1, axis[1]))
        Ns.append(max_per_block.sum(axis=axis) / q)

    Ns = np.stack(Ns, axis=-1)

    # Linear regression in log–log space
    y = np.log(Ns)
    my = y.mean(axis=-1)
    SS_xy = np.sum(y * log_q, axis=-1) - len(qs) * my * mx
    b1 = SS_xy / SS_xx
    output_block = -b1

    return output_block


def _fractal_roughness_func_scipy(
    dem: NDArrayf,
    window_size: int,
    out_dtype: DTypeLike = np.float32,
    force_backend: Literal["generic", "vectorized"] | None = None,
) -> NDArrayf:
    """SciPy wrapper for fractal roughness implementation, with option of forcing backend for tests."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice.")
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in .*",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in .*",
        )

        # If vectorized is available, use it
        if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":

            # Pre-compute scale-dependent constants
            qs, log_q, mx, SS_xx = _fractal_precompute(window_size)

            _part_func = partial(
                _fractal_roughness_func_vectorized, window_size=window_size, qs=qs, log_q=log_q, mx=mx, SS_xx=SS_xx
            )
            return scipy.ndimage.vectorized_filter(
                dem,
                _part_func,
                size=window_size,
                mode="constant",
                cval=np.nan,
            )

        # Otherwise fallback on generic function
        else:
            return scipy.ndimage.generic_filter(
                dem,
                _fractal_roughness_func,
                size=window_size,
                mode="constant",
                cval=np.nan,
                extra_arguments=(window_size, out_dtype),
            )


##############################
# Rugosity from Jenness (2004)
##############################
# Ref link: https://doi.org/10.2193/0091-7648(2004)032%5B0829:CLSAFD%5D2.0.CO;2


def _rugosity_func(arr: NDArrayf, resolution: float, out_dtype: DTypeLike = np.float32) -> float:
    """
    Non-vectorized rugosity, the difference between real surface area and planimetric surface area.

    The below computation only works for a 3x3 array, would need more effort to generalize it.
    """

    # Works only on a 3x3 block
    Z = arr
    L = resolution

    # Rugosity is computed on a 3x3 window like the quadratic coefficients, see Jenness (2004) for details

    # For this, we need elevation differences and horizontal length of 16 segments
    dzs = np.zeros((16,), dtype=out_dtype)
    dls = np.zeros((16,), dtype=out_dtype)

    count_without_center = 0
    count_all = 0
    # First, the 8 connected segments from the center cells, the center cell is index 4
    for j in range(-1, 2):
        for k in range(-1, 2):

            # Skip if this is the center pixel
            if j == 0 and k == 0:
                count_all += 1
                continue
            # The first eight elevation differences from the cell center
            dzs[count_without_center] = Z[4] - Z[count_all]
            # The first eight planimetric length that can be diagonal or straight from the center
            dls[count_without_center] = np.sqrt(j**2 + k**2) * L
            count_all += 1
            count_without_center += 1

    # Manually for the remaining eight segments between surrounding pixels:
    # First, four elevation differences along the x axis
    dzs[8] = Z[0] - Z[1]
    dzs[9] = Z[1] - Z[2]
    dzs[10] = Z[6] - Z[7]
    dzs[11] = Z[7] - Z[8]
    # Second, along the y axis
    dzs[12] = Z[0] - Z[3]
    dzs[13] = Z[3] - Z[6]
    dzs[14] = Z[2] - Z[5]
    dzs[15] = Z[5] - Z[8]
    # For the planimetric lengths, all are equal to one
    dls[8:] = L

    # Finally, the half-surface length of each segment
    hsl = np.sqrt(dzs**2 + dls**2) / 2

    # Starting from up direction anticlockwise, every triangle has 2 segments between center and surrounding
    # pixels and 1 segment between surrounding pixels; pixel 4 is the center
    # above 4 the index of center-surrounding segment decrease by 1, as the center pixel was skipped
    # Triangle 1: pixels 3 and 0
    T1 = (hsl[3], hsl[0], hsl[12])
    # Triangle 2: pixels 0 and 1
    T2 = (hsl[0], hsl[1], hsl[8])
    # Triangle 3: pixels 1 and 2
    T3 = (hsl[1], hsl[2], hsl[9])
    # Triangle 4: pixels 2 and 5
    T4 = (hsl[2], hsl[4], hsl[14])
    # Triangle 5: pixels 5 and 8
    T5 = (hsl[4], hsl[7], hsl[15])
    # Triangle 6: pixels 8 and 7
    T6 = (hsl[7], hsl[6], hsl[11])
    # Triangle 7: pixels 7 and 6
    T7 = (hsl[6], hsl[5], hsl[10])
    # Triangle 8: pixels 6 and 3
    T8 = (hsl[5], hsl[3], hsl[13])

    list_T = [T1, T2, T3, T4, T5, T6, T7, T8]

    # Finally, we compute the 3D surface areas of the 8 triangles
    A = np.empty((8,), dtype=out_dtype)
    count = 0
    for T in list_T:
        # Half sum of lengths
        hs = sum(T) / 2
        # Surface area of triangle
        A[count] = np.sqrt(hs * (hs - T[0]) * (hs - T[1]) * (hs - T[2]))
        count += 1

    rug = sum(A) / L**2

    return rug


# The inline="always" is required to have the nested jit code behaving similarly as if it was in the original function
# We lose speed-up by a factor of ~5 without it
_rugosity_func_numba = njit(inline="always", cache=True)(_rugosity_func)


def _rugosity_func_vectorized(input_block: NDArrayf, resolution: float, axis: tuple[int, ...] = (-2, -1)) -> NDArrayf:
    """
    Vectorized rugosity from Jenness, with input sizes (N, 3, 3) and output size (N).

    Works on a 3x3 window only.
    Argument "axis" is required by scipy.ndimage.vectorized_filter.
    """

    # Alias for readability
    Z = input_block
    L = resolution

    # Center-relative dz for the 8 neighbors
    Zc = Z[..., 1, 1]

    dz_center = np.stack(
        [
            Zc - Z[..., 0, 0],  # 0
            Zc - Z[..., 0, 1],  # 1
            Zc - Z[..., 0, 2],  # 2
            Zc - Z[..., 1, 0],  # 3
            Zc - Z[..., 1, 2],  # 4
            Zc - Z[..., 2, 0],  # 5
            Zc - Z[..., 2, 1],  # 6
            Zc - Z[..., 2, 2],  # 7
        ],
        axis=-1,
    )

    # Planimetric lengths for center segments
    dl_center = (
        np.array(
            [np.sqrt(2), 1, np.sqrt(2), 1, 1, np.sqrt(2), 1, np.sqrt(2)],
            dtype=Z.dtype,
        )
        * L
    )

    # Compute dz for surrounding-surrounding segments
    dz_edges = np.stack(
        [
            Z[..., 0, 0] - Z[..., 0, 1],  # 8
            Z[..., 0, 1] - Z[..., 0, 2],  # 9
            Z[..., 2, 0] - Z[..., 2, 1],  # 10
            Z[..., 2, 1] - Z[..., 2, 2],  # 11
            Z[..., 0, 0] - Z[..., 1, 0],  # 12
            Z[..., 1, 0] - Z[..., 2, 0],  # 13
            Z[..., 0, 2] - Z[..., 1, 2],  # 14
            Z[..., 1, 2] - Z[..., 2, 2],  # 15
        ],
        axis=-1,
    )

    dl_edges = np.full(8, L, dtype=Z.dtype)

    # Then, combine all segments
    dzs = np.concatenate([dz_center, dz_edges], axis=-1)
    dls = np.concatenate([dl_center, dl_edges])

    # Derive half surface lengths (hsl)
    hsl = np.sqrt(dzs * dzs + dls * dls) / 2

    # Make table of triangle indices into hsl
    tri_idx = np.array(
        [
            [3, 0, 12],
            [0, 1, 8],
            [1, 2, 9],
            [2, 4, 14],
            [4, 7, 15],
            [7, 6, 11],
            [6, 5, 10],
            [5, 3, 13],
        ]
    )

    a = hsl[..., tri_idx[:, 0]]
    b = hsl[..., tri_idx[:, 1]]
    c = hsl[..., tri_idx[:, 2]]

    s = (a + b + c) / 2
    A = np.sqrt(s * (s - a) * (s - b) * (s - c))

    output_block = np.sum(A, axis=-1) / (L * L)

    return output_block


def _rugosity_func_scipy(
    dem: NDArrayf,
    resolution: float,
    out_dtype: DTypeLike = np.float32,
    force_backend: Literal["generic", "vectorized"] | None = None,
) -> NDArrayf:
    """SciPy wrapper for rugosity implementation, with option of forcing backend for tests."""

    # If vectorized is available, use it
    if _HAS_VECTORIZED_FILTER or force_backend == "vectorized":
        _part_func = partial(_rugosity_func_vectorized, resolution=resolution)
        return scipy.ndimage.vectorized_filter(
            dem,
            _part_func,
            size=3,
            mode="constant",
            cval=np.nan,
        )
    # Otherwise fallback on generic function
    else:
        return scipy.ndimage.generic_filter(
            dem,
            _rugosity_func,
            size=3,
            mode="constant",
            cval=np.nan,
            extra_arguments=(resolution, out_dtype),
        )


####################################
# PREPROCESSING AND HEADER FUNCTIONS
####################################


def _preprocess_windowed_indexes(
    windowed_indexes: list[str],
) -> tuple[list[int], list[bool], int]:
    """
    Pre-processing for windowed indexes.

    Map ordered indexes to be used to derive them through SciPy or Numba loop efficiently. (to minimize memory and CPU
    usage)

    Returns list of indexes to map attributes, list of booleans to make attributes, and the size of the output
    attribute array.
    """

    # Step 2: Derive ordered indexes for attributes/coefs outside of SciPy/Numba processing for speed

    # Define booleans for generating attributes
    make_tpi = "topographic_position_index" in windowed_indexes
    make_tri = "terrain_ruggedness_index" in windowed_indexes
    make_roughness = "roughness" in windowed_indexes
    make_rugosity = "rugosity" in windowed_indexes
    make_fractal_roughness = "fractal_roughness" in windowed_indexes

    make_attrs = [
        make_tpi,
        make_tri,
        make_roughness,
        make_rugosity,
        make_fractal_roughness,
    ]

    # Map index of attributes and coefficients to defined order
    order_attrs = [
        "topographic_position_index",
        "terrain_ruggedness_index",
        "roughness",
        "rugosity",
        "fractal_roughness",
    ]
    idx_attrs = [windowed_indexes.index(oa) if oa in windowed_indexes else 99 for oa in order_attrs]

    # Because of the above indexes, we don't store the length of the output attributes anymore
    attrs_size = len(windowed_indexes)

    return idx_attrs, make_attrs, attrs_size


@njit(inline="always", cache=True)  # type: ignore
def _make_windowed_indexes(
    dem_window: NDArrayf,
    window_size: int,
    resolution: float,
    make_attrs: list[bool],
    tpi_idx: int,
    tri_idx: int,
    roughness_idx: int,
    rugosity_idx: int,
    frac_roughness_idx: int,
    tri_method_id: int,
    out_size: tuple[int, ...],
    out_dtype: DTypeLike,
) -> NDArrayf:

    attrs = np.full(out_size, fill_value=np.nan, dtype=out_dtype)

    make_tpi, make_tri, make_roughness, make_rugosity, make_fractal_roughness = make_attrs

    # Topographic position index
    if make_tpi:

        attrs[tpi_idx] = _tpi_func_numba(dem_window, window_size=window_size)

    if make_tri:

        if tri_method_id == 0:
            attrs[tri_idx] = _tri_riley_func_numba(dem_window)

        elif tri_method_id == 1:
            attrs[tri_idx] = _tri_wilson_func_numba(dem_window, window_size=window_size)

    if make_roughness:

        attrs[roughness_idx] = _roughness_func_numba(dem_window)

    if make_rugosity:

        attrs[rugosity_idx] = _rugosity_func_numba(dem_window, resolution=resolution, out_dtype=out_dtype)

    if make_fractal_roughness:

        attrs[frac_roughness_idx] = _fractal_roughness_func_numba(
            dem_window, window_size=window_size, out_dtype=out_dtype
        )

    return attrs


@njit(parallel=True, cache=True)  # type: ignore
def _get_windowed_indexes_numba(
    dem: NDArrayf,
    window_size: int,
    resolution: float,
    out_dtype: DTypeLike,
    attrs_size: int,
    make_attrs: list[bool],
    idx_attrs: list[int],
    tri_method_id: int,
) -> NDArrayf:
    """
    Run the pixel-wise analysis in parallel for any window size without using the resolution.

    See the xdem.terrain.get_windowed_indexes() docstring for more info.
    """

    # Get input shapes
    N1, N2 = dem.shape

    # Define ranges to loop through given padding
    row_range = N1 - window_size + 1
    col_range = N2 - window_size + 1

    # Ugly unpacking as integers outside loop required for Numba to speed-up
    tpi_idx, tri_idx, roughness_idx, rugosity_idx, frac_roughness_idx = idx_attrs

    # Allocate output array
    outputs = np.full((attrs_size, row_range, col_range), fill_value=np.nan, dtype=out_dtype)

    # Loop over every pixel concurrently by using prange
    for row in prange(row_range):
        for col in prange(col_range):

            dem_window = dem[row : row + window_size, col : col + window_size].flatten()
            out_size = (attrs_size,)
            attrs = _make_windowed_indexes(
                dem_window,
                window_size=window_size,
                resolution=resolution,
                make_attrs=make_attrs,
                tpi_idx=tpi_idx,
                tri_idx=tri_idx,
                roughness_idx=roughness_idx,
                rugosity_idx=rugosity_idx,
                frac_roughness_idx=frac_roughness_idx,
                tri_method_id=tri_method_id,
                out_size=out_size,
                out_dtype=out_dtype,
            )

            outputs[:, row, col] = attrs

    return outputs


def _get_windowed_indexes_scipy(
    dem: NDArrayf,
    window_size: int,
    resolution: float,
    make_attrs: list[bool],
    idx_attrs: list[int],
    tri_method_id: int,
    attrs_size: int,
    out_dtype: DTypeLike = np.float32,
    force_backend: Literal["generic", "vectorized"] | None = None,
) -> NDArrayf:

    outputs = np.full((attrs_size, dem.shape[0], dem.shape[1]), fill_value=np.nan, dtype=out_dtype)

    make_tpi, make_tri, make_roughness, make_rugosity, make_fractal_roughness = make_attrs

    if make_tpi:
        tpi = _tpi_func_scipy(dem=dem, window_size=window_size, force_backend=force_backend)
        tpi_idx = idx_attrs[0]
        outputs[tpi_idx] = tpi

    if make_tri:
        tri_idx = idx_attrs[1]
        if tri_method_id == 0:
            tri_ril = _tri_riley_func_scipy(dem=dem, window_size=window_size, force_backend=force_backend)
            outputs[tri_idx] = tri_ril

        elif tri_method_id == 1:
            tri_wil = _tri_wilson_func_scipy(dem=dem, window_size=window_size, force_backend=force_backend)
            outputs[tri_idx] = tri_wil

    if make_roughness:
        roughness_idx = idx_attrs[2]
        roughness = _roughness_func_scipy(dem=dem, window_size=window_size, force_backend=force_backend)
        outputs[roughness_idx] = roughness

    if make_rugosity:
        rugosity_idx = idx_attrs[3]
        rugosity = _rugosity_func_scipy(
            dem=dem, resolution=resolution, out_dtype=out_dtype, force_backend=force_backend
        )
        outputs[rugosity_idx] = rugosity

    if make_fractal_roughness:
        frac_roughness_idx = idx_attrs[4]
        frac_roughness = _fractal_roughness_func_scipy(
            dem=dem, window_size=window_size, out_dtype=out_dtype, force_backend=force_backend
        )
        outputs[frac_roughness_idx] = frac_roughness

    return outputs


def _get_windowed_indexes(
    dem: NDArrayf,
    window_size: int,
    windowed_indexes: list[str],
    resolution: float,
    out_dtype: DTypeLike = np.float32,
    tri_method: Literal["Riley", "Wilson"] = "Riley",
    engine: Literal["scipy", "numba"] = "scipy",
    force_scipy_backend: Literal["generic", "vectorized"] | None = None,
) -> NDArrayf:
    """
    Derive windowed terrain indexes using SciPy or Numba based on a windowed calculation of variable size.

    Includes:

    - Terrain Ruggedness Index from Riley et al. (1999),
        http://download.osgeo.org/qgis/doc/reference-docs/Terrain_Ruggedness_Index.pdf, for topography and from Wilson
        et al. (2007), http://dx.doi.org/10.1080/01490410701295962, for bathymetry.
    - Topographic Position Index from Weiss (2001), http://www.jennessent.com/downloads/TPI-poster-TNC_18x22.pdf.
    - Roughness from Dartnell (2000), thesis referenced in Wilson et al. (2007) above.
    - Fractal roughness from Taud et Parrot (2005), https://doi.org/10.4000/geomorphologie.622.

    Nearly all methods are also referenced in Wilson et al. (2007).

    :param dem: Input DEM as 2D array.
    :param window_size: Window size to compute the index.
    :param windowed_indexes: Names of windowed indexes to compute.
    :param out_dtype: Output dtype of the terrain attributes, can only be a floating type. Defaults to that of the
        input DEM if floating type or to float32 if integer type.
    :param tri_method: Method for the terrain ruggedness index ("Riley" or "Wilson").
    :param engine: Engine to compute the windowed indexes ("scipy" or "numba").
    :param force_scipy_backend: (For testing and SciPy only) Whether to use generic_filter or vectorized_filter.
    """

    # Get list of necessary coefficients depending on method and resolution
    idx_attrs, make_attrs, attrs_size = _preprocess_windowed_indexes(windowed_indexes=windowed_indexes)

    # Map slope method to integer ID to improve efficiency in Numba loop
    tri_method_id = 0 if tri_method.lower() == "riley" else 1

    # Run convolution to compute all coefficients, then reduce those to attributes through either SciPy or Numba
    # (For Numba: Reduction is done within loop to reduce memory usage of computing dozens of full-array coefficients)
    if engine == "scipy":
        output = _get_windowed_indexes_scipy(
            dem=dem,
            window_size=window_size,
            resolution=resolution,
            idx_attrs=idx_attrs,
            make_attrs=make_attrs,
            tri_method_id=tri_method_id,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
            force_backend=force_scipy_backend,
        )
    elif engine == "numba":

        # Fail and raise error if optional dependency is not installed
        numba = import_optional("numba")

        hw = int((window_size - 1) / 2)
        dem = np.pad(dem, pad_width=((hw, hw), (hw, hw)), constant_values=np.nan)
        # Now required to declare list typing in latest Numba before deprecation
        typed_make_attrs, typed_idx_attrs = numba.typed.List(), numba.typed.List()
        [typed_make_attrs.append(x) for x in make_attrs]
        [typed_idx_attrs.append(x) for x in idx_attrs]
        output = _get_windowed_indexes_numba(
            dem=dem,
            window_size=window_size,
            resolution=resolution,
            make_attrs=typed_make_attrs,
            idx_attrs=typed_idx_attrs,
            attrs_size=attrs_size,
            out_dtype=out_dtype,
            tri_method_id=tri_method_id,
        )

    return output

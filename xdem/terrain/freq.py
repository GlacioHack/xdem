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

"""Terrain submodule on frequency attributes computed on the full array, such as texture shading."""
from __future__ import annotations

import numpy as np
import scipy.fft as fft

from xdem._typing import NDArrayf

############################
# FREQUENCY-BASED ATTRIBUTES
############################


def _nextprod_fft(n: int) -> int:
    """
    Find the next valid FFT size (power of 2, 3, 5, or 7).

    Based on MATLAB's nextpow2 and optimized for scipy.fft.

    :param n: Input size
    :returns: Next valid FFT size
    """
    if n <= 1:
        return 1

    # For small sizes, use powers of 2
    if n <= 1024:
        return int(2 ** np.ceil(np.log2(n)))

    # For larger sizes, find the smallest m >= n such that m = 2^a * 3^b * 5^c * 7^d
    factors = [2, 3, 5, 7]
    candidate = n

    while True:
        temp = candidate
        for factor in factors:
            while temp % factor == 0:
                temp //= factor
        if temp == 1:
            return candidate
        candidate += 1


def _texture_shading_fft(
    dem: NDArrayf,
    alpha: float | None = 0.8,
) -> NDArrayf:
    """
    Core texture shading implementation using fractional Laplacian operator.

    Based on Leland Brown's texture shading technique from:
    Brown, L. (2010). Texture Shading: A New Technique for Depicting Terrain Relief.
    Workshop on Mountain Cartography, Banff, Canada.

    :param dem: Input DEM array
    :param alpha: Fractional exponent for Laplacian operator (0-2, default 0.8)
    :returns: Texture shaded array
    """
    # Validate inputs
    if alpha is None:
        alpha = 0.8  # Use default value if None
    if not 0 <= alpha <= 2:
        raise ValueError(f"Alpha must be between 0 and 2, got {alpha}")

    # Handle NaN values by creating a mask
    valid_mask = np.isfinite(dem)
    if not np.any(valid_mask):
        return np.full_like(dem, np.nan)

    # Work with a copy to avoid modifying input
    result = dem.copy()

    # Fill NaN values with mean of valid values for processing
    if not np.all(valid_mask):
        result[~valid_mask] = np.nanmean(dem)

    # Get dimensions
    rows, cols = result.shape

    # Determine FFT sizes for optimal performance
    fft_rows = _nextprod_fft(rows)
    fft_cols = _nextprod_fft(cols)

    # Pad the array for FFT
    pad_rows = (fft_rows - rows) // 2
    pad_cols = (fft_cols - cols) // 2

    # Use symmetric padding to reduce edge effects
    result = np.pad(
        result,
        (
            (pad_rows, fft_rows - rows - pad_rows),
            (pad_cols, fft_cols - cols - pad_cols),
        ),
        mode="symmetric",
    )

    # Create frequency domain grids
    fy = fft.fftfreq(fft_rows)[:, None]
    fx = fft.rfftfreq(fft_cols)[None, :]

    # Calculate frequency magnitude (avoiding division by zero)
    freq_magnitude = np.hypot(fx, fy)
    freq_magnitude[0, 0] = 1.0

    # Create fractional Laplacian filter in frequency domain
    # For alpha=1, this is the standard Laplacian
    # For alpha<1, it emphasizes low frequencies
    # For alpha>1, it emphasizes high frequencies
    laplacian_filter = freq_magnitude**alpha
    if alpha > 0:
        laplacian_filter[0, 0] = 0.0  # only zero DC when alpha>0

    # Apply FFT
    result = fft.rfft2(result, s=(fft_rows, fft_cols), overwrite_x=True)

    # Apply fractional Laplacian in frequency domain in-place
    result *= laplacian_filter

    # Transform back to spatial domain
    result = fft.irfft2(result, s=(fft_rows, fft_cols), overwrite_x=True)

    # Extract the original size from padded result
    result = result[pad_rows : pad_rows + rows, pad_cols : pad_cols + cols]

    # Restore NaN values where original data was invalid
    result[~valid_mask] = np.nan

    return result

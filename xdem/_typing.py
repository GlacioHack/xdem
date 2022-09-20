from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numpy.typing import NDArray  # this syntax works starting on Python 3.9

    NDArrayf = NDArray[np.floating[Any]]  # type: ignore
except ImportError:
    NDArrayf = np.array[Any, np.dtype[np.floating[Any]]]  # type: ignore

MArrayf = np.ma.masked_array[Any, np.dtype[np.floating[Any]]]

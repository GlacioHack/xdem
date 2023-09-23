from __future__ import annotations

import sys
from typing import Any

import numpy as np

# Only for Python >= 3.9
if sys.version_info.minor >= 9:

    from numpy.typing import NDArray  # this syntax works starting on Python 3.9

    NDArrayf = NDArray[np.floating[Any]]
    NDArrayb = NDArray[np.bool_]
    MArrayf = np.ma.masked_array[Any, np.dtype[np.floating[Any]]]

else:
    NDArrayf = np.ndarray  # type: ignore
    NDArrayb = np.ndarray  # type: ignore
    MArrayf = np.ma.masked_array  # type: ignore

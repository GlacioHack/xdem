from __future__ import annotations

from typing import Any

import numpy as np

# from numpy.typing import NDArray # this syntax will only work starting on Python 3.9
# NDArrayf = NDArray[np.floating[Any]]

NDArrayf = np.array[Any, np.dtype[np.floating[Any]]]
MArrayf = np.ma.masked_array[Any, np.dtype[np.floating[Any]]]

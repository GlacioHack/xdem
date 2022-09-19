from __future__ import annotations

from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

NDArrayf = NDArray[np.floating[Any]]
MArrayf = np.ma.masked_array[Any, np.dtype[np.floating[Any]]]
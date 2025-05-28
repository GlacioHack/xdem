# Copyright (c) 2024 xDEM developers
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

from __future__ import annotations

import sys
from typing import Any, List, Tuple, Union

import numpy as np

# Only for Python >= 3.9
if sys.version_info.minor >= 9:

    from numpy.typing import (  # this syntax works starting on Python 3.9
        ArrayLike,
        DTypeLike,
        NDArray,
    )

    # Simply define here if they exist
    DTypeLike = DTypeLike
    ArrayLike = ArrayLike

    NDArrayf = NDArray[np.floating[Any]]
    NDArrayb = NDArray[np.bool_]
    MArrayf = np.ma.masked_array[Any, np.dtype[np.floating[Any]]]

else:

    # Make an array-like type (since the array-like numpy type only exists in numpy>=1.20)
    DTypeLike = Union[str, type, np.dtype]  # type: ignore
    ArrayLike = Union[np.ndarray, np.ma.masked_array, List[Any], Tuple[Any]]  # type: ignore

    NDArrayf = np.ndarray  # type: ignore
    NDArrayb = np.ndarray  # type: ignore
    MArrayf = np.ma.masked_array  # type: ignore

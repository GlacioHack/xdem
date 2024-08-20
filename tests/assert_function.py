from typing import Any

import numpy as np
import pandas as pd


def assert_coreg_meta_equal(input1: Any, input2: Any) -> bool:
    """Short test function to check equality of coreg dictionary values."""
    if type(input1) != type(input2):
        return False
    elif isinstance(input1, (str, float, int, np.floating, np.integer, tuple, list)) or callable(input1):
        return input1 == input2
    elif isinstance(input1, np.ndarray):
        return np.array_equal(input1, input2, equal_nan=True)
    elif isinstance(input1, pd.DataFrame):
        return input1.equals(input2)
    else:
        raise TypeError(f"Input type {type(input1)} not supported for this test function.")

"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

import cv2
import numpy as np

import xdem.version


def generate_random_field(shape: tuple[int, int], corr_size: int) -> np.ndarray:
    """
    Generate a semi-random gaussian field (to simulate a DEM or DEM error)

    :param shape: The output shape of the field.
    :param corr_size: The correlation size of the field.

    :examples:
        >>> np.random.seed(1)
        >>> generate_random_field((4, 5), corr_size=2).round(2)
        array([[0.47, 0.5 , 0.56, 0.63, 0.65],
               [0.49, 0.51, 0.56, 0.62, 0.64],
               [0.56, 0.56, 0.57, 0.59, 0.59],
               [0.57, 0.57, 0.57, 0.58, 0.58]])

    :returns: A numpy array of semi-random values from 0 to 1
    """
    field = cv2.resize(
        cv2.GaussianBlur(
            np.repeat(
                np.repeat(
                    np.random.randint(0, 255, (shape[0] // corr_size, shape[1] // corr_size), dtype="uint8"),
                    corr_size,
                    axis=0,
                ),
                corr_size,
                axis=1,
            ),
            ksize=(2 * corr_size + 1, 2 * corr_size + 1),
            sigmaX=corr_size,
        )
        / 255,
        dsize=(shape[1], shape[0]),
    )
    return field


def deprecate(removal_version: str | None = None, details: str | None = None):
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed. 
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """
    def deprecator_func(func):

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warning_text = f"Call to deprecated function '{func.__name__}'."
            if details is not None:
                formatted_details = " " + details.strip().capitalize()
                if not any(formatted_details.endswith(c) for c in ".!?"):
                    formatted_details += "."
                warning_text += formatted_details
            else:
                formatted_details = ""
            if removal_version is not None:
                warning_text = warning_text.strip()
                if not any(warning_text.endswith(c) for c in ".!?"):
                    warning_text += "."
                warning_text += f" This functionality will be removed in version {removal_version}."

                if xdem.version.version >= removal_version:
                    raise ValueError(
                        f"Function '{func.__name__}' was deprecated in {removal_version}."
                        + formatted_details + 
                        f" Current version: {xdem.version.version}."
                    )
            warnings.warn(warning_text, category=DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func

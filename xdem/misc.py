"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import cv2
import numpy as np

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
            np.repeat(np.repeat(
                np.random.randint(0, 255, (shape[0]//corr_size,
                                           shape[1]//corr_size), dtype='uint8'),
                corr_size, axis=0), corr_size, axis=1),
            ksize=(2*corr_size + 1, 2*corr_size + 1),
            sigmaX=corr_size) / 255,
        dsize=(shape[1], shape[0])
    )
    return field

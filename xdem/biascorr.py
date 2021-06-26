"""Bias corrections for DEMs"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import rasterio as rio

import xdem


class AlongTrack(xdem.coreg.Coreg):
    """
    DEM along-track bias correction.

    Estimates the mean (or median, weighted avg., etc.) offset between two DEMs.
    """

    def __init__(self, bias_func : Callable = xdem.robust_stats.robust_polynomial_fit):  # pylint: disable=super-init-not-called
        """
        Instantiate an along-track correction object.

        :param bias_func: The function to use for calculating the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(meta={"bias_func": bias_func})

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], along_angle: Optional[float] = None, verbose: bool = False,**kwargs):
        """Estimate the bias using the bias_func."""

        if verbose:
            print('Getting along-track coordinates')

        diff = ref_dem - tba_dem
        xx, _ = xdem.spatial_tools.get_along(ref_dem,along_angle=along_angle)

        if verbose:
            print("Estimating along-track bias correction with function "+ self.meta['bias_func'].__name__)
        deg, coefs = self.meta["bias_func"](xx,diff,**kwargs)

        if verbose:
            print("Along-track bias estimated")

        self._meta['deg'] = deg
        self._meta["coefs"] = coefs

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the bias to a transform matrix."""

        raise ValueError(
            "Along-track bias-corrections cannot be represented as transformation matrices.")


class AcrossTrack(xdem.coreg.Coreg):
    """
    DEM bias correction.

    Estimates the mean (or median, weighted avg., etc.) offset between two DEMs.
    """

    def __init__(self,
                 bias_func: Callable = xdem.robust_stats.robust_polynomial_fit):  # pylint: disable=super-init-not-called
        """
        Instantiate an across-track correction object.

        :param bias_func: The function to use for calculating the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(meta={"bias_func": bias_func})

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], along_angle: Optional[float] = None, verbose: bool = False, **kwargs):
        """Estimate the bias using the bias_func."""

        if verbose:
            print('Getting across-track coordinates')

        diff = ref_dem - tba_dem
        _, yy = xdem.spatial_tools.get_along(ref_dem, along_angle=along_angle)

        if verbose:
            print("Estimating across-track bias correction with function " + self.meta['bias_func'].__name__)
        deg, coefs = self.meta["bias_func"](yy, diff, **kwargs)

        if verbose:
            print("Across-track bias estimated")

        self._meta['deg'] = deg
        self._meta["coefs"] = coefs

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the bias to a transform matrix."""

        raise ValueError(
            "Across-track bias-corrections cannot be represented as transformation matrices.")


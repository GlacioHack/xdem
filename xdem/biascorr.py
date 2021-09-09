"""Bias corrections for DEMs"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import rasterio as rio

import xdem

class DirectionalBias(xdem.coreg.Coreg):
    """
    For example for DEM along- or across-track bias correction.
    """

    def __init__(self, bias_func : Callable[..., tuple[int, np.ndarray]] = xdem.fit.robust_polynomial_fit):  # pylint: disable=super-init-not-called
        """
        Instantiate an directional bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(meta={"bias_func": bias_func})
        self._is_affine = False

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], angle: Optional[float] = None, verbose: bool = False, **kwargs):
        """Estimate the bias using the bias_func."""

        if verbose:
            print('Getting directional coordinates')

        diff = ref_dem - tba_dem
        x, _ = xdem.spatial_tools.get_xy_rotated(ref_dem,angle=angle)

        if verbose:
            print("Estimating directional bias correction with function "+ self.meta['bias_func'].__name__)
        deg, coefs = self._meta["bias_func"](x,diff,**kwargs)

        if verbose:
            print("Directional bias estimated")

        self._meta['angle'] = angle
        self._meta['degree'] = deg
        self._meta["coefs"] = coefs

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the bias to a transform matrix."""

        raise ValueError(
            "Directional bias-corrections cannot be represented as transformation matrices.")


class TerrainBias(xdem.coreg.Coreg):
    """
    Correct a bias according to terrain, such as elevation or curvature.

    With elevation: often useful for nadir image DEM correction, where the focal length is slightly miscalculated.
    With curvature: often useful for a difference of DEMs with different effective resolution.

    DISCLAIMER: An elevation correction may introduce error when correcting non-photogrammetric biases, as generally
    elevation biases are interlinked with curvature biases.
    See Gardelle et al. (2012) (Figure 2), http://dx.doi.org/10.3189/2012jog11j175, for curvature-related biases.
    """

    def __init__(self, bias_func: Callable[..., tuple[int, np.ndarray]] = xdem.robust_stats.robust_polynomial_fit):
        """
        Instantiate an terrain bias correction object

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(meta={"bias_func": bias_func})
        self._is_affine = False


    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, attribute: np.ndarray,
                  transform: Optional[rio.transform.Affine], weights: Optional[np.ndarray], verbose: bool = False,
                  **kwargs):
        """Estimate the bias using the bias_func."""

        diff = ref_dem - tba_dem

        if verbose:
            print("Estimating terrain bias correction with function " + self.meta['bias_func'].__name__)
        deg, coefs = self._meta["bias_func"](attribute, diff, **kwargs)

        if verbose:
            print("Terrain bias estimated")

        self._meta['degree'] = deg
        self._meta['coefs'] = coefs

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the scaling model to a DEM."""
        model = np.poly1d(self._meta['coefs'])

        return dem + model(dem)

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the scaling model to a set of points."""
        model = np.poly1d(self._meta['coefs'])

        new_coords = coords.copy()
        new_coords[:, 2] += model(new_coords[:, 2])
        return new_coords

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the transform to a matrix, if possible."""
        if self.degree == 0:  # If it's just a bias correction.
            return self._meta["coefficients"][-1]
        elif self.degree < 2:
            raise NotImplementedError
        else:
            raise ValueError("A 2nd degree or higher ZScaleCorr cannot be described as a 4x4 matrix!")



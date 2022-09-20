"""Bias corrections for DEMs"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import rasterio as rio

import xdem
from xdem._typing import NDArrayf


class BiasCorr(xdem.coreg.Coreg):
    """
    Parent class of bias-corrections methods: subclass of Coreg for non-affine methods.
    This is a class made to be subclassed, that simply writes the bias function in the Coreg metadata and defines the
    _is_affine tag as False.
    """

    def __init__(
        self, bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(meta={"bias_func": bias_func})
        self._is_affine = False

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_var: None | dict[str, NDArrayf] = None,
        transform: rio.transform.Affine | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        # FOR DEVELOPERS: This function needs to be implemented in a subclass.
        raise NotImplementedError("This step has to be implemented by subclassing.")


class BiasCorr1D(BiasCorr):
    """
    Bias-correction along a single variable (e.g., angle, terrain attribute, or any other).
    """

    def __init__(
        self, bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 1D bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(bias_func=bias_func)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_var: None | dict[str, NDArrayf] = None,
        transform: None | rio.transform.Affine = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the single provided variable using the bias function."""

        diff = ref_dem - tba_dem

        # Check length of bias variable
        if bias_var is None or len(bias_var) != 1:
            raise ValueError('A single variable has to be provided through the argument "bias_var".')

        # Get variable name
        var_name = list(bias_var.keys())[0]

        if verbose:
            print(
                "Estimating a 1D bias correction along variable {} "
                "with function {}...".format(var_name, self._meta["bias_func"].__name__)
            )

        params = self._meta["bias_func"](bias_var[var_name], diff, **kwargs)

        if verbose:
            print("1D bias estimated.")

        # Save method results and variable name
        self._meta["params"] = params
        self._meta["bias_var"] = var_name


class BiasCorr2D(BiasCorr):
    """
    Bias-correction along two variables (e.g., simultaneously slope and curvature, or simply x/y coordinates).
    """

    def __init__(
        self, bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 2D bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(bias_func=bias_func)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_var: None | dict[str, NDArrayf] = None,
        transform: None | rio.transform.Affine = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the two provided variable using the bias function."""

        diff = ref_dem - tba_dem

        # Check bias variable
        if bias_var is None or len(bias_var) != 2:
            raise ValueError('Two variables have to be provided through the argument "bias_var".')

        # Get variable names
        var_name_1 = list(bias_var.keys())[0]
        var_name_2 = list(bias_var.keys())[1]

        if verbose:
            print(
                "Estimating a 2D bias correction along variables {} and {} "
                "with function {}...".format(var_name_1, var_name_2, self._meta["bias_func"].__name__)
            )

        params = self._meta["bias_func"](bias_var[var_name_1], bias_var[var_name_2], diff, **kwargs)

        if verbose:
            print("2D bias estimated.")

        self._meta["params"] = params
        self._meta["bias_vars"] = [var_name_1, var_name_2]


class BiasCorrND(BiasCorr):
    """
    Bias-correction along N variables (e.g., simultaneously slope, curvature, aspect and elevation).
    """

    def __init__(
        self, bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 2D bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(bias_func=bias_func)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_var: None | dict[str, NDArrayf] = None,
        transform: None | rio.transform.Affine = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the two provided variable using the bias function."""

        diff = ref_dem - tba_dem

        # Check bias variable
        if bias_var is None or len(bias_var) <= 2:
            raise ValueError('More than two variables have to be provided through the argument "bias_var".')

        # Get variable names
        list_var_names = list(bias_var.keys())

        if verbose:
            print(
                "Estimating a 2D bias correction along variables {} "
                "with function {}...".format(", ".join(list_var_names), self._meta["bias_func"].__name__)
            )

        params = self._meta["bias_func"](*list(bias_var.values()), diff, **kwargs)

        if verbose:
            print("2D bias estimated.")

        self._meta["params"] = params
        self._meta["bias_vars"] = list_var_names


class DirectionalBias(BiasCorr1D):
    """
    Bias correction for directional biases, for example along- or across-track of satellite angle.
    """

    def __init__(
        self, bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit, angle: float = 0
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate an directional bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(bias_func=bias_func)
        self._meta["angle"] = angle

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: None | rio.transform.Affine = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias using the bias_func."""

        if verbose:
            print("Getting directional coordinates")

        diff = ref_dem - tba_dem
        x, _ = xdem.spatial_tools.get_xy_rotated(ref_dem, along_track_angle=self._meta["angle"])

        if verbose:
            print("Estimating directional bias correction with function {}".format(self._meta["bias_func"].__name__))

        deg, coefs = self._meta["bias_func"](x, diff, **kwargs)

        if verbose:
            print("Directional bias estimated")

        self._meta["degree"] = deg
        self._meta["coefs"] = coefs


class TerrainBias(BiasCorr1D):
    """
    Correct a bias according to terrain, such as elevation or curvature.

    With elevation: often useful for nadir image DEM correction, where the focal length is slightly miscalculated.
    With curvature: often useful for a difference of DEMs with different effective resolution.

    DISCLAIMER: An elevation correction may introduce error when correcting non-photogrammetric biases, as generally
    elevation biases are interlinked with curvature biases.
    See Gardelle et al. (2012) (Figure 2), http://dx.doi.org/10.3189/2012jog11j175, for curvature-related biases.
    """

    def __init__(
        self,
        bias_func: Callable[..., tuple[int, NDArrayf]] = xdem.fit.robust_polynomial_fit,
        terrain_attribute="maximum_curvature",
    ):
        """
        Instantiate an terrain bias correction object

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        """
        super().__init__(bias_func=bias_func)
        self._meta["terrain_attribute"] = terrain_attribute

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: None | rio.transform.Affine = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias using the bias_func."""

        diff = ref_dem - tba_dem

        if verbose:
            print("Estimating terrain bias correction with function {}".format(self._meta["bias_func"].__name__))
        deg, coefs = self._meta["bias_func"](self._meta["terrain_attribute"], diff, **kwargs)

        if verbose:
            print("Terrain bias estimated")

        self._meta["degree"] = deg
        self._meta["coefs"] = coefs

    def _apply_func(self, dem: NDArrayf, transform: rio.transform.Affine) -> NDArrayf:
        """Apply the scaling model to a DEM."""
        model = np.poly1d(self._meta["coefs"])

        return dem + model(dem)

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        """Apply the scaling model to a set of points."""
        model = np.poly1d(self._meta["coefs"])

        new_coords = coords.copy()
        new_coords[:, 2] += model(new_coords[:, 2])
        return new_coords

    def _to_matrix_func(self) -> NDArrayf:
        """Convert the transform to a matrix, if possible."""
        if self.degree == 0:  # If it's just a bias correction.
            return self._meta["coefficients"][-1]
        elif self.degree < 2:
            raise NotImplementedError
        else:
            raise ValueError("A 2nd degree or higher ZScaleCorr cannot be described as a 4x4 matrix!")

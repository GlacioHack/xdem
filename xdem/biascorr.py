"""Bias corrections for DEMs"""
from __future__ import annotations

from typing import Callable, Any, Literal

import numpy as np
import rasterio as rio
import scipy

import geoutils as gu

from xdem.fit import robust_norder_polynomial_fit, robust_nfreq_sumsin_fit, polynomial_1d, sumsin_1d
from xdem.coreg import Coreg
from xdem._typing import NDArrayf

workflows = {"norder_polynomial_fit": {"bias_func": polynomial_1d, "optimizer": robust_norder_polynomial_fit},
             "nfreq_sumsin_fit": {"bias_func": sumsin_1d, "optimizer": robust_nfreq_sumsin_fit}}

class BiasCorr(Coreg):
    """
    Parent class of bias-corrections methods: subclass of Coreg for non-affine methods.
    This is a class made to be subclassed, that simply writes the bias function in the Coreg metadata and defines the
    _is_affine tag as False.
    """

    def __init__(
        self,
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a bias correction object.

        Using ``workflow_func_and_optimizer`` will call the function

        :param bias_func: A function to fit to the bias.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """

        # TODO: Move this logic to parent class? Depends how we deal with Coreg inheritance eventually
        if bias_workflow is None and bias_func is None:
            raise ValueError("Either `bias_func` or `bias_workflow` need to be defined.")

        if bias_workflow is not None:
            if bias_workflow in workflows.keys():
                bias_func = workflows[bias_workflow]["bias_func"]
                optimizer = workflows[bias_workflow]["optimizer"]
            else:
                raise ValueError("Argument `bias_workflow` must be one of '{}', got {}.".format("', '".join(workflows.keys()), bias_workflow))
        else:
            bias_func = bias_func
            optimizer = None

        super().__init__(meta={"bias_func": bias_func, "optimizer": optimizer})
        self._is_affine = False

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        # FOR DEVELOPERS: This function needs to be implemented in a subclass.
        raise NotImplementedError("This step has to be implemented by subclassing.")

    def _apply_func(
            self,
            dem: NDArrayf,
            transform: rio.transform.Affine,
            crs: rio.crs.CRS,
            bias_vars: None | dict[str, NDArrayf] = None,
            **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        dem + self._meta["bias_func"](*tuple(bias_vars.values()), **self._meta["bias_params"])


class BiasCorr1D(BiasCorr):
    """
    Bias-correction along a single variable (e.g., angle, terrain attribute).
    """

    def __init__(
        self,
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 1D bias correction object.

        :param bias_func: The function to fit the bias.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """
        super().__init__(bias_func=bias_func, bias_workflow=bias_workflow)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the single provided variable using the bias function."""

        diff = ref_dem - tba_dem

        # Check length of bias variable
        if bias_vars is None or len(bias_vars) != 1:
            raise ValueError('A single variable has to be provided through the argument "bias_vars".')

        # Get variable name
        var_name = list(bias_vars.keys())[0]

        if verbose:
            print(
                "Estimating a 1D bias correction along variable {} "
                "with function {}...".format(var_name, self._meta["bias_func"].__name__)
            )

        params = optimizer(f=self._meta["bias_func"],
                           xdata=bias_vars[var_name],
                           ydata=diff,
                           sigma=weights,
                           absolute_sigma=True,
                           **kwargs)

        if verbose:
            print("1D bias estimated.")

        # Save method results and variable name
        self._meta["optimizer"] = optimizer.__name__
        self._meta["bias_params"] = params
        self._meta["bias_vars"] = [var_name]


class BiasCorr2D(BiasCorr):
    """
    Bias-correction along two variables (e.g., X/Y coordinates, slope and curvature simultaneously).
    """

    def __init__(
        self,
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 2D bias correction object.

        :param bias_func: The function to fit the bias.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """
        super().__init__(bias_func=bias_func, bias_workflow=bias_workflow)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the two provided variables using the bias function."""

        diff = ref_dem - tba_dem

        # Check bias variable
        if bias_vars is None or len(bias_vars) != 2:
            raise ValueError('Two variables have to be provided through the argument "bias_vars".')

        # Get variable names
        var_name_1 = list(bias_vars.keys())[0]
        var_name_2 = list(bias_vars.keys())[1]

        if verbose:
            print(
                "Estimating a 2D bias correction along variables {} and {} "
                "with function {}...".format(var_name_1, var_name_2, self._meta["bias_func"].__name__)
            )

        params = optimizer(f=self._meta["bias_func"],
                           xdata=(bias_vars[var_name_1], bias_vars[var_name_2]),
                           ydata=diff,
                           sigma=weights,
                           absolute_sigma=True,
                           **kwargs)

        if verbose:
            print("2D bias estimated.")

        self._meta["bias_params"] = params
        self._meta["bias_vars"] = [var_name_1, var_name_2]


class BiasCorrND(BiasCorr):
    """
    Bias-correction along N variables (e.g., simultaneously slope, curvature, aspect and elevation).
    """

    def __init__(
        self,
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a 2D bias correction object.

        :param bias_func: The function to fit the bias.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """
        super().__init__(bias_func=bias_func, bias_workflow=bias_workflow)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias along the two provided variable using the bias function."""

        diff = ref_dem - tba_dem

        # Check bias variable
        if bias_vars is None or len(bias_vars) <= 2:
            raise ValueError('More than two variables have to be provided through the argument "bias_vars".')

        # Get variable names
        list_var_names = list(bias_vars.keys())

        if verbose:
            print(
                "Estimating a 2D bias correction along variables {} "
                "with function {}.".format(", ".join(list_var_names), self._meta["bias_func"].__name__)
            )

        params = optimizer(f=self._meta["bias_func"],
                           xdata=tuple(bias_vars.values()),
                           ydata=diff,
                           sigma=weights,
                           absolute_sigma=True,
                           **kwargs)
        if verbose:
            print("2D bias estimated.")

        self._meta["bias_params"] = params
        self._meta["bias_vars"] = list_var_names


class DirectionalBias(BiasCorr1D):
    """
    Bias correction for directional biases, for example along- or across-track of satellite angle.
    """

    def __init__(
        self,
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
        angle: float = 0
    ):  # pylint: disable=super-init-not-called
        """
        Instantiate a directional bias correction object.

        :param bias_func: The function to fit the bias. Default: robust polynomial of degree 1 to 6.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """
        super().__init__(bias_func=bias_func, bias_workflow=bias_workflow)
        self._meta["angle"] = angle

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias using the bias_func."""

        if verbose:
            print("Getting directional coordinates.")

        diff = ref_dem - tba_dem
        x, _ = gu.raster.get_xy_rotated(ref_dem, along_track_angle=self._meta["angle"])

        if verbose:
            print("Estimating directional bias correction with function {}...".format(self._meta["bias_func"].__name__))

        params = optimizer(f=self._meta["bias_func"],
                           xdata=x,
                           ydata=diff,
                           sigma=weights,
                           absolute_sigma=True,
                           **kwargs)

        if verbose:
            print("Directional bias estimated.")

        self._meta["degree"] = deg
        self._meta["coefs"] = coefs
        self._meta["bias_vars"] = ["angle"]


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
        bias_func: Callable[..., NDArrayf] = None,
        bias_workflow: Literal["norder_polynomial_fit"] | Literal["nfreq_sumsin_fit"] | None = "norder_polynomial_fit",
        terrain_attribute="maximum_curvature",
    ):
        """
        Instantiate a terrain bias correction object

        :param bias_func: The function to fit the bias.
        :param bias_workflow: A pre-defined function + optimizer workflow to fit the bias.
            Overrides ``bias_func`` and the ``optimizer`` later used in ``fit()``.
        """
        super().__init__(bias_func=bias_func, bias_workflow=bias_workflow)
        self._meta["terrain_attribute"] = terrain_attribute

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Estimate the bias using the bias_func."""

        diff = ref_dem - tba_dem

        if verbose:
            print("Estimating terrain bias correction with function {}...".format(self._meta["bias_func"].__name__))
        deg, coefs = self._meta["bias_func"](self._meta["terrain_attribute"], diff, **kwargs)

        if verbose:
            print("Terrain bias estimated.")

        self._meta["degree"] = deg
        self._meta["coefs"] = coefs
        self._meta["bias_vars"] = [self._meta["terrain_attribute"]]

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
            raise ValueError("A 2nd degree or higher terrain cannot be described as a 4x4 matrix!")

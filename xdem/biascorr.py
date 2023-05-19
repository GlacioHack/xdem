"""Bias corrections for DEMs"""
from __future__ import annotations

from typing import Callable, Any, Literal

import numpy as np
import rasterio as rio
import scipy

import geoutils as gu
import xdem.spatialstats

from xdem.fit import robust_norder_polynomial_fit, robust_nfreq_sumsin_fit, polynomial_1d, sumsin_1d
from xdem.coreg import Coreg
from xdem._typing import NDArrayf

fit_workflows = {"norder_polynomial_fit": {"func": polynomial_1d, "optimizer": robust_norder_polynomial_fit},
                 "nfreq_sumsin_fit": {"func": sumsin_1d, "optimizer": robust_nfreq_sumsin_fit}}

class BiasCorr(Coreg):
    """
    Parent class of bias correction methods: non-rigid coregistrations.

    Made to be subclassed to pass default parameters/dimensions more intuitively, or to provide wrappers for specific
    types of bias corrections (directional, terrain, etc).
    """

    def __init__(
        self,
        fit_or_bin: str = "fit",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a bias correction object.
        """
        # Raise error if fit_or_bin is not defined
        if fit_or_bin not in ["fit", "bin"]:
            raise ValueError("Argument `fit_or_bin` must be 'fit' or 'bin'.")

        # Pass the arguments to the class metadata
        if fit_or_bin == "fit":

            # If a workflow was called, override optimizer and pass proper function
            if isinstance(fit_func, str) and fit_func in fit_workflows.keys():
                fit_optimizer = fit_workflows[fit_func]["optimizer"]
                fit_func = fit_workflows[fit_func]["func"]

            super().__init__(meta={"fit_func": fit_func, "fit_optimizer": fit_optimizer})
        else:
            super().__init__(meta={"bin_sizes": bin_sizes, "bin_statistic": bin_statistic,
                                   "bin_apply_method": bin_apply_method})

        # Update attributes
        self._fit_or_bin = fit_or_bin
        self._is_affine = False

    def _fit_func(
            self,
            ref_dem: NDArrayf,
            tba_dem: NDArrayf,
            bias_vars: None | dict[str, NDArrayf] = None,
            transform: None | rio.transform.Affine = None,
            crs: rio.crs.CRS | None = None,
            weights: None | NDArrayf = None,
            verbose: bool = False,
            **kwargs,
    ):
        """Should only be called through subclassing."""

        # Compute difference and mask of valid data
        diff = ref_dem - tba_dem
        ind_valid = np.logical_and.reduce((np.isfinite(diff), *(np.isfinite(var) for var in bias_vars.values())))

        # Get number of variables
        nd = len(bias_vars)

        # Run fit and save optimized function parameters
        if self._fit_or_bin == "fit":

            if verbose:
                print(
                    "Estimating bias correction along variables {} by fitting "
                    "with function {}.".format(", ".join(list(bias_vars.keys())), self._meta["fit_func"].__name__)
                )

            params = self._meta["fit_optimizer"] \
                (f=self._meta["fit_func"],
                 xdata=[var[ind_valid] for var in bias_vars.values()],
                 ydata=diff[ind_valid],
                 sigma=weights[ind_valid] if weights is not None else None,
                 absolute_sigma=True,
                 **kwargs)

            self._meta["fit_params"] = params
        # Or run binning and save dataframe of result
        else:

            print(
                "Estimating bias correction along variables {} by binning "
                "with statistic {}.".format(", ".join(list(bias_vars.keys())), self._meta["bin_statistic"].__name__)
            )

            df = xdem.spatialstats.nd_binning(values=diff[ind_valid],
                                              list_var=list(bias_vars.values()),
                                              list_var_names=list(bias_vars.keys()),
                                              list_var_bins=self._meta["bin_sizes"],
                                              statistics=(self._meta["bin_statistic"]),
                                              )

            self._meta["bin_dataframe"] = df

        if verbose:
            print("{}D bias estimated.".format(nd))

        # Save bias variable names
        self._meta["bias_vars"] = list(bias_vars.keys())

    def _apply_func(
            self,
            dem: NDArrayf,
            transform: rio.transform.Affine,
            crs: rio.crs.CRS,
            bias_vars: None | dict[str, NDArrayf] = None,
            **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # Apply function to get correction
        if self._fit_or_bin == "fit":
            corr = self._meta["fit_func"](*bias_vars, *self._meta["fit_params"])
        # Apply binning to get correction
        else:
            if self._meta["bin_apply"] == "linear":
                bin_interpolator = xdem.spatialstats.interp_nd_binning(df=self._meta["bin_dataframe"],
                                                                       list_var_names=list(bias_vars.keys()),
                                                                       statistic=self._meta["bin_statistic"])
            else:
                pass
                # TODO: !
                # bin_interpolator =

            corr = bin_interpolator(*bias_vars)

        return corr, transform


class BiasCorr1D(BiasCorr):
    """
    Bias-correction along a single variable (e.g., angle, terrain attribute).

    The correction can be done by fitting a function along the variable, or binning with that variable.
    """

    def __init__(
        self,
        fit_or_bin: str = "fit",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] |
                  Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] | None = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] | None = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] | None = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a 1D bias correction.

        :param fit_or_bin: Whether to fit or bin. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        """
        super().__init__(fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method)

    def _fit_func(
            self,
            ref_dem: NDArrayf,
            tba_dem: NDArrayf,
            bias_vars: None | dict[str, NDArrayf] = None,
            transform: None | rio.transform.Affine = None,
            crs: rio.crs.CRS | None = None,
            weights: None | NDArrayf = None,
            verbose: bool = False,
            **kwargs,
    ):
        """Estimate the bias along the single provided variable using the bias function."""

        # Check number of variables
        if bias_vars is None or len(bias_vars) != 1:
            raise ValueError('A single variable has to be provided through the argument "bias_vars".')

        super()._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, bias_vars=bias_vars, transform=transform, crs=crs,
                          weights=weights, verbose=verbose, **kwargs)



class BiasCorr2D(BiasCorr):
    """
    Bias-correction along two variables (e.g., X/Y coordinates, slope and curvature simultaneously).
    """

    def __init__(
        self,
        fit_or_bin: str = "fit",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] |
                  Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] | None = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] | None = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] | None = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a 2D bias correction.

        :param fit_or_bin: Whether to fit or bin. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        """
        super().__init__(fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):

        # Check number of variables
        if bias_vars is None or len(bias_vars) != 2:
            raise ValueError('Only two variable have to be provided through the argument "bias_vars".')

        super()._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, bias_vars=bias_vars, transform=transform, crs=crs,
                          weights=weights, verbose=verbose, **kwargs)


class BiasCorrND(BiasCorr):
    """
    Bias-correction along N variables (e.g., simultaneously slope, curvature, aspect and elevation).
    """

    def __init__(
        self,
        fit_or_bin: str = "bin",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] |
                  Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] | None = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] | None = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] | None = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a N-D bias correction.

        :param fit_or_bin: Whether to fit or bin. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        """
        super().__init__(fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method)

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):

        # Check bias variable
        if bias_vars is None or len(bias_vars) <= 2:
            raise ValueError('More than two variables have to be provided through the argument "bias_vars".')

        super()._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, bias_vars=bias_vars, transform=transform, crs=crs,
                          weights=weights, verbose=verbose, **kwargs)


class DirectionalBias(BiasCorr1D):
    """
    Bias correction for directional biases, for example along- or across-track of satellite angle.
    """

    def __init__(
        self,
        angle: float = 0,
        fit_or_bin: str = "bin",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] |
                  Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] | None = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] | None = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] | None = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a directional bias correction.

        :param angle: Angle in which to perform the directional correction.
        :param fit_or_bin: Whether to fit or bin. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        """
        super().__init__(fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method)
        self._meta["angle"] = angle

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):

        if verbose:
            print("Estimating rotated coordinates.")

        x, _ = gu.raster.get_xy_rotated(raster=gu.Raster.from_array(data=ref_dem, crs=crs, transform=transform),
                                        along_track_angle=self._meta["angle"])

        super()._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, bias_vars={"angle": x}, transform=transform, crs=crs,
                          weights=weights, verbose=verbose, **kwargs)


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
        terrain_attribute="maximum_curvature",
        fit_or_bin: str = "bin",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] |
                  Literal["nfreq_sumsin"] = "norder_polynomial_fit",
        fit_optimizer: Callable[..., tuple[float]] | None = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | tuple[float]] | None = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] | None = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a terrain bias correction.

        :param terrain_attribute: Terrain attribute to use for correction.
        :param fit_or_bin: Whether to fit or bin. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        """

        super().__init__(fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method)
        self._meta["terrain_attribute"] = terrain_attribute

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: None | rio.transform.Affine = None,
        crs: rio.crs.CRS | None = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ):

        # Derive terrain attribute
        attr = xdem.terrain.get_terrain_attribute(dem=ref_dem,
                                                  attribute=self._meta["attribute"],
                                                  resolution=(transform[0], transform[4]))

        # Run the parent function
        super()._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, bias_vars={self._meta["attribute"]: attr},
                          transform=transform, crs=crs, weights=weights, verbose=verbose, **kwargs)

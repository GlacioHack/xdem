"""Bias corrections (i.e., non-affine coregistration) classes."""
from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Literal

import geoutils as gu
import numpy as np
import pandas as pd
import rasterio as rio
import scipy
from geoutils import Mask
from geoutils.raster import RasterType

import xdem.spatialstats
from xdem._typing import MArrayf, NDArrayf
from xdem.coreg.base import Coreg, CoregType
from xdem.fit import (
    polynomial_1d,
    polynomial_2d,
    robust_nfreq_sumsin_fit,
    robust_norder_polynomial_fit,
    sumsin_1d,
)

fit_workflows = {
    "norder_polynomial": {"func": polynomial_1d, "optimizer": robust_norder_polynomial_fit},
    "nfreq_sumsin": {"func": sumsin_1d, "optimizer": robust_nfreq_sumsin_fit},
}


class BiasCorr(Coreg):
    """
    Parent class of bias correction methods: non-rigid coregistrations.

    Made to be subclassed to pass default parameters/dimensions more intuitively, or to provide wrappers for specific
    types of bias corrections (directional, terrain, etc).
    """

    def __init__(
        self,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "fit",
        fit_func: Callable[..., NDArrayf]
        | Literal["norder_polynomial"]
        | Literal["nfreq_sumsin"] = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a bias correction object.
        """
        # Raise error if fit_or_bin is not defined
        if fit_or_bin not in ["fit", "bin", "bin_and_fit"]:
            raise ValueError(f"Argument `fit_or_bin` must be 'bin_and_fit', 'fit' or 'bin', got {fit_or_bin}.")

        # Pass the arguments to the class metadata
        if fit_or_bin in ["fit", "bin_and_fit"]:

            # Check input types for "fit" to raise user-friendly errors
            if not (callable(fit_func) or (isinstance(fit_func, str) and fit_func in fit_workflows.keys())):
                raise TypeError(
                    "Argument `fit_func` must be a function (callable) "
                    "or the string '{}', got {}.".format("', '".join(fit_workflows.keys()), type(fit_func))
                )
            if not callable(fit_optimizer):
                raise TypeError(
                    "Argument `fit_optimizer` must be a function (callable), " "got {}.".format(type(fit_optimizer))
                )

            # If a workflow was called, override optimizer and pass proper function
            if isinstance(fit_func, str) and fit_func in fit_workflows.keys():
                # Looks like a typing bug here, see: https://github.com/python/mypy/issues/10740
                fit_optimizer = fit_workflows[fit_func]["optimizer"]  # type: ignore
                fit_func = fit_workflows[fit_func]["func"]  # type: ignore

        if fit_or_bin in ["bin", "bin_and_fit"]:

            # Check input types for "bin" to raise user-friendly errors
            if not (
                isinstance(bin_sizes, int)
                or (isinstance(bin_sizes, dict) and all(isinstance(val, (int, Iterable)) for val in bin_sizes.values()))
            ):
                raise TypeError(
                    "Argument `bin_sizes` must be an integer, or a dictionary of integers or iterables, "
                    "got {}.".format(type(bin_sizes))
                )

            if not callable(bin_statistic):
                raise TypeError(
                    "Argument `bin_statistic` must be a function (callable), " "got {}.".format(type(bin_statistic))
                )

            if not isinstance(bin_apply_method, str):
                raise TypeError(
                    "Argument `bin_apply_method` must be the string 'linear' or 'per_bin', "
                    "got {}.".format(type(bin_apply_method))
                )

        # Now we write the relevant attributes to the class metadata
        # For fitting
        if fit_or_bin == "fit":
            meta_fit = {"fit_func": fit_func, "fit_optimizer": fit_optimizer}
            # Somehow mypy doesn't understand that fit_func and fit_optimizer can only be callables now,
            # even writing the above "if" in a more explicit "if; else" loop with new variables names and typing
            super().__init__(meta=meta_fit)  # type: ignore

        # For binning
        elif fit_or_bin == "bin":
            meta_bin = {"bin_sizes": bin_sizes, "bin_statistic": bin_statistic, "bin_apply_method": bin_apply_method}
            super().__init__(meta=meta_bin)  # type: ignore

        # For both
        else:
            meta_bin_and_fit = {
                "fit_func": fit_func,
                "fit_optimizer": fit_optimizer,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
            }
            super().__init__(meta=meta_bin_and_fit)  # type: ignore

        # Update attributes
        self._fit_or_bin = fit_or_bin
        self._is_affine = False

    def fit(  # type: ignore
        self: CoregType,
        reference_dem: NDArrayf | MArrayf | RasterType,
        dem_to_be_aligned: NDArrayf | MArrayf | RasterType,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,  # None if subclass derives biasvar itself
        inlier_mask: NDArrayf | Mask | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int = 1.0,
        verbose: bool = False,
        random_state: None | np.random.RandomState | np.random.Generator | int = None,
        **kwargs: Any,
    ) -> CoregType:

        # Change dictionary content to array
        if bias_vars is not None:
            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

        # Call parent fit to do the pre-processing and return itself
        return super().fit(  # type: ignore
            reference_dem=reference_dem,
            dem_to_be_aligned=dem_to_be_aligned,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            weights=weights,
            subsample=subsample,
            verbose=verbose,
            random_state=random_state,
            bias_vars=bias_vars,
            **kwargs,
        )

    def apply(  # type: ignore
        self,
        dem: RasterType | NDArrayf | MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        resample: bool = True,
        **kwargs: Any,
    ) -> tuple[RasterType | NDArrayf | MArrayf, rio.transform.Affine]:

        # Change dictionary content to array
        if bias_vars is not None:
            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

        # Call parent fit to do the pre-processing and return itself
        return super().apply(
            dem=dem,
            transform=transform,
            crs=crs,
            resample=resample,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        bias_vars: None | dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Should only be called through subclassing."""

        # This is called by subclasses, so the bias_var should always be defined
        # TODO: Move this up to Coreg class, checking kwargs of fit(), or better to overload function
        #  description in fit() here?
        if bias_vars is None:
            raise ValueError("At least one `bias_var` should be passed to the fitting function, got None.")

        # Compute difference and mask of valid data
        diff = ref_dem - tba_dem
        ind_valid = np.logical_and.reduce((np.isfinite(diff), *(np.isfinite(var) for var in bias_vars.values())))

        # Raise errors if all values are NaN after introducing masks from the variables
        # (Others are already checked in Coreg.fit())
        if np.all(~ind_valid):
            raise ValueError("One of the 'bias_vars' had only NaNs.")

        # Get number of variables
        nd = len(bias_vars)

        # Remove random state for keyword argument if its value is not in the optimizer function
        if self._fit_or_bin in ["fit", "bin_and_fit"]:
            fit_func_args = inspect.getfullargspec(self._meta["fit_optimizer"]).args
            if "random_state" not in fit_func_args:
                kwargs.pop("random_state")

        # We need to sort the bin sizes in the same order as the bias variables if a dict is passed for bin_sizes
        if self._fit_or_bin in ["bin", "bin_and_fit"]:
            if isinstance(self._meta["bin_sizes"], dict):
                var_order = list(bias_vars.keys())
                # Declare type to write integer or tuple to the variable
                bin_sizes: int | tuple[int, ...] | tuple[NDArrayf, ...] = tuple(
                    np.array(self._meta["bin_sizes"][var]) for var in var_order
                )
            # Otherwise, write integer directly
            else:
                bin_sizes = self._meta["bin_sizes"]

        # Option 1: Run fit and save optimized function parameters
        if self._fit_or_bin == "fit":

            # Print if verbose
            if verbose:
                print(
                    "Estimating bias correction along variables {} by fitting "
                    "with function {}.".format(", ".join(list(bias_vars.keys())), self._meta["fit_func"].__name__)
                )

            results = self._meta["fit_optimizer"](
                f=self._meta["fit_func"],
                xdata=np.array([var[ind_valid].flatten() for var in bias_vars.values()]).squeeze(),
                ydata=diff[ind_valid].flatten(),
                sigma=weights[ind_valid].flatten() if weights is not None else None,
                absolute_sigma=True,
                **kwargs,
            )

        # Option 2: Run binning and save dataframe of result
        elif self._fit_or_bin == "bin":

            if verbose:
                print(
                    "Estimating bias correction along variables {} by binning "
                    "with statistic {}.".format(", ".join(list(bias_vars.keys())), self._meta["bin_statistic"].__name__)
                )

            df = xdem.spatialstats.nd_binning(
                values=diff[ind_valid],
                list_var=[var[ind_valid] for var in bias_vars.values()],
                list_var_names=list(bias_vars.keys()),
                list_var_bins=bin_sizes,
                statistics=(self._meta["bin_statistic"], "count"),
            )

        # Option 3: Run binning, then fitting, and save both results
        else:

            # Print if verbose
            if verbose:
                print(
                    "Estimating bias correction along variables {} by binning with statistic {} and then fitting "
                    "with function {}.".format(
                        ", ".join(list(bias_vars.keys())),
                        self._meta["bin_statistic"].__name__,
                        self._meta["fit_func"].__name__,
                    )
                )

            df = xdem.spatialstats.nd_binning(
                values=diff[ind_valid],
                list_var=[var[ind_valid] for var in bias_vars.values()],
                list_var_names=list(bias_vars.keys()),
                list_var_bins=bin_sizes,
                statistics=(self._meta["bin_statistic"], "count"),
            )

            # Now, we need to pass this new data to the fitting function and optimizer
            # We use only the N-D binning estimates (maximum dimension, equal to length of variable list)
            df_nd = df[df.nd == len(bias_vars)]

            # We get the middle of bin values for variable, and statistic for the diff
            new_vars = [pd.IntervalIndex(df_nd[var_name]).mid.values for var_name in bias_vars.keys()]
            new_diff = df_nd[self._meta["bin_statistic"].__name__].values
            # TODO: pass a new sigma based on "count" and original sigma (and correlation?)?
            #  sigma values would have to be binned above also

            ind_valid = np.logical_and.reduce((np.isfinite(new_diff), *(np.isfinite(var) for var in new_vars)))

            if np.all(~ind_valid):
                raise ValueError("Only NaNs values after binning, did you pass the right bin edges?")

            results = self._meta["fit_optimizer"](
                f=self._meta["fit_func"],
                xdata=np.array([var[ind_valid].flatten() for var in new_vars]).squeeze(),
                ydata=new_diff[ind_valid].flatten(),
                sigma=weights[ind_valid].flatten() if weights is not None else None,
                absolute_sigma=True,
                **kwargs,
            )

        if verbose:
            print(f"{nd}D bias estimated.")

        # Save results if fitting was performed
        if self._fit_or_bin in ["fit", "bin_and_fit"]:

            # Write the results to metadata in different ways depending on optimizer returns
            if self._meta["fit_optimizer"] in (w["optimizer"] for w in fit_workflows.values()):
                params = results[0]
                order_or_freq = results[1]
                if self._meta["fit_optimizer"] == robust_norder_polynomial_fit:
                    self._meta["poly_order"] = order_or_freq
                else:
                    self._meta["nb_sin_freq"] = order_or_freq

            elif self._meta["fit_optimizer"] == scipy.optimize.curve_fit:
                params = results[0]
                # Calculation to get the error on parameters (see description of scipy.optimize.curve_fit)
                perr = np.sqrt(np.diag(results[1]))
                self._meta["fit_perr"] = perr

            else:
                params = results[0]

            self._meta["fit_params"] = params

        # Save results of binning if it was perfrmed
        elif self._fit_or_bin in ["bin", "bin_and_fit"]:
            self._meta["bin_dataframe"] = df

        # Save bias variable names in any case
        self._meta["bias_vars"] = list(bias_vars.keys())

    def _apply_func(  # type: ignore
        self,
        dem: NDArrayf,
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if bias_vars is None:
            raise ValueError("At least one `bias_var` should be passed to the `apply` function, got None.")

        # Apply function to get correction (including if binning was done before)
        if self._fit_or_bin in ["fit", "bin_and_fit"]:
            corr = self._meta["fit_func"](tuple(bias_vars.values()), *self._meta["fit_params"])

        # Apply binning to get correction
        else:
            if self._meta["bin_apply_method"] == "linear":
                # N-D interpolation of binning
                bin_interpolator = xdem.spatialstats.interp_nd_binning(
                    df=self._meta["bin_dataframe"],
                    list_var_names=list(bias_vars.keys()),
                    statistic=self._meta["bin_statistic"],
                )
                corr = bin_interpolator(tuple(var.flatten() for var in bias_vars.values()))
                first_var = list(bias_vars.keys())[0]
                corr = corr.reshape(np.shape(bias_vars[first_var]))

            else:
                # Get N-D binning statistic for each pixel of the new list of variables
                corr = xdem.spatialstats.get_perbin_nd_binning(
                    df=self._meta["bin_dataframe"],
                    list_var=list(bias_vars.values()),
                    list_var_names=list(bias_vars.keys()),
                    statistic=self._meta["bin_statistic"],
                )

        dem_corr = dem + corr

        return dem_corr, transform


class BiasCorr1D(BiasCorr):
    """
    Bias-correction along a single variable (e.g., angle, terrain attribute).

    The correction can be done by fitting a function along the variable, or binning with that variable.
    """

    def __init__(
        self,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "fit",
        fit_func: Callable[..., NDArrayf]
        | Literal["norder_polynomial"]
        | Literal["nfreq_sumsin"] = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
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

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """Estimate the bias along the single provided variable using the bias function."""

        # Check number of variables
        if len(bias_vars) != 1:
            raise ValueError(
                "A single variable has to be provided through the argument 'bias_vars', "
                "got {}.".format(len(bias_vars))
            )

        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars=bias_vars,
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            **kwargs,
        )


class BiasCorr2D(BiasCorr):
    """
    Bias-correction along two variables (e.g., X/Y coordinates, slope and curvature simultaneously).
    """

    def __init__(
        self,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "fit",
        fit_func: Callable[..., NDArrayf] = polynomial_2d,
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
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

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        # Check number of variables
        if len(bias_vars) != 2:
            raise ValueError(
                "Exactly two variables have to be provided through the argument 'bias_vars'"
                ", got {}.".format(len(bias_vars))
            )

        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars=bias_vars,
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            **kwargs,
        )


class BiasCorrND(BiasCorr):
    """
    Bias-correction along N variables (e.g., simultaneously slope, curvature, aspect and elevation).
    """

    def __init__(
        self,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin",
        fit_func: Callable[..., NDArrayf]
        | Literal["norder_polynomial"]
        | Literal["nfreq_sumsin"] = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate an N-D bias correction.

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

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],  # Never None thanks to BiasCorr.fit() pre-process
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        # Check bias variable
        if bias_vars is None or len(bias_vars) <= 2:
            raise ValueError('At least three variables have to be provided through the argument "bias_vars".')

        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars=bias_vars,
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            **kwargs,
        )


class DirectionalBias(BiasCorr1D):
    """
    Bias correction for directional biases, for example along- or across-track of satellite angle.
    """

    def __init__(
        self,
        angle: float = 0,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin_and_fit",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"] = "nfreq_sumsin",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a directional bias correction.

        :param angle: Angle in which to perform the directional correction (degrees).
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

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        if verbose:
            print("Estimating rotated coordinates.")

        x, _ = gu.raster.get_xy_rotated(
            raster=gu.Raster.from_array(data=ref_dem, crs=crs, transform=transform),
            along_track_angle=self._meta["angle"],
        )

        # Parameters dependent on resolution cannot be derived from the rotated x coordinates, need to be passed below
        if "hop_length" not in kwargs:
            # The hop length will condition jump in function values, need to be larger than average resolution
            average_res = (transform[0] + abs(transform[4])) / 2
            kwargs.update({"hop_length": average_res})

        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars={"angle": x},
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            **kwargs,
        )

    def _apply_func(
        self,
        dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # Define the coordinates for applying the correction
        x, _ = gu.raster.get_xy_rotated(
            raster=gu.Raster.from_array(data=dem, crs=crs, transform=transform),
            along_track_angle=self._meta["angle"],
        )

        return super()._apply_func(dem=dem, transform=transform, crs=crs, bias_vars={"angle": x}, **kwargs)


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
        terrain_attribute: str = "maximum_curvature",
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin",
        fit_func: Callable[..., NDArrayf]
        | Literal["norder_polynomial"]
        | Literal["nfreq_sumsin"] = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 100,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
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

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        # Derive terrain attribute
        if self._meta["terrain_attribute"] == "elevation":
            attr = ref_dem
        else:
            attr = xdem.terrain.get_terrain_attribute(
                dem=ref_dem, attribute=self._meta["terrain_attribute"], resolution=(transform[0], abs(transform[4]))
            )

        # Run the parent function
        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars={self._meta["terrain_attribute"]: attr},
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            **kwargs,
        )

    def _apply_func(
        self,
        dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if bias_vars is None:
            # Derive terrain attribute
            if self._meta["terrain_attribute"] == "elevation":
                attr = dem
            else:
                attr = xdem.terrain.get_terrain_attribute(
                    dem=dem, attribute=self._meta["terrain_attribute"], resolution=(transform[0], abs(transform[4]))
                )
            bias_vars = {self._meta["terrain_attribute"]: attr}

        return super()._apply_func(dem=dem, transform=transform, crs=crs, bias_vars=bias_vars, **kwargs)


class Deramp(BiasCorr2D):
    """
    Correct for a 2D polynomial along X/Y coordinates, for example from residual camera model deformations.
    """

    def __init__(
        self,
        poly_order: int = 2,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "fit",
        fit_func: Callable[..., NDArrayf] = polynomial_2d,
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
    ):
        """
        Instantiate a directional bias correction.

        :param poly_order: Order of the 2D polynomial to fit.
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
        self._meta["poly_order"] = poly_order

    def _fit_func(  # type: ignore
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        bias_vars: dict[str, NDArrayf],
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: None | NDArrayf = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        # The number of parameters in the first guess defines the polynomial order when calling np.polyval2d
        p0 = np.ones(shape=((self._meta["poly_order"] + 1) * (self._meta["poly_order"] + 1)))

        # Coordinates (we don't need the actual ones, just array coordinates)
        xx, yy = np.meshgrid(np.arange(0, ref_dem.shape[1]), np.arange(0, ref_dem.shape[0]))

        super()._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            bias_vars={"xx": xx, "yy": yy},
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            p0=p0,
            **kwargs,
        )

    def _apply_func(
        self,
        dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # Define the coordinates for applying the correction
        xx, yy = np.meshgrid(np.arange(0, dem.shape[1]), np.arange(0, dem.shape[0]))

        return super()._apply_func(dem=dem, transform=transform, crs=crs, bias_vars={"xx": xx, "yy": yy}, **kwargs)

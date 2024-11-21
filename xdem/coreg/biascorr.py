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

"""Bias corrections (i.e., non-affine coregistration) classes."""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Literal, TypeVar

import geopandas as gpd
import geoutils as gu
import numpy as np
import rasterio as rio
import scipy

import xdem.spatialstats
from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import Coreg, fit_workflows
from xdem.fit import polynomial_2d

BiasCorrType = TypeVar("BiasCorrType", bound="BiasCorr")


class BiasCorr(Coreg):
    """
    Bias-correction (non-rigid alignment) simultaneously with any number and type of variables.

    Variables for bias-correction can include the elevation coordinates (deramping, directional biases), terrain
    attributes (terrain corrections), or any other user-input variable (quality metrics, land cover).

    The binning and/or fitting correction parameters are stored in the `self.meta["outputs"]["fitorbin"]`.
    """

    def __init__(
        self,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "fit",
        fit_func: (
            Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"]
        ) = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 10,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
        bias_var_names: Iterable[str] = None,
        subsample: float | int = 1.0,
    ):
        """
        Instantiate an N-dimensional bias correction using binning, fitting or both sequentially.

        All fit arguments apply to "fit" and "bin_and_fit", and bin arguments to "bin" and "bin_and_fit".

        :param fit_or_bin: Whether to fit or bin, or both. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins, or "bin_and_fit" to perform a fit on
            the binned statistics.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
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

        list_bias_var_names = list(bias_var_names) if bias_var_names is not None else None

        # Now we write the relevant attributes to the class metadata
        # For fitting
        if fit_or_bin == "fit":
            meta_fit = {"fit_func": fit_func, "fit_optimizer": fit_optimizer, "bias_var_names": list_bias_var_names}
            # Somehow mypy doesn't understand that fit_func and fit_optimizer can only be callables now,
            # even writing the above "if" in a more explicit "if; else" loop with new variables names and typing
            super().__init__(meta=meta_fit)  # type: ignore

        # For binning
        elif fit_or_bin == "bin":
            meta_bin = {
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
                "bin_apply_method": bin_apply_method,
                "bias_var_names": list_bias_var_names,
            }
            super().__init__(meta=meta_bin)  # type: ignore

        # For both
        else:
            meta_bin_and_fit = {
                "fit_func": fit_func,
                "fit_optimizer": fit_optimizer,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
                "bias_var_names": list_bias_var_names,
            }
            super().__init__(meta=meta_bin_and_fit)  # type: ignore

        # Add subsample attribute
        self._meta["inputs"]["fitorbin"]["fit_or_bin"] = fit_or_bin
        self._meta["inputs"]["random"]["subsample"] = subsample

        # Add number of dimensions attribute (length of bias_var_names, counted generically for iterator)
        self._meta["inputs"]["fitorbin"]["nd"] = sum(1 for _ in bias_var_names) if bias_var_names is not None else None

        # Update attributes
        self._is_affine = False
        self._needs_vars = True

    def _fit_rst_rst_and_rst_pts(  # type: ignore
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: None | dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:
        """Function for fitting raster-raster and raster-point for bias correction methods."""

        # Pre-process raster-point input
        sub_ref, sub_tba, sub_bias_vars = self._preprocess_rst_pts_subsample(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            aux_vars=bias_vars,
        )

        # Derive difference to get dh
        diff = sub_ref - sub_tba

        # Send to bin and fit
        self._bin_or_and_fit_nd(
            values=diff,
            bias_vars=sub_bias_vars,
            weights=weights,
            **kwargs,
        )

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called by other classes"""

        self._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _fit_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called by other classes"""

        self._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _apply_rst(  # type: ignore
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
        crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if bias_vars is None:
            raise ValueError("At least one `bias_var` should be passed to the `apply` function, got None.")

        # Check the bias_vars passed match the ones stored for this bias correction class
        if not sorted(bias_vars.keys()) == sorted(self._meta["inputs"]["fitorbin"]["bias_var_names"]):
            raise ValueError(
                "The keys of `bias_vars` do not match the `bias_var_names` defined during "
                "instantiation or fitting: {}.".format(self._meta["inputs"]["fitorbin"]["bias_var_names"])
            )

        # Apply function to get correction (including if binning was done before)
        if self.meta["inputs"]["fitorbin"]["fit_or_bin"] in ["fit", "bin_and_fit"]:
            corr = self._meta["inputs"]["fitorbin"]["fit_func"](
                tuple(bias_vars.values()), *self._meta["outputs"]["fitorbin"]["fit_params"]
            )

        # Apply binning to get correction
        else:
            if self._meta["inputs"]["fitorbin"]["bin_apply_method"] == "linear":
                # N-D interpolation of binning
                bin_interpolator = xdem.spatialstats.interp_nd_binning(
                    df=self._meta["outputs"]["fitorbin"]["bin_dataframe"],
                    list_var_names=list(bias_vars.keys()),
                    statistic=self._meta["inputs"]["fitorbin"]["bin_statistic"],
                )
                corr = bin_interpolator(tuple(var.flatten() for var in bias_vars.values()))
                first_var = list(bias_vars.keys())[0]
                corr = corr.reshape(np.shape(bias_vars[first_var]))

            else:
                # Get N-D binning statistic for each pixel of the new list of variables
                corr = xdem.spatialstats.get_perbin_nd_binning(
                    df=self._meta["outputs"]["fitorbin"]["bin_dataframe"],
                    list_var=list(bias_vars.values()),
                    list_var_names=list(bias_vars.keys()),
                    statistic=self._meta["inputs"]["fitorbin"]["bin_statistic"],
                )

        dem_corr = elev + corr

        return dem_corr, transform


class DirectionalBias(BiasCorr):
    """
    Bias correction for directional biases, for example along- or across-track of satellite angle.

    The binning and/or fitting correction parameters are stored in the `self.meta["outputs"]["fitorbin"]`.
    """

    def __init__(
        self,
        angle: float = 0,
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin_and_fit",
        fit_func: Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"] = "nfreq_sumsin",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 100,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
        subsample: float | int = 1.0,
    ):
        """
        Instantiate a directional bias correction.

        :param angle: Angle in which to perform the directional correction (degrees) with 0° corresponding to X axis
            direction and increasing clockwise.
        :param fit_or_bin: Whether to fit or bin, or both. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins, or "bin_and_fit" to perform a fit on
            the binned statistics.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        super().__init__(
            fit_or_bin, fit_func, fit_optimizer, bin_sizes, bin_statistic, bin_apply_method, ["angle"], subsample
        )
        self._meta["inputs"]["specific"]["angle"] = angle
        self._needs_vars = False

    def _fit_rst_rst(  # type: ignore
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        logging.info("Estimating rotated coordinates.")

        x, _ = gu.raster.get_xy_rotated(
            raster=gu.Raster.from_array(data=ref_elev, crs=crs, transform=transform, nodata=-9999),
            along_track_angle=self._meta["inputs"]["specific"]["angle"],
        )

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={"angle": x},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            **kwargs,
        )

    def _fit_rst_pts(  # type: ignore
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # Figure out which data is raster format to get gridded attributes
        rast_elev = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev

        logging.info("Estimating rotated coordinates.")

        x, _ = gu.raster.get_xy_rotated(
            raster=gu.Raster.from_array(data=rast_elev, crs=crs, transform=transform, nodata=-9999),
            along_track_angle=self._meta["inputs"]["specific"]["angle"],
        )

        # Parameters dependent on resolution cannot be derived from the rotated x coordinates, need to be passed below
        if "hop_length" not in kwargs:
            # The hop length will condition jump in function values, need to be larger than average resolution
            average_res = (transform[0] + abs(transform[4])) / 2
            kwargs.update({"hop_length": average_res})

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={"angle": x},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # Define the coordinates for applying the correction
        x, _ = gu.raster.get_xy_rotated(
            raster=gu.Raster.from_array(data=elev, crs=crs, transform=transform, nodata=-9999),
            along_track_angle=self._meta["inputs"]["specific"]["angle"],
        )

        return super()._apply_rst(elev=elev, transform=transform, crs=crs, bias_vars={"angle": x}, **kwargs)


class TerrainBias(BiasCorr):
    """
    Correct a bias according to terrain, such as elevation or curvature.

    With elevation: often useful for nadir image DEM correction, where the focal length is slightly miscalculated.
    With curvature: often useful for a difference of DEMs with different effective resolution.

    The binning and/or fitting correction parameters are stored in the `self.meta["outputs"]["fitorbin"]`.

    DISCLAIMER: An elevation correction may introduce error when correcting non-photogrammetric biases, as generally
    elevation biases are interlinked with curvature biases.
    See Gardelle et al. (2012) (Figure 2), http://dx.doi.org/10.3189/2012jog11j175, for curvature-related biases.
    """

    def __init__(
        self,
        terrain_attribute: str = "maximum_curvature",
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin",
        fit_func: (
            Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"]
        ) = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 100,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        bin_apply_method: Literal["linear"] | Literal["per_bin"] = "linear",
        subsample: float | int = 1.0,
    ):
        """
        Instantiate a terrain bias correction.

        :param terrain_attribute: Terrain attribute to use for correction.
        :param fit_or_bin: Whether to fit or bin, or both. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins, or "bin_and_fit" to perform a fit on
            the binned statistics.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        super().__init__(
            fit_or_bin,
            fit_func,
            fit_optimizer,
            bin_sizes,
            bin_statistic,
            bin_apply_method,
            [terrain_attribute],
            subsample,
        )
        # This is the same as bias_var_names, but let's leave the duplicate for clarity
        self._meta["inputs"]["specific"]["terrain_attribute"] = terrain_attribute
        self._needs_vars = False

    def _fit_rst_rst(  # type: ignore
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # If already passed by user, pass along
        if bias_vars is not None and self._meta["inputs"]["specific"]["terrain_attribute"] in bias_vars:
            attr = bias_vars[self._meta["inputs"]["specific"]["terrain_attribute"]]

        # If only declared during instantiation
        else:
            # Derive terrain attribute
            if self._meta["inputs"]["specific"]["terrain_attribute"] == "elevation":
                attr = ref_elev
            else:
                attr = xdem.terrain.get_terrain_attribute(
                    dem=ref_elev,
                    attribute=self._meta["inputs"]["specific"]["terrain_attribute"],
                    resolution=(transform[0], abs(transform[4])),
                )

        # Run the parent function
        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={self._meta["inputs"]["specific"]["terrain_attribute"]: attr},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            **kwargs,
        )

    def _fit_rst_pts(  # type: ignore
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # If already passed by user, pass along
        if bias_vars is not None and self._meta["inputs"]["specific"]["terrain_attribute"] in bias_vars:
            attr = bias_vars[self._meta["inputs"]["specific"]["terrain_attribute"]]

        # If only declared during instantiation
        else:
            # Figure out which data is raster format to get gridded attributes
            rast_elev = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev

            # Derive terrain attribute
            if self._meta["inputs"]["specific"]["terrain_attribute"] == "elevation":
                attr = rast_elev
            else:
                attr = xdem.terrain.get_terrain_attribute(
                    dem=rast_elev,
                    attribute=self._meta["inputs"]["specific"]["terrain_attribute"],
                    resolution=(transform[0], abs(transform[4])),
                )

        # Run the parent function
        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={self._meta["inputs"]["specific"]["terrain_attribute"]: attr},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if bias_vars is None:
            # Derive terrain attribute
            if self._meta["inputs"]["specific"]["terrain_attribute"] == "elevation":
                attr = elev
            else:
                attr = xdem.terrain.get_terrain_attribute(
                    dem=elev,
                    attribute=self._meta["inputs"]["specific"]["terrain_attribute"],
                    resolution=(transform[0], abs(transform[4])),
                )
            bias_vars = {self._meta["inputs"]["specific"]["terrain_attribute"]: attr}

        return super()._apply_rst(elev=elev, transform=transform, crs=crs, bias_vars=bias_vars, **kwargs)


class Deramp(BiasCorr):
    """
    Correct for a 2D polynomial along X/Y coordinates, for example from residual camera model deformations
    (dome-like errors) or tilts (rotational errors).

    The correction parameters are stored in the `self.meta["outputs"]["fitorbin"]` key "fit_params", that can be passed
    to the associated function in key "fit_func".
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
        subsample: float | int = 5e5,
    ):
        """
        Instantiate a directional bias correction.

        :param poly_order: Order of the 2D polynomial to fit.
        :param fit_or_bin: Whether to fit or bin, or both. Use "fit" to correct by optimizing a function or
            "bin" to correct with a statistic of central tendency in defined bins, or "bin_and_fit" to perform a fit on
            the binned statistics.
        :param fit_func: Function to fit to the bias with variables later passed in .fit().
        :param fit_optimizer: Optimizer to minimize the function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        super().__init__(
            fit_or_bin,
            fit_func,
            fit_optimizer,
            bin_sizes,
            bin_statistic,
            bin_apply_method,
            ["xx", "yy"],
            subsample,
        )
        self._meta["inputs"]["specific"]["poly_order"] = poly_order
        self._needs_vars = False

    def _fit_rst_rst(  # type: ignore
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] | None = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # The number of parameters in the first guess defines the polynomial order when calling np.polyval2d
        p0 = np.ones(shape=((self._meta["inputs"]["specific"]["poly_order"] + 1) ** 2))

        # Coordinates (we don't need the actual ones, just array coordinates)
        xx, yy = np.meshgrid(np.arange(0, ref_elev.shape[1]), np.arange(0, ref_elev.shape[0]))

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={"xx": xx, "yy": yy},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            p0=p0,
            **kwargs,
        )

    def _fit_rst_pts(  # type: ignore
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str,
        bias_vars: dict[str, NDArrayf] | None = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # Figure out which data is raster format to get gridded attributes
        rast_elev = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev

        # The number of parameters in the first guess defines the polynomial order when calling np.polyval2d
        p0 = np.ones(shape=((self._meta["inputs"]["specific"]["poly_order"] + 1) ** 2))

        # Coordinates (we don't need the actual ones, just array coordinates)
        xx, yy = np.meshgrid(np.arange(0, rast_elev.shape[1]), np.arange(0, rast_elev.shape[0]))

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={"xx": xx, "yy": yy},
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            p0=p0,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: None | dict[str, NDArrayf] = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # Define the coordinates for applying the correction
        xx, yy = np.meshgrid(np.arange(0, elev.shape[1]), np.arange(0, elev.shape[0]))

        return super()._apply_rst(elev=elev, transform=transform, crs=crs, bias_vars={"xx": xx, "yy": yy}, **kwargs)

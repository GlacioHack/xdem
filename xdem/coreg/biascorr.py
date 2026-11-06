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
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, TypeVar

import geoutils as gu
import numpy as np
import rasterio as rio
from geoutils._dispatch import _get_raster_interface, _is_raster, is_dask_array
from geoutils.multiproc.chunked import GeoGrid
from geoutils.raster import Raster
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.referencing import _coords
from geoutils.raster.transformation import _crop_window
from geoutils.raster.xr_accessor import DataArrayRasterAccessor
from geoutils.sampling.support import _raster_from_input

import xdem.fit
from xdem._typing import NDArrayf
from xdem.coreg.base import (
    Coreg,
    InRandomDict,
    _sample_coreg_inputs,
    fit_workflows,
)
from xdem.fit import design_matrix_polynomial_2d, polynomial_2d

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.raster.base import RasterBase, RasterLike

BiasCorrType = TypeVar("BiasCorrType", bound="BiasCorr")


############################################################
# Biascorr for raster chunks
############################################################


def _apply_biascorr_array_block(
    values: NDArrayf,
    *predictors: NDArrayf,
    coreg: Coreg,
    predictor_names: tuple[str, ...],
    predictor_nodata: tuple[float | int | None, ...],
    grid: GeoGrid,
    nodata: float | int | None,
    options: dict[str, Any],
    block_info: Any,
) -> NDArrayf:
    """Apply a fitted bias correction to one raster block using its full raster coordinates."""

    # Locate this block on the original raster grid
    (row, row_stop), (column, col_stop) = block_info[0]["array-location"]
    _, tile_transform = _crop_window(grid, (column, row, col_stop, row_stop), distance_unit="pixel")
    tile = DataArrayRasterAccessor.from_array(values, tile_transform, grid.crs, nodata=nodata)
    if not np.isfinite(tile.data).any():
        return np.full(values.shape, np.nan, dtype="float32")

    # Apply
    bias_vars = {
        name: DataArrayRasterAccessor.from_array(predictor, tile_transform, grid.crs, nodata=missing)
        for name, predictor, missing in zip(predictor_names, predictors, predictor_nodata)
    }
    result = coreg.apply(
        tile,
        bias_vars=bias_vars or None,
        raster_grid=grid,
        pixel_offset=(row, column),
        **options,
    )
    return result.data


def _apply_biascorr_raster_block(
    tile: RasterLike,
    coreg: Coreg,
    predictors: dict[str, Any],
    source: RasterBase,
    options: dict[str, Any],
) -> Raster:
    """Read bias predictor variables tiles and apply a fitted correction in a worker."""

    # Locate the tile and read predictors from the same pixels
    tile = _get_raster_interface(tile)
    row, column = source.xy2ij(*tile.ij2xy(0, 0), op=np.float64)
    window = rio.windows.Window(int(np.rint(column).item()), int(np.rint(row).item()), tile.width, tile.height)
    (row, row_stop), (column, col_stop) = window.toranges()
    bias_vars = {name: predictor.icrop((column, row, col_stop, row_stop)) for name, predictor in predictors.items()}

    # Apply the bias correction, including its output dtype and mask
    if not np.isfinite(tile.data).any():
        result = tile.copy(new_array=tile.data.astype("float32"))
    else:
        result = coreg.apply(
            tile,
            bias_vars=bias_vars or None,
            raster_grid=GeoGrid(source.transform, source.shape, source.crs),
            pixel_offset=(row, column),
            **options,
        )
    result = _get_raster_interface(result)
    return Raster.from_array(
        result.data, tile.transform, tile.crs, nodata=tile.nodata, area_or_point=tile.area_or_point
    )


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
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["ols"] | None = None,
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
        :param fit_optimizer: Optimizer to minimize the function. If None, use OLS for a built-in linear model, the
            optimizer associated with a named fit workflow, or :func:`scipy.optimize.curve_fit` otherwise.
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
            is_ols = isinstance(fit_optimizer, str) and fit_optimizer == "ols"
            if fit_optimizer is not None and not is_ols and not callable(fit_optimizer):
                raise TypeError(
                    "Argument `fit_optimizer` must be a function (callable), 'ols' or None, "
                    "got {}.".format(type(fit_optimizer))
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
            meta_fit = {
                "fit_or_bin": fit_or_bin,
                "fit_func": fit_func,
                "fit_optimizer": fit_optimizer,
                "bias_var_names": list_bias_var_names,
            }
            # Somehow mypy doesn't understand that fit_func and fit_optimizer can only be callables now,
            # even writing the above "if" in a more explicit "if; else" loop with new variables names and typing
            super().__init__(meta=meta_fit)  # type: ignore

        # For binning
        elif fit_or_bin == "bin":
            meta_bin = {
                "fit_or_bin": fit_or_bin,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
                "bin_apply_method": bin_apply_method,
                "bias_var_names": list_bias_var_names,
            }
            super().__init__(meta=meta_bin)  # type: ignore

        # For both
        else:
            meta_bin_and_fit = {
                "fit_or_bin": fit_or_bin,
                "fit_func": fit_func,
                "fit_optimizer": fit_optimizer,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
                "bias_var_names": list_bias_var_names,
            }
            super().__init__(meta=meta_bin_and_fit)  # type: ignore

        # Add subsample attribute
        self._meta["inputs"]["random"]["subsample"] = subsample

        # Add number of dimensions attribute (length of bias_var_names, counted generically for iterator)
        self._meta["inputs"]["fitorbin"]["nd"] = sum(1 for _ in bias_var_names) if bias_var_names is not None else None

        # Update attributes
        self._is_affine = False
        self._needs_vars = True

    def _fit_rst_rst_and_rst_pts(  # type: ignore
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        bias_vars: dict[str, Any] | Callable[[NDArrayf, NDArrayf], dict[str, NDArrayf]] | None = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:
        """Function for fitting raster-raster and raster-point for bias correction methods."""

        # Get multiprocessing arguments, if passed
        mp_config = kwargs.pop("mp_config", None)
        kwargs.pop("temporary_files", None)

        # Get random parameters
        params_random: InRandomDict = self._meta["inputs"]["random"]

        # Pre-process raster-point input
        auxiliary_at = {}
        if bias_vars is not None and not callable(bias_vars):
            for name, value in bias_vars.items():
                raster_array = (isinstance(value, np.ndarray) or is_dask_array(value)) and value.ndim >= 2
                auxiliary_at[name] = "tba" if raster_array and not _is_raster(ref_elev) else "ref"
        sub_ref, sub_tba, sub_bias_vars = _sample_coreg_inputs(
            mp_config=mp_config,
            subsample=params_random["subsample"],
            random_state=params_random["random_state"],
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            # Compare elevations and predictors at identical locations for bias correction
            sampling_strategy="same_xy",
            aux_vars=None if callable(bias_vars) else bias_vars,
            aux_tied_to=auxiliary_at,
        )

        # Calculate coordinate predictors only at the sampled points
        if callable(bias_vars):
            sub_bias_vars = bias_vars(sub_ref[0], sub_ref[1])

        # Write final subsample to class
        self._meta["outputs"]["random"] = {"subsample_final": sub_ref.shape[1]}

        # Derive difference of Z axis to get dh
        diff = sub_ref[2, :] - sub_tba[2, :]

        # Send to bin and fit
        self._bin_or_and_fit_nd(
            values=diff,
            bias_vars=sub_bias_vars,
            weights=weights,
            **kwargs,
        )

    def _fit_rst_rst(
        self,
        ref_elev: RasterBase,
        tba_elev: RasterBase,
        inlier_mask: Any,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called by other classes"""

        self._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _fit_rst_pts(
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called by other classes"""

        self._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: RasterBase,
        bias_vars: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike, rio.transform.Affine]:
        """
        Apply a fitted bias correction to a raster or its blocks.

        _apply_biascorr_array_block() and _apply_biascorr_raster_block() apply the same correction
        to each Dask block or multiprocessing tile.
        """

        # Inspect predictor storage before reading raster values
        supplied_predictors = bias_vars or {}
        predictor_interfaces = {name: _get_raster_interface(value) for name, value in supplied_predictors.items()}
        lazy_predictors = any(
            is_dask_array(value)
            or predictor_interfaces[name] is not None
            and predictor_interfaces[name]._chunks is not None
            for name, value in supplied_predictors.items()
        )
        if elev._chunks is not None or lazy_predictors or mp_config is not None:
            from geoutils.uncertainty.predictors import (
                _raster_chunk_predictors,
                _raster_chunk_source,
                _raster_dask_chunks,
            )

            # 1/ Check predictor grids without loading raster files
            predictors = {name: _raster_from_input(value, elev, name) for name, value in (bias_vars or {}).items()}
            if any(not elev.georeferenced_grid_equal(predictor) for predictor in predictors.values()):
                raise ValueError("Bias predictor rasters must have the same grid and CRS as the elevation raster.")
            options = {name: value for name, value in kwargs.items() if name not in {"transform", "crs"}}
            options["resample"] = False

            # 2/ If using multiprocessing, we write independent raster tiles
            if mp_config is not None:
                from geoutils.multiproc import map_overlap

                result = map_overlap(
                    _apply_biascorr_raster_block, elev, mp_config, self, predictors, elev, options, depth=0
                )
                if elev._is_xr:
                    return elev._cast_raster_output(result), elev.transform
                return elev.__class__(result), elev.transform

            # 3/ If using Dask, we apply the correction lazily
            from xdem._misc import import_optional

            import_optional("dask")
            import dask.array as da

            chunk_source = elev
            if elev._chunks is None:
                for predictor in predictors.values():
                    if _is_raster(predictor) and predictor._chunks is not None:
                        chunk_source = predictor
                        break
            chunks = _raster_dask_chunks(chunk_source, predictors, None)
            tile_size = max(chunks[0]), max(chunks[1])
            source = da.asarray(_raster_chunk_source(elev, tile_size)).reshape(elev.shape).rechunk(chunks)
            data = _raster_chunk_predictors(elev, predictors, tile_size)
            predictor_blocks = [da.asarray(value).rechunk(chunks) for value in data.values()]
            values = da.map_blocks(
                _apply_biascorr_array_block,
                source,
                *predictor_blocks,
                coreg=self,
                predictor_names=tuple(predictors),
                predictor_nodata=tuple(getattr(value, "nodata", None) for value in predictors.values()),
                grid=GeoGrid(elev.transform, elev.shape, elev.crs),
                nodata=elev.nodata,
                options=options,
                dtype="float32",
                meta=np.empty((0, 0), dtype="float32"),
            )
            result = DataArrayRasterAccessor.from_array(
                values, elev.transform, elev.crs, nodata=elev.nodata, area_or_point=elev.area_or_point
            )
            return result, elev.transform

        # Read arrays only for the correction in memory
        values, invalid = get_array_and_mask(elev)
        if np.all(invalid):
            raise ValueError("'dem' had only NaNs")
        # Preserve raster rows and columns, including one-pixel blocks
        values = values.reshape(elev.shape)
        if bias_vars is not None:
            prepared_predictors = {}
            for name, value in bias_vars.items():
                predictor = value if predictor_interfaces[name] is None else predictor_interfaces[name]
                prepared_predictors[name] = get_array_and_mask(predictor)[0].reshape(predictor.shape[-2:])
            bias_vars = prepared_predictors

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
                bin_interpolator = xdem.fit.interp_binning(
                    table=self._meta["outputs"]["fitorbin"]["bin_dataframe"],
                    value_name="bias",
                    statistic=self._meta["inputs"]["fitorbin"]["bin_statistic"],
                    min_count=kwargs.get("min_count", 0),
                )
                corr = bin_interpolator(bias_vars)
                first_var = list(bias_vars.keys())[0]
                corr = corr.reshape(np.shape(bias_vars[first_var]))

            else:
                # Get N-D binning statistic for each pixel of the new list of variables
                corr = xdem.fit.get_perbin_binning(
                    table=self._meta["outputs"]["fitorbin"]["bin_dataframe"],
                    value_name="bias",
                    predictors=bias_vars,
                    statistic=self._meta["inputs"]["fitorbin"]["bin_statistic"],
                )

        dem_corr = values + corr

        return dem_corr, elev.transform


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
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["ols"] | None = None,
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
        :param fit_optimizer: Optimizer to minimize the function. If None, use the default for the selected fit model.
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

    def _fit_rst_rst_and_rst_pts(  # type: ignore
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        bias_vars: dict[str, Any] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # Figure out which data is raster format to get gridded attributes
        rast_elev = ref_elev if _is_raster(ref_elev) else tba_elev

        logging.info("Estimating rotated coordinates.")

        # Locate the raster corners without building coordinate grids
        # Use the same zero origin as get_xy_rotated() when the correction is applied
        def rotated_coordinates(x: NDArrayf, y: NDArrayf) -> dict[str, NDArrayf]:
            """Calculate along-track distances at the sampled raster or point coordinates."""

            rotated_x, _ = gu.raster.get_xy_rotated(
                rast_elev, self._meta["inputs"]["specific"]["angle"], coordinates=(x, y), force_offset="center"
            )
            return {"angle": rotated_x}

        # Parameters dependent on resolution cannot be derived from the rotated x coordinates, need to be passed below
        if _is_raster(ref_elev) != _is_raster(tba_elev) and "hop_length" not in kwargs:
            # The hop length will condition jump in function values, need to be larger than average resolution
            average_res = sum(rast_elev.res) / 2
            kwargs.update({"hop_length": average_res})

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars=rotated_coordinates,
            weights=weights,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: RasterBase,
        bias_vars: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike, rio.transform.Affine]:

        # Calculate coordinate predictors on each block for Dask or multiprocessing
        if elev._chunks is not None or mp_config is not None:
            return super()._apply_rst(elev, bias_vars=bias_vars, mp_config=mp_config, **kwargs)

        # Define the coordinates for applying the correction
        grid = GeoGrid(elev.transform, elev.shape, elev.crs)
        coordinates = None
        if "raster_grid" in kwargs:
            coordinates = _coords(elev.transform, elev.shape, force_offset="ll")
            grid = kwargs["raster_grid"]
        x, _ = gu.raster.get_xy_rotated(grid, self._meta["inputs"]["specific"]["angle"], coordinates=coordinates)

        return super()._apply_rst(elev=elev, bias_vars={"angle": x}, mp_config=mp_config, **kwargs)


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
        terrain_attribute: str = "max_curvature",
        fit_or_bin: Literal["bin_and_fit"] | Literal["fit"] | Literal["bin"] = "bin",
        fit_func: (
            Callable[..., NDArrayf] | Literal["norder_polynomial"] | Literal["nfreq_sumsin"]
        ) = "norder_polynomial",
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["ols"] | None = None,
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
        :param fit_optimizer: Optimizer to minimize the function. If None, use the default for the selected fit model.
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

    def _fit_rst_rst_and_rst_pts(  # type: ignore
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        bias_vars: dict[str, Any] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # If already passed by user, pass along
        if bias_vars is not None and self._meta["inputs"]["specific"]["terrain_attribute"] in bias_vars:
            attr = bias_vars[self._meta["inputs"]["specific"]["terrain_attribute"]]

        # If only declared during instantiation
        else:
            # Figure out which data is raster format to get gridded attributes
            rast_elev = ref_elev if _is_raster(ref_elev) else tba_elev

            # Derive terrain attribute
            if self._meta["inputs"]["specific"]["terrain_attribute"] == "elevation":
                attr = rast_elev
            else:
                config = kwargs.get("mp_config")
                if config is not None:
                    config = kwargs["temporary_files"].enter_context(config.temporary())
                attr = xdem.terrain.get_terrain_attribute(
                    mp_config=config,
                    dem=rast_elev,
                    attribute=self._meta["inputs"]["specific"]["terrain_attribute"],
                    resolution=rast_elev.res,
                )

        # Run the parent function
        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars={self._meta["inputs"]["specific"]["terrain_attribute"]: attr},
            weights=weights,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: RasterBase,
        bias_vars: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike, rio.transform.Affine]:
        """Calculate terrain predictors before applying the correction."""

        with ExitStack() as temporary_files:
            if bias_vars is None:
                # Derive terrain attribute
                attribute = self._meta["inputs"]["specific"]["terrain_attribute"]
                if attribute == "elevation":
                    predictor = elev
                else:
                    # Terrain gradients need neighboring pixels across tile boundaries
                    config = mp_config
                    if config is not None:
                        config = temporary_files.enter_context(config.temporary())
                    predictor = xdem.terrain.get_terrain_attribute(
                        dem=elev,
                        attribute=attribute,
                        resolution=(elev.transform[0], abs(elev.transform[4])),
                        mp_config=config,
                    )
                bias_vars = {attribute: predictor}
            return super()._apply_rst(elev, bias_vars=bias_vars, mp_config=mp_config, **kwargs)


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
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["ols"] | None = None,
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
        :param fit_optimizer: Optimizer to minimize the function. If None, use OLS for the standard polynomial model.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param bin_apply_method: Method to correct with the binned statistics, either "linear" to interpolate linearly
            between bins, or "per_bin" to apply the statistic for each bin.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        # Use the linear solver only for the standard polynomial model
        self._fit_linear = fit_func is polynomial_2d

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
        self._design_matrix_func = design_matrix_polynomial_2d(poly_order)
        self._needs_vars = False

    def _fit_rst_rst_and_rst_pts(  # type: ignore
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        bias_vars: dict[str, Any] | None = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:

        # Figure out which data is raster format to get gridded attributes
        rast_elev = ref_elev if _is_raster(ref_elev) else tba_elev

        # The number of parameters in the first guess defines the polynomial order when calling np.polyval2d
        p0 = np.ones(shape=((self._meta["inputs"]["specific"]["poly_order"] + 1) ** 2))

        # Coordinates (we don't need the actual ones, just array coordinates)
        def pixel_coordinates(x: NDArrayf, y: NDArrayf) -> dict[str, NDArrayf]:
            """Calculate polynomial predictors at the sampled raster or point coordinates."""

            rows, columns = rast_elev.xy2ij(x, y, op=np.float64)
            return {"xx": columns, "yy": rows}

        super()._fit_rst_rst_and_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            bias_vars=pixel_coordinates,
            weights=weights,
            p0=p0,
            **kwargs,
        )

    def _apply_rst(
        self,
        elev: RasterBase,
        bias_vars: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike, rio.transform.Affine]:

        # Calculate coordinate predictors on each block for Dask or multiprocessing
        if elev._chunks is not None or mp_config is not None:
            return super()._apply_rst(elev, bias_vars=bias_vars, mp_config=mp_config, **kwargs)

        # Define the coordinates for applying the correction
        row, column = kwargs.get("pixel_offset", (0, 0))
        xx, yy = np.meshgrid(np.arange(column, column + elev.shape[1]), np.arange(row, row + elev.shape[0]))

        return super()._apply_rst(elev=elev, bias_vars={"xx": xx, "yy": yy}, mp_config=mp_config, **kwargs)

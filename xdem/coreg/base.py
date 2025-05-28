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

"""Base coregistration classes to define generic methods and pre/post-processing of input data."""

from __future__ import annotations

import copy
import inspect
import logging
import warnings
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    TypedDict,
    TypeVar,
    overload,
)

import affine
import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from geoutils._typing import Number
from geoutils.interface.gridding import _grid_pointcloud
from geoutils.interface.interpolate import _interp_points
from geoutils.raster import Mask, RasterType, raster
from geoutils.raster._geotransformations import _resampling_method_from_str
from geoutils.raster.array import get_array_and_mask
from geoutils.raster.georeferencing import _cast_pixel_interpretation, _coords
from geoutils.raster.geotransformations import _translate
from geoutils.stats import nmad

from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.fit import (
    polynomial_1d,
    robust_nfreq_sumsin_fit,
    robust_norder_polynomial_fit,
    sumsin_1d,
)
from xdem.spatialstats import nd_binning

try:
    import pytransform3d.rotations
    import pytransform3d.transformations
    from pytransform3d.transform_manager import TransformManager

    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False


# Map each workflow name to a function and optimizer
fit_workflows = {
    "norder_polynomial": {"func": polynomial_1d, "optimizer": robust_norder_polynomial_fit},
    "nfreq_sumsin": {"func": sumsin_1d, "optimizer": robust_nfreq_sumsin_fit},
}

# Map each key name to a descriptor string
dict_key_to_str = {
    "subsample": "Subsample size requested",
    "random_state": "Random generator",
    "subsample_final": "Subsample size drawn from valid values",
    "fit_or_bin": "Fit, bin or bin+fit",
    "fit_func": "Function to fit",
    "fit_optimizer": "Optimizer for fitting",
    "fit_minimizer": "Minimizer of method",
    "fit_loss_func": "Loss function of method",
    "bin_statistic": "Binning statistic",
    "bin_sizes": "Bin sizes or edges",
    "bin_apply_method": "Bin apply method",
    "bias_var_names": "Names of bias variables",
    "nd": "Number of dimensions of binning and fitting",
    "fit_params": "Optimized function parameters",
    "fit_perr": "Error on optimized function parameters",
    "bin_dataframe": "Binning output dataframe",
    "max_iterations": "Maximum number of iterations",
    "tolerance": "Tolerance to reach (pixel size)",
    "last_iteration": "Iteration at which algorithm stopped",
    "all_tolerances": "Tolerances at each iteration",
    "terrain_attribute": "Terrain attribute used for correction",
    "angle": "Angle of directional correction",
    "poly_order": "Polynomial order",
    "best_poly_order": "Best polynomial order",
    "best_nb_sin_freq": "Best number of sinusoid frequencies",
    "vshift_reduc_func": "Reduction function used to remove vertical shift",
    "apply_vshift": "Vertical shift activated",
    "centroid": "Centroid found for affine rotation",
    "shift_x": "Eastward shift estimated (georeferenced unit)",
    "shift_y": "Northward shift estimated (georeferenced unit)",
    "shift_z": "Vertical shift estimated (elevation unit)",
    "matrix": "Affine transformation matrix estimated",
    "only_translation": "Only translations are considered",
    "standardize": "Input data was standardized",
    "icp_method": "Type of ICP method",
    "icp_picky": "Picky closest pair selection",
    "cpd_weight": "Weight of CPD outlier removal",
}
#####################################
# Generic functions for preprocessing
###########################################


def _calculate_ddem_stats(
    ddem: NDArrayf | MArrayf,
    inlier_mask: NDArrayb | None = None,
    stats_list: tuple[Callable[[NDArrayf], Number], ...] | None = None,
    stats_labels: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """
    Calculate standard statistics of ddem, e.g., to be used to compare before/after coregistration.
    Default statistics are: count, mean, median, NMAD and std.

    :param ddem: The DEM difference to be analyzed.
    :param inlier_mask: 2D boolean array of areas to include in the analysis (inliers=True).
    :param stats_list: Statistics to compute on the DEM difference.
    :param stats_labels: Labels of the statistics to compute (same length as stats_list).

    Returns: a dictionary containing the statistics
    """
    # Default stats - Cannot be put in default args due to circular import with gu.stats.nmad.
    if (stats_list is None) or (stats_labels is None):
        stats_list = (np.size, np.mean, np.median, nmad, np.std)
        stats_labels = ("count", "mean", "median", "nmad", "std")

    # Check that stats_list and stats_labels are correct
    if len(stats_list) != len(stats_labels):
        raise ValueError("Number of items in `stats_list` and `stats_labels` should be identical.")
    for stat, label in zip(stats_list, stats_labels):
        if not callable(stat):
            raise ValueError(f"Item {stat} in `stats_list` should be a callable/function.")
        if not isinstance(label, str):
            raise ValueError(f"Item {label} in `stats_labels` should be a string.")

    # Get the mask of valid and inliers pixels
    nan_mask = ~np.isfinite(ddem)
    if inlier_mask is None:
        inlier_mask = np.ones(ddem.shape, dtype="bool")
    valid_ddem = ddem[~nan_mask & inlier_mask]

    # Calculate stats
    stats = {}
    for stat, label in zip(stats_list, stats_labels):
        stats[label] = stat(valid_ddem)

    return stats


def _preprocess_coreg_fit_raster_raster(
    reference_dem: NDArrayf | MArrayf | RasterType,
    dem_to_be_aligned: NDArrayf | MArrayf | RasterType,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
) -> tuple[NDArrayf, NDArrayf, NDArrayb, affine.Affine, rio.crs.CRS, Literal["Area", "Point"] | None]:
    """Pre-processing and checks of fit() for two raster input."""

    # Validate that both inputs are valid array-like (or Raster) types.
    if not all(isinstance(dem, (np.ndarray, gu.Raster)) for dem in (reference_dem, dem_to_be_aligned)):
        raise ValueError(
            "Both DEMs need to be array-like (implement a numpy array interface)."
            f"'reference_dem': {reference_dem}, 'dem_to_be_aligned': {dem_to_be_aligned}"
        )

    # If both DEMs are Rasters, validate that 'dem_to_be_aligned' is in the right grid. Then extract its data.
    if isinstance(dem_to_be_aligned, gu.Raster) and isinstance(reference_dem, gu.Raster):
        dem_to_be_aligned = dem_to_be_aligned.reproject(reference_dem, silent=True)

    # If both inputs are raster, cast their pixel interpretation and override any individual interpretation
    indiv_check = True
    new_aop = None
    if isinstance(reference_dem, gu.Raster) and isinstance(dem_to_be_aligned, gu.Raster):
        # Casts pixel interpretation, raises a warning if they differ (can be silenced with global config)
        new_aop = _cast_pixel_interpretation(reference_dem.area_or_point, dem_to_be_aligned.area_or_point)
        if area_or_point is not None:
            warnings.warn("Pixel interpretation cast from the two input rasters overrides the given 'area_or_point'.")
        indiv_check = False

    # If any input is a Raster, use its transform if 'transform is None'.
    # If 'transform' was given and any input is a Raster, trigger a warning.
    # Finally, extract only the data of the raster.
    new_transform = None
    new_crs = None
    for name, dem in [("reference_dem", reference_dem), ("dem_to_be_aligned", dem_to_be_aligned)]:
        if isinstance(dem, gu.Raster):
            # If a raster was passed, override the transform, reference raster has priority to set new_transform.
            if transform is None:
                new_transform = dem.transform
            elif transform is not None and new_transform is None:
                new_transform = dem.transform
                warnings.warn(f"'{name}' of type {type(dem)} overrides the given 'transform'")
            # Same for crs
            if crs is None:
                new_crs = dem.crs
            elif crs is not None and new_crs is None:
                new_crs = dem.crs
                warnings.warn(f"'{name}' of type {type(dem)} overrides the given 'crs'")
            # Same for pixel interpretation, only if both inputs aren't rasters (which requires casting, see above)
            if indiv_check:
                if area_or_point is None:
                    new_aop = dem.area_or_point
                elif crs is not None and new_aop is None:
                    new_aop = dem.area_or_point
                    warnings.warn(f"'{name}' of type {type(dem)} overrides the given 'area_or_point'")

    # Override transform, CRS and pixel interpretation
    if new_transform is not None:
        transform = new_transform
    if new_crs is not None:
        crs = new_crs
    if new_aop is not None:
        area_or_point = new_aop

    if transform is None:
        raise ValueError("'transform' must be given if both DEMs are array-like.")

    if crs is None:
        raise ValueError("'crs' must be given if both DEMs are array-like.")

    # Get a NaN array covering nodatas from the raster, masked array or integer-type array
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        ref_dem, ref_mask = get_array_and_mask(reference_dem, copy=True)
        tba_dem, tba_mask = get_array_and_mask(dem_to_be_aligned, copy=True)

    # Make sure that the mask has an expected format.
    if inlier_mask is not None:
        if isinstance(inlier_mask, Mask):
            inlier_mask = inlier_mask.data.filled(False).squeeze()
        else:
            inlier_mask = np.asarray(inlier_mask).squeeze()
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

        if np.all(~inlier_mask):
            raise ValueError("'inlier_mask' had no inliers.")
    else:
        inlier_mask = np.ones(np.shape(ref_dem), dtype=bool)

    if np.all(ref_mask):
        raise ValueError("'reference_dem' had only NaNs")
    if np.all(tba_mask):
        raise ValueError("'dem_to_be_aligned' had only NaNs")

    # Isolate all invalid values
    invalid_mask = np.logical_or.reduce((~inlier_mask, ref_mask, tba_mask))

    if np.all(invalid_mask):
        raise ValueError("All values of the inlier mask are NaNs in either 'reference_dem' or 'dem_to_be_aligned'.")

    return ref_dem, tba_dem, inlier_mask, transform, crs, area_or_point


def _preprocess_coreg_fit_raster_point(
    raster_elev: NDArrayf | MArrayf | RasterType,
    point_elev: gpd.GeoDataFrame,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
) -> tuple[NDArrayf, gpd.GeoDataFrame, NDArrayb, affine.Affine, rio.crs.CRS, Literal["Area", "Point"] | None]:
    """Pre-processing and checks of fit for raster-point input."""

    # TODO: Convert to point cloud once class is done
    # TODO: Raise warnings consistently with raster-raster function, see Amelie's Dask PR? #525
    if isinstance(raster_elev, gu.Raster):
        rst_elev = raster_elev.data
        crs = raster_elev.crs
        transform = raster_elev.transform
        area_or_point = raster_elev.area_or_point
    else:
        rst_elev = raster_elev
        crs = crs
        transform = transform
        area_or_point = area_or_point

    if transform is None:
        raise ValueError("'transform' must be given if both DEMs are array-like.")

    if crs is None:
        raise ValueError("'crs' must be given if both DEMs are array-like.")

    # Make sure that the mask has an expected format.
    if inlier_mask is not None:
        if isinstance(inlier_mask, Mask):
            inlier_mask = inlier_mask.data.filled(False).squeeze()
        else:
            inlier_mask = np.asarray(inlier_mask).squeeze()
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

        if np.all(~inlier_mask):
            raise ValueError("'inlier_mask' had no inliers.")
    else:
        inlier_mask = np.ones(np.shape(rst_elev), dtype=bool)

    # TODO: Convert to point cloud?
    # Convert geodataframe to vector
    point_elev = point_elev.to_crs(crs=crs)

    return rst_elev, point_elev, inlier_mask, transform, crs, area_or_point


def _preprocess_coreg_fit_point_point(
    reference_elev: gpd.GeoDataFrame, to_be_aligned_elev: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Pre-processing and checks of fit for point-point input."""

    ref_elev = reference_elev
    tba_elev = to_be_aligned_elev.to_crs(crs=reference_elev.crs)

    return ref_elev, tba_elev


def _preprocess_coreg_fit(
    reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
) -> tuple[
    NDArrayf | gpd.GeoDataFrame,
    NDArrayf | gpd.GeoDataFrame,
    NDArrayb | None,
    affine.Affine | None,
    rio.crs.CRS | None,
    Literal["Area", "Point"] | None,
]:
    """Pre-processing and checks of fit for any input."""

    if not all(
        isinstance(elev, (np.ndarray, gu.Raster, gpd.GeoDataFrame)) for elev in (reference_elev, to_be_aligned_elev)
    ):
        raise ValueError("Input elevation data should be a raster, an array or a geodataframe.")

    # If both inputs are raster or arrays, reprojection on the same grid is needed for raster-raster methods
    if all(isinstance(elev, (np.ndarray, gu.Raster)) for elev in (reference_elev, to_be_aligned_elev)):
        ref_elev, tba_elev, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit_raster_raster(
            reference_dem=reference_elev,
            dem_to_be_aligned=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

    # If one input is raster, and the other is point, we reproject the point data to the same CRS and extract arrays
    elif any(isinstance(dem, (np.ndarray, gu.Raster)) for dem in (reference_elev, to_be_aligned_elev)):
        if isinstance(reference_elev, (np.ndarray, gu.Raster)):
            raster_elev = reference_elev
            point_elev = to_be_aligned_elev
            ref = "raster"
        else:
            raster_elev = to_be_aligned_elev
            point_elev = reference_elev
            ref = "point"

        raster_elev, point_elev, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit_raster_point(
            raster_elev=raster_elev,
            point_elev=point_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        if ref == "raster":
            ref_elev = raster_elev
            tba_elev = point_elev
        else:
            ref_elev = point_elev
            tba_elev = raster_elev

    # If both inputs are points, simply reproject to the same CRS
    else:
        ref_elev, tba_elev = _preprocess_coreg_fit_point_point(
            reference_elev=reference_elev, to_be_aligned_elev=to_be_aligned_elev
        )

    return ref_elev, tba_elev, inlier_mask, transform, crs, area_or_point


def _preprocess_coreg_apply(
    elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine, rio.crs.CRS]:
    """Pre-processing and checks of apply for any input."""

    if not isinstance(elev, (np.ndarray, gu.Raster, gpd.GeoDataFrame)):
        raise ValueError("Input elevation data should be a raster, an array or a geodataframe.")

    # If input is geodataframe
    if isinstance(elev, gpd.GeoDataFrame):
        elev_out = elev
        new_transform = None
        new_crs = None

    # If input is a raster or array
    else:
        # If input is raster
        if isinstance(elev, gu.Raster):
            if transform is not None:
                warnings.warn(f"DEM of type {type(elev)} overrides the given 'transform'")
            if crs is not None:
                warnings.warn(f"DEM of type {type(elev)} overrides the given 'crs'")
            new_transform = elev.transform
            new_crs = elev.crs

        # If input is an array
        else:
            if transform is None:
                raise ValueError("'transform' must be given if DEM is array-like.")
            if crs is None:
                raise ValueError("'crs' must be given if DEM is array-like.")
            new_transform = transform
            new_crs = crs

        # The array to provide the functions will be an ndarray with NaNs for masked out areas.
        elev_out, elev_mask = get_array_and_mask(elev)

        if np.all(elev_mask):
            raise ValueError("'dem' had only NaNs")

    return elev_out, new_transform, new_crs


def _postprocess_coreg_apply_pts(
    applied_elev: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Post-processing and checks of apply for point input."""

    # TODO: Convert CRS back if the CRS did not match the one of the fit?
    return applied_elev


def _postprocess_coreg_apply_rst(
    elev: NDArrayf | gu.Raster,
    applied_elev: NDArrayf,
    transform: affine.Affine,
    out_transform: affine.Affine,
    crs: rio.crs.CRS,
    resample: bool,
    resampling: rio.warp.Resampling | None = None,
) -> tuple[NDArrayf | gu.Raster, affine.Affine]:
    """
    Post-processing and checks of apply for raster input.

    Here, "elev" and "transform" corresponds to user input, and are required to transform back the output that is
    composed of "applied_elev" and "out_transform".
    """

    # Ensure the dtype is OK
    applied_elev = applied_elev.astype("float32")

    # Set default dst_nodata
    if isinstance(elev, gu.Raster):
        nodata = elev.nodata
    else:
        nodata = raster._default_nodata(elev.dtype)

    # Resample the array on the original grid
    if resample:

        # TODO: Use this function for a translation only, for consistency with the rest of Coreg?
        #  (would require checking transform difference is only a translation)
        # applied_elev = _reproject_horizontal_shift_samecrs(raster_arr=applied_elev, src_transform=out_transform,
        #                                                    dst_transform=transform)

        # Reproject the DEM from its out_transform onto the transform
        applied_rst = gu.Raster.from_array(applied_elev, out_transform, crs=crs, nodata=nodata)
        if not isinstance(elev, gu.Raster):
            match_rst = gu.Raster.from_array(elev, transform, crs=crs, nodata=nodata)
        else:
            match_rst = elev
        applied_rst = applied_rst.reproject(match_rst, resampling=resampling, silent=True)
        applied_elev = applied_rst.data
        # Now that the raster data is reprojected, the new out_transform is set as the original transform
        out_transform = transform

    # Calculate final mask
    final_mask = np.logical_or(~np.isfinite(applied_elev), applied_elev == nodata)

    # If the DEM was a masked_array, copy the mask to the new DEM
    if isinstance(elev, (np.ma.masked_array, gu.Raster)):
        applied_elev = np.ma.masked_array(applied_elev, mask=final_mask)  # type: ignore
    else:
        applied_elev[final_mask] = np.nan

    # If the input was a Raster, returns a Raster, else returns array and transform
    if isinstance(elev, gu.Raster):
        out_dem = elev.from_array(applied_elev, out_transform, crs, nodata=elev.nodata)
        return out_dem, out_transform
    else:
        return applied_elev, out_transform


def _postprocess_coreg_apply(
    elev: NDArrayf | gu.Raster | gpd.GeoDataFrame,
    applied_elev: NDArrayf | gpd.GeoDataFrame,
    transform: affine.Affine,
    out_transform: affine.Affine,
    crs: rio.crs.CRS,
    resample: bool,
    resampling: rio.warp.Resampling | None = None,
) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine]:
    """
    Post-processing and checks of apply for any input.

    Here, "elev" and "transform" corresponds to user input, and are required to transform back the output that is
    composed of "applied_elev" and "out_transform".
    """

    # Define resampling
    resampling = resampling if isinstance(resampling, rio.warp.Resampling) else _resampling_method_from_str(resampling)

    # Distribute between raster and point apply methods
    if isinstance(applied_elev, np.ndarray):
        applied_elev, out_transform = _postprocess_coreg_apply_rst(
            elev=elev,
            applied_elev=applied_elev,
            transform=transform,
            crs=crs,
            out_transform=out_transform,
            resample=resample,
            resampling=resampling,
        )
    else:
        applied_elev = _postprocess_coreg_apply_pts(applied_elev)

    return applied_elev, out_transform


###############################################
# Statistical functions (to be moved in future)
###############################################


def _get_subsample_on_valid_mask(params_random: InRandomDict, valid_mask: NDArrayb) -> NDArrayb:
    """
    Get mask of values to subsample on valid mask (works for both 1D or 2D arrays).

    :param valid_mask: Mask of valid values (inlier and not nodata).
    """

    # This should never happen
    if params_random["subsample"] is None:
        raise ValueError("Subsample should have been defined in metadata before reaching this class method.")

    # If valid mask is empty
    if np.count_nonzero(valid_mask) == 0:
        raise ValueError(
            "There is no valid points common to the input and auxiliary data (bias variables, or "
            "derivatives required for this method, for example slope, aspect, etc)."
        )

    # If subsample is not equal to one, subsampling should be performed.
    elif params_random["subsample"] != 1.0:

        # Build a low memory masked array with invalid values masked to pass to subsampling
        ma_valid = np.ma.masked_array(data=np.ones(np.shape(valid_mask), dtype=bool), mask=~valid_mask)
        # Take a subsample within the valid values
        indices = gu.raster.subsample_array(
            ma_valid,
            subsample=params_random["subsample"],
            return_indices=True,
            random_state=params_random["random_state"],
        )

        # We return a boolean mask of the subsample within valid values
        subsample_mask = np.zeros(np.shape(valid_mask), dtype=bool)
        if len(indices) == 2:
            subsample_mask[indices[0], indices[1]] = True
        else:
            subsample_mask[indices[0]] = True
    else:
        # If no subsample is taken, use all valid values
        subsample_mask = valid_mask

    logging.debug(
        "Using a subsample of %d among %d valid values.", np.count_nonzero(subsample_mask), np.count_nonzero(valid_mask)
    )

    return subsample_mask


def _get_subsample_mask_pts_rst(
    params_random: InRandomDict,
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
    z_name: str,
    area_or_point: Literal["Area", "Point"] | None,
    aux_vars: None | dict[str, NDArrayf] = None,
) -> NDArrayb:
    """
    Get subsample mask for raster-raster or point-raster datasets on valid points of all inputs (including
    potential auxiliary variables).

    Returns a boolean array to use for subsampling (2D for raster-raster, 1D for point-raster to be used on point).
    """

    # TODO: Return more detailed error message for no valid points (which variable was full of NaNs?)

    if isinstance(ref_elev, gpd.GeoDataFrame) and isinstance(tba_elev, gpd.GeoDataFrame):
        raise TypeError(
            "This pre-processing function is only intended for raster-point or raster-raster methods, "
            "not point-point methods."
        )

    # For two rasters
    if isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):

        # Compute mask of valid data
        if aux_vars is not None:
            valid_mask = np.logical_and.reduce(
                (
                    inlier_mask,
                    np.isfinite(ref_elev),
                    np.isfinite(tba_elev),
                    *(np.isfinite(var) for var in aux_vars.values()),
                )
            )
        else:
            valid_mask = np.logical_and.reduce((inlier_mask, np.isfinite(ref_elev), np.isfinite(tba_elev)))

        # Raise errors if all values are NaN after introducing masks from the variables
        # (Others are already checked in pre-processing of Coreg.fit())

        # Perform subsampling
        sub_mask = _get_subsample_on_valid_mask(params_random=params_random, valid_mask=valid_mask)

    # For one raster and one point cloud
    else:

        # Interpolate inlier mask and bias vars at point coordinates
        pts_elev: gpd.GeoDataFrame = ref_elev if isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev
        rst_elev: NDArrayf = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev

        # Remove non-finite values from point dataset
        pts_elev = pts_elev[np.isfinite(pts_elev[z_name].values)]

        # Get coordinates
        pts = (pts_elev.geometry.x.values, pts_elev.geometry.y.values)

        # Get valid mask ahead of subsampling to have the exact number of requested subsamples
        if aux_vars is not None:
            valid_mask = np.logical_and.reduce(
                (inlier_mask, np.isfinite(rst_elev), *(np.isfinite(var) for var in aux_vars.values()))
            )
        else:
            valid_mask = np.logical_and.reduce((inlier_mask, np.isfinite(rst_elev)))

        # Convert inlier mask to points to be able to determine subsample later
        # The location needs to be surrounded by inliers, use floor to get 0 for at least one outlier
        # Interpolates boolean mask as integers
        # TODO: Create a function in GeoUtils that can compute the valid boolean mask of an interpolation without
        #  having to convert data to float32
        valid_mask = valid_mask.astype(np.float32)
        valid_mask[valid_mask == 0] = np.nan
        valid_mask = np.isfinite(
            _interp_points(array=valid_mask, transform=transform, points=pts, area_or_point=area_or_point)
        )

        # If there is a subsample, it needs to be done now on the point dataset to reduce later calculations
        sub_mask = _get_subsample_on_valid_mask(params_random=params_random, valid_mask=valid_mask)

    return sub_mask


def _subsample_on_mask(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    aux_vars: None | dict[str, NDArrayf],
    sub_mask: NDArrayb,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    return_coords: bool = False,
) -> tuple[NDArrayf, NDArrayf, None | dict[str, NDArrayf], None | tuple[NDArrayf, NDArrayf]]:
    """
    Perform subsampling on mask for raster-raster or point-raster datasets on valid points of all inputs (including
    potential auxiliary variables).

    Returns 1D arrays of subsampled inputs: reference elevation, to-be-aligned elevation and auxiliary variables
    (in dictionary), and (optionally) tuple of X/Y coordinates.
    """

    # For two rasters
    if isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):

        # Subsample all datasets with the mask
        sub_ref = ref_elev[sub_mask]
        sub_tba = tba_elev[sub_mask]
        if aux_vars is not None:
            sub_bias_vars = {}
            for var in aux_vars.keys():
                sub_bias_vars[var] = aux_vars[var][sub_mask]
        else:
            sub_bias_vars = None

        # Return coordinates if required
        if return_coords:
            coords = _coords(transform=transform, shape=ref_elev.shape, area_or_point=area_or_point)
            sub_coords = (coords[0][sub_mask], coords[1][sub_mask])
        else:
            sub_coords = None

    # For one raster and one point cloud
    else:

        # Identify which dataset is point or raster
        pts_elev: gpd.GeoDataFrame = ref_elev if isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev
        rst_elev: NDArrayf = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev

        # Remove invalid points
        pts_elev = pts_elev[np.isfinite(pts_elev[z_name].values)]

        # Subsample point coordinates
        pts = (pts_elev.geometry.x.values, pts_elev.geometry.y.values)
        pts = (pts[0][sub_mask], pts[1][sub_mask])

        # Interpolate raster array to the subsample point coordinates
        # Convert ref or tba depending on which is the point dataset
        sub_rst = _interp_points(array=rst_elev, transform=transform, points=pts, area_or_point=area_or_point)
        sub_pts = pts_elev[z_name].values[sub_mask]

        # Assign arrays depending on which one is the reference
        if isinstance(ref_elev, gpd.GeoDataFrame):
            sub_ref = sub_pts
            sub_tba = sub_rst
        else:
            sub_ref = sub_rst
            sub_tba = sub_pts

        # Interpolate arrays of bias variables to the subsample point coordinates
        if aux_vars is not None:
            sub_bias_vars = {}
            for var in aux_vars.keys():
                sub_bias_vars[var] = _interp_points(
                    array=aux_vars[var], transform=transform, points=pts, area_or_point=area_or_point
                )
        else:
            sub_bias_vars = None

        # Return coordinates if required
        if return_coords:
            sub_coords = pts
        else:
            sub_coords = None

    return sub_ref, sub_tba, sub_bias_vars, sub_coords


@overload
def _preprocess_pts_rst_subsample(
    params_random: InRandomDict,
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    aux_vars: None | dict[str, NDArrayf] = None,
    *,
    return_coords: Literal[False] = False,
) -> tuple[NDArrayf, NDArrayf, None | dict[str, NDArrayf], None]: ...


@overload
def _preprocess_pts_rst_subsample(
    params_random: InRandomDict,
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    aux_vars: None | dict[str, NDArrayf] = None,
    *,
    return_coords: Literal[True],
) -> tuple[NDArrayf, NDArrayf, None | dict[str, NDArrayf], tuple[NDArrayf, NDArrayf]]: ...


def _preprocess_pts_rst_subsample(
    params_random: InRandomDict,
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
    crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    aux_vars: None | dict[str, NDArrayf] = None,
    return_coords: bool = False,
) -> tuple[NDArrayf, NDArrayf, None | dict[str, NDArrayf], None | tuple[NDArrayf, NDArrayf]]:
    """
    Pre-process raster-raster or point-raster datasets into 1D arrays subsampled at the same points
    (and interpolated in the case of point-raster input).

    Return 1D arrays of reference elevation, to-be-aligned elevation and dictionary of 1D arrays of auxiliary variables
    at subsampled points.
    """

    # Get subsample mask (a 2D array for raster-raster, a 1D array of length the point data for point-raster)
    sub_mask = _get_subsample_mask_pts_rst(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
        aux_vars=aux_vars,
    )

    # Perform subsampling on mask for all inputs
    sub_ref, sub_tba, sub_bias_vars, sub_coords = _subsample_on_mask(
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        aux_vars=aux_vars,
        sub_mask=sub_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
        return_coords=return_coords,
    )

    # Return 1D arrays of subsampled points at the same location
    return sub_ref, sub_tba, sub_bias_vars, sub_coords


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    **kwargs: Any,
) -> tuple[None, tuple[NDArrayf, Any]]: ...


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["bin"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, None]: ...


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["bin_and_fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, tuple[NDArrayf, Any]]: ...


def _bin_or_and_fit_nd(
    fit_or_bin: Literal["fit", "bin", "bin_and_fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame | None, tuple[NDArrayf, Any] | None]:
    """
    Generic binning and/or fitting method to model values along N variables for a coregistration/correction,
    used for all affine and bias-correction subclasses. Expects either 2D arrays for rasters, or 1D arrays for
    points.

    :param params_fit_or_bin: Dictionary of parameters for fitting and/or binning (bias variable names,
    optimizer, bin sizes, etc), see FitOrBinDict for details.
    :param values: Valid values to bin or fit.
    :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
    :param weights: Array of weights for the coregistration.
    """

    if fit_or_bin is None:
        raise ValueError("This function should not be called for methods not supporting fit_or_bin logic.")

    # This is called by subclasses, so the bias_var should always be defined
    if bias_vars is None:
        raise ValueError("At least one `bias_var` should be passed to the fitting function, got None.")

    # Check number of variables
    nd = params_fit_or_bin["nd"]
    if nd is not None and len(bias_vars) != nd:
        raise ValueError(
            "A number of {} variable(s) has to be provided through the argument 'bias_vars', "
            "got {}.".format(nd, len(bias_vars))
        )

    # If bias var names were explicitly passed at instantiation, check that they match the one from the dict
    if params_fit_or_bin["bias_var_names"] is not None:
        if not sorted(bias_vars.keys()) == sorted(params_fit_or_bin["bias_var_names"]):
            raise ValueError(
                "The keys of `bias_vars` do not match the `bias_var_names` defined during "
                "instantiation: {}.".format(params_fit_or_bin["bias_var_names"])
            )

    # Get number of variables
    nd = len(bias_vars)

    # Remove random state for keyword argument if its value is not in the optimizer function
    if fit_or_bin in ["fit", "bin_and_fit"]:
        fit_func_args = inspect.getfullargspec(params_fit_or_bin["fit_optimizer"]).args
        if "random_state" not in fit_func_args and "random_state" in kwargs:
            kwargs.pop("random_state")

    # We need to sort the bin sizes in the same order as the bias variables if a dict is passed for bin_sizes
    if fit_or_bin in ["bin", "bin_and_fit"]:
        if isinstance(params_fit_or_bin["bin_sizes"], dict):
            var_order = list(bias_vars.keys())
            # Declare type to write integer or tuple to the variable
            bin_sizes: int | tuple[int, ...] | tuple[NDArrayf, ...] = tuple(
                np.array(params_fit_or_bin["bin_sizes"][var]) for var in var_order
            )
        # Otherwise, write integer directly
        else:
            bin_sizes = params_fit_or_bin["bin_sizes"]

    # Option 1: Run fit and save optimized function parameters
    if fit_or_bin == "fit":
        logging.debug(
            "Estimating alignment along variables %s by fitting with function %s.",
            ", ".join(list(bias_vars.keys())),
            params_fit_or_bin["fit_func"].__name__,
        )

        results = params_fit_or_bin["fit_optimizer"](
            f=params_fit_or_bin["fit_func"],
            xdata=np.array([var.flatten() for var in bias_vars.values()]).squeeze(),
            ydata=values.flatten(),
            sigma=weights.flatten() if weights is not None else None,
            absolute_sigma=True,
            **kwargs,
        )
        df = None

    # Option 2: Run binning and save dataframe of result
    elif fit_or_bin == "bin":
        logging.debug(
            "Estimating alignment along variables %s by binning with statistic %s.",
            ", ".join(list(bias_vars.keys())),
            params_fit_or_bin["bin_statistic"].__name__,
        )

        df = nd_binning(
            values=values,
            list_var=list(bias_vars.values()),
            list_var_names=list(bias_vars.keys()),
            list_var_bins=bin_sizes,
            statistics=(params_fit_or_bin["bin_statistic"], "count"),
        )
        results = None

    # Option 3: Run binning, then fitting, and save both results
    else:
        logging.debug(
            "Estimating alignment along variables %s by binning with statistic %s and then fitting with function %s.",
            ", ".join(list(bias_vars.keys())),
            params_fit_or_bin["bin_statistic"].__name__,
            params_fit_or_bin["fit_func"].__name__,
        )

        df = nd_binning(
            values=values,
            list_var=list(bias_vars.values()),
            list_var_names=list(bias_vars.keys()),
            list_var_bins=bin_sizes,
            statistics=(params_fit_or_bin["bin_statistic"], "count"),
        )

        # Now, we need to pass this new data to the fitting function and optimizer
        # We use only the N-D binning estimates (maximum dimension, equal to length of variable list)
        df_nd = df[df.nd == len(bias_vars)]

        # We get the middle of bin values for variable, and statistic for the diff
        new_vars = [pd.IntervalIndex(df_nd[var_name]).mid.values for var_name in bias_vars.keys()]
        new_diff = df_nd[params_fit_or_bin["bin_statistic"].__name__].values
        # TODO: pass a new sigma based on "count" and original sigma (and correlation?)?
        #  sigma values would have to be binned above also

        # Valid values for the binning output
        ind_valid = np.logical_and.reduce((np.isfinite(new_diff), *(np.isfinite(var) for var in new_vars)))

        if np.all(~ind_valid):
            raise ValueError("Only NaN values after binning, did you pass the right bin edges?")

        results = params_fit_or_bin["fit_optimizer"](
            f=params_fit_or_bin["fit_func"],
            xdata=np.array([var[ind_valid].flatten() for var in new_vars]).squeeze(),
            ydata=new_diff[ind_valid].flatten(),
            sigma=weights[ind_valid].flatten() if weights is not None else None,
            absolute_sigma=True,
            **kwargs,
        )
    logging.debug("%dD bias estimated.", nd)

    return df, results


###############################################
# Affine matrix manipulation and transformation
###############################################


def matrix_from_translations_rotations(
    t1: float = 0.0,
    t2: float = 0.0,
    t3: float = 0.0,
    alpha1: float = 0.0,
    alpha2: float = 0.0,
    alpha3: float = 0.0,
    use_degrees: bool = True,
) -> NDArrayf:
    """
    Build rigid affine matrix based on 3 translations (unit of coordinates) and 3 rotations (degrees or radians).

    The euler rotations use the extrinsic convention.

    :param t1: Translation in the X (west-east) direction (unit of coordinates).
    :param t2: Translation in the Y (south-north) direction (unit of coordinates).
    :param t3: Translation in the Z (vertical) direction (unit of DEM).
    :param alpha1: Rotation around the X (west-east) direction.
    :param alpha2: Rotation around the Y (south-north) direction.
    :param alpha3: Rotation around the Z (vertical) direction.
    :param use_degrees: Whether to use degrees for input rotations, otherwise radians.

    :raises ValueError: If the given translation or rotations contained invalid values.

    :return: Rigid affine matrix of transformation.
    """

    # Initialize diagonal matrix
    matrix = np.eye(4)
    # Convert euler angles to rotation matrix
    e = np.array([alpha1, alpha2, alpha3])
    # If angles were given in degrees
    if use_degrees:
        e = np.deg2rad(e)
    rot_matrix = pytransform3d.rotations.matrix_from_euler(e=e, i=0, j=1, k=2, extrinsic=True)

    # Add rotation matrix, and translations
    matrix[0:3, 0:3] = rot_matrix
    matrix[:3, 3] = [t1, t2, t3]

    return matrix


def translations_rotations_from_matrix(
    matrix: NDArrayf, return_degrees: bool = True
) -> tuple[float, float, float, float, float, float]:
    """
    Extract 3 translations (unit of coordinates) and 3 rotations (degrees or radians) from rigid affine matrix.

    The extracted euler rotations use the extrinsic convention.

    :param matrix: Rigid affine matrix of transformation.
    :param return_degrees: Whether to return rotations in degrees, otherwise radians.

    :return: Translations in the X, Y and Z direction and rotations around the X, Y and Z directions.
    """

    # Extract translations
    t1, t2, t3 = matrix[:3, 3]

    # Get rotations from affine matrix
    rots = pytransform3d.rotations.euler_from_matrix(matrix[:3, :3], i=0, j=1, k=2, extrinsic=True, strict_check=True)
    if return_degrees:
        rots = np.rad2deg(np.array(rots))

    # Extract rotations
    alpha1, alpha2, alpha3 = rots

    return t1, t2, t3, alpha1, alpha2, alpha3


def invert_matrix(matrix: NDArrayf) -> NDArrayf:
    """
    Invert a transformation matrix.

    :param matrix: Affine transformation matrix.

    :return: Inverted transformation matrix.
    """
    with warnings.catch_warnings():
        # Deprecation warning from pytransform3d. Let's hope that is fixed in the near future.
        warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")

        checked_matrix = pytransform3d.transformations.check_transform(matrix)
        # Invert the transform if wanted.
        return pytransform3d.transformations.invert_transform(checked_matrix)


def _apply_matrix_pts_mat(
    mat: NDArrayf,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    invert: bool = False,
) -> NDArrayf:
    """Apply matrix to points as a 3D array with 3D array output (to improve speed in some functions)."""

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # First, get 4xN array, adding a column of ones for translations during matrix multiplication
    points = np.concatenate((mat, np.ones((1, mat.shape[1]))))

    # Temporarily subtract centroid coordinates
    if centroid is not None:
        points[:3, :] -= np.array(centroid)[:, None]

    # Transform using matrix multiplication, and get only the first three columns
    transformed_points = (matrix @ points)[:3, :]

    # Add back centroid coordinates
    if centroid is not None:
        transformed_points += np.array(centroid)[:, None]

    return transformed_points


def _apply_matrix_pts_arr(
    x: NDArrayf,
    y: NDArrayf,
    z: NDArrayf,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    invert: bool = False,
) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
    """Apply matrix to points as arrays with array outputs (to improve speed in some functions)."""

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # First, get 4xN array, adding a column of ones for translations during matrix multiplication
    points = np.vstack([x, y, z, np.ones(len(x))])

    # Temporarily subtract centroid coordinates
    if centroid is not None:
        points[:3, :] -= np.array(centroid)[:, None]

    # Transform using matrix multiplication, and get only the first three columns
    transformed_points = (matrix @ points)[:3, :]

    # Add back centroid coordinates
    if centroid is not None:
        transformed_points += np.array(centroid)[:, None]

    return transformed_points[0, :], transformed_points[1, :], transformed_points[2, :]


def _apply_matrix_pts(
    epc: gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    z_name: str = "z",
) -> gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud.

    :param epc: Elevation point cloud.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.

    :return: Transformed elevation point cloud.
    """

    # Apply transformation to X/Y/Z arrays
    tx, ty, tz = _apply_matrix_pts_arr(
        x=epc.geometry.x.values,
        y=epc.geometry.y.values,
        z=epc[z_name].values,
        matrix=matrix,
        centroid=centroid,
        invert=invert,
    )

    # Finally, transform back to a new GeoDataFrame
    transformed_epc = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=tx, y=ty, crs=epc.crs),
        data={z_name: tz},
    )

    return transformed_epc


def _iterate_affine_regrid_small_rotations(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
) -> tuple[NDArrayf, rio.transform.Affine]:
    """
    Iterative process to find the best reprojection of affine transformation for small rotations.

    Faster than regridding point cloud by triangulation of points (for instance with scipy.interpolate.griddata).
    """

    # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
    dem_rst = gu.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
    epc = dem_rst.to_pointcloud(data_column_name="z", skip_nodata=False).ds

    # Exact affine transform of elevation point cloud (which yields irregular coordinates in 2D)
    tz0 = _apply_matrix_pts_arr(
        x=epc.geometry.x.values, y=epc.geometry.y.values, z=epc.z.values, matrix=matrix, centroid=centroid
    )[2]

    # We need to find the elevation Z of a transformed DEM at the exact grid coordinates X,Y
    # Which means we need to find coordinates X',Y',Z' of the original DEM that, after the exact affine transform,
    # fall exactly on regular X,Y coordinates

    # 1/ The elevation of the original DEM, Z', is simply a 2D interpolator function of X',Y' (bilinear, typically)
    # (We create the interpolator only once here for computational speed, instead of using Raster.interp_points)
    xycoords = dem_rst.coords(grid=False)
    z_interp = scipy.interpolate.RegularGridInterpolator(
        points=(np.flip(xycoords[1], axis=0), xycoords[0]), values=dem, method=resampling, bounds_error=False
    )

    # 2/ As a first guess of a transformed DEM elevation Z near the grid coordinates, we initialize with the elevations
    # of the nearest point from the transformed elevation point cloud

    # OLD METHOD
    # (Longest step computationally)
    # with warnings.catch_warnings():
    #     warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
    #     nearest = gpd.sjoin_nearest(epc, trans_epc)
    #
    # # In case several points are found at exactly the same distance, take the mean of their elevations
    # new_z = nearest.groupby(by=nearest.index)["z_left"].mean().values

    # NEW METHOD: Use the transformed elevation instead of searching for a nearest neighbour,
    # is close enough for small rotations! (and only creates a couple more iterations instead of a full search)
    new_z = tz0

    # 3/ We then iterate between two steps until convergence:
    # a/ Use the Z guess to derive invert affine transform X',Y' coordinates for the original DEM,
    # b/ Interpolate Z' at new coordinates X',Y' on the original DEM, and apply affine transform to get updated Z guess

    # Start with full array of X/Y regular coordinates (subset during iterations to improve computational efficiency)
    x = epc.geometry.x.values
    y = epc.geometry.y.values

    # Initialize output z array, and array to store points that have converged
    zfinal = np.ones(len(x), dtype=dem.dtype)
    ind_converged = np.zeros(len(x), dtype=bool)

    # For small rotations, and large DEMs (elevation range smaller than the DEM extent), this converges fast
    max_niter = 20  # Maximum iteration number
    niter_check = 5  # Number of iterations between residual checks
    tolerance = 10 ** (-4)  # Tolerance in X/Y relative to resolution of X/Y
    res_x = dem_rst.res[0]  # Resolution in X
    res_y = dem_rst.res[1]  # Resolution in Y
    niter = 1  # Starting iteration

    while niter < max_niter:

        # Invert X,Y (exact grid coordinates) with Z guess to find X',Y' coordinates on original DEM
        tx, ty = _apply_matrix_pts_arr(x=x, y=y, z=new_z, matrix=matrix, invert=True, centroid=centroid)[:2]

        # Interpolate original DEM at X', Y' to get Z', and convert to point cloud
        tz = z_interp((ty, tx))

        # Transform to see if we fall back on our feet (on the regular grid), or if we need to iterate more
        x0, y0, z0 = _apply_matrix_pts_arr(x=tx, y=ty, z=tz, matrix=matrix, centroid=centroid)

        # Only check residuals after first iteration (to remove NaNs) then every 5 iterations to reduce computing time
        if niter == 1 or niter == niter_check:

            # Compute difference between exact grid coordinates and current coordinates, and stop if tolerance reached
            diff_x = x0 - x
            diff_y = y0 - y

            logging.debug(
                "Residual check at iteration number %d:" "\n    Mean diff x: %f" "\n    Mean diff y: %f",
                niter,
                np.nanmean(np.abs(diff_x)),
                np.nanmean(np.abs(diff_y)),
            )

            # Get index of points below tolerance in both X/Y for this subsample (all points before convergence update)
            # Nodata values are considered having converged
            subind_diff_x = np.logical_or(np.abs(diff_x) < (tolerance * res_x), ~np.isfinite(diff_x))
            subind_diff_y = np.logical_or(np.abs(diff_y) < (tolerance * res_y), ~np.isfinite(diff_y))
            subind_converged = np.logical_and(subind_diff_x, subind_diff_y)

            logging.debug(
                "    Points not within tolerance: %d for X; %d for Y",
                np.count_nonzero(~subind_diff_x),
                np.count_nonzero(~subind_diff_y),
            )

            # If all points left are below convergence, update Z one final time and stop here
            if all(subind_converged):
                zfinal[~ind_converged] = z0
                break
            # Otherwise, save Z for new converged points and keep only not converged in next iterations (for speed)
            else:
                zfinal[~ind_converged] = z0
                x = x[~subind_converged]
                y = y[~subind_converged]
                z0 = z0[~subind_converged]

            # Otherwise, for this check, update convergence status for points not having converged yet
            ind_converged[~ind_converged] = subind_converged

        # If another iteration is required, update Z guess and increment
        new_z = z0
        niter += 1

    # 4/ Write the regular-grid point cloud back into a raster
    epc.z = zfinal  # We just replace the Z of the original grid to ensure exact coordinates
    transformed_dem = dem_rst.from_pointcloud_regular(
        epc, transform=transform, shape=dem.shape, data_column_name="z", nodata=-99999
    )

    return transformed_dem.data.filled(np.nan), transform


def _apply_matrix_rst(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    force_regrid_method: Literal["iterative", "griddata"] | None = None,
) -> tuple[NDArrayf, rio.transform.Affine]:
    """
    Apply a 3D affine transformation matrix to a 2.5D DEM.

    See details in description of apply_matrix().

    :param dem: DEM to transform.
    :param transform: Geotransform of the DEM.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resampling: Point interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more
    information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.
    :param force_regrid_method: Force re-gridding method to convert 3D point cloud to 2.5 DEM, only for testing.

    :returns: Transformed DEM, Transform.
    """

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # Check DEM has valid values
    if np.count_nonzero(np.isfinite(dem)) == 0:
        raise ValueError("Input DEM has all nans.")

    shift_z_only_matrix = np.diag(np.ones(4, float))
    shift_z_only_matrix[2, 3] = matrix[2, 3]

    shift_only_matrix = np.diag(np.ones(4, float))
    shift_only_matrix[:3, 3] = matrix[:3, 3]

    # 1/ Check if the matrix only contains a Z correction, in that case only shift the DEM values by the vertical shift
    if np.array_equal(shift_z_only_matrix, matrix) and force_regrid_method is None:
        return dem + matrix[2, 3], transform

    # 2/ Check if the matrix contains only translations, in that case only shift the DEM only by translation
    if np.array_equal(shift_only_matrix, matrix) and force_regrid_method is None:
        new_transform = _translate(transform, xoff=matrix[0, 3], yoff=matrix[1, 3])
        return dem + matrix[2, 3], new_transform

    # 3/ If matrix contains only small rotations (less than 20 degrees), use the fast iterative reprojection
    rotations = translations_rotations_from_matrix(matrix)[3:]
    if all(np.abs(rot) < 20 for rot in rotations) and force_regrid_method is None or force_regrid_method == "iterative":
        new_dem, transform = _iterate_affine_regrid_small_rotations(
            dem=dem, transform=transform, matrix=matrix, centroid=centroid, resampling=resampling
        )
        return new_dem, transform

    # 4/ Otherwise, use a delauney triangulation interpolation of the transformed point cloud
    # Convert DEM to elevation point cloud, keeping all exact grid coordinates X/Y even for NaNs
    dem_rst = gu.Raster.from_array(dem, transform=transform, crs=None, nodata=99999)
    epc = dem_rst.to_pointcloud(data_column_name="z").ds
    trans_epc = _apply_matrix_pts(epc, matrix=matrix, centroid=centroid)

    new_dem = _grid_pointcloud(
        trans_epc, grid_coords=dem_rst.coords(grid=False), data_column_name="z", resampling=resampling
    )

    return new_dem, transform


@overload
def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    *,
    return_interpolator: Literal[False] = False,
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> NDArrayf: ...


@overload
def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    *,
    return_interpolator: Literal[True],
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf]: ...


def _reproject_horizontal_shift_samecrs(
    raster_arr: NDArrayf,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine = None,
    return_interpolator: bool = False,
    resampling: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"] = "linear",
) -> NDArrayf | Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf]:
    """
    Reproject a raster only for a horizontal shift (transform update) in the same CRS.

    This function exists independently of Raster.reproject() because Rasterio has unexplained reprojection issues
    that can create non-negligible sub-pixel shifts that should be crucially avoided for coregistration.
    See https://github.com/rasterio/rasterio/issues/2052#issuecomment-2078732477.

    Here we use SciPy interpolation instead, modified for nodata propagation in geoutils.interp_points().
    """

    # We are reprojecting the raster array relative to itself without changing its pixel interpretation, so we can
    # force any pixel interpretation (area_or_point) without it having any influence on the result, here "Area"
    if not return_interpolator:
        coords_dst = _coords(transform=dst_transform, area_or_point="Area", shape=raster_arr.shape)
        # Flatten the arrays (only 1D supported in rowcol/xy after Rasterio 1.4)
        coords_dst = (coords_dst[0].ravel(), coords_dst[1].ravel())
    # If we just want the interpolator, we don't need to coordinates of destination points
    else:
        coords_dst = None

    output = _interp_points(
        array=raster_arr,
        area_or_point="Area",
        transform=src_transform,
        points=coords_dst,
        method=resampling,
        return_interpolator=return_interpolator,
    )

    # Reshape output
    if coords_dst is not None:
        output = output.reshape(np.shape(raster_arr))

    return output


@overload
def apply_matrix(
    elev: NDArrayf,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
    **kwargs: Any,
) -> tuple[NDArrayf, affine.Affine]: ...


@overload
def apply_matrix(
    elev: gu.Raster | gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
    **kwargs: Any,
) -> gu.Raster | gpd.GeoDataFrame: ...


def apply_matrix(
    elev: gu.Raster | NDArrayf | gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resample: bool = True,
    resampling: Literal["nearest", "linear", "cubic", "quintic"] = "linear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
    **kwargs: Any,
) -> tuple[NDArrayf, affine.Affine] | gu.Raster | gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud or 2.5D DEM.

    For an elevation point cloud, the transformation is exact.

    For a DEM, it requires re-gridding because the affine-transformed point cloud of the DEM does not fall onto a
    regular grid anymore (except if the affine transformation only has translations). For this, this function uses the
    three following methods:

    1. For transformations with only translations, the transform is updated and vertical shift added to the array,

    2. For transformations with a small rotation (20 degrees or less for all axes), this function maps which 2D
    point coordinates will fall back exactly onto the original DEM grid coordinates after affine transformation by
    searching iteratively using the invert affine transformation and 2D point regular-grid interpolation on the
    original DEM (see geoutils.Raster.interp_points, or scipy.interpolate.interpn),

    3. For transformations with large rotations (20 degrees or more), scipy.interpolate.griddata is used to
    re-grid the irregular affine-transformed 3D point cloud using Delauney triangulation interpolation (slower).

    :param elev: Elevation point cloud or DEM to transform, either a 2D array (requires transform) or
        geodataframe (requires z_name).
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resample: (For translations) If set to True, will resample output on the translated grid to match the input
        transform. Otherwise, only the transform will be updated and no resampling is done.
    :param resampling: Point interpolation method, one of 'nearest', 'linear', 'cubic', or 'quintic'. For more
        information, see scipy.ndimage.map_coordinates and scipy.interpolate.interpn. Default is linear.
    :param transform: Geotransform of the DEM, only for DEM passed as 2D array.
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
    :param kwargs: Keywords passed to _apply_matrix_rst for testing.

    :return: Affine transformed elevation point cloud or DEM.
    """

    # Apply matrix to elevation point cloud
    if isinstance(elev, gpd.GeoDataFrame):
        return _apply_matrix_pts(epc=elev, matrix=matrix, invert=invert, centroid=centroid, z_name=z_name)
    # Or apply matrix to raster (often requires re-gridding)
    else:

        # First, we apply the affine matrix for the array/transform
        if isinstance(elev, gu.Raster):
            transform = elev.transform
            dem = elev.data.filled(np.nan)
        else:
            dem = elev
        applied_dem, out_transform = _apply_matrix_rst(
            dem=dem,
            transform=transform,
            matrix=matrix,
            invert=invert,
            centroid=centroid,
            resampling=resampling,
            **kwargs,
        )

        # Then, if resample is True, we reproject the DEM from its out_transform onto the transform
        if resample:
            applied_dem = _reproject_horizontal_shift_samecrs(
                applied_dem, src_transform=out_transform, dst_transform=transform, resampling=resampling
            )
            out_transform = transform

        # We return a raster if input was a raster
        if isinstance(elev, gu.Raster):
            applied_dem = gu.Raster.from_array(applied_dem, out_transform, elev.crs, elev.nodata)
            return applied_dem
        return applied_dem, out_transform


###########################################
# Generic coregistration processing classes
###########################################


class NotImplementedCoregFit(NotImplementedError):
    """
    Error subclass for not implemented coregistration fit methods; mainly to differentiate with NotImplementedError
    """


class NotImplementedCoregApply(NotImplementedError):
    """
    Error subclass for not implemented coregistration fit methods; mainly to differentiate with NotImplementedError
    """


class InRandomDict(TypedDict, total=False):
    """Keys and types of inputs associated with randomization and subsampling."""

    # Subsample size input by user
    subsample: int | float
    # Random state (for subsampling, but also possibly for some fitting methods)
    random_state: int | np.random.Generator | None


class OutRandomDict(TypedDict, total=False):
    """Keys and types of outputs associated with randomization and subsampling."""

    # Final subsample size available from valid data
    subsample_final: int


class InFitOrBinDict(TypedDict, total=False):
    """Keys and types of inputs associated with binning and/or fitting."""

    # Whether to fit, bin or bin then fit
    fit_or_bin: Literal["fit", "bin", "bin_and_fit"]

    # Fit parameters: function to fit and optimizer
    fit_func: Callable[..., NDArrayf]
    fit_optimizer: Callable[..., tuple[NDArrayf, Any]]

    # TODO: Solve redundancy between optimizer and minimizer (curve_fit or minimize as default?)
    # For a minimization problem
    fit_minimizer: Callable[..., tuple[NDArrayf, Any]]
    fit_loss_func: Callable[[NDArrayf], np.floating[Any]]

    # Bin parameters: bin sizes, statistic and apply method
    bin_sizes: int | dict[str, int | Iterable[float]]
    bin_statistic: Callable[[NDArrayf], np.floating[Any]]
    bin_apply_method: Literal["linear", "per_bin"]
    # Name of variables, and number of dimensions
    bias_var_names: list[str]
    nd: int | None


class OutFitOrBinDict(TypedDict, total=False):
    """Keys and types of outputs associated with binning and/or fitting."""

    # Optimized parameters for fitted function, and its error
    fit_params: NDArrayf
    fit_perr: NDArrayf
    # Binning dataframe
    bin_dataframe: pd.DataFrame


class InIterativeDict(TypedDict, total=False):
    """Keys and types of inputs associated with iterative methods."""

    # Maximum number of iterations
    max_iterations: int
    # Tolerance at which to stop algorithm (unit specified in method)
    tolerance: float


class OutIterativeDict(TypedDict, total=False):
    """Keys and types of outputs associated with iterative methods."""

    # Iteration at which the algorithm stopped
    last_iteration: int
    # Tolerances of each iteration until threshold
    all_tolerances: list[float]


class InSpecificDict(TypedDict, total=False):
    """Keys and types of inputs associated with specific methods."""

    # (Using TerrainBias) Selected terrain attribute
    terrain_attribute: str
    # (Using DirectionalBias) Angle for directional correction
    angle: float
    # (Using Deramp) Polynomial order selected for deramping
    poly_order: int
    # (Using ICP) Method type to compute 3D distances
    icp_method: Literal["point-to-point", "point-to-plane"]
    # (Using ICP) Picky selection of closest pairs
    icp_picky: bool

    # (Using CPD) Weight for outlier removal
    cpd_weight: float


class OutSpecificDict(TypedDict, total=False):
    """Keys and types of outputs associated with specific methods."""

    # (Using multi-order polynomial fit) Best performing polynomial order
    best_poly_order: int
    # (Using multi-frequency sum of sinusoids fit) Best performing number of frequencies
    best_nb_sin_freq: int


class InAffineDict(TypedDict, total=False):
    """Keys and types of inputs associated with affine methods."""

    # Vertical shift reduction function for methods focusing on translation coregistration
    vshift_reduc_func: Callable[[NDArrayf], np.floating[Any]]
    # Vertical shift activated
    apply_vshift: bool
    # Apply coregistration method only for translations
    only_translation: bool
    # Standardize input data for numerics
    standardize: bool


class OutAffineDict(TypedDict, total=False):
    """Keys and types of outputs associated with affine methods."""

    # Output common to all affine transforms
    centroid: tuple[float, float, float]
    matrix: NDArrayf

    # For translation methods
    shift_x: float
    shift_y: float
    shift_z: float


class InputCoregDict(TypedDict, total=False):

    random: InRandomDict
    fitorbin: InFitOrBinDict
    iterative: InIterativeDict
    specific: InSpecificDict
    affine: InAffineDict


class OutputCoregDict(TypedDict, total=False):
    random: OutRandomDict
    fitorbin: OutFitOrBinDict
    iterative: OutIterativeDict
    specific: OutSpecificDict
    affine: OutAffineDict


class CoregDict(TypedDict, total=False):
    """
    Defining the type of each possible key in the metadata dictionary of Coreg classes.
    The parameter total=False means that the key are not required. In the recent PEP 655 (
    https://peps.python.org/pep-0655/) there is an easy way to specific Required or NotRequired for each key, if we
    want to change this in the future.
    """

    # For a classic coregistration
    inputs: InputCoregDict
    outputs: OutputCoregDict

    # For pipelines and blocks
    # TODO: Move out to separate TypedDict?
    step_meta: list[Any]
    pipeline: list[Any]


CoregType = TypeVar("CoregType", bound="Coreg")


class Coreg:
    """
    Generic co-registration processing class.

    Used to implement methods common to all processing steps (rigid alignment, bias corrections, filtering).
    Those are: instantiation, copying and addition (which casts to a Pipeline object).

    Made to be subclassed.
    """

    _fit_called: bool = False  # Flag to check if the .fit() method has been called.
    _is_affine: bool | None = None
    _is_translation: bool | None = None
    _needs_vars: bool = False
    _meta: CoregDict

    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        """Instantiate a generic processing step method."""

        # Automatically sort input keys into their appropriate nested level using only the TypedDicts defined
        # above which make up the CoregDict altogether
        dict_meta = CoregDict(inputs={}, outputs={})
        if meta is not None:
            # First, we get the typed dictionary keys ("random", "fitorbin", etc),
            # this is a typing class so requires to get its keys in __annotations__
            list_input_levels = list(InputCoregDict.__annotations__.keys())
            # Then the list of keys per level, getting the nested class value for each key (via __forward_arg__)
            keys_per_level = [
                list(globals()[InputCoregDict.__annotations__[lv].__forward_arg__].__annotations__.keys())
                for lv in list_input_levels
            ]

            # Join all keys for input check
            all_keys = [k for lv in keys_per_level for k in lv]
            for k in meta.keys():
                if k not in all_keys:
                    raise ValueError(
                        f"Coregistration metadata key {k} is not supported. " f"Should be one of {', '.join(all_keys)}"
                    )

            # Add keys to inputs
            for k, v in meta.items():
                for i, lv in enumerate(list_input_levels):
                    # If level does not exist, create it
                    if lv not in dict_meta["inputs"]:
                        dict_meta["inputs"].update({lv: {}})  # type: ignore
                    # If key exist, write and continue
                    if k in keys_per_level[i]:
                        dict_meta["inputs"][lv][k] = v  # type: ignore
                        continue

        self._meta: CoregDict = dict_meta

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        # Need a deepcopy for dictionaries, or it would just point towards the copied coreg
        new_coreg.__dict__ = {key: copy.deepcopy(value) for key, value in self.__dict__.items()}

        return new_coreg

    def __add__(self, other: CoregType) -> CoregPipeline:
        """Return a pipeline consisting of self and the other processing function."""
        if not isinstance(other, Coreg):
            raise ValueError(f"Incompatible add type: {type(other)}. Expected 'Coreg' subclass")
        return CoregPipeline([self, other])

    @property
    def is_affine(self) -> bool:
        """Check if the transform be explained by a 3D affine transform."""
        # _is_affine is found by seeing if to_matrix() raises an error.
        # If this hasn't been done yet, it will be None
        if self._is_affine is None:
            try:  # See if to_matrix() raises an error.
                self.to_matrix()
                self._is_affine = True
            except (AttributeError, ValueError, NotImplementedError):
                self._is_affine = False

        return self._is_affine

    @property
    def is_translation(self) -> bool | None:

        # If matrix exists in keys, or can be derived from to_matrix(), we conclude
        if "matrix" in self._meta["outputs"]["affine"].keys():
            matrix = self._meta["outputs"]["affine"]["matrix"]
        else:
            try:
                matrix = self.to_matrix()
            # Otherwise we can't yet and return None
            except (AttributeError, ValueError, NotImplementedError):
                self._is_translation = None
                return None

        # If the 3x3 rotation sub-matrix is the identity matrix, we have a translation
        return np.allclose(matrix[:3, :3], np.diag(np.ones(3)), rtol=10e-3)

    @property
    def meta(self) -> CoregDict:
        """Metadata dictionary of the coregistration."""

        return self._meta

    @overload
    def info(self, as_str: Literal[False] = ...) -> None: ...

    @overload
    def info(self, as_str: Literal[True]) -> str: ...

    def info(self, as_str: bool = False) -> None | str:
        """Summarize information about this coregistration."""

        # Define max tabulation: longest name + 2 spaces
        tab = np.max([len(v) for v in dict_key_to_str.values()]) + 2

        # Get list of existing deepest level keys in this coreg metadata
        def recursive_items(dictionary: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
            for key, value in dictionary.items():
                if type(value) is dict:
                    yield from recursive_items(value)
                else:
                    yield (key, value)

        existing_deep_keys = [k for k, v in recursive_items(self._meta)]

        # Formatting function for key values, rounding up digits for numbers and returning function names
        def format_coregdict_values(val: Any, tab: int) -> str:
            """
            Format coregdict values for printing.

            :param val: Input value.
            :param tab: Tabulation (if value is printed on multiple lines).

            :return: String representing input value.
            """

            # Function to get decimal to round to a certain number of digits relative to magnitude, for floating numbers
            def dec_round_to_n(x: float | np.floating[Any], n: int) -> int:
                return -int(np.floor(np.log10(np.abs(x)))) + (n - 1)

            # Different formatting to string depending on key value type
            if isinstance(val, (float, np.floating)):
                if np.isfinite(val):
                    str_val = str(round(val, dec_round_to_n(val, 3)))
                else:
                    str_val = str(val)
            elif isinstance(val, np.ndarray):
                min_val = np.min(val)
                str_val = str(np.round(val, decimals=dec_round_to_n(min_val, 3)))
            elif callable(val):
                str_val = val.__name__
            else:
                str_val = str(val)

            # Add tabulation if string has a return to line
            if "\n" in str_val:
                str_val = "\n".ljust(tab).join(str_val.split("\n"))

            return str_val

        # Sublevels of metadata to show
        sublevels = {
            "random": "Randomization",
            "fitorbin": "Fitting and binning",
            "affine": "Affine",
            "iterative": "Iterative",
            "specific": "Specific",
        }

        header_str = [
            "Generic coregistration information \n",
            f"  Method:       {self.__class__.__name__} \n",
            f"  Is affine?    {self.is_affine} \n",
            f"  Fit called?   {self._fit_called} \n",
        ]

        # Add lines for inputs
        inputs_str = [
            "Inputs\n",
        ]
        for lk, lv in sublevels.items():
            if lk in self._meta["inputs"].keys():
                existing_level_keys = [
                    (k, v) for k, v in self._meta["inputs"][lk].items() if k in existing_deep_keys  # type: ignore
                ]
                if len(existing_level_keys) > 0:
                    inputs_str += [f"  {lv}\n"]
                    inputs_str += [
                        f"    {dict_key_to_str[k]}:".ljust(tab) + f"{format_coregdict_values(v, tab)}\n"
                        for k, v in existing_level_keys
                    ]

        # And for outputs
        outputs_str = ["Outputs\n"]
        # If dict not empty
        if self._meta["outputs"]:
            for lk, lv in sublevels.items():
                if lk in self._meta["outputs"].keys():
                    existing_level_keys = [
                        (k, v) for k, v in self._meta["outputs"][lk].items() if k in existing_deep_keys  # type: ignore
                    ]
                    if len(existing_level_keys) > 0:
                        outputs_str += [f"  {lv}\n"]
                        outputs_str += [
                            f"    {dict_key_to_str[k]}:".ljust(tab) + f"{format_coregdict_values(v, tab)}\n"
                            for k, v in existing_level_keys
                        ]
        elif not self._fit_called:
            outputs_str += ["  None yet (fit not called)"]
        # Not sure this case can happen, but just in case
        else:
            outputs_str += ["  None"]

        # Combine into final string
        final_str = header_str + inputs_str + outputs_str

        # Return as string or print (default)
        if as_str:
            return "".join(final_str)
        else:
            print("".join(final_str))
            return None

    def _get_subsample_on_valid_mask(self, valid_mask: NDArrayb) -> NDArrayb:
        """
        Get mask of values to subsample on valid mask.

        :param valid_mask: Mask of valid values (inlier and not nodata).
        """

        # Get random parameters
        params_random = self._meta["inputs"]["random"]

        # Derive subsampling mask
        sub_mask = _get_subsample_on_valid_mask(
            params_random=params_random,
            valid_mask=valid_mask,
        )

        # Write final subsample to class
        self._meta["outputs"]["random"] = {"subsample_final": int(np.count_nonzero(sub_mask))}

        return sub_mask

    def _preprocess_rst_pts_subsample(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        aux_vars: dict[str, NDArrayf] | None = None,
        weights: NDArrayf | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
    ) -> tuple[NDArrayf, NDArrayf, None | dict[str, NDArrayf]]:
        """
        Pre-process raster-raster or point-raster datasets into 1D arrays subsampled at the same points
        (and interpolated in the case of point-raster input).

        Return 1D arrays of reference elevation, to-be-aligned elevation and dictionary of 1D arrays of auxiliary
        variables at subsampled points.
        """

        # Get random parameters
        params_random: InRandomDict = self._meta["inputs"]["random"]

        # Get subsample mask (a 2D array for raster-raster, a 1D array of length the point data for point-raster)
        sub_mask = _get_subsample_mask_pts_rst(
            params_random=params_random,
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            area_or_point=area_or_point,
            z_name=z_name,
            aux_vars=aux_vars,
        )

        # Perform subsampling on mask for all inputs
        sub_ref, sub_tba, sub_bias_vars, _ = _subsample_on_mask(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            aux_vars=aux_vars,
            sub_mask=sub_mask,
            transform=transform,
            area_or_point=area_or_point,
            z_name=z_name,
        )

        # Write final subsample to class
        self._meta["outputs"]["random"] = {"subsample_final": int(np.count_nonzero(sub_mask))}

        return sub_ref, sub_tba, sub_bias_vars

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:
        """
        Estimate the coregistration transform on the given DEMs.

        :param reference_elev: Reference elevation, either a DEM or an elevation point cloud.
        :param to_be_aligned_elev: To-be-aligned elevation, either a DEM or an elevation point cloud.
        :param inlier_mask: Mask or boolean array of areas to include (inliers=True).
        :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
        :param weights: Array of weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param transform: Transform of the reference elevation, only if provided as 2D array.
        :param crs: CRS of the reference elevation, only if provided as 2D array.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling).
        """

        if weights is not None:
            raise NotImplementedError("Weights have not yet been implemented")

        # Override subsample argument of instantiation if passed to fit
        if subsample is not None:

            # Check if subsample argument was also defined at instantiation (not default value), and raise warning
            argspec = inspect.getfullargspec(self.__class__)
            sub_meta = self._meta["inputs"]["random"]["subsample"]
            if argspec.defaults is None or "subsample" not in argspec.args:
                raise ValueError("The subsample argument and default need to be defined in this Coreg class.")
            sub_is_default = argspec.defaults[argspec.args.index("subsample") - 1] == sub_meta  # type: ignore
            if not sub_is_default:
                warnings.warn(
                    "Subsample argument passed to fit() will override non-default subsample value defined at "
                    "instantiation. To silence this warning: only define 'subsample' in either fit(subsample=...) or "
                    "instantiation e.g. VerticalShift(subsample=...)."
                )

            # In any case, override!
            self._meta["inputs"]["random"]["subsample"] = subsample

        # Save random_state if a subsample is used
        if self._meta["inputs"]["random"]["subsample"] != 1:
            self._meta["inputs"]["random"]["random_state"] = random_state

        # Pre-process the inputs, by reprojecting and converting to arrays
        ref_elev, tba_elev, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        main_args = {
            "ref_elev": ref_elev,
            "tba_elev": tba_elev,
            "inlier_mask": inlier_mask,
            "transform": transform,
            "crs": crs,
            "area_or_point": area_or_point,
            "z_name": z_name,
            "weights": weights,
        }

        # If bias_vars are defined, update dictionary content to array
        if bias_vars is not None:
            # Check if the current class actually requires bias_vars
            if self._is_affine:
                warnings.warn("This coregistration method is affine, ignoring `bias_vars` passed to fit().")

            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

            main_args.update({"bias_vars": bias_vars})

        # Run the associated fitting function, which has fallback logic for "raster-raster", "raster-point" or
        # "point-point" depending on what is available for a certain Coreg function
        self._fit_func(
            **main_args,
            **kwargs,
        )

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame: ...

    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:
        """
        Apply the estimated transform to a DEM.

        :param elev: Elevation to apply the transform to, either a DEM or an elevation point cloud.
        :param bias_vars: Only for some bias correction classes. 2D array of bias variables used.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
            only the transform might be updated and no resampling is done.
        :param resampling: Resampling method if resample is used. Defaults to "bilinear".
        :param transform: Geotransform of the elevation, only if provided as 2D array.
        :param crs: CRS of elevation, only if provided as 2D array.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param kwargs: Any optional arguments to be passed to either self._apply_rst or apply_matrix.

        :returns: The transformed DEM.
        """
        if not self._fit_called and self._meta["outputs"]["affine"].get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        elev_array, transform, crs = _preprocess_coreg_apply(elev=elev, transform=transform, crs=crs)

        main_args = {"elev": elev_array, "transform": transform, "crs": crs, "resample": resample, "z_name": z_name}

        # If bias_vars are defined, update dictionary content to array
        if bias_vars is not None:
            # Check if the current class actually requires bias_vars
            if self._is_affine:
                warnings.warn("This coregistration method is affine, ignoring `bias_vars` passed to apply().")

            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

            main_args.update({"bias_vars": bias_vars})

        # Call _apply_func to choose method depending on point/raster input and if specific apply method exists
        applied_elev, out_transform = self._apply_func(**main_args, **kwargs)

        # Post-process output depending on input type
        applied_elev, out_transform = _postprocess_coreg_apply(
            elev=elev,
            applied_elev=applied_elev,
            transform=transform,
            out_transform=out_transform,
            crs=crs,
            resample=resample,
            resampling=resampling,
        )

        # Only return object if raster or geodataframe, also return transform if object was an array
        if isinstance(applied_elev, (gu.Raster, gpd.GeoDataFrame)):
            return applied_elev
        else:
            return applied_elev, out_transform

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: MArrayf,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
    ) -> RasterType | gpd.GeoDataFrame: ...

    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:
        """Estimate and apply the coregistration to a pair of elevation data.

        :param reference_elev: Reference elevation, either a DEM or an elevation point cloud.
        :param to_be_aligned_elev: To-be-aligned elevation, either a DEM or an elevation point cloud.
        :param inlier_mask: Mask or boolean array of areas to include (inliers=True).
        :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
        :param weights: Array of weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param transform: Transform of the reference elevation, only if provided as 2D array.
        :param crs: CRS of the reference elevation, only if provided as 2D array.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
            only the transform might be updated and no resampling is done.
        :param resampling: Resampling method if resample is used. Defaults to "bilinear".
        :param random_state: Random state or seed number to use for calculations (to fix random sampling).
        :param fit_kwargs: Keyword arguments to be passed to fit.
        :param apply_kwargs: Keyword argument to be passed to apply.
        """

        if fit_kwargs is None:
            fit_kwargs = {}
        if apply_kwargs is None:
            apply_kwargs = {}

        self.fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            bias_vars=bias_vars,
            weights=weights,
            subsample=subsample,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            random_state=random_state,
            **fit_kwargs,
        )

        aligned_dem = self.apply(
            elev=to_be_aligned_elev,
            bias_vars=bias_vars,
            resample=resample,
            resampling=resampling,
            transform=transform,
            crs=crs,
            z_name=z_name,
            **apply_kwargs,
        )

        return aligned_dem

    def residuals(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        subsample: float | int = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayf:
        """
        Calculate the residual offsets (the difference) between two DEMs after applying the transformation.

        :param reference_elev: 2D array of elevation values acting reference.
        :param to_be_aligned_elev: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

        :returns: A 1D array of finite residuals.
        """

        # Apply the transformation to the dem to be aligned
        aligned_elev = self.apply(to_be_aligned_elev, transform=transform, crs=crs)[0]

        # Pre-process the inputs, by reprojecting and subsampling
        ref_dem, align_elev, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        # Calculate the DEM difference
        diff = ref_dem - align_elev

        # Sometimes, the float minimum (for float32 = -3.4028235e+38) is returned. This and inf should be excluded.
        full_mask = np.isfinite(diff)
        if "float" in str(diff.dtype):
            full_mask[(diff == np.finfo(diff.dtype).min) | np.isinf(diff)] = False

        # Return the difference values within the full inlier mask
        return diff[full_mask]

    @overload
    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: list[str],
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
    ) -> list[np.floating[Any] | float | np.integer[Any] | int]: ...

    @overload
    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: str = "nmad",
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
    ) -> np.floating[Any] | float | np.integer[Any] | int: ...

    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: str | list[str] = "nmad",
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
    ) -> np.floating[Any] | float | np.integer[Any] | int | list[np.floating[Any] | float | np.integer[Any] | int]:
        """
        Calculate the error of a coregistration approach.

        Choices:
            - "nmad": Default. The Normalized Median Absolute Deviation of the residuals.
            - "median": The median of the residuals.
            - "mean": The mean/average of the residuals
            - "std": The standard deviation of the residuals.
            - "rms": The root mean square of the residuals.
            - "mae": The mean absolute error of the residuals.
            - "count": The residual count.

        :param reference_elev: 2D array of elevation values acting reference.
        :param to_be_aligned_elev: 2D array of elevation values to be aligned.
        :param error_type: The type of error measure to calculate. May be a list of error types.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.

        :returns: The error measure of choice for the residuals.
        """
        if isinstance(error_type, str):
            error_type = [error_type]

        residuals = self.residuals(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        def rms(res: NDArrayf) -> np.floating[Any]:
            return np.sqrt(np.mean(np.square(res)))

        def mae(res: NDArrayf) -> np.floating[Any]:
            return np.mean(np.abs(res))

        def count(res: NDArrayf) -> int:
            return res.size

        error_functions: dict[str, Callable[[NDArrayf], np.floating[Any] | float | np.integer[Any] | int]] = {
            "nmad": nmad,
            "median": np.median,
            "mean": np.mean,
            "std": np.std,
            "rms": rms,
            "mae": mae,
            "count": count,
        }

        try:
            errors = [error_functions[err_type](residuals) for err_type in error_type]
        except KeyError as exception:
            raise ValueError(
                f"Invalid 'error_type'{'s' if len(error_type) > 1 else ''}: "
                f"'{error_type}'. Choices: {list(error_functions.keys())}"
            ) from exception

        return errors if len(errors) > 1 else errors[0]

    def _fit_func(
        self,
        **kwargs: Any,
    ) -> None:
        """
        Distribute to _fit_rst_rst, fit_rst_pts or fit_pts_pts depending on input and method availability.
        Needs to be _fit_func of the main class to simplify calls from CoregPipeline and BlockwiseCoreg.
        """

        # Determine if input is raster-raster, raster-point or point-point
        if all(isinstance(dem, np.ndarray) for dem in (kwargs["ref_elev"], kwargs["tba_elev"])):
            rop = "r-r"
        elif all(isinstance(dem, gpd.GeoDataFrame) for dem in (kwargs["ref_elev"], kwargs["tba_elev"])):
            rop = "p-p"
        else:
            rop = "r-p"

        # Fallback logic is always the same: 1/ raster-raster, 2/ raster-point, 3/ point-point
        try_rp = False
        try_pp = False

        # For raster-raster
        if rop == "r-r":
            # Check if raster-raster function exists, if yes run it and stop
            try:
                self._fit_rst_rst(**kwargs)
            # Otherwise, convert the tba raster to points and try raster-points
            except NotImplementedCoregFit:
                warnings.warn(
                    f"No raster-raster method found for coregistration {self.__class__.__name__}, "
                    f"trying raster-point method by converting to-be-aligned DEM to points.",
                    UserWarning,
                )
                tba_elev_pts = (
                    gu.Raster.from_array(data=kwargs["tba_elev"], transform=kwargs["transform"], crs=kwargs["crs"])
                    .to_pointcloud()
                    .ds
                )
                kwargs.update({"tba_elev": tba_elev_pts})
                try_rp = True

        # For raster-point
        if rop == "r-p" or try_rp:
            try:
                self._fit_rst_pts(**kwargs)
            except NotImplementedCoregFit:
                warnings.warn(
                    f"No raster-point method found for coregistration {self.__class__.__name__}, "
                    f"trying point-point method by converting all elevation data to points.",
                    UserWarning,
                )
                ref_elev_pts = (
                    gu.Raster.from_array(data=kwargs["ref_elev"], transform=kwargs["transform"], crs=kwargs["crs"])
                    .to_pointcloud()
                    .ds
                )
                kwargs.update({"ref_elev": ref_elev_pts})
                try_pp = True

        # For point-point
        if rop == "p-p" or try_pp:
            try:
                self._fit_pts_pts(**kwargs)
            except NotImplementedCoregFit:
                if try_pp and try_rp:
                    raise NotImplementedCoregFit(
                        f"No raster-raster, raster-point or point-point method found for "
                        f"coregistration {self.__class__.__name__}."
                    )
                elif try_pp:
                    raise NotImplementedCoregFit(
                        f"No raster-point or point-point method found for coregistration {self.__class__.__name__}."
                    )
                else:
                    raise NotImplementedCoregFit(
                        f"No point-point method found for coregistration {self.__class__.__name__}."
                    )

    def _apply_func(self, **kwargs: Any) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine]:
        """Distribute to _apply_rst and _apply_pts based on input and method availability."""

        # If input is a raster
        if isinstance(kwargs["elev"], np.ndarray):

            # See if a _apply_rst exists
            try:
                # Run the associated apply function
                applied_elev, out_transform = self._apply_rst(**kwargs)  # pylint: disable=assignment-from-no-return

            # If it doesn't exist, use apply_matrix()
            except NotImplementedCoregApply:

                if self.is_affine:  # This only works for affine, however.

                    # Not resampling is only possible for translation methods, fail with warning if passed by user
                    if not self.is_translation:
                        if not kwargs["resample"]:
                            raise NotImplementedError(
                                f"Option `resample=False` not supported by {self.__class__},"
                                f" only available for translation coregistrations such as NuthKaab."
                            )

                    # Apply the matrix around the centroid (if defined, otherwise just from the center).
                    transform = kwargs.pop("transform")
                    applied_elev, out_transform = _apply_matrix_rst(
                        dem=kwargs.pop("elev"),
                        transform=transform,
                        matrix=self.to_matrix(),
                        centroid=self._meta["outputs"]["affine"].get("centroid"),
                    )
                else:
                    raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_rst.")

        # If input is a point
        else:
            out_transform = None

            # See if an _apply_pts_func exists
            try:
                applied_elev = self._apply_pts(**kwargs)

            # If it doesn't exist, use apply_matrix()
            except NotImplementedCoregApply:
                if self.is_affine:

                    applied_elev = _apply_matrix_pts(
                        epc=kwargs["elev"],
                        matrix=self.to_matrix(),
                        centroid=self._meta["outputs"]["affine"].get("centroid"),
                        z_name=kwargs.pop("z_name"),
                    )

                else:
                    raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_pts.")

        return applied_elev, out_transform

    def _bin_or_and_fit_nd(  # type: ignore
        self,
        values: NDArrayf,
        bias_vars: None | dict[str, NDArrayf] = None,
        weights: None | NDArrayf = None,
        **kwargs,
    ) -> None:
        """
        Generic binning and/or fitting method to model values along N variables for a coregistration/correction,
        used for all affine and bias-correction subclasses. Expects either 2D arrays for rasters, or 1D arrays for
        points.

        Should only be called through subclassing.
        """

        # Store bias variable names from the dictionary if undefined
        if self._meta["inputs"]["fitorbin"]["bias_var_names"] is None:
            self._meta["inputs"]["fitorbin"]["bias_var_names"] = list(bias_vars.keys())

        # Run the fit or bin, passing the dictionary of parameters
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]
        df, results = _bin_or_and_fit_nd(
            fit_or_bin=self._meta["inputs"]["fitorbin"]["fit_or_bin"],
            params_fit_or_bin=params_fit_or_bin,
            values=values,
            bias_vars=bias_vars,
            weights=weights,
            **kwargs,
        )

        # Initialize output dictionary
        self.meta["outputs"]["fitorbin"] = {}

        # Save results if fitting was performed
        if self._meta["inputs"]["fitorbin"]["fit_or_bin"] in ["fit", "bin_and_fit"] and results is not None:

            # Write the results to metadata in different ways depending on optimizer returns
            if self._meta["inputs"]["fitorbin"]["fit_optimizer"] in (w["optimizer"] for w in fit_workflows.values()):
                params = results[0]
                order_or_freq = results[1]
                if self._meta["inputs"]["fitorbin"]["fit_optimizer"] == robust_norder_polynomial_fit:
                    self._meta["outputs"]["specific"] = {"best_poly_order": order_or_freq}
                else:
                    self._meta["outputs"]["specific"] = {"best_nb_sin_freq": order_or_freq}

            elif self._meta["inputs"]["fitorbin"]["fit_optimizer"] == scipy.optimize.curve_fit:
                params = results[0]
                # Calculation to get the error on parameters (see description of scipy.optimize.curve_fit)
                perr = np.sqrt(np.diag(results[1]))
                self._meta["outputs"]["fitorbin"].update({"fit_perr": perr})

            else:
                params = results[0]

            self._meta["outputs"]["fitorbin"].update({"fit_params": params})

        # Save results of binning if it was performed
        elif self._meta["inputs"]["fitorbin"]["fit_or_bin"] in ["bin", "bin_and_fit"] and df is not None:
            self._meta["outputs"]["fitorbin"].update({"bin_dataframe": df})

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
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

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
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _fit_pts_pts(
        self,
        ref_elev: gpd.GeoDataFrame,
        tba_elev: gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")

    def _apply_pts(
        self,
        elev: gpd.GeoDataFrame,
        z_name: str = "z",
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:

        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")


class CoregPipeline(Coreg):
    """
    A sequential set of co-registration processing steps.
    """

    def __init__(self, pipeline: list[Coreg]) -> None:
        """
        Instantiate a new processing pipeline.

        :param: Processing steps to run in the sequence they are given.
        """
        self.pipeline = pipeline

        super().__init__()

    def __repr__(self) -> str:
        return f"Pipeline: {self.pipeline}"

    @overload
    def info(self, as_str: Literal[False] = ...) -> None: ...

    @overload
    def info(self, as_str: Literal[True]) -> str: ...

    def info(self, as_str: bool = False) -> None | str:
        """Summarize information about this coregistration."""

        # Get the pipeline information for each step as a string
        final_str = []
        for i, step in enumerate(self.pipeline):

            final_str.append(f"Pipeline step {i}:\n" f"################\n")
            step_str = step.info(as_str=True)
            final_str.append(step_str)

        # Return as string or print (default)
        if as_str:
            return "".join(final_str)
        else:
            print("".join(final_str))
            return None

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.deepcopy(value) for key, value in self.__dict__.items() if key != "pipeline"}
        new_coreg.pipeline = [step.copy() for step in self.pipeline]

        return new_coreg

    def _parse_bias_vars(self, step: int, bias_vars: dict[str, NDArrayf] | None) -> dict[str, NDArrayf]:
        """Parse bias variables for a pipeline step requiring them."""

        # Get number of non-affine coregistration requiring bias variables to be passed
        nb_needs_vars = sum(c._needs_vars for c in self.pipeline)

        # Get step object
        coreg = self.pipeline[step]

        # Check that all variable names of this were passed
        var_names = coreg._meta["inputs"]["fitorbin"]["bias_var_names"]

        # Raise error if bias_vars is None
        if bias_vars is None:
            msg = f"No `bias_vars` passed to .fit() for bias correction step {coreg.__class__} of the pipeline."
            if nb_needs_vars > 1:
                msg += (
                    " As you are using several bias correction steps requiring `bias_vars`, don't forget to "
                    "explicitly define their `bias_var_names` during "
                    "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
                )
            raise ValueError(msg)

        # Raise error if no variable were explicitly assigned and there is more than 1 step with bias_vars
        if var_names is None and nb_needs_vars > 1:
            raise ValueError(
                "When using several bias correction steps requiring `bias_vars` in a pipeline,"
                "the `bias_var_names` need to be explicitly defined at each step's "
                "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
            )

        # Raise error if the variables explicitly assigned don't match the ones passed in bias_vars
        if not all(n in bias_vars.keys() for n in var_names):
            raise ValueError(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step {}: {}.".format(coreg.__class__, var_names)
            )

        # Add subset dict for this pipeline step to args of fit and apply
        return {n: bias_vars[n] for n in var_names}

    # Need to override base Coreg method to work on pipeline steps
    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:

        # Check if subsample arguments are different from their default value for any of the coreg steps:
        # get default value in argument spec and "subsample" stored in meta, and compare both are consistent
        argspec = [inspect.getfullargspec(c.__class__) for c in self.pipeline]
        sub_meta = [c.meta["inputs"]["random"]["subsample"] for c in self.pipeline]
        sub_is_default = [
            argspec[i].defaults[argspec[i].args.index("subsample") - 1] == sub_meta[i]  # type: ignore
            for i in range(len(argspec))
        ]
        if subsample is not None and not all(sub_is_default):
            warnings.warn(
                "Subsample argument passed to fit() will override non-default subsample values defined for"
                " individual steps of the pipeline. To silence this warning: only define 'subsample' in "
                "either fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
            )
            # Filter warnings of individual pipelines now that the one above was raised
            warnings.filterwarnings("ignore", message="Subsample argument passed to*", category=UserWarning)

        # Pre-process the inputs, by reprojecting and subsampling, without any subsampling (done in each step)
        ref_dem, tba_dem, inlier_mask, transform, crs, area_or_point = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
        )

        tba_dem_mod = tba_dem.copy()
        out_transform = transform

        for i, coreg in enumerate(self.pipeline):
            logging.debug("Running pipeline step: %d / %d", i + 1, len(self.pipeline))

            main_args_fit = {
                "reference_elev": ref_dem,
                "to_be_aligned_elev": tba_dem_mod,
                "inlier_mask": inlier_mask,
                "transform": out_transform,
                "crs": crs,
                "z_name": z_name,
                "weights": weights,
                "subsample": subsample,
                "random_state": random_state,
            }

            main_args_apply = {"elev": tba_dem_mod, "transform": out_transform, "crs": crs, "z_name": z_name}

            # If non-affine method that expects a bias_vars argument
            if coreg._needs_vars:
                step_bias_vars = self._parse_bias_vars(step=i, bias_vars=bias_vars)

                main_args_fit.update({"bias_vars": step_bias_vars})
                main_args_apply.update({"bias_vars": step_bias_vars})

            # Perform the step fit
            coreg.fit(**main_args_fit)

            # Step apply: one output for a geodataframe, two outputs for array/transform
            # We only run this step if it's not the last, otherwise it is unused!
            if i != (len(self.pipeline) - 1):
                if isinstance(tba_dem_mod, gpd.GeoDataFrame):
                    tba_dem_mod = coreg.apply(**main_args_apply)
                else:
                    tba_dem_mod, out_transform = coreg.apply(**main_args_apply)

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame: ...

    # Need to override base Coreg method to work on pipeline steps
    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:

        # First step and preprocessing
        if not self._fit_called and self._meta["outputs"]["affine"].get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        elev_array, transform, crs = _preprocess_coreg_apply(elev=elev, transform=transform, crs=crs)

        elev_mod = elev_array.copy()
        out_transform = copy.copy(transform)

        # Apply each step of the coregistration
        for i, coreg in enumerate(self.pipeline):

            main_args_apply = {
                "elev": elev_mod,
                "transform": out_transform,
                "crs": crs,
                "z_name": z_name,
                "resample": resample,
                "resampling": resampling,
            }

            # If non-affine method that expects a bias_vars argument
            if coreg._needs_vars:
                step_bias_vars = self._parse_bias_vars(step=i, bias_vars=bias_vars)
                main_args_apply.update({"bias_vars": step_bias_vars})

            # Step apply: one return for a geodataframe, two returns for array/transform
            if isinstance(elev_mod, gpd.GeoDataFrame):
                elev_mod = coreg.apply(**main_args_apply, **kwargs)
            else:
                elev_mod, out_transform = coreg.apply(**main_args_apply, **kwargs)

        # Post-process output depending on input type
        applied_elev, out_transform = _postprocess_coreg_apply(
            elev=elev,
            applied_elev=elev_mod,
            transform=transform,
            out_transform=out_transform,
            crs=crs,
            resample=resample,
            resampling=resampling,
        )

        # Only return object if raster or geodataframe, also return transform if object was an array
        if isinstance(applied_elev, (gu.Raster, gpd.GeoDataFrame)):
            return applied_elev
        else:
            return applied_elev, out_transform

    def __iter__(self) -> Generator[Coreg]:
        """Iterate over the pipeline steps."""
        yield from self.pipeline

    def __add__(self, other: list[Coreg] | Coreg | CoregPipeline) -> CoregPipeline:
        """Append a processing step or a pipeline to the pipeline."""
        if not isinstance(other, Coreg):
            other = list(other)
        else:
            other = [other]

        pipelines = self.pipeline + other

        return CoregPipeline(pipelines)

    def to_matrix(self) -> NDArrayf:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def _to_matrix_func(self) -> NDArrayf:
        """Try to join the coregistration steps to a single transformation matrix."""
        if not _HAS_P3D:
            raise ValueError("Optional dependency needed. Install 'pytransform3d'")

        transform_mgr = TransformManager()

        with warnings.catch_warnings():
            # Deprecation warning from pytransform3d. Let's hope that is fixed in the near future.
            warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
            for i, coreg in enumerate(self.pipeline):
                new_matrix = coreg.to_matrix()

                transform_mgr.add_transform(i, i + 1, new_matrix)

            return transform_mgr.get_transform(0, len(self.pipeline))

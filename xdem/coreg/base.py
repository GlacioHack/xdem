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
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
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
import scipy.optimize
from geoutils import profiler
from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_raster,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils.multiproc import MultiprocConfig
from geoutils.operators.interpolator import Interpolator
from geoutils.operators.reducer import Reducer
from geoutils.pointcloud.dataframe import _get_dataframe_attrs, _set_dataframe_attrs
from geoutils.pointcloud.pointcloud import PointCloudType
from geoutils.raster import Raster, RasterType, raster
from geoutils.raster.referencing import _cast_pixel_interpretation
from geoutils.raster.transformation import _resampling_method_from_str
from geoutils.sampling.support import _raster_from_input

from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.fit import (
    polynomial_1d,
    robust_nfreq_sumsin_fit,
    robust_norder_polynomial_fit,
    sumsin_1d,
)

if TYPE_CHECKING:
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike

    from xdem.coreg.pipeline import CoregPipeline

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
    "fit_pcov": "Covariance of optimized function parameters",
    "bin_dataframe": "Binning output dataframe",
    "max_iterations": "Maximum number of iterations",
    "tolerance_translation": "Tolerance in translation (georeferenced unit)",
    "tolerance_rotation": "Tolerance in rotation (degrees)",
    "tolerance_objective_func": "Tolerance in objective function",
    "trim_residuals": "Residuals are trimmed for outliers",
    "trim_central_statistic": "Central statistic for trimming",
    "trim_spread_statistic": "Spread statistic for trimming",
    "trim_spread_coverage": "Spread coverage for trimming",
    "trim_iterative": "Trimming statistics are updated iteratively until no more value is excluded",
    "last_iteration": "Iteration at which algorithm stopped",
    "iteration_stats": "Statistics compared to tolerances at each iteration",
    "converged": "All stopping tolerances satisfied",
    "terrain_attribute": "Terrain attribute used for correction",
    "angle": "Angle of directional correction",
    "poly_order": "Polynomial order",
    "best_poly_order": "Best polynomial order",
    "best_nb_sin_freq": "Best number of sinusoid frequencies",
    "vshift_reduc_func": "Reduction function used to remove vertical shift",
    "apply_vshift": "Vertical shift activated",
    "min_slope": "Minimum terrain slope (degrees)",
    "centroid": "Centroid found for affine rotation",
    "shift_x": "Eastward shift estimated (georeferenced unit)",
    "shift_y": "Northward shift estimated (georeferenced unit)",
    "shift_z": "Vertical shift estimated (elevation unit)",
    "initial_shift": "Estimated initial shift (georeferenced unit)",
    "matrix": "Affine transformation matrix estimated",
    "only_translation": "Only translations are considered",
    "standardize": "Input data was standardized",
    "icp_method": "Type of ICP method",
    "icp_picky": "Picky ICP variant keeping only closest pairs",
    "linearized": "Rotation linearized during optimization",
    "anisotropic": "Weights axis distances separately",
    "sampling_strategy": "Sampling strategy for point-point registration",
    "iterative_sampling": "Point samples are updated after each optimization step",
    "cpd_weight": "Weight of CPD outlier removal",
    "cpd_lsg": "CPD variant LSG using normals",
    "cpd_estep_knearest": "Number of nearest neighbours used by CPD",
}


###########################################
# Generic functions for preprocessing
###########################################


def _check_coreg_input_families(*values: Any, mp_config: MultiprocConfig | None = None) -> bool | None:
    """
    Check raster and point cloud types without loading their values.

    Return True for Xarray/GeoPandas inputs, False for GeoUtils inputs, or None when neither is supplied.
    """

    import xarray as xr

    # Check elevation, mask and predictor types
    family = None
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            pending.extend(value.values())
            continue
        if isinstance(value, tuple):
            pending.extend(value)
            continue

        # We raise an error if Dask and multiprocessing are passed together
        if mp_config is not None and (is_dask_array(value) or is_dask_dataframe(value)):
            raise ValueError("Cannot use multiprocessing and Dask simultaneously. Remove mp_config for Dask inputs.")
        if isinstance(value, xr.Dataset) or isinstance(getattr(value, "_obj", None), xr.Dataset):
            raise TypeError(
                "Select an elevation DataArray from the Dataset before coregistration or uncertainty analysis."
            )

        # Identify Raster/PointCloud or Xarray/GeoPandas inputs
        interface = _get_raster_interface(value)
        if interface is None:
            interface = _get_pointcloud_interface(value)
        if interface is not None:
            accessor = interface._is_xr or getattr(interface, "_is_pd", False)
            chunked = interface._chunks is not None if _is_raster(interface) else interface._is_dask
            if mp_config is not None and chunked:
                raise ValueError(
                    "Cannot use multiprocessing and Dask simultaneously. Remove mp_config for Dask inputs."
                )
        elif isinstance(value, (xr.DataArray, gpd.GeoDataFrame)) or is_dask_dataframe(value):
            accessor = True
        elif isinstance(value, gu.Vector):
            accessor = False
        else:
            continue

        # Reject mixed GeoUtils and Xarray/GeoPandas inputs
        if family is not None and family != accessor:
            raise TypeError(
                "Cannot mix Raster/PointCloud objects with DataArray/GeoDataFrame inputs in coregistration. "
                "Use one family for all geospatial inputs, including masks and predictors."
            )
        family = accessor
    return family


def _as_coreg_elevation(
    elev: Any,
    transform: affine.Affine | None = None,
    crs: Any = None,
    area_or_point: Literal["Area", "Point"] | None = None,
    z_name: str | None = None,
    as_accessor: bool = False,
) -> RasterBase | PointCloudBase:
    """Select a raster or point cloud interface, creating a raster for array input."""

    import xarray as xr

    # 1/ Check the elevation variable and point column
    # Require a single elevation variable
    if isinstance(elev, xr.Dataset):
        raise TypeError("Select an elevation DataArray from the Dataset before coregistration or uncertainty analysis.")

    # Select the point elevation column
    dataframe_input = isinstance(elev, gpd.GeoDataFrame) or is_dask_dataframe(elev)
    if is_dask_dataframe(elev):
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()
    attributes = _get_dataframe_attrs(elev) if dataframe_input else {}
    selected_column = attributes.get("data_name")
    selected_metadata = "data_name" in attributes
    if (
        dataframe_input
        and z_name is not None
        and not (selected_metadata and (selected_column is None or selected_column in elev.columns))
    ):
        if z_name not in elev:
            raise ValueError(f"'z_name' {z_name} is not a column of the elevation point cloud dataframe.")
        selected = elev.copy(deep=False)
        _set_dataframe_attrs(selected, {**attributes, "data_name": z_name})
        return _get_pointcloud_interface(selected)

    # 2/ Use existing raster or point cloud interfaces
    raster = _get_raster_interface(elev)
    if raster is not None:
        return raster
    points = _get_pointcloud_interface(elev)
    if points is not None:
        return points

    # 3/ Convert arrays to rasters using transform and crs
    if isinstance(elev, np.ndarray) or is_dask_array(elev):
        if transform is None:
            raise ValueError("'transform' must be given if any DEM is array-like.")
        if crs is None:
            raise ValueError("'crs' must be given if any DEM is array-like.")

        # Create a DataArray for Dask arrays or Xarray/GeoPandas inputs
        if is_dask_array(elev) or as_accessor:
            from geoutils.raster.xr_accessor import DataArrayRasterAccessor

            return _get_raster_interface(
                DataArrayRasterAccessor.from_array(
                    elev,
                    transform,
                    crs,
                    nodata=np.nan if np.issubdtype(elev.dtype, np.floating) else None,
                    area_or_point=area_or_point,
                )
            )
        return Raster.from_array(
            elev,
            transform,
            crs,
            area_or_point=area_or_point,
            nodata=np.nan if np.issubdtype(elev.dtype, np.floating) else None,
        )

    raise ValueError(f"Input elevation data should be a raster, array, geodataframe or point cloud, got {type(elev)}.")


def _preprocess_coreg_fit(
    reference_elev: Any,
    to_be_aligned_elev: Any,
    inlier_mask: Any,
    transform: affine.Affine | None,
    crs: Any,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str | None,
    mp_config: MultiprocConfig | None = None,
    bias_vars: Mapping[str, Any] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pre-processing and checks of fit for any input."""

    # 1/ Check raster, point cloud, mask and predictor types
    family = _check_coreg_input_families(
        reference_elev, to_be_aligned_elev, inlier_mask, bias_vars, fit_options, mp_config=mp_config
    )

    # 2/ Convert arrays and select point elevation columns
    # Use either Raster or DataArray for array inputs
    as_accessor = family is True or any(is_dask_array(value) for value in (reference_elev, to_be_aligned_elev))

    # Get raster grids and point elevation columns
    inputs = [
        _as_coreg_elevation(value, transform, crs, area_or_point, z_name, as_accessor=as_accessor)
        for value in (reference_elev, to_be_aligned_elev)
    ]

    # Reject mixed GeoUtils and Xarray/GeoPandas types after array conversion
    _check_coreg_input_families(*inputs, inlier_mask, bias_vars, fit_options, mp_config=mp_config)

    # 3/ Check raster grid metadata and pixel interpretation
    # Warn if rasters override transform or crs
    rasters = [value for value in inputs if _is_raster(value)]
    for name, value in (("reference_dem", reference_elev), ("dem_to_be_aligned", to_be_aligned_elev)):
        if _get_raster_interface(value) is not None:
            for key, supplied in (("transform", transform), ("crs", crs)):
                if supplied is not None:
                    warnings.warn(f"'{name}' of type {type(value)} overrides the given '{key}'")

    # Match raster pixel interpretations
    if len(rasters) == 2:
        interpretation = _cast_pixel_interpretation(rasters[0].area_or_point, rasters[1].area_or_point)
        if rasters[0].area_or_point != rasters[1].area_or_point:
            # Change pixel interpretation without moving raster coordinates
            for index, elevation in enumerate(inputs):
                if elevation.area_or_point != interpretation:
                    adjusted = _get_raster_interface(elevation.copy(deep=False))
                    adjusted.tags = elevation.tags.copy()
                    adjusted.set_area_or_point(interpretation)
                    inputs[index] = adjusted
                    rasters[index] = adjusted
        if area_or_point is not None:
            warnings.warn("Pixel interpretation cast from the two input rasters overrides the given 'area_or_point'.")

    # 4/ Check mask dtype and raster shape
    if inlier_mask is not None:
        mask = _get_raster_interface(inlier_mask)
        if mask is None and (isinstance(inlier_mask, np.ndarray) or is_dask_array(inlier_mask)):
            if inlier_mask.dtype != bool:
                raise AssertionError(f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'")
            if rasters and tuple(inlier_mask.shape) != rasters[0].shape:
                raise ValueError("Input mask array can't be a different size array as input elevation.")
            if rasters and inlier_mask.ndim == 2:
                mask_values = inlier_mask.filled(False) if np.ma.isMaskedArray(inlier_mask) else inlier_mask
                inlier_mask = _raster_from_input(mask_values, rasters[0], "inlier_mask")

    # Return raster and point cloud interfaces for fitting
    return dict(ref_elev=inputs[0], tba_elev=inputs[1], inlier_mask=inlier_mask)


def _preprocess_coreg_apply(
    elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
    transform: affine.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    z_name: str | None = None,
    bias_vars: Mapping[str, Any] | None = None,
    mp_config: MultiprocConfig | None = None,
) -> tuple[RasterBase | PointCloudBase, affine.Affine | None, rio.crs.CRS | None, str | None]:
    """Pre-processing and checks of apply for any input."""

    # 1/ Check raster, point cloud and array metadata
    # Check raster, point cloud and predictor types
    family = _check_coreg_input_families(elev, bias_vars, mp_config=mp_config)

    # 2/ Return raster metadata or the point elevation column
    interface = _as_coreg_elevation(elev, transform=transform, crs=crs, z_name=z_name, as_accessor=family is True)
    if _is_raster(interface):
        # Warn if the raster overrides transform or crs
        if _get_raster_interface(elev) is not None:
            for name, supplied in (("transform", transform), ("crs", crs)):
                if supplied is not None:
                    warnings.warn(f"DEM of type {type(elev)} overrides the given '{name}'")
        return interface, interface.transform, interface.crs, None

    # Return the point cloud CRS and elevation column
    return interface, None, interface.crs, interface.data_name


def _postprocess_coreg_apply(
    elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
    applied_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
    transform: affine.Affine | None,
    out_transform: affine.Affine | None,
    crs: rio.crs.CRS | None,
    resample: bool,
    resampling: str | rio.warp.Resampling | Interpolator | Reducer,
    out_shape: tuple[int, int] | None = None,
) -> RasterLike | PointCloudLike | tuple[NDArrayf | MArrayf, affine.Affine]:
    """Post-processing of apply() for all inputs"""

    # 1/ Format raster or point cloud results
    raster_result = _get_raster_interface(applied_elev)
    if raster_result is not None:
        if isinstance(elev, np.ndarray) or is_dask_array(elev):
            values = raster_result.data if is_dask_array(elev) else raster_result.to_nanarray()
            values = values.reshape(raster_result.shape if out_shape is not None else elev.shape)
            if np.ma.isMaskedArray(elev):
                values = np.ma.masked_invalid(values)
            return values, raster_result.transform
        return applied_elev
    if _get_pointcloud_interface(applied_elev) is not None:
        return applied_elev

    # 2/ Post-process array results from raster corrections
    resampling_method = (
        resampling if isinstance(resampling, rio.warp.Resampling) else _resampling_method_from_str(resampling)
    )
    applied_elev, out_transform = _postprocess_coreg_apply_rst(
        elev, applied_elev, transform, out_transform, crs, resample, resampling_method
    )
    if _get_raster_interface(applied_elev) is not None:
        return applied_elev
    return applied_elev, out_transform


def _postprocess_coreg_apply_rst(
    elev: NDArrayf | RasterLike,
    applied_elev: NDArrayf,
    transform: affine.Affine,
    out_transform: affine.Affine,
    crs: rio.crs.CRS,
    resample: bool,
    resampling: rio.warp.Resampling | None = None,
) -> tuple[NDArrayf | RasterLike, affine.Affine]:
    """
    Post-processing and checks of apply for raster input.

    Here, "elev" and "transform" corresponds to user input, and are required to transform back the output that is
    composed of "applied_elev" and "out_transform".
    """

    # 1/ Use raster nodata and pixel interpretation
    input_raster = _get_raster_interface(elev)
    nodata = input_raster.nodata if input_raster is not None else raster._default_nodata(elev.dtype)
    area_or_point = input_raster.area_or_point if input_raster is not None else None
    constructor = Raster.from_array if input_raster is None else input_raster.from_array
    # Ensure the dtype is OK
    applied_elev = applied_elev.astype("float32")

    # 2/ Reproject onto the original raster grid
    if resample:
        # Reproject the DEM from its out_transform onto the transform
        applied_raster = constructor(applied_elev, out_transform, crs, nodata=nodata, area_or_point=area_or_point)
        original_grid = input_raster
        if original_grid is None:
            original_grid = Raster.from_array(elev, transform, crs, nodata=nodata)
        applied_raster = _get_raster_interface(applied_raster).reproject(
            original_grid, resampling=resampling, silent=True
        )
        applied_elev = _get_raster_interface(applied_raster).data
        # Now that the raster data is reprojected, the new out_transform is set as the original transform
        out_transform = transform

    # 3/ Apply the nodata mask and return the raster or array
    final_mask = np.logical_or(~np.isfinite(applied_elev), applied_elev == nodata)
    # If the DEM was a masked_array, copy the mask to the new DEM
    if input_raster is not None or np.ma.isMaskedArray(elev):
        applied_elev = np.ma.masked_array(applied_elev, mask=final_mask)
    else:
        applied_elev[final_mask] = np.nan
    # If the input was a raster, returns a raster, else returns array and transform
    if input_raster is not None:
        result = constructor(applied_elev, out_transform, crs, nodata=nodata, area_or_point=area_or_point)
        return result, out_transform
    return applied_elev, out_transform


########################################
# Sampling strategy for coregistration
########################################


def _sample_coreg_same_xy(
    first: RasterBase | PointCloudBase,
    second: RasterBase | PointCloudBase,
    mask: Any,
    subsample: int | float,
    random_state: Any,
    auxiliary: Mapping[str, Any],
    auxiliary_at: Literal["self", "other"] | Mapping[str, str],
    mp_config: MultiprocConfig | None,
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf]]:
    """Sample paired elevations and auxiliaries at common X/Y coordinates."""

    # 1/ Sample elevations and auxiliary values, using cosample()
    with ExitStack() as temporary_files:
        config = None if mp_config is None else temporary_files.enter_context(mp_config.temporary())
        sample = first.cosample(
            second,
            auxiliary=auxiliary,
            auxiliary_at=auxiliary_at,
            mask=mask,
            subsample=subsample,
            random_state=random_state,
            align="reproject",
            mp_config=config,
        )

        # 2/ Read the selected coordinates and values
        raster = _get_raster_interface(sample)
        if raster is not None:
            names = raster.tags["long_name"]
            array = raster.to_pointcloud(
                as_array=True,
                force_output_to_memory=True,
                force_pixel_offset="center",
                mp_config=mp_config,
            )
            coordinates = array[:, :2].T
            values = dict(zip(names, array[:, 2:].T))
        else:
            points = _get_pointcloud_interface(sample)
            coordinates = points.to_xyz()[:2]
            values = {name: points._get_column_values(name) for name in points.columns if name != "geometry"}
            if points._is_dask:
                from xdem._misc import import_optional

                coordinates, values = import_optional("dask").compute(coordinates, values)
            values = {name: np.asarray(value) for name, value in values.items()}

        # Keep the coordinate orientation consumed by the existing fitting algorithms
        reference_xyz = np.vstack((*coordinates, values["self"]))
        other_xyz = np.vstack((*coordinates, values["other"]))
        aux_values = {name: values[name] for name in values if name not in {"self", "other"}}
        return reference_xyz, other_xyz, aux_values


def _sample_coreg_independent(
    ref_elev: RasterBase | PointCloudBase,
    tba_elev: RasterBase | PointCloudBase,
    mask: Any,
    subsample: int | float,
    random_state: Any,
    auxiliary: Mapping[str, Any],
    owners: Mapping[str, str],
    mp_config: MultiprocConfig | None,
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf]]:
    """Sample each elevation dataset and its auxiliaries at independent coordinates."""

    # Sample independently for point-point algorithms such as ICP/CPD
    samples = []
    aux_values = {}
    for side, elevation in (("ref", ref_elev), ("tba", tba_elev)):
        selected_auxiliary = {name: value for name, value in auxiliary.items() if owners[name] == side}
        coordinates, _, selected_aux_values = _sample_coreg_same_xy(
            elevation,
            elevation,
            mask,
            subsample,
            random_state,
            selected_auxiliary,
            "self",
            mp_config,
        )
        samples.append(coordinates)
        aux_values.update(selected_aux_values)
    return samples[0], samples[1], aux_values


def _sample_coreg_nearest_xy(
    ref_elev: RasterBase | PointCloudBase,
    tba_elev: RasterBase | PointCloudBase,
    mask: Any,
    subsample: int | float,
    random_state: Any,
    auxiliary: Mapping[str, Any],
    owners: Mapping[str, str],
    mp_config: MultiprocConfig | None,
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf]]:
    """
    Sample xy coordinates and select nearest points in the other dataset (raster or point cloud).

    GeoUtils selects the nearest finite raster cells without interpolating their elevations.
    """

    # 1/ Select the target and its auxiliary values
    # A point cloud defines target coordinates when paired with a raster
    reverse = _is_raster(ref_elev) and not _is_raster(tba_elev)
    target, other = (tba_elev, ref_elev) if reverse else (ref_elev, tba_elev)
    target_side = "tba" if reverse else "ref"
    target_auxiliary = {name: value for name, value in auxiliary.items() if owners[name] == target_side}
    other_auxiliary = {name: value for name, value in auxiliary.items() if owners[name] != target_side}

    # 2/ Sample the target before searching the other raster or point cloud
    target_xyz, _, target_aux_values = _sample_coreg_same_xy(
        target,
        target,
        mask,
        subsample,
        random_state,
        target_auxiliary,
        "self",
        mp_config,
    )

    # 3/ Match the nearest original raster cells or points
    if _is_raster(other):
        with ExitStack() as temporary_files:
            config = None if mp_config is None else temporary_files.enter_context(mp_config.temporary())
            sample = other.cosample(
                other,
                auxiliary=other_auxiliary,
                auxiliary_at="self",
                mask=mask,
                subsample=1,
                random_state=random_state,
                align="reproject",
                mp_config=config,
            )
            raster = _get_raster_interface(sample)

            from geoutils.interface.search import _prediction_search
            from geoutils.operators.neighbours import PointNeighbours

            # Select the nearest finite cells and their coordinates with GeoUtils
            search = _prediction_search(raster, mp_config)
            if raster._chunks is not None:
                search = search.compute()
            elevations, _, _, offsets, selected_coordinates = search.query(
                target_xyz[:2].T, 0, PointNeighbours(max_points=1), OrderedDict(), return_coordinates=True
            )
            if np.any(np.diff(offsets) != 1):
                raise ValueError("No finite raster observations are available for the coregistration sample.")
            selected_coordinates = tuple(selected_coordinates.T)
            del search

            # Read auxiliary values at the selected raster cells
            sampled_values = {"self": elevations, "other": elevations}
            for band, name in enumerate(raster.tags["long_name"][2:], start=3):
                sampled = raster.interp_at_points(
                    selected_coordinates, method="nearest", band=band, as_array=True, mp_config=mp_config
                )
                sampled_values[name] = sampled.compute() if is_dask_array(sampled) else np.asarray(sampled)
            other_xyz = np.vstack((*selected_coordinates, sampled_values["self"]))
            other_aux_values = {name: value for name, value in sampled_values.items() if name not in {"self", "other"}}
    else:
        other_xyz, _, other_aux_values = _sample_coreg_same_xy(
            other,
            other,
            mask,
            1,
            random_state,
            other_auxiliary,
            "self",
            mp_config,
        )

        # Match nearby points without replacing their horizontal coordinates
        tree = scipy.spatial.cKDTree(other_xyz[:2].T)
        indexes = tree.query(target_xyz[:2].T)[1]
        other_xyz = other_xyz[:, indexes]
        other_aux_values = {name: aux[indexes] for name, aux in other_aux_values.items()}

    # 4/ Return samples and auxiliary values in reference and aligned order
    aux_values = {**target_aux_values, **other_aux_values}
    if reverse:
        return other_xyz, target_xyz, aux_values
    return target_xyz, other_xyz, aux_values


def _sample_coreg_inputs(
    ref_elev: RasterBase | PointCloudBase,
    tba_elev: RasterBase | PointCloudBase,
    inlier_mask: Any,
    subsample: int | float = 1,
    random_state: Any = None,
    sampling_strategy: Literal["independent", "same_xy", "nearest_xy"] = "same_xy",
    aux_vars: Mapping[str, Any] | None = None,
    aux_tied_to: str | Mapping[str, str] = "ref",
    mp_config: MultiprocConfig | None = None,
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf] | None]:
    """Sample according to strategy and return 3-by-N coordinate arrays required by coregistration optimizers."""

    # 1/ Check sampling options
    if sampling_strategy not in {"same_xy", "nearest_xy", "independent"}:
        raise ValueError("sampling_strategy must be 'same_xy', 'nearest_xy' or 'independent'.")

    owner_names = aux_tied_to.values() if isinstance(aux_tied_to, Mapping) else (aux_tied_to,)
    if any(owner not in {"ref", "tba"} for owner in owner_names):
        raise ValueError("aux_tied_to must select 'ref' or 'tba' for each auxiliary.")

    # 2/ Pass predictors and their input locations to cosample()
    owners = dict(aux_tied_to) if isinstance(aux_tied_to, Mapping) else dict.fromkeys(aux_vars or {}, aux_tied_to)
    auxiliary = aux_vars or {}

    # 3/ Sample rasters and point clouds depending on sampling strategy
    with warnings.catch_warnings():
        # Silence subsample too large
        warnings.filterwarnings(
            "ignore",
            message=r"Argument ``subsample``.*is larger than the number of valid pixels.*",
            category=UserWarning,
        )
        warnings.filterwarnings("ignore", message="All provided points were outside of raster bounds.*")
        if sampling_strategy == "same_xy":
            reference, aligned, aux_values = _sample_coreg_same_xy(
                ref_elev,
                tba_elev,
                inlier_mask,
                subsample,
                random_state,
                auxiliary,
                {name: "self" if owner == "ref" else "other" for name, owner in owners.items()},
                mp_config,
            )
        elif sampling_strategy == "independent":
            reference, aligned, aux_values = _sample_coreg_independent(
                ref_elev,
                tba_elev,
                inlier_mask,
                subsample,
                random_state,
                auxiliary,
                owners,
                mp_config,
            )
        else:
            reference, aligned, aux_values = _sample_coreg_nearest_xy(
                ref_elev,
                tba_elev,
                inlier_mask,
                subsample,
                random_state,
                auxiliary,
                owners,
                mp_config,
            )
    return reference, aligned, aux_values if aux_vars is not None else None


def _ols_fit(
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None,
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None,
) -> tuple[NDArrayf, None]:
    """
    Solve a linear model with direct OLS via np.linalg.lstsq.

    A design matrix builder can normalize predictor coordinates before constructing poorly scaled columns. If the
    builder defines ``unnormalize_coeffs``, the fitted coefficients are converted back to the original coordinates.

    If design_matrix_func is None, xdata already contains the design matrix of shape (N, P).

    :param design_matrix_func: Callable that takes xdata and returns the design matrix (N, P).
    :param xdata: Predictor data passed to design_matrix_func.
    :param ydata: Observations to fit (N,).
    :param sigma: Optional per-observation standard deviations; used for weighted OLS.

    :returns: Tuple of (coefficients, None), mimicking the (popt, pcov) return of curve_fit.
    """
    # Build the linear system and apply optional inverse-standard-deviation weights
    design_matrix = xdata if design_matrix_func is None else design_matrix_func(xdata)
    if sigma is not None:
        inverse_sigma = 1.0 / sigma
        design_matrix = design_matrix * inverse_sigma[:, np.newaxis]
        ydata = ydata * inverse_sigma

    # Solve the system and restore coefficients from any internal coordinate normalization
    coeffs = np.linalg.lstsq(design_matrix, ydata, rcond=None)[0]
    if design_matrix_func is not None and hasattr(design_matrix_func, "unnormalize_coeffs"):
        coeffs = design_matrix_func.unnormalize_coeffs(coeffs)
    return coeffs, None


def _call_fit_optimizer(
    params_fit_or_bin: InFitOrBinDict,
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None,
    xdata: NDArrayf,
    ydata: NDArrayf,
    sigma: NDArrayf | None,
    **kwargs: Any,
) -> tuple[NDArrayf, Any]:
    """Fit observations with the optimizer selected when the coregistration was initialized."""

    fit_optimizer = params_fit_or_bin["fit_optimizer"]

    # Use the dedicated design matrix for a linear model
    if fit_optimizer == "ols":
        if design_matrix_func is None:
            raise ValueError("The 'ols' fit optimizer requires a design matrix function.")
        return _ols_fit(design_matrix_func, xdata, ydata, sigma)

    # Call user optimizers through the established curve_fit-compatible interface
    return fit_optimizer(
        f=params_fit_or_bin["fit_func"],
        xdata=xdata,
        ydata=ydata,
        sigma=sigma,
        absolute_sigma=True,
        **kwargs,
    )


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None = None,
    **kwargs: Any,
) -> tuple[None, tuple[NDArrayf, Any]]: ...


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["bin"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, None]: ...


@overload
def _bin_or_and_fit_nd(
    fit_or_bin: Literal["bin_and_fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None = None,
    **kwargs: Any,
) -> tuple[pd.DataFrame, tuple[NDArrayf, Any]]: ...


def _bin_or_and_fit_nd(
    fit_or_bin: Literal["fit", "bin", "bin_and_fit"],
    params_fit_or_bin: InFitOrBinDict,
    values: NDArrayf,
    bias_vars: None | dict[str, NDArrayf] = None,
    weights: None | NDArrayf = None,
    design_matrix_func: Callable[[NDArrayf], NDArrayf] | None = None,
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
    :param design_matrix_func: Function that builds the design matrix when the selected optimizer is "ols".
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
            "A number of {} variable(s) has to be provided through the argument 'bias_vars', got {}.".format(
                nd, len(bias_vars)
            )
        )

    # If bias var names were explicitly passed at instantiation, check that they match the one from the dict
    if params_fit_or_bin["bias_var_names"] is not None:
        if not sorted(bias_vars.keys()) == sorted(params_fit_or_bin["bias_var_names"]):
            raise ValueError(
                "The keys of `bias_vars` do not match the `bias_var_names` defined during instantiation: {}.".format(
                    params_fit_or_bin["bias_var_names"]
                )
            )

    # Get number of variables
    nd = len(bias_vars)

    # Remove random_state if the selected optimizer does not accept it
    if fit_or_bin in ["fit", "bin_and_fit"]:
        fit_optimizer = params_fit_or_bin["fit_optimizer"]
        if fit_optimizer == "ols":
            kwargs.pop("random_state", None)
        else:
            fit_func_args = inspect.getfullargspec(fit_optimizer).args
            if "random_state" not in fit_func_args:
                kwargs.pop("random_state", None)

    # We need to sort the bin sizes in the same order as the bias variables if a dict is passed for bin_sizes
    if fit_or_bin in ["bin", "bin_and_fit"]:
        if isinstance(params_fit_or_bin["bin_sizes"], dict):
            var_order = list(bias_vars.keys())
            # Declare type to write integer or tuple to the variable
            bin_sizes: Any = tuple(params_fit_or_bin["bin_sizes"][var] for var in var_order)
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

        xdata = np.array([var.flatten() for var in bias_vars.values()]).squeeze()
        ydata = values.flatten()
        sigma = weights.flatten() if weights is not None else None

        results = _call_fit_optimizer(
            params_fit_or_bin=params_fit_or_bin,
            design_matrix_func=design_matrix_func,
            xdata=xdata,
            ydata=ydata,
            sigma=sigma,
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

        df = gu.stats.stats(
            {"bias": values},
            (params_fit_or_bin["bin_statistic"], "count"),
            by=bias_vars,
            bins=(
                dict.fromkeys(bias_vars, bin_sizes)
                if isinstance(bin_sizes, (int, np.integer))
                else dict(zip(bias_vars, bin_sizes))
            ),
            observed=False,
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

        df = gu.stats.stats(
            {"bias": values},
            (params_fit_or_bin["bin_statistic"], "count"),
            by=bias_vars,
            bins=(
                dict.fromkeys(bias_vars, bin_sizes)
                if isinstance(bin_sizes, (int, np.integer))
                else dict(zip(bias_vars, bin_sizes))
            ),
            observed=False,
        )

        # Now, we need to pass this new data to the fitting function and optimizer
        # Read the predictor centres and bias statistic from GeoUtils labelled groups
        new_vars = [pd.IntervalIndex(df.index.get_level_values(name)).mid.values for name in bias_vars]
        new_diff = df[("bias", params_fit_or_bin["bin_statistic"].__name__)].values
        # TODO: pass a new sigma based on "count" and original sigma (and correlation?)?
        #  sigma values would have to be binned above also

        # Valid values for the binning output
        ind_valid = np.logical_and.reduce((np.isfinite(new_diff), *(np.isfinite(var) for var in new_vars)))

        if np.all(~ind_valid):
            raise ValueError("Only NaN values after binning, did you pass the right bin edges?")

        xdata_bin = np.array([var[ind_valid].flatten() for var in new_vars]).squeeze()
        ydata_bin = new_diff[ind_valid].flatten()
        sigma_bin = weights[ind_valid].flatten() if weights is not None else None

        results = _call_fit_optimizer(
            params_fit_or_bin=params_fit_or_bin,
            design_matrix_func=design_matrix_func,
            xdata=xdata_bin,
            ydata=ydata_bin,
            sigma=sigma_bin,
            **kwargs,
        )
    logging.debug("%dD bias estimated.", nd)

    return df, results


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
    fit_optimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["ols"]

    # For a minimization problem
    fit_minimizer: Callable[..., Any] | Literal["ols"]
    fit_loss_func: Callable[[NDArrayf], np.floating[Any]]

    # Bin parameters: bin sizes, statistic and apply method
    bin_sizes: int | dict[str, int | Iterable[float]]
    bin_statistic: Callable[[NDArrayf], np.floating[Any]]
    bin_apply_method: Literal["linear", "per_bin"]
    # Name of variables, and number of dimensions
    bias_var_names: list[str]
    nd: int | None

    # Whether to trim residuals, and related parameters
    trim_residuals: bool
    # Trimming statistics and coverage (always symmetric)
    trim_central_statistic: Callable[[NDArrayf], np.floating[Any]]
    trim_spread_statistic: Callable[[NDArrayf], np.floating[Any]]
    trim_spread_coverage: float
    # Whether to trim the residuals iteratively until trimming is over
    trim_iterative: bool


class OutFitOrBinDict(TypedDict, total=False):
    """Keys and types of outputs associated with binning and/or fitting."""

    # Optimized parameters for fitted function, and its error
    fit_params: NDArrayf
    fit_perr: NDArrayf
    fit_pcov: NDArrayf
    # Binning dataframe
    bin_dataframe: pd.DataFrame


class InIterativeDict(TypedDict, total=False):
    """Keys and types of inputs associated with iterative methods."""

    # Maximum number of iterations
    max_iterations: int
    # Tolerance at which to stop algorithm (unit specified in method)
    tolerance_translation: float | None
    tolerance_rotation: float | None
    tolerance_objective_func: float | None


class OutIterativeDict(TypedDict, total=False):
    """Keys and types of outputs associated with iterative methods."""

    # Iteration at which the algorithm stopped
    last_iteration: int
    # Statistics (often offsets) compared to tolerances at each iteration
    iteration_stats: pd.DataFrame
    # Whether the final iteration satisfied all stopping tolerances
    converged: bool


class InSpecificDict(TypedDict, total=False):
    """Keys and types of inputs associated with specific methods."""

    # (Using TerrainBias) Selected terrain attribute
    terrain_attribute: str
    # (Using DirectionalBias) Angle for directional correction
    angle: float
    # (Using Deramp) Polynomial order selected for deramping
    poly_order: int

    # (Using NuthKaab) Minimum terrain slope in degrees
    min_slope: float

    # (Using ICP) Method type to compute 3D distances
    icp_method: Literal["point-to-point", "point-to-plane"]
    # (Using ICP) Picky selection of closest pairs
    icp_picky: bool

    # (Using CPD) Weight for outlier removal
    cpd_weight: float
    # (Using CPD) Use local surface geometry method
    cpd_lsg: bool
    # (Using CPD) Use kNN to speed up E-step
    cpd_estep_knearest: int

    # (Using ICP or CPD or other point-point registration)
    sampling_strategy: Literal["independent", "same_xy", "nearest_xy"]
    # (Using ICP or CPD) Update point samples after each optimization step
    iterative_sampling: bool


class OutSpecificDict(TypedDict, total=False):
    """Keys and types of outputs associated with specific methods."""

    # (Using multi-order polynomial fit) Best performing polynomial order
    best_poly_order: int
    # (Using multi-frequency sum of sinusoids fit) Best performing number of frequencies
    best_nb_sin_freq: int


class InAffineDict(TypedDict, total=False):
    """Keys and types of inputs associated with affine methods."""

    # Estimated initial shift (z currently always equal to zero)
    initial_shift: tuple[float, float, float] | None
    # Vertical shift reduction function for methods focusing on translation coregistration
    vshift_reduc_func: Callable[[NDArrayf], np.floating[Any]]
    # Vertical shift activated
    apply_vshift: bool
    # Apply coregistration method only for translations
    only_translation: bool
    # Standardize input data for numerics
    standardize: bool
    # Linearize the rotation during optimization (if available)
    linearized: bool
    # Assign anisotropic weights based on X/Y/Z axis to account for unit differences
    anisotropic: Literal["xy_vs_z", "per_axis"] | None


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
    _fit_linear: bool = False
    _design_matrix_func: Callable[[NDArrayf], NDArrayf] | None = None
    _meta: CoregDict

    def __init__(self, meta: dict[str, Any] | None = None) -> None:
        """Instantiate a generic processing step method."""

        # Automatically sort input keys into their appropriate nested level using only the TypedDicts defined
        # above which make up the CoregDict altogether
        dict_meta = CoregDict(inputs={}, outputs={})
        if meta is not None:
            meta = meta.copy()

            # Resolve the default once so metadata and info() expose the optimizer that will run
            if (
                meta.get("fit_or_bin") in ["fit", "bin_and_fit"]
                and "fit_optimizer" in meta
                and meta["fit_optimizer"] is None
            ):
                meta["fit_optimizer"] = "ols" if self._fit_linear else scipy.optimize.curve_fit

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
                        f"Coregistration metadata key {k} is not supported. Should be one of {', '.join(all_keys)}"
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

        from xdem.coreg.pipeline import (
            CoregPipeline,  # Local import to avoid circularity
        )

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
                if np.isfinite(val) and val != 0:
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
                    (k, v)
                    for k, v in self._meta["inputs"][lk].items()  # type: ignore
                    if k in existing_deep_keys
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
                        (k, v)
                        for k, v in self._meta["outputs"][lk].items()  # type: ignore
                        if k in existing_deep_keys
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

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str | None = None,
        random_state: int | np.random.Generator | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> CoregType:  # type: ignore
        """
        Estimate the coregistration transform on the given DEMs.

        :param reference_elev: Reference elevation, either a DEM or an elevation point cloud.
        :param to_be_aligned_elev: To-be-aligned elevation, either a DEM or an elevation point cloud.
        :param inlier_mask: Raster or boolean array of areas to include (inliers=True).
        :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
        :param weights: Array of weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
            A value of 1 selects all valid observations.
        :param transform: Transform of the reference elevation, only if provided as 2D array.
        :param crs: CRS of the reference elevation, only if provided as 2D array.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling).
        :param mp_config: Worker and tile settings for multiprocessing preprocessing.
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
        # TODO: Always save a random state?
        # if self._meta["inputs"]["random"]["subsample"] != 1:
        self._meta["inputs"]["random"]["random_state"] = random_state

        # Pre-process the inputs, by checking raster and point cloud metadata
        # Get main arguments
        main_args = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            mp_config=mp_config,
            bias_vars=bias_vars,
            fit_options=kwargs,
        )

        # Apply the shift to the source dem if given
        initial_shift_apply = False
        if self._meta["inputs"]["affine"].get("initial_shift") is not None:
            shift_x = self._meta["inputs"]["affine"]["initial_shift"][0]  # type: ignore
            shift_y = self._meta["inputs"]["affine"]["initial_shift"][1]  # type: ignore

            # shift_z is currently always equal to zero
            reference = main_args["ref_elev"]
            shifted = reference.translate(-shift_x, -shift_y)
            if _is_raster(reference):
                main_args["ref_elev"] = _get_raster_interface(shifted)
                if isinstance(inlier_mask, np.ndarray) or is_dask_array(inlier_mask):
                    shifted_mask = main_args["inlier_mask"].translate(-shift_x, -shift_y)
                    main_args["inlier_mask"] = _get_raster_interface(shifted_mask)
            else:
                main_args["ref_elev"] = _get_pointcloud_interface(shifted)
            initial_shift_apply = True

        # Pass predictors to sampling without reading their values
        if bias_vars is not None:
            # Check if the current class actually requires bias_vars
            if self._is_affine:
                warnings.warn("This coregistration method is affine, ignoring `bias_vars` passed to fit().")
            else:
                main_args.update({"bias_vars": bias_vars})

        # Run the associated fitting function, which has fallback logic for "raster-raster", "raster-point" or
        # "point-point" depending on what is available for a certain Coreg function
        self._fit_func(
            **main_args,
            mp_config=mp_config,
            **kwargs,
        )

        # Unapply the shift to the source dem if apply before
        if initial_shift_apply and "outputs" in self.meta and "affine" in self.meta["outputs"]:
            if "matrix" in self.meta["outputs"]["affine"]:
                matrix = self.meta["outputs"]["affine"]["matrix"]
                matrix[:2, 3] += (shift_x, shift_y)
            if "shift_x" in self.meta["outputs"]["affine"]:
                self.meta["outputs"]["affine"]["shift_x"] += shift_x
                logging.debug(f"Updated shift_x by {shift_x} in {self}")
            if "shift_y" in self.meta["outputs"]["affine"]:
                self.meta["outputs"]["affine"]["shift_y"] += shift_y
                logging.debug(f"Updated shift_y by {shift_y} in {self}")

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame | PointCloudType,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | PointCloudType: ...

    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame | PointCloudType,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str | None = None,
        **kwargs: Any,
    ) -> (
        RasterType
        | gpd.GeoDataFrame
        | PointCloudType
        | tuple[NDArrayf, rio.transform.Affine]
        | tuple[MArrayf, rio.transform.Affine]
    ):
        """
        Apply the estimated transform to a DEM.

        :param elev: Elevation to apply the transform to, either a DEM or an elevation point cloud.
        :param bias_vars: Only for some bias correction classes. 2D array of bias variables used.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
            only the transform might be updated and no resampling is done.
        :param resampling: Resampling method if resample is used. Affine methods also accept GeoUtils gridding
            names and Interpolator/Reducer objects. Defaults to "bilinear".
        :param transform: Geotransform of the elevation, only if provided as 2D array.
        :param crs: CRS of elevation, only if provided as 2D array.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param kwargs: Any optional arguments to be passed to either self._apply_rst or apply_matrix. Affine raster
            calls accept out_transform and out_shape to select the destination grid, and gridding_options
            to configure GeoUtils grid().

        :returns: The transformed DEM.
        """
        if not self._fit_called and self._meta["outputs"]["affine"].get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        spatial, transform, crs, z_name = _preprocess_coreg_apply(
            elev, transform, crs, z_name, bias_vars=bias_vars, mp_config=kwargs.get("mp_config")
        )

        # Apply through the subclass implementation
        main_args: dict[str, Any] = dict(
            elev=spatial, transform=transform, crs=crs, resample=resample, resampling=resampling
        )
        if bias_vars is not None:
            main_args["bias_vars"] = bias_vars
        applied_elev, out_transform = self._apply(**main_args, **kwargs)

        # Post-process output depending on input type
        return _postprocess_coreg_apply(
            elev,
            applied_elev,
            transform,
            out_transform,
            crs,
            resample,
            resampling,
            out_shape=kwargs.get("out_shape"),
        )

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        to_be_aligned_elev: MArrayf,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> tuple[MArrayf, rio.transform.Affine]: ...

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        to_be_aligned_elev: NDArrayf,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> tuple[NDArrayf, rio.transform.Affine]: ...

    @overload
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        to_be_aligned_elev: RasterType | gpd.GeoDataFrame | PointCloudType,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType | gpd.GeoDataFrame: ...

    @profiler.profile("xdem.coreg.base.fit_and_apply", memprof=True)
    def fit_and_apply(
        self,
        reference_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterLike | PointCloudLike,
        inlier_mask: NDArrayb | Raster | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        area_or_point: Literal["Area", "Point"] | None = None,
        z_name: str = "z",
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        random_state: int | np.random.Generator | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        apply_kwargs: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:
        """Estimate and apply the coregistration to a pair of elevation data.

        :param reference_elev: Reference elevation, either a DEM or an elevation point cloud.
        :param to_be_aligned_elev: To-be-aligned elevation, either a DEM or an elevation point cloud.
        :param inlier_mask: Raster or boolean array of areas to include (inliers=True).
        :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
        :param weights: Array of weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
            A value of 1 selects all valid observations.
        :param transform: Transform of the reference elevation, only if provided as 2D array.
        :param crs: CRS of the reference elevation, only if provided as 2D array.
        :param area_or_point: Pixel interpretation of the DEMs, only if provided as 2D arrays.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
            only the transform might be updated and no resampling is done.
        :param resampling: Resampling method if resample is used. Affine methods also accept GeoUtils gridding
            names and Interpolator/Reducer objects. Defaults to "bilinear".
        :param random_state: Random state or seed number to use for calculations (to fix random sampling).
        :param fit_kwargs: Keyword arguments to be passed to fit.
        :param apply_kwargs: Keyword argument to be passed to apply.
        :param mp_config: Worker and tile settings for fitting and applying the transformation.
        """

        if fit_kwargs is None:
            fit_kwargs = {}
        if apply_kwargs is None:
            apply_kwargs = {}

        if mp_config is not None:
            fit_kwargs = {"mp_config": mp_config, **fit_kwargs}
            apply_kwargs = {"mp_config": mp_config, **apply_kwargs}

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

    def _fit_func(
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        **kwargs: Any,
    ) -> None:
        """
        Distribute to _fit_rst_rst, fit_rst_pts or fit_pts_pts depending on input and method availability.
        Needs to be _fit_func of the main class to simplify calls from CoregPipeline and BlockwiseCoreg.
        """

        reference_is_raster, aligned_is_raster = _is_raster(ref_elev), _is_raster(tba_elev)
        kwargs.update(ref_elev=ref_elev, tba_elev=tba_elev, inlier_mask=inlier_mask)

        # Share the same dispatch and resource lifetime across affine methods and bias corrections
        with ExitStack() as temporary_files:
            kwargs["temporary_files"] = temporary_files
            if reference_is_raster and aligned_is_raster:
                self._fit_rst_rst(**kwargs)
            elif not reference_is_raster and not aligned_is_raster:
                self._fit_pts_pts(**kwargs)
            else:
                self._fit_rst_pts(**kwargs)

    def _apply(
        self,
        elev: RasterBase | PointCloudBase,
        resample: bool = True,
        resampling: str | rio.warp.Resampling | Interpolator | Reducer = "bilinear",
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike | PointCloudLike, affine.Affine | None]:
        """Distribute to _apply_rst and _apply_pts based on input and method availability."""

        # Leave resampling to apply() when it formats non-affine raster outputs
        if _is_raster(elev):
            try:
                return self._apply_rst(elev=elev, resample=resample, **kwargs)
            except NotImplementedCoregApply:
                raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_rst.")

        # Apply the point correction without raster post-processing
        try:
            applied_elev = self._apply_pts(elev=elev, resample=resample, **kwargs)
        except NotImplementedCoregApply:
            raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_pts.")
        return applied_elev, None

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
            design_matrix_func=self._design_matrix_func,
            **kwargs,
        )

        # Initialize output dictionary
        self.meta["outputs"]["fitorbin"] = {}

        # Save results if fitting was performed
        if self._meta["inputs"]["fitorbin"]["fit_or_bin"] in ["fit", "bin_and_fit"] and results is not None:
            # Write the results to metadata in different ways depending on optimizer returns
            fit_optimizer = self._meta["inputs"]["fitorbin"]["fit_optimizer"]
            if callable(fit_optimizer) and fit_optimizer in (w["optimizer"] for w in fit_workflows.values()):
                params = results[0]
                order_or_freq = results[1]
                if fit_optimizer == robust_norder_polynomial_fit:
                    self._meta["outputs"]["specific"] = {"best_poly_order": order_or_freq}
                else:
                    self._meta["outputs"]["specific"] = {"best_nb_sin_freq": order_or_freq}

            elif fit_optimizer == scipy.optimize.curve_fit:
                params = results[0]
                # Calculate parameter errors from the covariance returned by scipy.optimize.curve_fit
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
        ref_elev: RasterBase,
        tba_elev: RasterBase,
        inlier_mask: Any,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _fit_rst_pts(
        self,
        ref_elev: RasterBase | PointCloudBase,
        tba_elev: RasterBase | PointCloudBase,
        inlier_mask: Any,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _fit_pts_pts(
        self,
        ref_elev: PointCloudBase,
        tba_elev: PointCloudBase,
        inlier_mask: Any,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _apply_rst(
        self,
        elev: RasterBase,
        bias_vars: dict[str, Any] | None = None,
        mp_config: MultiprocConfig | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf | RasterLike, rio.transform.Affine]:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")

    def _apply_pts(
        self,
        elev: PointCloudBase,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> PointCloudLike:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")

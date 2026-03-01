# Copyright (c) 2025 xDEM developers
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

"""This module contains helper for 'co-sampling', i.e. sampling multiple geospatial data (rasters or point clouds)
 at the same locations, optionally with random subsampling and vector masking."""

from __future__ import annotations

import logging
from typing import Literal, Any

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio as rio

import geoutils as gu

from geoutils.raster.georeferencing import _coords
from geoutils.interface.interpolate import _interp_points
from xdem.coreg.base import _reproject_horizontal_shift_samecrs
from geoutils import Raster, PointCloud

from xdem._typing import NDArrayf, NDArrayb

def _get_subsample_on_valid_mask(subsample: float | int,
                                 random_state: int | None | np.random.RandomState,
                                 valid_mask: NDArrayb) -> NDArrayb:
    """
    Get mask of values to subsample on valid mask (works for both 1D or 2D arrays).

    :param valid_mask: Raster of valid values (inlier and not nodata).
    """

    # This should never happen
    if subsample is None:
        raise ValueError("Subsample should have been defined in metadata before reaching this class method.")

    # If valid mask is empty
    if np.count_nonzero(valid_mask) == 0:
        raise ValueError(
            "There is no valid points common to the input and auxiliary data (bias variables, or "
            "derivatives required for this method, for example slope, aspect, etc)."
        )

    # If subsample is not equal to one, subsampling should be performed.
    elif subsample != 1.0:

        # Build a low memory masked array with invalid values masked to pass to subsampling
        ma_valid = np.ma.masked_array(data=np.ones(np.shape(valid_mask), dtype=bool), mask=~valid_mask)
        # Take a subsample within the valid values
        indices = gu.stats.sampling.subsample_array(
            ma_valid,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
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
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,  # Never None thanks to Coreg.fit() pre-process
    z_name: str,
    area_or_point: Literal["Area", "Point"] | None,
    subsample: float | int = 1,
    random_state: int | np.random.RandomState | None = None,
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
        sub_mask = _get_subsample_on_valid_mask(subsample=subsample, random_state=random_state, valid_mask=valid_mask)

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
        sub_mask = _get_subsample_on_valid_mask(subsample=subsample, random_state=random_state, valid_mask=valid_mask)

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


def _subsample_pts_rst_independent(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine | None,
    tba_transform: rio.transform.Affine | None,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    subsample: int | float = 1,
    random_state: float | np.random.RandomState | None = None,
    raster_to_point: Literal["on_grid", "off_grid"] = "off_grid",
    aux_vars: None | dict[str, NDArrayf] = None,
    aux_tied_to: Literal["ref", "tba"] = "ref",
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf] | None]:
    """
    Subsample raster-raster, raster-point or point-point independently into two point clouds.

    Each subsampling respects the valid values of each input and (optionally) of auxiliary variables tied to one of
    the inputs.
    """

    # 1/ Reference elevation subsampling

    # If reference is a point cloud
    if isinstance(ref_elev, gpd.GeoDataFrame):
        # Subsample geodataframe from only valid Z values
        sub_mask = _get_subsample_on_valid_mask(
            subsample=subsample, random_state=random_state, valid_mask=np.isfinite(ref_elev[z_name].values)
        )
        sub_ref = ref_elev[sub_mask]
        # Convert to Nx3 array
        sub_ref = np.vstack((sub_ref.geometry.x.values, sub_ref.geometry.y.values, sub_ref[z_name].values))
        sub_aux_ref = None

    # Or if it is a raster
    else:
        # We can use the _get_subsample_mask_pts_rst with a placeholder
        placeholder_tba = np.ones(ref_elev.shape, dtype=bool)
        # If auxiliary variables are tied to reference, pass them here
        aux_vars_ref = aux_vars if aux_tied_to == "ref" else None
        sub_mask = _get_subsample_mask_pts_rst(
            subsample=subsample,
            random_state=random_state,
            ref_elev=ref_elev,
            tba_elev=placeholder_tba,
            inlier_mask=inlier_mask,
            transform=ref_transform,
            z_name=z_name,
            area_or_point=area_or_point,
            aux_vars=aux_vars_ref,
        )
        sub_ref, _, sub_aux_ref, sub_coords = _subsample_on_mask(
            ref_elev=ref_elev,
            tba_elev=placeholder_tba,
            aux_vars=aux_vars_ref,
            sub_mask=sub_mask,
            transform=ref_transform,
            area_or_point=area_or_point,
            z_name=z_name,
            return_coords=True,
        )
        # Convert to Nx3 array
        sub_ref = np.vstack((sub_coords[0], sub_coords[1], sub_ref))

    # 2/ To-be-aligned elevation subsampling, independently of reference

    # If to-be-aligned is a point cloud
    if isinstance(tba_elev, gpd.GeoDataFrame):
        # Subsample geodataframe from only valid Z values
        sub_mask = _get_subsample_on_valid_mask(
            subsample=subsample, random_state=random_state, valid_mask=np.isfinite(tba_elev[z_name].values)
        )
        sub_tba = tba_elev[sub_mask]
        # Convert to Nx3 array
        sub_tba = np.vstack((sub_tba.geometry.x.values, sub_tba.geometry.y.values, sub_tba[z_name].values))
        sub_aux_tba = None

    # Or if it is a raster
    else:
        # We can use the _get_subsample_mask_pts_rst with a placeholder
        placeholder_ref = np.ones(tba_elev.shape, dtype=bool)
        # If auxiliary variables are tied to to-be-aligned, pass them here
        aux_vars_tba = aux_vars if aux_tied_to == "tba" else None
        sub_mask = _get_subsample_mask_pts_rst(
            subsample=subsample,
            random_state=random_state,
            ref_elev=placeholder_ref,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=tba_transform,
            z_name=z_name,
            area_or_point=area_or_point,
            aux_vars=aux_vars_tba,
        )
        _, sub_tba, sub_aux_tba, sub_coords = _subsample_on_mask(
            ref_elev=placeholder_ref,
            tba_elev=tba_elev,
            aux_vars=aux_vars_tba,
            sub_mask=sub_mask,
            transform=tba_transform,
            area_or_point=area_or_point,
            z_name=z_name,
            return_coords=True,
        )
        # Convert to Nx3 array
        sub_tba = np.vstack((sub_coords[0], sub_coords[1], sub_tba))

    # Retrieve subsampled auxiliary if tied
    if aux_tied_to == "ref":
        sub_aux = sub_aux_ref
    else:
        sub_aux = sub_aux_tba

    # If lengths differ, cut to same? Not mandatory
    # min_samp = min(sub_tba.shape[1], sub_ref.shape[1])
    # print(min_samp)
    # sub_ref = sub_ref[:, :min_samp]
    # sub_tba = sub_tba[:, :min_samp]
    # if sub_aux is not None:
    #     sub_aux = {k: v[:min_samp] for k, v in sub_aux.items()}

    return sub_ref, sub_tba, sub_aux


def _subsample_pts_rst_same_xy(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine | None,
    tba_transform: rio.transform.Affine | None,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    subsample: float | int = 1,
    random_state: int | np.random.RandomState | None = None,
    raster_to_point: Literal["on_grid", "off_grid"] = "off_grid",
    aux_vars: None | dict[str, NDArrayf] = None,
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf] | None]:
    """
    Subsample raster-raster or raster-point into two point clouds at same X/Y coordinantes.

    The common subsampling respects the valid values of both input and (optionally) of auxiliary variables.
    """

    # This sampling strategy cannot work for two elevation point clouds
    if isinstance(ref_elev, gpd.GeoDataFrame) and isinstance(tba_elev, gpd.GeoDataFrame):
        raise ValueError(
            "Sampling strategy 'same_xy' is only available if at least one of the two elevation datasets inputs is a raster."
        )

    # If both are rasters, the two grids should be projected on each other
    if isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):
        tba_elev = _reproject_horizontal_shift_samecrs(
            raster_arr=tba_elev, src_transform=tba_transform, dst_transform=ref_transform
        )
        transform = ref_transform
    else:
        transform = ref_transform if isinstance(ref_elev, np.ndarray) else tba_transform

    # Get subsample mask (a 2D array for raster-raster, a 1D array of length the point data for point-raster)
    sub_mask = _get_subsample_mask_pts_rst(
        subsample=subsample,
        random_state=random_state,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
        aux_vars=aux_vars,
    )

    # Perform subsampling on mask for all inputs
    sub_ref, sub_tba, sub_aux, sub_coords = _subsample_on_mask(
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        aux_vars=aux_vars,
        sub_mask=sub_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
        return_coords=True,
    )

    # Convert to Nx3 arrays
    sub_ref = np.vstack((sub_coords[0], sub_coords[1], sub_ref))
    sub_tba = np.vstack((sub_coords[0], sub_coords[1], sub_tba))

    return sub_ref, sub_tba, sub_aux


def _subsample_rst_pts(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine | None,
    tba_transform: rio.transform.Affine | None,
    crs: rio.crs.CRS,  # Never None thanks to Coreg.fit() pre-process
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    subsample: float | int = 1,
    random_state: int | np.random.RandomState | None = None,
    sampling_strategy: Literal["independent", "same_xy"] = "same_xy",
    raster_to_point: Literal["on_grid", "off_grid"] = "off_grid",  # TODO: IMPLEMENT THIS
    aux_vars: None | dict[str, NDArrayf] = None,
    aux_tied_to: Literal["ref", "tba"] = "ref",
) -> tuple[NDArrayf, NDArrayf, dict[str, NDArrayf] | None]:
    """
    Pre-process raster-raster, point-raster or point-point datasets into two random point subsamples.

    Additionally, simultaneously subsample auxiliary variables (e.g., normals, gradient) tied to one of the inputs.
    This ensures that all subsampled values are valid, both for main and auxiliary data.

    Different sampling strategies:
        - "independent": Each point subsample is drawn independently of the other dataset,
        - "same_xy": Each point subsample is drawn at the same X/Y coordinates as the other dataset (where both dataset have valid values).

    Different raster-to-point conversion strategies:
        - "on_grid": Each raster is converted to point using only coordinates on its regular grid.
        - "off_grid": Each raster is converted to point using any point coordinate within the valid extent of the raster (interpolated).

    :param subsample: Subsample fraction.
    :param random_state: Random seed/state.
    :param ref_elev: Reference elevation data.
    :param tba_elev: To-be-aligned elevation data.
    :param inlier_mask: Inlier mask data.
    :param ref_transform: Geotransform of reference data.
    :param tba_transform: Geotransform of to-be-aligned data.
    :param crs: Coordinate reference system.
    :param area_or_point: Pixel interpretation of raster data.
    :param z_name: Name of elevation point cloud column.
    :param sampling_strategy: Sampling strategy for random subsampling of point-raster data, either "independent"
        to sample different points between the two datasets , or "same_xy" to sample at the same X/Y coordinates.
    :param raster_to_point: Conversion from raster to point, either "on_grid" to use only points at regular grid
        coordinates or "off_grid" to use point at any interpolated coordinates.
    :param aux_vars: Auxiliary variables.
    :param aux_tied_to: What input data the auxiliary variables are tied to (only relevant for "independent" sampling).

    :returns: Nx3 array of subsampled reference elevation points, Nx3 array of subsampled to-be-aligned elevation
        points and dictionary of 1D arrays of auxiliary variables values subsampled at the same points as either
        reference or to-be-aligned.
    """

    # If sampling is done at independent coordinates
    if sampling_strategy == "independent":

        # This function requires an additional "aux_tied_to" argument to know how to deal with subsampling of auxiliary variables
        sub_ref, sub_tba, sub_aux = _subsample_pts_rst_independent(
            subsample=subsample,
            random_state=random_state,
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            aux_vars=aux_vars,
            aux_tied_to=aux_tied_to,
            area_or_point=area_or_point,
            z_name=z_name,
            raster_to_point=raster_to_point,
        )

    # If sampling is done at the same X/Y coordinates
    else:

        sub_ref, sub_tba, sub_aux = _subsample_pts_rst_same_xy(
            subsample=subsample,
            random_state=random_state,
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            aux_vars=aux_vars,
            area_or_point=area_or_point,
            z_name=z_name,
            raster_to_point=raster_to_point,
        )

    # Replace everything to NaN arrays
    if np.ma.isMaskedArray(sub_ref):
        sub_ref = sub_ref.filled()
    if np.ma.isMaskedArray(sub_tba):
        sub_tba = sub_tba.filled()
    if sub_aux is not None:
        for k, v in sub_aux.items():
            if np.ma.isMaskedArray(v):
                sub_aux.update({k: v.filled()})

    # Return two geodataframes of subsampled points
    return sub_ref, sub_tba, sub_aux

def _is_dask_array(x: object) -> bool:
    # Avoid importing dask; rely on duck-typing.
    return hasattr(x, "compute") and hasattr(x, "dtype") and hasattr(x, "shape")

def _compute_if_dask(x: NDArrayf | NDArrayb) -> NDArrayf | NDArrayb:
    return x.compute() if _is_dask_array(x) else x

def _ndim(x: object) -> int:
    if hasattr(x, "ndim"):
        return int(getattr(x, "ndim"))
    return int(np.asarray(x).ndim)

def _shape(x: object) -> tuple[int, ...]:
    if hasattr(x, "shape"):
        return tuple(getattr(x, "shape"))
    return tuple(np.asarray(x).shape)

def _normalize_aux(aux: object) -> dict[str, NDArrayf | gu.Raster | gpd.GeoDataFrame]:
    if aux is None:
        return {}
    if isinstance(aux, dict):
        return aux
    return {f"var{i+1}": v for i, v in enumerate(aux)}

def _as_raster_array_and_meta(
    obj: NDArrayf | gu.Raster,
    *,
    transform_fallback: rio.transform.Affine | None,
    crs_fallback: rio.crs.CRS | None,
    aop_fallback: Literal["Area", "Point"] | None,
) -> tuple[NDArrayf, rio.transform.Affine, rio.crs.CRS, Literal["Area", "Point"] | None]:
    if isinstance(obj, gu.Raster):
        arr = obj.data.filled(np.nan) if np.ma.isMaskedArray(obj.data) else obj.data
        return arr, obj.transform, obj.crs, obj.area_or_point
    if transform_fallback is None or crs_fallback is None:
        raise ValueError("'transform' and 'crs' must be provided when passing raster data as ndarray.")
    return obj, transform_fallback, crs_fallback, aop_fallback  # type: ignore[return-value]

def _as_point_gdf(obj: gpd.GeoDataFrame | PointCloud, *, z_name: str) -> tuple[gpd.GeoDataFrame, str]:
    """
    Normalize point cloud-like inputs to a GeoDataFrame.

    :param obj: Point data as GeoDataFrame or EPC.
    :param z_name: Elevation/value column name (only used for GeoDataFrame validation).
    :return: GeoDataFrame view of the point cloud.
    """
    if isinstance(obj, PointCloud):
        gdf = obj.ds
        z_name = obj.data_column
        # EPC defines its own data column; cosample() should use that column for values.
        # We do NOT rename columns here; we just return the GeoDataFrame.
        return gdf, z_name

    # GeoDataFrame input: validate z_name exists
    if z_name not in obj.columns:
        raise ValueError(f"Point GeoDataFrame must contain column '{z_name}'.")
    return obj, z_name

def _ensure_same_points(aux_gdf: gpd.GeoDataFrame, main_gdf: gpd.GeoDataFrame, *, name: str) -> gpd.GeoDataFrame:
    if aux_gdf.crs != main_gdf.crs:
        raise ValueError(f"Aux point cloud '{name}' has a different CRS than the main point cloud.")
    if not (
        np.array_equal(aux_gdf.geometry.x.values, main_gdf.geometry.x.values)
        and np.array_equal(aux_gdf.geometry.y.values, main_gdf.geometry.y.values)
    ):
        raise ValueError(f"Aux point cloud '{name}' does not have the same point coordinates as the main point cloud.")
    return aux_gdf

def _estimate_finite_fraction_grid(arr: NDArrayf) -> float:
    """
    Estimate fraction of finite values in a grid.

    For Dask, use a small window to avoid large computations.
    For NumPy, compute exact fraction.
    """
    # Masked arrays: consider filled data
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    if _is_dask_array(arr):
        # Use a small window (top-left) for a cheap estimate.
        ny, nx = arr.shape[:2]
        wy = min(int(ny), 512)
        wx = min(int(nx), 512)
        sample = np.asarray(arr[:wy, :wx].compute())
        if sample.size == 0:
            return 0.0
        return float(np.count_nonzero(np.isfinite(sample)) / sample.size)

    # NumPy / eager array
    a = np.asarray(arr)
    if a.size == 0:
        return 0.0
    return float(np.count_nonzero(np.isfinite(a)) / a.size)

def _warn_if_aux_sparse(*, aux_name: str, aux_frac: float, base_name: str, base_frac: float) -> None:
    # Warn if aux has <50% of the finite-data fraction of the base.
    # Use strict "<" as requested.
    if base_frac > 0 and aux_frac < 0.5 * base_frac:
        logging.warning(
            "Auxiliary variable '%s' has substantially fewer finite values than '%s' "
            "(finite fraction: %.3f vs %.3f). This may reduce the effective sample size.",
            aux_name,
            base_name,
            aux_frac,
            base_frac,
        )

def cosample(
    rst_pc1: NDArrayf | gu.Raster | gpd.GeoDataFrame,
    rst_pc2: NDArrayf | gu.Raster | gpd.GeoDataFrame | None = None,
    *,
    aux_vars: None
    | dict[str, NDArrayf | gu.Raster | np.ndarray | pd.Series[Any]]
    | list[NDArrayf | gu.Raster | np.ndarray | pd.Series[Any]] = None,
    inlier_mask: NDArrayb | gu.Raster | gu.Vector | gpd.GeoDataFrame | None = None,
    vector_mask_mode: Literal["inside", "outside"] = "inside",
    # Raster georeferencing if arrays are passed
    rst_pc1_transform: rio.transform.Affine | None = None,
    rst_pc2_transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    area_or_point: Literal["Area", "Point"] | None = None,
    # Point-cloud value column
    z_name: str = "z",
    # Subsampling
    subsample: float | int = 1,
    random_state: int | np.random.RandomState | None = None,
    # Optional outputs
    return_coords: bool = False,
    # Output formatting
    preserve_shape: bool = False,
) -> tuple[
    NDArrayf,                  # rst_pc1 values (1D or 2D if preserve_shape=True and raster-only)
    NDArrayf | None,           # rst_pc2 values (1D or 2D if preserve_shape=True and raster-only)
    dict[str, NDArrayf],       # auxiliary values (1D or 2D if preserve_shape=True and raster-only)
    None | tuple[NDArrayf, NDArrayf],  # coords (kept-only)
]:
    """
    Co-sample multiple rasters or point clouds at common locations with valid (finite) data, with optional vector/raster
    masking, random subsampling and return of subsampled coordinates.

    Main inputs can be either a single raster, or a single point cloud, or two rasters, or one raster and a point cloud.
    Auxiliary variables matching the grid/coords from any of the two inputs can be passed (as raster or 2D array, or
    as 1D array/Series aligned to the point coordinates if a point cloud input is used).
    An inlier mask can be passed as a raster or array the shape of any of the inputs (True = inlier), or as a vector
    object which will use to mask any value within or without geometry (from raster or point cloud).

    This function is designed to remain minimal and compatible with Dask-backed rasters:
    - Validity on rasters is computed on-grid (can remain lazy),
    - Point validity from point-native inputs is applied *before* any raster interpolation (to reduce work),
    - Point validity from rasters is derived by interpolating a *single* raster-validity mask to point coordinates,
    - Interpolation of rasters to points is performed only after subsampling (to avoid unnecessary work),
    - Outputs are returned eagerly (NumPy) at the subsample size to avoid building a massive Dask graph.

    preserve_shape behaviour
    ------------------------
    If preserve_shape=True:
    - raster-only mode returns 2D arrays on the reference grid, with NaNs outside (valid ∧ inlier ∧ subsampled).
    - point modes return 1D arrays of length equal to the *original* input point cloud (before pre-filtering), with
      NaNs outside (valid ∧ inlier ∧ subsampled).
    - return_coords remains "kept-only" to avoid returning very large coordinate arrays.

    Rules:
    - There are up to two main inputs (rst_pc1, rst_pc2). At least one must be a raster (never two point clouds).
    - If one main input is a point cloud, all rasters (mains and aux) are sampled at those point coordinates.
    - aux_vars can be provided as a dict or list. Elements can be:
        * raster-like (2D numpy/dask array, or Raster) on the same grid as the main raster reference,
        * 1D arrays/Series (NumPy/Pandas) of length n_points, aligned to the main point cloud ordering.
    - The inlier mask can be:
        * None -> all True on grid,
        * Vector/GeoDataFrame -> evaluated on raster grid (raster-only) or directly on points (point mode),
        * Raster/ndarray (2D) -> used on-grid,
        * ndarray (1D) -> only accepted if a main point cloud exists (mask on points).

    Modes:
    - raster-only: sample on the raster grid (2D mask, then boolean indexing).
    - raster-point: reduce points with point-native validity first, then evaluate raster validity at points once,
      then subsample.
    - point-only: subsample points directly (Vector masking is evaluated on points; raster aux vars are forbidden).

    Warnings:
    - If an auxiliary variable has <50% of the finite/valid data fraction of the main input domain it relates to,
      a warning is logged. For Dask-backed rasters, the fraction is estimated from a small window to avoid
      triggering large computations.
    """

    # -------------------------------------------------------------------------
    # Local helpers (minimal, to keep function self-contained)
    # -------------------------------------------------------------------------

    def _as_float_with_nan(arr: NDArrayf) -> NDArrayf:
        """Ensure array can hold NaNs (casts integer arrays to float)."""
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            return a.astype(np.float32)
        return a

    def _expand_from_sampled(
        *,
        original_len: int,
        reduced_to_original_mask: NDArrayb,
        keep_mask_reduced: NDArrayb,
        sampled_values: NDArrayf,
    ) -> NDArrayf:
        """
        Expand sampled 1D values (defined on reduced points) back to original point ordering,
        filling non-kept entries with NaN.
        """
        sampled_values = _as_float_with_nan(sampled_values)
        out = np.full(original_len, np.nan, dtype=sampled_values.dtype)

        keep_full = np.zeros(original_len, dtype=bool)
        keep_full[reduced_to_original_mask] = keep_mask_reduced

        out[keep_full] = sampled_values
        return out

    # -------------------------------------------------------------------------
    # 1) Input checks and normalization
    # -------------------------------------------------------------------------

    if vector_mask_mode not in ("inside", "outside"):
        raise ValueError("mask_mode must be either 'inside' or 'outside'.")

    aux_dict = _normalize_aux(aux_vars)

    # Accept EPC as a point cloud input (same as GeoDataFrame)
    is_pc1_pts = isinstance(rst_pc1, (gpd.GeoDataFrame, PointCloud))
    is_pc2_pts = isinstance(rst_pc2, (gpd.GeoDataFrame, PointCloud)) if rst_pc2 is not None else False
    if is_pc1_pts and is_pc2_pts:
        raise TypeError("cosample does not support two point clouds as main inputs.")

    point_mode = is_pc1_pts or is_pc2_pts

    # Identify the main point cloud (if any)
    pts_gdf: gpd.GeoDataFrame | None
    if point_mode:
        pts_gdf, z_name = _as_point_gdf(rst_pc1 if is_pc1_pts else rst_pc2, z_name=z_name)  # type: ignore[arg-type]
    else:
        pts_gdf, z_name = None, None

    # Identify a reference raster among main inputs (optional in point-only mode)
    raster_ref_arr: NDArrayf | None
    raster_ref_transform: rio.transform.Affine | None
    raster_ref_crs: rio.crs.CRS | None
    raster_ref_aop: Literal["Area", "Point"] | None = area_or_point
    if not is_pc1_pts:
        raster_ref_arr, raster_ref_transform, raster_ref_crs, raster_ref_aop = _as_raster_array_and_meta(
            rst_pc1,
            transform_fallback=rst_pc1_transform,
            crs_fallback=crs,
            aop_fallback=area_or_point,
        )
    elif rst_pc2 is not None and not is_pc2_pts:
        raster_ref_arr, raster_ref_transform, raster_ref_crs, raster_ref_aop = _as_raster_array_and_meta(
            rst_pc2,
            transform_fallback=rst_pc2_transform,
            crs_fallback=crs,
            aop_fallback=area_or_point,
        )
    else:
        # Point-only mode (no raster main input)
        if not point_mode or pts_gdf is None:
            raise TypeError("cosample requires at least one raster if two main inputs are passed.")
        raster_ref_arr = None
        raster_ref_transform = None
        raster_ref_crs = pts_gdf.crs

    # CRS selection
    if crs is None:
        crs = raster_ref_crs if raster_ref_crs is not None else (pts_gdf.crs if pts_gdf is not None else None)
    if crs is None:
        raise ValueError("'crs' must be provided if no CRS can be inferred from inputs.")

    # Strict CRS consistency for point mode
    if point_mode and pts_gdf is not None and pts_gdf.crs != crs:
        raise ValueError("Main point cloud CRS differs from 'crs'; reproject before calling cosample().")

    if subsample is None:
        raise ValueError("'subsample' cannot be None.")
    if isinstance(subsample, (int, float)) and float(subsample) <= 0:
        raise ValueError("'subsample' must be > 0.")

    # Extract main rasters to arrays once (after CRS/grid checks), keep None if point
    r1_arr: NDArrayf | None = None
    r2_arr: NDArrayf | None = None
    if not is_pc1_pts:
        r1_arr, _, _, _ = _as_raster_array_and_meta(
            rst_pc1,
            transform_fallback=rst_pc1_transform,
            crs_fallback=crs,
            aop_fallback=raster_ref_aop,
        )
    if rst_pc2 is not None and not is_pc2_pts:
        r2_arr, _, _, _ = _as_raster_array_and_meta(
            rst_pc2,
            transform_fallback=rst_pc2_transform,
            crs_fallback=crs,
            aop_fallback=raster_ref_aop,
        )

    # -------------------------------------------------------------------------
    # 2) Normalize point-domain inputs early (reduce points before raster interpolation)
    # -------------------------------------------------------------------------

    # 1D inlier mask on points (only allowed if point_mode)
    inlier_mask_pts_user: NDArrayb | None = None

    # Point-domain auxiliary variables stored as 1D arrays only (aligned to pts_gdf ordering)
    aux_pts: dict[str, NDArrayf] = {}

    # Mapping information to expand back to original point ordering if preserve_shape=True
    pts_original_len: int | None = None
    pts_reduced_to_original_mask: NDArrayb | None = None

    if point_mode and pts_gdf is not None:

        # Store original length for preserve_shape=True
        pts_original_len = len(pts_gdf)

        # Initialize pre-valid mask from main point Z
        valid_pts_pre = np.isfinite(pts_gdf[z_name].values)

        # If user provided vector mask, evaluate it directly on points (avoids rasterization + interpolation)
        if isinstance(inlier_mask, (gu.Vector, gpd.GeoDataFrame)):
            vec = inlier_mask if isinstance(inlier_mask, gu.Vector) else gu.Vector(inlier_mask)
            m = vec.create_mask(pts_gdf, as_array=True).astype(bool).squeeze()
            if vector_mask_mode == "outside":
                m = ~m
            valid_pts_pre = valid_pts_pre & m

        # If user provided 1D inlier mask, combine now (will be reduced alongside points)
        if isinstance(inlier_mask, np.ndarray) and _ndim(inlier_mask) == 1:
            inlier_mask_pts_user = np.asarray(inlier_mask).astype(bool).squeeze()
            if inlier_mask_pts_user.shape[0] != valid_pts_pre.shape[0]:
                raise ValueError("1D inlier_mask length does not match the number of input points.")
            valid_pts_pre = valid_pts_pre & inlier_mask_pts_user

        # Collect point-domain aux variables (Series or 1D arrays) and combine finite validity now
        for k, v in list(aux_dict.items()):

            if isinstance(v, pd.Series):
                v1 = v.to_numpy()
                if v1.shape[0] != valid_pts_pre.shape[0]:
                    raise ValueError(
                        f"Aux variable '{k}' is a Series but length does not match the number of input points."
                    )
                aux_pts[k] = v1
                valid_pts_pre = valid_pts_pre & np.isfinite(v1)
                aux_dict.pop(k)
                continue

            if isinstance(v, np.ndarray) and _ndim(v) == 1:
                v1 = np.asarray(v).squeeze()
                if v1.shape[0] != valid_pts_pre.shape[0]:
                    raise ValueError(f"Aux variable '{k}' is 1D but length does not match the number of input points.")
                aux_pts[k] = v1
                valid_pts_pre = valid_pts_pre & np.isfinite(v1)
                aux_dict.pop(k)
                continue

        # Reduce point dataset and point-domain aux variables once, consistently
        if np.count_nonzero(valid_pts_pre) == 0:
            raise ValueError("There is no valid point data common to the point inputs (mask and point auxiliaries).")

        # Save mapping mask (original -> reduced)
        pts_reduced_to_original_mask = valid_pts_pre.copy()

        if inlier_mask_pts_user is not None:
            inlier_mask_pts_user = inlier_mask_pts_user[valid_pts_pre]

        pts_gdf = pts_gdf[valid_pts_pre]
        for k in list(aux_pts.keys()):
            aux_pts[k] = aux_pts[k][valid_pts_pre]

    # -------------------------------------------------------------------------
    # 3) Derive grid-domain inlier mask and classify grid-domain auxiliaries
    # -------------------------------------------------------------------------

    inlier_mask_grid: NDArrayb | None = None

    # Grid-domain auxiliaries stored as arrays only (2D numpy/dask)
    aux_grid: dict[str, NDArrayf] = {}

    if raster_ref_arr is None:
        # Point-only mode forbids any raster/grid aux vars and any raster/grid inlier mask
        for k in aux_dict.keys():
            raise TypeError(
                f"Aux variable '{k}' is raster/grid-like, but point-only mode has no raster grid to sample on."
            )

        if inlier_mask is not None and not isinstance(inlier_mask, (gu.Vector, gpd.GeoDataFrame, np.ndarray)):
            raise TypeError(
                "Point-only mode only supports inlier_mask as None, Vector/GeoDataFrame (evaluated on points), "
                "or a 1D boolean array of length n_points."
            )

        # At this point, points were already reduced by all point-native validity in step 2.
        # Build a final valid_pts mask as all True (further raster validity does not exist in point-only mode).
        if pts_gdf is None:
            raise AssertionError

        valid_pts = np.ones(len(pts_gdf), dtype=bool)

    else:
        # Raster grid exists: derive a 2D inlier mask on the reference grid unless point-native mask already applied
        grid_shape = raster_ref_arr.shape
        grid_raster = gu.Raster.from_array(
            data=np.zeros(grid_shape, dtype=np.uint8),
            transform=raster_ref_transform,
            crs=crs,
            area_or_point=raster_ref_aop,
        )

        # If a vector mask was provided, it was already applied on points in step 2 for point_mode.
        # For raster-only mode, it still needs to be rasterized here.
        if inlier_mask is None:
            inlier_mask_grid = np.ones(grid_shape, dtype=bool)

        elif isinstance(inlier_mask, (gu.Vector, gpd.GeoDataFrame)):
            if point_mode:
                # In point mode, vector masks are handled on points for efficiency.
                inlier_mask_grid = np.ones(grid_shape, dtype=bool)
            else:
                vec = inlier_mask if isinstance(inlier_mask, gu.Vector) else gu.Vector(inlier_mask)
                inlier_mask_grid = vec.create_mask(grid_raster, as_array=True).astype(bool).squeeze()
                if vector_mask_mode == "outside":
                    inlier_mask_grid = ~inlier_mask_grid

        elif isinstance(inlier_mask, gu.Raster):
            if not grid_raster.georeferenced_grid_equal(inlier_mask):
                inlier_mask = inlier_mask.reproject(grid_raster, resampling=rio.warp.Resampling.nearest, silent=True)
            inlier_mask_grid = inlier_mask.data.filled(False).astype(bool).squeeze()

        else:
            arr = np.asarray(inlier_mask).squeeze()
            if arr.ndim == 1:
                if not point_mode:
                    raise ValueError("1D inlier_mask is only supported when a main point cloud is provided.")
                # 1D point mask was already handled in step 2.
                inlier_mask_grid = np.ones(grid_shape, dtype=bool)
            else:
                if arr.dtype != bool:
                    raise ValueError(f"Invalid inlier_mask dtype: '{arr.dtype}'. Expected 'bool'.")
                if arr.shape != grid_shape:
                    raise ValueError(f"inlier_mask shape {arr.shape} does not match expected grid shape {grid_shape}.")
                inlier_mask_grid = arr

        # Classify remaining aux variables as grid-domain (Raster or 2D arrays)
        for k, v in aux_dict.items():
            if isinstance(v, gu.Raster):
                v_arr = v.data.filled(np.nan) if np.ma.isMaskedArray(v.data) else v.data
                if _shape(v_arr) != raster_ref_arr.shape:
                    raise ValueError(
                        f"Aux Raster '{k}' shape {_shape(v_arr)} does not match grid shape {raster_ref_arr.shape}."
                    )
                aux_grid[k] = v_arr
                continue

            if _ndim(v) != 2:
                raise ValueError(
                    f"Aux variable '{k}' must be a 2D raster/grid array or Raster, or a 1D array/Series in point mode."
                )
            if _shape(v) != raster_ref_arr.shape:
                raise ValueError(
                    f"Aux variable '{k}' shape {_shape(v)} does not match grid shape {raster_ref_arr.shape}."
                )
            aux_grid[k] = v  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # 4) Warnings for sparse auxiliary validity (relative to their base domain)
    # -------------------------------------------------------------------------

    # Point-domain warnings: after early reduction, the base finite fraction is effectively 1.0
    if point_mode and pts_gdf is not None:
        base_pts_name = "main point cloud"
        base_pts_frac = 1.0
        for k, v1 in aux_pts.items():
            aux_frac = float(np.count_nonzero(np.isfinite(v1)) / max(v1.size, 1))
            _warn_if_aux_sparse(aux_name=k, aux_frac=aux_frac, base_name=base_pts_name, base_frac=base_pts_frac)

    # Grid-domain warnings: compare finite fractions to the reference raster finite fraction.
    if raster_ref_arr is not None:
        base_grid_name = "reference raster grid"
        base_grid_frac = _estimate_finite_fraction_grid(raster_ref_arr)
        for k, v_arr in aux_grid.items():
            aux_frac = _estimate_finite_fraction_grid(v_arr)
            _warn_if_aux_sparse(aux_name=k, aux_frac=aux_frac, base_name=base_grid_name, base_frac=base_grid_frac)

    # -------------------------------------------------------------------------
    # 5) Build final validity and subsample, then sample outputs
    # -------------------------------------------------------------------------

    out_aux: dict[str, NDArrayf] = {}

    # 5A) Point-only: subsample points (already reduced), then return 1D outputs
    if raster_ref_arr is None:
        if pts_gdf is None:
            raise AssertionError

        sub_mask_pts = _get_subsample_on_valid_mask(
            subsample=subsample, random_state=random_state, valid_mask=valid_pts  # type: ignore[name-defined]
        )

        # Sample kept values on reduced point set
        out1_sampled = _as_float_with_nan(pts_gdf[z_name].values)[sub_mask_pts]
        out2_sampled = None
        aux_sampled = {k: _as_float_with_nan(v1)[sub_mask_pts] for k, v1 in aux_pts.items()}

        if preserve_shape:
            if pts_original_len is None or pts_reduced_to_original_mask is None:
                raise AssertionError

            out_rst_pc1 = _expand_from_sampled(
                original_len=pts_original_len,
                reduced_to_original_mask=pts_reduced_to_original_mask,
                keep_mask_reduced=sub_mask_pts,
                sampled_values=out1_sampled,
            )
            out_rst_pc2 = None

            for k, v_s in aux_sampled.items():
                out_aux[k] = _expand_from_sampled(
                    original_len=pts_original_len,
                    reduced_to_original_mask=pts_reduced_to_original_mask,
                    keep_mask_reduced=sub_mask_pts,
                    sampled_values=v_s,
                )
        else:
            out_rst_pc1 = out1_sampled
            out_rst_pc2 = None
            out_aux.update(aux_sampled)

        if return_coords:
            pts = (pts_gdf.geometry.x.values, pts_gdf.geometry.y.values)
            sub_pts = (pts[0][sub_mask_pts], pts[1][sub_mask_pts])
            sub_coords = sub_pts
        else:
            sub_coords = None

        return out_rst_pc1, out_rst_pc2, out_aux, sub_coords

    # 5B) Raster-only: build grid valid mask (lazy if Dask), subsample on-grid, then index / preserve
    if not point_mode:

        if inlier_mask_grid is None:
            raise AssertionError
        if r1_arr is None:
            raise AssertionError  # raster-only implies rst_pc1 is raster-like

        valid_grid = inlier_mask_grid

        valid_grid = valid_grid & np.isfinite(r1_arr)
        if r2_arr is not None:
            valid_grid = valid_grid & np.isfinite(r2_arr)
        for v_arr in aux_grid.values():
            valid_grid = valid_grid & np.isfinite(v_arr)

        valid_grid_np = np.asarray(_compute_if_dask(valid_grid)).astype(bool)

        if np.count_nonzero(valid_grid_np) == 0:
            raise ValueError("There is no valid data common to the input and auxiliary variables.")

        sub_mask_grid = _get_subsample_on_valid_mask(
            subsample=subsample, random_state=random_state, valid_mask=valid_grid_np
        )

        if preserve_shape:
            # keep_grid True only where subsampled among valid points
            keep_grid = np.zeros_like(valid_grid_np, dtype=bool)
            keep_grid[sub_mask_grid] = True

            r1_full = _as_float_with_nan(np.asarray(_compute_if_dask(r1_arr)))
            out_rst_pc1 = np.where(keep_grid, r1_full, np.nan)

            if r2_arr is not None:
                r2_full = _as_float_with_nan(np.asarray(_compute_if_dask(r2_arr)))
                out_rst_pc2 = np.where(keep_grid, r2_full, np.nan)
            else:
                out_rst_pc2 = None

            for k, v_arr in aux_grid.items():
                v_full = _as_float_with_nan(np.asarray(_compute_if_dask(v_arr)))
                out_aux[k] = np.where(keep_grid, v_full, np.nan)

        else:
            out_rst_pc1 = np.asarray(_compute_if_dask(r1_arr[sub_mask_grid]))
            out_rst_pc2 = np.asarray(_compute_if_dask(r2_arr[sub_mask_grid])) if r2_arr is not None else None
            for k, v_arr in aux_grid.items():
                out_aux[k] = np.asarray(_compute_if_dask(v_arr[sub_mask_grid]))

        if return_coords:
            coords = _coords(transform=raster_ref_transform, shape=raster_ref_arr.shape, area_or_point=raster_ref_aop)
            sub_coords = (coords[0][sub_mask_grid], coords[1][sub_mask_grid])
        else:
            sub_coords = None

        return out_rst_pc1, out_rst_pc2, out_aux, sub_coords

    # 5C) Raster-point: evaluate raster validity at (already reduced) points once, then subsample on points
    if pts_gdf is None:
        raise AssertionError
    if inlier_mask_grid is None:
        raise AssertionError

    # Build raster-grid validity once (lazy if Dask)
    valid_grid = inlier_mask_grid
    if r1_arr is not None:
        valid_grid = valid_grid & np.isfinite(r1_arr)
    if r2_arr is not None:
        valid_grid = valid_grid & np.isfinite(r2_arr)
    for v_arr in aux_grid.values():
        valid_grid = valid_grid & np.isfinite(v_arr)

    pts = (pts_gdf.geometry.x.values, pts_gdf.geometry.y.values)

    # Interpolate raster-validity mask to points (only once, on reduced point set)
    valid_grid_f = valid_grid.astype(np.float32)
    valid_grid_f = np.where(valid_grid_f, 1.0, np.nan)

    valid_pts = np.isfinite(
        _interp_points(array=valid_grid_f, transform=raster_ref_transform, points=pts, area_or_point=raster_ref_aop)
    )

    if np.count_nonzero(valid_pts) == 0:
        raise ValueError("There is no valid data common to the input and auxiliary variables.")

    sub_mask_pts = _get_subsample_on_valid_mask(
        subsample=subsample, random_state=random_state, valid_mask=valid_pts
    )
    sub_pts = (pts[0][sub_mask_pts], pts[1][sub_mask_pts])

    # Sample mains (rasters interpolated only on subsample, points indexed)
    if is_pc1_pts:
        out1_sampled = _as_float_with_nan(pts_gdf[z_name].values)[sub_mask_pts]
    else:
        out1_sampled = _as_float_with_nan(
            np.asarray(
                _interp_points(
                    array=r1_arr,
                    transform=raster_ref_transform,
                    points=sub_pts,
                    area_or_point=raster_ref_aop,
                )
            )
        )

    if rst_pc2 is not None:
        if is_pc2_pts:
            out2_sampled = _as_float_with_nan(pts_gdf[z_name].values)[sub_mask_pts]
        else:
            out2_sampled = _as_float_with_nan(
                np.asarray(
                    _interp_points(
                        array=r2_arr,
                        transform=raster_ref_transform,
                        points=sub_pts,
                        area_or_point=raster_ref_aop,
                    )
                )
            )
    else:
        out2_sampled = None

    # Sample aux grid variables at subsampled points (interpolate only on subsample)
    aux_sampled: dict[str, NDArrayf] = {}
    for k, v_arr in aux_grid.items():
        aux_sampled[k] = _as_float_with_nan(
            np.asarray(
                _interp_points(array=v_arr, transform=raster_ref_transform, points=sub_pts, area_or_point=raster_ref_aop)
            )
        )

    # Sample aux point variables by indexing (on reduced point set)
    for k, v1 in aux_pts.items():
        aux_sampled[k] = _as_float_with_nan(v1)[sub_mask_pts]

    if preserve_shape:
        if pts_original_len is None or pts_reduced_to_original_mask is None:
            raise AssertionError

        out_rst_pc1 = _expand_from_sampled(
            original_len=pts_original_len,
            reduced_to_original_mask=pts_reduced_to_original_mask,
            keep_mask_reduced=sub_mask_pts,
            sampled_values=out1_sampled,
        )

        if out2_sampled is None:
            out_rst_pc2 = None
        else:
            out_rst_pc2 = _expand_from_sampled(
                original_len=pts_original_len,
                reduced_to_original_mask=pts_reduced_to_original_mask,
                keep_mask_reduced=sub_mask_pts,
                sampled_values=out2_sampled,
            )

        for k, v_s in aux_sampled.items():
            out_aux[k] = _expand_from_sampled(
                original_len=pts_original_len,
                reduced_to_original_mask=pts_reduced_to_original_mask,
                keep_mask_reduced=sub_mask_pts,
                sampled_values=v_s,
            )

    else:
        out_rst_pc1 = out1_sampled
        out_rst_pc2 = out2_sampled
        out_aux.update(aux_sampled)

    if return_coords:
        sub_coords = sub_pts
    else:
        sub_coords = None

    return out_rst_pc1, out_rst_pc2, out_aux, sub_coords
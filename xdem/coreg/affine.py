# Copyright (c) 2025 xDEM developers
# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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

"""Affine coregistration classes."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Iterable, Literal, TypeVar, TypedDict

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.optimize
import scipy.spatial
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from geoutils._typing import Number
from geoutils.interface.interpolate import _interp_points
from geoutils.raster.georeferencing import _coords, _res
from geoutils.stats import nmad

from xdem._misc import get_progress, import_optional
from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    InFitOrBinDict,
    InRandomDict,
    OutAffineDict,
    OutIterativeDict,
    _apply_matrix_pts,
    _apply_matrix_pts_mat,
    _apply_matrix_rst,
    _bin_or_and_fit_nd,
    _make_matrix_valid,
    _reproject_horizontal_shift_samecrs,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)
from xdem.cosampling import _subsample_rst_pts, _get_subsample_mask_pts_rst
from xdem.fit import index_trimmed

######################################
# Generic functions for affine methods
######################################


def _check_inputs_bin_before_fit(
    bin_before_fit: bool,
    fit_minimizer: Callable[..., tuple[NDArrayf, Any]],
    bin_sizes: int | dict[str, int | Iterable[float]],
    bin_statistic: Callable[[NDArrayf], np.floating[Any]],
) -> None:
    """
    Check input types of fit or bin_and_fit affine functions.

    :param bin_before_fit: Whether to bin data before fitting the coregistration function.
    :param fit_minimizer: Minimizer for the coregistration.
    :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
    :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
    """

    if not callable(fit_minimizer):
        raise TypeError(
            "Argument `fit_minimizer` must be a function (callable), " "got {}.".format(type(fit_minimizer))
        )

    if bin_before_fit:

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


def _iterate_method(
    method: Callable[..., Any],
    iterating_input: Any,
    constant_inputs: tuple[Any, ...],
    tolerances: dict[str, float],
    max_iterations: int,
) -> tuple[Any, Any, OutAffineDict, Any]:
    """
    Function to iterate a method (e.g. ICP, Nuth and Kääb) until it reaches tolerances or maximum number of iterations.

    :param method: Method that needs to be iterated to derive a transformation. Takes argument "inputs" as its input,
        and outputs three terms: a "statistic" to compare to tolerance, "updated inputs" with this transformation, and
        the parameters of the transformation.
    :param iterating_input: Iterating input to method, should be first argument.
    :param constant_inputs: Constant inputs to method, should be all positional arguments after first.
    :param tolerances: Tolerances to reach for the method statistics (i.e. maximum value for the statistic).
    :param max_iterations: Maximum number of iterations for the method.

    :return: Final output of iterated method.
    """

    # Initiate inputs
    new_inputs = iterating_input

    # Initiate storage of iteration statistics
    list_df = []

    # Iteratively run the analysis until the maximum iterations or until the error gets low enough
    # If logging level <= INFO, will use progressbar and print additional statements
    pbar = get_progress(
        range(max_iterations), disable=logging.getLogger().getEffectiveLevel() > logging.INFO, desc="   Progress"
    )
    for i in pbar:

        # Apply method and get new statistics to compare to tolerances, new inputs for next iterations, and
        # outputs in case this is the final one
        new_inputs, new_statistics, static_outputs = method(new_inputs, *constant_inputs)

        # Store statistics to dataframe and append to list
        df_iteration = pd.DataFrame(new_statistics, index=[i + 1])
        df_iteration["iteration"] = i + 1
        list_df.append(df_iteration)

        # Check that all statistics have a matching tolerance, otherwise the process should fail with a dev error
        new_statistics_keys = list(new_statistics.keys())
        tolerance_keys = list(tolerances.keys())
        if not all([n in tolerance_keys for n in new_statistics_keys]):
            raise NotImplementedError(
                "Developer Error: The keys of the tolerances dictionary passed "
                "to _iterate_method in the coregistration method call should match the keys of "
                "the statistics return by the method's iteration_step function."
            )

        # Print final results
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            pbar.write(f"      Iteration #{i + 1:d}")
            k = list(tolerances.keys())
            for j in range(len(tolerances)):
                pbar.write(f"   Last {k[j]} offset: {new_statistics[k[j]]}")

        # Check that all statistics are below their respective tolerance
        if all(new_statistics[k] < tolerances[k] if k is not None else True for k in tolerances.keys()):
            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                pbar.write(f"   The last offset(s) were all below the set tolerance(s) -> stopping")
                all_tolerances = ";".join(f"{k}: {v}" for k, v in tolerances.items())
                pbar.write(f"   Set tolerance(s) were: {all_tolerances}.")

            break

    df_all_it = pd.concat(list_df)
    output_iterative: OutAffineDict = {"last_iteration": i + 1, "iteration_stats": df_all_it}

    return new_inputs, new_statistics, output_iterative, static_outputs


def _subsample_on_mask_interpolator(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    aux_vars: None | dict[str, NDArrayf],
    sub_mask: NDArrayb,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
) -> tuple[Callable[[float, float], NDArrayf], None | dict[str, NDArrayf]]:
    """
    Mirrors coreg.base._subsample_on_mask, but returning an interpolator of elevation difference and subsampled
    coordinates for efficiency in iterative affine methods.

    Perform subsampling on mask for raster-raster or point-raster datasets on valid points of all inputs (including
    potential auxiliary variables), returning coordinates along with an interpolator.
    """

    # For two rasters
    if isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):

        # Derive coordinates and interpolator
        coords = _coords(transform=transform, shape=ref_elev.shape, area_or_point=area_or_point, grid=True)
        tba_elev_interpolator = _reproject_horizontal_shift_samecrs(
            tba_elev, src_transform=transform, return_interpolator=True
        )

        # Subsample coordinates
        sub_coords = (coords[0][sub_mask], coords[1][sub_mask])

        def sub_dh_interpolator(shift_x: float, shift_y: float) -> NDArrayf:
            """Elevation difference interpolator for shifted coordinates of the subsample."""

            # TODO: Align array axes in _reproject_horizontal... ?
            # Get interpolator of dh for shifted coordinates; Y and X are inverted here due to raster axes
            return ref_elev[sub_mask] - tba_elev_interpolator((sub_coords[1] + shift_y, sub_coords[0] + shift_x))

        # Subsample auxiliary variables with the mask
        if aux_vars is not None:
            sub_bias_vars = {}
            for var in aux_vars.keys():
                sub_bias_vars[var] = aux_vars[var][sub_mask]
        else:
            sub_bias_vars = None

    # For one raster and one point cloud
    else:

        # Identify which dataset is point or raster
        pts_elev: gpd.GeoDataFrame = ref_elev if isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev
        rst_elev: NDArrayf = ref_elev if not isinstance(ref_elev, gpd.GeoDataFrame) else tba_elev
        # Check which input is reference, to compute the dh always in the same direction (ref minus tba) further below
        ref = "point" if isinstance(ref_elev, gpd.GeoDataFrame) else "raster"

        # Subsample point coordinates
        coords = (pts_elev.geometry.x.values, pts_elev.geometry.y.values)
        sub_coords = (coords[0][sub_mask], coords[1][sub_mask])

        # Interpolate raster array to the subsample point coordinates
        # Convert ref or tba depending on which is the point dataset
        rst_elev_interpolator = _interp_points(
            array=rst_elev,
            transform=transform,
            area_or_point=area_or_point,
            points=sub_coords,
            return_interpolator=True,
        )

        def sub_dh_interpolator(shift_x: float, shift_y: float) -> NDArrayf:
            """Elevation difference interpolator for shifted coordinates of the subsample."""

            # Always return ref minus tba
            if ref == "point":
                return pts_elev[z_name][sub_mask].values - rst_elev_interpolator(
                    (sub_coords[1] + shift_y, sub_coords[0] + shift_x)
                )
            # Also invert the shift direction on the raster interpolator, so that the shift is the same relative to
            # the reference (returns the right shift relative to the reference no matter if it is point or raster)
            else:
                return (
                    rst_elev_interpolator((sub_coords[1] - shift_y, sub_coords[0] - shift_x))
                    - pts_elev[z_name][sub_mask].values
                )

        # Interpolate arrays of bias variables to the subsample point coordinates
        if aux_vars is not None:
            sub_bias_vars = {}
            for var in aux_vars.keys():
                sub_bias_vars[var] = _interp_points(
                    array=aux_vars[var], transform=transform, points=sub_coords, area_or_point=area_or_point
                )
        else:
            sub_bias_vars = None

    return sub_dh_interpolator, sub_bias_vars


def _subsample_rst_pts_interpolator(
    params_random: InRandomDict,
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    aux_vars: None | dict[str, NDArrayf] = None,
) -> tuple[Callable[[float, float], NDArrayf], None | dict[str, NDArrayf], int]:
    """
    Mirrors coreg.base._subsample_rst_pts, but returning an interpolator for efficiency in iterative methods.

    Pre-process raster-raster or point-raster datasets into an elevation difference interpolator at the same
    points, and subsample arrays for auxiliary variables, with subsampled coordinates to evaluate the interpolator.

    Returns dh interpolator, tuple of 1D arrays of subsampled coordinates, and dictionary of 1D arrays of subsampled
    auxiliary variables.
    """

    # Get subsample mask (a 2D array for raster-raster, a 1D array of length the point data for point-raster)
    sub_mask = _get_subsample_mask_pts_rst(
        subsample=params_random["subsample"],
        random_state=params_random["random_state"],
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
        aux_vars=aux_vars,
    )

    # Return interpolator of elevation differences and subsampled auxiliary variables
    sub_dh_interpolator, sub_bias_vars = _subsample_on_mask_interpolator(
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        aux_vars=aux_vars,
        sub_mask=sub_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
    )

    # Derive subsample size to pass back to class
    subsample_final = np.count_nonzero(sub_mask)

    # Return 1D arrays of subsampled points at the same location
    return sub_dh_interpolator, sub_bias_vars, subsample_final


def _get_centroid_scale(
    ref_elev: NDArrayf | gpd.GeoDataFrame, transform: affine.Affine | None, z_name: str | None = None,
) -> tuple[tuple[float, float, float], float]:
    """
    Get centroid and standardization factor from reference elevation (whether it is a DEM or an elevation point cloud).

    This step needs to be computed before subsampling to avoid inconsistencies between random samplings.

    :param ref_elev: Reference elevation, either an array or a point cloud.
    :param transform: Geotransform if reference elevation is a DEM.

    :return: Centroid of elevation object, Scale factor of elevation object.
    """

    # For a DEM
    if isinstance(ref_elev, np.ndarray):

        # Get coordinates of DEM
        coords_x, coords_y = _coords(transform=transform, shape=ref_elev.shape, area_or_point=None, grid=False)
        # Derive centroid
        centroid_x = np.nanmedian(coords_x)
        centroid_y = np.nanmedian(coords_y)
        if np.ma.isMaskedArray(ref_elev):
            centroid_z = np.ma.median(ref_elev)
        else:
            centroid_z = np.nanmedian(ref_elev)
        centroid = (centroid_x, centroid_y, centroid_z)

        # Derive standardization factor
        std_fac = np.mean([nmad(coords_x - centroid[0]), nmad(coords_y - centroid[1]), nmad(ref_elev - centroid[2])])

    # For an elevation point cloud
    else:
        # Derive centroid
        centroid = (np.nanmedian(ref_elev.geometry.x.values),
                    np.nanmedian(ref_elev.geometry.y.values),
                    np.nanmedian(ref_elev[z_name].values))

        # Derive standardization factor
        std_fac = float(
            np.mean(
                [
                    nmad(ref_elev.geometry.x.values - centroid[0]),
                    nmad(ref_elev.geometry.y.values - centroid[1]),
                    nmad(ref_elev[z_name].values - centroid[2]),
                ]
            )
        )

    return centroid, std_fac


def _standardize_epc(
    ref_epc: NDArrayf, tba_epc: NDArrayf, centroid: tuple[float, float, float], scale: float | int = 1
) -> tuple[NDArrayf, NDArrayf]:
    """
    Standardize elevation point clouds by subtracting a centroid and dividing per scale factor.

    Usually paired with _get_centroid_scale() to get the centroid and scale factor.
    To avoid applying the scale, simply leave to default (=1).

    :param ref_epc: Reference point cloud.
    :param tba_epc: To-be-aligned point cloud.
    :param centroid: Centroid of point cloud.
    :param scale: Scale of point cloud (defaults to 1).

    :return: Standardized point clouds.
    """

    # Convert centroid to array
    centroid = np.array(centroid)

    # Subtract centroid from point clouds
    ref_epc = ref_epc - centroid[:, None]
    tba_epc = tba_epc - centroid[:, None]

    # Standardize point clouds
    if scale != 1:
        ref_epc = ref_epc / scale
        tba_epc = tba_epc / scale

    return ref_epc, tba_epc


# Helper for computing normals
##############################

class DemGeometryDict(TypedDict, total=False):
    """Keys and types of DEM-derived geometry rasters."""

    nx: NDArrayf
    ny: NDArrayf
    nz: NDArrayf
    curv: NDArrayf

def _dem_normals_curvature(
    dem: NDArrayf,
    transform: affine.Affine,
    *,
    return_curvature: bool = False,
) -> DemGeometryDict:
    """
    Derive fixed-surface geometry from a DEM: unit normals (and optionally a curvature proxy).

    This helper is meant to be used consistently by both ICP ("point-to-plane") and CPD (LSG variant),
    so that "fixed surface" geometry is derived the same way across methods.

    Notes
    -----
    - Normals are derived from the height field z(x, y) using:
          n ∝ (-dz/dx, -dz/dy, 1)
      then normalized to unit length.
    - Curvature is an optional, lightweight proxy derived from the Hessian magnitude:
          curvature ≈ ||[d²z/dx², d²z/dy²]||
      scaled by pixel size so it has comparable behavior across resolutions.
      This is not a differential-geometry "true curvature", but it is a stable surface-variation indicator
      suitable to modulate planarity weights (as in LSG-CPD).

    :param dem: DEM array.
    :param transform: Affine transform of DEM.
    :param return_curvature: Whether to derive and return a curvature-like proxy.

    :return: Dictionary of DEM-derived rasters:
        - nx: East component of unit normal.
        - ny: North component of unit normal.
        - nz: Up component of unit normal.
        - curvature: Curvature proxy (if return_curvature=True).
    """
    # Get resolution
    res_x, res_y = _res(transform)

    # If masked array input
    if np.ma.isMaskedArray(dem):
        dem = dem.filled(np.nan)

    # 1) First derivatives (dz/dx, dz/dy) in map units (need to scale per resolution)
    dz_drow, dz_dcol = np.gradient(dem.astype(float), edge_order=1)
    dz_dx = dz_dcol / res_x
    dz_dy = -dz_drow / res_y

    # 2) Unit normals: (-dz/dx, -dz/dy, 1)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(nx)
    # Normalize relative to precision
    n_norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    n_norm = np.clip(n_norm, np.finfo(float).eps, None)
    nx = nx / n_norm
    ny = ny / n_norm
    nz = nz / n_norm

    out: DemGeometryDict = {"nx": nx, "ny": ny, "nz": nz}

    # 3) Curvature proxy, scaled + mapped to (0, 1)
    if return_curvature:
        # Second derivatives in index space
        d2z_drow2, _ = np.gradient(dz_drow, edge_order=1)
        _, d2z_dcol2 = np.gradient(dz_dcol, edge_order=1)

        # Convert to map-coordinate second derivatives
        d2z_dx2 = d2z_dcol2 / (res_x * res_x)
        d2z_dy2 = d2z_drow2 / (res_y * res_y)

        # Raw (units ~ 1/length): magnitude of principal second derivatives
        curv_raw = np.sqrt(d2z_dx2 * d2z_dx2 + d2z_dy2 * d2z_dy2)
        curv_raw = np.clip(curv_raw, np.finfo(float).eps, None)

        # Make approximately dimensionless by multiplying with a pixel length scale
        pix = float(np.sqrt(res_x * res_y))
        curv_dimless = curv_raw * pix

        # Map to (0, 1) with a robust scale c0 (median is a good default)
        c0 = float(np.nanmedian(curv_dimless))
        c0 = max(c0, np.finfo(float).eps)
        curvature = curv_dimless / (curv_dimless + c0)

        # Final clip so 1/curvature is well-behaved
        curvature = np.clip(curvature, 1e-12, 1.0)

        out["curv"] = curvature

    return out

def _epc_normals_curvature(points: NDArrayf, neighbours: int) -> tuple[NDArrayf, NDArrayf]:
    """
    Compute normals and curvature-like variation measure from a point cloud using kNN PCA.

    :param points: Point cloud (M,3).
    :param neighbours: Number of neighbors for local PCA.
    :return: normals (M,3) unit, curvature (M,) = smallest_eig / sum_eigs.
    """
    Y = np.asarray(points, dtype=float)
    M = Y.shape[0]
    k = max(int(neighbours), 3)

    tree = cKDTree(Y)
    _, idx = tree.query(Y, k=k + 1)  # includes itself
    idx = idx[:, 1:]                 # drop self
    neigh = Y[idx]                   # (M,k,3)

    mu = neigh.mean(axis=1, keepdims=True)
    Xc = neigh - mu
    C = np.einsum("mki,mkj->mij", Xc, Xc) / max(k - 1, 1)

    evals, evecs = np.linalg.eigh(C)      # ascending
    normals = evecs[:, :, 0]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    curvature = evals[:, 0] / (evals.sum(axis=1) + 1e-12)
    curvature = np.clip(curvature, 1e-12, None)

    return normals.astype(float), curvature.astype(float)

def _axis_weights_from_epc(
    epc: NDArrayf,
    *,
    anisotropic: Literal["xy_vs_z", "per_axis"] | None = None,
    eps: float = 1e-12,
) -> NDArrayf:
    """
    Compute anisotropic axis weights from a 3xN EPC.

    Either None (isotropic), "per_axis" (each axis weighted independent by its std), or "xy_vs_z" where X/Y share the
    same weight pooled from X and Y std.

    :param epc: EPC array (3, N).
    :param anisotropic: Type of anisotropic weighting.
    :param eps: Small value to avoid division by zero.

    :return: w shape (3,), axis weights.
    """
    xyz = np.asarray(epc, dtype=float)

    if xyz.shape[0] != 3:
        raise ValueError("epc must have shape (3, N).")

    if anisotropic is None:
        return np.ones(3, dtype=float)

    # Get STD of each axis, avoid division per zero
    std = np.nanstd(xyz, axis=1)
    std = np.clip(std, eps, None)

    # Per axis
    if anisotropic == "per_axis":
        w = 1.0 / (std ** 2)

    # For XY vs Z, we sum the variances of X/Y and take the half-squareroot
    elif anisotropic == "xy_vs_z":
        std_xy = np.sqrt(0.5 * (std[0] ** 2 + std[1] ** 2))
        std_xy = max(std_xy, eps)
        w_xy = 1.0 / (std_xy ** 2)
        w_z = 1.0 / (std[2] ** 2)
        w = np.array([w_xy, w_xy, w_z], dtype=float)

    else:
        raise ValueError("anisotropic must be None, 'xy_vs_z', or 'per_axis'.")

    return w.astype(float)

################################
# Affine coregistrations methods
# ##############################

##################
# 1/ Nuth and Kääb
##################


def _nuth_kaab_fit_func(xx: NDArrayf, *params: tuple[float, float, float]) -> NDArrayf:
    """
    Nuth and Kääb (2011) fitting function.

    Describes the elevation differences divided by the slope tangent (y) as a 1D function of the aspect.

    y(x) = a * cos(b - x) + c

    where y = dh/tan(slope) and x = aspect.

    :param xx: The aspect in radians.
    :param params: Parameters a, b and c of above function.

    :returns: Estimated y-values with the same shape as the given x-values
    """
    return params[0] * np.cos(params[1] - xx) + params[2]


def _nuth_kaab_bin_fit(
    dh: NDArrayf,
    slope_tan: NDArrayf,
    aspect: NDArrayf,
    params_fit_or_bin: InFitOrBinDict,
) -> tuple[float, float, float]:
    """
    Optimize the Nuth and Kääb (2011) function based on observed values of elevation differences, slope tangent and
    aspect at the same locations, using either fitting or binning + fitting.

    Called at each iteration step.

    :param dh: 1D array of elevation differences (in georeferenced unit, typically meters).
    :param slope_tan: 1D array of slope tangent (unitless).
    :param aspect: 1D array of aspect (units = radians).
    :param params_fit_or_bin: Dictionary of parameters for fitting or binning.

    :returns: Optimized parameters of Nuth and Kääb (2011) fit function: easting, northing, and vertical offsets
        (in georeferenced unit).
    """

    # Slope tangents near zero were removed beforehand, so errors should never happen here
    with np.errstate(divide="ignore", invalid="ignore"):
        y = dh / slope_tan
        valids = np.isfinite(y)
        y = y[valids]
        dh = dh[valids]

    # Trim if required
    if "trim_residuals" in params_fit_or_bin.keys() and params_fit_or_bin["trim_residuals"]:
        ind = index_trimmed(dh,
                            central_estimator=params_fit_or_bin["trim_central_statistic"],
                            spread_estimator=params_fit_or_bin["trim_spread_statistic"],
                            spread_coverage=params_fit_or_bin["trim_spread_coverage"],
                            iterative=params_fit_or_bin["trim_iterative"])
        logging.info(f"Trimmed {np.count_nonzero(ind)} residuals.")
        # Keep data not trimmed
        y = y[~ind]
        aspect = aspect[~ind]
        dh = dh[~ind]

    # Make an initial guess of the a, b, and c parameters
    x0 = (1, 1, np.nanmedian(dh))

    # For this type of method, the procedure can only be fit, or bin + fit (binning alone does not estimate parameters)
    if params_fit_or_bin["fit_or_bin"] not in ["fit", "bin_and_fit"]:
        raise ValueError("Nuth and Kääb method only supports 'fit' or 'bin_and_fit'.")

    # Define fit and bin parameters
    params_fit_or_bin["fit_func"] = _nuth_kaab_fit_func
    params_fit_or_bin["nd"] = 1
    params_fit_or_bin["bias_var_names"] = ["aspect"]

    # Run bin and fit, returning dataframe of binning and parameters of fitting
    _, results = _bin_or_and_fit_nd(
        fit_or_bin=params_fit_or_bin["fit_or_bin"],
        params_fit_or_bin=params_fit_or_bin,
        values=y,
        bias_vars={"aspect": aspect},
        x0=x0,
    )
    # Mypy: having results as "None" is impossible, but not understood through overloading of _bin_or_and_fit_nd...
    assert results is not None
    easting_offset = results[0] * np.sin(results[1])
    northing_offset = results[0] * np.cos(results[1])
    vertical_offset = results[2] * np.nanmean(slope_tan)

    return easting_offset, northing_offset, vertical_offset


def _nuth_kaab_aux_vars(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
) -> tuple[NDArrayf, NDArrayf]:
    """
    Deriving slope tangent and aspect auxiliary variables expected by the Nuth and Kääb (2011) algorithm.

    :return: Slope tangent and aspect (radians).
    """

    def _calculate_slope_and_aspect_nuthkaab(dem: NDArrayf) -> tuple[NDArrayf, NDArrayf]:
        """
        Calculate the tangent of slope and aspect of a DEM, in radians, as needed for the Nuth & Kaab algorithm.

        For now, this method using the gradient is more efficient than slope/aspect derived in the terrain module.

        :param dem: A numpy array of elevation values.

        :returns:  The tangent of slope and aspect (in radians) of the DEM.
        """

        # Gradient implementation
        # # Calculate the gradient of the slope
        gradient_y, gradient_x = np.gradient(dem)
        slope_tan = np.sqrt(gradient_x**2 + gradient_y**2)
        aspect = np.arctan2(-gradient_x, gradient_y)
        aspect += np.pi

        # Terrain module implementation
        # slope, aspect = xdem.terrain.get_terrain_attribute(
        #     dem, attribute=["slope", "aspect"], resolution=1, degrees=False
        # )
        # slope_tan = np.tan(slope)
        # aspect = (aspect + np.pi) % (2 * np.pi)

        return slope_tan, aspect

    # If inputs are both point clouds, raise an error
    if isinstance(ref_elev, gpd.GeoDataFrame) and isinstance(tba_elev, gpd.GeoDataFrame):

        raise TypeError(
            "The Nuth and Kääb (2011) coregistration does not support two point clouds, one elevation "
            "dataset in the pair must be a DEM."
        )

    # If inputs are both rasters, derive terrain attributes from ref and get 2D dh interpolator
    elif isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):

        # Derive slope and aspect from the reference as default
        slope_tan, aspect = _calculate_slope_and_aspect_nuthkaab(ref_elev)

    # If inputs are one raster and one point cloud, derive terrain attribute from raster and get 1D dh interpolator
    else:

        if isinstance(ref_elev, gpd.GeoDataFrame):
            rst_elev = tba_elev
        else:
            rst_elev = ref_elev

        # Derive slope and aspect from the raster dataset
        slope_tan, aspect = _calculate_slope_and_aspect_nuthkaab(rst_elev)

    return slope_tan, aspect


def _nuth_kaab_iteration_step(
    coords_offsets: tuple[float, float, float],
    dh_interpolator: Callable[[float, float], NDArrayf],
    slope_tan: NDArrayf,
    aspect: NDArrayf,
    res: tuple[int, int],
    params_fit_bin: InFitOrBinDict,
) -> tuple[tuple[float, float, float], dict[str, float], None]:
    """
    Iteration step of Nuth and Kääb (2011).

    Returns newly incremented coordinate offsets, and new statistic to compare to tolerance to reach.

    :param coords_offsets: Coordinate offsets at this iteration (easting, northing, vertical) in georeferenced unit.
    :param dh_interpolator: Interpolator returning elevation differences at the subsampled points for a certain
        horizontal offset (see _preprocess_pts_rst_subsample_interpolator).
    :param slope_tan: Array of slope tangent.
    :param aspect: Array of aspect.
    :param res: Resolution of DEM.

    :return X/Y/Z offsets, Horizontal tolerance.
    """

    # Calculate the elevation difference with offsets
    dh_step = dh_interpolator(coords_offsets[0], coords_offsets[1])
    # Tests show that using the median vertical offset significantly speeds up the algorithm compared to
    # using the vertical offset output of the fit function below
    vshift = np.nanmean(dh_step)
    dh_step -= vshift

    # Interpolating with an offset creates new invalid values, so the subsample is reduced
    # TODO: Add an option to re-subsample at every iteration step?
    mask_valid = np.isfinite(dh_step)
    if np.count_nonzero(mask_valid) == 0:
        raise ValueError(
            "The subsample contains no more valid values. This can happen is the horizontal shift to "
            "correct is very large, or if the algorithm diverged. To ensure all possible points can "
            "be used at any iteration step, use subsample=1."
        )
    dh_step = dh_step[mask_valid]
    slope_tan = slope_tan[mask_valid]
    aspect = aspect[mask_valid]

    # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
    easting_offset, northing_offset, _ = _nuth_kaab_bin_fit(
        dh=dh_step, slope_tan=slope_tan, aspect=aspect, params_fit_or_bin=params_fit_bin,
    )

    # Increment the offsets by the new offset
    new_coords_offsets = (
        coords_offsets[0] + easting_offset * res[0],
        coords_offsets[1] + northing_offset * res[1],
        float(vshift),
    )

    # Compute statistic on offset to know if it reached tolerance
    # The easting and northing are here in pixels because of the slope/aspect derivation
    offset_horizontal_translation = np.sqrt(easting_offset**2 + northing_offset**2)

    step_statistics = {"translation": offset_horizontal_translation}

    return new_coords_offsets, step_statistics, None


def nuth_kaab(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    tolerance_translation: float,
    max_iterations: int,
    params_fit_or_bin: InFitOrBinDict,
    params_random: InRandomDict,
    z_name: str,
    weights: NDArrayf | None = None,
    **kwargs: Any,
) -> tuple[tuple[float, float, float], int, OutIterativeDict]:
    """
    Nuth and Kääb (2011) iterative coregistration.

    This function subsamples input data, then runs Nuth and Kääb iteration steps to optimize its fit function until
    convergence or a maximum of iterations is reached.

    :return: Final estimated offset: east, north, vertical (in georeferenced units).
    """
    logging.info("Running Nuth and Kääb (2011) coregistration")

    transform = ref_transform if ref_transform is not None else tba_transform

    # Check that DEM CRS is projected, otherwise slope is not correctly calculated
    if not crs.is_projected:
        raise NotImplementedError(
            f"NuthKaab coregistration only works with a projected CRS, current CRS is {crs}. Reproject "
            f"your DEMs with DEM.reproject() in a local projected CRS such as UTM, that you can find "
            f"using DEM.get_metric_crs()."
        )

    # First, derive auxiliary variables of Nuth and Kääb (slope tangent, and aspect) for any point-raster input
    slope_tan, aspect = _nuth_kaab_aux_vars(ref_elev=ref_elev, tba_elev=tba_elev)

    # Add slope tangents near zero to outliers, to avoid infinite values from later division by slope tangent, and to
    # subsample the right number of subsample points straight ahead
    mask_zero_slope_tan = np.isclose(slope_tan, 0)
    slope_tan[mask_zero_slope_tan] = np.nan

    # Then, perform preprocessing: subsampling and interpolation of inputs and auxiliary vars at same points
    aux_vars = {"slope_tan": slope_tan, "aspect": aspect}  # Wrap auxiliary data in dictionary to use generic function
    sub_dh_interpolator, sub_aux_vars, subsample_final = _subsample_rst_pts_interpolator(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        aux_vars=aux_vars,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
    )

    # Initialise east, north and vertical offset variables (these will be incremented up and down)
    initial_offset = (0.0, 0.0, 0.0)
    # Resolution
    res = _res(transform)
    # Iterate through method of Nuth and Kääb (2011) until tolerance or max number of iterations is reached
    assert sub_aux_vars is not None  # Mypy: dictionary cannot be None here
    constant_inputs = (sub_dh_interpolator, sub_aux_vars["slope_tan"], sub_aux_vars["aspect"], res, params_fit_or_bin)
    final_offsets, _, output_iterative, _ = _iterate_method(
        method=_nuth_kaab_iteration_step,
        iterating_input=initial_offset,
        constant_inputs=constant_inputs,
        tolerances={"translation": tolerance_translation},
        max_iterations=max_iterations,
    )

    return final_offsets, subsample_final, output_iterative


####################
# 2/ Dh minimization
####################


def _dh_minimize_fit_func(
    coords_offsets: tuple[float, float],
    dh_interpolator: Callable[[float, float], NDArrayf],
) -> NDArrayf:
    """
    Fitting function of dh minimization method, returns the NMAD of elevation differences.

    :param coords_offsets: Coordinate offsets at this iteration (easting, northing) in georeferenced unit.
    :param dh_interpolator: Interpolator returning elevation differences at the subsampled points for a certain
        horizontal offset (see _preprocess_pts_rst_subsample_interpolator).
    :returns: NMAD of residuals.
    """

    # Calculate the elevation difference
    dh = dh_interpolator(coords_offsets[0], coords_offsets[1]).flatten()

    return dh


def _dh_minimize_fit(
    dh_interpolator: Callable[[float, float], NDArrayf],
    params_fit_or_bin: InFitOrBinDict,
    **kwargs: Any,
) -> tuple[float, float, float]:
    """
    Optimize the statistical dispersion of the elevation differences residuals.

    :param dh_interpolator: Interpolator returning elevation differences at the subsampled points for a certain
        horizontal offset (see _preprocess_pts_rst_subsample_interpolator).
    :param params_fit_or_bin: Parameters for fitting or binning.

    :return: Optimized offsets (easing, northing, vertical) in georeferenced unit.
    """
    # Define partial function
    loss_func = params_fit_or_bin["fit_loss_func"]

    def fit_func(coords_offsets: tuple[float, float]) -> np.floating[Any]:
        return loss_func(_dh_minimize_fit_func(coords_offsets=coords_offsets, dh_interpolator=dh_interpolator))

    # Initial offset near zero
    init_offsets = (0, 0)

    # Default parameters depending on optimizer used
    if params_fit_or_bin["fit_minimizer"] == scipy.optimize.minimize:
        if "method" not in kwargs.keys():
            kwargs.update({"method": "Nelder-Mead"})
            # This method has trouble when initialized with 0,0, so defaulting to 1,1
            # (tip from Simon Gascoin: https://github.com/GlacioHack/xdem/pull/595#issuecomment-2387104719)
            init_offsets = (1, 1)

    results = params_fit_or_bin["fit_minimizer"](fit_func, init_offsets, **kwargs)

    # Get final offsets with the right sign direction
    offset_east = -results.x[0]
    offset_north = -results.x[1]
    offset_vertical = float(np.nanmedian(dh_interpolator(-offset_east, -offset_north)))

    return offset_east, offset_north, offset_vertical


def dh_minimize(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    area_or_point: Literal["Area", "Point"] | None,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    z_name: str,
    weights: NDArrayf | None = None,
    **kwargs: Any,
) -> tuple[tuple[float, float, float], int]:
    """
    Elevation difference minimization coregistration method, for any point-raster or raster-raster input,
    including subsampling and interpolation to the same points.

    :return: Final estimated offset: east, north, vertical (in georeferenced units).
    """

    logging.info("Running dh minimization coregistration.")

    transform = ref_transform if ref_transform is not None else tba_transform

    # Perform preprocessing: subsampling and interpolation of inputs and auxiliary vars at same points
    dh_interpolator, _, subsample_final = _subsample_rst_pts_interpolator(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        area_or_point=area_or_point,
        z_name=z_name,
    )

    # Perform fit
    # TODO: To match original implementation, need to add back weight support for point data
    final_offsets = _dh_minimize_fit(dh_interpolator=dh_interpolator, params_fit_or_bin=params_fit_or_bin)

    return final_offsets, subsample_final


###################
# 3/ Vertical shift
###################


def vertical_shift(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    params_random: InRandomDict,
    vshift_reduc_func: Callable[[NDArrayf], np.floating[Any]],
    z_name: str,
    weights: NDArrayf | None = None,
    **kwargs: Any,
) -> tuple[float, int]:
    """
    Vertical shift coregistration, for any point-raster or raster-raster input, including subsampling.
    """

    logging.info("Running vertical shift coregistration")

    # Pre-process point-raster inputs to the same subsampled points
    sub_ref, sub_tba, _ = _subsample_rst_pts(
        subsample=params_random["subsample"],
        random_state=params_random["random_state"],
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        ref_transform=ref_transform,
        tba_transform=tba_transform,
        sampling_strategy="same_xy",  # This needs to be enforced for a vertical shift based on mean elevation differences
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
    )
    # Get elevation difference
    dh = sub_ref[2] - sub_tba[2]

    # Get vertical shift on subsa weights if those were provided.
    vshift = float(vshift_reduc_func(dh) if weights is None else vshift_reduc_func(dh, weights))  # type: ignore

    # TODO: We might need to define the type of bias_func with Callback protocols to get the optional argument,
    # TODO: once we have the weights implemented

    logging.info("Vertical shift estimated")

    # Get final subsample size
    subsample_final = len(sub_ref)

    return vshift, subsample_final


############################
# 4/ Iterative closest point
############################

def _collapse_weights_to_points(weights: NDArrayf | None, n: int) -> NDArrayf | None:
    """
    Convert various weight shapes to per-point weights of shape (N,).

    Accepts (N,), (N,1) or (N,3)/(3,N) collapsed to (N,) using a mean across xyz.
    """
    if weights is None:
        return None

    w = np.asarray(weights)

    if w.ndim == 1:
        if w.shape[0] != n:
            raise ValueError(f"weights has length {w.shape[0]} but expected {n}.")
        return w.astype(float, copy=False)

    if w.ndim == 2:
        if w.shape == (n, 1):
            return w[:, 0].astype(float, copy=False)
        if w.shape == (n, 3):
            return np.nanmean(w, axis=1).astype(float, copy=False)
        if w.shape == (3, n):
            return np.nanmean(w, axis=0).astype(float, copy=False)
    raise ValueError(
        f"Unsupported weights shape {w.shape}. Expected (N,), (N,1), (N,3) or (3,N) where N={n}."
    )


def _icp_fit_func(
    inputs: tuple[NDArrayf, NDArrayf, NDArrayf | None],
    t1: float,
    t2: float,
    t3: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    method: Literal["point-to-point", "point-to-plane"],
    linearized: bool = False,
    weights: NDArrayf | None = None,
    axis_weights: NDArrayf | None = None,
) -> NDArrayf:
    """
    Fit function of ICP, a rigid transformation with 6 parameters (3 translations and 3 rotations) between closest
    points (that are fixed for this optimization and located at the same indexes, and update at each iterative step).

    If method is "point-to-plane" and linearized=True, returns the Low (2004) linearized residual r = A x - B
    for x = [alpha1, alpha2, alpha3, t1, t2, t3] (internally mapped from (t, alpha) args).
    See Low (2004), https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf for the
    "point-to-plane" approximation.

    If weights is provided, applies weighted LS as sqrt(w_i) * r_i (per correspondence).

    :param inputs: Constant input for the fit function: three arrays of size 3xN, for the reference point cloud, the
        to-be-aligned point cloud ordered for its nearest points to the reference, and the plane normals.
    :param t1: Translation in X.
    :param t2: Translation in Y.
    :param t3: Translation in Z.
    :param alpha1: Rotation around X.
    :param alpha2: Rotation around Y.
    :param alpha3: Rotation around Z.
    :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
        that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
        projected on normals.

    :return: Array of distances between closest points.
    """

    # Get inputs
    ref, tba, norm = inputs
    n_pts = ref.shape[1]

    # Per point weights
    w_point = _collapse_weights_to_points(weights, n_pts)
    w_sqrt = None if w_point is None else np.sqrt(w_point)

    # Axis weights
    if axis_weights is None:
        sqrt_w_axis = None
        w_axis = None
    else:
        w_axis = np.asarray(axis_weights, dtype=float).reshape(3)
        sqrt_w_axis = np.sqrt(np.clip(w_axis, 0.0, None))

    # 1/ Point-to-point method, from Besl and McKay (1992), https://doi.org/10.1117/12.57955
    if method == "point-to-point":
        # No linearization possible, get the exact transformation
        matrix = matrix_from_translations_rotations(t1, t2, t3, alpha1, alpha2, alpha3, use_degrees=False)
        trans_tba = _apply_matrix_pts_mat(mat=tba, matrix=matrix)

        # Vector residuals (3, N)
        r = (trans_tba - ref)

        # Apply axis weight for anisotropic X/Y versus Z (3, N)
        if sqrt_w_axis is not None:
            r = r * sqrt_w_axis[:, None]

        # Apply per-point weights (3, N)
        if w_sqrt is not None:
            r = r * w_sqrt[None, :]

        # Return 1D residual vector
        res = r.reshape(-1)

    # 2/ Point-to-plane used the normals, from Chen and Medioni (1992), https://doi.org/10.1016/0262-8856(92)90066-C
    # A priori, this method is faster based on Rusinkiewicz and Levoy (2001), https://doi.org/10.1109/IM.2001.924423
    elif method == "point-to-plane":
        assert norm is not None
        # If using linearized point-to-plane (Low, 2004)
        if linearized:
            # Work in Nx3 for dot products
            p = tba.T   # (N,3)
            q = ref.T   # (N,3)
            n = norm.T  # (N,3)

            # Build residual directly without forming A explicitly: A_rot = cross(p, n); A_trans = n; B = dot(n, q - p)
            A_rot = np.cross(p, n)  # (N,3)
            A_trans = n  # (N,3)
            B = np.sum(n * (q - p), axis=1)  # (N,)
            alpha = np.array([alpha1, alpha2, alpha3], dtype=float)  # (3,)
            t = np.array([t1, t2, t3], dtype=float)  # (3,)
            res = (A_rot @ alpha) + (A_trans @ t) - B  # (N,)

        else:
            # Otherwise, use nonlinear point-to-plane
            matrix = matrix_from_translations_rotations(t1, t2, t3, alpha1, alpha2, alpha3, use_degrees=False)
            trans_tba = _apply_matrix_pts_mat(mat=tba, matrix=matrix)

            # Distance projected on 3D normal
            diffs = (trans_tba - ref) * norm
            res = np.sum(diffs, axis=0)  # shape (N,)

            # Axis weighting for plane residuals
            if w_axis is not None:
                # The norm is (3, N)
                nWn = np.sum((norm * norm) * w_axis[:, None], axis=0)  # (N,)
                plane_sqrt = np.sqrt(np.clip(nWn, 0.0, None))
                res = plane_sqrt * res

            # Apply per-point weight
            if w_sqrt is not None:
                res = w_sqrt * res
    else:
        raise ValueError("ICP method must be 'point-to-point' or 'point-to-plane'.")

    return res

def _icp_fit(
    ref: NDArrayf,
    tba: NDArrayf,
    norms: NDArrayf | None,
    weights: NDArrayf | None,
    axis_weights: NDArrayf | None,
    method: Literal["point-to-point", "point-to-plane"],
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
    linearized: bool,
    **kwargs: Any,
) -> NDArrayf:
    """
    Optimization of ICP fit function, using either any optimizer or specific linearized approximations for ICP.

    Returns affine transform optimized for this iteration.

    :param ref: Reference point cloud as 3xN array.
    :param tba: To-be-aligned point cloud as 3xN array.
    :param norms: Plane normals as 3xN array.
    :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
        that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
        projected on normals.
    :param params_fit_or_bin: Dictionary of parameters for fitting or binning.
    :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
        rotation as default.
    :param **kwargs: Keyword arguments passed to fit optimizer.

    :return: Affine transform matrix.
    """

    # Trim if required
    if "trim_residuals" in params_fit_or_bin.keys() and params_fit_or_bin["trim_residuals"]:
        res = _icp_fit_func((ref, tba, norms), 0, 0, 0, 0, 0, 0, method=method)
        ind = index_trimmed(res, central_estimator=params_fit_or_bin["trim_central_statistic"],
                            spread_estimator=params_fit_or_bin["trim_spread_statistic"],
                            spread_coverage=params_fit_or_bin["trim_spread_coverage"],
                            iterative=params_fit_or_bin["trim_iterative"])
        logging.info(f"Trimmed {np.count_nonzero(ind)} residuals.")
        # Keep data not trimmed
        ref = ref[:, ~ind]
        tba = tba[:, ~ind]
        norms = norms[:, ~ind]

    # Group inputs into a single array
    inputs = (ref, tba, norms)

    # Define loss function
    loss_func = params_fit_or_bin["fit_loss_func"]

    # With rotation
    if not only_translation:

        def fit_func(offsets: tuple[float, float, float, float, float, float]) -> NDArrayf:
            return _icp_fit_func(
                inputs=inputs,
                t1=offsets[0],
                t2=offsets[1],
                t3=offsets[2],
                alpha1=offsets[3],
                alpha2=offsets[4],
                alpha3=offsets[5],
                method=method,
                linearized=linearized,
                weights=weights,
                axis_weights=axis_weights
            )

        # Initial offset near zero
        init_offsets = np.zeros(6)

    # Without rotation
    else:

        def fit_func(offsets: tuple[float, float, float, float, float, float]) -> NDArrayf:
            return _icp_fit_func(
                inputs=inputs,
                t1=offsets[0],
                t2=offsets[1],
                t3=offsets[2],
                alpha1=0.0,
                alpha2=0.0,
                alpha3=0.0,
                method=method,
                linearized=linearized,
                weights=weights,
                axis_weights=axis_weights
            )

        # Initial offset near zero
        init_offsets = np.zeros(3)

    results = params_fit_or_bin["fit_minimizer"](fit_func, init_offsets, **kwargs, loss=loss_func)
    # Mypy: having results as "None" is impossible, but not understood through overloading of _bin_or_and_fit_nd...
    assert results is not None
    # Build matrix out of optimized parameters
    matrix = matrix_from_translations_rotations(*results.x, use_degrees=False)  # type: ignore

    return matrix


def _icp_iteration_step(
    matrix: NDArrayf,
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    norms: NDArrayf,
    axis_weights: NDArrayf,
    ref_epc_nearest_tree: scipy.spatial.KDTree,
    params_fit_or_bin: InFitOrBinDict,
    method: Literal["point-to-point", "point-to-plane"],
    picky: bool,
    linearized: bool,
    only_translation: bool,
) -> tuple[NDArrayf, dict[str, float], None]:
    """
    Iteration step of Iterative Closest Point coregistration.

    Returns optimized affine matrix and statistics (mostly offsets) to compare to tolerances for this iteration step.

    :param matrix: Affine transform matrix.
    :param ref_epc: Reference point cloud as 3xN array.
    :param tba_epc: To-be-aligned point cloud as 3xN array.
    :param norms: Plane normals as 3xN array.
    :param ref_epc_nearest_tree: Nearest neighbour mapping to reference point cloud as scipy.KDTree.
    :param params_fit_or_bin: Dictionary of parameters for fitting or binning.
    :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
        that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
        projected on normals.
    :param picky: Whether to use the duplicate removal for pairs of closest points of Zinsser et al. (2003).
    :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
        rotation as default.

    :return Affine matrix, Iteration statistics to compare to tolerances.
    """

    # Apply transform matrix from previous steps
    trans_tba_epc = _apply_matrix_pts_mat(tba_epc, matrix=matrix)

    # Create nearest neighbour tree from reference elevations, and query for transformed point cloud
    dists, ind = ref_epc_nearest_tree.query(trans_tba_epc.T, k=1)

    # Picky ICP: Remove duplicates of transformed points with the same closest reference points
    # Keep only the one with minimum distance
    if picky:
        init_len = len(ind)
        df = pd.DataFrame(data={"ind": ind, "dists": dists})
        ind_tba = df.groupby(["ind"]).idxmin()["dists"].values
        logging.info(f"Picky ICP duplicate removal: Reducing from {init_len} to {len(ind)} point pairs.")
    # In case the number of points is different for ref and tba (can't happen with current subsampling)
    else:
        ind_tba = ind < ref_epc.shape[1]

    # Reference index is the original indexed by the other
    ind_ref = ind[ind_tba]

    # Index points to get nearest
    step_ref = ref_epc[:, ind_ref]
    step_trans_tba = trans_tba_epc[:, ind_tba]

    if method == "point-to-plane":
        step_normals = norms[:, ind_ref]
    else:
        step_normals = None

    # Fit to get new step transform
    step_matrix = _icp_fit(
        ref=step_ref,
        tba=step_trans_tba,
        norms=step_normals,
        params_fit_or_bin=params_fit_or_bin,
        method=method,
        only_translation=only_translation,
        linearized=linearized,
        weights=None,
        axis_weights=axis_weights,
    )

    # Increment transformation matrix by step
    new_matrix = step_matrix @ matrix

    # Compute statistics to know if they reached tolerance
    # (offsets in translation/rotation, but can also be other statistics)
    translations = step_matrix[:3, 3]
    offset_translation = np.sqrt(np.sum(translations ** 2))
    rotations = step_matrix[:3, :3]
    offset_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    step_statistics = {"translation": offset_translation, "rotation": offset_rotation}

    return new_matrix, step_statistics, None

def icp(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    crs: rio.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    max_iterations: int,
    tolerance_translation: float | None,
    tolerance_rotation: float | None,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    method: Literal["point-to-point", "point-to-plane"] = "point-to-plane",
    sampling_strategy: Literal["independent", "same_xy", "iterative_same_xy"] = "same_xy",
    picky: bool = False,
    linearized: bool = False,
    only_translation: bool = False,
    anisotropic: Literal["xy_vs_z", "per_axis"] | None = "xy_vs_z",
    standardize: bool = True,
) -> tuple[NDArrayf, tuple[float, float, float], int, OutIterativeDict]:
    """
    Main function for Iterative Closest Point coregistration.

    This function subsamples input data, then runs ICP iteration steps to optimize its fit function until
    convergence or a maximum of iterations is reached.

    Based on Besl and McKay (1992), https://doi.org/10.1117/12.57955 for "point-to-point" and on
    Chen and Medioni (1992), https://doi.org/10.1016/0262-8856(92)90066-C for "point-to-plane".

    The function assumes we have two DEMs, or DEM and an elevation point cloud, in the same CRS.
    The normals for "point-to-plane" are computed on the DEM for speed.

    :return: Affine transform matrix, Centroid, Subsample size.
    """

    # Derive centroid and scale ahead of any potential iterative sampling loop, and scale translation tolerance
    centroid, scale = _get_centroid_scale(ref_elev=ref_elev, transform=ref_transform, z_name=z_name)
    if not standardize:
        scale = 1
    tolerance_translation /= scale

    # Initial parameters and tolerances as dictionary
    init_matrix = np.eye(4)  # Initial matrix is the identity transform
    tolerances = {"translation": tolerance_translation, "rotation": tolerance_rotation}

    # Derive normals if method is point-to-plane, otherwise not
    # (This has to happen before subsampling, because it creates new invalid data
    # and we wanted a fixed valid data size for subsampling)
    invert_output = False
    if method == "point-to-plane":

        # Fixed reference must be an array to derive normals
        if isinstance(ref_elev, np.ndarray):
            fixed_elev = "ref"
        elif isinstance(tba_elev, np.ndarray):
            fixed_elev = "tba"
        else:
            raise TypeError(
                "point-to-plane ICP requires one input to be a DEM array to derive normals."
            )

        # The normals must always be derived relative to a "fixed reference" (we don't rotate them during iterations),
        # so internally we'll need to invert inputs if 'tba' is the array, then invert the matrix at the end
        if fixed_elev == "tba":
            invert_output = True
            ref_elev, tba_elev = tba_elev, ref_elev
            ref_transform, tba_transform = tba_transform, ref_transform

        # Now ref is always the array
        aux_vars = _dem_normals_curvature(ref_elev, ref_transform)
        aux_tied_to = "ref"

    else:
        aux_vars = None
        aux_tied_to = "ref"

    # If we iterate the sampling, we re-define maximum iterations to be in the outside loop,
    # running a single iteration with every sample
    if sampling_strategy == "iterative_same_xy":
        in_loop_max_it = max_iterations
        out_loop_max_it = max_iterations
        in_loop_sampling_strategy = "same_xy"
    # Otherwise, we run a single iteration of outside loop, as if there was no loop
    else:
        in_loop_max_it = max_iterations
        out_loop_max_it = 1
        in_loop_sampling_strategy = sampling_strategy

    # Initialize values for iterative sampling loop (tba_elev updates out_step_matrix that updates tba_elev)
    list_iteration_stats = []
    final_matrix = np.eye(4)
    tba_elev_orig = tba_elev.copy()
    tba_transform_orig = tba_transform
    for i in range(out_loop_max_it):

        # Pre-process point-raster inputs to the same subsampled points
        ref_epc, tba_epc, sub_aux = _subsample_rst_pts(
            subsample=params_random["subsample"],
            random_state=params_random["random_state"],
            ref_elev=ref_elev,
            tba_elev=tba_elev,  # For iterative sampling, tba_elev is updated below
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,  # For iterative sampling, tba_transform can be updated below
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            sampling_strategy=in_loop_sampling_strategy,
            aux_vars=aux_vars,
            aux_tied_to=aux_tied_to,
        )

        # Re-stack normals if defined
        if sub_aux is not None:
            norms = np.vstack((sub_aux["nx"], sub_aux["ny"], sub_aux["nz"]))
        else:
            norms = None

        # Remove centroid and standardize to facilitate numerical convergence
        ref_epc, tba_epc = _standardize_epc(ref_epc, tba_epc, centroid=centroid, scale=scale)

        # Calculate standard deviation for axes for optional anisotropic weighting
        w_axis = _axis_weights_from_epc(ref_epc, anisotropic=anisotropic)

        # Define search tree outside of loop for performance
        ref_epc_nearest_tree = scipy.spatial.KDTree(ref_epc.T)

        # Iterate through method until tolerance or max number of iterations is reached
        constant_inputs = (
            ref_epc,
            tba_epc,  # For iterative sampling, tba_epc is updated above
            norms,
            w_axis,
            ref_epc_nearest_tree,
            params_fit_or_bin,
            method,
            picky,
            linearized,
            only_translation,
        )
        out_step_matrix, new_stats, output_iterative, _ = _iterate_method(
            method=_icp_iteration_step,
            iterating_input=init_matrix,
            constant_inputs=constant_inputs,
            tolerances=tolerances,
            max_iterations=in_loop_max_it,
        )

        # We destandardize and update the matrix
        out_step_matrix[:3, 3] *= scale
        final_matrix = out_step_matrix @ final_matrix

        # If we apply iterative sampling, we need to do additional things each loop
        if sampling_strategy == "iterative_same_xy":
            # We update the to-be-aligned elevation (input before subsampling) here directly, so that X/Y sampling will be updated
            if isinstance(tba_elev, np.ndarray):
                tba_elev, tba_transform = _apply_matrix_rst(
                    tba_elev_orig, transform=tba_transform_orig, centroid=centroid, matrix=final_matrix
                )
            else:
                tba_elev = _apply_matrix_pts(tba_elev_orig, matrix=final_matrix, centroid=centroid, z_name=z_name)
            # We update the iterative output with the outside loop iteration number
            it_stats = output_iterative["iteration_stats"]
            it_stats["iteration"] = i + 1
            list_iteration_stats.append(it_stats)

            # Check exit condition was reached in inside loop
            if all(new_stats[k] < tolerances[k] if k is not None else True for k in tolerances.keys()):
                logging.debug("Exiting outside loop of iterative sampling as statistics have all reached tolerance.")
                break

    # Over-write iterative output for iterative sampling
    if sampling_strategy == "iterative_same_xy":
        iteration_stats = pd.concat(list_iteration_stats)
        output_iterative: OutAffineDict = {"last_iteration": i + 1, "iteration_stats": iteration_stats}

    # Get subsample size
    # TODO: Support reporting different number of subsamples when independent?
    subsample_final = min(ref_epc.shape[1], tba_epc.shape[1])

    # If we solved the inverted problem (fixed surface was originally tba), we invert to keep the external convention
    if method == "point-to-plane" and invert_output:
        final_matrix = invert_matrix(final_matrix)

    return final_matrix, centroid, subsample_final, output_iterative


#########################
# 5/ Coherent Point Drift
#########################

def _plane_ratio_from_curvature(curvature: NDArrayf, lsg_lambda: float, max_plane_ratio: float) -> NDArrayf:
    """
    Port of Matlab pre-calculation:
      a = max(2 ./ (1 + exp(lambda .* (3 - 1 ./ Curvature))) - 1, 0) .* alimit;
    """
    curv = np.asarray(curvature, dtype=float)
    a = 2.0 / (1.0 + np.exp(float(lsg_lambda) * (3.0 - 1.0 / (curv + 1e-12)))) - 1.0
    a = np.maximum(a, 0.0) * float(max_plane_ratio)
    return a

# CPD cache precomputation (reused across iterations)
#####################################################
def _cpd_precompute(
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    weight_cpd: float,
    *,
    axis_weights: NDArrayf | None = None,
    lsg: bool = False,
    # LSG-CPD options
    ref_normals: NDArrayf | None = None,
    curvature: NDArrayf | None = None,
    neighbours: int = 10,
    max_plane_ratio: float = 30.0,
    truncation_threshold: float = 0.19,
    lsg_lambda: float = 0.2,
) -> dict[str, Any]:
    """
    Precompute quantities that can be reused across CPD iterations.

    This is meant to be called once per subsample set (i.e. inside your outer sampling loop),
    then passed through `constant_inputs` of _iterate_method to avoid recomputing invariant terms.

    :return: Cache dictionary.
    """
    X = ref_epc.T  # (N,3)
    Y = tba_epc.T  # (M,3)

    # Get shape of inputs
    N, D = X.shape
    M, _ = Y.shape

    if axis_weights is None:
        axis_weights = np.ones(3, dtype=float)
    axis_weights = np.asarray(axis_weights, dtype=float).reshape(3)

    cache: dict[str, Any] = {
        "classic": {
            "X": X,
            "Y": Y,
            "N": N,
            "M": M,
            "D": D,
            "dtype": X.dtype,
            "w": axis_weights,
        },
        "lsg": None,
        "weight_cpd": float(weight_cpd),
        "truncation_threshold": float(truncation_threshold),
    }

    if not lsg:
        return cache

    # We precompute geometry-heavy terms independent of sigma2.
    V = float((X[:, 0].max() - X[:, 0].min())
              * (X[:, 1].max() - X[:, 1].min())
              * (X[:, 2].max() - X[:, 2].min()))
    V = max(V, 1e-12)

    # Normals and curvature
    if ref_normals is None or curvature is None:
        normals_cpu, curv = _epc_normals_curvature(X, neighbours=neighbours)
        if ref_normals is None:
            ref_normals = normals_cpu
        if curvature is None:
            curvature = curv
    else:
        ref_normals = np.asarray(ref_normals, dtype=float)
        if ref_normals.shape != X.shape:
            raise ValueError("ref_normals must have shape (N,3) matching ref_epc.")
        ref_normals /= np.linalg.norm(ref_normals, axis=1, keepdims=True) + 1e-12

        curvature = np.asarray(curvature, dtype=float).reshape(-1)
        if curvature.shape[0] != N:
            raise ValueError("curvature must have shape (N,) matching ref_epc.")

    # Plane ratio a(m)
    a = _plane_ratio_from_curvature(curvature, lsg_lambda=lsg_lambda, max_plane_ratio=max_plane_ratio)  # (N,)

    w = cache["classic"]["w"]  # (3,)
    W = np.diag(w)  # (3,3)
    Wn = ref_normals * w[None, :]  # (N,3)
    nWn = np.sum(ref_normals * Wn, axis=1)  # (N,) = n^T W n
    nWn = np.clip(nWn, 1e-12, None)

    # Precompute invSigma per target point m: invSigma_m = a(m) * n n^T + I
    invSigma = np.empty((N, 3, 3), dtype=float)
    for i in range(N):
        wn = Wn[i].reshape(3, 1)  # (3,1) = W n
        invSigma[i] = W + (float(a[i]) / float(nWn[i])) * (wn @ wn.T)

    invSigma_flat = invSigma.reshape(N, 9)  # (N,9)
    x_invSigma = np.einsum("ni,nij->nj", X, invSigma)                # (N,3)
    x_invSigma_x = np.einsum("ni,nij,nj->n", X, invSigma, X)         # (N,)

    # Precompute dot(x, n)
    X_normal = np.sum(X * Wn, axis=1)  # (N,)

    # Precompute X_Y = ||x||^2 + ||y||^2 (N,M) (Matlab uses this structure)
    X3 = X.T  # (3,N)
    Y3 = Y.T  # (3,M)
    X_X2 = np.sum((X * X) * w[None, :], axis=1)  # (N,)
    Y_Y2 = np.sum(Y3 * Y3, axis=0)   # (M,)
    X_Y = X_X2[:, None] + Y_Y2[None, :]  # (N,M)

    # pi(m) base (before sigma2-dependent scaling and outlier reweighting)
    f_X_base = np.ones(N, dtype=float) / N
    vol = np.sqrt(a + 1.0)
    f_X_scaled = f_X_base * vol

    # Confidence weights (Matlab has optional confidence filtering; we default to ones)
    confidence_Y = np.ones(M, dtype=float)

    cache["lsg"] = {
        "X": X,                     # target/GMM (N,3)
        "Y": Y,                     # source/moving (M,3)
        "N": N,
        "M": M,
        "V": V,
        "w": axis_weights,
        "Wn": Wn,
        "nWn": nWn,
        "ref_normals": ref_normals, # (N,3)
        "curvature": curvature,     # (N,)
        "a": a,                     # (N,)
        "invSigma_flat": invSigma_flat,
        "x_invSigma": x_invSigma,
        "x_invSigma_x": x_invSigma_x,
        "X_normal": X_normal,
        "X_X2": X_X2,
        "f_X_scaled": f_X_scaled,   # (N,)
        "confidence_Y": confidence_Y,
        "lsg_lambda": float(lsg_lambda),
        "max_plane_ratio": float(max_plane_ratio),
        "neighbours": int(neighbours),
    }

    return cache


def _lsg_update_sigma_dependent(cache_lsg: dict[str, Any], sigma2: float, weight_cpd: float) -> dict[str, Any]:
    """
    Update LSG-CPD quantities that depend on sigma2 (and weight).

    This is called at every iteration, but it is cheap compared to recomputing normals/curvature/invSigma.
    """
    sigma2 = float(sigma2)
    N = cache_lsg["N"]
    M = cache_lsg["M"]
    V = cache_lsg["V"]
    a = cache_lsg["a"]
    f_X_scaled = cache_lsg["f_X_scaled"]
    confidence_Y = cache_lsg["confidence_Y"]

    # Estimate outlier weight (Matlab block)
    # w0 = V * w * f_Y * (2*pi*sigma2)^(-3/2) * sqrt(a+1)
    # Here, target is X (size N), source is Y (size M). We keep the same structure:
    w0 = V * float(weight_cpd) * float(f_X_scaled @ ((2.0 * np.pi * sigma2) ** (-1.5) * np.sqrt(a + 1.0)))
    w0 = float(w0 / (1.0 - float(weight_cpd) + w0))

    # Outlier mixing for source points (Matlab uses confidence_X on source; here source is Y)
    wn = (1.0 - (1.0 - w0) * confidence_Y)              # (M,)
    f_Y = (1.0 - wn) / np.clip(wn, 1e-12, None)         # (M,)

    # F_matrix = f_X .* f_Y (N,M)
    F_matrix = f_X_scaled[:, None] * f_Y[None, :]       # (N,M)

    # E-step constants
    C_const = float((2.0 * np.pi * sigma2) ** 1.5 * (1.0 / V))
    c_const = float(-1.0 / (2.0 * sigma2))

    return {"F_matrix": F_matrix, "C_const": C_const, "c_const": c_const}


# -----------------------------------------------------------------------------
# Classic CPD helpers (E-step / M-step / shrink step)
# -----------------------------------------------------------------------------

def _cpd_estep_classic(
    X: NDArrayf,
    Y: NDArrayf,
    TY: NDArrayf,
    weight_cpd: float,
    sigma2: float,
    w: NDArrayf,
) -> dict[str, NDArrayf]:
    """
    Expectation step of classic CPD.

    Returns a dictionary of precomputed quantities (P, Pt1, P1, Np, PX) used by the M-step and shrink step.
    """
    N, D = X.shape
    M, _ = Y.shape

    # Sum only over D axis for numerator
    diff2 = (X[None, :, :] - TY[:, None, :]) ** 2 * w[None, None, :]
    P = np.sum(diff2, axis=2)
    P = np.exp(-P / (2 * sigma2))

    # Re-sum over M axis for denominator
    Pden = np.sum(P, axis=0, keepdims=True)
    c = (2 * np.pi * sigma2) ** (D / 2) * weight_cpd / (1.0 - weight_cpd) * M / N
    Pden = np.clip(Pden, np.finfo(X.dtype).eps, None) + c

    P = np.divide(P, Pden)

    # Extract P subterms useful for next steps
    Pt1 = np.sum(P, axis=0)
    P1 = np.sum(P, axis=1)
    Np = np.sum(P1)
    PX = np.matmul(P, X)

    return {"P": P, "Pt1": Pt1, "P1": P1, "Np": np.asarray(Np), "PX": PX}


def _cpd_mstep_classic_fit_minimizer(
    X: NDArrayf,
    TY: NDArrayf,
    estep: dict[str, NDArrayf],
    scale: bool,
    w: NDArrayf,
    only_translation: bool,
    params_fit_or_bin: Any,  # InFitOrBinDict
) -> tuple[NDArrayf, NDArrayf, float]:
    """
    Minimization step of classic CPD using a generic fit_minimizer.

    Returns (R, t, s) for the shrink step.

    Notes:
    - This optimizes the classic CPD M-step objective directly:
        sum_{m,n} P[m,n] * || X[n] - (s R Y[m] + t) ||^2
      by returning weighted residuals sqrt(P) * (X - (s R Y + t)).
    - This is mathematically faithful but can be expensive because it creates N*M residual blocks.
    """
    N, D = X.shape
    M, _ = TY.shape

    P = estep["P"]   # (M,N) in your implementation
    P1 = estep["P1"] # (M,)
    Np = float(estep["Np"])
    PX = estep["PX"] # (N,D) since PX = P @ X

    # Get centroid of each point cloud (unchanged, needed for shrink step outputs)
    muX = np.divide(np.sum(PX, axis=0), Np)
    muY = np.divide(np.sum(np.dot(np.transpose(P), TY), axis=0), Np)

    # Subtract centroid from each point cloud
    X_hat = X - np.tile(muX, (N, 1))
    Y_hat = TY - np.tile(muY, (M, 1))
    YPY = float(np.dot(np.transpose(P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1)))

    # Derive A as in Fig. 2 (kept so shrink step remains identical)
    A = np.dot(np.transpose(X_hat), np.transpose(P))
    A = np.dot(A, Y_hat)

    # Optimization hook
    fit_minimizer: Callable[..., Any] = params_fit_or_bin["fit_minimizer"]
    loss_func = params_fit_or_bin["fit_loss_func"]

    # Precompute pair indices once (constant shape each iteration)
    # P shape is (M,N), so pairs are (m,n)
    # Weighted residual for each pair: sqrt(P[m,n]) * (X[n] - (s R Y[m] + t))
    # We'll flatten to 1D vector for least_squares compatibility.
    P_sqrt = np.sqrt(np.clip(P, 0.0, None))  # (M,N)

    # With rotation (and maybe scale)
    if not only_translation:

        if scale is True:

            def fit_func(params: NDArrayf) -> NDArrayf:
                # params = [rx, ry, rz, tx, ty, tz, log_s]
                R = Rot.from_rotvec(params[:3]).as_matrix()
                t = params[3:6]
                s = float(np.exp(params[6]))  # keep scale positive

                trans_Y = (s * (TY @ R.T)) + t.reshape(1, 3)  # (M,3)

                # Build residuals for all pairs (m,n)
                # diff[m,n,:] = X[n,:] - trans_Y[m,:]
                diff = X[None, :, :] - trans_Y[:, None, :]     # (M,N,3)
                diff = diff * np.sqrt(w[None, None, :])
                r = P_sqrt[:, :, None] * diff                 # (M,N,3)
                return r.reshape(-1)

            init_offsets = np.zeros(7, dtype=float)  # log_s=0 => s=1

        else:

            def fit_func(params: NDArrayf) -> NDArrayf:
                # params = [rx, ry, rz, tx, ty, tz]
                R = Rot.from_rotvec(params[:3]).as_matrix()
                t = params[3:6]
                s = 1.0

                trans_Y = (s * (TY @ R.T)) + t.reshape(1, 3)   # (M,3)
                diff = X[None, :, :] - trans_Y[:, None, :]    # (M,N,3)
                diff = diff * np.sqrt(w[None, None, :])
                r = P_sqrt[:, :, None] * diff                 # (M,N,3)
                return r.reshape(-1)

            init_offsets = np.zeros(6, dtype=float)

    # Without rotation: translate only (and maybe scale)
    else:

        if scale is True:

            def fit_func(params: NDArrayf) -> NDArrayf:
                # params = [tx, ty, tz, log_s]
                t = params[0:3]
                s = float(np.exp(params[3]))

                trans_Y = (s * TY) + t.reshape(1, 3)           # (M,3)
                diff = X[None, :, :] - trans_Y[:, None, :]    # (M,N,3)
                diff = diff * np.sqrt(w[None, None, :])
                r = P_sqrt[:, :, None] * diff                 # (M,N,3)
                return r.reshape(-1)

            init_offsets = np.zeros(4, dtype=float)

        else:

            def fit_func(params: NDArrayf) -> NDArrayf:
                # params = [tx, ty, tz]
                t = params[0:3]
                s = 1.0

                trans_Y = (s * TY) + t.reshape(1, 3)           # (M,3)
                diff = X[None, :, :] - trans_Y[:, None, :]    # (M,N,3)
                diff = diff * np.sqrt(w[None, None, :])
                r = P_sqrt[:, :, None] * diff                 # (M,N,3)
                return r.reshape(-1)

            init_offsets = np.zeros(3, dtype=float)

    results = fit_minimizer(fit_func, init_offsets, loss=loss_func)
    assert results is not None

    # Decode solution
    if not only_translation:
        R_opt = Rot.from_rotvec(results.x[:3]).as_matrix()
        t_opt = results.x[3:6]
        if scale is True:
            s_opt = float(np.exp(results.x[6]))
        else:
            s_opt = 1.0
    else:
        R_opt = np.eye(3)
        t_opt = results.x[:3]
        if scale is True:
            s_opt = float(np.exp(results.x[3]))
        else:
            s_opt = 1.0

    return R_opt, t_opt, s_opt

def _cpd_mstep_classic_fit_minimizer_fast(
    X: NDArrayf,
    TY: NDArrayf,
    w: NDArrayf,
    estep: dict[str, NDArrayf],
    scale: bool,
    only_translation: bool,
    params_fit_or_bin: Any,  # InFitOrBinDict
) -> tuple[NDArrayf, NDArrayf, float]:
    """
    Minimization step of classic CPD using a generic fit_minimizer (fast collapsed form).

    Returns (R, t, s).

    This uses the identity (up to constants):
        sum_{m,n} P[m,n] ||x_n - (s R y_m + t)||^2
        ≈ sum_m P1[m] ||xbar_m - (s R y_m + t)||^2
    where xbar_m = (sum_n P[m,n] x_n) / P1[m].
    """
    N, D = X.shape
    M, _ = TY.shape

    P = estep["P"]    # (M,N)
    P1 = estep["P1"]  # (M,)
    Np = float(estep["Np"])
    PX = estep["PX"]  # (M,3) if computed as P @ X; if your PX is (N,3) swap accordingly

    # --- IMPORTANT ---
    # In your current classic CPD code, PX = np.matmul(P, X) where P is (M,N) and X is (N,3),
    # so PX is (M,3). (This matches the CPD derivation.)
    # If your actual PX in your dict is transposed, adjust here.
    if PX.shape[0] != M:
        # If PX is (N,3) we likely have P shaped (N,M) somewhere; fail loudly to avoid silent bugs.
        raise ValueError("estep['PX'] must have shape (M,3) for classic CPD fast M-step.")

    # Soft correspondence centroids for each Y_m
    P1_safe = np.clip(P1, 1e-12, None)
    Xbar = PX / P1_safe[:, None]  # (M,3)

    # Build the SVD-derived terms too (so shrink step is unchanged)
    muX = np.divide(np.sum(estep["PX"], axis=0), Np) if estep["PX"].shape[0] == N else np.divide(np.sum(P @ X, axis=0), Np)
    muY = np.divide(np.sum(np.dot(np.transpose(P), TY), axis=0), Np)
    X_hat = X - np.tile(muX, (N, 1))
    Y_hat = TY - np.tile(muY, (M, 1))
    YPY = float(np.dot(np.transpose(P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1)))
    A = np.dot(np.transpose(X_hat), np.transpose(P))
    A = np.dot(A, Y_hat)

    fit_minimizer: Callable[..., Any] = params_fit_or_bin["fit_minimizer"]
    loss_func = params_fit_or_bin["fit_loss_func"]

    w_sqrt = np.sqrt(P1_safe)  # (M,)

    if not only_translation:

        if scale is True:

            def fit_func(params: NDArrayf) -> NDArrayf:
                R = Rot.from_rotvec(params[:3]).as_matrix()
                t = params[3:6]
                s = float(np.exp(params[6]))

                pred = (s * (TY @ R.T)) + t.reshape(1, 3)   # (M,3)
                r = w_sqrt[:, None] * (Xbar - pred)  * np.sqrt(w[None, :])         # (M,3)
                return r.reshape(-1)

            init_offsets = np.zeros(7, dtype=float)

        else:

            def fit_func(params: NDArrayf) -> NDArrayf:
                R = Rot.from_rotvec(params[:3]).as_matrix()
                t = params[3:6]
                s = 1.0

                pred = (s * (TY @ R.T)) + t.reshape(1, 3)
                r = w_sqrt[:, None] * (Xbar - pred)  * np.sqrt(w[None, :])
                return r.reshape(-1)

            init_offsets = np.zeros(6, dtype=float)

    else:

        if scale is True:

            def fit_func(params: NDArrayf) -> NDArrayf:
                t = params[0:3]
                s = float(np.exp(params[3]))

                pred = (s * TY) + t.reshape(1, 3)
                r = w_sqrt[:, None] * (Xbar - pred)  * np.sqrt(w[None, :])
                return r.reshape(-1)

            init_offsets = np.zeros(4, dtype=float)

        else:

            def fit_func(params: NDArrayf) -> NDArrayf:
                t = params[0:3]
                pred = TY + t.reshape(1, 3)
                r = w_sqrt[:, None] * (Xbar - pred)  * np.sqrt(w[None, :])
                return r.reshape(-1)

            init_offsets = np.zeros(3, dtype=float)

    results = fit_minimizer(fit_func, init_offsets, loss=loss_func)
    assert results is not None

    if not only_translation:
        R_opt = Rot.from_rotvec(results.x[:3]).as_matrix()
        t_opt = results.x[3:6]
        s_opt = float(np.exp(results.x[6])) if scale else 1.0
    else:
        R_opt = np.eye(3)
        t_opt = results.x[:3]
        s_opt = float(np.exp(results.x[3])) if scale else 1.0

    return R_opt, t_opt, s_opt

def _cpd_shrink_classic(
    X: NDArrayf,
    Ycur: NDArrayf,
    estep: dict[str, NDArrayf],
    R: NDArrayf,
    s: float,
    w: NDArrayf,
    sigma2: float,
    sigma2_min: float,
) -> tuple[float, float]:
    """
    Update variance and objective function for classic CPD.
    """
    P = estep["P"]          # (M,N)
    Pt1 = estep["Pt1"]      # (N,)
    P1 = estep["P1"]        # (M,)
    Np = float(estep["Np"]) # scalar

    N, D = X.shape
    M, Dy = Ycur.shape
    if Dy != D:
        raise ValueError("X and Ycur must have the same dimensionality.")

    # Weighted means (Fig. 2, Eq. 6)
    muX = (Pt1 @ X) / max(Np, 1e-12)          # (D,)
    muY = (P1 @ Ycur) / max(Np, 1e-12)        # (D,)

    # Centered clouds
    X_hat = X - muX[None, :]
    Y_hat = Ycur - muY[None, :]

    Xh = X_hat * np.sqrt(w[None, :])  # (N,3)
    Yh = Y_hat * np.sqrt(w[None, :])  # (M,3)

    # Terms used in q and sigma2 update
    xPx = float(Pt1 @ np.sum(Xh * Xh, axis=1))
    YPY = float(P1 @ np.sum(Yh * Yh, axis=1))
    A = Xh.T @ (P.T @ Yh)
    trAR = float(np.trace(A @ R))

    # Objective (Eq. 7)
    q = (xPx - 2.0 * s * trAR + (s * s) * YPY) / (2.0 * sigma2) + (D * Np / 2.0) * np.log(sigma2)

    # Sigma2 update (Fig. 2)
    sigma2_new = (xPx - s * trAR) / (max(Np, 1e-12) * D)
    if sigma2_new <= 0:
        sigma2_new = float(sigma2_min)

    return float(sigma2_new), float(q)

def _cpd_estep_lsg(
    cache_lsg: dict[str, Any],
    sigma_terms: dict[str, Any],
    TY: NDArrayf,
) -> dict[str, NDArrayf]:
    """
    Expectation step of LSG-CPD.

    Returns a dictionary of precomputed quantities (P, M0, M1, M2, sum_P) used by the least-squares M-step.

    :param cache_lsg: LSG cache (contains X, normals, etc.).
    :param sigma_terms: Sigma-dependent terms returned by _lsg_update_sigma_dependent.
    :param TY: Transformed moving/source point cloud using the current estimate, shape (M,3).
        This mirrors classic CPD where the E-step is evaluated at the current transform.
    """
    X = cache_lsg["X"]                     # (N,3)
    ref_normals = cache_lsg["ref_normals"] # (N,3)
    a = cache_lsg["a"]                     # (N,)
    invSigma_flat = cache_lsg["invSigma_flat"]  # (N,9)
    x_invSigma = cache_lsg["x_invSigma"]        # (N,3)
    x_invSigma_x = cache_lsg["x_invSigma_x"]    # (N,)
    w = cache_lsg["w"]  # (3,)
    Wn = cache_lsg["Wn"]  # (N,3)
    nWn = cache_lsg["nWn"]  # (N,)
    X_normal = cache_lsg["X_normal"]  # (N,) = x^T W n
    X_X2 = cache_lsg["X_X2"]  # (N,)

    F_matrix = sigma_terms["F_matrix"]          # (N,M)
    C_const = sigma_terms["C_const"]            # scalar
    c_const = sigma_terms["c_const"]            # scalar

    # Use the transformed moving/source points (M,3) directly (no recompute from R,t)
    gY = np.asarray(TY, dtype=float)  # (M,3)
    gY2 = np.sum((gY * gY) * w[None, :], axis=1)  # (M,)

    # Plane term weighted
    dWn = (Wn @ gY.T) - X_normal[:, None]  # (N,M)
    plane_term = (dWn * dWn) / nWn[:, None]  # (N,M)

    # Quad term weighted
    xWy = (X * w[None, :]) @ gY.T  # (N,M)
    quad_term = X_X2[:, None] + gY2[None, :] - 2.0 * xWy

    # Responsibilities (N,M)
    P = F_matrix * np.exp(c_const * (a[:, None] * plane_term + quad_term))

    denom = P.sum(axis=0, keepdims=True) + C_const
    P = P / np.clip(denom, 1e-300, None)

    # Sufficient stats
    M0_flat = P.T @ invSigma_flat    # (M,9)
    M0 = M0_flat.reshape(gY.shape[0], 3, 3)  # (M,3,3)

    M1 = P.T @ x_invSigma            # (M,3)
    M2 = P.T @ x_invSigma_x          # (M,)
    sum_P = float(np.sum(P))

    return {"P": P, "M0": M0, "M1": M1, "M2": M2, "sum_P": np.asarray(sum_P)}


def _lsg_build_whitened_terms(
    M0: NDArrayf,
    M1: NDArrayf,
) -> tuple[NDArrayf, NDArrayf]:
    """
    Build Cholesky factors L_m and targets mu_m for the whitened least-squares residuals:
      r_m = L_m ( R y_m + t - mu_m )

    :param M0: (M,3,3) SPD matrices.
    :param M1: (M,3) vectors.
    :return: (Ls, mus) with Ls shape (M,3,3) and mus shape (M,3).
    """
    M = M0.shape[0]
    Ls = np.empty((M, 3, 3), dtype=float)
    mus = np.empty((M, 3), dtype=float)

    for m in range(M):
        A = M0[m]
        b = M1[m]

        # Stabilize SPD factorization (small jitter)
        jitter = 1e-12 * float(np.trace(A)) + 1e-15
        A_stable = A + jitter * np.eye(3)

        try:
            L = np.linalg.cholesky(A_stable)
        except np.linalg.LinAlgError:
            # Fallback: eigen clamp
            w, V = np.linalg.eigh(A_stable)
            w = np.clip(w, 1e-12, None)
            A_spd = (V * w) @ V.T
            L = np.linalg.cholesky(A_spd)

        mu = np.linalg.solve(A_stable, b)

        Ls[m] = L
        mus[m] = mu

    return Ls, mus


def _cpd_mstep_lsg_least_squares(
    TY: NDArrayf,
    estep: dict[str, NDArrayf],
    only_translation: bool,
    params_fit_or_bin: Any,  # InFitOrBinDict in your codebase
) -> tuple[NDArrayf, NDArrayf]:
    """
    Minimization step of LSG-CPD.

    Uses the generic fit_minimizer on a whitened residual:
      r_m = L_m ( R y_m + t - mu_m )
    with M0_m = L_m L_m^T and mu_m = M0_m^{-1} M1_m.

    This matches your existing ICP pattern:
        results = params_fit_or_bin["fit_minimizer"](fit_func, init_offsets, **kwargs, loss=loss_func)

    :param TY: Transformed moving points for this iteration (M, 3).
    :param estep: Output of LSG E-step with at least "M0" (M,3,3) and "M1" (M,3).
    :param only_translation: Whether to solve only translation, otherwise solves rotation+translation.
    :param params_fit_or_bin: Dict with "fit_minimizer" and "fit_loss_func" entries.

    :return: (R, t)
    """
    M0 = estep["M0"]    # (M,3,3)
    M1 = estep["M1"]    # (M,3)

    # Build whitened terms once for this M-step
    Ls, mus = _lsg_build_whitened_terms(M0=M0, M1=M1)

    # Define loss function (same naming as ICP)
    loss_func = params_fit_or_bin["fit_loss_func"]
    fit_minimizer: Callable[..., Any] = params_fit_or_bin["fit_minimizer"]

    # With rotation: optimize rotvec + translation
    if not only_translation:

        def fit_func(offsets: NDArrayf) -> NDArrayf:
            # offsets = [rx, ry, rz, tx, ty, tz]
            R = Rot.from_rotvec(offsets[:3]).as_matrix()
            t = offsets[3:6]
            gY = (TY @ R.T) + t.reshape(1, 3)               # (M,3)
            diff = gY - mus                                 # (M,3)
            r = np.einsum("mij,mj->mi", Ls, diff)           # (M,3)
            return r.reshape(-1)

        # Initial offset near zero
        init_offsets = np.zeros(6, dtype=float)

    # Without rotation: optimize translation only
    else:

        def fit_func(offsets: NDArrayf) -> NDArrayf:
            # offsets = [tx, ty, tz]
            t = offsets[:3]
            gY = TY + t.reshape(1, 3)                        # (M,3)
            diff = gY - mus                                  # (M,3)
            r = np.einsum("mij,mj->mi", Ls, diff)            # (M,3)
            return r.reshape(-1)

        # Initial offset near zero
        init_offsets = np.zeros(3, dtype=float)

    # Any extra optimizer kwargs can be passed via params_fit_or_bin["fit_func"] in your framework,
    # but here we mirror your ICP call signature exactly and keep it simple.
    results = fit_minimizer(fit_func, init_offsets, loss=loss_func)

    # Mypy: having results as "None" is impossible, but not understood through overloading of your fit wrapper
    assert results is not None

    if not only_translation:
        R_opt = Rot.from_rotvec(results.x[:3]).as_matrix()
        t_opt = results.x[3:6]
    else:
        R_opt = np.eye(3, dtype=float)
        t_opt = results.x[:3]

    return R_opt, t_opt


def _cpd_shrink_lsg(
    TY: NDArrayf,
    estep: dict[str, NDArrayf],
    R: NDArrayf,
    t: NDArrayf,
    sigma2_min: float,
) -> tuple[float, float]:
    """
    Shrink step of LSG-CPD.

    Updates sigma2 and returns loglikelihood-like objective, following the same quadratic structure used in LSG code.
    """
    M0 = estep["M0"]            # (M,3,3)
    M1 = estep["M1"]            # (M,3)
    M2 = estep["M2"]            # (M,)
    sum_P = float(estep["sum_P"])

    # g(y) = R y + t
    gY = (TY @ R.T) + t.reshape(1, 3)  # (M,3)

    # Equivalent of:
    #   sum_m gY_m^T M0_m gY_m - 2 sum_m gY_m^T M1_m + sum_m M2_m
    term1 = float(np.einsum("mi,mij,mj->", gY, M0, gY))
    term2 = float(2.0 * np.einsum("mi,mi->", gY, M1))
    term3 = float(np.sum(M2))
    q = term1 - term2 + term3

    sigma2_new = q / max(3.0 * sum_P, 1e-12)

    # If sigma2 gets negative, we use a minimal sigma value instead
    if sigma2_new <= 0:
        sigma2_new = sigma2_min

    return float(sigma2_new), float(q)


def _cpd_fit(
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    trans_tba_epc: NDArrayf,
    params_fit_or_bin: InFitOrBinDict,
    weight_cpd: float,
    sigma2: float,
    sigma2_min: float,
    scale: bool = False,
    only_translation: bool = False,
    lsg: bool = False,
    # Precomputed constants reused across iterations
    cache: dict[str, Any] | None = None,
) -> tuple[NDArrayf, float, float]:
    """
    Fit step of Coherent Point Drift by expectation-minimization, with variance updating.

    See Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46, Figure 2, for equations below.
    See Liu et al. (2021), https://doi.org/10.1109/ICCV48922.2021.01501 for local surface geometry method.

    Note: CPD often solves for the full absolute transform again at every step with different initialization.
    Here we solve for the step transform instead with no initialization (equivalent), but carry the absolute
    transform to properly update the shrinking step.
    """

    # Convert inputs from 3xN to Nx3 for CPD math (classic CPD notation uses row vectors)
    X = ref_epc.T
    Y = tba_epc.T
    TY = trans_tba_epc.T

    # Get shape of inputs
    N, D = X.shape
    M, _ = Y.shape

    # 0/ Initialize variance if not defined
    if sigma2 is None:
        diff2 = (X[None, :, :] - TY[:, None, :]) ** 2
        sigma2 = float(np.sum(diff2) / (D * N * M))

    # Classic CPD
    if not lsg:

        w = cache["classic"]["w"]

        # 1/ Expectation step
        estep = _cpd_estep_classic(X=X, Y=Y, TY=TY, w=w, weight_cpd=weight_cpd, sigma2=float(sigma2))

        # 2/ Minimization step
        use_generic_mstep = False  # To check internally the consistency with old implementation
        if use_generic_mstep:
            mstep_func = _cpd_mstep_classic_fit_minimizer
        else:
            mstep_func = _cpd_mstep_classic_fit_minimizer_fast
        R, t, s = mstep_func(
            X=X, TY=TY, estep=estep, scale=scale, w=w, only_translation=only_translation,
            params_fit_or_bin=params_fit_or_bin
        )

        # 3/ Update variance and objective function
        sigma2_new, q = _cpd_shrink_classic(X=X, Ycur=TY, estep=estep, R=R, s=s, w=w,
                                            sigma2=float(sigma2), sigma2_min=float(sigma2_min))

    # LSG-CPD
    else:
        # LSG-CPD uses normals and local plane ratios, and replaces the SVD M-step
        # by a least-squares minimization of whitened residuals.
        assert cache is not None and cache["lsg"] is not None
        cache_lsg = cache["lsg"]


        sigma_terms = _lsg_update_sigma_dependent(
            cache_lsg=cache_lsg,
            sigma2=float(sigma2),
            weight_cpd=float(weight_cpd),
        )

        # 1/ Expectation step evaluated at current transform (TY)
        estep_lsg = _cpd_estep_lsg(
            cache_lsg=cache_lsg,
            sigma_terms=sigma_terms,
            TY=TY,
        )

        # 2/ Minimization step
        R, t = _cpd_mstep_lsg_least_squares(
            TY,
            estep=estep_lsg,
            only_translation=only_translation,
            params_fit_or_bin=params_fit_or_bin,
        )

        # 3/ Update variance and objective function
        sigma2_new, q = _cpd_shrink_lsg(
            TY,
            estep=estep_lsg,
            R=R,
            t=t,
            sigma2_min=float(sigma2_min),
        )

    # Build output matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = t

    return matrix, float(sigma2_new), float(q)

def _cpd_iteration_step(
    iterating_input: tuple[NDArrayf, float, float],
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    params_fit_or_bin: InFitOrBinDict,
    weight_cpd: float,
    sigma2_min: float,
    only_translation: bool,
    lsg: bool = False,
    cache: dict[str, Any] | None = None,
) -> tuple[tuple[NDArrayf, float, float], dict[str, float], None]:
    """
    Iteration step for Coherent Point Drift algorithm.

    Returns the updated iterating input (affine matrix, variance and objective function).
    """
    matrix, sigma2, q = iterating_input

    # Apply transform matrix from previous step
    trans_tba_epc = _apply_matrix_pts_mat(tba_epc, matrix=matrix)

    # Fit to get new step transform
    # (Note: the CPD algorithm re-computes the full transform from the original target point cloud,
    # so there is no need to combine a step transform within the iteration as in ICP/LZD)
    step_matrix, new_sigma2, new_q = _cpd_fit(
        ref_epc=ref_epc,
        tba_epc=tba_epc,
        trans_tba_epc=trans_tba_epc,
        sigma2=sigma2,
        weight_cpd=weight_cpd,
        sigma2_min=sigma2_min,
        only_translation=only_translation,
        lsg=lsg,
        cache=cache,
        params_fit_or_bin=params_fit_or_bin
    )

    # Compute statistic on offset to know if it reached tolerance
    offset_q = np.abs(q - new_q)

    # Add step matrix to full matrix
    new_matrix = step_matrix @ matrix

    translations = step_matrix[:3, 3]
    rotations = step_matrix[:3, :3]
    offset_translation = np.sqrt(np.sum(translations ** 2))
    offset_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    step_statistics = {"objective_func": float(offset_q), "translation": float(offset_translation),
                       "rotation": float(offset_rotation)}

    return (new_matrix, new_sigma2, new_q), step_statistics, None

def cpd(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    weight_cpd: float,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    max_iterations: int,
    tolerance_q: float | None,
    tolerance_translation: float | None,
    tolerance_rotation: float | None,
    sampling_strategy: Literal["independent", "same_xy", "iterative_same_xy"] = "same_xy",
    anisotropic: Literal["xy_vs_z", "per_axis"] | None = "xy_vs_z",
    only_translation: bool = False,
    lsg: bool = False,
    standardize: bool = True,
) -> tuple[NDArrayf, tuple[float, float, float], int, OutIterativeDict]:
    """
    Main function for Coherent Point Drift coregistration including "local surface geometry" (LSG) variant.

    See Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46 for original implementation.
    See Liu et al. (2021), https://doi.org/10.1109/ICCV48922.2021.01501 for LSG variant.

    This function subsamples input data, then runs CPD iteration steps to optimize its expectation-minimization until
    convergence or a maximum of iterations is reached.

    The function assumes we have two DEMs, or DEM and an elevation point cloud, in the same CRS.
    """

    # Derive centroid and scale ahead of any potential iterative sampling loop, and scale translation tolerance
    centroid, scale = _get_centroid_scale(ref_elev=ref_elev, transform=ref_transform, z_name=z_name)
    if not standardize:
        scale = 1
    tolerance_translation /= scale

    # Initial values and tolerances as dictionary
    init_matrix = np.eye(4)  # Initial matrix is the identity transform
    init_q = np.inf
    init_sigma2 = None
    tolerances = {"translation": tolerance_translation, "rotation": tolerance_rotation, "objective_func": tolerance_q}

    # Derive normals if method is LSG, otherwise not
    invert_output = False
    if lsg:
        if isinstance(ref_elev, np.ndarray):
            fixed_elev = "ref"
        elif isinstance(tba_elev, np.ndarray):
            fixed_elev = "tba"
        else:
            raise TypeError(
                "LSG-CPD requires one input to be a DEM array to derive normals."
            )

        # The normals must always be derived relative to a "fixed reference" (we don't rotate them during iterations),
        # so internally we'll need to invert inputs if 'tba' is the array, then invert the matrix at the end
        if fixed_elev == "tba":
            invert_output = True
            ref_elev, tba_elev = tba_elev, ref_elev
            ref_transform, tba_transform = tba_transform, ref_transform

        # Now ref is always the array
        aux_vars = _dem_normals_curvature(ref_elev, ref_transform, return_curvature=True)
        aux_tied_to = "ref"
    else:
        aux_vars = None
        aux_tied_to = "ref"

    # If we iterate the sampling, we re-define maximum iterations to be in the outside loop,
    # running a single CPD iteration with every sample
    if sampling_strategy == "iterative_same_xy":
        in_loop_max_it = max_iterations
        out_loop_max_it = max_iterations
        in_loop_sampling_strategy = "same_xy"
    # Otherwise, we run a single iteration of outside loop, as if there was no loop
    else:
        in_loop_max_it = max_iterations
        out_loop_max_it = 1
        in_loop_sampling_strategy = sampling_strategy

    # Initialize values for iterative sampling loop (tba_elev updates out_step_matrix that updates tba_elev)
    list_iteration_stats = []
    final_matrix = np.eye(4)
    tba_elev_orig = tba_elev.copy()
    tba_transform_orig = tba_transform
    new_q = init_q
    new_sigma2 = init_sigma2
    for i in range(out_loop_max_it):

        # Pre-process point-raster inputs to the same subsampled points
        ref_epc, tba_epc, sub_aux = _subsample_rst_pts(
            subsample=params_random["subsample"],
            random_state=params_random["random_state"],
            ref_elev=ref_elev,
            tba_elev=tba_elev,  # For iterative sampling, tba_elev is updated below
            aux_vars=aux_vars,
            aux_tied_to=aux_tied_to,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,  # For iterative sampling, tba_transform can be updated below
            crs=crs,
            area_or_point=area_or_point,
            sampling_strategy=in_loop_sampling_strategy,
            z_name=z_name,
        )

        # Remove centroid and standardize to facilitate numerical convergence
        ref_epc, tba_epc = _standardize_epc(ref_epc, tba_epc, centroid=centroid, scale=scale)

        # Calculate standard deviation for axes for optional anisotropic weighting
        w_axis = _axis_weights_from_epc(ref_epc, anisotropic=anisotropic)

        # Re-stack normals if defined
        if sub_aux is not None:
            norms = np.vstack((sub_aux["nx"], sub_aux["ny"], sub_aux["nz"])).T
            curv = sub_aux["curv"]
        else:
            norms = None
            curv = None

        # Run rigid CPD registration
        # Iterate through method until tolerance or max number of iterations is reached
        iterating_input = (init_matrix, new_sigma2, new_q)
        sigma2_min = tolerance_translation / 10
        # Pre-compute values re-used through iteration for efficiency
        cpd_cache = _cpd_precompute(
            ref_epc=ref_epc,
            tba_epc=tba_epc,
            weight_cpd=weight_cpd,
            lsg=lsg,
            ref_normals=norms,
            curvature=curv,
            axis_weights=w_axis,
        )
        constant_inputs = (
            ref_epc,
            tba_epc,
            params_fit_or_bin,
            weight_cpd,
            sigma2_min,
            only_translation,
            lsg,
            cpd_cache
        )  # For iterative sampling, tba_epc is updated above
        new_output, new_stats, output_iterative, _ = _iterate_method(
            method=_cpd_iteration_step,
            iterating_input=iterating_input,
            constant_inputs=constant_inputs,
            tolerances=tolerances,
            max_iterations=in_loop_max_it,
        )

        # Re-define inputs
        out_step_matrix = new_output[0]
        new_sigma2 = new_output[1]
        new_q = new_output[2]

        # Prints to help debug
        # trans_rot = translations_rotations_from_matrix(out_step_matrix)
        # print(f"Sampling iteration {i}: step translation/rotations: {trans_rot[3]}")

        # We invert, destandardize and update the matrix
        out_step_matrix[:3, 3] *= scale
        final_matrix = out_step_matrix @ final_matrix

        # trans_rot = translations_rotations_from_matrix(final_matrix)
        # print(f"Sampling iteration {i}: full matrix translation/rotations: {trans_rot[3]}")

        # If we apply iterative sampling, we need to do additional things each loop
        if sampling_strategy == "iterative_same_xy":
            # We update the to-be-aligned elevation (input before subsampling) here directly, so that X/Y sampling will be updated
            if isinstance(tba_elev, np.ndarray):
                tba_elev, tba_transform = _apply_matrix_rst(
                    tba_elev_orig, transform=tba_transform_orig, centroid=centroid, matrix=final_matrix
                )
            else:
                tba_elev = _apply_matrix_pts(tba_elev_orig, matrix=final_matrix, centroid=centroid, z_name=z_name)
            # We update the iterative output with the outside loop iteration number
            it_stats = output_iterative["iteration_stats"]
            it_stats["iteration"] = i + 1
            list_iteration_stats.append(it_stats)

            # Check exit condition was reached in inside loop
            if all(new_stats[k] < tolerances[k] if k is not None else True for k in tolerances.keys()):
                logging.debug("Exiting outside loop of iterative sampling as statistics have all reached tolerance.")
                break

    # Over-write iterative output for iterative sampling
    if sampling_strategy == "iterative_same_xy":
        iteration_stats = pd.concat(list_iteration_stats)
        output_iterative: OutAffineDict = {"last_iteration": i + 1, "iteration_stats": iteration_stats}

    # Get subsample size
    # TODO: Support reporting different number of subsamples when independent?
    subsample_final = min(ref_epc.shape[1], tba_epc.shape[1])

    # If we solved the swapped problem for LSG (tba was the fixed reference), invert to keep external convention
    if lsg and invert_output:
        final_matrix = invert_matrix(final_matrix)

    return final_matrix, centroid, subsample_final, output_iterative


#######################
# 6/ Least Z-difference
#######################

def _lzd_aux_vars(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    transform: affine.Affine,
) -> tuple[NDArrayf, NDArrayf]:
    """
    Deriving gradient in X/Y expected by the Least Z-difference coregistration.

    :return: Gradient in X/Y, scaled based on the DEM resolution.
    """

    # If inputs are both point clouds, raise an error
    if isinstance(ref_elev, gpd.GeoDataFrame) and isinstance(tba_elev, gpd.GeoDataFrame):

        raise TypeError(
            "The LZD coregistration does not support two point clouds, one elevation "
            "dataset in the pair must be a DEM."
        )

    # If inputs are both rasters, derive terrain attributes from ref and get 2D dh interpolator
    elif isinstance(ref_elev, np.ndarray) and isinstance(tba_elev, np.ndarray):

        # Derive slope and aspect from the reference as default
        if np.ma.isMaskedArray(ref_elev):
            ref_arr = ref_elev.filled(np.nan)
        else:
            ref_arr = ref_elev
        gradient_y, gradient_x = np.gradient(ref_arr)

    # If inputs are one raster and one point cloud, derive terrain attribute from raster and get 1D dh interpolator
    else:

        if isinstance(ref_elev, gpd.GeoDataFrame):
            rst_elev = tba_elev
        else:
            rst_elev = ref_elev
        if np.ma.isMaskedArray(rst_elev):
            rst_arr = rst_elev.filled(np.nan)
        else:
            rst_arr = rst_elev
        # Derive slope and aspect from the raster dataset
        gradient_y, gradient_x = np.gradient(rst_arr)

    # Convert to unitary gradient depending on resolution
    res = _res(transform)
    gradient_x = gradient_x / res[0]
    gradient_y = -gradient_y / res[1]  # Because raster Y axis is inverted, need to add a minus

    return gradient_x, gradient_y

def _lzd_fit_func_nonlinear(
    inputs: tuple[Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf], NDArrayf, tuple[float, float, float]],
    t1: float,
    t2: float,
    t3: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
) -> NDArrayf:
    """
    Fit function for nonlinear LZD.

    Returns elevation residuals after applying a candidate rigid step transform to the point cloud:
        r = z_grid(x', y') - z'

    :param inputs: Tuple of (grid interpolator, point cloud, centroid).
        - sub_rst: Callable interpolator z_grid((y, x)) -> z, matching _reproject_horizontal_shift_samecrs signature.
        - pts_epc: Point cloud as 3xN array (in map coordinates).
        - centroid: Centroid used as rotation center for _apply_matrix_pts_mat.
    :param t1: Translation in X.
    :param t2: Translation in Y.
    :param t3: Translation in Z.
    :param alpha1: Rotation around X.
    :param alpha2: Rotation around Y.
    :param alpha3: Rotation around Z.

    :return: 1D array of finite residuals (dh) in map units.
    """

    sub_rst, pts_epc, centroid = inputs

    # Build candidate step matrix
    step = matrix_from_translations_rotations(t1, t2, t3, alpha1, alpha2, alpha3, use_degrees=False)

    # Apply candidate step to current points
    pts_step = _apply_matrix_pts_mat(pts_epc, matrix=step, centroid=centroid)

    x = pts_step[0, :]
    y = pts_step[1, :]
    z = pts_step[2, :]

    # Residuals: grid elevation at transformed XY minus transformed Z
    dh = sub_rst((y, x)) - z
    dh = np.asarray(dh).reshape(-1)

    # Remove invalid values (created by interpolation)
    return dh[np.isfinite(dh)]


def _lzd_fit_nonlinear(
    sub_rst: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    pts_epc: NDArrayf,
    centroid: tuple[float, float, float],
    params_fit_or_bin: Any,  # InFitOrBinDict
    only_translation: bool,
    **kwargs: Any,
) -> NDArrayf:
    """
    Optimize one nonlinear LZD step transform by minimizing dh residuals directly.

    This is analogous to ICP's point-to-plane nonlinear solve pattern:
    - return a residual vector (here: vertical residuals),
    - let fit_minimizer + loss handle robust losses.

    :param sub_rst: Interpolator of the reference grid elevation.
    :param pts_epc: Current point cloud as 3xN array (already in the current iteration frame).
    :param centroid: Centroid used as rotation center for _apply_matrix_pts_mat.
    :param params_fit_or_bin: Fit configuration dictionary (expects keys "fit_minimizer" and "fit_loss_func").
    :param only_translation: Whether to solve only for translation (no rotations).
    :param kwargs: Extra keyword arguments forwarded to fit_minimizer.

    :return: Step affine matrix (4x4) for this iteration.
    """

    inputs = (sub_rst, pts_epc, centroid)

    fit_minimizer: Callable[..., Any] = params_fit_or_bin["fit_minimizer"]
    loss_func = params_fit_or_bin["fit_loss_func"]

    # With rotation: 6 DOF
    if not only_translation:

        def fit_func(offsets: NDArrayf) -> NDArrayf:
            return _lzd_fit_func_nonlinear(
                inputs=inputs,
                t1=float(offsets[0]),
                t2=float(offsets[1]),
                t3=float(offsets[2]),
                alpha1=float(offsets[3]),
                alpha2=float(offsets[4]),
                alpha3=float(offsets[5]),
            )

        init_offsets = np.zeros(6, dtype=float)

    # Translation only: 3 DOF
    else:

        def fit_func(offsets: NDArrayf) -> NDArrayf:
            return _lzd_fit_func_nonlinear(
                inputs=inputs,
                t1=float(offsets[0]),
                t2=float(offsets[1]),
                t3=float(offsets[2]),
                alpha1=0.0,
                alpha2=0.0,
                alpha3=0.0,
            )

        init_offsets = np.zeros(3, dtype=float)

    results = fit_minimizer(fit_func, init_offsets, loss=loss_func, **kwargs)
    assert results is not None

    # Build step matrix
    if only_translation:
        beta = np.array([results.x[0], results.x[1], results.x[2], 0.0, 0.0, 0.0], dtype=float)
    else:
        beta = np.asarray(results.x, dtype=float)

    return matrix_from_translations_rotations(*beta, use_degrees=False)


def _lzd_fit_func(
    inputs: tuple[NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf, NDArrayf],
    t1: float,
    t2: float,
    t3: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    scale: float = 0.0,
) -> NDArrayf:
    """
    Fit function of Least Z-difference coregistration, Rosenholm and Torlegård (1988).

    Linearizes a rigid transformation for small rotations and utilizes dZ as a differential function of the plane
    coordinates (Equation 6).

    Will solve for the 7 parameters of a scaled rigid transform.

    :param inputs: Inputs not optimized by fit function: 1D arrays of X, Y, Z, as well as elevation change and elevation
        gradient along X and Y evaluated at X/Y coordinates.
    :param t1: Translation in X.
    :param t2: Translation in Y.
    :param t3: Translation in Z.
    :param alpha1: Rotation around X.
    :param alpha2: Rotation around Y.
    :param alpha3: Rotation around Z.
    :param scale: Scaling factor.

    :return: 1D array of residuals between elevation change and approximate rigid transformation in X/Y.
    """

    # Get constant inputs
    x, y, z, dh, gradx, grady = inputs

    # We compute lambda from estimated parameters (Equation 6)
    lda = (
        t3
        - x * alpha2
        + y * alpha1
        + z * scale
        - gradx * (t1 + x * scale - y * alpha3 + z * alpha2)
        - grady * (t2 + x * alpha3 + y * scale - z * alpha1)
    )

    # Get residuals with elevation change
    res = lda - dh

    return res

def _convert_lengthscale_gstools_gpytorch(correlation_range: float):

    # Divide by 2 because I used rescale=2 in GSTools (to get effective range)
    gp_lengthscale = correlation_range / 2 / np.sqrt(2)

    return gp_lengthscale

def _gls_lazy_gpytorch(
    xcoord: NDArrayf,
    ycoord: NDArrayf,
    X: NDArrayf,
    Y: NDArrayf,
    sig_Y: NDArrayf,
    lengthscale=0.2,
    outputscale=1.0,
    cg_tol=1e-3,
    max_preconditioner_size=100,
    jitter=1e-3):
    """
    Perform generalized least squares (GLS) using GPyTorch lazy covariances based on kernels to scale efficiently
    with a large number of points.

    This workflow has the benefit of never having to invert the full NxN covariance like for traditional GLS,
    but instead solves for covariance * vector, as usually done in Gaussian Processes.
    It also has a more stable inversion than Cholesky methods for kernel-type covariances through conjugate gradients.
    """
    # This feature requires GPytorch and dependencies (torch, linear_operator)
    import_optional("gpytorch")

    import torch
    import gpytorch
    # Dependency of GPyTorch (used to live in GPyTorch directly)
    from linear_operator.operators import (
        DiagLinearOperator,
        MatmulLinearOperator,
    )

    # Convert with from_numpy() to ensure it shares memory with the NumPy array
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    # 1/ Create lazy covariance from kernel
    coords = torch.stack([torch.from_numpy(xcoord), torch.from_numpy(ycoord)], dim=1)  # N x 2
    dtype = coords.dtype

    # Exponential kernel = RBF
    base = gpytorch.kernels.RBFKernel(ard_num_dims=2)
    base.lengthscale = torch.tensor(lengthscale, dtype=dtype)
    base.raw_lengthscale.requires_grad_(False)

    scale = gpytorch.kernels.ScaleKernel(base)
    scale.outputscale = torch.tensor(outputscale, dtype=dtype)
    scale.raw_outputscale.requires_grad_(False)

    # This returns a LazyEvaluatedKernelTensor (not a full NxN covariance matrix)
    K_lazy = scale(coords, coords)

    # Add jitter (for numerical stability)
    if jitter != 0:
        K_lazy = gpytorch.add_jitter(K_lazy, jitter)

    # Multiply the covariance by heteroscedastic noise in an outer product
    sigma_noise = torch.from_numpy(sig_Y)
    D = DiagLinearOperator(sigma_noise)  # diag(sigma)
    DK = MatmulLinearOperator(D, K_lazy)
    K_lazy = MatmulLinearOperator(DK, D)

    # 2/ Use conjugate-gradient settings for solving inversion
    with gpytorch.settings.cg_tolerance(cg_tol), \
         gpytorch.settings.max_preconditioner_size(max_preconditioner_size):

        Sinv_y = gpytorch.functions.solve(K_lazy, Yt)    # n×1
        Sinv_X = gpytorch.functions.solve(K_lazy, Xt)    # n×p

    # We compute the terms required for the GLS
    XtSinvX = Xt.T @ Sinv_X
    XtSinvY = Xt.T @ Sinv_y
    # We ensure full symmetry before final solve
    XtSinvX = (XtSinvX + XtSinvX.T) * 0.5
    # Derive coefficients, their covariance, and standard errors
    beta_hat = torch.linalg.solve(XtSinvX, XtSinvY)
    cov_beta = torch.linalg.inv(XtSinvX)
    se_beta = torch.sqrt(torch.diag(cov_beta))

    return beta_hat.numpy(), se_beta.numpy(), cov_beta.numpy()

def _lzd_fit_error_propag(x, y, z, dh, gx, gy, pixel_size, sig_h_other, corr_h_other,
                          sig_h_grid=None, corr_h_grid=None, force_opti: Literal["gls", "tls"] | None = None):
    """
    Error-aware LZD using either generalized least-squares (GLS) or total least-squares (TLS), depending on
    the error structure of the inputs.
    """

    # Test data
    # import logging
    # import numpy as np
    # print("Careful: TEST DATA UNCOMMENTED")
    # x = np.linspace(0, 50, 50)
    # y = np.linspace(0, 50, 50)
    # z = np.random.normal(size=50)
    # dh = np.random.normal(size=50)
    # gx = np.random.normal(size=50)
    # gy = np.random.normal(size=50)


    #
    # var_h_other = None
    # var_h_grid = np.ones(50)
    # corr_h_grid = None
    # var_h_other = np.ones(50)
    # var_h_grid = var_h_other
    # corr_h_other = None
    # corr_h_grid = corr_h_other
    # var_h_other = var_h_grid
    # corr_h_other = corr_h_grid
    # var_h_other = np.abs(np.random.normal(size=50))

    # print(np.min(var_h_grid), np.max(var_h_grid))
    # def corr_h_other(d):
    #     return np.exp(-(d/10)**2)
    # var_gx = np.abs(np.random.normal(size=50))
    # var_gy = np.abs(np.random.normal(size=50))


    # Linear regression Y = β X
    # Independent variable
    X = np.stack([
        -gx,
        -gy,
        y + gy * z,
        -x - gx * z,
        gx * y - gy * x
    ])
    # Dependent variable
    Y = dh

    # 1/ GENERALIZED LEAST SQUARES: Only error in dependent variable Y, no errors in the independent variable X
    # In our case, if only the secondary (not necessarily gridded) elevation has significant errors
    if force_opti == "gls" or (force_opti is None and sig_h_grid is None):

        if force_opti == "gls":
            logging.info("Forcing method optimization method 'gls' for LZD.")
        import statsmodels.api as sm
        from scipy.spatial.distance import pdist, squareform

        logging.info("No error passed for gridded elevation, using GLS for LZD error propagation.")

        # Add a constant to the independent variables for the intercept
        X2 = sm.add_constant(X.T)

        # Known error covariance matrix accounting for autocorrelation and heteroscedasticity
        # pdists = squareform(pdist(np.stack([x, y]).T))
        #
        # logging.info(f"Max distance: {np.max(pdists)}")
        #
        # if corr_h_other is not None:
        #
        #     corr_func = corr_h_other[0]
        #     logging.info("Using correlation function to estimate square covariance matrix.")
        #     logging.info(f"Mean correlation: {np.mean(corr_func(pdists))}")
        #
        #     covar = corr_func(pdists) * np.outer(np.sqrt(var_h_other), np.sqrt(var_h_other))
        #     covar += 10e-6 * np.mean(var_h_other) * np.eye(len(x))
        #
        #     # def clip_eigenvalues(Sigma, min_eig_frac=1e-6):
        #     #     Sigma = (Sigma + Sigma.T) / 2
        #     #     eigvals, eigvecs = np.linalg.eigh(Sigma)
        #     #     floor = max(eigvals.max() * min_eig_frac, 1e-12)
        #     #     eigvals_clipped = np.clip(eigvals, floor, None)
        #     #     return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        #     # covar = clip_eigenvalues(covar)
        # else:
        #     logging.info("No correlation, using diagonal variance.")
        #     covar = 1 / var_h_other

        # print(f"Cholesky: {np.linalg.cholesky(covar)}\n")
        # print(f"Cond: {np.linalg.cond(covar)}")
        # Z = np.linalg.solve(covar, X)
        # print(np.linalg.cond(X.T @ Z))

        # K is a LazyTensor (from scale(coords)) instead of a dense matrix

        # Instantiate and fit the GLS model
        # gls_model = sm.GLS(Y, X2, sigma=covar)
        # gls_results = gls_model.fit()

        # Print the summary of the results
        # logging.info("GLS summary from statsmodels:")
        # gls_results.summary()

        # Extract results
        # beta = gls_results.params
        # sd_beta = gls_results.bse
        # cov_beta = gls_results.cov_params()
        #
        # logging.info(f"Statsmodel GLS beta: {beta}")
        # logging.info(f"Statsmodel GLS SE: {sd_beta}")

        if corr_h_other is not None:
            gp_ls = _convert_lengthscale_gstools_gpytorch(corr_h_other[1])
        else:
            gp_ls = _convert_lengthscale_gstools_gpytorch(corr_h_grid[1])

        # We pass sigma for Y = dh, which can depend on both inputs
        sig_h_o = sig_h_other if sig_h_other is not None else 0
        sig_h_g = sig_h_grid if sig_h_grid is not None else 0
        sig_dh = np.sqrt(sig_h_o ** 2 + sig_h_g ** 2)
        # Perform GLS with GPyTorch
        beta, sd_beta, cov_beta = _gls_lazy_gpytorch(xcoord=x, ycoord=y, X=X2, Y=Y, sig_Y=sig_dh, lengthscale=gp_ls)

        logging.info(f"GPyTorch-based GLS beta: {beta}")
        logging.info(f"GPyTorch-based GLS SE: {sd_beta}")


    # 2/ TOTAL LEAST SQUARES: Errors in both dependent variable Y and independent variable X
    # In our case, if the gridded elevation and its gradient have significant errors
    else:

        if force_opti == "tls":
            logging.info("Forcing method optimization method 'tls' for LZD.")

        logging.info("Error passed for gridded elevation, using TLS for LZD error propagation.")

        # Transform sigma in variance to simplify writing below
        # If sig_h_grid is not defined, we simply apply a fraction of sig_h_other
        var_h_grid = sig_h_grid ** 2 if sig_h_grid is not None else np.mean(sig_h_other ** 2) / 1000 * np.ones(len(x))

        # Get amplitude of gradient errors from elevation errors and their correlations
        corr_func = corr_h_grid[0] if corr_h_grid is not None else None
        corr_grad_spacing = corr_func(2 * pixel_size) if corr_func is not None else 0

        logging.info("Correlation at gradient spacing: {:.2f}".format(corr_grad_spacing))
        var_gx = var_h_grid / 2 * (1 - corr_grad_spacing) / (pixel_size ** 2)
        var_gy = var_h_grid / 2 * (1 - corr_grad_spacing) / (pixel_size ** 2)
        var_dh = var_h_grid + sig_h_other**2 if sig_h_other is not None else var_h_grid
        var_z = var_h_grid / 10000
        var_x = var_h_grid / 10000
        var_y = var_h_grid / 10000

        from scipy.odr import ODR, multilinear, Data
        # CAVEAT: This ODR implementation doesn't support autocorrelation of the variables,
        # only heteroscedasticity and inter-correlation between independent variables

        # Thankfully, both elevation errors and its gradient errors share the same correlation lengths,
        # so we can simply sample sparse points according to the correlation error

        # 1/ Covariance of dependent variable
        cov_Y = var_dh
        # Convert to weight, easy for a vector variance
        w_Y = 1 / cov_Y

        # 2/ Covariance of independent variable
        # We need to build a (5, 5, N) covariance matrix describing the covariance between input variables
        # The 6th term, the intercept constant (z translation), doesn't need to be defined yet for the covariance

        # Diagonals
        var_X = np.array([
            var_gx,
            var_gy,
            z**2 * var_gy + var_z * gy**2 + var_y,
            z**2 * var_gx + var_z * gx**2 + var_x,
            y**2 * var_gx + x**2 * var_gy + var_x * gy**2 + var_y * gx**2
        ])

        ratio_X = np.mean(cov_Y) / np.var(Y)
        logging.info(f"Y error/variance ratio: {ratio_X}")
        ratio_X = np.mean(var_X, axis=1) / np.var(X, axis=1)
        logging.info(f"X error/variance ratio: {ratio_X}")

        # For zeros
        zeros = np.zeros(len(z))
        # Deactivate black formatting for readibility of the matrix
        # fmt: off
        cov_XX = np.stack(
            [[var_X[0],     zeros,       zeros,                        z * var_gx,                       -y * var_gx],
             [zeros,        var_X[1],   -z * var_gy,                   zeros,                             x * var_gy],
             [zeros,       -z * var_gy,  var_X[2],                     var_z * gx * gy, -z * x * var_gy + gx * var_y],
             [z * var_gx,   zeros,       var_z * gx * gy,              var_X[3],        -z * y * var_gx + gy * var_x],
             [-y * var_gx,  x * var_gy, -z * x * var_gy + gx * var_y, -z * y * var_gx + gy * var_x,         var_X[4]],
        ])
        # Reactivate black formatting
        # fmt: on

        # TODO: Move those to tests
        # ind_symmetric = [np.all(np.array_equal(cov_XX[:, :, i], cov_XX[:, :, i].T)) for i in range(cov_XX.shape[-1])]
        # ind_pos_semidef = np.array([np.all(np.linalg.eigvals(cov_XX[:, :, i]) > 0) for i in range(cov_XX.shape[-1])])
        #
        # indices_wrong = np.arange(len(ind_pos_semidef))[ind_pos_semidef]
        #
        # for i in range(len(indices_wrong)):
        #     print(f"Variance: {var_h_grid[indices_wrong[i]]}")
        #     print(f"Gradient X: {gx[indices_wrong[i]]}")
        #     print(f"Gradient Y: {gy[indices_wrong[i]]}")
        #     print(f"X: {x[indices_wrong[i]]}")
        #     print(f"Y: {y[indices_wrong[i]]}")
        #     print(f"Z: {z[indices_wrong[i]]}")

        # Convert to weight, inverting each 5x5 matrix for a given observation (as those are assumed uncorrelated)
        w_XX = np.ones(cov_XX.shape)
        for i in range(cov_XX.shape[-1]):
            # Regularize diagonal with 10e-3 term to ensure semi-positiveness
            reg = 10e-6 * np.eye(cov_XX.shape[0]) * np.mean(np.diag(cov_XX[:, :, i]))
            w_XX[:, :, i] = np.linalg.inv(cov_XX[:, :, i] + reg)

        # rd = RealData(x=X, y=Y, covx=cov_XX, covy=cov_Y)
        rd = Data(x=X, y=Y, we=w_Y, wd=w_XX)
        odr = ODR(rd, multilinear, beta0=np.zeros(6))  # We define the 6th term here in the multilinear model and beta0

        output = odr.run()
        # logging.info("TLS summary from scipy.odr:")
        # logging.info(output.pprint())

        # Get output
        beta = output.beta
        sd_beta = output.sd_beta
        cov_beta = output.cov_beta

    # logging.info(beta1)
    # logging.info(beta2)

    # Re-order output to be t1, t2, t3, r1, r2, r3, given that t3 (constant) was added at the end
    ind = np.array([1, 2, 0, 3, 4, 5])
    beta = beta[ind]
    sd_beta = sd_beta[ind]

    return beta, sd_beta, cov_beta

def _lzd_fit_linearized(
    x: NDArrayf,
    y: NDArrayf,
    z: NDArrayf,
    dh: NDArrayf,
    gradx: NDArrayf,
    grady: NDArrayf,
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
    pixel_size: float,
    errors: tuple[NDArrayf, Callable, NDArrayf, Callable] = None,
    force_opti: Literal["ols", "gls", "tls"] = None,
    **kwargs: Any,
) -> tuple[NDArrayf, NDArrayf]:
    """
    Optimization of fit function for Least Z-difference coregistration.

    :param x: X coordinate as 1D array.
    :param y: Y coordinate as 1D array.
    :param z: Z coordinate as 1D array.
    :param dh: Elevation change with other elevation dataset at X/Y coordinates as 1D array.
    :param gradx: DEM gradient along X axis at X/Y coordinates as 1D array.
    :param grady: DEM gradient along Y axis at X/Y coordinates as 1D array.
    :param params_fit_or_bin: Dictionary of fitting and binning parameters.
    :param only_translation: Whether to coregister only a translation, otherwise both translation and rotation.
    :param: **kwargs: Keyword arguments passed to fit optimizer.

    :return: Optimized affine matrix.
    """

    # Trim if required
    if "trim_residuals" in params_fit_or_bin.keys() and params_fit_or_bin["trim_residuals"]:
        res = _lzd_fit_func((x, y, z, dh, gradx, grady), 0, 0, 0, 0, 0, 0)
        ind = index_trimmed(res, central_estimator=params_fit_or_bin["trim_central_statistic"],
                            spread_estimator=params_fit_or_bin["trim_spread_statistic"],
                            spread_coverage=params_fit_or_bin["trim_spread_coverage"],
                            iterative=params_fit_or_bin["trim_iterative"])
        logging.info(f"Trimmed {np.count_nonzero(ind)} residuals.")
        # Keep data not trimmed
        x = x[~ind]
        y = y[~ind]
        z = z[~ind]
        dh = dh[~ind]
        gradx = gradx[~ind]
        grady = grady[~ind]
        if errors is not None:
            sig_other = errors[0][~ind] if errors[0] is not None else None
            sig_grid = errors[2][~ind] if errors[2] is not None else None
            errors = (sig_other, errors[1], sig_grid, errors[3])

    # Inputs that are not parameters to optimize
    inputs = (x, y, z, dh, gradx, grady)

    # Define loss function
    loss_func = params_fit_or_bin["fit_loss_func"]

    # For translation + rotation
    if not only_translation:

        def fit_func(offsets: tuple[float, float, float, float, float, float]) -> NDArrayf:
            return _lzd_fit_func(
                inputs=inputs,
                t1=offsets[0],
                t2=offsets[1],
                t3=offsets[2],
                alpha1=offsets[3],
                alpha2=offsets[4],
                alpha3=offsets[5],
            )

        # Initial offset near zero
        init_offsets = np.zeros(6)

    # For only translation
    else:

        def fit_func(offsets: tuple[float, float, float, float, float, float]) -> NDArrayf:
            return _lzd_fit_func(
                inputs=inputs,
                t1=offsets[0],
                t2=offsets[1],
                t3=offsets[2],
                alpha1=0.0,
                alpha2=0.0,
                alpha3=0.0,
            )

        # Initial offset near zero
        init_offsets = np.zeros(3)

    # Run optimizer on function
    if force_opti == "ols" or (force_opti is None and errors is None):

        if force_opti == "ols":
            logging.info("Forcing method optimization method 'ols' for LZD.")
        results = params_fit_or_bin["fit_minimizer"](fit_func, init_offsets, loss=loss_func, **kwargs)
        beta = results.x
        err_beta = None
    else:
        beta, err_beta, _ = _lzd_fit_error_propag(x=x, y=y, z=z, dh=dh, gx=gradx, gy=grady, pixel_size=pixel_size,
                                                  sig_h_other=errors[0], corr_h_other=errors[1],
                                                  sig_h_grid=errors[2], corr_h_grid=errors[3],
                                                  force_opti=force_opti)

    # Mypy: having beta as "None" is impossible, but not understood through overloading of _bin_or_and_fit_nd...
    assert beta is not None
    # Build matrix out of optimized parameters
    matrix = matrix_from_translations_rotations(*beta, use_degrees=False)  # type: ignore

    return matrix, err_beta


def _lzd_fit(
    matrix: NDArrayf,
    sub_rst: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    trans_pts_epc: NDArrayf,
    centroid: tuple[float, float, float],
    sub_gradx: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    sub_grady: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    params_fit_or_bin: Any,  # InFitOrBinDict
    only_translation: bool,
    sub_errors: tuple[
        Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf] | None,
        Callable | None,
        Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf] | None,
        Callable | None,
    ] | None,
    pixel_size: float,
    force_opti: Literal["ols", "gls", "tls"] | None = None,
    linearized: bool = True,
    **kwargs: Any,
) -> tuple[NDArrayf, NDArrayf | None]:
    """
    Fit dispatcher for LZD iteration step.

    This function:
    1) Applies the current matrix to the point cloud.
    2) Builds the residual inputs required by the selected solver.
    3) Solves for a step transform matrix (and optionally parameter uncertainties for linearized GLS/TLS).

    :param matrix: Current affine transform matrix of the iteration.
    :param sub_rst: Interpolator for the reference grid elevation.
    :param sub_pts: Transformed elevation point cloud for this step.
    :param centroid: Centroid used as rotation center for _apply_matrix_pts_mat.
    :param sub_gradx: Interpolator for grid gradient along X (used only for linearized solver).
    :param sub_grady: Interpolator for grid gradient along Y (used only for linearized solver).
    :param params_fit_or_bin: Fit configuration dictionary.
    :param only_translation: Whether to solve only for translation.
    :param sub_errors: Error inputs (used only for linearized GLS/TLS path).
    :param pixel_size: Pixel size in meters.
    :param force_opti: Force optimization method in linearized solver ("ols", "gls", "tls").
    :param linearized: Whether solver is "linearized" (R&T 1988 small-rotation) or not (exact vertical residual).
    :param kwargs: Extra keyword arguments forwarded to the fit minimizer.

    :return: (step_matrix, err_beta) where err_beta is only returned for the linearized GLS/TLS path.
    """

    # Nonlinear solve: optimize dh residuals directly
    if not linearized:
        if sub_errors is not None or force_opti is not None:
            # Nonlinear path currently ignores GLS/TLS error propagation
            logging.info(
                "Nonlinear LZD does not use GLS/TLS error propagation; using residual+loss only."
            )

        step_matrix = _lzd_fit_nonlinear(
            sub_rst=sub_rst,
            pts_epc=trans_pts_epc,
            centroid=centroid,
            params_fit_or_bin=params_fit_or_bin,
            only_translation=only_translation,
            **kwargs,
        )
        err_beta = None
        return step_matrix, err_beta

    # Linearized solve: follow Rosenholm & Torlegård (1988)
    else:
        x = trans_pts_epc[0, :].copy()
        y = trans_pts_epc[1, :].copy()
        z = trans_pts_epc[2, :].copy()

        # Evaluate dh and gradients at transformed coordinates
        dh = sub_rst((y, x)) - z
        gradx = sub_gradx((y, x))
        grady = sub_grady((y, x))

        # Resolve and sample errors at the transformed coordinates (if provided)
        if sub_errors is not None:
            sub_sig_other, corr_other, sub_sig_grid, corr_grid = sub_errors
            sig_other = sub_sig_other((y, x)) if sub_sig_other is not None else None
            sig_grid = sub_sig_grid((y, x)) if sub_sig_grid is not None else None
            errors = (sig_other, corr_other, sig_grid, corr_grid)
        else:
            errors = None

        # Remove centroid before fit for better numerical conditioning
        x -= centroid[0]
        y -= centroid[1]
        z -= centroid[2]

        # Remove invalid values sampled by interpolators
        valids = np.logical_and.reduce((np.isfinite(dh), np.isfinite(z), np.isfinite(gradx), np.isfinite(grady)))
        if errors is not None:
            if errors[0] is not None:
                valids = np.logical_and(valids, np.isfinite(errors[0]))
            if errors[2] is not None:
                valids = np.logical_and(valids, np.isfinite(errors[2]))

        if np.count_nonzero(valids) == 0:
            raise ValueError(
                "The subsample contains no more valid values. This can happen if the affine transformation to "
                "correct is larger than the data extent, or if the algorithm diverged. To ensure all possible points can "
                "be used at any iteration step, use subsample=1."
            )

        x = x[valids]
        y = y[valids]
        z = z[valids]
        dh = np.asarray(dh)[valids]
        gradx = np.asarray(gradx)[valids]
        grady = np.asarray(grady)[valids]

        if errors is not None:
            err_other, corr_other, err_grid, corr_grid = errors
            err_other = err_other[valids] if err_other is not None else None
            err_grid = err_grid[valids] if err_grid is not None else None
            errors = (err_other, corr_other, err_grid, corr_grid)

        step_matrix, err_beta = _lzd_fit_linearized(
            x=x,
            y=y,
            z=z,
            dh=dh,
            gradx=gradx,
            grady=grady,
            params_fit_or_bin=params_fit_or_bin,
            only_translation=only_translation,
            errors=errors,
            pixel_size=pixel_size,
            force_opti=force_opti,
            **kwargs,
        )
        return step_matrix, err_beta


def _lzd_iteration_step(
    matrix: NDArrayf,
    sub_rst: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    sub_pts: NDArrayf,
    sub_coords: tuple[NDArrayf, NDArrayf],
    centroid: tuple[float, float, float],
    sub_gradx: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    sub_grady: Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf],
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
    linearized: bool,
    sub_errors: tuple[Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf], Callable, Callable[[tuple[NDArrayf, NDArrayf]], NDArrayf], Callable],
    pixel_size: float,
    force_opti: Literal["ols", "gls", "tls"] = None,
) -> tuple[NDArrayf, dict[str, float], NDArrayf | None]:
    """
    Iteration step of Least Z-difference coregistration from Rosenholm and Torlegård (1988).

    The function uses 2D array interpolators of the DEM input and its gradient, computed only once outside iteration
    loops, to optimize computing time.

    Returns optimized affine matrix and statistics (mostly offsets) to compare to tolerances for this iteration step.

    :param matrix: Affine transform matrix.
    :param sub_rst: Interpolator for 2D array of DEM.
    :param sub_pts: Subsampled elevation for other elevation data (DEM or point) as 1D array.
    :param sub_coords: Subsampled X/Y coordinates arrays for point or second DEM input as 1D arrays.
    :param centroid: Centroid removed from point or second DEM input (to facilitate numerical convergence).
    :param sub_gradx: Interpolator for 2D array of DEM gradient along X axis.
    :param sub_grady: Interpolator for 2D array of DEM gradient along Y axis.
    :param params_fit_or_bin: Dictionary of fitting and binning parameters.
    :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
        rotation as default.
    :param sub_errors: Input errors.
    :param pixel_size: Pixel size in meters.

    :return Affine matrix, Iteration statistics to compare to tolerances, Error outputs.
    """

    # Apply transform matrix from previous steps
    pts_epc = np.vstack((sub_coords[0], sub_coords[1], sub_pts))
    trans_pts_epc = _apply_matrix_pts_mat(pts_epc, matrix=matrix, centroid=centroid)

    # Apply step fit
    step_matrix, err_beta = _lzd_fit(
        matrix=matrix,
        sub_rst=sub_rst,
        trans_pts_epc=trans_pts_epc,
        centroid=centroid,
        sub_gradx=sub_gradx,
        sub_grady=sub_grady,
        params_fit_or_bin=params_fit_or_bin,
        only_translation=only_translation,
        sub_errors=sub_errors,
        pixel_size=pixel_size,
        force_opti=force_opti,
        linearized=linearized,
    )

    # Increment transformation matrix by step
    new_matrix = step_matrix @ matrix

    # Compute statistics to know if they reached tolerance
    # (offsets in translation/rotation, but can also be other statistics)
    translations = step_matrix[:3, 3]
    offset_translation = np.sqrt(np.sum(translations ** 2))
    rotations = step_matrix[:3, :3]
    offset_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    step_statistics = {"translation": offset_translation, "rotation": offset_rotation}

    return new_matrix, step_statistics, err_beta


def lzd(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    ref_transform: rio.transform.Affine,
    tba_transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    max_iterations: int,
    tolerance_translation: float | None,
    tolerance_rotation: float | None,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
    linearized: bool = True,
    sig_ref: NDArrayf | gpd.GeoDataFrame = None,
    sig_tba: NDArrayf | gpd.GeoDataFrame = None,
    corr_ref: Callable[[NDArrayf, NDArrayf], NDArrayf] = None,
    corr_tba: Callable[[NDArrayf, NDArrayf], NDArrayf] = None,
    force_opti: Literal["ols", "gls", "tls"] | None = None,
) -> tuple[NDArrayf, tuple[float, float, float], int, OutIterativeDict, NDArrayf | None]:
    """
    Least Z-differences coregistration.
    See Rosenholm and Torlegård (1988),
    https://www.asprs.org/wp-content/uploads/pers/1988journal/oct/1988_oct_1385-1389.pdf.

    This function subsamples input data, then runs LZD iteration steps to optimize its fit function until
    convergence or a maximum of iterations is reached.

    The function assumes we have two DEMs, or a DEM and an elevation point cloud, in the same CRS.
    """

    logging.info("Running LZD coregistration")

    transform = ref_transform if ref_transform is not None else tba_transform
    pixel_size = _res(transform)[0]
    logging.info(f"Using {"reference" if ref_transform is not None else "to-be-aligned"} "
                 f"as continuous grid for deriving gradients.")

    # Check that DEM CRS is projected, otherwise slope is not correctly calculated
    if not crs.is_projected:
        raise NotImplementedError(
            f"LZD coregistration only works with a projected CRS, current CRS is {crs}. Reproject "
            f"your DEMs with DEM.reproject() in a local projected CRS such as UTM, that you can find "
            f"using DEM.get_metric_crs()."
        )

    # First, derive auxiliary variables of Nuth and Kääb (slope tangent, and aspect) for any point-raster input
    gradx, grady = _lzd_aux_vars(ref_elev=ref_elev, tba_elev=tba_elev, transform=transform)

    # Then, perform preprocessing: subsampling and interpolation of inputs and auxiliary vars at same points
    sub_ref, sub_tba, _ = _subsample_rst_pts(
        subsample=params_random["subsample"],
        random_state=params_random["random_state"],
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        ref_transform=ref_transform,
        tba_transform=tba_transform,
        sampling_strategy="same_xy",  # This sampling needs to be enforced for LZD
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
    )
    # Simplify output given that the X/Y coordinates are the same
    sub_coords = (sub_ref[0, :], sub_ref[1, :])
    sub_ref = sub_ref[2, :]
    sub_tba = sub_tba[2, :]

    # Define inputs of methods, depending on if they are point or raster data
    if not isinstance(ref_elev, gpd.GeoDataFrame):
        ref = "rst"
        sub_pts = sub_tba
        sub_rst = _reproject_horizontal_shift_samecrs(ref_elev, src_transform=transform, return_interpolator=True)
    else:
        ref = "pts"
        sub_pts = sub_ref
        sub_rst = _reproject_horizontal_shift_samecrs(tba_elev, src_transform=transform, return_interpolator=True)
    # We use interpolators of gradx and grady in any case
    sub_gradx = _reproject_horizontal_shift_samecrs(gradx, src_transform=transform, return_interpolator=True)
    sub_grady = _reproject_horizontal_shift_samecrs(grady, src_transform=transform, return_interpolator=True)

    # If input errors were defined
    if any(x is not None for x in [sig_ref, sig_tba, corr_ref, corr_tba]):
        if sig_ref is not None:
            sub_sig_ref = _reproject_horizontal_shift_samecrs(sig_ref, src_transform=transform, return_interpolator=True)
        else:
            sub_sig_ref = None
        if sig_tba is not None:
            sub_sig_tba = _reproject_horizontal_shift_samecrs(sig_tba, src_transform=transform, return_interpolator=True)
        else:
            sub_sig_tba = None
        # We want to pass the errors as "sigma_other", "correlation_other", "sigma_grid", "correlation_grid",
        # where "grid" is the DEM used for deriving the gradient
        if ref_transform is not None:
            sub_errors = (sub_sig_tba, corr_tba, sub_sig_ref, corr_ref)
        else:
            sub_errors = (sub_sig_ref, corr_ref, sub_sig_tba, corr_tba)
    else:
        sub_errors = None

    # Estimate centroid to use
    centroid = _get_centroid_scale(ref_elev=ref_elev, transform=transform, z_name=z_name)[0]

    logging.info("Iteratively estimating rigid transformation:")
    # Iterate through method until tolerance or max number of iterations is reached
    init_matrix = np.eye(4)  # Initial matrix is the identity transform
    constant_inputs = (
        sub_rst,
        sub_pts,
        sub_coords,
        centroid,
        sub_gradx,
        sub_grady,
        params_fit_or_bin,
        only_translation,
        linearized,
        sub_errors,
        pixel_size,
        force_opti
    )
    final_matrix, _, output_iterative, err_beta = _iterate_method(
        method=_lzd_iteration_step,
        iterating_input=init_matrix,
        constant_inputs=constant_inputs,
        tolerances={"translation": tolerance_translation, "rotation": tolerance_rotation},
        max_iterations=max_iterations,
    )

    # Invert matrix if reference was the point data
    if ref == "pts":
        final_matrix = invert_matrix(final_matrix)

    subsample_final = len(sub_pts)

    return final_matrix, centroid, subsample_final, output_iterative, err_beta


##################################
# Affine coregistration subclasses
##################################

AffineCoregType = TypeVar("AffineCoregType", bound="AffineCoreg")


class AffineCoreg(Coreg):
    """
    Generic affine coregistration class.

    Builds additional common affine methods on top of the generic Coreg class.
    Made to be subclassed.
    """

    _fit_called: bool = False  # Flag to check if the .fit() method has been called.
    _is_affine: bool | None = None
    _is_translation: bool | None = None

    def __init__(
        self,
        subsample: float | int = 1.0,
        matrix: NDArrayf | None = None,
        meta: dict[str, Any] | None = None,
        initial_shift: tuple[Number, Number] | tuple[Number, Number, Number] | None = None,
    ) -> None:
        """Instantiate a generic AffineCoreg method."""

        if meta is None:
            meta = {}
        # Define subsample size
        meta.update({"subsample": subsample})

        # Define initial shift
        if initial_shift is not None:
            meta.update({"initial_shift": initial_shift})

        super().__init__(meta=meta)

        if matrix is not None:
            valid_matrix = _make_matrix_valid(matrix)
            self._meta["outputs"]["affine"] = {"matrix": valid_matrix}

        self._is_affine = True

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        ** kwargs: Any,
    ) -> None:
        raise NotImplementedError("This method is meant to be subclassed.")

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # We re-direct to a common _fit_any_rst_pts for that subclass
        # The subclass will raise an error if a certain input type is not supported
        # Affine registration need the original raster data without reprojection as overlap is not required
        self._fit_any_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
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

        # We re-direct to a common _fit_any_rst_pts for that subclass
        # The subclass will raise an error if a certain input type is not supported
        # Affine registration need the original raster data without reprojection as overlap is not required
        self._fit_any_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=transform if isinstance(ref_elev, np.ndarray) else None,
            tba_transform=transform if isinstance(tba_elev, np.ndarray) else None,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def _fit_pts_pts(
        self,
        ref_elev: gpd.GeoDataFrame,
        tba_elev: gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # We re-direct to a common _fit_any_rst_pts for that subclass
        # The subclass will raise an error if a certain input type is not supported
        # Affine registration need the original raster data without reprojection as overlap is not required
        self._fit_any_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=None,
            tba_transform=None,
            crs=crs,
            area_or_point=None,
            z_name=z_name,
            weights=weights,
            bias_vars=bias_vars,
            **kwargs,
        )

    def to_matrix(self) -> NDArrayf:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def to_translations(self) -> tuple[float, float, float]:
        """
        Extract X/Y/Z translations from the affine transformation matrix.

        :return: Easting, northing and vertical translations (in georeferenced unit).
        """

        matrix = self.to_matrix()
        shift_x, shift_y, shift_z = translations_rotations_from_matrix(matrix)[:3]

        return shift_x, shift_y, shift_z

    def to_rotations(self, return_degrees: bool = True) -> tuple[float, float, float]:
        """
        Extract X/Y/Z euler rotations (extrinsic convention) from the affine transformation matrix.

        Warning: This function only works for a rigid transformation (rotation and translation).

        :param return_degrees: Whether to return degrees, otherwise radians.

        :return: Extrinsic Euler rotations along easting, northing and vertical directions (degrees).
        """

        matrix = self.to_matrix()
        alpha1, alpha2, alpha3 = translations_rotations_from_matrix(matrix, return_degrees=return_degrees)[3:]

        return alpha1, alpha2, alpha3

    def centroid(self) -> tuple[float, float, float] | None:
        """Get the centroid of the coregistration, if defined."""
        meta_centroid = self._meta["outputs"]["affine"].get("centroid")

        if meta_centroid is None:
            return None

        # Unpack the centroid in case it is in an unexpected format (an array, list or something else).
        return meta_centroid[0], meta_centroid[1], meta_centroid[2]

    def _preprocess_rst_pts_subsample_interpolator(
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
    ) -> tuple[Callable[[float, float], NDArrayf], None | dict[str, NDArrayf]]:
        """
        Pre-process raster-raster or point-raster datasets into 1D arrays subsampled at the same points
        (and interpolated in the case of point-raster input).

        Return 1D arrays of reference elevation, to-be-aligned elevation and dictionary of 1D arrays of auxiliary
        variables at subsampled points.
        """

        # Get random parameters
        params_random = self._meta["inputs"]["random"]

        # Get subsample mask (a 2D array for raster-raster, a 1D array of length the point data for point-raster)
        sub_mask = _get_subsample_mask_pts_rst(
            subsample=params_random["subsample"],
            random_state=params_random["random_state"],
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            area_or_point=area_or_point,
            z_name=z_name,
            aux_vars=aux_vars,
        )

        # Return interpolator of elevation differences and subsampled auxiliary variables
        sub_dh_interpolator, sub_bias_vars = _subsample_on_mask_interpolator(
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

        # Return 1D arrays of subsampled points at the same location
        return sub_dh_interpolator, sub_bias_vars

    @classmethod
    def from_matrix(cls, matrix: NDArrayf) -> AffineCoreg:
        """
        Instantiate a generic Coreg class from a transformation matrix.

        :param matrix: A 4x4 transformation matrix. Shape must be (4,4).

        :raises ValueError: If the matrix is incorrectly formatted.

        :returns: The instantiated generic Coreg class.
        """
        if np.any(~np.isfinite(matrix)):
            raise ValueError(f"Matrix has non-finite values:\n{matrix}")

        valid_matrix = _make_matrix_valid(matrix)

        return cls(matrix=valid_matrix)

    @classmethod
    def from_translations(cls, x_off: float = 0.0, y_off: float = 0.0, z_off: float = 0.0) -> AffineCoreg:
        """
        Instantiate a generic Coreg class from a X/Y/Z translation.

        :param x_off: The offset to apply in the X (west-east) direction.
        :param y_off: The offset to apply in the Y (south-north) direction.
        :param z_off: The offset to apply in the Z (vertical) direction.

        :raises ValueError: If the given translation contained invalid values.

        :returns: An instantiated generic Coreg class.
        """
        matrix = matrix_from_translations_rotations(t1=x_off, t2=y_off, t3=z_off, alpha1=0.0, alpha2=0.0, alpha3=0.0)

        return cls.from_matrix(matrix)

    @classmethod
    def from_rotations(
        cls, x_rot: float = 0.0, y_rot: float = 0.0, z_rot: float = 0.0, use_degrees: bool = True
    ) -> AffineCoreg:
        """
        Instantiate a generic Coreg class from a X/Y/Z rotation.

        :param x_rot: The rotation to apply around the X (west-east) direction.
        :param y_rot: The rotation to apply around the Y (south-north) direction.
        :param z_rot: The rotation to apply around the Z (vertical) direction.
        :param use_degrees: Whether to use degrees for input angles, otherwise radians.

        :raises ValueError: If the given rotation contained invalid values.

        :returns: An instantiated generic Coreg class.
        """

        matrix = matrix_from_translations_rotations(
            t1=0.0, t2=0.0, t3=0.0, alpha1=x_rot, alpha2=y_rot, alpha3=z_rot, use_degrees=use_degrees
        )

        return cls.from_matrix(matrix)

    def _to_matrix_func(self) -> NDArrayf:
        # FOR DEVELOPERS: This function needs to be implemented if the `self._meta['matrix']` keyword is not None.

        # Try to see if a matrix exists.
        meta_matrix = self._meta["outputs"]["affine"].get("matrix")
        if meta_matrix is not None:
            assert meta_matrix.shape == (4, 4), f"Invalid _meta matrix shape. Expected: (4, 4), got {meta_matrix.shape}"
            return meta_matrix

        raise NotImplementedError("This should be implemented by subclassing")


class VerticalShift(AffineCoreg):
    """
    Vertical translation alignment.

    Estimates the mean vertical offset between two elevation datasets based on a reductor function (median, mean, or
    any custom reductor function).

    The estimated vertical shift is stored in the `self.meta["outputs"]["affine"]` key "shift_z" (in unit of the
    elevation dataset inputs, typically meters).
    """

    def __init__(
        self, vshift_reduc_func: Callable[[NDArrayf], np.floating[Any]] = np.median, subsample: float | int = 1.0
    ) -> None:  # pylint:
        # disable=super-init-not-called
        """
        Instantiate a vertical shift alignment object.

        :param vshift_reduc_func: Reductor function to estimate the central tendency of the vertical shift.
            Defaults to the median.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        self._meta: CoregDict = {}  # All __init__ functions should instantiate an empty dict.

        super().__init__(meta={"vshift_reduc_func": vshift_reduc_func}, subsample=subsample)

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:
        """Estimate the vertical shift using the vshift_func."""

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]

        vshift, subsample_final = vertical_shift(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
            params_random=params_random,
            vshift_reduc_func=self._meta["inputs"]["affine"]["vshift_reduc_func"],
            z_name=z_name,
            weights=weights,
            **kwargs,
        )

        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}
        self._meta["outputs"]["affine"] = {"shift_z": vshift}

    def _to_matrix_func(self) -> NDArrayf:
        """Convert the vertical shift to a transform matrix."""
        empty_matrix = np.diag(np.ones(4, dtype=float))

        empty_matrix[2, 3] = self._meta["outputs"]["affine"]["shift_z"]

        return empty_matrix


class ICP(AffineCoreg):
    """
    Iterative closest point registration.

    Estimates a rigid transform (rotation + translation) between two elevation datasets.

    The ICP method can be:

     - Point-to-point of Besl and McKay (1992), https://doi.org/10.1117/12.57955, where the loss function is computed
       on the 3D distances of closest points (original method),
     - Point-to-plane of Chen and Medioni (1992), https://doi.org/10.1016/0262-8856(92)90066-C where the loss function
       is computed on the 3D distances of closest points projected on the plane normals (generally faster and more
       accurate, particularly for point clouds representing contiguous surfaces).


    Other ICP options are:

    - Linearized approximation of the point-to-plane least-square optimization of Low (2004),
      https://www.cs.unc.edu/techreports/04-004.pdf (faster, only for rotations below 30° degrees),
    - Picky ICP of Zinsser et al. (2003), https://doi.org/10.1109/ICIP.2003.1246775, where closest point pairs matched
      to the same point in the reference point cloud are removed, keeping only the pair with the minimum distance
      (useful for datasets with different extents that won't exactly overlap).

    The estimated transform is stored in the `self.meta["outputs"]["affine"]` key "matrix", with rotation centered
    on the coordinates in the key "centroid". The translation parameters are also stored individually in the
    keys "shift_x", "shift_y" and "shift_z" (in georeferenced units for horizontal shifts, and unit of the
    elevation dataset inputs for the vertical shift).
    """

    def __init__(
        self,
        method: Literal["point-to-point", "point-to-plane"] = "point-to-plane",
        picky: bool = True,
        linearized: bool = True,
        only_translation: bool = False,
        anisotropic: Literal["xy_vs_z", "per_axis"] | None = "xy_vs_z",
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        max_iterations: int = 20,
        tolerance_translation: float | None = 0.01,
        tolerance_rotation: float | None = 0.001,
        sampling_strategy: Literal["independent", "same_xy", "iterative_same_xy"] = "same_xy",
        standardize: bool = True,
        subsample: float | int = 5e5,
        trim_residuals: bool = False,
        trim_central_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        trim_spread_statistic: Callable[[NDArrayf], np.floating[Any]] = nmad,
        trim_spread_coverage: float = 3,
        trim_iterative: bool = False,
    ) -> None:
        """
        Instantiate an ICP coregistration object.

        :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
            that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
            projected on normals.
        :param picky: Whether to use the duplicate removal for pairs of closest points of Zinsser et al. (2003).
        :param linearized: Whether to use linearized rotation approximation of Low (2004) (only available for
            "point-to-plane").
        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance_translation: Magnitude of iteration translation (in georeferenced unit) at which to stop the
            iterations (once other tolerances are also reached, if any).
        :param tolerance_rotation: Magnitude of iteration rotation (in degrees) at which to stop the iterations (once
            other tolerances are also reached, if any)
        :param standardize: Whether to standardize input point clouds to the unit sphere for numerical convergence
            (tolerance is also standardized by the same factor).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        if tolerance_rotation is None and tolerance_translation is None:
            raise ValueError("At least one tolerance must be defined.")

        meta = {
            "icp_method": method,
            "icp_picky": picky,
            "linearized": linearized,
            "fit_minimizer": fit_minimizer,
            "fit_loss_func": fit_loss_func,
            "max_iterations": max_iterations,
            "tolerance_translation": tolerance_translation,
            "tolerance_rotation": tolerance_rotation,
            "only_translation": only_translation,
            "sampling_strategy": sampling_strategy,
            "standardize": standardize,
            "anisotropic": anisotropic,
        }
        if trim_residuals:
            meta_input_filtering = {
                "trim_residuals": trim_residuals,
                "trim_spread_statistic": trim_spread_statistic,
                "trim_spread_coverage": trim_spread_coverage,
                "trim_central_statistic": trim_central_statistic,
                "trim_iterative": trim_iterative,
            }
            meta.update(meta_input_filtering)

        super().__init__(subsample=subsample, meta=meta)

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        matrix, centroid, subsample_final, output_iterative = icp(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance_translation=self._meta["inputs"]["iterative"]["tolerance_translation"],
            tolerance_rotation=self._meta["inputs"]["iterative"]["tolerance_rotation"],
            method=self._meta["inputs"]["specific"]["icp_method"],
            picky=self._meta["inputs"]["specific"]["icp_picky"],
            linearized=self._meta["inputs"]["affine"]["linearized"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
            sampling_strategy=self._meta["inputs"]["specific"]["sampling_strategy"],
            standardize=self._meta["inputs"]["affine"]["standardize"],
            anisotropic=self._meta["inputs"]["affine"]["anisotropic"],
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            centroid=centroid,
            matrix=matrix,
            shift_x=matrix[0, 3],
            shift_y=matrix[1, 3],
            shift_z=matrix[2, 3],
        )
        self._meta["outputs"]["affine"] = output_affine
        self._meta["outputs"]["iterative"] = output_iterative
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}


class CPD(AffineCoreg):
    """
    Coherent Point Drift coregistration, based on Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46.

    Supports "local surface geometry" variant from Liu et al. (2021), https://doi.org/10.1109/ICCV48922.2021.0150, that
    relies on normals.

    Estimate translations and rotations.

    The estimated transform is stored in the `self.meta["outputs"]["affine"]` key "matrix", with rotation centered
    on the coordinates in the key "centroid". The translation parameters are also stored individually in the
    keys "shift_x", "shift_y" and "shift_z" (in georeferenced units for horizontal shifts, and unit of the
    elevation dataset inputs for the vertical shift).
    """

    def __init__(
        self,
        weight: float = 0,
        only_translation: bool = False,
        lsg: bool = True,
        anisotropic: Literal["xy_vs_z", "per_axis"] | None = "xy_vs_z",
        max_iterations: int = 100,
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        tolerance_translation: float | None = 0.01,
        tolerance_rotation: float | None = 0.001,
        tolerance_objective_func: float | None = 0.001,
        sampling_strategy: Literal["independent", "same_xy", "iterative_same_xy"] = "same_xy",
        standardize: bool = True,
        subsample: int | float = 5e3,
    ):
        """
        Instantiate a CPD coregistration object.

        :param weight: Weight contribution of the uniform distribution to account for outliers, from 0 (inclusive) to
            1 (exclusive).
        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param lsg: Whether to use local surface geometry (LSG) variant using normals from Liu et al. (2021).
        :param max_iterations: Maximum allowed iterations before stopping.
        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param tolerance_translation: Magnitude of iteration translation (in georeferenced unit) at which to stop the
            iterations (once other tolerances are also reached, if any).
        :param tolerance_rotation: Magnitude of iteration rotation (in degrees) at which to stop the iterations (once
            other tolerances are also reached, if any)
        :param tolerance_objective_func: Magnitude of iteration objective function value (see Q in Myronenko and Song (2010))
            at which to stop the iterations (once other tolerances are also reached, if any).
        :param standardize: Whether to standardize input point clouds to the unit sphere for numerical convergence
            (tolerance is also standardized by the same factor).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        if tolerance_objective_func is None and tolerance_rotation is None and tolerance_translation is None:
            raise ValueError("At least one tolerance must be defined.")

        meta_cpd = {
            "max_iterations": max_iterations,
            "tolerance_objective_func": tolerance_objective_func,
            "tolerance_translation": tolerance_translation,
            "tolerance_rotation": tolerance_rotation,
            "cpd_weight": weight,
            "fit_minimizer": fit_minimizer,
            "fit_loss_func": fit_loss_func,
            "cpd_lsg": lsg,
            "only_translation": only_translation,
            "sampling_strategy": sampling_strategy,
            "standardize": standardize,
            "anisotropic": anisotropic,
        }

        super().__init__(subsample=subsample, meta=meta_cpd)  # type: ignore

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fitorbin = self._meta["inputs"]["fitorbin"]
        # Call method
        matrix, centroid, subsample_final, output_iterative = cpd(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            lsg=self._meta["inputs"]["specific"]["cpd_lsg"],
            weight_cpd=self._meta["inputs"]["specific"]["cpd_weight"],
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance_translation=self._meta["inputs"]["iterative"]["tolerance_translation"],
            tolerance_rotation=self._meta["inputs"]["iterative"]["tolerance_rotation"],
            tolerance_q=self._meta["inputs"]["iterative"]["tolerance_objective_func"],
            sampling_strategy=self._meta["inputs"]["specific"]["sampling_strategy"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
            standardize=self._meta["inputs"]["affine"]["standardize"],
            anisotropic=self._meta["inputs"]["affine"]["anisotropic"],
            params_fit_or_bin=params_fitorbin,
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            centroid=centroid,
            matrix=matrix,
            shift_x=matrix[0, 3],
            shift_y=matrix[1, 3],
            shift_z=matrix[2, 3],
        )
        self._meta["outputs"]["affine"] = output_affine
        self._meta["outputs"]["iterative"] = output_iterative
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}


class NuthKaab(AffineCoreg):
    """
    Nuth and Kääb (2011) coregistration, https://doi.org/10.5194/tc-5-271-2011.

    Estimate horizontal and vertical translations by iterative slope/aspect alignment.

    The translation parameters are stored in the `self.meta["outputs"]["affine"]` keys "shift_x", "shift_y" and
    "shift_z" (in georeferenced units for horizontal shifts, and unit of the elevation dataset inputs for the
    vertical shift), as well as in the "matrix" transform.
    """

    def __init__(
        self,
        max_iterations: int = 20,
        tolerance_translation: float = 0.001,
        bin_before_fit: bool = True,
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        bin_sizes: int | dict[str, int | Iterable[float]] = 72,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        subsample: int | float = 5e5,
        trim_residuals: bool = False,
        trim_central_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        trim_spread_statistic: Callable[[NDArrayf], np.floating[Any]] = nmad,
        trim_spread_coverage: float = 3,
        trim_iterative: bool = False,
        vertical_shift: bool = True,
        initial_shift: tuple[Number, Number] | tuple[Number, Number, Number] | None = None,
    ) -> None:
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance_translation: Magnitude of iteration translation (in georeferenced unit) at which to stop the
            iterations.
        :param bin_before_fit: Whether to bin data before fitting the coregistration function. For the Nuth and Kääb
            (2011) algorithm, this corresponds to bins of aspect to compute statistics on dh/tan(slope).
        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        :param vertical_shift: Whether to apply the vertical shift or not (default is True).
        :param initial_shift: Tuple containing x, y and z shifts (in georeferenced units).
            These shifts are applied before the fit() part.
        """

        self.vertical_shift = vertical_shift

        # Input checks
        _check_inputs_bin_before_fit(
            bin_before_fit=bin_before_fit, bin_sizes=bin_sizes, bin_statistic=bin_statistic, fit_minimizer=fit_minimizer
        )

        # Define iterative parameters and vertical shift
        meta_input_iterative = {
            "max_iterations": max_iterations,
            "tolerance_translation": tolerance_translation,
            "apply_vshift": vertical_shift,
        }
        if trim_residuals:
            meta_input_filtering = {
                "trim_residuals": trim_residuals,
                "trim_spread_statistic": trim_spread_statistic,
                "trim_spread_coverage": trim_spread_coverage,
                "trim_central_statistic": trim_central_statistic,
                "trim_iterative": trim_iterative,
            }
            meta_input_iterative.update(meta_input_filtering)

        # Test consistency of the estimated initial shift given if provided
        if initial_shift:
            if not (
                isinstance(initial_shift, tuple)
                and (len(initial_shift) == 2 or len(initial_shift) == 3)
                and all(isinstance(val, (float, int)) for val in initial_shift)
            ):
                raise ValueError("Argument `initial_shift` must be a tuple of exactly two or three numerical values.")

            if len(initial_shift) == 2:
                initial_shift += (0,)
            elif initial_shift[2] != 0:  # initial z shift is not taken into account
                initial_shift = (*initial_shift[:2], 0)
                warnings.warn(
                    "Initial shift in altitude is currently work in progress.",
                    category=UserWarning,
                )

        # Define parameters exactly as in BiasCorr, but with only "fit" or "bin_and_fit" as option, so a bin_before_fit
        # boolean, no bin apply option, and fit_func is predefined
        if not bin_before_fit:
            meta_fit = {
                "fit_or_bin": "fit",
                "fit_func": _nuth_kaab_fit_func,
                "fit_minimizer": fit_minimizer,
                "fit_loss_func": fit_loss_func,
            }
            meta_fit.update(meta_input_iterative)
            super().__init__(subsample=subsample, meta=meta_fit, initial_shift=initial_shift)  # type: ignore
        else:
            meta_bin_and_fit = {
                "fit_or_bin": "bin_and_fit",
                "fit_func": _nuth_kaab_fit_func,
                "fit_minimizer": fit_minimizer,
                "fit_loss_func": fit_loss_func,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
            }
            meta_bin_and_fit.update(meta_input_iterative)
            super().__init__(
                subsample=subsample, meta=meta_bin_and_fit, initial_shift=initial_shift
            )  # t)  # type: ignore

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        (easting_offset, northing_offset, vertical_offset), subsample_final, output_iterative = nuth_kaab(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance_translation=self._meta["inputs"]["iterative"]["tolerance_translation"],
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            shift_x=-easting_offset, shift_y=-northing_offset, shift_z=vertical_offset * self.vertical_shift
        )
        self._meta["outputs"]["affine"] = output_affine
        self._meta["outputs"]["iterative"] = output_iterative
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transformation matrix from the estimated offsets."""

        # We add a translation, on the last column
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += self._meta["outputs"]["affine"]["shift_x"]
        matrix[1, 3] += self._meta["outputs"]["affine"]["shift_y"]
        matrix[2, 3] += self._meta["outputs"]["affine"]["shift_z"]

        return matrix


class LZD(AffineCoreg):
    """
    Least Z-difference coregistration.

    See Rosenholm and Torlegård (1988),
    https://www.asprs.org/wp-content/uploads/pers/1988journal/oct/1988_oct_1385-1389.pdf.

    Estimates a rigid transform (rotation + translation) between two elevation datasets.

    The estimated transform is stored in the `self.meta["outputs"]["affine"]` key "matrix", with rotation centered
    on the coordinates in the key "centroid". The translation parameters are also stored individually in the
    keys "shift_x", "shift_y" and "shift_z" (in georeferenced units for horizontal shifts, and unit of the
    elevation dataset inputs for the vertical shift).
    """

    def __init__(
        self,
        only_translation: bool = False,
        linearized: bool = True,
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        max_iterations: int = 20,
        tolerance_translation: float | None = 0.01,
        tolerance_rotation: float | None = 0.001,
        subsample: float | int = 5e5,
        trim_residuals: bool = False,
        trim_central_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        trim_spread_statistic: Callable[[NDArrayf], np.floating[Any]] = nmad,
        trim_spread_coverage: float = 3,
        trim_iterative: bool = False,
    ):
        """
         Instantiate an LZD coregistration object.

        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance_translation: Magnitude of iteration translation (in georeferenced unit) at which to stop the
            iterations (once other tolerances are also reached, if any).
        :param tolerance_rotation: Magnitude of iteration rotation (in degrees) at which to stop the iterations (once
            other tolerances are also reached, if any).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        if tolerance_rotation is None and tolerance_translation is None:
            raise ValueError("At least one tolerance must be defined.")

        meta = {
            "fit_minimizer": fit_minimizer,
            "fit_loss_func": fit_loss_func,
            "max_iterations": max_iterations,
            "tolerance_translation": tolerance_translation,
            "tolerance_rotation": tolerance_rotation,
            "only_translation": only_translation,
            "linearized": linearized,
        }
        if trim_residuals:
            meta_input_filtering = {
                "trim_residuals": trim_residuals,
                "trim_spread_statistic": trim_spread_statistic,
                "trim_spread_coverage": trim_spread_coverage,
                "trim_central_statistic": trim_central_statistic,
                "trim_iterative": trim_iterative,
            }
            meta.update(meta_input_filtering)
        super().__init__(subsample=subsample, meta=meta)

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> None:

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        matrix, centroid, subsample_final, output_iterative, err_beta = lzd(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance_translation=self._meta["inputs"]["iterative"]["tolerance_translation"],
            tolerance_rotation=self._meta["inputs"]["iterative"]["tolerance_rotation"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
            linearized=self._meta["inputs"]["affine"]["linearized"],
            sig_tba=kwargs.get("sig_tba"),
            sig_ref=kwargs.get("sig_ref"),
            corr_ref=kwargs.get("corr_ref"),
            corr_tba=kwargs.get("corr_tba"),
            force_opti=kwargs.get("force_opti"),
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            centroid=centroid,
            matrix=matrix,
            shift_x=matrix[0, 3],
            shift_y=matrix[1, 3],
            shift_z=matrix[2, 3],
        )
        self._meta["outputs"]["affine"] = output_affine
        self._meta["outputs"]["iterative"] = output_iterative
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}
        if err_beta is not None:
            self._meta["outputs"]["fitorbin"] = {"fit_perr": err_beta}


class DhMinimize(AffineCoreg):
    """
    (DEPRECATED: To replicate the same behaviour, use LZD with linearized=False, only_translation=True,
    fit_minimizer=scipy.optimize.minimize and fit_loss_func=nmad)

    Elevation difference minimization coregistration.

    Estimates vertical and horizontal translations.

    The translation parameters are stored in the `self.meta["outputs"]["affine"]` keys "shift_x", "shift_y" and
    "shift_z" (in georeferenced units for horizontal shifts, and unit of the elevation dataset inputs for the
    vertical shift), as well as in the "matrix" transform.
    """

    def __init__(
        self,
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.minimize,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] = nmad,
        subsample: int | float = 5e5,
    ) -> None:
        """
        Instantiate dh minimization object.

        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        warnings.warn(message="DhMinimize is deprecated due to redundancy with LZD. To replicate "
                              "the old behaviour, use LZD with linearized=False, only_translation=True, "
                              "fit_optimizer=scipy.optimize.minimize and fit_loss_func=nmad",
                      category=DeprecationWarning)

        meta_fit = {"fit_or_bin": "fit", "fit_minimizer": fit_minimizer, "fit_loss_func": fit_loss_func}
        super().__init__(subsample=subsample, meta=meta_fit)  # type: ignore

    def _fit_any_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        ref_transform: rio.transform.Affine,
        tba_transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        area_or_point: Literal["Area", "Point"] | None,
        z_name: str | None = None,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ):

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        (easting_offset, northing_offset, vertical_offset), subsample_final = dh_minimize(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            ref_transform=ref_transform,
            tba_transform=tba_transform,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            **kwargs,
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(shift_x=easting_offset, shift_y=northing_offset, shift_z=vertical_offset)
        self._meta["outputs"]["affine"] = output_affine
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transformation matrix from the estimated offsets."""

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += self._meta["outputs"]["affine"]["shift_x"]
        matrix[1, 3] += self._meta["outputs"]["affine"]["shift_y"]
        matrix[2, 3] += self._meta["outputs"]["affine"]["shift_z"]

        return matrix

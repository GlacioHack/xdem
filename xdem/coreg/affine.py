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

"""Affine coregistration classes."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Iterable, Literal, TypeVar

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import scipy.optimize
import scipy.spatial
from geoutils.interface.interpolate import _interp_points
from geoutils.raster.georeferencing import _coords, _res
from geoutils.stats import nmad
from tqdm import trange

from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    InFitOrBinDict,
    InRandomDict,
    OutAffineDict,
    _apply_matrix_pts_mat,
    _bin_or_and_fit_nd,
    _get_subsample_mask_pts_rst,
    _preprocess_pts_rst_subsample,
    _reproject_horizontal_shift_samecrs,
    invert_matrix,
    matrix_from_translations_rotations,
    translations_rotations_from_matrix,
)

try:
    import pytransform3d.rotations
    import pytransform3d.transformations

    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False

######################################
# Generic functions for affine methods
######################################


def _check_inputs_bin_before_fit(
    bin_before_fit: bool,
    fit_optimizer: Callable[..., tuple[NDArrayf, Any]],
    bin_sizes: int | dict[str, int | Iterable[float]],
    bin_statistic: Callable[[NDArrayf], np.floating[Any]],
) -> None:
    """
    Check input types of fit or bin_and_fit affine functions.

    :param bin_before_fit: Whether to bin data before fitting the coregistration function.
    :param fit_optimizer: Optimizer to minimize the coregistration function.
    :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
    :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
    """

    if not callable(fit_optimizer):
        raise TypeError(
            "Argument `fit_optimizer` must be a function (callable), " "got {}.".format(type(fit_optimizer))
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
    tolerance: float,
    max_iterations: int,
) -> Any:
    """
    Function to iterate a method (e.g. ICP, Nuth and Kääb) until it reaches a tolerance or maximum number of iterations.

    :param method: Method that needs to be iterated to derive a transformation. Takes argument "inputs" as its input,
        and outputs three terms: a "statistic" to compare to tolerance, "updated inputs" with this transformation, and
        the parameters of the transformation.
    :param iterating_input: Iterating input to method, should be first argument.
    :param constant_inputs: Constant inputs to method, should be all positional arguments after first.
    :param tolerance: Tolerance to reach for the method statistic (i.e. maximum value for the statistic).
    :param max_iterations: Maximum number of iterations for the method.

    :return: Final output of iterated method.
    """

    # Initiate inputs
    new_inputs = iterating_input

    # Iteratively run the analysis until the maximum iterations or until the error gets low enough
    # If logging level <= INFO, will use progressbar and print additional statements
    pbar = trange(max_iterations, disable=logging.getLogger().getEffectiveLevel() > logging.INFO, desc="   Progress")
    for i in pbar:

        # Apply method and get new statistic to compare to tolerance, new inputs for next iterations, and
        # outputs in case this is the final one
        new_inputs, new_statistic = method(new_inputs, *constant_inputs)

        # Print final results
        # TODO: Allow to pass a string to _iterate_method on how to print/describe exactly the iterating input
        if logging.getLogger().getEffectiveLevel() <= logging.INFO:
            pbar.write(f"      Iteration #{i + 1:d} - Offset: {new_inputs}; Magnitude: {new_statistic}")

        if i > 1 and new_statistic < tolerance:
            if logging.getLogger().getEffectiveLevel() <= logging.INFO:
                pbar.write(f"   Last offset was below the residual offset threshold of {tolerance} -> stopping")
            break

    return new_inputs


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


def _preprocess_pts_rst_subsample_interpolator(
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
    Mirrors coreg.base._preprocess_pts_rst_subsample, but returning an interpolator for efficiency in iterative methods.

    Pre-process raster-raster or point-raster datasets into an elevation difference interpolator at the same
    points, and subsample arrays for auxiliary variables, with subsampled coordinates to evaluate the interpolator.

    Returns dh interpolator, tuple of 1D arrays of subsampled coordinates, and dictionary of 1D arrays of subsampled
    auxiliary variables.
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


def _standardize_epc(
    ref_epc: NDArrayf, tba_epc: NDArrayf, scale_std: bool = True
) -> tuple[NDArrayf, NDArrayf, tuple[float, float, float], float]:
    """
    Standardize elevation point clouds by getting centroid and standardization factor using median statistics.

    :param ref_epc: Reference point cloud.
    :param tba_epc: To-be-aligned point cloud.
    :param scale_std: Whether to scale all axes by a factor.

    :return: Standardized point clouds, Centroid of standardization, Scale factor of standardization.
    """

    # Get centroid
    centroid = np.median(ref_epc, axis=1)

    # Subtract centroid from point clouds
    ref_epc = ref_epc - centroid[:, None]
    tba_epc = tba_epc - centroid[:, None]

    centroid = (centroid[0], centroid[1], centroid[2])

    if scale_std:
        # Get mean standardization factor for all axes
        std_fac = np.mean([nmad(ref_epc[0, :]), nmad(ref_epc[1, :]), nmad(ref_epc[2, :])])

        # Standardize point clouds
        ref_epc = ref_epc / std_fac
        tba_epc = tba_epc / std_fac
    else:
        std_fac = 1

    return ref_epc, tba_epc, centroid, std_fac


################################
# Affine coregistrations methods
# ##############################

##################
# 1/ Nuth and Kääb
##################


def _nuth_kaab_fit_func(xx: NDArrayf, *params: tuple[float, float, float]) -> NDArrayf:
    """
    Nuth and Kääb (2011) fitting function.

    Describes the elevation differences divided by the slope tangente (y) as a 1D function of the aspect.

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

    # Make an initial guess of the a, b, and c parameters
    p0 = (3 * np.nanstd(y) / (2**0.5), 0.0, np.nanmean(y))

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
        p0=p0,
    )
    # Mypy: having results as "None" is impossible, but not understood through overloading of _bin_or_and_fit_nd...
    assert results is not None
    easting_offset = results[0][0] * np.sin(results[0][1])
    northing_offset = results[0][0] * np.cos(results[0][1])
    vertical_offset = results[0][2]

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
) -> tuple[tuple[float, float, float], float]:
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
    vshift = np.nanmedian(dh_step)
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
        dh=dh_step, slope_tan=slope_tan, aspect=aspect, params_fit_or_bin=params_fit_bin
    )

    # Increment the offsets by the new offset
    new_coords_offsets = (
        coords_offsets[0] + easting_offset * res[0],
        coords_offsets[1] + northing_offset * res[1],
        float(vshift),
    )

    # Compute statistic on offset to know if it reached tolerance
    # The easting and northing are here in pixels because of the slope/aspect derivation
    tolerance_statistic = np.sqrt(easting_offset**2 + northing_offset**2)

    return new_coords_offsets, tolerance_statistic


def nuth_kaab(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    tolerance: float,
    max_iterations: int,
    params_fit_or_bin: InFitOrBinDict,
    params_random: InRandomDict,
    z_name: str,
    weights: NDArrayf | None = None,
    **kwargs: Any,
) -> tuple[tuple[float, float, float], int]:
    """
    Nuth and Kääb (2011) iterative coregistration.

    This function subsamples input data, then runs Nuth and Kääb iteration steps to optimize its fit function until
    convergence or a maximum of iterations is reached.


    :return: Final estimated offset: east, north, vertical (in georeferenced units).
    """
    logging.info("Running Nuth and Kääb (2011) coregistration")

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
    sub_dh_interpolator, sub_aux_vars, subsample_final = _preprocess_pts_rst_subsample_interpolator(
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
    final_offsets = _iterate_method(
        method=_nuth_kaab_iteration_step,
        iterating_input=initial_offset,
        constant_inputs=constant_inputs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    return final_offsets, subsample_final


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
    transform: rio.transform.Affine,
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

    # Perform preprocessing: subsampling and interpolation of inputs and auxiliary vars at same points
    dh_interpolator, _, subsample_final = _preprocess_pts_rst_subsample_interpolator(
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
    transform: rio.transform.Affine,
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
    sub_ref, sub_tba, _, _ = _preprocess_pts_rst_subsample(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
    )
    # Get elevation difference
    dh = sub_ref - sub_tba

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


def _icp_fit_func(
    inputs: tuple[NDArrayf, NDArrayf, NDArrayf | None],
    t1: float,
    t2: float,
    t3: float,
    alpha1: float,
    alpha2: float,
    alpha3: float,
    method: Literal["point-to-point", "point-to-plane"],
) -> NDArrayf:
    """
    Fit function of ICP, a rigid transformation with 6 parameters (3 translations and 3 rotations) between closest
    points (that are fixed for this optimization and located at the same indexes, and update at each iterative step).

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

    :return Array of distances between closest points.
    """

    # Get inputs
    ref, tba, norm = inputs

    # Build an affine matrix for 3D translations and rotations
    matrix = matrix_from_translations_rotations(t1, t2, t3, alpha1, alpha2, alpha3, use_degrees=False)

    # Apply affine transformation
    trans_tba = _apply_matrix_pts_mat(mat=tba, matrix=matrix)

    # Compute residuals between reference points and nearest to-be-aligned points (located at the same indexes)
    # depending on type of ICP method

    # Point-to-point is simply the difference, from Besl and McKay (1992), https://doi.org/10.1117/12.57955
    if method == "point-to-point":
        diffs = (trans_tba - ref) ** 2
    # Point-to-plane used the normals, from Chen and Medioni (1992), https://doi.org/10.1016/0262-8856(92)90066-C
    # A priori, this method is faster based on Rusinkiewicz and Levoy (2001), https://doi.org/10.1109/IM.2001.924423
    elif method == "point-to-plane":
        assert norm is not None
        diffs = (trans_tba - ref) * norm
    else:
        raise ValueError("ICP method must be 'point-to-point' or 'point-to-plane'.")

    # Distance residuals summed for all 3 dimensions
    res = np.sum(diffs, axis=0)

    # For point-to-point, take the squareroot of the sum
    if method == "point-to-point":
        res = np.sqrt(res)

    return res


def _icp_fit_approx_lsq(
    ref: NDArrayf,
    tba: NDArrayf,
    norms: NDArrayf | None,
    weights: NDArrayf | None = None,
    method: Literal["point-to-point", "point-to-plane"] = "point-to-point",
) -> NDArrayf:
    """
    Linear approximation of the rigid transformation least-square optimization.

    See Low (2004), https://www.comp.nus.edu.sg/~lowkl/publications/lowk_point-to-plane_icp_techrep.pdf for the
    "point-to-plane" approximation.

    :param ref: Reference point cloud as Nx3 array.
    :param tba: To-be-aligned point cloud as Nx3 array.
    :param norms: Plane normals as Nx3 array.
    :param weights: Weights as Nx3 array.
    :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
        that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
        projected on normals.

    :return Affine transform matrix.
    """

    # Linear approximation of ICP least-squares

    # Point-to-plane
    if method == "point-to-plane":

        assert norms is not None

        # Define A and B as in Low (2004)
        B = np.expand_dims(np.sum(ref * norms, axis=1) - np.sum(tba * norms, axis=1), axis=1)
        A = np.hstack((np.cross(tba, norms), norms))

        # Choose if using weights or not
        if weights is not None:
            x = np.linalg.inv(A.T @ weights @ A) @ A.T @ weights @ B
        else:
            x = np.linalg.inv(A.T @ A) @ A.T @ B
        x = x.squeeze()

        # Convert back to affine matrix
        matrix = matrix_from_translations_rotations(
            alpha1=x[0], alpha2=x[1], alpha3=x[2], t1=x[3], t2=x[4], t3=x[5], use_degrees=False
        )

    else:
        raise ValueError("Fit optimizer 'lst_approx' of ICP is only available for point-to-plane method.")

    return matrix


def _icp_fit(
    ref: NDArrayf,
    tba: NDArrayf,
    norms: NDArrayf | None,
    method: Literal["point-to-point", "point-to-plane"],
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
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

    # Group inputs into a single array
    inputs = (ref, tba, norms)

    # If we use the linear approximation
    if isinstance(params_fit_or_bin["fit_minimizer"], str) and params_fit_or_bin["fit_minimizer"] == "lsq_approx":

        assert norms is not None

        matrix = _icp_fit_approx_lsq(ref.T, tba.T, norms.T, method=method)

    # Or we solve using any optimizer and loss function
    else:
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
    ref_epc_nearest_tree: scipy.spatial.KDTree,
    params_fit_or_bin: InFitOrBinDict,
    method: Literal["point-to-point", "point-to-plane"],
    picky: bool,
    only_translation: bool,
) -> tuple[NDArrayf, float]:
    """
    Iteration step of Iterative Closest Point coregistration.

    Returns affine transform optimized for this iteration, and tolerance parameters.

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

    :return: Affine transform matrix, Tolerance.
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
    )

    # Increment transformation matrix by step
    new_matrix = step_matrix @ matrix

    # Compute statistic on offset to know if it reached tolerance
    translations = step_matrix[:3, 3]

    tolerance_translation = np.sqrt(np.sum(translations) ** 2)
    # TODO: If we allow multiple tolerances in the future, here's the rotation tolerance
    # rotations = step_matrix[:3, :3]
    # tolerance_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    return new_matrix, tolerance_translation


def _icp_norms(dem: NDArrayf, transform: affine.Affine) -> tuple[NDArrayf, NDArrayf, NDArrayf]:
    """
    Derive normals from the DEM for "point-to-plane" method.

    :param dem: Array of DEM.
    :param transform: Transform of DEM.

    :return: Arrays of 3D normals: east, north and upward.
    """

    # Get DEM resolution
    resolution = _res(transform)

    # Generate the X, Y and Z normals
    gradient_x, gradient_y = np.gradient(dem)
    normal_east = np.sin(np.arctan(gradient_y / resolution[1])) * -1
    normal_north = np.sin(np.arctan(gradient_x / resolution[0]))
    normal_up = 1 - np.linalg.norm([normal_east, normal_north], axis=0)

    return normal_east, normal_north, normal_up


def icp(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    max_iterations: int,
    tolerance: float,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    method: Literal["point-to-point", "point-to-plane"] = "point-to-plane",
    picky: bool = False,
    only_translation: bool = False,
    standardize: bool = True,
) -> tuple[NDArrayf, tuple[float, float, float], int]:
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

    # Derive normals if method is point-to-plane, otherwise not
    if method == "point-to-plane":
        # We use the DEM to derive the normals
        if isinstance(ref_elev, np.ndarray):
            dem = ref_elev
        else:
            dem = tba_elev
        nx, ny, nz = _icp_norms(dem, transform)
        aux_vars = {"nx": nx, "ny": ny, "nz": nz}
    else:
        aux_vars = None

    # Pre-process point-raster inputs to the same subsampled points
    sub_ref, sub_tba, sub_aux_vars, sub_coords = _preprocess_pts_rst_subsample(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
        aux_vars=aux_vars,
        return_coords=True,
    )

    # Convert point clouds to Nx3 arrays for efficient calculations below
    ref_epc = np.vstack((sub_coords[0], sub_coords[1], sub_ref))
    tba_epc = np.vstack((sub_coords[0], sub_coords[1], sub_tba))
    if sub_aux_vars is not None:
        norms = np.vstack((sub_aux_vars["nx"], sub_aux_vars["ny"], sub_aux_vars["nz"]))
    else:
        norms = None

    # Remove centroid and standardize to facilitate numerical convergence
    ref_epc, tba_epc, centroid, std_fac = _standardize_epc(ref_epc, tba_epc, scale_std=standardize)
    tolerance /= std_fac

    # Define search tree outside of loop for performance
    ref_epc_nearest_tree = scipy.spatial.KDTree(ref_epc.T)

    # Iterate through method until tolerance or max number of iterations is reached
    init_matrix = np.eye(4)  # Initial matrix is the identity transform
    constant_inputs = (
        ref_epc,
        tba_epc,
        norms,
        ref_epc_nearest_tree,
        params_fit_or_bin,
        method,
        picky,
        only_translation,
    )
    final_matrix = _iterate_method(
        method=_icp_iteration_step,
        iterating_input=init_matrix,
        constant_inputs=constant_inputs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    # De-standardize
    final_matrix[:3, 3] *= std_fac

    # Get subsample size
    subsample_final = len(sub_ref)

    return final_matrix, centroid, subsample_final


#########################
# 5/ Coherent Point Drift
#########################


def _cpd_fit(
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    trans_tba_epc: NDArrayf,
    weight_cpd: float,
    sigma2: float,
    sigma2_min: float,
    scale: bool = False,
    only_translation: bool = False,
) -> tuple[NDArrayf, float, float]:
    """
    Fit step of Coherent Point Drift by expectation-minimization, with variance updating.

    See Fig. 2 of Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46 for equations below.

    Inspired from pycpd implementation: https://github.com/siavashk/pycpd/blob/master/pycpd/rigid_registration.py.
    """

    X, Y, TY = (ref_epc, tba_epc, trans_tba_epc)

    # Get shape of inputs
    N, D = X.shape
    M, _ = Y.shape

    # 0/ Initialize variance if not defined
    diff2 = (X[None, :, :] - TY[:, None, :]) ** 2
    if sigma2 is None:
        sigma2 = np.sum(diff2) / (D * N * M)

    # 1/ Expectation step

    # Sum only over D axis for numerator
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

    # 2/ Minimization step

    # Get centroid of each point cloud
    muX = np.divide(np.sum(PX, axis=0), Np)
    muY = np.divide(np.sum(np.dot(np.transpose(P), Y), axis=0), Np)

    # Subtract centroid from each point cloud
    X_hat = X - np.tile(muX, (N, 1))
    Y_hat = Y - np.tile(muY, (M, 1))
    YPY = np.dot(np.transpose(P1), np.sum(np.multiply(Y_hat, Y_hat), axis=1))

    # Derive A as in Fig. 2
    A = np.dot(np.transpose(X_hat), np.transpose(P))
    A = np.dot(A, Y_hat)

    # Singular value decomposition as per lemma 1
    if not only_translation:
        try:
            U, _, V = np.linalg.svd(A, full_matrices=True)
        except np.linalg.LinAlgError:
            raise ValueError("CPD coregistration numerics during np.linalg.svd(), try setting standardize=True.")
        C = np.ones((D,))
        C[D - 1] = np.linalg.det(np.dot(U, V))

        # Calculate the rotation matrix using Eq. 9
        R = np.transpose(np.dot(np.dot(U, np.diag(C)), V))
    else:
        R = np.eye(3)

    # Update scale and translation using Fig. 2
    if scale is True:
        s = np.trace(np.dot(np.transpose(A), np.transpose(R))) / YPY
    else:
        s = 1
    t = np.transpose(muX) - s * np.dot(np.transpose(R), np.transpose(muY))

    # Store in affine matrix
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = -t  # Translation is inverted here

    # 3/ Update variance and objective function

    # Update objective function using Eq. 7
    trAR = np.trace(np.dot(A, R))
    xPx = np.dot(np.transpose(Pt1), np.sum(np.multiply(X_hat, X_hat), axis=1))
    q = (xPx - 2 * s * trAR + s * s * YPY) / (2 * sigma2) + D * Np / 2 * np.log(sigma2)

    # Update variance using Fig. 2
    sigma2 = (xPx - s * trAR) / (Np * D)

    # If sigma2 gets negative, we use a minimal sigma value instead
    if sigma2 <= 0:
        sigma2 = sigma2_min

    return matrix, sigma2, q


def _cpd_iteration_step(
    iterating_input: tuple[NDArrayf, float, float],
    ref_epc: NDArrayf,
    tba_epc: NDArrayf,
    weight_cpd: float,
    sigma2_min: float,
    only_translation: bool,
) -> tuple[tuple[NDArrayf, float, float], float]:
    """
    Iteration step for Coherent Point Drift algorithm.

    Returns the updated iterating input (affine matrix, variance and objective function).
    """

    matrix, sigma2, q = iterating_input

    # Apply transform matrix from previous step
    trans_tba_epc = _apply_matrix_pts_mat(tba_epc, matrix=matrix, invert=True)

    # Fit to get new step transform
    # (Note: the CPD algorithm re-computes the full transform from the original target point cloud,
    # so there is no need to combine a step transform within the iteration as in ICP/LZD)
    new_matrix, new_sigma2, new_q = _cpd_fit(
        ref_epc=ref_epc.T,
        tba_epc=tba_epc.T,
        trans_tba_epc=trans_tba_epc.T,
        sigma2=sigma2,
        weight_cpd=weight_cpd,
        sigma2_min=sigma2_min,
        only_translation=only_translation,
    )

    # Compute statistic on offset to know if it reached tolerance
    tolerance_q = np.abs(q - new_q)

    # TODO: If we allow multiple tolerances in the future, here are the translation and rotation tolerances
    # translations = new_matrix[:3, 3]
    # rotations = new_matrix[:3, :3]
    # tolerance_translation = np.sqrt(np.sum(translations) ** 2)
    # tolerance_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    return (new_matrix, new_sigma2, new_q), tolerance_q


def cpd(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    weight_cpd: float,
    params_random: InRandomDict,
    max_iterations: int,
    tolerance: float,
    only_translation: bool = False,
    standardize: bool = True,
) -> tuple[NDArrayf, tuple[float, float, float], int]:
    """
    Main function for Coherent Point Drift coregistration.
    See Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46.

    This function subsamples input data, then runs CPD iteration steps to optimize its expectation-minimization until
    convergence or a maximum of iterations is reached.

    The function assumes we have two DEMs, or DEM and an elevation point cloud, in the same CRS.
    """

    # Pre-process point-raster inputs to the same subsampled points
    sub_ref, sub_tba, _, sub_coords = _preprocess_pts_rst_subsample(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
        return_coords=True,
    )

    # Convert point clouds to Nx3 arrays for efficient calculations below
    ref_epc = np.vstack((sub_coords[0], sub_coords[1], sub_ref))
    tba_epc = np.vstack((sub_coords[0], sub_coords[1], sub_tba))

    # Remove centroid and standardize to facilitate numerical convergence
    ref_epc, tba_epc, centroid, std_fac = _standardize_epc(ref_epc=ref_epc, tba_epc=tba_epc, scale_std=standardize)
    tolerance /= std_fac

    # Run rigid CPD registration
    # Iterate through method until tolerance or max number of iterations is reached
    init_matrix = np.eye(4)  # Initial matrix is the identity transform
    init_q = np.inf
    init_sigma2 = None
    iterating_input = (init_matrix, init_sigma2, init_q)
    sigma2_min = tolerance / 10
    constant_inputs = (ref_epc, tba_epc, weight_cpd, sigma2_min, only_translation)
    final_matrix, _, _ = _iterate_method(
        method=_cpd_iteration_step,
        iterating_input=iterating_input,
        constant_inputs=constant_inputs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    final_matrix = invert_matrix(final_matrix)

    # De-standardize
    final_matrix[:3, 3] *= std_fac

    # Get subsample size
    subsample_final = len(sub_ref)

    return final_matrix, centroid, subsample_final


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
        gradient_y, gradient_x = np.gradient(ref_elev)

    # If inputs are one raster and one point cloud, derive terrain attribute from raster and get 1D dh interpolator
    else:

        if isinstance(ref_elev, gpd.GeoDataFrame):
            rst_elev = tba_elev
        else:
            rst_elev = ref_elev

        # Derive slope and aspect from the raster dataset
        gradient_y, gradient_x = np.gradient(rst_elev)

    # Convert to unitary gradient depending on resolution
    res = _res(transform)
    gradient_x = gradient_x / res[0]
    gradient_y = -gradient_y / res[1]  # Because raster Y axis is inverted, need to add a minus

    return gradient_x, gradient_y


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


def _lzd_fit(
    x: NDArrayf,
    y: NDArrayf,
    z: NDArrayf,
    dh: NDArrayf,
    gradx: NDArrayf,
    grady: NDArrayf,
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
    **kwargs: Any,
) -> NDArrayf:
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
    results = params_fit_or_bin["fit_minimizer"](fit_func, init_offsets, loss=loss_func, **kwargs)

    # Mypy: having results as "None" is impossible, but not understood through overloading of _bin_or_and_fit_nd...
    assert results is not None
    # Build matrix out of optimized parameters
    matrix = matrix_from_translations_rotations(*results.x, use_degrees=False)  # type: ignore

    return matrix


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
) -> tuple[NDArrayf, float]:
    """
    Iteration step of Least Z-difference coregistration from Rosenholm and Torlegård (1988).

    The function uses 2D array interpolators of the DEM input and its gradient, computed only once outside iteration
    loops, to optimize computing time.

    Returns optimized affine matrix and tolerance for this iteration step.

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

    :return Affine matrix, Tolerance.
    """

    # Apply transform matrix from previous steps
    pts_epc = np.vstack((sub_coords[0], sub_coords[1], sub_pts))
    trans_pts_epc = _apply_matrix_pts_mat(pts_epc, matrix=matrix, centroid=centroid)

    # Evaluate dh and gradients at new X/Y coordinates
    x = trans_pts_epc[0, :]
    y = trans_pts_epc[1, :]
    z = trans_pts_epc[2, :]
    dh = sub_rst((y, x)) - z
    gradx = sub_gradx((y, x))
    grady = sub_grady((y, x))

    # Remove centroid before fit for better convergence
    x -= centroid[0]
    y -= centroid[1]
    z -= centroid[2]

    # Remove invalid values sampled by interpolators
    valids = np.logical_and.reduce((np.isfinite(dh), np.isfinite(z), np.isfinite(gradx), np.isfinite(grady)))
    if np.count_nonzero(valids) == 0:
        raise ValueError(
            "The subsample contains no more valid values. This can happen if the affine transformation to "
            "correct is larger than the data extent, or if the algorithm diverged. To ensure all possible points can "
            "be used at any iteration step, use subsample=1."
        )
    x = x[valids]
    y = y[valids]
    z = z[valids]
    dh = dh[valids]
    gradx = gradx[valids]
    grady = grady[valids]

    # Fit to get new step transform
    step_matrix = _lzd_fit(
        x=x,
        y=y,
        z=z,
        dh=dh,
        gradx=gradx,
        grady=grady,
        params_fit_or_bin=params_fit_or_bin,
        only_translation=only_translation,
    )

    # Increment transformation matrix by step
    new_matrix = step_matrix @ matrix

    # Compute statistic on offset to know if it reached tolerance
    translations = step_matrix[:3, 3]

    tolerance_translation = np.sqrt(np.sum(translations) ** 2)
    # TODO: If we allow multiple tolerances in the future, here's the rotation tolerance
    # rotations = step_matrix[:3, :3]
    # tolerance_rotation = np.rad2deg(np.arccos(np.clip((np.trace(rotations) - 1) / 2, -1, 1)))

    return new_matrix, tolerance_translation


def lzd(
    ref_elev: NDArrayf | gpd.GeoDataFrame,
    tba_elev: NDArrayf | gpd.GeoDataFrame,
    inlier_mask: NDArrayb,
    transform: rio.transform.Affine,
    crs: rio.crs.CRS,
    area_or_point: Literal["Area", "Point"] | None,
    z_name: str,
    max_iterations: int,
    tolerance: float,
    params_random: InRandomDict,
    params_fit_or_bin: InFitOrBinDict,
    only_translation: bool,
) -> tuple[NDArrayf, tuple[float, float, float], int]:
    """
    Least Z-differences coregistration.
    See Rosenholm and Torlegård (1988),
    https://www.asprs.org/wp-content/uploads/pers/1988journal/oct/1988_oct_1385-1389.pdf.

    This function subsamples input data, then runs LZD iteration steps to optimize its fit function until
    convergence or a maximum of iterations is reached.

    The function assumes we have two DEMs, or a DEM and an elevation point cloud, in the same CRS.
    """

    logging.info("Running LZD coregistration")

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
    sub_ref, sub_tba, _, sub_coords = _preprocess_pts_rst_subsample(
        params_random=params_random,
        ref_elev=ref_elev,
        tba_elev=tba_elev,
        inlier_mask=inlier_mask,
        transform=transform,
        crs=crs,
        area_or_point=area_or_point,
        z_name=z_name,
        return_coords=True,
    )
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

    # Estimate centroid to use
    centroid_x = float(np.nanmean(sub_coords[0]))
    centroid_y = float(np.nanmean(sub_coords[1]))
    centroid_z = float(np.nanmean(sub_pts))
    centroid = (centroid_x, centroid_y, centroid_z)

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
    )
    final_matrix = _iterate_method(
        method=_lzd_iteration_step,
        iterating_input=init_matrix,
        constant_inputs=constant_inputs,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    # Invert matrix if reference was the point data
    if ref == "pts":
        final_matrix = invert_matrix(final_matrix)

    subsample_final = len(sub_pts)

    return final_matrix, centroid, subsample_final


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
    ) -> None:
        """Instantiate a generic AffineCoreg method."""

        if meta is None:
            meta = {}
        # Define subsample size
        meta.update({"subsample": subsample})
        super().__init__(meta=meta)

        if matrix is not None:
            with warnings.catch_warnings():
                # This error is fixed in the upcoming 1.8
                warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
                valid_matrix = pytransform3d.transformations.check_transform(matrix)
            self._meta["outputs"]["affine"] = {"matrix": valid_matrix}

        self._is_affine = True

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

        :return: Extrinsinc Euler rotations along easting, northing and vertical directions (degrees).
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
            params_random=params_random,
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
        with warnings.catch_warnings():
            # This error is fixed in the upcoming 1.8
            warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
            valid_matrix = pytransform3d.transformations.check_transform(matrix)
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
        """Estimate the vertical shift using the vshift_func."""

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
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
        """Estimate the vertical shift using the vshift_func."""

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]

        vshift, subsample_final = vertical_shift(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
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

        empty_matrix[2, 3] += self._meta["outputs"]["affine"]["shift_z"]

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
      https://www.cs.unc.edu/techreports/04-004.pdf (faster, only for rotations
      below 30° degrees),
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
        only_translation: bool = False,
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] | Literal["lsq_approx"] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        max_iterations: int = 20,
        tolerance: float = 0.01,
        standardize: bool = True,
        subsample: float | int = 5e5,
    ) -> None:
        """
        Instantiate an ICP coregistration object.

        :param method: Method of iterative closest point registration, either "point-to-point" of Besl and McKay (1992)
            that minimizes 3D distances, or "point-to-plane" of Chen and Medioni (1992) that minimizes 3D distances
            projected on normals.
        :param picky: Whether to use the duplicate removal for pairs of closest points of Zinsser et al. (2003).
        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param fit_minimizer: Minimizer for the coregistration function. Use "lsq_approx" for the linearized
            least-square approximation of Low (2004) (only available for "point-to-plane").
        :param fit_loss_func: Loss function for the minimization of residuals (if minimizer is not "lsq_approx").
        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance: Residual change threshold after which to stop the iterations.
        :param standardize: Whether to standardize input point clouds to the unit sphere for numerical convergence
            (tolerance is also standardized by the same factor).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        meta = {
            "icp_method": method,
            "icp_picky": picky,
            "fit_minimizer": fit_minimizer,
            "fit_loss_func": fit_loss_func,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "only_translation": only_translation,
            "standardize": standardize,
        }
        super().__init__(subsample=subsample, meta=meta)

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
        """Estimate the rigid transform from tba_dem to ref_dem."""

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
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

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        matrix, centroid, subsample_final = icp(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance=self._meta["inputs"]["iterative"]["tolerance"],
            method=self._meta["inputs"]["specific"]["icp_method"],
            picky=self._meta["inputs"]["specific"]["icp_picky"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
            standardize=self._meta["inputs"]["affine"]["standardize"],
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
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}


class CPD(AffineCoreg):
    """
    Coherent Point Drift coregistration, based on Myronenko and Song (2010), https://doi.org/10.1109/TPAMI.2010.46.

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
        max_iterations: int = 100,
        tolerance: float = 0.01,
        standardize: bool = True,
        subsample: int | float = 5e3,
    ):
        """
        Instantiate a CPD coregistration object.

        :param weight: Weight contribution of the uniform distribution to account for outliers, from 0 (inclusive) to
            1 (exclusive).
        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance: Residual change threshold after which to stop the iterations.
        :param standardize: Whether to standardize input point clouds to the unit sphere for numerical convergence
            (tolerance is also standardized by the same factor).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        meta_cpd = {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "cpd_weight": weight,
            "only_translation": only_translation,
            "standardize": standardize,
        }

        super().__init__(subsample=subsample, meta=meta_cpd)  # type: ignore

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
        """Estimate the rigid transform from tba_dem to ref_dem."""

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
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

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]

        # Call method
        matrix, centroid, subsample_final = cpd(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            weight_cpd=self._meta["inputs"]["specific"]["cpd_weight"],
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance=self._meta["inputs"]["iterative"]["tolerance"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
            standardize=self._meta["inputs"]["affine"]["standardize"],
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
        max_iterations: int = 10,
        offset_threshold: float = 0.001,
        bin_before_fit: bool = True,
        fit_optimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.curve_fit,
        bin_sizes: int | dict[str, int | Iterable[float]] = 72,
        bin_statistic: Callable[[NDArrayf], np.floating[Any]] = np.nanmedian,
        subsample: int | float = 5e5,
        vertical_shift: bool = True,
    ) -> None:
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: Maximum allowed iterations before stopping.
        :param offset_threshold: Residual offset threshold after which to stop the iterations (in pixels).
        :param bin_before_fit: Whether to bin data before fitting the coregistration function. For the Nuth and Kääb
            (2011) algorithm, this corresponds to bins of aspect to compute statistics on dh/tan(slope).
        :param fit_optimizer: Optimizer to minimize the coregistration function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        :param vertical_shift: Whether to apply the vertical shift or not (default is True).
        """

        self.vertical_shift = vertical_shift

        # Input checks
        _check_inputs_bin_before_fit(
            bin_before_fit=bin_before_fit, fit_optimizer=fit_optimizer, bin_sizes=bin_sizes, bin_statistic=bin_statistic
        )

        # Define iterative parameters and vertical shift
        meta_input_iterative = {
            "max_iterations": max_iterations,
            "tolerance": offset_threshold,
            "apply_vshift": vertical_shift,
        }

        # Define parameters exactly as in BiasCorr, but with only "fit" or "bin_and_fit" as option, so a bin_before_fit
        # boolean, no bin apply option, and fit_func is predefined
        if not bin_before_fit:
            meta_fit = {"fit_or_bin": "fit", "fit_func": _nuth_kaab_fit_func, "fit_optimizer": fit_optimizer}
            meta_fit.update(meta_input_iterative)
            super().__init__(subsample=subsample, meta=meta_fit)  # type: ignore
        else:
            meta_bin_and_fit = {
                "fit_or_bin": "bin_and_fit",
                "fit_func": _nuth_kaab_fit_func,
                "fit_optimizer": fit_optimizer,
                "bin_sizes": bin_sizes,
                "bin_statistic": bin_statistic,
            }
            meta_bin_and_fit.update(meta_input_iterative)
            super().__init__(subsample=subsample, meta=meta_bin_and_fit)  # type: ignore

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
        """Estimate the x/y/z offset between two DEMs."""

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
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
        """
        Estimate the x/y/z offset between a DEM and points cloud.
        """

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        (easting_offset, northing_offset, vertical_offset), subsample_final = nuth_kaab(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            weights=weights,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance=self._meta["inputs"]["iterative"]["tolerance"],
        )

        # Write output to class
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            shift_x=-easting_offset, shift_y=-northing_offset, shift_z=vertical_offset * self.vertical_shift
        )
        self._meta["outputs"]["affine"] = output_affine
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
        fit_minimizer: Callable[..., tuple[NDArrayf, Any]] = scipy.optimize.least_squares,
        fit_loss_func: Callable[[NDArrayf], np.floating[Any]] | str = "linear",
        max_iterations: int = 200,
        tolerance: float = 0.01,
        subsample: float | int = 5e5,
    ):
        """
         Instantiate an LZD coregistration object.

        :param only_translation: Whether to solve only for a translation, otherwise solves for both translation and
            rotation as default.
        :param fit_minimizer: Minimizer for the coregistration function.
        :param fit_loss_func: Loss function for the minimization of residuals.
        :param max_iterations: Maximum allowed iterations before stopping.
        :param tolerance: Residual change threshold after which to stop the iterations.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        meta = {
            "fit_minimizer": fit_minimizer,
            "fit_loss_func": fit_loss_func,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "only_translation": only_translation,
        }
        super().__init__(subsample=subsample, meta=meta)

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
        """Estimate the rigid transform from tba_dem to ref_dem."""

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
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

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        matrix, centroid, subsample_final = lzd(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name=z_name,
            params_random=params_random,
            params_fit_or_bin=params_fit_or_bin,
            max_iterations=self._meta["inputs"]["iterative"]["max_iterations"],
            tolerance=self._meta["inputs"]["iterative"]["tolerance"],
            only_translation=self._meta["inputs"]["affine"]["only_translation"],
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
        self._meta["outputs"]["random"] = {"subsample_final": subsample_final}


class DhMinimize(AffineCoreg):
    """
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

        meta_fit = {"fit_or_bin": "fit", "fit_minimizer": fit_minimizer, "fit_loss_func": fit_loss_func}
        super().__init__(subsample=subsample, meta=meta_fit)  # type: ignore

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

        # Method is the same for 2D or 1D elevation differences, so we can simply re-direct to fit_rst_pts
        self._fit_rst_pts(
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

        # Get parameters stored in class
        params_random = self._meta["inputs"]["random"]
        params_fit_or_bin = self._meta["inputs"]["fitorbin"]

        # Call method
        (easting_offset, northing_offset, vertical_offset), subsample_final = dh_minimize(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
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

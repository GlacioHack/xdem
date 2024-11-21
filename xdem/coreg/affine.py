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

import xdem.coreg.base

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import geopandas as gpd
import numpy as np
import rasterio as rio
import scipy.optimize
from geoutils.interface.interpolate import _interp_points
from geoutils.raster.georeferencing import _bounds, _coords, _res
from tqdm import trange

from xdem._typing import NDArrayb, NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    InFitOrBinDict,
    InRandomDict,
    OutAffineDict,
    _bin_or_and_fit_nd,
    _get_subsample_mask_pts_rst,
    _preprocess_pts_rst_subsample,
    _reproject_horizontal_shift_samecrs,
)
from xdem.spatialstats import nmad

try:
    import pytransform3d.rotations
    import pytransform3d.transformations

    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False

try:
    from noisyopt import minimizeCompass

    _HAS_NOISYOPT = True
except ImportError:
    _HAS_NOISYOPT = False

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

    :param method: Method that needs to be iterated to derive a transformation. Take argument "inputs" as its input,
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
        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
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
    Iteration step of Nuth and Kääb (2011), passed to the iterate_method function.

    Returns newly incremented coordinate offsets, and new statistic to compare to tolerance to reach.

    :param coords_offsets: Coordinate offsets at this iteration (easting, northing, vertical) in georeferenced unit.
    :param dh_interpolator: Interpolator returning elevation differences at the subsampled points for a certain
        horizontal offset (see _preprocess_pts_rst_subsample_interpolator).
    :param slope_tan: Array of slope tangent.
    :param aspect: Array of aspect.
    :param res: Resolution of DEM.
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

    logging.info("Iteratively estimating horizontal shift:")
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

    elif _HAS_NOISYOPT and params_fit_or_bin["fit_minimizer"] == minimizeCompass:
        kwargs.update({"errorcontrol": False})
        if "deltatol" not in kwargs.keys():
            kwargs.update({"deltatol": 0.004})
        if "feps" not in kwargs.keys():
            kwargs.update({"feps": 10e-5})

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
    sub_ref, sub_tba, _ = _preprocess_pts_rst_subsample(
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
        shift_x = matrix[0, 3]
        shift_y = matrix[1, 3]
        shift_z = matrix[2, 3]

        return shift_x, shift_y, shift_z

    def to_rotations(self) -> tuple[float, float, float]:
        """
        Extract X/Y/Z euler rotations (extrinsic convention) from the affine transformation matrix.

        Warning: This function only works for a rigid transformation (rotation and translation).

        :return: Extrinsinc Euler rotations along easting, northing and vertical directions (degrees).
        """

        matrix = self.to_matrix()
        rots = pytransform3d.rotations.euler_from_matrix(matrix, i=0, j=1, k=2, extrinsic=True, strict_check=True)
        rots = np.rad2deg(np.array(rots))
        return rots[0], rots[1], rots[2]

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
        # Initialize a diagonal matrix
        matrix = np.diag(np.ones(4, dtype=float))
        # Add the three translations (which are in the last column)
        matrix[0, 3] = x_off
        matrix[1, 3] = y_off
        matrix[2, 3] = z_off

        return cls.from_matrix(matrix)

    @classmethod
    def from_rotations(cls, x_rot: float = 0.0, y_rot: float = 0.0, z_rot: float = 0.0) -> AffineCoreg:
        """
        Instantiate a generic Coreg class from a X/Y/Z rotation.

        :param x_rot: The rotation (degrees) to apply around the X (west-east) direction.
        :param y_rot: The rotation (degrees) to apply around the Y (south-north) direction.
        :param z_rot: The rotation (degrees) to apply around the Z (vertical) direction.

        :raises ValueError: If the given rotation contained invalid values.

        :returns: An instantiated generic Coreg class.
        """

        # Initialize a diagonal matrix
        matrix = np.diag(np.ones(4, dtype=float))
        # Convert rotations to radians
        e = np.deg2rad(np.array([x_rot, y_rot, z_rot]))
        # Derive 3x3 rotation matrix, and insert in 4x4 affine matrix
        rot_matrix = pytransform3d.rotations.matrix_from_euler(e, i=0, j=1, k=2, extrinsic=True)
        matrix[0:3, 0:3] = rot_matrix

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
    Iterative closest point registration, based on Besl and McKay (1992), https://doi.org/10.1117/12.57955.

    Estimates a rigid transform (rotation + translation) between two elevation datasets.

    The estimated transform is stored in the `self.meta["outputs"]["affine"]` key "matrix", with rotation centered
    on the coordinates in the key "centroid". The translation parameters are also stored individually in the
    keys "shift_x", "shift_y" and "shift_z" (in georeferenced units for horizontal shifts, and unit of the
    elevation dataset inputs for the vertical shift).

    Requires 'opencv'. See opencv doc for more info:
    https://docs.opencv.org/master/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 0.05,
        rejection_scale: float = 2.5,
        num_levels: int = 6,
        subsample: float | int = 5e5,
    ) -> None:
        """
        Instantiate an ICP coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param tolerance: The residual change threshold after which to stop the iterations.
        :param rejection_scale: The threshold (std * rejection_scale) to consider points as outliers.
        :param num_levels: Number of octree levels to consider. A higher number is faster but may be more inaccurate.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        if not _has_cv2:
            raise ValueError("Optional dependency needed. Install 'opencv'")

        meta = {
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "rejection_scale": rejection_scale,
            "num_levels": num_levels,
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

        if weights is not None:
            warnings.warn("ICP was given weights, but does not support it.")

        resolution = _res(transform)

        # Generate the x and y coordinates for the reference_dem
        x_coords, y_coords = _coords(transform, ref_elev.shape, area_or_point=area_or_point)
        gradient_x, gradient_y = np.gradient(ref_elev)

        normal_east = np.sin(np.arctan(gradient_y / resolution[1])) * -1
        normal_north = np.sin(np.arctan(gradient_x / resolution[0]))
        normal_up = 1 - np.linalg.norm([normal_east, normal_north], axis=0)

        valid_mask = np.logical_and.reduce(
            (inlier_mask, np.isfinite(ref_elev), np.isfinite(normal_east), np.isfinite(normal_north))
        )
        subsample_mask = self._get_subsample_on_valid_mask(valid_mask=valid_mask)

        ref_pts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=x_coords[subsample_mask], y=y_coords[subsample_mask], crs=None),
            data={
                "z": ref_elev[subsample_mask],
                "nx": normal_east[subsample_mask],
                "ny": normal_north[subsample_mask],
                "nz": normal_up[subsample_mask],
            },
        )

        self._fit_rst_pts(
            ref_elev=ref_pts,
            tba_elev=tba_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            area_or_point=area_or_point,
            z_name="z",
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

        # Check which one is reference
        if isinstance(ref_elev, gpd.GeoDataFrame):
            point_elev = ref_elev
            rst_elev = tba_elev
            ref = "point"
        else:
            point_elev = tba_elev
            rst_elev = ref_elev
            ref = "raster"

        # Pre-process point data
        point_elev = point_elev.dropna(how="any", subset=[z_name])

        bounds = _bounds(transform=transform, shape=rst_elev.shape)
        resolution = _res(transform)

        # Generate the x and y coordinates for the TBA DEM
        x_coords, y_coords = _coords(transform, rst_elev.shape, area_or_point=area_or_point)
        centroid = (float(np.mean([bounds.left, bounds.right])), float(np.mean([bounds.bottom, bounds.top])), 0.0)
        # Subtract by the bounding coordinates to avoid float32 rounding errors.
        x_coords -= centroid[0]
        y_coords -= centroid[1]

        gradient_x, gradient_y = np.gradient(rst_elev)

        # This CRS is temporary and doesn't affect the result. It's just needed for Raster instantiation.
        normal_east = np.sin(np.arctan(gradient_y / resolution[1])) * -1
        normal_north = np.sin(np.arctan(gradient_x / resolution[0]))
        normal_up = 1 - np.linalg.norm([normal_east.data, normal_north.data], axis=0)

        valid_mask = ~np.isnan(rst_elev) & ~np.isnan(normal_east.data) & ~np.isnan(normal_north.data)

        points: dict[str, NDArrayf] = {}
        points["raster"] = np.dstack(
            [
                x_coords[valid_mask],
                y_coords[valid_mask],
                rst_elev[valid_mask],
                normal_east[valid_mask],
                normal_north[valid_mask],
                normal_up[valid_mask],
            ]
        ).squeeze()

        # TODO: Should be a way to not duplicate this column and just feed it directly
        point_elev["E"] = point_elev.geometry.x.values
        point_elev["N"] = point_elev.geometry.y.values

        if any(col not in point_elev for col in ["nx", "ny", "nz"]):
            for key, arr in [("nx", normal_east), ("ny", normal_north), ("nz", normal_up)]:
                point_elev[key] = _interp_points(
                    arr,
                    transform=transform,
                    area_or_point=area_or_point,
                    points=(point_elev["E"].values, point_elev["N"].values),
                )

        point_elev["E"] -= centroid[0]
        point_elev["N"] -= centroid[1]

        points["point"] = point_elev[["E", "N", z_name, "nx", "ny", "nz"]].values

        for key in points:
            points[key] = points[key][~np.any(np.isnan(points[key]), axis=1)].astype("float32")
            points[key][:, 0] -= resolution[0] / 2
            points[key][:, 1] -= resolution[1] / 2

        # Extract parameters and pass them to method
        max_it = self._meta["inputs"]["iterative"]["max_iterations"]
        tol = self._meta["inputs"]["iterative"]["tolerance"]
        rej = self._meta["inputs"]["specific"]["rejection_scale"]
        num_lv = self._meta["inputs"]["specific"]["num_levels"]
        icp = cv2.ppf_match_3d_ICP(max_it, tol, rej, num_lv)
        logging.info("Running ICP...")
        try:
            # Use points as reference
            _, residual, matrix = icp.registerModelToScene(points["raster"], points["point"])
        except cv2.error as exception:
            if "(expected: 'n > 0'), where" not in str(exception):
                raise exception

            raise ValueError(
                "Not enough valid points in input data."
                f"'reference_dem' had {points['ref'].size} valid points."
                f"'dem_to_be_aligned' had {points['tba'].size} valid points."
            )

        # If raster was reference, invert the matrix
        if ref == "raster":
            matrix = xdem.coreg.base.invert_matrix(matrix)

        logging.info("ICP finished")

        assert residual < 1000, f"ICP coregistration failed: residual={residual}, threshold: 1000"

        # Save outputs
        # (Mypy does not pass with normal dict, requires "OutAffineDict" here for some reason...)
        output_affine = OutAffineDict(
            centroid=centroid,
            matrix=matrix,
            shift_x=matrix[0, 3],
            shift_y=matrix[1, 3],
            shift_z=matrix[2, 3],
        )
        self._meta["outputs"]["affine"] = output_affine


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
    ) -> None:
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param offset_threshold: The residual offset threshold after which to stop the iterations (in pixels).
        :param bin_before_fit: Whether to bin data before fitting the coregistration function. For the Nuth and Kääb
            (2011) algorithm, this corresponds to bins of aspect to compute statistics on dh/tan(slope).
        :param fit_optimizer: Optimizer to minimize the coregistration function.
        :param bin_sizes: Size (if integer) or edges (if iterable) for binning variables later passed in .fit().
        :param bin_statistic: Statistic of central tendency (e.g., mean) to apply during the binning.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """

        # Input checks
        _check_inputs_bin_before_fit(
            bin_before_fit=bin_before_fit, fit_optimizer=fit_optimizer, bin_sizes=bin_sizes, bin_statistic=bin_statistic
        )

        # Define iterative parameters
        meta_input_iterative = {"max_iterations": max_iterations, "tolerance": offset_threshold}

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
        output_affine = OutAffineDict(shift_x=-easting_offset, shift_y=-northing_offset, shift_z=vertical_offset)
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

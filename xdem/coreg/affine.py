"""Affine coregistration classes."""

from __future__ import annotations

import warnings
from typing import Any, Callable, TypeVar

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import numpy as np
import pandas as pd
import rasterio as rio
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from geoutils.raster import RasterType, get_array_and_mask
from tqdm import trange

from xdem._typing import NDArrayf
from xdem.coreg.base import (
    Coreg,
    CoregDict,
    _get_x_and_y_coords,
    _mask_dataframe_by_dem,
    _residuals_df,
    _transform_to_bounds_and_res,
    deramping,
)
from xdem.spatialstats import nmad

try:
    import pytransform3d.transformations

    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False

try:
    from noisyopt import minimizeCompass

    _has_noisyopt = True
except ImportError:
    _has_noisyopt = False

######################################
# Generic functions for affine methods
######################################


def apply_xy_shift(transform: rio.transform.Affine, dx: float, dy: float) -> rio.transform.Affine:
    """
    Apply horizontal shift to a rasterio Affine transform
    :param transform: The Affine transform of the raster
    :param dx: dx shift value
    :param dy: dy shift value

    Returns: Updated transform
    """
    transform_shifted = rio.transform.Affine(
        transform.a, transform.b, transform.c + dx, transform.d, transform.e, transform.f + dy
    )
    return transform_shifted


######################################
# Functions for affine coregistrations
######################################


def _calculate_slope_and_aspect_nuthkaab(dem: NDArrayf) -> tuple[NDArrayf, NDArrayf]:
    """
    Calculate the tangent of slope and aspect of a DEM, in radians, as needed for the Nuth & Kaab algorithm.

    :param dem: A numpy array of elevation values.

    :returns:  The tangent of slope and aspect (in radians) of the DEM.
    """
    # Old implementation
    # # Calculate the gradient of the slope
    gradient_y, gradient_x = np.gradient(dem)
    slope_tan = np.sqrt(gradient_x**2 + gradient_y**2)
    aspect = np.arctan2(-gradient_x, gradient_y)
    aspect += np.pi

    # xdem implementation
    # slope, aspect = xdem.terrain.get_terrain_attribute(
    #     dem, attribute=["slope", "aspect"], resolution=1, degrees=False
    # )
    # slope_tan = np.tan(slope)
    # aspect = (aspect + np.pi) % (2 * np.pi)

    return slope_tan, aspect


def get_horizontal_shift(
    elevation_difference: NDArrayf, slope: NDArrayf, aspect: NDArrayf, min_count: int = 20
) -> tuple[float, float, float]:
    """
    Calculate the horizontal shift between two DEMs using the method presented in Nuth and Kääb (2011).

    :param elevation_difference: The elevation difference (reference_dem - aligned_dem).
    :param slope: A slope map with the same shape as elevation_difference (units = pixels?).
    :param aspect: An aspect map with the same shape as elevation_difference (units = radians).
    :param min_count: The minimum allowed bin size to consider valid.

    :raises ValueError: If very few finite values exist to analyse.

    :returns: The pixel offsets in easting, northing, and the c_parameter (altitude?).
    """
    input_x_values = aspect

    with np.errstate(divide="ignore", invalid="ignore"):
        input_y_values = elevation_difference / slope

    # Remove non-finite values
    x_values = input_x_values[np.isfinite(input_x_values) & np.isfinite(input_y_values)]
    y_values = input_y_values[np.isfinite(input_x_values) & np.isfinite(input_y_values)]

    assert y_values.shape[0] > 0

    # Remove outliers
    lower_percentile = np.percentile(y_values, 1)
    upper_percentile = np.percentile(y_values, 99)
    valids = np.where((y_values > lower_percentile) & (y_values < upper_percentile) & (np.abs(y_values) < 200))
    x_values = x_values[valids]
    y_values = y_values[valids]

    # Slice the dataset into appropriate aspect bins
    step = np.pi / 36
    slice_bounds = np.arange(start=0, stop=2 * np.pi, step=step)
    y_medians = np.zeros([len(slice_bounds)])
    count = y_medians.copy()
    for i, bound in enumerate(slice_bounds):
        y_slice = y_values[(bound < x_values) & (x_values < (bound + step))]
        if y_slice.shape[0] > 0:
            y_medians[i] = np.median(y_slice)
        count[i] = y_slice.shape[0]

    # Filter out bins with counts below threshold
    y_medians = y_medians[count > min_count]
    slice_bounds = slice_bounds[count > min_count]

    if slice_bounds.shape[0] < 10:
        raise ValueError("Less than 10 different cells exist.")

    # Make an initial guess of the a, b, and c parameters
    initial_guess: tuple[float, float, float] = (3 * np.std(y_medians) / (2**0.5), 0.0, np.mean(y_medians))

    def estimate_ys(x_values: NDArrayf, parameters: tuple[float, float, float]) -> NDArrayf:
        """
        Estimate y-values from x-values and the current parameters.

        y(x) = a * cos(b - x) + c

        :param x_values: The x-values to feed the above function.
        :param parameters: The a, b, and c parameters to feed the above function

        :returns: Estimated y-values with the same shape as the given x-values
        """
        return parameters[0] * np.cos(parameters[1] - x_values) + parameters[2]

    def residuals(parameters: tuple[float, float, float], y_values: NDArrayf, x_values: NDArrayf) -> NDArrayf:
        """
        Get the residuals between the estimated and measured values using the given parameters.

        err(x, y) = est_y(x) - y

        :param parameters: The a, b, and c parameters to use for the estimation.
        :param y_values: The measured y-values.
        :param x_values: The measured x-values

        :returns: An array of residuals with the same shape as the input arrays.
        """
        err = estimate_ys(x_values, parameters) - y_values
        return err

    # Estimate the a, b, and c parameters with least square minimisation
    results = scipy.optimize.least_squares(
        fun=residuals, x0=initial_guess, args=(y_medians, slice_bounds), xtol=1e-8, gtol=None, ftol=None
    )

    # Round results above the tolerance to get fixed results on different OS
    a_parameter, b_parameter, c_parameter = results.x
    a_parameter = np.round(a_parameter, 2)
    b_parameter = np.round(b_parameter, 2)

    # Calculate the easting and northing offsets from the above parameters
    east_offset = a_parameter * np.sin(b_parameter)
    north_offset = a_parameter * np.cos(b_parameter)

    return east_offset, north_offset, c_parameter


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

    def __init__(self, meta: CoregDict | None = None, matrix: NDArrayf | None = None) -> None:
        """Instantiate a generic Coreg method."""

        super().__init__(meta=meta)

        if matrix is not None:
            with warnings.catch_warnings():
                # This error is fixed in the upcoming 1.8
                warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
                valid_matrix = pytransform3d.transformations.check_transform(matrix)
            self._meta["matrix"] = valid_matrix
        self._is_affine = True

    @property
    def is_affine(self) -> bool:
        """Check if the transform be explained by a 3D affine transform."""
        # _is_affine is found by seeing if to_matrix() raises an error.
        # If this hasn't been done yet, it will be None
        if self._is_affine is None:
            try:  # See if to_matrix() raises an error.
                self.to_matrix()
                self._is_affine = True
            except (ValueError, NotImplementedError):
                self._is_affine = False

        return self._is_affine

    def to_matrix(self) -> NDArrayf:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def centroid(self) -> tuple[float, float, float] | None:
        """Get the centroid of the coregistration, if defined."""
        meta_centroid = self._meta.get("centroid")

        if meta_centroid is None:
            return None

        # Unpack the centroid in case it is in an unexpected format (an array, list or something else).
        return meta_centroid[0], meta_centroid[1], meta_centroid[2]

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
    def from_translation(cls, x_off: float = 0.0, y_off: float = 0.0, z_off: float = 0.0) -> AffineCoreg:
        """
        Instantiate a generic Coreg class from a X/Y/Z translation.

        :param x_off: The offset to apply in the X (west-east) direction.
        :param y_off: The offset to apply in the Y (south-north) direction.
        :param z_off: The offset to apply in the Z (vertical) direction.

        :raises ValueError: If the given translation contained invalid values.

        :returns: An instantiated generic Coreg class.
        """
        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] = x_off
        matrix[1, 3] = y_off
        matrix[2, 3] = z_off

        return cls.from_matrix(matrix)

    def _to_matrix_func(self) -> NDArrayf:
        # FOR DEVELOPERS: This function needs to be implemented if the `self._meta['matrix']` keyword is not None.

        # Try to see if a matrix exists.
        meta_matrix = self._meta.get("matrix")
        if meta_matrix is not None:
            assert meta_matrix.shape == (4, 4), f"Invalid _meta matrix shape. Expected: (4, 4), got {meta_matrix.shape}"
            return meta_matrix

        raise NotImplementedError("This should be implemented by subclassing")

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: NDArrayf | None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented.
        raise NotImplementedError("This step has to be implemented by subclassing.")

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        # FOR DEVELOPERS: This function is only needed for non-rigid transforms.
        raise NotImplementedError("This should have been implemented by subclassing")

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        # FOR DEVELOPERS: This function is only needed for non-rigid transforms.
        raise NotImplementedError("This should have been implemented by subclassing")


class VerticalShift(AffineCoreg):
    """
    DEM vertical shift correction.

    Estimates the mean (or median, weighted avg., etc.) vertical offset between two DEMs.
    """

    def __init__(self, vshift_func: Callable[[NDArrayf], np.floating[Any]] = np.average) -> None:  # pylint:
        # disable=super-init-not-called
        """
        Instantiate a vertical shift correction object.

        :param vshift_func: The function to use for calculating the vertical shift. Default: (weighted) average.
        """
        self._meta: CoregDict = {}  # All __init__ functions should instantiate an empty dict.

        super().__init__(meta={"vshift_func": vshift_func})

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: NDArrayf | None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the vertical shift using the vshift_func."""
        if verbose:
            print("Estimating the vertical shift...")
        diff = ref_dem - tba_dem
        diff = diff[np.isfinite(diff)]

        if np.count_nonzero(np.isfinite(diff)) == 0:
            raise ValueError("No finite values in vertical shift comparison.")

        # Use weights if those were provided.
        vshift = (
            self._meta["vshift_func"](diff)
            if weights is None
            else self._meta["vshift_func"](diff, weights)  # type: ignore
        )

        # TODO: We might need to define the type of bias_func with Callback protocols to get the optional argument,
        # TODO: once we have the weights implemented

        if verbose:
            print("Vertical shift estimated")

        self._meta["vshift"] = vshift

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the VerticalShift function to a DEM."""
        return dem + self._meta["vshift"], transform

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        """Apply the VerticalShift function to a set of points."""
        new_coords = coords.copy()
        new_coords[:, 2] += self._meta["vshift"]
        return new_coords

    def _to_matrix_func(self) -> NDArrayf:
        """Convert the vertical shift to a transform matrix."""
        empty_matrix = np.diag(np.ones(4, dtype=float))

        empty_matrix[2, 3] += self._meta["vshift"]

        return empty_matrix


class ICP(AffineCoreg):
    """
    Iterative Closest Point DEM coregistration.
    Based on 3D registration of Besl and McKay (1992), https://doi.org/10.1117/12.57955.

    Estimates a rigid transform (rotation + translation) between two DEMs.

    Requires 'opencv'
    See opencv doc for more info: https://docs.opencv.org/master/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
    """

    def __init__(
        self, max_iterations: int = 100, tolerance: float = 0.05, rejection_scale: float = 2.5, num_levels: int = 6
    ) -> None:
        """
        Instantiate an ICP coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param tolerance: The residual change threshold after which to stop the iterations.
        :param rejection_scale: The threshold (std * rejection_scale) to consider points as outliers.
        :param num_levels: Number of octree levels to consider. A higher number is faster but may be more inaccurate.
        """
        if not _has_cv2:
            raise ValueError("Optional dependency needed. Install 'opencv'")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rejection_scale = rejection_scale
        self.num_levels = num_levels

        super().__init__()

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: NDArrayf | None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the rigid transform from tba_dem to ref_dem."""

        if weights is not None:
            warnings.warn("ICP was given weights, but does not support it.")

        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        points: dict[str, NDArrayf] = {}
        # Generate the x and y coordinates for the reference_dem
        x_coords, y_coords = _get_x_and_y_coords(ref_dem.shape, transform)

        centroid = (np.mean([bounds.left, bounds.right]), np.mean([bounds.bottom, bounds.top]), 0.0)
        # Subtract by the bounding coordinates to avoid float32 rounding errors.
        x_coords -= centroid[0]
        y_coords -= centroid[1]
        for key, dem in zip(["ref", "tba"], [ref_dem, tba_dem]):

            gradient_x, gradient_y = np.gradient(dem)

            normal_east = np.sin(np.arctan(gradient_y / resolution)) * -1
            normal_north = np.sin(np.arctan(gradient_x / resolution))
            normal_up = 1 - np.linalg.norm([normal_east, normal_north], axis=0)

            valid_mask = ~np.isnan(dem) & ~np.isnan(normal_east) & ~np.isnan(normal_north)

            point_cloud = np.dstack(
                [
                    x_coords[valid_mask],
                    y_coords[valid_mask],
                    dem[valid_mask],
                    normal_east[valid_mask],
                    normal_north[valid_mask],
                    normal_up[valid_mask],
                ]
            ).squeeze()

            points[key] = point_cloud[~np.any(np.isnan(point_cloud), axis=1)].astype("float32")

            icp = cv2.ppf_match_3d_ICP(self.max_iterations, self.tolerance, self.rejection_scale, self.num_levels)
        if verbose:
            print("Running ICP...")
        try:
            _, residual, matrix = icp.registerModelToScene(points["tba"], points["ref"])
        except cv2.error as exception:
            if "(expected: 'n > 0'), where" not in str(exception):
                raise exception

            raise ValueError(
                "Not enough valid points in input data."
                f"'reference_dem' had {points['ref'].size} valid points."
                f"'dem_to_be_aligned' had {points['tba'].size} valid points."
            )

        if verbose:
            print("ICP finished")

        assert residual < 1000, f"ICP coregistration failed: residual={residual}, threshold: 1000"

        self._meta["centroid"] = centroid
        self._meta["matrix"] = matrix


class Tilt(AffineCoreg):
    """
    DEM tilting.

    Estimates an 2-D plan correction between the difference of two DEMs.
    """

    def __init__(self, subsample: int | float = 5e5) -> None:
        """
        Instantiate a tilt correction object.

        :param subsample: Factor for subsampling the input raster for speed-up.
            If <= 1, will be considered a fraction of valid pixels to extract.
            If > 1 will be considered the number of pixels to extract.

        """
        self.poly_order = 1
        self.subsample = subsample

        super().__init__()

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: NDArrayf | None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Fit the dDEM between the DEMs to a least squares polynomial equation."""
        ddem = ref_dem - tba_dem
        x_coords, y_coords = _get_x_and_y_coords(ref_dem.shape, transform)
        fit_ramp, coefs = deramping(
            ddem, x_coords, y_coords, degree=self.poly_order, subsample=self.subsample, verbose=verbose
        )

        self._meta["coefficients"] = coefs[0]
        self._meta["func"] = fit_ramp

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the deramp function to a DEM."""
        x_coords, y_coords = _get_x_and_y_coords(dem.shape, transform)

        ramp = self._meta["func"](x_coords, y_coords)

        return dem + ramp, transform

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        """Apply the deramp function to a set of points."""
        new_coords = coords.copy()

        new_coords[:, 2] += self._meta["func"](new_coords[:, 0], new_coords[:, 1])

        return new_coords

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transform matrix if possible."""
        if self.degree > 1:
            raise ValueError(
                "Nonlinear deramping degrees cannot be represented as transformation matrices."
                f" (max 1, given: {self.poly_order})"
            )
        if self.degree == 1:
            raise NotImplementedError("Vertical shift, rotation and horizontal scaling has to be implemented.")

        # If degree==0, it's just a bias correction
        empty_matrix = np.diag(np.ones(4, dtype=float))

        empty_matrix[2, 3] += self._meta["coefficients"][0]

        return empty_matrix


class NuthKaab(AffineCoreg):
    """
    Nuth and Kääb (2011) DEM coregistration.

    Implemented after the paper:
    https://doi.org/10.5194/tc-5-271-2011
    """

    def __init__(self, max_iterations: int = 10, offset_threshold: float = 0.05) -> None:
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param offset_threshold: The residual offset threshold after which to stop the iterations.
        """
        self._meta: CoregDict
        self.max_iterations = max_iterations
        self.offset_threshold = offset_threshold

        super().__init__()

    def _fit_func(
        self,
        ref_dem: NDArrayf,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        weights: NDArrayf | None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the x/y/z offset between two DEMs."""
        if verbose:
            print("Running Nuth and Kääb (2011) coregistration")

        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        # Make a new DEM which will be modified inplace
        aligned_dem = tba_dem.copy()

        # Check that DEM CRS is projected, otherwise slope is not correctly calculated
        if not crs.is_projected:
            raise NotImplementedError(
                f"DEMs CRS is {crs}. NuthKaab coregistration only works with \
projected CRS. First, reproject your DEMs in a local projected CRS, e.g. UTM, and re-run."
            )

        # Calculate slope and aspect maps from the reference DEM
        if verbose:
            print("   Calculate slope and aspect")

        slope_tan, aspect = _calculate_slope_and_aspect_nuthkaab(ref_dem)

        # Make index grids for the east and north dimensions
        east_grid = np.arange(ref_dem.shape[1])
        north_grid = np.arange(ref_dem.shape[0])

        # Make a function to estimate the aligned DEM (used to construct an offset DEM)
        elevation_function = scipy.interpolate.RectBivariateSpline(
            x=north_grid, y=east_grid, z=np.where(np.isnan(aligned_dem), -9999, aligned_dem), kx=1, ky=1
        )

        # Make a function to estimate nodata gaps in the aligned DEM (used to fix the estimated offset DEM)
        # Use spline degree 1, as higher degrees will create instabilities around 1 and mess up the nodata mask
        nodata_function = scipy.interpolate.RectBivariateSpline(
            x=north_grid, y=east_grid, z=np.isnan(aligned_dem), kx=1, ky=1
        )

        # Initialise east and north pixel offset variables (these will be incremented up and down)
        offset_east, offset_north = 0.0, 0.0

        # Calculate initial dDEM statistics
        elevation_difference = ref_dem - aligned_dem

        vshift = np.nanmedian(elevation_difference)
        nmad_old = nmad(elevation_difference)

        if verbose:
            print("   Statistics on initial dh:")
            print(f"      Median = {vshift:.2f} - NMAD = {nmad_old:.2f}")

        # Iteratively run the analysis until the maximum iterations or until the error gets low enough
        if verbose:
            print("   Iteratively estimating horizontal shift:")

        # If verbose is True, will use progressbar and print additional statements
        pbar = trange(self.max_iterations, disable=not verbose, desc="   Progress")
        for i in pbar:

            # Calculate the elevation difference and the residual (NMAD) between them.
            elevation_difference = ref_dem - aligned_dem
            vshift = np.nanmedian(elevation_difference)
            # Correct potential vertical shifts
            elevation_difference -= vshift

            # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
            east_diff, north_diff, _ = get_horizontal_shift(  # type: ignore
                elevation_difference=elevation_difference, slope=slope_tan, aspect=aspect
            )
            if verbose:
                pbar.write(f"      #{i + 1:d} - Offset in pixels : ({east_diff:.2f}, {north_diff:.2f})")

            # Increment the offsets with the overall offset
            offset_east += east_diff
            offset_north += north_diff

            # Calculate new elevations from the offset x- and y-coordinates
            new_elevation = elevation_function(y=east_grid + offset_east, x=north_grid - offset_north)

            # Set NaNs where NaNs were in the original data
            new_nans = nodata_function(y=east_grid + offset_east, x=north_grid - offset_north)
            new_elevation[new_nans > 0] = np.nan

            # Assign the newly calculated elevations to the aligned_dem
            aligned_dem = new_elevation

            # Update statistics
            elevation_difference = ref_dem - aligned_dem

            vshift = np.nanmedian(elevation_difference)
            nmad_new = nmad(elevation_difference)

            nmad_gain = (nmad_new - nmad_old) / nmad_old * 100

            if verbose:
                pbar.write(f"      Median = {vshift:.2f} - NMAD = {nmad_new:.2f}  ==>  Gain = {nmad_gain:.2f}%")

            # Stop if the NMAD is low and a few iterations have been made
            assert ~np.isnan(nmad_new), (offset_east, offset_north)

            offset = np.sqrt(east_diff**2 + north_diff**2)
            if i > 1 and offset < self.offset_threshold:
                if verbose:
                    pbar.write(
                        f"   Last offset was below the residual offset threshold of {self.offset_threshold} -> stopping"
                    )
                break

            nmad_old = nmad_new

        # Print final results
        if verbose:
            print(f"\n   Final offset in pixels (east, north) : ({offset_east:f}, {offset_north:f})")
            print("   Statistics on coregistered dh:")
            print(f"      Median = {vshift:.2f} - NMAD = {nmad_new:.2f}")

        self._meta["offset_east_px"] = offset_east
        self._meta["offset_north_px"] = offset_north
        self._meta["vshift"] = vshift
        self._meta["resolution"] = resolution

    def _fit_pts_func(
        self,
        ref_dem: pd.DataFrame,
        tba_dem: RasterType,
        transform: rio.transform.Affine | None,
        weights: NDArrayf | None,
        verbose: bool = False,
        order: int = 1,
        z_name: str = "z",
    ) -> None:
        """
        Estimate the x/y/z offset between a DEM and points cloud.
        1. deleted elevation_function and nodata_function, shifting dataframe (points) instead of DEM.
        2. do not support latitude and longitude as inputs.

        :param z_name: the column name of dataframe used for elevation differencing

        """

        if verbose:
            print("Running Nuth and Kääb (2011) coregistration. Shift pts instead of shifting dem")

        tba_arr, _ = get_array_and_mask(tba_dem)

        resolution = tba_dem.res[0]

        # Make a new DEM which will be modified inplace
        aligned_dem = tba_dem.copy()

        x_coords, y_coords = (ref_dem["E"].values, ref_dem["N"].values)
        pts = np.array((x_coords, y_coords)).T

        # Calculate slope and aspect maps from the reference DEM
        if verbose:
            print("   Calculate slope and aspect")
        slope, aspect = _calculate_slope_and_aspect_nuthkaab(tba_arr)

        slope_r = tba_dem.copy(new_array=np.ma.masked_array(slope[None, :, :], mask=~np.isfinite(slope[None, :, :])))
        aspect_r = tba_dem.copy(new_array=np.ma.masked_array(aspect[None, :, :], mask=~np.isfinite(aspect[None, :, :])))

        # Initialise east and north pixel offset variables (these will be incremented up and down)
        offset_east, offset_north, vshift = 0.0, 0.0, 0.0

        # Calculate initial DEM statistics
        slope_pts = slope_r.interp_points(pts, mode="nearest")
        aspect_pts = aspect_r.interp_points(pts, mode="nearest")
        tba_pts = aligned_dem.interp_points(pts, mode="nearest")

        # Treat new_pts as a window, every time we shift it a little bit to fit the correct view
        new_pts = pts.copy()

        elevation_difference = ref_dem[z_name].values - tba_pts
        vshift = float(np.nanmedian(elevation_difference))
        nmad_old = nmad(elevation_difference)

        if verbose:
            print("   Statistics on initial dh:")
            print(f"      Median = {vshift:.3f} - NMAD = {nmad_old:.3f}")

        # Iteratively run the analysis until the maximum iterations or until the error gets low enough
        if verbose:
            print("   Iteratively estimating horizontal shit:")

        # If verbose is True, will use progressbar and print additional statements
        pbar = trange(self.max_iterations, disable=not verbose, desc="   Progress")
        for i in pbar:

            # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
            east_diff, north_diff, _ = get_horizontal_shift(  # type: ignore
                elevation_difference=elevation_difference, slope=slope_pts, aspect=aspect_pts
            )
            if verbose:
                pbar.write(f"      #{i + 1:d} - Offset in pixels : ({east_diff:.3f}, {north_diff:.3f})")

            # Increment the offsets with the overall offset
            offset_east += east_diff
            offset_north += north_diff

            # Assign offset to the coordinates of the pts
            # Treat new_pts as a window, every time we shift it a little bit to fit the correct view
            new_pts += [east_diff * resolution, north_diff * resolution]

            # Get new values
            tba_pts = aligned_dem.interp_points(new_pts, mode="nearest")
            elevation_difference = ref_dem[z_name].values - tba_pts

            # Mask out no data by dem's mask
            pts_, mask_ = _mask_dataframe_by_dem(new_pts, tba_dem)

            # Update values relataed to shifted pts
            elevation_difference = elevation_difference[mask_]
            slope_pts = slope_r.interp_points(pts_, mode="nearest")
            aspect_pts = aspect_r.interp_points(pts_, mode="nearest")
            vshift = float(np.nanmedian(elevation_difference))

            # Update statistics
            elevation_difference -= vshift
            nmad_new = nmad(elevation_difference)
            nmad_gain = (nmad_new - nmad_old) / nmad_old * 100

            if verbose:
                pbar.write(f"      Median = {vshift:.3f} - NMAD = {nmad_new:.3f}  ==>  Gain = {nmad_gain:.3f}%")

            # Stop if the NMAD is low and a few iterations have been made
            assert ~np.isnan(nmad_new), (offset_east, offset_north)

            offset = np.sqrt(east_diff**2 + north_diff**2)
            if i > 1 and offset < self.offset_threshold:
                if verbose:
                    pbar.write(
                        f"   Last offset was below the residual offset threshold of {self.offset_threshold} -> stopping"
                    )
                break

            nmad_old = nmad_new

        # Print final results
        if verbose:
            print(
                "\n   Final offset in pixels (east, north, bais) : ({:f}, {:f},{:f})".format(
                    offset_east, offset_north, vshift
                )
            )
            print("   Statistics on coregistered dh:")
            print(f"      Median = {vshift:.3f} - NMAD = {nmad_new:.3f}")

        self._meta["offset_east_px"] = offset_east
        self._meta["offset_north_px"] = offset_north
        self._meta["vshift"] = vshift
        self._meta["resolution"] = resolution
        self._meta["nmad"] = nmad_new

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transformation matrix from the estimated offsets."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += offset_east
        matrix[1, 3] += offset_north
        matrix[2, 3] += self._meta["vshift"]

        return matrix

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the Nuth & Kaab shift to a DEM."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        updated_transform = apply_xy_shift(transform, -offset_east, -offset_north)
        vshift = self._meta["vshift"]
        return dem + vshift, updated_transform

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        """Apply the Nuth & Kaab shift to a set of points."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        new_coords = coords.copy()
        new_coords[:, 0] += offset_east
        new_coords[:, 1] += offset_north
        new_coords[:, 2] += self._meta["vshift"]

        return new_coords


class GradientDescending(AffineCoreg):
    """
    Gradient Descending coregistration by Zhihao
    """

    def __init__(
        self,
        downsampling: int = 6000,
        x0: tuple[float, float] = (0, 0),
        bounds: tuple[float, float] = (-3, 3),
        deltainit: int = 2,
        deltatol: float = 0.004,
        feps: float = 0.0001,
    ) -> None:
        """
        Instantiate gradient descending coregistration object.

        :param downsampling: The number of points of downsampling the df to run the coreg. Set None to disable it.
        :param x0: The initial point of gradient descending iteration.
        :param bounds: The boundary of the maximum shift.
        :param deltainit: Initial pattern size.
        :param deltatol: Target pattern size, or the precision you want achieve.
        :param feps: Parameters for algorithm. Smallest difference in function value to resolve.

        The algorithm terminates when the iteration is locally optimal at the target pattern size 'deltatol',
        or when the function value differs by less than the tolerance 'feps' along all directions.

        """
        self._meta: CoregDict
        self.downsampling = downsampling
        self.bounds = bounds
        self.x0 = x0
        self.deltainit = deltainit
        self.deltatol = deltatol
        self.feps = feps

        super().__init__()

    def _fit_pts_func(
        self,
        ref_dem: pd.DataFrame,
        tba_dem: NDArrayf,
        transform: rio.transform.Affine | None,
        verbose: bool = False,
        order: int | None = 1,
        z_name: str = "z",
        weights: str | None = None,
    ) -> None:
        """Estimate the x/y/z offset between two DEMs.
        :param ref_dem: the dataframe used as ref
        :param tba_dem: the dem to be aligned
        :param z_name: the column name of dataframe used for elevation differencing
        :param weights: the column name of dataframe used for weight, should have the same length with z_name columns
        :param order and transform is no needed but kept temporally for consistency.

        """

        if not _has_noisyopt:
            raise ValueError("Optional dependency needed. Install 'noisyopt'")

        # downsampling if downsampling != None
        if self.downsampling and len(ref_dem) > self.downsampling:
            ref_dem = ref_dem.sample(frac=self.downsampling / len(ref_dem), random_state=42).copy()

        resolution = tba_dem.res[0]

        if verbose:
            print("Running Gradient Descending Coreg - Zhihao (in preparation) ")
            if self.downsampling:
                print("Running on downsampling. The length of the gdf:", len(ref_dem))

            elevation_difference = _residuals_df(tba_dem, ref_dem, (0, 0), 0, z_name=z_name)
            nmad_old = nmad(elevation_difference)
            vshift = np.nanmedian(elevation_difference)
            print("   Statistics on initial dh:")
            print(f"      Median = {vshift:.4f} - NMAD = {nmad_old:.4f}")

        # start iteration, find the best shifting px
        def func_cost(x: tuple[float, float]) -> np.floating[Any]:
            return nmad(_residuals_df(tba_dem, ref_dem, x, 0, z_name=z_name, weight=weights))

        res = minimizeCompass(
            func_cost,
            x0=self.x0,
            deltainit=self.deltainit,
            deltatol=self.deltatol,
            feps=self.feps,
            bounds=(self.bounds, self.bounds),
            disp=verbose,
            errorcontrol=False,
        )

        # Send the best solution to find all results
        elevation_difference = _residuals_df(tba_dem, ref_dem, (res.x[0], res.x[1]), 0, z_name=z_name)

        # results statistics
        vshift = np.nanmedian(elevation_difference)
        nmad_new = nmad(elevation_difference)

        # Print final results
        if verbose:

            print(f"\n   Final offset in pixels (east, north) : ({res.x[0]:f}, {res.x[1]:f})")
            print("   Statistics on coregistered dh:")
            print(f"      Median = {vshift:.4f} - NMAD = {nmad_new:.4f}")

        self._meta["offset_east_px"] = res.x[0]
        self._meta["offset_north_px"] = res.x[1]
        self._meta["vshift"] = vshift
        self._meta["resolution"] = resolution

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transformation matrix from the estimated offsets."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += offset_east
        matrix[1, 3] += offset_north
        matrix[2, 3] += self._meta["vshift"]

        return matrix

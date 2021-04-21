"""
DEM coregistration functions.

"Amaury's" functions are adapted from Amaury Dehecq.
Source: https://github.com/GeoUtils/geoutils/blob/master/geoutils/dem_coregistration.py

Author(s):
    Erik Schytt Holmlund (holmlund@vaw.baug.ethz.ch)

Date: 13 November 2020.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from enum import Enum
from typing import Any, Callable, Optional, Union

import fiona
import geoutils as gu
import numpy as np
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import rasterio.windows  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from rasterio import Affine
from tqdm import trange

import xdem

try:
    import richdem as rd
    _has_rd = True
except ImportError:
    _has_rd = False

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False

try:
    from pytransform3d.transform_manager import TransformManager
    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False


def filter_by_range(ds: rio.DatasetReader, rangelim: tuple[float, float]):
    """
    Function to filter values using a range.
    """
    print('Excluding values outside of range: {0:f} to {1:f}'.format(*rangelim))
    out = np.ma.masked_outside(ds, *rangelim)
    out.set_fill_value(ds.fill_value)
    return out


def filtered_slope(ds_slope, slope_lim=(0.1, 40)):
    print("Slope filter: %0.2f - %0.2f" % slope_lim)
    print("Initial count: %i" % ds_slope.count())
    flt_slope = filter_by_range(ds_slope, slope_lim)
    print(flt_slope.count())
    return flt_slope


def apply_xy_shift(ds: rio.DatasetReader, dx: float, dy: float) -> np.ndarray:
    """
    Apply horizontal shift to rio dataset using Transform affine matrix
    :param ds: DEM
    :param dx: dx shift value
    :param dy: dy shift value

    Returns:
    Rio Dataset with updated transform
    """
    print("X shift: ", dx)
    print("Y shift: ", dy)

    # Update geotransform
    ds_meta = ds.meta
    gt_orig = ds.transform
    gt_align = Affine(gt_orig.a, gt_orig.b, gt_orig.c+dx,
                      gt_orig.d, gt_orig.e, gt_orig.f+dy)

    print("Original transform:", gt_orig)
    print("Updated transform:", gt_align)

    # Update ds Geotransform
    ds_align = ds
    meta_update = ds.meta.copy()
    meta_update({"driver": "GTiff", "height": ds.shape[1],
                 "width": ds.shape[2], "transform": gt_align, "crs": ds.crs})
    # to split this part in two?
    with rasterio.open(ds_align, "w", **meta_update) as dest:
        dest.write(ds_align)

    return ds_align


def apply_z_shift(ds: rio.DatasetReader, dz: float):
    """
    Apply vertical shift to rio dataset using Transform affine matrix
    :param ds: DEM
    :param dx: dz shift value
    """
    src_dem = rio.open(ds)
    a = src_dem.read(1)
    ds_shift = a + dz
    return ds_shift


def rio_to_rda(ds: rio.DatasetReader) -> rd.rdarray:
    """
    Get georeferenced richDEM array from rasterio dataset
    :param ds: DEM
    :return: DEM
    """
    arr = ds.read(1)
    rda = rd.rdarray(arr, no_data=ds.get_nodatavals()[0])
    rda.geotransform = ds.get_transform()
    rda.projection = ds.get_gcps()

    return rda


def get_terrainattr(ds: rio.DatasetReader, attrib='slope_degrees') -> rd.rdarray:
    """
    Derive terrain attribute for DEM opened with rasterio. One of "slope_degrees", "slope_percentage", "aspect",
    "profile_curvature", "planform_curvature", "curvature" and others (see richDEM documentation)
    :param ds: DEM
    :param attrib: terrain attribute
    :return:
    """
    rda = rio_to_rda(ds)
    terrattr = rd.TerrainAttribute(rda, attrib=attrib)

    return terrattr


def get_horizontal_shift(elevation_difference: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                         min_count: int = 20) -> tuple[float, float, float]:
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
    initial_guess: tuple[float, float, float] = (3 * np.std(y_medians) / (2 ** 0.5), 0.0, np.mean(y_medians))

    def estimate_ys(x_values: np.ndarray, parameters: tuple[float, float, float]) -> np.ndarray:
        """
        Estimate y-values from x-values and the current parameters.

        y(x) = a * cos(b - x) + c

        :param x_values: The x-values to feed the above function.
        :param parameters: The a, b, and c parameters to feed the above function

        :returns: Estimated y-values with the same shape as the given x-values
        """
        return parameters[0] * np.cos(parameters[1] - x_values) + parameters[2]

    def residuals(parameters: tuple[float, float, float], y_values: np.ndarray, x_values: np.ndarray):
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
    plsq = scipy.optimize.leastsq(func=residuals, x0=initial_guess, args=(y_medians, slice_bounds), full_output=1)

    a_parameter, b_parameter, c_parameter = plsq[0]

    # Calculate the easting and northing offsets from the above parameters
    east_offset = a_parameter * np.sin(b_parameter)
    north_offset = a_parameter * np.cos(b_parameter)

    return east_offset, north_offset, c_parameter


def calculate_slope_and_aspect(dem: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the slope and aspect of a DEM.

    :param dem: A numpy array of elevation values.

    :returns:  The slope (in pixels??) and aspect (in radians) of the DEM.
    """
    # TODO: Figure out why slope is called slope_px. What unit is it in?
    # TODO: Change accordingly in the get_horizontal_shift docstring.

    # Calculate the gradient of the slope
    gradient_y, gradient_x = np.gradient(dem)

    slope_px = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    aspect = np.arctan(-gradient_x, gradient_y)
    aspect += np.pi

    return slope_px, aspect


def deramping(elevation_difference, x_coordinates: np.ndarray, y_coordinates: np.ndarray,
              degree: int, verbose: bool = False,
              metadata: Optional[dict[str, Any]] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Calculate a deramping function to account for rotational and non-rigid components of the elevation difference.

    :param elevation_difference: The elevation difference array to analyse.
    :param x_coordinates: x-coordinates of the above array (must have the same shape as elevation_difference)
    :param y_coordinates: y-coordinates of the above array (must have the same shape as elevation_difference)
    :param degree: The polynomial degree to estimate the ramp.
    :param verbose: Print the least squares optimization progress.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "deramp".

    :returns: A callable function to estimate the ramp.
    """
    warnings.warn("This function is deprecated in favour of the new Coreg class.", DeprecationWarning)
    # Extract only the finite values of the elevation difference and corresponding coordinates.
    valid_diffs = elevation_difference[np.isfinite(elevation_difference)]
    valid_x_coords = x_coordinates[np.isfinite(elevation_difference)]
    valid_y_coords = y_coordinates[np.isfinite(elevation_difference)]

    # Randomly subsample the values if there are more than 500,000 of them.
    if valid_x_coords.shape[0] > 500_000:
        random_indices = np.random.randint(0, valid_x_coords.shape[0] - 1, 500_000)
        valid_diffs = valid_diffs[random_indices]
        valid_x_coords = valid_x_coords[random_indices]
        valid_y_coords = valid_y_coords[random_indices]

    # Create a function whose residuals will be attempted to minimise
    def estimate_values(x_coordinates: np.ndarray, y_coordinates: np.ndarray,
                        coefficients: np.ndarray, degree: int) -> np.ndarray:
        """
        Estimate values from a 2D-polynomial.

        :param x_coordinates: x-coordinates of the difference array (must have the same shape as elevation_difference).
        :param y_coordinates: y-coordinates of the difference array (must have the same shape as elevation_difference).
        :param coefficients: The coefficients (a, b, c, etc.) of the polynomial.
        :param degree: The degree of the polynomial.

        :raises ValueError: If the length of the coefficients list is not compatible with the degree.

        :returns: The values estimated by the polynomial.
        """
        # Check that the coefficient size is correct.
        coefficient_size = (degree + 1) * (degree + 2) / 2
        if len(coefficients) != coefficient_size:
            raise ValueError()

        # Do Amaury's black magic to estimate the values.
        estimated_values = np.sum([coefficients[k * (k + 1) // 2 + j] * x_coordinates ** (k - j) *
                                   y_coordinates ** j for k in range(degree + 1) for j in range(k + 1)], axis=0)
        return estimated_values  # type: ignore

    # Creat the error function
    def residuals(coefficients: np.ndarray, values: np.ndarray, x_coordinates: np.ndarray,
                  y_coordinates: np.ndarray, degree: int) -> np.ndarray:
        """
        Calculate the difference between the estimated and measured values.

        :param coefficients: Coefficients for the estimation.
        :param values: The measured values.
        :param x_coordinates: The x-coordinates of the values.
        :param y_coordinates: The y-coordinates of the values.
        :param degree: The degree of the polynomial to estimate.

        :returns: An array of residuals.
        """
        error = estimate_values(x_coordinates, y_coordinates, coefficients, degree) - values
        error = error[np.isfinite(error)]

        return error

    # Run a least-squares minimisation to estimate the correct coefficients.
    # TODO: Maybe remove the full_output?
    initial_guess = np.zeros(shape=((degree + 1) * (degree + 2) // 2))
    if verbose:
        print("Deramping...")
    coefficients = scipy.optimize.least_squares(
        fun=residuals,
        x0=initial_guess,
        args=(valid_diffs, valid_x_coords, valid_y_coords, degree),
        verbose=2 if verbose and degree > 1 else 0
    ).x

    # Generate the return-function which can correctly estimate the ramp

    def ramp(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> np.ndarray:
        """
        Get the values of the ramp that corresponds to given coordinates.

        :param x_coordinates: x-coordinates of interest.
        :param y_coordinates: y-coordinates of interest.

        :returns: The estimated ramp offsets.
        """
        return estimate_values(x_coordinates, y_coordinates, coefficients, degree)

    if metadata is not None:
        metadata["deramp"] = {
            "coefficients": coefficients,
            "nmad": xdem.spatial_tools.nmad(residuals(coefficients, valid_diffs, valid_x_coords, valid_y_coords, degree))
        }

    # Return the function which can be used later.
    return ramp


def mask_as_array(reference_raster: gu.georaster.Raster, mask: Union[str, gu.geovector.Vector, gu.georaster.Raster]) -> np.ndarray:
    """
    Convert a given mask into an array.

    :param reference_raster: The raster to use for rasterizing the mask if the mask is a vector.
    :param mask: A valid Vector, Raster or a respective filepath to a mask.

    :raises: ValueError: If the mask path is invalid.
    :raises: TypeError: If the wrong mask type was given.

    :returns: The mask as a squeezed array.
    """
    # Try to load the mask file if it's a filepath
    if isinstance(mask, str):
        # First try to load it as a Vector
        try:
            mask = gu.geovector.Vector(mask)
        # If the format is unsopported, try loading as a Raster
        except fiona.errors.DriverError:
            try:
                mask = gu.georaster.Raster(mask)
            # If that fails, raise an error
            except rio.errors.RasterioIOError:
                raise ValueError(f"Mask path not in a supported Raster or Vector format: {mask}")

    # At this point, the mask variable is either a Raster or a Vector
    # Now, convert the mask into an array by either rasterizing a Vector or by fetching a Raster's data
    if isinstance(mask, gu.geovector.Vector):
        mask_array = mask.create_mask(reference_raster)
    elif isinstance(mask, gu.georaster.Raster):
        # The true value is the maximum value in the raster, unless the maximum value is 0 or False
        true_value = np.nanmax(mask.data) if not np.nanmax(mask.data) in [0, False] else True
        mask_array = (mask.data == true_value).squeeze()
    else:
        raise TypeError(
            f"Mask has invalid type: {type(mask)}. Expected one of: "
            f"{[gu.georaster.Raster, gu.geovector.Vector, str, type(None)]}"
        )

    return mask_array


def _transform_to_bounds_and_res(shape: tuple[int, int],
                                 transform: rio.transform.Affine) -> tuple[rio.coords.BoundingBox, float]:
    """Get the bounding box and (horizontal) resolution from a transform and the shape of a DEM."""
    bounds = rio.coords.BoundingBox(
        *rio.transform.array_bounds(shape[0], shape[1], transform=transform))
    resolution = (bounds.right - bounds.left) / shape[1]

    return bounds, resolution


def _get_x_and_y_coords(shape: tuple[int, int], transform: rio.transform.Affine):
    """Generate center coordinates from a transform and the shape of a DEM."""
    bounds, resolution = _transform_to_bounds_and_res(shape, transform)
    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, num=shape[1]),
        np.linspace(bounds.bottom + resolution / 2, bounds.top - resolution / 2, num=shape[0])[::-1]
    )
    return x_coords, y_coords


class Coreg:
    _meta: Optional[dict[str, Any]] = None  # All __init__ functions should instantiate an empty dict.
    _fit_called = False  # Flag to check if the .fit() method has been called.

    def __init__(self):
        """This function should have been overwritten by subclassing."""
        raise ValueError("Coreg class should not be instantiated directly.")

    def fit(self, reference_dem: Union[np.ndarray, np.ma.masked_array],
            dem_to_be_aligned: Union[np.ndarray, np.ma.masked_array],
            inlier_mask: Optional[np.ndarray] = None,
            transform: Optional[rio.transform.Affine] = None,
            weights: Optional[np.ndarray] = None,
            subsample: Union[float, int] = 1.0,
            verbose: bool = False):
        """
        Estimate the coregistration transform on the given DEMs.

        :param reference_dem: 2D array of elevation values acting reference. 
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param weights: Optional. Per-pixel weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param verbose: Print progress messages to stdout.
        """
        # Make sure that the mask has an expected format.
        if inlier_mask is not None:
            inlier_mask = np.asarray(inlier_mask)
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

        if weights is not None:
            raise NotImplementedError("Weights have not yet been implemented")

        # The reference mask is the union of the nan occurrence and the (potential) ma mask.
        ref_mask = np.isnan(reference_dem) | (reference_dem.mask
                                              if isinstance(reference_dem, np.ma.masked_array) else False)
        # The to-be-aligned mask is the union of the nan occurrence and the (potential) ma mask.
        tba_mask = np.isnan(dem_to_be_aligned) | (dem_to_be_aligned.mask
                                                  if isinstance(dem_to_be_aligned, np.ma.masked_array) else False)

        # The full mask (inliers=True) is the inverse of the above masks and the provided mask.
        full_mask = (~ref_mask & ~tba_mask & (np.asarray(inlier_mask) if inlier_mask is not None else True)).squeeze()

        # If subsample is not equal to one, subsampling should be performed.
        if subsample != 1.0:
            # If subsample is less than one, it is parsed as a fraction (e.g. 0.8 => retain 80% of the values)
            if subsample < 1.0:
                subsample = int(np.count_nonzero(full_mask) * (1 - subsample))

            # Randomly pick N inliers in the full_mask where N=subsample
            random_falses = np.random.choice(np.argwhere(full_mask.flatten()).squeeze(), int(subsample), replace=False)
            # Convert the 1D indices to 2D indices
            cols = (random_falses // full_mask.shape[0]).astype(int)
            rows = random_falses % full_mask.shape[0]
            # Set the N random inliers to be parsed as outliers instead.
            full_mask[rows, cols] = False

        # The arrays to provide the functions will be ndarrays with NaNs for masked out areas.
        ref_dem = np.where(full_mask, np.asarray(reference_dem), np.nan).squeeze()
        tba_dem = np.where(full_mask, np.asarray(dem_to_be_aligned), np.nan).squeeze()

        # Run the associated fitting function
        self._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, transform=transform, weights=weights, verbose=verbose)

        # Flag that the fitting function has been called.
        self._fit_called = True

    def apply(self, dem: Union[np.ndarray, np.ma.masked_array],
              transform: rio.transform.Affine) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Apply the estimated transform to a DEM.

        :param dem: A DEM to apply the transform on.
        :param transform: The transform object of the DEM. TODO: Remove??

        :returns: The transformed DEM.
        """
        if not self._fit_called:
            raise AssertionError(".fit() does not seem to have been called yet")

        # The mask is the union of the nan occurrence and the (potential) ma mask.
        dem_mask = (np.isnan(dem) | (dem.mask if isinstance(dem, np.ma.masked_array) else False)).squeeze()

        # The array to provide the functions will be an ndarray with NaNs for masked out areas.
        dem_array = np.where(~dem_mask, np.asarray(dem), np.nan).squeeze()

        # Run the associated apply function
        applied_dem = self._apply_func(dem_array, transform)

        # Return the array in the same format as it was given (ndarray or masked_array)
        return np.ma.masked_array(applied_dem, mask=dem.mask) if isinstance(dem, np.ma.masked_array) else applied_dem

    def apply_pts(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply the estimated transform to a set of 3D points.

        :param coords: A (N, 3) array of X/Y/Z coordinates.

        :returns: The transformed coordinates.
        """
        if not self._fit_called:
            raise AssertionError(".fit() does not seem to have been called yet")
        assert coords.shape[1] == 3, f"'coords' shape must be (N, 3). Given shape: {coords.shape}"

        return self._apply_pts_func(coords)

    def to_matrix(self) -> np.ndarray:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def __add__(self, other: Coreg) -> CoregPipeline:
        """Return a pipeline consisting of self and the other coreg function."""
        if not isinstance(other, Coreg):
            raise ValueError(f"Incompatible add type: {type(other)}. Expected 'Coreg' subclass")
        return CoregPipeline([self, other])

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        raise NotImplementedError("This should have been implemented by subclassing")

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        raise NotImplementedError("This should have been implemented by subclassing")

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        raise NotImplementedError("This should have been implemented by subclassing")

    def _to_matrix_func(self) -> np.ndarray:
        raise NotImplementedError("This should be implemented by subclassing")


class BiasCorr(Coreg):
    """
    DEM bias correction.

    Estimates the mean (or median, weighted avg., etc.) offset between two DEMs.
    """

    def __init__(self, bias_func=np.average):  # pylint: disable=super-init-not-called
        """
        Instantiate a bias correction object.

        :param bias_func: The function to use for calculating the bias. Default: (weighted) average.
        """
        self._meta: dict[str, Any] = {"bias_func": bias_func}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the bias using the bias_func."""
        if verbose:
            print("Estimating bias...")
        diff = ref_dem - tba_dem
        diff = diff[np.isfinite(diff)]

        # Use weights if those were provided.
        bias = self._meta["bias_func"](diff) if weights is None \
            else self._meta["bias_func"](diff, weights=weights)

        if verbose:
            print("Bias estimated")

        self._meta["bias"] = bias

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the bias to a DEM."""
        return dem + self._meta["bias"]

    def _apply_pts_func(self, coords: np.ndarray):
        """Apply the bias to a given coordinate array."""
        new_coords = coords.copy()
        new_coords[:, 2] += self._apply_func(coords[:, 2], None)  # type: ignore

        return new_coords

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the bias to a transform matrix."""
        empty_matrix = np.diag(np.ones(4, dtype=float))

        empty_matrix[2, 3] += self._meta["bias"]

        return empty_matrix


class ICP(Coreg):
    """
    Iterative Closest Point DEM coregistration.

    Estimates a rigid transform (rotation + translation) between two DEMs.

    Requires 'opencv'
    See opencv docs for more info: https://docs.opencv.org/master/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
    """

    def __init__(self, max_iterations=100, tolerance=0.05, rejection_scale=2.5, num_levels=6):  # pylint: disable=super-init-not-called
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
        self._meta: dict[str, Any] = {}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the rigid transform from tba_dem to ref_dem."""
        if weights is not None:
            warnings.warn("ICP was given weights, but does not support it.")

        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        points: dict[str, np.ndarray] = {}
        # Generate the x and y coordinates for the reference_dem
        x_coords, y_coords = _get_x_and_y_coords(ref_dem.shape, transform)
        # Subtract by the bounding coordinates to avoid float32 rounding errors.
        x_coords -= bounds.left
        y_coords -= bounds.bottom
        for key, dem in zip(["ref", "tba"], [ref_dem, tba_dem]):

            gradient_x, gradient_y = np.gradient(dem)

            normal_east = np.sin(np.arctan(gradient_y / resolution)) * -1
            normal_north = np.sin(np.arctan(gradient_x / resolution))
            normal_up = 1 - np.linalg.norm([normal_east, normal_north], axis=0)

            valid_mask = ~np.isnan(dem) & ~np.isnan(normal_east) & ~np.isnan(normal_north)

            point_cloud = np.dstack([
                x_coords[valid_mask],
                y_coords[valid_mask],
                dem[valid_mask],
                normal_east[valid_mask],
                normal_north[valid_mask],
                normal_up[valid_mask]
            ]).squeeze()

            points[key] = point_cloud[~np.any(np.isnan(point_cloud), axis=1)].astype("float32")

        icp = cv2.ppf_match_3d_ICP(self.max_iterations, self.tolerance, self.rejection_scale, self.num_levels)
        if verbose:
            print("Running ICP...")
        _, residual, matrix = icp.registerModelToScene(points["tba"], points["ref"])
        if verbose:
            print("ICP finished")

        assert residual < 1000, f"ICP coregistration failed: residual={residual}, threshold: 1000"

        self._meta["matrix"] = matrix

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the coregistration matrix to a DEM."""
        bounds, resolution = _transform_to_bounds_and_res(dem.shape, transform)
        x_coords, y_coords = _get_x_and_y_coords(dem.shape, transform)
        x_coords -= bounds.left
        y_coords -= bounds.bottom

        valid_mask = np.isfinite(dem)
        transformed_points = self._apply_pts_func(np.dstack([
            x_coords[valid_mask],
            y_coords[valid_mask],
            dem[valid_mask]
        ]).squeeze())

        aligned_dem = scipy.interpolate.griddata(
            points=transformed_points[:, :2],
            values=transformed_points[:, 2],
            xi=tuple(np.meshgrid(
                np.linspace(bounds.left, bounds.right, dem.shape[1]) - bounds.left,
                np.linspace(bounds.bottom, bounds.top, dem.shape[0])[::-1] - bounds.bottom
            )),
            method="cubic"
        )
        aligned_dem[~valid_mask] = np.nan

        return aligned_dem

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the coregistration matrix to a set of points."""
        transformed_points = cv2.perspectiveTransform(coords.reshape(1, -1, 3), self.to_matrix()).squeeze()
        return transformed_points

    def _to_matrix_func(self) -> np.ndarray:
        """Return the coregistration matrix."""
        return self._meta["matrix"]


class Deramp(Coreg):
    """
    Polynomial DEM deramping.

    Estimates an n-D polynomial between the difference of two DEMs.
    """

    def __init__(self, degree: int = 1):
        """
        Instantiate a deramping correction object.

        :param degree: The polynomial degree to estimate. degree=0 is a simple bias correction.
        """
        self.degree = degree

        self._meta: dict[str, Any] = {}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Fit the dDEM between the DEMs to a least squares polynomial equation."""
        x_coords, y_coords = _get_x_and_y_coords(ref_dem.shape, transform)

        ddem = ref_dem - tba_dem
        valid_mask = np.isfinite(ddem)
        ddem = ddem[valid_mask]
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Formulate the 2D polynomial whose coefficients will be solved for.
        def poly2d(x_coordinates: np.ndarray, y_coordinates: np.ndarray,
                   coefficients: np.ndarray) -> np.ndarray:
            """
            Estimate values from a 2D-polynomial.

            :param x_coordinates: x-coordinates of the difference array (must have the same shape as elevation_difference).
            :param y_coordinates: y-coordinates of the difference array (must have the same shape as elevation_difference).
            :param coefficients: The coefficients (a, b, c, etc.) of the polynomial.
            :param degree: The degree of the polynomial.

            :raises ValueError: If the length of the coefficients list is not compatible with the degree.

            :returns: The values estimated by the polynomial.
            """
            # Check that the coefficient size is correct.
            coefficient_size = (self.degree + 1) * (self.degree + 2) / 2
            if len(coefficients) != coefficient_size:
                raise ValueError()

            # Do Amaury's black magic to formulate and calculate the polynomial equation.
            estimated_values = np.sum([coefficients[k * (k + 1) // 2 + j] * x_coordinates ** (k - j) *
                                       y_coordinates ** j for k in range(self.degree + 1) for j in range(k + 1)], axis=0)
            return estimated_values  # type: ignore

        def residuals(coefs: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, targets: np.ndarray):
            return np.median(np.abs(targets - poly2d(x_coords, y_coords, coefs)))

        if verbose:
            print("Estimating deramp function...")
        coefs = scipy.optimize.fmin(
            func=residuals,
            x0=np.zeros(shape=((self.degree + 1) * (self.degree + 2) // 2)),
            args=(x_coords, y_coords, ddem),
            disp=verbose
        )

        self._meta["coefficients"] = coefs
        self._meta["func"] = lambda x, y: poly2d(x, y, coefs)

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the deramp function to a DEM."""
        x_coords, y_coords = _get_x_and_y_coords(dem.shape, transform)

        ramp = self._meta["func"](x_coords, y_coords)

        return dem + ramp

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the deramp function to a set of points."""
        new_coords = coords.copy()

        new_coords[:, 2] += self._meta["func"](new_coords[:, 0], new_coords[:, 1])

        return new_coords

    def _to_matrix_func(self) -> np.ndarray:
        """Return a transform matrix if possible."""
        if self.degree > 1:
            raise ValueError(
                "Nonlinear deramping degrees cannot be represented as transformation matrices."
                f" (max 1, given: {self.degree})")
        if self.degree == 1:
            raise NotImplementedError("Vertical shift, rotation and horizontal scaling has to be implemented.")

        # If degree==0, it's just a bias correction
        empty_matrix = np.diag(np.ones(4, dtype=float))

        empty_matrix[2, 3] += self._meta["coefficients"][0]

        return empty_matrix


class CoregPipeline(Coreg):
    """
    A sequential set of coregistration steps.
    """

    def __init__(self, pipeline: list[Coreg]):  # pylint: disable=super-init-not-called
        """
        Instantiate a new coregistration pipeline.

        :param: Coregistration steps to run in the sequence they are given.
        """
        self.pipeline = pipeline
        self._meta = {}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Fit each coregistration step with the previously transformed DEM."""
        tba_dem_mod = tba_dem.copy()

        for i, coreg in enumerate(self.pipeline):
            if verbose:
                print(f"Running pipeline step: {i + 1} / {len(self.pipeline)}")
            coreg._fit_func(ref_dem, tba_dem_mod, transform=transform, weights=weights, verbose=verbose)
            coreg._fit_called = True

            tba_dem_mod = coreg._apply_func(tba_dem_mod, transform)

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the coregistration steps sequentially to a DEM."""
        dem_mod = dem.copy()

        for coreg in self.pipeline:
            dem_mod = coreg.apply(dem_mod, transform)

        return dem_mod

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the coregistration steps sequentially to a set of points."""
        coords_mod = coords.copy()

        for coreg in self.pipeline:
            coords_mod = coreg._apply_pts_func(coords_mod)

        return coords_mod

    def _to_matrix_func(self) -> np.ndarray:
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

    def __iter__(self):
        """Iterate over the pipeline steps."""
        for coreg in self.pipeline:
            yield coreg

    def __add__(self, other: Union[list[Coreg], Coreg, CoregPipeline]) -> CoregPipeline:
        """Append Coreg(s) or a CoregPipeline to the pipeline."""
        if not isinstance(other, Coreg):
            other = list(other)
        else:
            other = [other]

        pipelines = self.pipeline + other

        return CoregPipeline(pipelines)


class NuthKaab(Coreg):
    """
    Nuth and Kääb (2011) DEM coregistration.

    Implemented after the paper:
    https://doi.org/10.5194/tc-5-271-2011
    """

    def __init__(self, max_iterations: int = 50, error_threshold: float = 0.05):  # pylint: disable=super-init-not-called
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param error_threshold: The residual change threshold after which to stop the iterations.
        """
        self.max_iterations = max_iterations
        self.error_threshold = error_threshold

        self._meta: dict[str, Any] = {}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the x/y/z offset between two DEMs."""
        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        # Make a new DEM which will be modified inplace
        aligned_dem = tba_dem.copy()

        # Calculate slope and aspect maps from the reference DEM
        slope, aspect = calculate_slope_and_aspect(ref_dem)

        # Make index grids for the east and north dimensions
        east_grid = np.arange(ref_dem.shape[1])
        north_grid = np.arange(ref_dem.shape[0])

        # Make a function to estimate the aligned DEM (used to construct an offset DEM)
        elevation_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid,
                                                                   z=np.where(np.isnan(aligned_dem), -9999, aligned_dem))
        # Make a function to estimate nodata gaps in the aligned DEM (used to fix the estimated offset DEM)
        nodata_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=np.isnan(aligned_dem))
        # Initialise east and north pixel offset variables (these will be incremented up and down)
        offset_east, offset_north, bias = 0.0, 0.0, 0.0

        if verbose:
            print("Running Nuth and Kääb (2011) coregistration")
        # Iteratively run the analysis until the maximum iterations or until the error gets low enough
        for i in trange(self.max_iterations, disable=not verbose, desc="Iteratively correcting dataset"):

            # Calculate the elevation difference and the residual (NMAD) between them.
            elevation_difference = ref_dem - aligned_dem
            bias = np.nanmedian(elevation_difference)
            # Correct potential biases
            elevation_difference -= bias

            nmad = xdem.spatial_tools.nmad(elevation_difference)

            assert ~np.isnan(nmad), (offset_east, offset_north)

            # Stop if the NMAD is low and a few iterations have been made
            if i > 5 and nmad < self.error_threshold:
                if verbose:
                    print(f"NMAD went below the error threshold of {self.error_threshold}")
                break

            # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
            east_diff, north_diff, _ = get_horizontal_shift(  # type: ignore
                elevation_difference=elevation_difference,
                slope=slope,
                aspect=aspect
            )
            # Increment the offsets with the overall offset
            offset_east += east_diff
            offset_north += north_diff

            # Calculate new elevations from the offset x- and y-coordinates
            new_elevation = elevation_function(y=east_grid + offset_east, x=north_grid - offset_north)

            # Set NaNs where NaNs were in the original data
            new_nans = nodata_function(y=east_grid + offset_east, x=north_grid - offset_north)
            new_elevation[new_nans >= 1] = np.nan

            # Assign the newly calculated elevations to the aligned_dem
            aligned_dem = new_elevation

        self._meta["offset_east_px"] = offset_east
        self._meta["offset_north_px"] = offset_north
        self._meta["bias"] = bias
        self._meta["resolution"] = resolution

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the estimated x/y/z offsets to a DEM."""
        bounds, resolution = _transform_to_bounds_and_res(dem.shape, transform)
        scaling_factor = self._meta["resolution"] / resolution

        # Make index grids for the east and north dimensions
        east_grid = np.arange(dem.shape[1]) * scaling_factor
        north_grid = np.arange(dem.shape[0]) * scaling_factor

        # Make a function to estimate the DEM (used to construct an offset DEM)
        elevation_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid,
                                                                   z=np.where(np.isnan(dem), -9999, dem))
        # Make a function to estimate nodata gaps in the aligned DEM (used to fix the estimated offset DEM)
        nodata_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=np.isnan(dem))

        shifted_east_grid = east_grid + self._meta["offset_east_px"]
        shifted_north_grid = north_grid - self._meta["offset_north_px"]

        shifted_dem = elevation_function(y=shifted_east_grid, x=shifted_north_grid)
        new_nans = nodata_function(y=shifted_east_grid, x=shifted_north_grid)
        shifted_dem[new_nans >= 1] = np.nan

        shifted_dem += self._meta["bias"]

        return shifted_dem

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the estimated x/y/z offsets to a set of points."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        new_coords = coords.copy()
        new_coords[:, 0] -= offset_east
        new_coords[:, 1] -= offset_north
        new_coords[:, 2] += self._meta["bias"]

        return new_coords

    def _to_matrix_func(self) -> np.ndarray:
        """Return a transformation matrix from the estimated offsets."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += offset_east
        matrix[1, 3] += offset_north
        matrix[2, 3] += self._meta["bias"]

        return matrix


class ZScaleCorr(Coreg):
    """
    Correct linear or nonlinear elevation scale errors.

    Often useful for nadir image DEM correction, where the focal length is slightly miscalculated.

    DISCLAIMER: This function may introduce error when correcting non-photogrammetric biases.
    See Gardelle et al. (2012) (Figure 2), http://dx.doi.org/10.3189/2012jog11j175, for curvature-related biases.
    """

    def __init__(self, degree=1, bin_count=100):
        """
        Instantiate a elevation scale correction object.

        :param degree: The polynomial degree to estimate.
        :param bin_count: The amount of bins to divide the elevation change in.
        """
        self.degree = degree
        self.bin_count = bin_count
        self._meta: dict[str, Any] = {}

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the scale difference between the two DEMs."""
        ddem = ref_dem - tba_dem

        medians = xdem.volume.hypsometric_binning(
            ddem=ddem,
            ref_dem=tba_dem,
            bins=self.bin_count,
            kind="count"
        )["value"]

        coefficients = np.polyfit(medians.index.mid, medians.values, deg=self.degree)
        self._meta["coefficients"] = coefficients

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine) -> np.ndarray:
        """Apply the scaling model to a DEM."""
        model = np.poly1d(self._meta["coefficients"])

        return dem + model(dem)

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the scaling model to a set of points."""
        model = np.poly1d(self._meta["coefficients"])

        new_coords = coords.copy()
        new_coords[:, 2] += model(new_coords[:, 2])
        return new_coords

    def _to_matrix_func(self) -> np.ndarray:
        """Convert the transform to a matrix, if possible."""
        if self.degree < 2:
            raise NotImplementedError

        raise ValueError("Model cannot be described as a rigid transformation matrix.")

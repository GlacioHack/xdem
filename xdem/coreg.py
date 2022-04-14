"""DEM coregistration classes and functions."""
from __future__ import annotations

import copy
import concurrent.futures
import json
import os
import subprocess
import tempfile
import warnings
from enum import Enum
from typing import Any, Callable, Optional, overload, Union, Sequence, TypeVar

try:
    import cv2
    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import fiona
import geoutils as gu
from geoutils.georaster import RasterType
from geoutils import spatial_tools
import numpy as np
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import rasterio.windows  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import skimage.transform
from rasterio import Affine
from tqdm import trange, tqdm
import pandas as pd

import xdem

try:
    from pytransform3d.transform_manager import TransformManager
    import pytransform3d.transformations
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
    aspect = np.arctan2(-gradient_x, gradient_y)
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
    #warnings.warn("This function is deprecated in favour of the new Coreg class.", DeprecationWarning)
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
            "nmad": xdem.spatialstats.nmad(residuals(coefficients, valid_diffs, valid_x_coords, valid_y_coords, degree))
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


def _transform_to_bounds_and_res(shape: tuple[int, ...],
                                 transform: rio.transform.Affine) -> tuple[rio.coords.BoundingBox, float]:
    """Get the bounding box and (horizontal) resolution from a transform and the shape of a DEM."""
    bounds = rio.coords.BoundingBox(
        *rio.transform.array_bounds(shape[0], shape[1], transform=transform))
    resolution = (bounds.right - bounds.left) / shape[1]

    return bounds, resolution


def _get_x_and_y_coords(shape: tuple[int, ...], transform: rio.transform.Affine):
    """Generate center coordinates from a transform and the shape of a DEM."""
    bounds, resolution = _transform_to_bounds_and_res(shape, transform)
    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, num=shape[1]),
        np.linspace(bounds.bottom + resolution / 2, bounds.top - resolution / 2, num=shape[0])[::-1]
    )
    return x_coords, y_coords


CoregType = TypeVar("CoregType", bound="Coreg")

class Coreg:
    """
    Generic Coreg class.

    Made to be subclassed.
    """

    _fit_called: bool = False  # Flag to check if the .fit() method has been called.
    _is_affine: Optional[bool] = None

    def __init__(self, meta: Optional[dict[str, Any]] = None, matrix: Optional[np.ndarray] = None):
        """Instantiate a generic Coreg method."""
        self._meta: dict[str, Any] = meta or {}  # All __init__ functions should instantiate an empty dict.

        if matrix is not None:
            with warnings.catch_warnings():
                # This error is fixed in the upcoming 1.8
                warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
                valid_matrix = pytransform3d.transformations.check_transform(matrix)
            self._meta["matrix"] = valid_matrix

    def fit(self: CoregType, reference_dem: np.ndarray | np.ma.masked_array | RasterType,
            dem_to_be_aligned: np.ndarray | np.ma.masked_array | RasterType,
            inlier_mask: Optional[np.ndarray] = None,
            transform: Optional[rio.transform.Affine] = None,
            weights: Optional[np.ndarray] = None,
            subsample: Union[float, int] = 1.0,
            verbose: bool = False) -> CoregType:
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

        if weights is not None:
            raise NotImplementedError("Weights have not yet been implemented")

        # Validate that both inputs are valid array-like (or Raster) types.
        if not all(hasattr(dem, "__array_interface__") for dem in (reference_dem, dem_to_be_aligned)):
            raise ValueError(
                "Both DEMs need to be array-like (implement a numpy array interface)."
                f"'reference_dem': {reference_dem}, 'dem_to_be_aligned': {dem_to_be_aligned}"
            )

        # If both DEMs are Rasters, validate that 'dem_to_be_aligned' is in the right grid. Then extract its data.
        if isinstance(dem_to_be_aligned, gu.Raster) and isinstance(reference_dem, gu.Raster):
            dem_to_be_aligned = dem_to_be_aligned.reproject(reference_dem, silent=True).data

        # If any input is a Raster, use its transform if 'transform is None'.
        # If 'transform' was given and any input is a Raster, trigger a warning.
        # Finally, extract only the data of the raster.
        for name, dem in [("reference_dem", reference_dem), ("dem_to_be_aligned", dem_to_be_aligned)]:
            if hasattr(dem, "transform"):
                if transform is None:
                    transform = getattr(dem, "transform")
                elif transform is not None:
                    warnings.warn(f"'{name}' of type {type(dem)} overrides the given 'transform'")

                """
                if name == "reference_dem":
                    reference_dem = dem.data
                else:
                    dem_to_be_aligned = dem.data
                """

        if transform is None:
            raise ValueError("'transform' must be given if both DEMs are array-like.")

        ref_dem, ref_mask = spatial_tools.get_array_and_mask(reference_dem)
        tba_dem, tba_mask = spatial_tools.get_array_and_mask(dem_to_be_aligned)

        # Make sure that the mask has an expected format.
        if inlier_mask is not None:
            inlier_mask = np.asarray(inlier_mask).squeeze()
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

            if np.all(~inlier_mask):
                raise ValueError("'inlier_mask' had no inliers.")

            ref_dem[~inlier_mask] = np.nan
            tba_dem[~inlier_mask] = np.nan

        if np.all(ref_mask):
            raise ValueError("'reference_dem' had only NaNs")
        if np.all(tba_mask):
            raise ValueError("'dem_to_be_aligned' had only NaNs")

        # If subsample is not equal to one, subsampling should be performed.
        if subsample != 1.0:
            # The full mask (inliers=True) is the inverse of the above masks and the provided mask.
            full_mask = (~ref_mask & ~tba_mask & (np.asarray(inlier_mask) if inlier_mask is not None else True)).squeeze()
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


        # Run the associated fitting function
        self._fit_func(ref_dem=ref_dem, tba_dem=tba_dem, transform=transform, weights=weights, verbose=verbose)

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(self, dem: RasterType, transform: rio.transform.Affine | None, **kwargs) -> RasterType: ...

    @overload
    def apply(self, dem: np.ndarray, transform: rio.transform.Affine | None, **kwargs) -> np.ndarray: ...

    @overload
    def apply(self, dem: np.ma.masked_array, transform: rio.transform.Affine | None, **kwargs) -> np.ma.masked_array: ...


    def apply(self, dem: np.ndarray | np.ma.masked_array | RasterType,
              transform: rio.transform.Affine | None = None, **kwargs) -> RasterType | np.ndarray | np.ma.masked_array:
        """
        Apply the estimated transform to a DEM.

        :param dem: A DEM array or Raster to apply the transform on.
        :param transform: The transform object of the DEM. Required if 'dem' is an array and not a Raster.
        :param kwargs: Any optional arguments to be passed to either self._apply_func or apply_matrix.

        :returns: The transformed DEM.
        """
        if not self._fit_called and self._meta.get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        if isinstance(dem, gu.Raster):
            if transform is None:
                transform = dem.transform
            else:
                warnings.warn(f"DEM of type {type(dem)} overrides the given 'transform'")
        else:
            if transform is None:
                raise ValueError("'transform' must be given if DEM is array-like.")

        # The array to provide the functions will be an ndarray with NaNs for masked out areas.
        dem_array, dem_mask = spatial_tools.get_array_and_mask(dem)

        if np.all(dem_mask):
            raise ValueError("'dem' had only NaNs")

        # See if a _apply_func exists
        try:
            # Run the associated apply function
            applied_dem = self._apply_func(dem_array, transform, **kwargs)  # pylint: disable=assignment-from-no-return
        # If it doesn't exist, use apply_matrix()
        except NotImplementedError:
            if self.is_affine:  # This only works on it's affine, however.
                # If dilate_mask is not specified, set it to True by default
                if "dilate_mask" in kwargs.keys():
                    dilate_mask = kwargs["dilate_mask"]
                    kwargs.pop("dilate_mask")
                else:
                    dilate_mask = True

                # Apply the matrix around the centroid (if defined, otherwise just from the center).
                applied_dem = apply_matrix(
                    dem_array,
                    transform=transform,
                    matrix=self.to_matrix(),
                    centroid=self._meta.get("centroid"),
                    dilate_mask=dilate_mask,
                    **kwargs
                )
            else:
                raise ValueError("Coreg method is non-rigid but has no implemented _apply_func")

        # Calculate final mask
        final_mask = dem_mask + np.isnan(applied_dem)

        # If the DEM was a masked_array, copy the mask to the new DEM
        if hasattr(dem, "mask"):
            applied_dem = np.ma.masked_array(applied_dem, mask=final_mask)  # type: ignore
        # If the DEM was a Raster with a mask, copy the mask to the new DEM
        elif hasattr(dem, "data") and hasattr(dem.data, "mask"):
            applied_dem = np.ma.masked_array(applied_dem, mask=final_mask)  # type: ignore
        else:
            applied_dem[final_mask] = np.nan

        # If the input was a Raster, return a Raster as well.
        if isinstance(dem, gu.Raster):
            return dem.from_array(applied_dem, transform, dem.crs, nodata=dem.nodata)

        return applied_dem

    def apply_pts(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply the estimated transform to a set of 3D points.

        :param coords: A (N, 3) array of X/Y/Z coordinates or one coordinate of shape (3,).

        :returns: The transformed coordinates.
        """
        if not self._fit_called and self._meta.get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")
        # If the coordinates represent just one coordinate
        if np.shape(coords) == (3,):
            coords = np.reshape(coords, (1, 3))

        assert len(np.shape(coords)) == 2 and np.shape(coords)[1] == 3, f"'coords' shape must be (N, 3). Given shape: {np.shape(coords)}"

        coords_c = coords.copy()

        # See if an _apply_pts_func exists
        try:
            transformed_points = self._apply_pts_func(coords)
        # If it doesn't exist, use opencv's perspectiveTransform
        except NotImplementedError:
            if self.is_affine:  # This only works on it's rigid, however.
                # Transform the points (around the centroid if it exists).
                if self._meta.get("centroid") is not None:
                    coords_c -= self._meta["centroid"]
                transformed_points = cv2.perspectiveTransform(coords_c.reshape(1, -1, 3), self.to_matrix()).squeeze()
                if self._meta.get("centroid") is not None:
                    transformed_points += self._meta["centroid"]

            else:
                raise ValueError("Coreg method is non-rigid but has not implemented _apply_pts_func")

        return transformed_points

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

    def to_matrix(self) -> np.ndarray:
        """Convert the transform to a 4x4 transformation matrix."""
        return self._to_matrix_func()

    def centroid(self) -> Optional[tuple[float, float, float]]:
        """Get the centroid of the coregistration, if defined."""
        meta_centroid = self._meta.get("centroid")

        if meta_centroid is None:
            return None

        # Unpack the centroid in case it is in an unexpected format (an array, list or something else).
        return (meta_centroid[0], meta_centroid[1], meta_centroid[2])

    def residuals(self, reference_dem: Union[np.ndarray, np.ma.masked_array],
                  dem_to_be_aligned: Union[np.ndarray, np.ma.masked_array],
                  inlier_mask: Optional[np.ndarray] = None,
                  transform: Optional[rio.transform.Affine] = None) -> np.ndarray:
        """
        Calculate the residual offsets (the difference) between two DEMs after applying the transformation.

        :param reference_dem: 2D array of elevation values acting reference. 
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.

        :returns: A 1D array of finite residuals.
        """
        # Use the transform to correct the DEM to be aligned.
        aligned_dem = self.apply(dem_to_be_aligned, transform=transform)

        # Format the reference DEM
        ref_arr, ref_mask = spatial_tools.get_array_and_mask(reference_dem)

        if inlier_mask is None:
            inlier_mask = np.ones(ref_arr.shape, dtype=bool)

        # Create the full inlier mask (manual inliers plus non-nans)
        full_mask = (~ref_mask) & np.isfinite(aligned_dem) & inlier_mask

        # Calculate the DEM difference
        diff = ref_arr - aligned_dem

        # Sometimes, the float minimum (for float32 = -3.4028235e+38) is returned. This and inf should be excluded.
        if "float" in str(diff.dtype):
            full_mask[(diff == np.finfo(diff.dtype).min) | np.isinf(diff)] = False

        # Return the difference values within the full inlier mask
        return diff[full_mask]

    def error(self, reference_dem: Union[np.ndarray, np.ma.masked_array],
              dem_to_be_aligned: Union[np.ndarray, np.ma.masked_array],
              error_type: str | list[str] = "nmad",
              inlier_mask: Optional[np.ndarray] = None,
              transform: Optional[rio.transform.Affine] = None) -> float | list[float]:
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

        :param reference_dem: 2D array of elevation values acting reference. 
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param error_type: The type of error meaure to calculate. May be a list of error types.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.

        :returns: The error measure of choice for the residuals.
        """
        if isinstance(error_type, str):
            error_type = [error_type]

        residuals = self.residuals(reference_dem=reference_dem, dem_to_be_aligned=dem_to_be_aligned,
                                   inlier_mask=inlier_mask, transform=transform)

        error_functions = {
            "nmad": xdem.spatialstats.nmad,
            "median": np.median,
            "mean": np.mean,
            "std": np.std,
            "rms": lambda residuals: np.sqrt(np.mean(np.square(residuals))),
            "mae": lambda residuals: np.mean(np.abs(residuals)),
            "count": lambda residuals: residuals.size
        }

        try:
            errors = [error_functions[err_type](residuals) for err_type in error_type]
        except KeyError as exception:
            raise ValueError(
                    f"Invalid 'error_type'{'s' if len(error_type) > 1 else ''}: "
                    f"'{error_type}'. Choices: {list(error_functions.keys())}"
                    ) from exception

        return errors if len(errors) > 1 else errors[0]

    @classmethod
    def from_matrix(cls, matrix: np.ndarray):
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
    def from_translation(cls, x_off: float = 0.0, y_off: float = 0.0, z_off: float = 0.0):
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

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.copy(value) for key, value in self.__dict__.items()}

        return new_coreg

    def __add__(self, other: Coreg) -> CoregPipeline:
        """Return a pipeline consisting of self and the other coreg function."""
        if not isinstance(other, Coreg):
            raise ValueError(f"Incompatible add type: {type(other)}. Expected 'Coreg' subclass")
        return CoregPipeline([self, other])

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        # FOR DEVELOPERS: This function needs to be implemented.
        raise NotImplementedError("This should have been implemented by subclassing")

    def _to_matrix_func(self) -> np.ndarray:
        # FOR DEVELOPERS: This function needs to be implemented if the `self._meta['matrix']` keyword is not None.

        # Try to see if a matrix exists.
        meta_matrix = self._meta.get("matrix")
        if meta_matrix is not None:
            assert meta_matrix.shape == (4, 4), f"Invalid _meta matrix shape. Expected: (4, 4), got {meta_matrix.shape}"
            return meta_matrix

        raise NotImplementedError("This should be implemented by subclassing")

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine, **kwargs) -> np.ndarray:
        # FOR DEVELOPERS: This function is only needed for non-rigid transforms.
        raise NotImplementedError("This should have been implemented by subclassing")

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        # FOR DEVELOPERS: This function is only needed for non-rigid transforms.
        raise NotImplementedError("This should have been implemented by subclassing")


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
        super().__init__(meta={"bias_func": bias_func})

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the bias using the bias_func."""
        if verbose:
            print("Estimating bias...")
        diff = ref_dem - tba_dem
        diff = diff[np.isfinite(diff)]

        if np.count_nonzero(np.isfinite(diff)) == 0:
            raise ValueError("No finite values in bias comparison.")

        # Use weights if those were provided.
        bias = self._meta["bias_func"](diff) if weights is None \
            else self._meta["bias_func"](diff, weights=weights)

        if verbose:
            print("Bias estimated")

        self._meta["bias"] = bias

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

    def __init__(self, max_iterations=100, tolerance=0.05, rejection_scale=2.5, num_levels=6):
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

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the rigid transform from tba_dem to ref_dem."""

        if weights is not None:
            warnings.warn("ICP was given weights, but does not support it.")


        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        points: dict[str, np.ndarray] = {}
        # Generate the x and y coordinates for the reference_dem
        x_coords, y_coords = _get_x_and_y_coords(ref_dem.shape, transform)

        centroid = np.array([np.mean([bounds.left, bounds.right]), np.mean([bounds.bottom, bounds.top]), 0.0])
        # Subtract by the bounding coordinates to avoid float32 rounding errors.
        x_coords -= centroid[0]
        y_coords -= centroid[1]
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


class Deramp(Coreg):
    """
    Polynomial DEM deramping.

    Estimates an n-D polynomial between the difference of two DEMs.
    """

    def __init__(self, degree: int = 1, subsample: Union[int, float] = 5e5):
        """
        Instantiate a deramping correction object.

        :param degree: The polynomial degree to estimate. degree=0 is a simple bias correction.
        :param subsample: Factor for subsampling the input raster for speed-up.
        If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of pixels to extract.
        """
        self.degree = degree
        self.subsample = subsample

        super().__init__()

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
            res = targets - poly2d(x_coords, y_coords, coefs)
            return res[np.isfinite(res)]

        if verbose:
            print("Estimating deramp function...")

        # reduce number of elements for speed
        # Get number of points to extract
        max_points = np.size(x_coords)
        if (self.subsample <= 1) & (self.subsample >= 0):
            npoints = int(self.subsample * max_points)
        elif self.subsample > 1:
            npoints = int(self.subsample)
        else:
            raise ValueError("`subsample` must be >= 0")

        if max_points > npoints:
            indices = np.random.choice(max_points, npoints, replace=False)
            x_coords = x_coords[indices]
            y_coords = y_coords[indices]
            ddem = ddem[indices]

        # Optimize polynomial parameters
        coefs = scipy.optimize.leastsq(
            func=residuals,
            x0=np.zeros(shape=((self.degree + 1) * (self.degree + 2) // 2)),
            args=(x_coords, y_coords, ddem)
        )

        self._meta["coefficients"] = coefs[0]
        self._meta["func"] = lambda x, y: poly2d(x, y, coefs[0])

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine, **kwargs) -> np.ndarray:
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

    def __init__(self, pipeline: list[Coreg]):
        """
        Instantiate a new coregistration pipeline.

        :param: Coregistration steps to run in the sequence they are given.
        """
        self.pipeline = pipeline

        super().__init__()

    def __repr__(self):
        return f"CoregPipeline: {self.pipeline}"

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.copy(value) for key, value in self.__dict__.items() if key != "pipeline"}
        new_coreg.pipeline = [step.copy() for step in self.pipeline]

        return new_coreg

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Fit each coregistration step with the previously transformed DEM."""
        tba_dem_mod = tba_dem.copy()

        for i, coreg in enumerate(self.pipeline):
            if verbose:
                print(f"Running pipeline step: {i + 1} / {len(self.pipeline)}")
            coreg._fit_func(ref_dem, tba_dem_mod, transform=transform, weights=weights, verbose=verbose)
            coreg._fit_called = True

            tba_dem_mod = coreg.apply(tba_dem_mod, transform)

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine, **kwargs) -> np.ndarray:
        """Apply the coregistration steps sequentially to a DEM."""
        dem_mod = dem.copy()
        for coreg in self.pipeline:
            dem_mod = coreg.apply(dem_mod, transform, **kwargs)

        return dem_mod

    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the coregistration steps sequentially to a set of points."""
        coords_mod = coords.copy()

        for coreg in self.pipeline:
            coords_mod = coreg.apply_pts(coords_mod).reshape(coords_mod.shape)

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

    def __init__(self, max_iterations: int = 10, offset_threshold: float = 0.05):
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param offset_threshold: The residual offset threshold after which to stop the iterations.
        """
        self.max_iterations = max_iterations
        self.offset_threshold = offset_threshold

        super().__init__()

    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: Optional[rio.transform.Affine],
                  weights: Optional[np.ndarray], verbose: bool = False):
        """Estimate the x/y/z offset between two DEMs."""
        if verbose:
            print("Running Nuth and Kääb (2011) coregistration")

        bounds, resolution = _transform_to_bounds_and_res(ref_dem.shape, transform)
        # Make a new DEM which will be modified inplace
        aligned_dem = tba_dem.copy()

        # Calculate slope and aspect maps from the reference DEM
        if verbose:
            print("   Calculate slope and aspect")
        slope, aspect = calculate_slope_and_aspect(ref_dem)

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
        offset_east, offset_north, bias = 0.0, 0.0, 0.0

        # Calculate initial dDEM statistics
        elevation_difference = ref_dem - aligned_dem
        bias = np.nanmedian(elevation_difference)
        nmad_old = xdem.spatialstats.nmad(elevation_difference)
        if verbose:
            print("   Statistics on initial dh:")
            print("      Median = {:.2f} - NMAD = {:.2f}".format(bias, nmad_old))

        # Iteratively run the analysis until the maximum iterations or until the error gets low enough
        if verbose:
            print("   Iteratively estimating horizontal shit:")

        # If verbose is True, will use progressbar and print additional statements
        pbar = trange(self.max_iterations, disable=not verbose, desc="   Progress")
        for i in pbar:

            # Calculate the elevation difference and the residual (NMAD) between them.
            elevation_difference = ref_dem - aligned_dem
            bias = np.nanmedian(elevation_difference)
            # Correct potential biases
            elevation_difference -= bias

            # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
            east_diff, north_diff, _ = get_horizontal_shift(  # type: ignore
                elevation_difference=elevation_difference,
                slope=slope,
                aspect=aspect
            )
            if verbose:
                pbar.write("      #{:d} - Offset in pixels : ({:.2f}, {:.2f})".format(i + 1, east_diff, north_diff))

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
            bias = np.nanmedian(elevation_difference)
            nmad_new = xdem.spatialstats.nmad(elevation_difference)
            nmad_gain = (nmad_new - nmad_old) / nmad_old*100

            if verbose:
                pbar.write("      Median = {:.2f} - NMAD = {:.2f}  ==>  Gain = {:.2f}%".format(bias, nmad_new, nmad_gain))

            # Stop if the NMAD is low and a few iterations have been made
            assert ~np.isnan(nmad_new), (offset_east, offset_north)

            offset = np.sqrt(east_diff**2 + north_diff**2)
            if i > 1 and offset < self.offset_threshold:
                if verbose:
                    pbar.write(f"   Last offset was below the residual offset threshold of {self.offset_threshold} -> stopping")
                break

            nmad_old = nmad_new

        # Print final results
        if verbose:
            print("\n   Final offset in pixels (east, north) : ({:f}, {:f})".format(offset_east, offset_north))
            print("   Statistics on coregistered dh:")
            print("      Median = {:.2f} - NMAD = {:.2f}".format(bias, nmad_new))

        self._meta["offset_east_px"] = offset_east
        self._meta["offset_north_px"] = offset_north
        self._meta["bias"] = bias
        self._meta["resolution"] = resolution

    def _to_matrix_func(self) -> np.ndarray:
        """Return a transformation matrix from the estimated offsets."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += offset_east
        matrix[1, 3] += offset_north
        matrix[2, 3] += self._meta["bias"]

        return matrix


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    """Invert a transformation matrix."""
    with warnings.catch_warnings():
        # Deprecation warning from pytransform3d. Let's hope that is fixed in the near future.
        warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")

        checked_matrix = pytransform3d.transformations.check_matrix(matrix)
        # Invert the transform if wanted.
        return pytransform3d.transformations.invert_transform(checked_matrix)


def apply_matrix(dem: np.ndarray, transform: rio.transform.Affine, matrix: np.ndarray, invert: bool = False,
                 centroid: Optional[tuple[float, float, float]] = None,
                 resampling: Union[int, str] = "bilinear",
                 dilate_mask: bool = False, **kwargs) -> np.ndarray:
    """
    Apply a 3D transformation matrix to a 2.5D DEM.

    The transformation is applied as a value correction using linear deramping, and 2D image warping.

    1. Convert the DEM into a point cloud (not for gridding; for estimating the DEM shifts).
    2. Transform the point cloud in 3D using the 4x4 matrix.
    3. Measure the difference in elevation between the original and transformed points.
    4. Estimate a linear deramp from the elevation difference, and apply the correction to the DEM values.
    5. Convert the horizontal coordinates of the transformed points to pixel index coordinates.
    6. Apply the pixel-wise displacement in 2D using the new pixel coordinates.
    7. Apply the same displacement to a nodata-mask to exclude previous and/or new nans.

    :param dem: The DEM to transform.
    :param transform: The Affine transform object (georeferencing) of the DEM.
    :param matrix: A 4x4 transformation matrix to apply to the DEM.
    :param invert: Invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations. Defaults to the midpoint (Z=0)
    :param resampling: The resampling method to use. Can be `nearest`, `bilinear`, `cubic` or an integer from 0-5.
    :param dilate_mask: Dilate the nan mask to exclude edge pixels that could be wrong.

    :returns: The transformed DEM with NaNs as nodata values (replaces a potential mask of the input `dem`).
    """
    # Parse the resampling argument given.
    if isinstance(resampling, int):
        resampling_order = resampling
    elif resampling == "cubic":
        resampling_order = 3
    elif resampling == "bilinear":
        resampling_order = 1
    elif resampling == "nearest":
        resampling_order = 0
    else:
        raise ValueError(
            f"`{resampling}` is not a valid resampling mode."
            " Choices: [`nearest`, `bilinear`, `cubic`] or an integer."
        )
    # Copy the DEM to make sure the original is not modified, and convert it into an ndarray
    demc = np.array(dem)

    # Check if the matrix only contains a Z correction. In that case, only shift the DEM values by the bias.
    empty_matrix = np.diag(np.ones(4, float))
    empty_matrix[2, 3] = matrix[2, 3]
    if np.mean(np.abs(empty_matrix - matrix)) == 0.0:
        return demc + matrix[2, 3]

    # Opencv is required down from here
    if not _has_cv2:
        raise ValueError("Optional dependency needed. Install 'opencv'")

    nan_mask = spatial_tools.get_mask(dem)
    assert np.count_nonzero(~nan_mask) > 0, "Given DEM had all nans."
    # Create a filled version of the DEM. (skimage doesn't like nans)
    filled_dem = np.where(~nan_mask, demc, np.nan)

    # Get the centre coordinates of the DEM pixels.
    x_coords, y_coords = _get_x_and_y_coords(demc.shape, transform)

    bounds, resolution = _transform_to_bounds_and_res(dem.shape, transform)

    # If a centroid was not given, default to the center of the DEM (at Z=0).
    if centroid is None:
        centroid = (np.mean([bounds.left, bounds.right]), np.mean([bounds.bottom, bounds.top]), 0.0)
    else:
        assert len(centroid) == 3, f"Expected centroid to be 3D X/Y/Z coordinate. Got shape of {len(centroid)}"

    # Shift the coordinates to centre around the centroid.
    x_coords -= centroid[0]
    y_coords -= centroid[1]

    # Create a point cloud of X/Y/Z coordinates
    point_cloud = np.dstack((x_coords, y_coords, filled_dem))

    # Shift the Z components by the centroid.
    point_cloud[:, 2] -= centroid[2]

    if invert:
        matrix = invert_matrix(matrix)

    # Transform the point cloud using the matrix.
    transformed_points = cv2.perspectiveTransform(
        point_cloud.reshape((1, -1, 3)),
        matrix,
    ).reshape(point_cloud.shape)

    # Estimate the vertical difference of old and new point cloud elevations.
    deramp = deramping(
        (point_cloud[:, :, 2] - transformed_points[:, :, 2])[~nan_mask].flatten(),
        point_cloud[:, :, 0][~nan_mask].flatten(),
        point_cloud[:, :, 1][~nan_mask].flatten(),
        degree=1
    )
    # Shift the elevation values of the soon-to-be-warped DEM.
    filled_dem -= deramp(x_coords, y_coords)

    # Create gap-free arrays of x and y coordinates to be converted into index coordinates.
    x_inds = rio.fill.fillnodata(transformed_points[:, :, 0].copy(), mask=(~nan_mask).astype("uint8"))
    y_inds = rio.fill.fillnodata(transformed_points[:, :, 1].copy(), mask=(~nan_mask).astype("uint8"))

    # Divide the coordinates by the resolution to create index coordinates.
    x_inds /= resolution
    y_inds /= resolution
    # Shift the x coords so that bounds.left is equivalent to xindex -0.5
    x_inds -= x_coords.min() / resolution
    # Shift the y coords so that bounds.top is equivalent to yindex -0.5
    y_inds = (y_coords.max() / resolution) - y_inds

    # Create a skimage-compatible array of the new index coordinates that the pixels shall have after warping.
    inds = np.vstack((y_inds.reshape((1,) + y_inds.shape), x_inds.reshape((1,) + x_inds.shape)))

    with warnings.catch_warnings():
        # An skimage warning that will hopefully be fixed soon. (2021-07-30)
        warnings.filterwarnings("ignore", message="Passing `np.nan` to mean no clipping in np.clip")
        # Warp the DEM
        transformed_dem = skimage.transform.warp(
            filled_dem,
            inds,
            order=resampling_order,
            mode="constant",
            cval=np.nan,
            preserve_range=True
        )
        # Warp the NaN mask, setting true to all values outside the new frame.
        tr_nan_mask = skimage.transform.warp(
            nan_mask.astype("uint8"),
            inds,
            order=resampling_order,
            mode="constant",
            cval=1,
            preserve_range=True
        ) > 0

    if dilate_mask:
        tr_nan_mask = scipy.ndimage.binary_dilation(tr_nan_mask, iterations=resampling_order)

    # Apply the transformed nan_mask
    transformed_dem[tr_nan_mask] = np.nan

    assert np.count_nonzero(~np.isnan(transformed_dem)) > 0, "Transformed DEM has all nans."

    return transformed_dem


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

        super().__init__()

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

    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine, **kwargs) -> np.ndarray:
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
        if self.degree == 0:  # If it's just a bias correction.
            return self._meta["coefficients"][-1]
        elif self.degree < 2:
            raise NotImplementedError
        else:
            raise ValueError("A 2nd degree or higher ZScaleCorr cannot be described as a 4x4 matrix!")

class BlockwiseCoreg(Coreg):
    """
    Block-wise coreg class for nonlinear estimations.

    A coreg class of choice is run on an arbitrary subdivision of the raster. When later applying the coregistration,\
        the optimal warping is interpolated based on X/Y/Z shifts from the coreg algorithm at the grid points.

    E.g. a subdivision of 4 means to divide the DEM in four equally sized parts. These parts are then coregistered\
        separately, creating four Coreg.fit results. If the subdivision is not divisible by the raster shape,\
        subdivision is made as best as possible to have approximately equal pixel counts.
    """

    def __init__(self, coreg: Coreg | CoregPipeline, subdivision: int, success_threshold: float = 0.8, n_threads: int | None = None, warn_failures: bool = False):
        """
        Instantiate a blockwise coreg object.

        :param coreg: An instantiated coreg object to fit in the subdivided DEMs.
        :param subdivision: The number of chunks to divide the DEMs in. E.g. 4 means four different transforms.
        :param success_threshold: Raise an error if fewer chunks than the fraction failed for any reason.
        :param n_threads: The maximum amount of threads to use. Default=auto
        :param warn_failures: Trigger or ignore warnings for each exception/warning in each block.
        """
        if isinstance(coreg, type):
            raise ValueError(
                    "The 'coreg' argument must be an instantiated Coreg subclass. "
                    "Hint: write e.g. ICP() instead of ICP"
            )
        self.coreg = coreg
        self.subdivision = subdivision
        self.success_threshold = success_threshold
        self.n_threads = n_threads
        self.warn_failures = warn_failures

        super().__init__()

        self._meta["coreg_meta"] = []



    def _fit_func(self, ref_dem: np.ndarray, tba_dem: np.ndarray, transform: rio.transform.Affine | None,
                  weights: np.ndarray | None, verbose: bool = False):
        """Fit the coreg approach for each subdivision."""

        groups = self.subdivide_array(tba_dem.shape)

        indices = np.unique(groups)

        progress_bar = tqdm(total=indices.size, desc="Coregistering chunks", disable=(not verbose))

        def coregister(i: int) -> dict[str, Any] | BaseException | None:
            """
            Coregister a chunk in a thread-safe way.

            :returns:
                * If it succeeds: A dictionary of the fitting metadata.
                * If it fails: The associated exception.
                * If the block is empty: None
            """
            inlier_mask = groups == i

            # Find the corresponding slice of the inlier_mask to subset the data
            rows, cols = np.where(inlier_mask)
            arrayslice = np.s_[rows.min():rows.max() + 1, cols.min(): cols.max() + 1]

            # Copy a subset of the two DEMs, the mask, the coreg instance, and make a new subset transform
            ref_subset = ref_dem[arrayslice].copy()
            tba_subset = tba_dem[arrayslice].copy()

            if any(np.all(~np.isfinite(dem)) for dem in (ref_subset, tba_subset)):
                return None
            mask_subset = inlier_mask[arrayslice].copy()
            west, top = rio.transform.xy(transform, min(rows), min(cols), offset="ul")
            transform_subset = rio.transform.from_origin(west, top, transform.a, -transform.e)
            coreg = self.coreg.copy()


            # Try to run the coregistration. If it fails for any reason, skip it and save the exception.
            try:
                coreg.fit(
                    reference_dem=ref_subset,
                    dem_to_be_aligned=tba_subset,
                    transform=transform_subset,
                    inlier_mask=mask_subset
                )

                nmad, median = coreg.error(
                    reference_dem=ref_subset,
                    dem_to_be_aligned=tba_subset,
                    error_type=["nmad", "median"],
                    inlier_mask=mask_subset,
                    transform=transform_subset
                )
            except Exception as exception:
                return exception

            meta: dict[str, Any] = {
                "i": i,
                "transform": transform_subset,
                "inlier_count": np.count_nonzero(mask_subset & np.isfinite(ref_subset) & np.isfinite(tba_subset)),
                "nmad": nmad,
                "median": median
            }
            # Find the center of the inliers.
            inlier_positions = np.argwhere(mask_subset)
            mid_row = np.mean(inlier_positions[:, 0]).astype(int)
            mid_col = np.mean(inlier_positions[:, 1]).astype(int)

            # Find the indices of all finites within the mask
            finites = np.argwhere(np.isfinite(tba_subset) & mask_subset)
            # Calculate the distance between the approximate center and all finite indices
            distances = np.linalg.norm(finites - np.array([mid_row, mid_col]), axis=1)
            # Find the index representing the closest finite value to the center.
            closest = np.argwhere(distances == distances.min())

            # Assign the closest finite value as the representative point
            representative_row, representative_col = finites[closest][0][0]
            meta["representative_x"], meta["representative_y"] = rio.transform.xy(transform_subset, representative_row, representative_col)
            meta["representative_val"] = ref_subset[representative_row, representative_col]

            # If the coreg is a pipeline, copy its metadatas to the output meta
            if hasattr(coreg, "pipeline"):
                meta["pipeline"] = [step._meta.copy() for step in coreg.pipeline]

            # Copy all current metadata (except for the alreay existing keys like "i", "min_row", etc, and the "coreg_meta" key)
            # This can then be iteratively restored when the apply function should be called.
            meta.update({key: value for key, value in coreg._meta.items() if key not in ["coreg_meta"] + list(meta.keys())})

            progress_bar.update()

            return meta.copy()

        # Catch warnings; only show them if
        exceptions: list[BaseException | warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("default")
            with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                results = executor.map(coregister, indices)

            exceptions += list(caught_warnings)

        empty_blocks = 0
        for result in results:
            if isinstance(result, BaseException):
                exceptions.append(result)
            elif result is None:
                empty_blocks += 1
                continue
            else:
                self._meta["coreg_meta"].append(result)

        progress_bar.close()

        # Stop if the success rate was below the threshold
        if ((len(self._meta["coreg_meta"]) + empty_blocks) / self.subdivision) <= self.success_threshold:
            raise ValueError(
                f"Fitting failed for {len(exceptions)} chunks:\n" +
                "\n".join(map(str, exceptions[:5])) +
                f"\n... and {len(exceptions) - 5} more" if len(exceptions) > 5 else ""
            )

        if self.warn_failures:
            for exception in exceptions:
                warnings.warn(str(exception))

        # Set the _fit_called parameters (only identical copies of self.coreg have actually been called)
        self.coreg._fit_called = True
        if hasattr(self.coreg, "pipeline"):
            for step in self.coreg.pipeline:
                step._fit_called = True

    def _restore_metadata(self, meta: dict[str, Any]) -> None:
        """
        Given some metadata, set it in the right place.

        :param meta: A metadata file to update self._meta
        """
        self.coreg._meta.update(meta)

        if hasattr(self.coreg, "pipeline") and "pipeline" in meta:
            for i, step in enumerate(self.coreg.pipeline):
                step._meta.update(meta["pipeline"][i])

    def to_points(self) -> np.ndarray:
        """
        Convert the blockwise coregistration matrices to 3D (source -> destination) points.

        The returned shape is (N, 3, 2) where the dimensions represent:
            0. The point index where N is equal to the amount of subdivisions.
            1. The X/Y/Z coordinate of the point.
            2. The old/new position of the point.

        To acquire the first point's original position: points[0, :, 0]
        To acquire the first point's new position: points[0, :, 1]
        To acquire the first point's Z difference: points[0, 2, 1] - points[0, 2, 0]

        :returns: An array of 3D source -> destionation points.
        """
        if len(self._meta["coreg_meta"]) == 0:
            raise AssertionError("No coreg results exist. Has '.fit()' been called?")
        points = np.empty(shape=(0, 3, 2))
        for meta in self._meta["coreg_meta"]:
            self._restore_metadata(meta)

            #x_coord, y_coord = rio.transform.xy(meta["transform"], meta["representative_row"], meta["representative_col"])
            x_coord, y_coord = meta["representative_x"], meta["representative_y"]


            old_position = np.reshape([x_coord, y_coord, meta["representative_val"]], (1, 3))
            new_position = self.coreg.apply_pts(old_position)
        
            points = np.append(points, np.dstack((old_position, new_position)), axis=0)

        return points

    def stats(self) -> pd.DataFrame:
        """
        Return statistics for each chunk in the blockwise coregistration.

            * center_{x,y,z}: The center coordinate of the chunk in georeferenced units.
            * {x,y,z}_off: The calculated offset in georeferenced units.
            * inlier_count: The number of pixels that were inliers in the chunk.
            * nmad: The NMAD after coregistration.
            * median: The bias after coregistration.

        :raises ValueError: If no coregistration results exist yet.

        :returns: A dataframe of statistics for each chunk.
        """
        points = self.to_points()

        chunk_meta = {meta["i"]: meta for meta in self._meta["coreg_meta"]}

        statistics: list[dict[str, Any]] = []
        for i in range(points.shape[0]):
            if i not in chunk_meta:
                continue
            statistics.append(
                {
                    "center_x": points[i, 0, 0],
                    "center_y": points[i, 1, 0],
                    "center_z": points[i, 2, 0],
                    "x_off": points[i, 0, 1] - points[i, 0, 0],
                    "y_off": points[i, 1, 1] - points[i, 1, 0],
                    "z_off": points[i, 2, 1] - points[i, 2, 0],
                    "inlier_count": chunk_meta[i]["inlier_count"],
                    "nmad": chunk_meta[i]["nmad"],
                    "median": chunk_meta[i]["median"]
                }
            )

        stats_df = pd.DataFrame(statistics)
        stats_df.index.name = "chunk"

        return stats_df
            

    def subdivide_array(self, shape: tuple[int, ...]) -> np.ndarray:
        """
        Return the grid subdivision for a given DEM shape.

        :param shape: The shape of the input DEM.
        
        :returns: An array of shape 'shape' with 'self.subdivision' unique indices.
        """
        if len(shape) == 3 and shape[0] == 1:  # Account for (1, row, col) shapes
            shape = (shape[1], shape[2])
        return spatial_tools.subdivide_array(shape, count=self.subdivision)


    def _apply_func(self, dem: np.ndarray, transform: rio.transform.Affine, **kwargs) -> np.ndarray:

        points = self.to_points()

        bounds, resolution = _transform_to_bounds_and_res(dem.shape, transform)
        
        representative_height = np.nanmean(dem)
        edges_source = np.array([
            [bounds.left + resolution / 2, bounds.top - resolution / 2, representative_height],
            [bounds.right - resolution / 2, bounds.top - resolution / 2, representative_height],
            [bounds.left + resolution / 2, bounds.bottom + resolution / 2, representative_height],
            [bounds.right - resolution / 2, bounds.bottom + resolution / 2, representative_height]
        ])
        edges_dest = self.apply_pts(edges_source)
        edges = np.dstack((edges_source, edges_dest))

        all_points = np.append(points, edges, axis=0)

        warped_dem = warp_dem(
            dem=dem,
            transform=transform,
            source_coords=all_points[:, :, 0],
            destination_coords=all_points[:, :, 1],
            resampling="linear"
        )

        return warped_dem



    def _apply_pts_func(self, coords: np.ndarray) -> np.ndarray:
        """Apply the scaling model to a set of points."""
        points = self.to_points()

        new_coords = coords.copy()

        for dim in range(0, 3):
            with warnings.catch_warnings():
                # ZeroDivisionErrors may happen when the transformation is empty (which is fine)
                warnings.filterwarnings("ignore", message="ZeroDivisionError")
                model = scipy.interpolate.Rbf(
                    points[:, 0, 0],
                    points[:, 1, 0],
                    points[:, dim, 1] - points[:, dim, 0],
                    function="linear",
                )

            new_coords[:, dim] += model(coords[:, 0], coords[:, 1])

        return new_coords


def warp_dem(
        dem: np.ndarray,
        transform: rio.transform.Affine,
        source_coords: np.ndarray,
        destination_coords: np.ndarray,
        resampling: str = "cubic",
        trim_border: bool = True,
        dilate_mask: bool = True,
    ) -> np.ndarray:
    """
    Warp a DEM using a set of source-destination 2D or 3D coordinates.

    :param dem: The DEM to warp. Allowed shapes are (1, row, col) or (row, col)
    :param transform: The Affine transform of the DEM.
    :param source_coords: The source 2D or 3D points. must be X/Y/(Z) coords of shape (N, 2) or (N, 3).
    :param destination_coords: The destination 2D or 3D points. Must have the exact same shape as 'source_coords'
    :param resampling: The resampling order to use. Choices: ['nearest', 'linear', 'cubic'].
    :param trim_border: Remove values outside of the interpolation regime (True) or leave them unmodified (False).
    :param dilate_mask: Dilate the nan mask to exclude edge pixels that could be wrong.

    :raises ValueError: If the inputs are poorly formatted.
    :raises AssertionError: For unexpected outputs.

    :returns: A warped DEM with the same shape as the input.
    """
    if source_coords.shape != destination_coords.shape:
        raise ValueError(
            f"Incompatible shapes: source_coords '({source_coords.shape})' and "
            f"destination_coords '({destination_coords.shape})' shapes must be the same"
        )
    if (len(source_coords.shape) > 2) or (source_coords.shape[1] < 2) or (source_coords.shape[1] > 3):
        raise ValueError(
                "Invalid coordinate shape. Expected 2D or 3D coordinates of shape (N, 2) or (N, 3). "
                f"Got '{source_coords.shape}'"
        )
    allowed_resampling_strs = ["nearest", "linear", "cubic"]
    if resampling not in allowed_resampling_strs:
        raise ValueError(f"Resampling type '{resampling}' not understood. Choices: {allowed_resampling_strs}")

    dem_arr, dem_mask = spatial_tools.get_array_and_mask(dem)

    bounds, resolution = _transform_to_bounds_and_res(dem_arr.shape, transform)

    no_horizontal =  np.sum(np.linalg.norm(destination_coords[:, :2] - source_coords[:, :2], axis=1)) < 1e-6
    no_vertical = source_coords.shape[1] > 2 and np.sum(np.abs(destination_coords[:, 2] - source_coords[:, 2])) < 1e-6

    if no_horizontal and no_vertical:
        warnings.warn("No difference between source and destination coordinates. Returning self.")
        return dem

    source_coords_scaled = source_coords.copy()
    destination_coords_scaled = destination_coords.copy()
    # Scale the coordinates to index-space
    for coords in (source_coords_scaled, destination_coords_scaled):
        coords[:, 0] = (
            dem_arr.shape[1] *
            (coords[:, 0] - bounds.left) /
            (bounds.right - bounds.left)
        )
        coords[:, 1] = (
            dem_arr.shape[0] *
            (1 - (
                coords[:, 1] - bounds.bottom) /
                (bounds.top - bounds.bottom)
            )
        )

    # Generate a grid of x and y index coordinates.
    grid_y, grid_x = np.mgrid[0:dem_arr.shape[0], 0:dem_arr.shape[1]]

    if no_horizontal:
        warped = dem_arr.copy()
    else:
        # Interpolate the sparse source-destination points to a grid.
        # (row, col, 0) represents the destination y-coordinates of the pixels.
        # (row, col, 1) represents the destination x-coordinates of the pixels.
        new_indices = scipy.interpolate.griddata(
            source_coords_scaled[:, [1, 0]],
            destination_coords_scaled[:, [1, 0]],  # Coordinates should be in y/x (not x/y) for some reason..
            (grid_y, grid_x),
            method="linear"
        )

        # If the border should not be trimmed, just assign the original indices to the missing values.
        if not trim_border:
            missing_ys = np.isnan(new_indices[:, :, 0])
            missing_xs = np.isnan(new_indices[:, :, 1])
            new_indices[:, :, 0][missing_ys] = grid_y[missing_ys]
            new_indices[:, :, 1][missing_xs] = grid_x[missing_xs]

        order = {"nearest": 0, "linear": 1, "cubic": 3}

        with warnings.catch_warnings():
            # An skimage warning that will hopefully be fixed soon. (2021-06-08)
            warnings.filterwarnings("ignore", message="Passing `np.nan` to mean no clipping in np.clip")
            warped = skimage.transform.warp(
                image=np.where(dem_mask, np.nan, dem_arr),
                inverse_map=np.moveaxis(new_indices, 2, 0),
                output_shape=dem_arr.shape,
                preserve_range=True,
                order=order[resampling],
                cval=np.nan
            )
            new_mask = skimage.transform.warp(
                image=dem_mask,
                inverse_map=np.moveaxis(new_indices, 2, 0),
                output_shape=dem_arr.shape,
                cval=False
            ) > 0

        if dilate_mask:
            new_mask = scipy.ndimage.binary_dilation(new_mask, iterations=order[resampling]).astype(new_mask.dtype)

        warped[new_mask] = np.nan


    # If the coordinates are 3D (N, 3), apply a Z correction as well.
    if not no_vertical:
        grid_offsets = scipy.interpolate.griddata(
            points=destination_coords_scaled[:, :2],
            values=destination_coords_scaled[:, 2] - source_coords_scaled[:, 2],
            xi=(grid_x, grid_y),
            method=resampling,
            fill_value=np.nan
        )
        if not trim_border:
            grid_offsets[np.isnan(grid_offsets)] = np.nanmean(grid_offsets)

        warped += grid_offsets

    assert not np.all(np.isnan(warped)), "All-NaN output."

    return warped.reshape(dem.shape)

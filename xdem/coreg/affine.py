"""Affine coregistration classes."""

from __future__ import annotations

import warnings
from typing import Any, Callable, TypeVar

import xdem.coreg.base

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import geopandas as gpd
import numpy as np
import rasterio as rio
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from geoutils.raster import Raster, get_array_and_mask
from tqdm import trange

from xdem._typing import NDArrayb, NDArrayf
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
    c_parameter = np.round(c_parameter, 3)

    # Calculate the easting and northing offsets from the above parameters
    east_offset = np.round(a_parameter * np.sin(b_parameter), 3)
    north_offset = np.round(a_parameter * np.cos(b_parameter), 3)

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

    def __init__(
        self,
        subsample: float | int = 1.0,
        matrix: NDArrayf | None = None,
        meta: CoregDict | None = None,
    ) -> None:
        """Instantiate a generic AffineCoreg method."""

        super().__init__(meta=meta)

        # Define subsample size
        self._meta["subsample"] = subsample

        if matrix is not None:
            with warnings.catch_warnings():
                # This error is fixed in the upcoming 1.8
                warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")
                valid_matrix = pytransform3d.transformations.check_transform(matrix)
            self._meta["matrix"] = valid_matrix
        self._is_affine = True

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


class VerticalShift(AffineCoreg):
    """
    DEM vertical shift correction.

    Estimates the mean (or median, weighted avg., etc.) vertical offset between two DEMs.
    """

    def __init__(
        self, vshift_func: Callable[[NDArrayf], np.floating[Any]] = np.average, subsample: float | int = 1.0
    ) -> None:  # pylint:
        # disable=super-init-not-called
        """
        Instantiate a vertical shift correction object.

        :param vshift_func: The function to use for calculating the vertical shift. Default: (weighted) average.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        self._meta: CoregDict = {}  # All __init__ functions should instantiate an empty dict.

        super().__init__(meta={"vshift_func": vshift_func}, subsample=subsample)

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the vertical shift using the vshift_func."""

        if verbose:
            print("Estimating the vertical shift...")
        diff = ref_elev - tba_elev

        valid_mask = np.logical_and.reduce((inlier_mask, np.isfinite(diff)))
        subsample_mask = self._get_subsample_on_valid_mask(valid_mask=valid_mask)

        diff = diff[subsample_mask]

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

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the VerticalShift function to a DEM."""
        return elev + self._meta["vshift"], transform

    def _apply_pts(
        self,
        elev: gpd.GeoDataFrame,
        z_name: str = "z",
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:

        """Apply the VerticalShift function to a set of points."""
        dem_copy = elev.copy()
        dem_copy[z_name] += self._meta["vshift"]
        return dem_copy

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

        # TODO: Move these to _meta?
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rejection_scale = rejection_scale
        self.num_levels = num_levels

        super().__init__(subsample=subsample)

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the rigid transform from tba_dem to ref_dem."""

        if weights is not None:
            warnings.warn("ICP was given weights, but does not support it.")

        bounds, resolution = _transform_to_bounds_and_res(ref_elev.shape, transform)
        # Generate the x and y coordinates for the reference_dem
        x_coords, y_coords = _get_x_and_y_coords(ref_elev.shape, transform)
        gradient_x, gradient_y = np.gradient(ref_elev)

        normal_east = np.sin(np.arctan(gradient_y / resolution)) * -1
        normal_north = np.sin(np.arctan(gradient_x / resolution))
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
            verbose=verbose,
            z_name="z",
        )

    def _fit_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
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
        bounds, resolution = _transform_to_bounds_and_res(rst_elev.shape, transform)

        # Generate the x and y coordinates for the TBA DEM
        x_coords, y_coords = _get_x_and_y_coords(rst_elev.shape, transform)
        centroid = (np.mean([bounds.left, bounds.right]), np.mean([bounds.bottom, bounds.top]), 0.0)
        # Subtract by the bounding coordinates to avoid float32 rounding errors.
        x_coords -= centroid[0]
        y_coords -= centroid[1]

        gradient_x, gradient_y = np.gradient(rst_elev)

        # This CRS is temporary and doesn't affect the result. It's just needed for Raster instantiation.
        dem_kwargs = {"transform": transform, "crs": rio.CRS.from_epsg(32633), "nodata": -9999.0}
        normal_east = Raster.from_array(np.sin(np.arctan(gradient_y / resolution)) * -1, **dem_kwargs)
        normal_north = Raster.from_array(np.sin(np.arctan(gradient_x / resolution)), **dem_kwargs)
        normal_up = Raster.from_array(1 - np.linalg.norm([normal_east.data, normal_north.data], axis=0), **dem_kwargs)

        valid_mask = ~np.isnan(rst_elev) & ~np.isnan(normal_east.data) & ~np.isnan(normal_north.data)

        points: dict[str, NDArrayf] = {}
        points["raster"] = np.dstack(
            [
                x_coords[valid_mask],
                y_coords[valid_mask],
                rst_elev[valid_mask],
                normal_east.data[valid_mask],
                normal_north.data[valid_mask],
                normal_up.data[valid_mask],
            ]
        ).squeeze()

        # TODO: Should be a way to not duplicate this column and just feed it directly
        point_elev["E"] = point_elev.geometry.x.values
        point_elev["N"] = point_elev.geometry.y.values

        if any(col not in point_elev for col in ["nx", "ny", "nz"]):
            for key, raster in [("nx", normal_east), ("ny", normal_north), ("nz", normal_up)]:
                raster.tags["AREA_OR_POINT"] = "Area"
                point_elev[key] = raster.interp_points(
                    point_elev[["E", "N"]].values,
                    shift_area_or_point=True,
                )

        point_elev["E"] -= centroid[0]
        point_elev["N"] -= centroid[1]

        points["point"] = point_elev[["E", "N", z_name, "nx", "ny", "nz"]].values

        for key in points:
            points[key] = points[key][~np.any(np.isnan(points[key]), axis=1)].astype("float32")
            points[key][:, :2] -= resolution / 2

        icp = cv2.ppf_match_3d_ICP(self.max_iterations, self.tolerance, self.rejection_scale, self.num_levels)
        if verbose:
            print("Running ICP...")
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
        # TODO: Move matrix/invert_matrix to affine module?
        if ref == "raster":
            matrix = xdem.coreg.base.invert_matrix(matrix)

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

        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        self.poly_order = 1

        super().__init__(subsample=subsample)

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Fit the dDEM between the DEMs to a least squares polynomial equation."""
        ddem = ref_elev - tba_elev
        ddem[~inlier_mask] = np.nan
        x_coords, y_coords = _get_x_and_y_coords(ref_elev.shape, transform)
        fit_ramp, coefs = deramping(
            ddem, x_coords, y_coords, degree=self.poly_order, subsample=self._meta["subsample"], verbose=verbose
        )

        self._meta["coefficients"] = coefs[0]
        self._meta["func"] = fit_ramp

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the deramp function to a DEM."""
        x_coords, y_coords = _get_x_and_y_coords(elev.shape, transform)

        ramp = self._meta["func"](x_coords, y_coords)

        return elev + ramp, transform

    def _apply_pts(
        self,
        elev: gpd.GeoDataFrame,
        z_name: str = "z",
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Apply the deramp function to a set of points."""
        dem_copy = elev.copy()
        dem_copy[z_name].values += self._meta["func"](dem_copy.geometry.x.values, dem_copy.geometry.y.values)

        return dem_copy

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
    Nuth and Kääb (2011) DEM coregistration: iterative registration of horizontal and vertical shift using slope/aspect.

    Implemented after the paper: https://doi.org/10.5194/tc-5-271-2011.
    """

    def __init__(self, max_iterations: int = 10, offset_threshold: float = 0.05, subsample: int | float = 5e5) -> None:
        """
        Instantiate a new Nuth and Kääb (2011) coregistration object.

        :param max_iterations: The maximum allowed iterations before stopping.
        :param offset_threshold: The residual offset threshold after which to stop the iterations (in pixels).
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.
        """
        self._meta: CoregDict
        self.max_iterations = max_iterations
        self.offset_threshold = offset_threshold

        super().__init__(subsample=subsample)

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the x/y/z offset between two DEMs."""
        if verbose:
            print("Running Nuth and Kääb (2011) coregistration")

        bounds, resolution = _transform_to_bounds_and_res(ref_elev.shape, transform)
        # Make a new DEM which will be modified inplace
        aligned_dem = tba_elev.copy()

        # Check that DEM CRS is projected, otherwise slope is not correctly calculated
        if not crs.is_projected:
            raise NotImplementedError(
                f"DEMs CRS is {crs}. NuthKaab coregistration only works with \
projected CRS. First, reproject your DEMs in a local projected CRS, e.g. UTM, and re-run."
            )

        # Calculate slope and aspect maps from the reference DEM
        if verbose:
            print("   Calculate slope and aspect")

        slope_tan, aspect = _calculate_slope_and_aspect_nuthkaab(ref_elev)

        valid_mask = np.logical_and.reduce(
            (inlier_mask, np.isfinite(ref_elev), np.isfinite(tba_elev), np.isfinite(slope_tan))
        )
        subsample_mask = self._get_subsample_on_valid_mask(valid_mask=valid_mask)

        ref_elev[~subsample_mask] = np.nan

        # Make index grids for the east and north dimensions
        east_grid = np.arange(ref_elev.shape[1])
        north_grid = np.arange(ref_elev.shape[0])

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
        elevation_difference = ref_elev - aligned_dem

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
            elevation_difference = ref_elev - aligned_dem
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
            elevation_difference = ref_elev - aligned_dem

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

    def _fit_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Estimate the x/y/z offset between a DEM and points cloud.
        1. deleted elevation_function and nodata_function, shifting dataframe (points) instead of DEM.
        2. do not support latitude and longitude as inputs.

        :param z_name: the column name of dataframe used for elevation differencing

        """

        # Check which one is reference
        if isinstance(ref_elev, gpd.GeoDataFrame):
            point_elev = ref_elev
            rst_elev = tba_elev
            ref = "point"
        else:
            point_elev = tba_elev
            rst_elev = ref_elev
            ref = "raster"

        if verbose:
            print("Running Nuth and Kääb (2011) coregistration. Shift pts instead of shifting dem")

        rst_elev = Raster.from_array(rst_elev, transform=transform, crs=crs)
        tba_arr, _ = get_array_and_mask(rst_elev)

        bounds, resolution = _transform_to_bounds_and_res(ref_elev.shape, transform)
        x_coords, y_coords = (point_elev["E"].values, point_elev["N"].values)

        # Assume that the coordinates represent the center of a theoretical pixel.
        # The raster sampling is done in the upper left corner, meaning all point have to be respectively shifted
        x_coords -= resolution / 2
        y_coords += resolution / 2

        pts = np.array((x_coords, y_coords)).T
        # This needs to be consistent, so it's cardcoded here
        area_or_point = "Area"
        # Make a new DEM which will be modified inplace
        aligned_dem = rst_elev.copy()
        aligned_dem.tags["AREA_OR_POINT"] = area_or_point

        # Calculate slope and aspect maps from the reference DEM
        if verbose:
            print("   Calculate slope and aspect")
        slope, aspect = _calculate_slope_and_aspect_nuthkaab(tba_arr)

        slope_r = rst_elev.copy(new_array=np.ma.masked_array(slope[None, :, :], mask=~np.isfinite(slope[None, :, :])))
        slope_r.tags["AREA_OR_POINT"] = area_or_point
        aspect_r = rst_elev.copy(
            new_array=np.ma.masked_array(aspect[None, :, :], mask=~np.isfinite(aspect[None, :, :]))
        )
        aspect_r.tags["AREA_OR_POINT"] = area_or_point

        # Initialise east and north pixel offset variables (these will be incremented up and down)
        offset_east, offset_north, vshift = 0.0, 0.0, 0.0

        # Calculate initial DEM statistics
        slope_pts = slope_r.interp_points(pts, shift_area_or_point=True)
        aspect_pts = aspect_r.interp_points(pts, shift_area_or_point=True)
        tba_pts = aligned_dem.interp_points(pts, shift_area_or_point=True)

        # Treat new_pts as a window, every time we shift it a little bit to fit the correct view
        new_pts = pts.copy()

        elevation_difference = point_elev[z_name].values - tba_pts
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
            tba_pts = aligned_dem.interp_points(new_pts, shift_area_or_point=True)
            elevation_difference = point_elev[z_name].values - tba_pts

            # Mask out no data by dem's mask
            pts_, mask_ = _mask_dataframe_by_dem(new_pts, rst_elev)

            # Update values relataed to shifted pts
            elevation_difference = elevation_difference[mask_]
            slope_pts = slope_r.interp_points(pts_, shift_area_or_point=True)
            aspect_pts = aspect_r.interp_points(pts_, shift_area_or_point=True)
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

        self._meta["offset_east_px"] = offset_east if ref == "point" else -offset_east
        self._meta["offset_north_px"] = offset_north if ref == "point" else -offset_north
        self._meta["vshift"] = vshift if ref == "point" else -vshift
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

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the Nuth & Kaab shift to a DEM."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        updated_transform = apply_xy_shift(transform, -offset_east, -offset_north)
        vshift = self._meta["vshift"]
        return elev + vshift, updated_transform

    def _apply_pts(
        self,
        elev: gpd.GeoDataFrame,
        z_name: str = "z",
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:

        """Apply the Nuth & Kaab shift to a set of points."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        applied_epc = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                x=elev.geometry.x.values + offset_east, y=elev.geometry.y.values + offset_north, crs=elev.crs
            ),
            data={z_name: elev[z_name].values + self._meta["vshift"]},
        )

        return applied_epc


class GradientDescending(AffineCoreg):
    """
    Gradient Descending coregistration by Zhihao
    """

    def __init__(
        self,
        x0: tuple[float, float] = (0, 0),
        bounds: tuple[float, float] = (-3, 3),
        deltainit: int = 2,
        deltatol: float = 0.004,
        feps: float = 0.0001,
        subsample: int | float = 6000,
    ) -> None:
        """
        Instantiate gradient descending coregistration object.

        :param x0: The initial point of gradient descending iteration.
        :param bounds: The boundary of the maximum shift.
        :param deltainit: Initial pattern size.
        :param deltatol: Target pattern size, or the precision you want achieve.
        :param feps: Parameters for algorithm. Smallest difference in function value to resolve.
        :param subsample: Subsample the input for speed-up. <1 is parsed as a fraction. >1 is a pixel count.

        The algorithm terminates when the iteration is locally optimal at the target pattern size 'deltatol',
        or when the function value differs by less than the tolerance 'feps' along all directions.

        """
        self._meta: CoregDict
        self.bounds = bounds
        self.x0 = x0
        self.deltainit = deltainit
        self.deltatol = deltatol
        self.feps = feps

        super().__init__(subsample=subsample)

    def _fit_rst_pts(
        self,
        ref_elev: NDArrayf | gpd.GeoDataFrame,
        tba_elev: NDArrayf | gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Estimate the x/y/z offset between two DEMs.
        :param point_elev: the dataframe used as ref
        :param rst_elev: the dem to be aligned
        :param z_name: the column name of dataframe used for elevation differencing
        :param weights: the column name of dataframe used for weight, should have the same length with z_name columns
        :param random_state: The random state of the subsampling.
        """
        if not _has_noisyopt:
            raise ValueError("Optional dependency needed. Install 'noisyopt'")

        # Check which one is reference
        if isinstance(ref_elev, gpd.GeoDataFrame):
            point_elev = ref_elev
            rst_elev = tba_elev
            ref = "point"
        else:
            point_elev = tba_elev
            rst_elev = ref_elev
            ref = "raster"

        rst_elev = Raster.from_array(rst_elev, transform=transform, crs=crs)

        # Perform downsampling if subsample != None
        if self._meta["subsample"] and len(point_elev) > self._meta["subsample"]:
            point_elev = point_elev.sample(
                frac=self._meta["subsample"] / len(point_elev), random_state=self._meta["random_state"]
            ).copy()
        else:
            point_elev = point_elev.copy()

        bounds, resolution = _transform_to_bounds_and_res(ref_elev.shape, transform)
        # Assume that the coordinates represent the center of a theoretical pixel.
        # The raster sampling is done in the upper left corner, meaning all point have to be respectively shifted

        # TODO: Should be a way to not duplicate this column and just feed it directly
        point_elev["E"] = point_elev.geometry.x.values
        point_elev["N"] = point_elev.geometry.y.values
        point_elev["E"] -= resolution / 2
        point_elev["N"] += resolution / 2

        area_or_point = "Area"
        old_aop = rst_elev.tags.get("AREA_OR_POINT", None)
        rst_elev.tags["AREA_OR_POINT"] = area_or_point

        if verbose:
            print("Running Gradient Descending Coreg - Zhihao (in preparation) ")
            if self._meta["subsample"]:
                print("Running on downsampling. The length of the gdf:", len(point_elev))

            elevation_difference = _residuals_df(rst_elev, point_elev, (0, 0), 0, z_name=z_name)
            nmad_old = nmad(elevation_difference)
            vshift = np.nanmedian(elevation_difference)
            print("   Statistics on initial dh:")
            print(f"      Median = {vshift:.4f} - NMAD = {nmad_old:.4f}")

        # start iteration, find the best shifting px
        def func_cost(x: tuple[float, float]) -> np.floating[Any]:
            return nmad(_residuals_df(rst_elev, point_elev, x, 0, z_name=z_name))

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
        elevation_difference = _residuals_df(rst_elev, point_elev, (res.x[0], res.x[1]), 0, z_name=z_name)

        if old_aop is None:
            del rst_elev.tags["AREA_OR_POINT"]
        else:
            rst_elev.tags["AREA_OR_POINT"] = old_aop

        # results statistics
        vshift = np.nanmedian(elevation_difference)
        nmad_new = nmad(elevation_difference)

        # Print final results
        if verbose:

            print(f"\n   Final offset in pixels (east, north) : ({res.x[0]:f}, {res.x[1]:f})")
            print("   Statistics on coregistered dh:")
            print(f"      Median = {vshift:.4f} - NMAD = {nmad_new:.4f}")

        offset_east = res.x[0]
        offset_north = res.x[1]

        self._meta["offset_east_px"] = offset_east if ref == "point" else -offset_east
        self._meta["offset_north_px"] = offset_north if ref == "point" else -offset_north
        self._meta["vshift"] = vshift if ref == "point" else -vshift
        self._meta["resolution"] = resolution

    def _fit_rst_rst(
        self,
        ref_elev: NDArrayf,
        tba_elev: NDArrayf,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:

        ref_elev = (
            Raster.from_array(ref_elev, transform=transform, crs=crs, nodata=-9999.0)
            .to_points(as_array=False, pixel_offset="center")
            .ds
        )
        ref_elev["E"] = ref_elev.geometry.x
        ref_elev["N"] = ref_elev.geometry.y
        ref_elev.rename(columns={"b1": z_name}, inplace=True)
        self._fit_rst_pts(
            ref_elev=ref_elev,
            tba_elev=tba_elev,
            transform=transform,
            crs=crs,
            inlier_mask=inlier_mask,
            z_name=z_name,
            **kwargs,
        )

    def _to_matrix_func(self) -> NDArrayf:
        """Return a transformation matrix from the estimated offsets."""
        offset_east = self._meta["offset_east_px"] * self._meta["resolution"]
        offset_north = self._meta["offset_north_px"] * self._meta["resolution"]

        matrix = np.diag(np.ones(4, dtype=float))
        matrix[0, 3] += offset_east
        matrix[1, 3] += offset_north
        matrix[2, 3] += self._meta["vshift"]

        return matrix

"""Base coregistration classes to define generic methods and pre/post-processing of input data."""

from __future__ import annotations

import concurrent.futures
import copy
import warnings
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    TypedDict,
    TypeVar,
    overload,
)

import affine

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False
import fiona
import geoutils as gu
import numpy as np
import pandas as pd
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
import skimage.transform
from geoutils._typing import AnyNumber
from geoutils.raster import (
    Mask,
    RasterType,
    get_array_and_mask,
    raster,
    subdivide_array,
    subsample_array,
)
from tqdm import tqdm

from xdem._typing import MArrayf, NDArrayf
from xdem.spatialstats import nmad
from xdem.terrain import get_terrain_attribute

try:
    import pytransform3d.transformations
    from pytransform3d.transform_manager import TransformManager

    _HAS_P3D = True
except ImportError:
    _HAS_P3D = False


###########################################
# Generic functions for preprocessing
###########################################


def _transform_to_bounds_and_res(
    shape: tuple[int, ...], transform: rio.transform.Affine
) -> tuple[rio.coords.BoundingBox, float]:
    """Get the bounding box and (horizontal) resolution from a transform and the shape of a DEM."""
    bounds = rio.coords.BoundingBox(*rio.transform.array_bounds(shape[0], shape[1], transform=transform))
    resolution = (bounds.right - bounds.left) / shape[1]

    return bounds, resolution


def _get_x_and_y_coords(shape: tuple[int, ...], transform: rio.transform.Affine) -> tuple[NDArrayf, NDArrayf]:
    """Generate center coordinates from a transform and the shape of a DEM."""
    bounds, resolution = _transform_to_bounds_and_res(shape, transform)
    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, num=shape[1]),
        np.linspace(bounds.bottom + resolution / 2, bounds.top - resolution / 2, num=shape[0])[::-1],
    )
    return x_coords, y_coords


def _apply_xyz_shift_df(df: pd.DataFrame, dx: float, dy: float, dz: float, z_name: str) -> NDArrayf:
    """
    Apply shift to dataframe using Transform affine matrix

    :param df: DataFrame with columns 'E','N',z_name (height)
    :param dz: dz shift value
    """

    new_df = df.copy()
    new_df["E"] += dx
    new_df["N"] += dy
    new_df[z_name] -= dz

    return new_df


def _residuals_df(
    dem: NDArrayf,
    df: pd.DataFrame,
    shift_px: tuple[float, float],
    dz: float,
    z_name: str,
    weight: str = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Calculate the difference between the DEM and points (a dataframe has 'E','N','z') after applying a shift.

    :param dem: DEM
    :param df: A dataframe has 'E','N' and has been subseted according to DEM bonds and masks.
    :param shift_px: The coordinates of shift pixels (e_px,n_px).
    :param dz: The bias.
    :param z_name: The column that be used to compare with dem_h.
    :param weight: The column that be used as weights

    :returns: An array of residuals.
    """

    # shift ee,nn
    ee, nn = (i * dem.res[0] for i in shift_px)
    df_shifted = _apply_xyz_shift_df(df, ee, nn, dz, z_name=z_name)

    # prepare DEM
    arr_ = dem.data.astype(np.float32)

    # get residual error at the point on DEM.
    i, j = dem.xy2ij(df_shifted["E"].values, df_shifted["N"].values, op=np.float32)

    # ndimage return
    dem_h = scipy.ndimage.map_coordinates(arr_, [i, j], order=1, mode="nearest", **kwargs)
    weight_ = df[weight] if weight else 1

    return (df_shifted[z_name].values - dem_h) * weight_


def _df_sampling_from_dem(
    dem: RasterType, tba_dem: RasterType, samples: int = 10000, order: int = 1, offset: str | None = None
) -> pd.DataFrame:
    """
    Generate a dataframe from a dem by random sampling.

    :param offset: The pixel’s center is returned by default, but a corner can be returned
    by setting offset to one of ul, ur, ll, lr.

    :returns dataframe: N,E coordinates and z of DEM at sampling points.
    """
    try:
        if offset is None and (dem.tags["AREA_OR_POINT"] in ["Point", "point"]):
            offset = "center"
        else:
            offset = "ul"
    except KeyError:
        offset = "ul"

    # Avoid edge, and mask-out area in sampling
    width, length = dem.shape
    i, j = np.random.randint(10, width - 10, samples), np.random.randint(10, length - 10, samples)
    mask = dem.data.mask

    # Get value
    x, y = dem.ij2xy(i[~mask[i, j]], j[~mask[i, j]], offset=offset)
    z = scipy.ndimage.map_coordinates(
        dem.data.astype(np.float32), [i[~mask[i, j]], j[~mask[i, j]]], order=order, mode="nearest"
    )
    df = pd.DataFrame({"z": z, "N": y, "E": x})

    # mask out from tba_dem
    if tba_dem is not None:
        df, _ = _mask_dataframe_by_dem(df, tba_dem)

    return df


def _mask_dataframe_by_dem(df: pd.DataFrame | NDArrayf, dem: RasterType) -> pd.DataFrame | NDArrayf:
    """
    Mask out the dataframe (has 'E','N' columns), or np.ndarray ([E,N]) by DEM's mask.

    Return new dataframe and mask.
    """

    final_mask = ~dem.data.mask
    mask_raster = dem.copy(new_array=final_mask.astype(np.float32))

    if isinstance(df, pd.DataFrame):
        pts = np.array((df["E"].values, df["N"].values)).T
    elif isinstance(df, np.ndarray):
        pts = df

    ref_inlier = mask_raster.interp_points(pts, input_latlon=False, order=0)
    new_df = df[ref_inlier.astype(bool)].copy()

    return new_df, ref_inlier.astype(bool)


def _calculate_ddem_stats(
    ddem: NDArrayf | MArrayf,
    inlier_mask: NDArrayf | None = None,
    stats_list: tuple[Callable[[NDArrayf], AnyNumber], ...] | None = None,
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
    # Default stats - Cannot be put in default args due to circular import with xdem.spatialstats.nmad.
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


def _mask_as_array(reference_raster: gu.Raster, mask: str | gu.Vector | gu.Raster) -> NDArrayf:
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
            mask = gu.Vector(mask)
        # If the format is unsopported, try loading as a Raster
        except fiona.errors.DriverError:
            try:
                mask = gu.Raster(mask)
            # If that fails, raise an error
            except rio.errors.RasterioIOError:
                raise ValueError(f"Mask path not in a supported Raster or Vector format: {mask}")

    # At this point, the mask variable is either a Raster or a Vector
    # Now, convert the mask into an array by either rasterizing a Vector or by fetching a Raster's data
    if isinstance(mask, gu.Vector):
        mask_array = mask.create_mask(reference_raster, as_array=True)
    elif isinstance(mask, gu.Raster):
        # The true value is the maximum value in the raster, unless the maximum value is 0 or False
        true_value = np.nanmax(mask.data) if not np.nanmax(mask.data) in [0, False] else True
        mask_array = (mask.data == true_value).squeeze()
    else:
        raise TypeError(
            f"Mask has invalid type: {type(mask)}. Expected one of: " f"{[gu.Raster, gu.Vector, str, type(None)]}"
        )

    return mask_array


def _preprocess_coreg_raster_input(
    reference_dem: NDArrayf | MArrayf | RasterType,
    dem_to_be_aligned: NDArrayf | MArrayf | RasterType,
    inlier_mask: NDArrayf | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
    subsample: float | int = 1.0,
    random_state: None | np.random.RandomState | np.random.Generator | int = None,
) -> tuple[NDArrayf, NDArrayf, affine.Affine, rio.crs.CRS]:

    # Validate that both inputs are valid array-like (or Raster) types.
    if not all(isinstance(dem, (np.ndarray, gu.Raster)) for dem in (reference_dem, dem_to_be_aligned)):
        raise ValueError(
            "Both DEMs need to be array-like (implement a numpy array interface)."
            f"'reference_dem': {reference_dem}, 'dem_to_be_aligned': {dem_to_be_aligned}"
        )

    # If both DEMs are Rasters, validate that 'dem_to_be_aligned' is in the right grid. Then extract its data.
    if isinstance(dem_to_be_aligned, gu.Raster) and isinstance(reference_dem, gu.Raster):
        dem_to_be_aligned = dem_to_be_aligned.reproject(reference_dem, silent=True)

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
    # Override transform and CRS
    if new_transform is not None:
        transform = new_transform
    if new_crs is not None:
        crs = new_crs

    if transform is None:
        raise ValueError("'transform' must be given if both DEMs are array-like.")

    if crs is None:
        raise ValueError("'crs' must be given if both DEMs are array-like.")

    # Get a NaN array covering nodatas from the raster, masked array or integer-type array
    ref_dem, ref_mask = get_array_and_mask(reference_dem)
    tba_dem, tba_mask = get_array_and_mask(dem_to_be_aligned)

    # Make sure that the mask has an expected format.
    if inlier_mask is not None:
        if isinstance(inlier_mask, Mask):
            inlier_mask = inlier_mask.data.filled(False).squeeze()
        else:
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

        indices = gu.raster.subsample_array(
            ref_dem, subsample=subsample, return_indices=True, random_state=random_state
        )

        mask_subsample = np.zeros(np.shape(ref_dem), dtype=bool)
        mask_subsample[indices[0], indices[1]] = True

        ref_dem[~mask_subsample] = np.nan
        tba_dem[~mask_subsample] = np.nan

        # # TODO: Use tested subsampling function from geoutils?
        # # The full mask (inliers=True) is the inverse of the above masks and the provided mask.
        # full_mask = (
        #         ~ref_mask & ~tba_mask & (np.asarray(inlier_mask) if inlier_mask is not None else True)
        # ).squeeze()
        # random_indices = subsample_array(full_mask, subsample=subsample, return_indices=True)
        # full_mask[random_indices] = False
        #
        # # Remove the data, keep the shape
        # # TODO: there's likely a better way to go about this...
        # ref_dem[~full_mask] = np.nan
        # tba_dem[~full_mask] = np.nan

    return ref_dem, tba_dem, transform, crs


# TODO: Re-structure AffineCoreg apply function and move there?


def deramping(
    ddem: NDArrayf | MArrayf,
    x_coords: NDArrayf,
    y_coords: NDArrayf,
    degree: int,
    subsample: float | int = 1.0,
    verbose: bool = False,
) -> tuple[Callable[[NDArrayf, NDArrayf], NDArrayf], tuple[NDArrayf, int]]:
    """
    Calculate a deramping function to remove spatially correlated elevation differences that can be explained by \
    a polynomial of degree `degree`.

    :param ddem: The elevation difference array to analyse.
    :param x_coords: x-coordinates of the above array (must have the same shape as elevation_difference)
    :param y_coords: y-coordinates of the above array (must have the same shape as elevation_difference)
    :param degree: The polynomial degree to estimate the ramp.
    :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
    :param verbose: Print the least squares optimization progress.

    :returns: A callable function to estimate the ramp and the output of scipy.optimize.leastsq
    """
    # Extract only valid pixels
    valid_mask = np.isfinite(ddem)
    ddem = ddem[valid_mask]
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]

    # Formulate the 2D polynomial whose coefficients will be solved for.
    def poly2d(x_coords: NDArrayf, y_coords: NDArrayf, coefficients: NDArrayf) -> NDArrayf:
        """
        Estimate values from a 2D-polynomial.

        :param x_coords: x-coordinates of the difference array (must have the same shape as
            elevation_difference).
        :param y_coords: y-coordinates of the difference array (must have the same shape as
            elevation_difference).
        :param coefficients: The coefficients (a, b, c, etc.) of the polynomial.
        :param degree: The degree of the polynomial.

        :raises ValueError: If the length of the coefficients list is not compatible with the degree.

        :returns: The values estimated by the polynomial.
        """
        # Check that the coefficient size is correct.
        coefficient_size = (degree + 1) * (degree + 2) / 2
        if len(coefficients) != coefficient_size:
            raise ValueError()

        # Build the polynomial of degree `degree`
        estimated_values = np.sum(
            [
                coefficients[k * (k + 1) // 2 + j] * x_coords ** (k - j) * y_coords**j
                for k in range(degree + 1)
                for j in range(k + 1)
            ],
            axis=0,
        )
        return estimated_values  # type: ignore

    def residuals(coefs: NDArrayf, x_coords: NDArrayf, y_coords: NDArrayf, targets: NDArrayf) -> NDArrayf:
        """Return the optimization residuals"""
        res = targets - poly2d(x_coords, y_coords, coefs)
        return res[np.isfinite(res)]

    if verbose:
        print("Estimating deramp function...")

    # reduce number of elements for speed
    rand_indices = subsample_array(x_coords, subsample=subsample, return_indices=True)
    x_coords = x_coords[rand_indices]
    y_coords = y_coords[rand_indices]
    ddem = ddem[rand_indices]

    # Optimize polynomial parameters
    coefs = scipy.optimize.leastsq(
        func=residuals,
        x0=np.zeros(shape=((degree + 1) * (degree + 2) // 2)),
        args=(x_coords, y_coords, ddem),
    )

    def fit_ramp(x: NDArrayf, y: NDArrayf) -> NDArrayf:
        """
        Get the elevation difference biases (ramp) at the given coordinates.

        :param x_coordinates: x-coordinates of interest.
        :param y_coordinates: y-coordinates of interest.

        :returns: The estimated elevation difference bias.
        """
        return poly2d(x, y, coefs[0])

    return fit_ramp, coefs


def invert_matrix(matrix: NDArrayf) -> NDArrayf:
    """Invert a transformation matrix."""
    with warnings.catch_warnings():
        # Deprecation warning from pytransform3d. Let's hope that is fixed in the near future.
        warnings.filterwarnings("ignore", message="`np.float` is a deprecated alias for the builtin `float`")

        checked_matrix = pytransform3d.transformations.check_matrix(matrix)
        # Invert the transform if wanted.
        return pytransform3d.transformations.invert_transform(checked_matrix)


def apply_matrix(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: int | str = "bilinear",
    fill_max_search: int = 0,
) -> NDArrayf:
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
    :param fill_max_search: Set to > 0 value to fill the DEM before applying the transformation, to avoid spreading\
    gaps. The DEM will be filled with rasterio.fill.fillnodata with max_search_distance set to fill_max_search.\
    This is experimental, use at your own risk !

    :returns: The transformed DEM with NaNs as nodata values (replaces a potential mask of the input `dem`).
    """
    # Parse the resampling argument given.
    if isinstance(resampling, (int, np.integer)):
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

    # Check if the matrix only contains a Z correction. In that case, only shift the DEM values by the vertical shift.
    empty_matrix = np.diag(np.ones(4, float))
    empty_matrix[2, 3] = matrix[2, 3]
    if np.mean(np.abs(empty_matrix - matrix)) == 0.0:
        return demc + matrix[2, 3]

    # Opencv is required down from here
    if not _has_cv2:
        raise ValueError("Optional dependency needed. Install 'opencv'")

    nan_mask = ~np.isfinite(dem)
    assert np.count_nonzero(~nan_mask) > 0, "Given DEM had all nans."
    # Optionally, fill DEM around gaps to reduce spread of gaps
    if fill_max_search > 0:
        filled_dem = rio.fill.fillnodata(demc, mask=(~nan_mask).astype("uint8"), max_search_distance=fill_max_search)
    else:
        filled_dem = demc  # np.where(~nan_mask, demc, np.nan)  # I don't know why this was needed - to delete

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
    deramp, coeffs = deramping(
        (point_cloud[:, :, 2] - transformed_points[:, :, 2])[~nan_mask].flatten(),
        point_cloud[:, :, 0][~nan_mask].flatten(),
        point_cloud[:, :, 1][~nan_mask].flatten(),
        degree=1,
    )
    # Shift the elevation values of the soon-to-be-warped DEM.
    filled_dem -= deramp(x_coords, y_coords)

    # Create arrays of x and y coordinates to be converted into index coordinates.
    x_inds = transformed_points[:, :, 0].copy()
    x_inds[x_inds == 0] = np.nan
    y_inds = transformed_points[:, :, 1].copy()
    y_inds[y_inds == 0] = np.nan

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
            filled_dem, inds, order=resampling_order, mode="constant", cval=np.nan, preserve_range=True
        )

    assert np.count_nonzero(~np.isnan(transformed_dem)) > 0, "Transformed DEM has all nans."

    return transformed_dem


###########################################
# Generic coregistration processing classes
###########################################


class CoregDict(TypedDict, total=False):
    """
    Defining the type of each possible key in the metadata dictionary of Process classes.
    The parameter total=False means that the key are not required. In the recent PEP 655 (
    https://peps.python.org/pep-0655/) there is an easy way to specific Required or NotRequired for each key, if we
    want to change this in the future.
    """

    # TODO: homogenize the naming mess!
    vshift_func: Callable[[NDArrayf], np.floating[Any]]
    func: Callable[[NDArrayf, NDArrayf], NDArrayf]
    vshift: np.floating[Any] | float | np.integer[Any] | int
    matrix: NDArrayf
    centroid: tuple[float, float, float]
    offset_east_px: float
    offset_north_px: float
    coefficients: NDArrayf
    step_meta: list[Any]
    resolution: float
    nmad: np.floating[Any]

    # The pipeline metadata can have any value of the above
    pipeline: list[Any]

    # BiasCorr classes generic metadata

    # 1/ Inputs
    fit_or_bin: Literal["fit"] | Literal["bin"]
    fit_func: Callable[..., NDArrayf]
    fit_optimizer: Callable[..., tuple[NDArrayf, Any]]
    bin_sizes: int | dict[str, int | Iterable[float]]
    bin_statistic: Callable[[NDArrayf], np.floating[Any]]
    bin_apply_method: Literal["linear"] | Literal["per_bin"]

    # 2/ Outputs
    bias_vars: list[str]
    fit_params: NDArrayf
    fit_perr: NDArrayf
    bin_dataframe: pd.DataFrame

    # 3/ Specific inputs or outputs
    terrain_attribute: str
    angle: float
    poly_order: int
    nb_sin_freq: int


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

    def __init__(self, meta: CoregDict | None = None) -> None:
        """Instantiate a generic processing step method."""
        self._meta: CoregDict = meta or {}  # All __init__ functions should instantiate an empty dict.

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.copy(value) for key, value in self.__dict__.items()}

        return new_coreg

    def __add__(self, other: CoregType) -> CoregPipeline:
        """Return a pipeline consisting of self and the other processing function."""
        if not isinstance(other, Coreg):
            raise ValueError(f"Incompatible add type: {type(other)}. Expected 'Coreg' subclass")
        return CoregPipeline([self, other])

    def fit(
        self: CoregType,
        reference_dem: NDArrayf | MArrayf | RasterType,
        dem_to_be_aligned: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayf | Mask | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int = 1.0,
        verbose: bool = False,
        random_state: None | np.random.RandomState | np.random.Generator | int = None,
        **kwargs: Any,
    ) -> CoregType:
        """
        Estimate the coregistration transform on the given DEMs.

        :param reference_dem: 2D array of elevation values acting reference.
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory if DEM provided as array.
        :param crs: Optional. CRS of the reference_dem. Mandatory if DEM provided as array.
        :param weights: Optional. Per-pixel weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param verbose: Print progress messages to stdout.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)
        """

        if weights is not None:
            raise NotImplementedError("Weights have not yet been implemented")

        # Pre-process the inputs, by reprojecting and subsampling
        ref_dem, tba_dem, transform, crs = _preprocess_coreg_raster_input(
            reference_dem=reference_dem,
            dem_to_be_aligned=dem_to_be_aligned,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            subsample=subsample,
            random_state=random_state,
        )

        # Run the associated fitting function
        self._fit_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            transform=transform,
            crs=crs,
            weights=weights,
            verbose=verbose,
            random_state=random_state,
            **kwargs,
        )

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    def residuals(
        self,
        reference_dem: NDArrayf,
        dem_to_be_aligned: NDArrayf,
        inlier_mask: NDArrayf | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        subsample: float | int = 1.0,
        random_state: None | np.random.RandomState | np.random.Generator | int = None,
    ) -> NDArrayf:
        """
        Calculate the residual offsets (the difference) between two DEMs after applying the transformation.

        :param reference_dem: 2D array of elevation values acting reference.
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

        :returns: A 1D array of finite residuals.
        """

        # Apply the transformation to the dem to be aligned
        aligned_dem = self.apply(dem_to_be_aligned, transform=transform, crs=crs)[0]

        # Pre-process the inputs, by reprojecting and subsampling
        ref_dem, align_dem, transform, crs = _preprocess_coreg_raster_input(
            reference_dem=reference_dem,
            dem_to_be_aligned=aligned_dem,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
            subsample=subsample,
            random_state=random_state,
        )

        # Calculate the DEM difference
        diff = ref_dem - align_dem

        # Sometimes, the float minimum (for float32 = -3.4028235e+38) is returned. This and inf should be excluded.
        full_mask = np.isfinite(diff)
        if "float" in str(diff.dtype):
            full_mask[(diff == np.finfo(diff.dtype).min) | np.isinf(diff)] = False

        # Return the difference values within the full inlier mask
        return diff[full_mask]

    def fit_pts(
        self: CoregType,
        reference_dem: NDArrayf | MArrayf | RasterType | pd.DataFrame,
        dem_to_be_aligned: RasterType,
        inlier_mask: NDArrayf | Mask | None = None,
        transform: rio.transform.Affine | None = None,
        samples: int = 10000,
        subsample: float | int = 1.0,
        verbose: bool = False,
        mask_high_curv: bool = False,
        order: int = 1,
        z_name: str = "z",
        weights: str | None = None,
    ) -> CoregType:
        """
        Estimate the coregistration transform between a DEM and a reference point elevation data.

        :param reference_dem: Point elevation data acting reference.
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param verbose: Print progress messages to stdout.
        :param order: interpolation 0=nearest, 1=linear, 2=cubic.
        :param z_name: the column name of dataframe used for elevation differencing
        :param weights: the column name of dataframe used for weight, should have the same length with z_name columns
        """

        # Validate that at least one input is a valid array-like (or Raster) types.
        if not isinstance(dem_to_be_aligned, (np.ndarray, gu.Raster)):
            raise ValueError(
                "The dem_to_be_aligned needs to be array-like (implement a numpy array interface)."
                f"'dem_to_be_aligned': {dem_to_be_aligned}"
            )

        # DEM to dataframe if ref_dem is raster
        # How to make sure sample point locates in stable terrain?
        if isinstance(reference_dem, (np.ndarray, gu.Raster)):
            reference_dem = _df_sampling_from_dem(
                reference_dem, dem_to_be_aligned, samples=samples, order=1, offset=None
            )

        # Validate that at least one input is a valid point data type.
        if not isinstance(reference_dem, pd.DataFrame):
            raise ValueError(
                "The reference_dem needs to be point data format (pd.Dataframe)." f"'reference_dem': {reference_dem}"
            )

        # If any input is a Raster, use its transform if 'transform is None'.
        # If 'transform' was given and any input is a Raster, trigger a warning.
        # Finally, extract only the data of the raster.
        for name, dem in [("dem_to_be_aligned", dem_to_be_aligned)]:
            if hasattr(dem, "transform"):
                if transform is None:
                    transform = dem.transform
                elif transform is not None:
                    warnings.warn(f"'{name}' of type {type(dem)} overrides the given 'transform'")

        if transform is None:
            raise ValueError("'transform' must be given if the dem_to_be_align DEM is array-like.")

        _, tba_mask = get_array_and_mask(dem_to_be_aligned)

        if np.all(tba_mask):
            raise ValueError("'dem_to_be_aligned' had only NaNs")

        tba_dem = dem_to_be_aligned.copy()
        ref_valid = np.isfinite(reference_dem["z"].values)

        if np.all(~ref_valid):
            raise ValueError("'reference_dem' point data only contains NaNs")

        ref_dem = reference_dem[ref_valid]

        if mask_high_curv:
            maxc = np.maximum(
                np.abs(get_terrain_attribute(tba_dem, attribute=["planform_curvature", "profile_curvature"])), axis=0
            )
            # Mask very high curvatures to avoid resolution biases
            mask_hc = maxc.data > 5.0
        else:
            mask_hc = np.zeros(tba_dem.data.mask.shape, dtype=bool)
            if "planc" in ref_dem.columns and "profc" in ref_dem.columns:
                ref_dem = ref_dem.query("planc < 5 and profc < 5")
            else:
                print("Warning: There is no curvature in dataframe. Set mask_high_curv=True for more robust results")

        points = np.array((ref_dem["E"].values, ref_dem["N"].values)).T

        # Make sure that the mask has an expected format.
        if inlier_mask is not None:
            if isinstance(inlier_mask, Mask):
                inlier_mask = inlier_mask.data.filled(False).squeeze()
            else:
                inlier_mask = np.asarray(inlier_mask).squeeze()
                assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

            if np.all(~inlier_mask):
                raise ValueError("'inlier_mask' had no inliers.")

            final_mask = np.logical_and.reduce((~tba_dem.data.mask, inlier_mask, ~mask_hc))
        else:
            final_mask = np.logical_and(~tba_dem.data.mask, ~mask_hc)

        mask_raster = tba_dem.copy(new_array=final_mask.astype(np.float32))

        ref_inlier = mask_raster.interp_points(points, order=0)
        ref_inlier = ref_inlier.astype(bool)

        if np.all(~ref_inlier):
            raise ValueError("Intersection of 'reference_dem' and 'dem_to_be_aligned' had only NaNs")

        ref_dem = ref_dem[ref_inlier]

        # If subsample is not equal to one, subsampling should be performed.
        if subsample != 1.0:

            # Randomly pick N inliers in the full_mask where N=subsample
            random_valids = subsample_array(ref_dem["z"].values, subsample=subsample, return_indices=True)

            # Subset to the N random inliers
            ref_dem = ref_dem.iloc[random_valids]

        # Run the associated fitting function
        self._fit_pts_func(
            ref_dem=ref_dem,
            tba_dem=tba_dem,
            transform=transform,
            weights=weights,
            verbose=verbose,
            order=order,
            z_name=z_name,
        )

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        dem: MArrayf,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        resample: bool = True,
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        dem: NDArrayf,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        resample: bool = True,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        dem: RasterType,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        resample: bool = True,
        **kwargs: Any,
    ) -> RasterType:
        ...

    def apply(
        self,
        dem: RasterType | NDArrayf | MArrayf,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        resample: bool = True,
        **kwargs: Any,
    ) -> RasterType | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:
        """
        Apply the estimated transform to a DEM.

        :param dem: A DEM array or Raster to apply the transform on.
        :param transform: Optional. The transform object of the DEM. Mandatory if 'dem' provided as array.
        :param crs: Optional. CRS of the reference_dem. Mandatory if 'dem' provided as array.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
        only the transform might be updated and no resampling is done.
        :param kwargs: Any optional arguments to be passed to either self._apply_func or apply_matrix.
        Kwarg `resampling` can be set to any rio.warp.Resampling to use a different resampling in case \
        `resample` is True, default is bilinear.

        :returns: The transformed DEM.
        """
        if not self._fit_called and self._meta.get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        if isinstance(dem, gu.Raster):
            if transform is None:
                transform = dem.transform
            else:
                warnings.warn(f"DEM of type {type(dem)} overrides the given 'transform'")
            if crs is None:
                crs = dem.crs
            else:
                warnings.warn(f"DEM of type {type(dem)} overrides the given 'crs'")

        else:
            if transform is None:
                raise ValueError("'transform' must be given if DEM is array-like.")
            if crs is None:
                raise ValueError("'crs' must be given if DEM is array-like.")

        # The array to provide the functions will be an ndarray with NaNs for masked out areas.
        dem_array, dem_mask = get_array_and_mask(dem)

        if np.all(dem_mask):
            raise ValueError("'dem' had only NaNs")

        # See if a _apply_func exists
        try:
            # arg `resample` must be passed to _apply_func, otherwise will be overwritten in CoregPipeline
            kwargs["resample"] = resample

            # Run the associated apply function
            applied_dem, out_transform = self._apply_func(
                dem_array, transform, crs, **kwargs
            )  # pylint: disable=assignment-from-no-return

        # If it doesn't exist, use apply_matrix()
        except NotImplementedError:

            # In this case, resampling is necessary
            if not resample:
                raise NotImplementedError(f"Option `resample=False` not implemented for coreg method {self.__class__}")
            kwargs.pop("resample")  # Need to removed before passing to apply_matrix

            if self.is_affine:  # This only works on it's affine, however.

                # Apply the matrix around the centroid (if defined, otherwise just from the center).
                applied_dem = apply_matrix(
                    dem_array,
                    transform=transform,
                    matrix=self.to_matrix(),
                    centroid=self._meta.get("centroid"),
                    **kwargs,
                )
                out_transform = transform
            else:
                raise ValueError("Coreg method is non-rigid but has no implemented _apply_func")

        # Ensure the dtype is OK
        applied_dem = applied_dem.astype("float32")

        # Set default dst_nodata
        if isinstance(dem, gu.Raster):
            dst_nodata = dem.nodata
        else:
            dst_nodata = raster._default_nodata(applied_dem.dtype)

        # Resample the array on the original grid
        if resample:
            # Set default resampling method if not specified in kwargs
            resampling = kwargs.get("resampling", rio.warp.Resampling.bilinear)
            if not isinstance(resampling, rio.warp.Resampling):
                raise ValueError("`resampling` must be a rio.warp.Resampling algorithm")

            applied_dem, out_transform = rio.warp.reproject(
                applied_dem,
                destination=applied_dem,
                src_transform=out_transform,
                dst_transform=transform,
                src_crs=crs,
                dst_crs=crs,
                resampling=resampling,
                dst_nodata=dst_nodata,
            )

        # Calculate final mask
        final_mask = np.logical_or(~np.isfinite(applied_dem), applied_dem == dst_nodata)

        # If the DEM was a masked_array, copy the mask to the new DEM
        if isinstance(dem, (np.ma.masked_array, gu.Raster)):
            applied_dem = np.ma.masked_array(applied_dem, mask=final_mask)  # type: ignore
        else:
            applied_dem[final_mask] = np.nan

        # If the input was a Raster, returns a Raster, else returns array and transform
        if isinstance(dem, gu.Raster):
            out_dem = dem.from_array(applied_dem, out_transform, crs, nodata=dem.nodata)
            return out_dem
        else:
            return applied_dem, out_transform

    def apply_pts(self, coords: NDArrayf) -> NDArrayf:
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

        assert (
            len(np.shape(coords)) == 2 and np.shape(coords)[1] == 3
        ), f"'coords' shape must be (N, 3). Given shape: {np.shape(coords)}"

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

    @overload
    def error(
        self,
        reference_dem: NDArrayf,
        dem_to_be_aligned: NDArrayf,
        error_type: list[str],
        inlier_mask: NDArrayf | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
    ) -> list[np.floating[Any] | float | np.integer[Any] | int]:
        ...

    @overload
    def error(
        self,
        reference_dem: NDArrayf,
        dem_to_be_aligned: NDArrayf,
        error_type: str = "nmad",
        inlier_mask: NDArrayf | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
    ) -> np.floating[Any] | float | np.integer[Any] | int:
        ...

    def error(
        self,
        reference_dem: NDArrayf,
        dem_to_be_aligned: NDArrayf,
        error_type: str | list[str] = "nmad",
        inlier_mask: NDArrayf | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
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

        :param reference_dem: 2D array of elevation values acting reference.
        :param dem_to_be_aligned: 2D array of elevation values to be aligned.
        :param error_type: The type of error measure to calculate. May be a list of error types.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.

        :returns: The error measure of choice for the residuals.
        """
        if isinstance(error_type, str):
            error_type = [error_type]

        residuals = self.residuals(
            reference_dem=reference_dem,
            dem_to_be_aligned=dem_to_be_aligned,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
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

    def copy(self: CoregType) -> CoregType:
        """Return an identical copy of the class."""
        new_coreg = self.__new__(type(self))

        new_coreg.__dict__ = {key: copy.copy(value) for key, value in self.__dict__.items() if key != "pipeline"}
        new_coreg.pipeline = [step.copy() for step in self.pipeline]

        return new_coreg

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
        """Fit each processing step with the previously transformed DEM."""
        tba_dem_mod = tba_dem.copy()

        for i, coreg in enumerate(self.pipeline):
            if verbose:
                print(f"Running pipeline step: {i + 1} / {len(self.pipeline)}")
            coreg._fit_func(ref_dem, tba_dem_mod, transform=transform, crs=crs, weights=weights, verbose=verbose)
            coreg._fit_called = True

            tba_dem_mod, out_transform = coreg.apply(tba_dem_mod, transform, crs)

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        """Apply the coregistration steps sequentially to a DEM."""
        dem_mod = dem.copy()
        out_transform = copy.copy(transform)
        for coreg in self.pipeline:
            dem_mod, out_transform = coreg.apply(dem_mod, out_transform, crs, **kwargs)

        return dem_mod, out_transform

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
        """Apply the coregistration steps sequentially to a set of points."""
        coords_mod = coords.copy()

        for coreg in self.pipeline:
            coords_mod = coreg.apply_pts(coords_mod).reshape(coords_mod.shape)

        return coords_mod

    def __iter__(self) -> Generator[Coreg, None, None]:
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


class BlockwiseCoreg(Coreg):
    """
    Block-wise co-registration processing class to run a step in segmented parts of the grid.

    A processing class of choice is run on an arbitrary subdivision of the raster. When later applying the step
    the optimal warping is interpolated based on X/Y/Z shifts from the coreg algorithm at the grid points.

    For instance: a subdivision of 4 triggers a division of the DEM in four equally sized parts. These parts are then
    processed separately, with 4 .fit() results. If the subdivision is not divisible by the raster shape,
    subdivision is made as good as possible to have approximately equal pixel counts.
    """

    def __init__(
        self,
        step: Coreg | CoregPipeline,
        subdivision: int,
        success_threshold: float = 0.8,
        n_threads: int | None = None,
        warn_failures: bool = False,
    ) -> None:
        """
        Instantiate a blockwise processing object.

        :param step: An instantiated co-registration step object to fit in the subdivided DEMs.
        :param subdivision: The number of chunks to divide the DEMs in. E.g. 4 means four different transforms.
        :param success_threshold: Raise an error if fewer chunks than the fraction failed for any reason.
        :param n_threads: The maximum amount of threads to use. Default=auto
        :param warn_failures: Trigger or ignore warnings for each exception/warning in each block.
        """
        if isinstance(step, type):
            raise ValueError(
                "The 'step' argument must be an instantiated Coreg subclass. " "Hint: write e.g. ICP() instead of ICP"
            )
        self.procstep = step
        self.subdivision = subdivision
        self.success_threshold = success_threshold
        self.n_threads = n_threads
        self.warn_failures = warn_failures

        super().__init__()

        self._meta: CoregDict = {"step_meta": []}

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
        """Fit the coreg approach for each subdivision."""

        groups = self.subdivide_array(tba_dem.shape)

        indices = np.unique(groups)

        progress_bar = tqdm(total=indices.size, desc="Processing chunks", disable=(not verbose))

        def process(i: int) -> dict[str, Any] | BaseException | None:
            """
            Process a chunk in a thread-safe way.

            :returns:
                * If it succeeds: A dictionary of the fitting metadata.
                * If it fails: The associated exception.
                * If the block is empty: None
            """
            inlier_mask = groups == i

            # Find the corresponding slice of the inlier_mask to subset the data
            rows, cols = np.where(inlier_mask)
            arrayslice = np.s_[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

            # Copy a subset of the two DEMs, the mask, the coreg instance, and make a new subset transform
            ref_subset = ref_dem[arrayslice].copy()
            tba_subset = tba_dem[arrayslice].copy()

            if any(np.all(~np.isfinite(dem)) for dem in (ref_subset, tba_subset)):
                return None
            mask_subset = inlier_mask[arrayslice].copy()
            west, top = rio.transform.xy(transform, min(rows), min(cols), offset="ul")
            transform_subset = rio.transform.from_origin(west, top, transform.a, -transform.e)
            procstep = self.procstep.copy()

            # Try to run the coregistration. If it fails for any reason, skip it and save the exception.
            try:
                procstep.fit(
                    reference_dem=ref_subset,
                    dem_to_be_aligned=tba_subset,
                    transform=transform_subset,
                    inlier_mask=mask_subset,
                    crs=crs,
                )
                nmad, median = procstep.error(
                    reference_dem=ref_subset,
                    dem_to_be_aligned=tba_subset,
                    error_type=["nmad", "median"],
                    inlier_mask=mask_subset,
                    transform=transform_subset,
                    crs=crs,
                )
            except Exception as exception:
                return exception

            meta: dict[str, Any] = {
                "i": i,
                "transform": transform_subset,
                "inlier_count": np.count_nonzero(mask_subset & np.isfinite(ref_subset) & np.isfinite(tba_subset)),
                "nmad": nmad,
                "median": median,
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
            meta["representative_x"], meta["representative_y"] = rio.transform.xy(
                transform_subset, representative_row, representative_col
            )

            repr_val = ref_subset[representative_row, representative_col]
            if ~np.isfinite(repr_val):
                repr_val = 0
            meta["representative_val"] = repr_val

            # If the coreg is a pipeline, copy its metadatas to the output meta
            if hasattr(procstep, "pipeline"):
                meta["pipeline"] = [step._meta.copy() for step in procstep.pipeline]

            # Copy all current metadata (except for the already existing keys like "i", "min_row", etc, and the
            # "coreg_meta" key)
            # This can then be iteratively restored when the apply function should be called.
            meta.update(
                {key: value for key, value in procstep._meta.items() if key not in ["step_meta"] + list(meta.keys())}
            )

            progress_bar.update()

            return meta.copy()

        # Catch warnings; only show them if
        exceptions: list[BaseException | warnings.WarningMessage] = []
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("default")
            with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
                results = executor.map(process, indices)

            exceptions += list(caught_warnings)

        empty_blocks = 0
        for result in results:
            if isinstance(result, BaseException):
                exceptions.append(result)
            elif result is None:
                empty_blocks += 1
                continue
            else:
                self._meta["step_meta"].append(result)

        progress_bar.close()

        # Stop if the success rate was below the threshold
        if ((len(self._meta["step_meta"]) + empty_blocks) / self.subdivision) <= self.success_threshold:
            raise ValueError(
                f"Fitting failed for {len(exceptions)} chunks:\n"
                + "\n".join(map(str, exceptions[:5]))
                + f"\n... and {len(exceptions) - 5} more"
                if len(exceptions) > 5
                else ""
            )

        if self.warn_failures:
            for exception in exceptions:
                warnings.warn(str(exception))

        # Set the _fit_called parameters (only identical copies of self.coreg have actually been called)
        self.procstep._fit_called = True
        if isinstance(self.procstep, CoregPipeline):
            for step in self.procstep.pipeline:
                step._fit_called = True

    def _restore_metadata(self, meta: CoregDict) -> None:
        """
        Given some metadata, set it in the right place.

        :param meta: A metadata file to update self._meta
        """
        self.procstep._meta.update(meta)

        if isinstance(self.procstep, CoregPipeline) and "pipeline" in meta:
            for i, step in enumerate(self.procstep.pipeline):
                step._meta.update(meta["pipeline"][i])

    def to_points(self) -> NDArrayf:
        """
        Convert the blockwise coregistration matrices to 3D (source -> destination) points.

        The returned shape is (N, 3, 2) where the dimensions represent:
            0. The point index where N is equal to the amount of subdivisions.
            1. The X/Y/Z coordinate of the point.
            2. The old/new position of the point.

        To acquire the first point's original position: points[0, :, 0]
        To acquire the first point's new position: points[0, :, 1]
        To acquire the first point's Z difference: points[0, 2, 1] - points[0, 2, 0]

        :returns: An array of 3D source -> destination points.
        """
        if len(self._meta["step_meta"]) == 0:
            raise AssertionError("No coreg results exist. Has '.fit()' been called?")
        points = np.empty(shape=(0, 3, 2))
        for meta in self._meta["step_meta"]:
            self._restore_metadata(meta)

            # x_coord, y_coord = rio.transform.xy(meta["transform"], meta["representative_row"],
            # meta["representative_col"])
            x_coord, y_coord = meta["representative_x"], meta["representative_y"]

            old_position = np.reshape([x_coord, y_coord, meta["representative_val"]], (1, 3))
            new_position = self.procstep.apply_pts(old_position)

            points = np.append(points, np.dstack((old_position, new_position)), axis=0)

        return points

    def stats(self) -> pd.DataFrame:
        """
        Return statistics for each chunk in the blockwise coregistration.

            * center_{x,y,z}: The center coordinate of the chunk in georeferenced units.
            * {x,y,z}_off: The calculated offset in georeferenced units.
            * inlier_count: The number of pixels that were inliers in the chunk.
            * nmad: The NMAD of elevation differences (robust dispersion) after coregistration.
            * median: The median of elevation differences (vertical shift) after coregistration.

        :raises ValueError: If no coregistration results exist yet.

        :returns: A dataframe of statistics for each chunk.
        """
        points = self.to_points()

        chunk_meta = {meta["i"]: meta for meta in self._meta["step_meta"]}

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
                    "median": chunk_meta[i]["median"],
                }
            )

        stats_df = pd.DataFrame(statistics)
        stats_df.index.name = "chunk"

        return stats_df

    def subdivide_array(self, shape: tuple[int, ...]) -> NDArrayf:
        """
        Return the grid subdivision for a given DEM shape.

        :param shape: The shape of the input DEM.

        :returns: An array of shape 'shape' with 'self.subdivision' unique indices.
        """
        if len(shape) == 3 and shape[0] == 1:  # Account for (1, row, col) shapes
            shape = (shape[1], shape[2])
        return subdivide_array(shape, count=self.subdivision)

    def _apply_func(
        self, dem: NDArrayf, transform: rio.transform.Affine, crs: rio.crs.CRS, **kwargs: Any
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if np.count_nonzero(np.isfinite(dem)) == 0:
            return dem, transform

        # Other option than resample=True is not implemented for this case
        if "resample" in kwargs and kwargs["resample"] is not True:
            raise NotImplementedError()

        points = self.to_points()

        bounds, resolution = _transform_to_bounds_and_res(dem.shape, transform)

        representative_height = np.nanmean(dem)
        edges_source = np.array(
            [
                [bounds.left + resolution / 2, bounds.top - resolution / 2, representative_height],
                [bounds.right - resolution / 2, bounds.top - resolution / 2, representative_height],
                [bounds.left + resolution / 2, bounds.bottom + resolution / 2, representative_height],
                [bounds.right - resolution / 2, bounds.bottom + resolution / 2, representative_height],
            ]
        )
        edges_dest = self.apply_pts(edges_source)
        edges = np.dstack((edges_source, edges_dest))

        all_points = np.append(points, edges, axis=0)

        warped_dem = warp_dem(
            dem=dem,
            transform=transform,
            source_coords=all_points[:, :, 0],
            destination_coords=all_points[:, :, 1],
            resampling="linear",
        )

        return warped_dem, transform

    def _apply_pts_func(self, coords: NDArrayf) -> NDArrayf:
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
    dem: NDArrayf,
    transform: rio.transform.Affine,
    source_coords: NDArrayf,
    destination_coords: NDArrayf,
    resampling: str = "cubic",
    trim_border: bool = True,
    dilate_mask: bool = True,
) -> NDArrayf:
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

    dem_arr, dem_mask = get_array_and_mask(dem)

    bounds, resolution = _transform_to_bounds_and_res(dem_arr.shape, transform)

    no_horizontal = np.sum(np.linalg.norm(destination_coords[:, :2] - source_coords[:, :2], axis=1)) < 1e-6
    no_vertical = source_coords.shape[1] > 2 and np.sum(np.abs(destination_coords[:, 2] - source_coords[:, 2])) < 1e-6

    if no_horizontal and no_vertical:
        warnings.warn("No difference between source and destination coordinates. Returning self.")
        return dem

    source_coords_scaled = source_coords.copy()
    destination_coords_scaled = destination_coords.copy()
    # Scale the coordinates to index-space
    for coords in (source_coords_scaled, destination_coords_scaled):
        coords[:, 0] = dem_arr.shape[1] * (coords[:, 0] - bounds.left) / (bounds.right - bounds.left)
        coords[:, 1] = dem_arr.shape[0] * (1 - (coords[:, 1] - bounds.bottom) / (bounds.top - bounds.bottom))

    # Generate a grid of x and y index coordinates.
    grid_y, grid_x = np.mgrid[0 : dem_arr.shape[0], 0 : dem_arr.shape[1]]

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
            method="linear",
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
                cval=np.nan,
            )
            new_mask = (
                skimage.transform.warp(
                    image=dem_mask, inverse_map=np.moveaxis(new_indices, 2, 0), output_shape=dem_arr.shape, cval=False
                )
                > 0
            )

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
            fill_value=np.nan,
        )
        if not trim_border:
            grid_offsets[np.isnan(grid_offsets)] = np.nanmean(grid_offsets)

        warped += grid_offsets

    assert not np.all(np.isnan(warped)), "All-NaN output."

    return warped.reshape(dem.shape)

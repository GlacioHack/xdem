"""Base coregistration classes to define generic methods and pre/post-processing of input data."""

from __future__ import annotations

import concurrent.futures
import copy
import inspect
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
import geopandas as gpd
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
from geoutils._typing import Number
from geoutils.misc import resampling_method_from_str
from geoutils.raster import (
    Mask,
    RasterType,
    get_array_and_mask,
    raster,
    subdivide_array,
    subsample_array,
)
from tqdm import tqdm

from xdem._typing import MArrayf, NDArrayb, NDArrayf
from xdem.dem import DEM
from xdem.spatialstats import nmad

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
    weight_name: str = None,
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
    :param area_or_point: Use the GDAL Area or Point sampling method.

    :returns: An array of residuals.
    """

    # shift ee,nn
    ee, nn = (i * dem.res[0] for i in shift_px)
    df_shifted = _apply_xyz_shift_df(df, ee, nn, dz, z_name=z_name)

    # prepare DEM
    arr_ = dem.data.astype(np.float32)

    # get residual error at the point on DEM.
    i, j = dem.xy2ij(df_shifted["E"].values, df_shifted["N"].values)

    # ndimage return
    dem_h = scipy.ndimage.map_coordinates(arr_, [i, j], order=1, mode="nearest", **kwargs)
    weight_ = df[weight_name] if weight_name else 1

    return (df_shifted[z_name].values - dem_h) * weight_


def _df_sampling_from_dem(
    dem: RasterType, tba_dem: RasterType, subsample: float | int = 10000, order: int = 1, offset: str | None = None
) -> pd.DataFrame:
    """
    Generate a dataframe from a dem by random sampling.

    :param offset: The pixel’s center is returned by default, but a corner can be returned
    by setting offset to one of ul, ur, ll, lr.

    :returns dataframe: N,E coordinates and z of DEM at sampling points.
    """

    if offset is None:
        if dem.tags.get("AREA_OR_POINT", "").lower() == "area":
            offset = "ul"
        else:
            offset = "center"

    # Convert subsample to int
    valid_mask = np.logical_and(~dem.mask, ~tba_dem.mask)
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * np.count_nonzero(valid_mask))
    elif subsample > 1:
        npoints = int(subsample)
    else:
        raise ValueError("Argument `subsample` must be > 0.")

    # Avoid edge, and mask-out area in sampling
    width, length = dem.shape
    rng = np.random.default_rng()
    i, j = rng.integers(10, width - 10, npoints), rng.integers(10, length - 10, npoints)
    mask = dem.data.mask

    # Get value
    x, y = dem.ij2xy(i[~mask[i, j]], j[~mask[i, j]])
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

    ref_inlier = mask_raster.interp_points(pts)
    new_df = df[ref_inlier.astype(bool)].copy()

    return new_df, ref_inlier.astype(bool)


def _calculate_ddem_stats(
    ddem: NDArrayf | MArrayf,
    inlier_mask: NDArrayb | None = None,
    stats_list: tuple[Callable[[NDArrayf], Number], ...] | None = None,
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


def _preprocess_coreg_fit_raster_raster(
    reference_dem: NDArrayf | MArrayf | RasterType,
    dem_to_be_aligned: NDArrayf | MArrayf | RasterType,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
) -> tuple[NDArrayf, NDArrayf, NDArrayb, affine.Affine, rio.crs.CRS]:
    """Pre-processing and checks of fit() for two raster input."""

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
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        ref_dem, ref_mask = get_array_and_mask(reference_dem, copy=True)
        tba_dem, tba_mask = get_array_and_mask(dem_to_be_aligned, copy=True)

    # Make sure that the mask has an expected format.
    if inlier_mask is not None:
        if isinstance(inlier_mask, Mask):
            inlier_mask = inlier_mask.data.filled(False).squeeze()
        else:
            inlier_mask = np.asarray(inlier_mask).squeeze()
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

        if np.all(~inlier_mask):
            raise ValueError("'inlier_mask' had no inliers.")
    else:
        inlier_mask = np.ones(np.shape(ref_dem), dtype=bool)

    if np.all(ref_mask):
        raise ValueError("'reference_dem' had only NaNs")
    if np.all(tba_mask):
        raise ValueError("'dem_to_be_aligned' had only NaNs")

    # Isolate all invalid values
    invalid_mask = np.logical_or.reduce((~inlier_mask, ref_mask, tba_mask))

    if np.all(invalid_mask):
        raise ValueError("All values of the inlier mask are NaNs in either 'reference_dem' or 'dem_to_be_aligned'.")

    return ref_dem, tba_dem, inlier_mask, transform, crs


def _preprocess_coreg_fit_raster_point(
    raster_elev: NDArrayf | MArrayf | RasterType,
    point_elev: gpd.GeoDataFrame,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
) -> tuple[NDArrayf, gpd.GeoDataFrame, NDArrayb, affine.Affine, rio.crs.CRS]:
    """Pre-processing and checks of fit for raster-point input."""

    # TODO: Convert to point cloud once class is done
    if isinstance(raster_elev, gu.Raster):
        rst_elev = raster_elev.data
        crs = raster_elev.crs
        transform = raster_elev.transform
    else:
        rst_elev = raster_elev
        crs = crs
        transform = transform

    if transform is None:
        raise ValueError("'transform' must be given if both DEMs are array-like.")

    if crs is None:
        raise ValueError("'crs' must be given if both DEMs are array-like.")

    # Make sure that the mask has an expected format.
    if inlier_mask is not None:
        if isinstance(inlier_mask, Mask):
            inlier_mask = inlier_mask.data.filled(False).squeeze()
        else:
            inlier_mask = np.asarray(inlier_mask).squeeze()
            assert inlier_mask.dtype == bool, f"Invalid mask dtype: '{inlier_mask.dtype}'. Expected 'bool'"

        if np.all(~inlier_mask):
            raise ValueError("'inlier_mask' had no inliers.")
    else:
        inlier_mask = np.ones(np.shape(rst_elev), dtype=bool)

    # TODO: Convert to point cloud?
    # Convert geodataframe to vector
    point_elev = point_elev.to_crs(crs=crs)

    return rst_elev, point_elev, inlier_mask, transform, crs


def _preprocess_coreg_fit_point_point(
    reference_elev: gpd.GeoDataFrame, to_be_aligned_elev: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Pre-processing and checks of fit for point-point input."""

    ref_elev = reference_elev
    tba_elev = to_be_aligned_elev.to_crs(crs=reference_elev.crs)

    return ref_elev, tba_elev


def _preprocess_coreg_fit(
    reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    inlier_mask: NDArrayb | Mask | None = None,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
) -> tuple[
    NDArrayf | gpd.GeoDataFrame, NDArrayf | gpd.GeoDataFrame, NDArrayb | None, affine.Affine | None, rio.crs.CRS | None
]:
    """Pre-processing and checks of fit for any input."""

    if not all(
        isinstance(elev, (np.ndarray, gu.Raster, gpd.GeoDataFrame)) for elev in (reference_elev, to_be_aligned_elev)
    ):
        raise ValueError("Input elevation data should be a raster, an array or a geodataframe.")

    # If both inputs are raster or arrays, reprojection on the same grid is needed for raster-raster methods
    if all(isinstance(elev, (np.ndarray, gu.Raster)) for elev in (reference_elev, to_be_aligned_elev)):
        ref_elev, tba_elev, inlier_mask, transform, crs = _preprocess_coreg_fit_raster_raster(
            reference_dem=reference_elev,
            dem_to_be_aligned=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
        )

    # If one input is raster, and the other is point, we reproject the point data to the same CRS and extract arrays
    elif any(isinstance(dem, (np.ndarray, gu.Raster)) for dem in (reference_elev, to_be_aligned_elev)):
        if isinstance(reference_elev, (np.ndarray, gu.Raster)):
            raster_elev = reference_elev
            point_elev = to_be_aligned_elev
            ref = "raster"
        else:
            raster_elev = to_be_aligned_elev
            point_elev = reference_elev
            ref = "point"

        raster_elev, point_elev, inlier_mask, transform, crs = _preprocess_coreg_fit_raster_point(
            raster_elev=raster_elev, point_elev=point_elev, inlier_mask=inlier_mask, transform=transform, crs=crs
        )

        if ref == "raster":
            ref_elev = raster_elev
            tba_elev = point_elev
        else:
            ref_elev = point_elev
            tba_elev = raster_elev

    # If both inputs are points, simply reproject to the same CRS
    else:
        ref_elev, tba_elev = _preprocess_coreg_fit_point_point(
            reference_elev=reference_elev, to_be_aligned_elev=to_be_aligned_elev
        )

    return ref_elev, tba_elev, inlier_mask, transform, crs


def _preprocess_coreg_apply(
    elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
    transform: rio.transform.Affine | None = None,
    crs: rio.crs.CRS | None = None,
) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine, rio.crs.CRS]:
    """Pre-processing and checks of apply for any input."""

    if not isinstance(elev, (np.ndarray, gu.Raster, gpd.GeoDataFrame)):
        raise ValueError("Input elevation data should be a raster, an array or a geodataframe.")

    # If input is geodataframe
    if isinstance(elev, gpd.GeoDataFrame):
        elev_out = elev
        new_transform = None
        new_crs = None

    # If input is a raster or array
    else:
        # If input is raster
        if isinstance(elev, gu.Raster):
            if transform is not None:
                warnings.warn(f"DEM of type {type(elev)} overrides the given 'transform'")
            if crs is not None:
                warnings.warn(f"DEM of type {type(elev)} overrides the given 'crs'")
            new_transform = elev.transform
            new_crs = elev.crs

        # If input is an array
        else:
            if transform is None:
                raise ValueError("'transform' must be given if DEM is array-like.")
            if crs is None:
                raise ValueError("'crs' must be given if DEM is array-like.")
            new_transform = transform
            new_crs = crs

        # The array to provide the functions will be an ndarray with NaNs for masked out areas.
        elev_out, elev_mask = get_array_and_mask(elev)

        if np.all(elev_mask):
            raise ValueError("'dem' had only NaNs")

    return elev_out, new_transform, new_crs


def _postprocess_coreg_apply_pts(
    applied_elev: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Post-processing and checks of apply for point input."""

    # TODO: Convert CRS back if the CRS did not match the one of the fit?
    return applied_elev


def _postprocess_coreg_apply_rst(
    elev: NDArrayf | gu.Raster,
    applied_elev: NDArrayf,
    transform: affine.Affine,
    out_transform: affine.Affine,
    crs: rio.crs.CRS,
    resample: bool,
    resampling: rio.warp.Resampling | None = None,
) -> tuple[NDArrayf | gu.Raster, affine.Affine]:
    """
    Post-processing and checks of apply for raster input.

    Here, "elev" and "transform" corresponds to user input, and are required to transform back the output that is
    composed of "applied_elev" and "out_transform".
    """

    # Ensure the dtype is OK
    applied_elev = applied_elev.astype("float32")

    # Set default dst_nodata
    if isinstance(elev, gu.Raster):
        nodata = elev.nodata
    else:
        nodata = raster._default_nodata(elev.dtype)

    # Resample the array on the original grid
    if resample:
        # Reproject the DEM from its out_transform onto the transform
        applied_rst = gu.Raster.from_array(applied_elev, out_transform, crs=crs, nodata=nodata)
        if not isinstance(elev, gu.Raster):
            match_rst = gu.Raster.from_array(elev, transform, crs=crs, nodata=nodata)
        else:
            match_rst = elev
        applied_rst = applied_rst.reproject(match_rst, resampling=resampling, silent=True)
        applied_elev = applied_rst.data
        # Now that the raster data is reprojected, the new out_transform is set as the original transform
        out_transform = transform

    # Calculate final mask
    final_mask = np.logical_or(~np.isfinite(applied_elev), applied_elev == nodata)

    # If the DEM was a masked_array, copy the mask to the new DEM
    if isinstance(elev, (np.ma.masked_array, gu.Raster)):
        applied_elev = np.ma.masked_array(applied_elev, mask=final_mask)  # type: ignore
    else:
        applied_elev[final_mask] = np.nan

    # If the input was a Raster, returns a Raster, else returns array and transform
    if isinstance(elev, gu.Raster):
        out_dem = elev.from_array(applied_elev, out_transform, crs, nodata=elev.nodata)
        return out_dem, out_transform
    else:
        return applied_elev, out_transform


def _postprocess_coreg_apply(
    elev: NDArrayf | gu.Raster | gpd.GeoDataFrame,
    applied_elev: NDArrayf | gpd.GeoDataFrame,
    transform: affine.Affine,
    out_transform: affine.Affine,
    crs: rio.crs.CRS,
    resample: bool,
    resampling: rio.warp.Resampling | None = None,
) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine]:
    """
    Post-processing and checks of apply for any input.

    Here, "elev" and "transform" corresponds to user input, and are required to transform back the output that is
    composed of "applied_elev" and "out_transform".
    """

    # Define resampling
    resampling = resampling if isinstance(resampling, rio.warp.Resampling) else resampling_method_from_str(resampling)

    # Distribute between raster and point apply methods
    if isinstance(applied_elev, np.ndarray):
        applied_elev, out_transform = _postprocess_coreg_apply_rst(
            elev=elev,
            applied_elev=applied_elev,
            transform=transform,
            crs=crs,
            out_transform=out_transform,
            resample=resample,
            resampling=resampling,
        )
    else:
        applied_elev = _postprocess_coreg_apply_pts(applied_elev)

    return applied_elev, out_transform


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

        checked_matrix = pytransform3d.transformations.check_transform(matrix)
        # Invert the transform if wanted.
        return pytransform3d.transformations.invert_transform(checked_matrix)


def apply_matrix(
    elev: gu.Raster | NDArrayf | gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: int | str = "bilinear",
    transform: rio.transform.Affine = None,
    z_name: str = "z",
) -> NDArrayf | gu.Raster | gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud or 2.5D DEM.

    :param elev: Elevation point cloud or DEM to transform, either a 2D array (requires transform) or
        geodataframe (requires z_name).
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resampling: The resampling method to use, only for DEM 2.5D transformation. Can be `nearest`, `bilinear`,
        `cubic` or an integer from 0-5.
    :param transform: Geotransform of the DEM, only for DEM passed as 2D array.
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
    :return:
    """

    if isinstance(elev, gpd.GeoDataFrame):
        return _apply_matrix_pts(epc=elev, matrix=matrix, invert=invert, centroid=centroid, z_name=z_name)
    else:
        if isinstance(elev, gu.Raster):
            transform = elev.transform
            dem = elev.data
        else:
            dem = elev

        # TODO: Add exception for translation to update only geotransform, maybe directly in apply_matrix?
        applied_dem = _apply_matrix_rst(
            dem=dem, transform=transform, matrix=matrix, invert=invert, centroid=centroid, resampling=resampling
        )
        if isinstance(elev, gu.Raster):
            applied_dem = DEM.from_array(applied_dem, transform, elev.crs, elev.nodata)
        return applied_dem


def _apply_matrix_rst(
    dem: NDArrayf,
    transform: rio.transform.Affine,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    resampling: int | str = "bilinear",
    fill_max_search: int = 0,
) -> NDArrayf:
    """
    Apply a 3D affine transformation matrix to a 2.5D DEM.

    The transformation is applied as a value correction using linear deramping, and 2D image warping.

    1. Convert the DEM into a point cloud (not for gridding; for estimating the DEM shifts).
    2. Transform the point cloud in 3D using the 4x4 matrix.
    3. Measure the difference in elevation between the original and transformed points.
    4. Estimate a linear deramp from the elevation difference, and apply the correction to the DEM values.
    5. Convert the horizontal coordinates of the transformed points to pixel index coordinates.
    6. Apply the pixel-wise displacement in 2D using the new pixel coordinates.
    7. Apply the same displacement to a nodata-mask to exclude previous and/or new nans.

    :param dem: DEM to transform.
    :param transform: Geotransform of the DEM.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param resampling: The resampling method to use. Can be `nearest`, `bilinear`, `cubic` or an integer from 0-5.
    :param fill_max_search: Set to > 0 value to fill the DEM before applying the transformation, to avoid spreading\
    gaps. The DEM will be filled with rasterio.fill.fillnodata with max_search_distance set to fill_max_search.\
    This is experimental, use at your own risk !

    :returns: Transformed DEM with NaNs as nodata values (replaces a potential mask of the input `dem`).
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


def _apply_matrix_pts(
    epc: gpd.GeoDataFrame,
    matrix: NDArrayf,
    invert: bool = False,
    centroid: tuple[float, float, float] | None = None,
    z_name: str = "z",
) -> gpd.GeoDataFrame:
    """
    Apply a 3D affine transformation matrix to a 3D elevation point cloud.

    :param epc: Elevation point cloud.
    :param matrix: Affine (4x4) transformation matrix to apply to the DEM.
    :param invert: Whether to invert the transformation matrix.
    :param centroid: The X/Y/Z transformation centroid. Irrelevant for pure translations.
        Defaults to the midpoint (Z=0).
    :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.

    :return: Transformed elevation point cloud.
    """

    # Invert matrix if required
    if invert:
        matrix = invert_matrix(matrix)

    # First, get Nx3 array to pass to opencv
    points = np.array([epc.geometry.x.values, epc.geometry.y.values, epc[z_name].values]).T

    # Transform the points (around the centroid if it exists).
    if centroid is not None:
        points -= centroid
    transformed_points = cv2.perspectiveTransform(points.reshape(1, -1, 3), matrix)[
        0, :, :
    ]  # Select the first dimension that is one
    if centroid is not None:
        transformed_points += centroid

    # Finally, transform back to a new GeoDataFrame
    transformed_epc = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=transformed_points[:, 0], y=transformed_points[:, 1], crs=epc.crs),
        data={"z": transformed_points[:, 2]},
    )

    return transformed_epc


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

    # Affine + BiasCorr classes
    subsample: int | float
    subsample_final: int
    random_state: int | np.random.Generator | None

    # BiasCorr classes generic metadata

    # 1/ Inputs
    fit_or_bin: Literal["fit"] | Literal["bin"]
    fit_func: Callable[..., NDArrayf]
    fit_optimizer: Callable[..., tuple[NDArrayf, Any]]
    bin_sizes: int | dict[str, int | Iterable[float]]
    bin_statistic: Callable[[NDArrayf], np.floating[Any]]
    bin_apply_method: Literal["linear"] | Literal["per_bin"]
    bias_var_names: list[str]
    nd: int | None

    # 2/ Outputs
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
    _needs_vars: bool = False

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

    def _get_subsample_on_valid_mask(self, valid_mask: NDArrayb, verbose: bool = False) -> NDArrayb:
        """
        Get mask of values to subsample on valid mask.

        :param valid_mask: Mask of valid values (inlier and not nodata).
        """

        # This should never happen
        if self._meta["subsample"] is None:
            raise ValueError("Subsample should have been defined in metadata before reaching this class method.")

        # If subsample is not equal to one, subsampling should be performed.
        elif self._meta["subsample"] != 1.0:

            # Build a low memory masked array with invalid values masked to pass to subsampling
            ma_valid = np.ma.masked_array(data=np.ones(np.shape(valid_mask), dtype=bool), mask=~valid_mask)
            # Take a subsample within the valid values
            indices = gu.raster.subsample_array(
                ma_valid,
                subsample=self._meta["subsample"],
                return_indices=True,
                random_state=self._meta["random_state"],
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

        if verbose:
            print(
                "Using a subsample of {} among {} valid values.".format(
                    np.count_nonzero(valid_mask), np.count_nonzero(subsample_mask)
                )
            )

        # Write final subsample to class
        self._meta["subsample_final"] = np.count_nonzero(subsample_mask)

        return subsample_mask

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        verbose: bool = False,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:
        """
        Estimate the coregistration transform on the given DEMs.

        :param reference_elev: Reference elevation, either a DEM or an elevation point cloud.
        :param to_be_aligned_elev: To-be-aligned elevation, either a DEM or an elevation point cloud.
        :param inlier_mask: Mask or boolean array of areas to include (inliers=True).
        :param bias_vars: Auxiliary variables for certain bias correction classes, as raster or arrays.
        :param weights: Array of weights for the coregistration.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param transform: Transform of the reference elevation, only if provided as 2D array.
        :param crs: CRS of the reference elevation, only if provided as 2D array.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param verbose: Print progress messages.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling).
        """

        if weights is not None:
            raise NotImplementedError("Weights have not yet been implemented")

        # Override subsample argument of instantiation if passed to fit
        if subsample is not None:

            # Check if subsample argument was also defined at instantiation (not default value), and raise warning
            argspec = inspect.getfullargspec(self.__class__)
            sub_meta = self._meta["subsample"]
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
            self._meta["subsample"] = subsample

        # Save random_state if a subsample is used
        if self._meta["subsample"] != 1:
            self._meta["random_state"] = random_state

        # Pre-process the inputs, by reprojecting and converting to arrays
        ref_elev, tba_elev, inlier_mask, transform, crs = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
        )

        main_args = {
            "ref_elev": ref_elev,
            "tba_elev": tba_elev,
            "inlier_mask": inlier_mask,
            "transform": transform,
            "crs": crs,
            "z_name": z_name,
            "weights": weights,
            "verbose": verbose,
        }

        # If bias_vars are defined, update dictionary content to array
        if bias_vars is not None:
            # Check if the current class actually requires bias_vars
            if self._is_affine:
                warnings.warn("This coregistration method is affine, ignoring `bias_vars` passed to fit().")

            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

            main_args.update({"bias_vars": bias_vars})

        # Run the associated fitting function, which has fallback logic for "raster-raster", "raster-point" or
        # "point-point" depending on what is available for a certain Coreg function
        self._fit_func(
            **main_args,
            **kwargs,
        )

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame:
        ...

    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:
        """
        Apply the estimated transform to a DEM.

        :param elev: Elevation to apply the transform to, either a DEM or an elevation point cloud.
        :param bias_vars: Only for some bias correction classes. 2D array of bias variables used.
        :param resample: If set to True, will reproject output Raster on the same grid as input. Otherwise, \
            only the transform might be updated and no resampling is done.
        :param resampling: Resampling method if resample is used. Defaults to "bilinear".
        :param transform: Geotransform of the elevation, only if provided as 2D array.
        :param crs: CRS of elevation, only if provided as 2D array.
        :param z_name: Column name to use as elevation, only for point elevation data passed as geodataframe.
        :param kwargs: Any optional arguments to be passed to either self._apply_rst or apply_matrix.

        :returns: The transformed DEM.
        """
        if not self._fit_called and self._meta.get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        elev_array, transform, crs = _preprocess_coreg_apply(elev=elev, transform=transform, crs=crs)

        main_args = {"elev": elev_array, "transform": transform, "crs": crs, "resample": resample, "z_name": z_name}

        # If bias_vars are defined, update dictionary content to array
        if bias_vars is not None:
            # Check if the current class actually requires bias_vars
            if self._is_affine:
                warnings.warn("This coregistration method is affine, ignoring `bias_vars` passed to apply().")

            for var in bias_vars.keys():
                bias_vars[var] = gu.raster.get_array_and_mask(bias_vars[var])[0]

            main_args.update({"bias_vars": bias_vars})

        # Call _apply_func to choose method depending on point/raster input and if specific apply method exists
        applied_elev, out_transform = self._apply_func(**main_args, **kwargs)

        # Post-process output depending on input type
        applied_elev, out_transform = _postprocess_coreg_apply(
            elev=elev,
            applied_elev=applied_elev,
            transform=transform,
            out_transform=out_transform,
            crs=crs,
            resample=resample,
            resampling=resampling,
        )

        # Only return object if raster or geodataframe, also return transform if object was an array
        if isinstance(applied_elev, (gu.Raster, gpd.GeoDataFrame)):
            return applied_elev
        else:
            return applied_elev, out_transform

    def residuals(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        subsample: float | int = 1.0,
        random_state: int | np.random.Generator | None = None,
    ) -> NDArrayf:
        """
        Calculate the residual offsets (the difference) between two DEMs after applying the transformation.

        :param reference_elev: 2D array of elevation values acting reference.
        :param to_be_aligned_elev: 2D array of elevation values to be aligned.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.
        :param subsample: Subsample the input to increase performance. <1 is parsed as a fraction. >1 is a pixel count.
        :param random_state: Random state or seed number to use for calculations (to fix random sampling during testing)

        :returns: A 1D array of finite residuals.
        """

        # Apply the transformation to the dem to be aligned
        aligned_elev = self.apply(to_be_aligned_elev, transform=transform, crs=crs)[0]

        # Pre-process the inputs, by reprojecting and subsampling
        ref_dem, align_elev, inlier_mask, transform, crs = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
        )

        # Calculate the DEM difference
        diff = ref_dem - align_elev

        # Sometimes, the float minimum (for float32 = -3.4028235e+38) is returned. This and inf should be excluded.
        full_mask = np.isfinite(diff)
        if "float" in str(diff.dtype):
            full_mask[(diff == np.finfo(diff.dtype).min) | np.isinf(diff)] = False

        # Return the difference values within the full inlier mask
        return diff[full_mask]

    @overload
    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: list[str],
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
    ) -> list[np.floating[Any] | float | np.integer[Any] | int]:
        ...

    @overload
    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: str = "nmad",
        inlier_mask: NDArrayb | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
    ) -> np.floating[Any] | float | np.integer[Any] | int:
        ...

    def error(
        self,
        reference_elev: NDArrayf,
        to_be_aligned_elev: NDArrayf,
        error_type: str | list[str] = "nmad",
        inlier_mask: NDArrayb | None = None,
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

        :param reference_elev: 2D array of elevation values acting reference.
        :param to_be_aligned_elev: 2D array of elevation values to be aligned.
        :param error_type: The type of error measure to calculate. May be a list of error types.
        :param inlier_mask: Optional. 2D boolean array of areas to include in the analysis (inliers=True).
        :param transform: Optional. Transform of the reference_dem. Mandatory in some cases.
        :param crs: Optional. CRS of the reference_dem. Mandatory in some cases.

        :returns: The error measure of choice for the residuals.
        """
        if isinstance(error_type, str):
            error_type = [error_type]

        residuals = self.residuals(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
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
        **kwargs: Any,
    ) -> None:
        """
        Distribute to _fit_rst_rst, fit_rst_pts or fit_pts_pts depending on input and method availability.
        Needs to be _fit_func of the main class to simplify calls from CoregPipeline and BlockwiseCoreg.
        """

        # Determine if input is raster-raster, raster-point or point-point
        if all(isinstance(dem, np.ndarray) for dem in (kwargs["ref_elev"], kwargs["tba_elev"])):
            rop = "r-r"
        elif all(isinstance(dem, gpd.GeoDataFrame) for dem in (kwargs["ref_elev"], kwargs["tba_elev"])):
            rop = "p-p"
        else:
            rop = "r-p"

        # Fallback logic is always the same: 1/ raster-raster, 2/ raster-point, 3/ point-point
        try_rp = False
        try_pp = False

        # For raster-raster
        if rop == "r-r":
            # Check if raster-raster function exists, if yes run it and stop
            try:
                self._fit_rst_rst(**kwargs)
            # Otherwise, convert the tba raster to points and try raster-points
            except NotImplementedCoregFit:
                warnings.warn(
                    f"No raster-raster method found for coregistration {self.__class__.__name__}, "
                    f"trying raster-point method by converting to-be-aligned DEM to points.",
                    UserWarning,
                )
                tba_elev_pts = (
                    gu.Raster.from_array(data=kwargs["tba_elev"], transform=kwargs["transform"], crs=kwargs["crs"])
                    .to_pointcloud()
                    .ds
                )
                kwargs.update({"tba_elev": tba_elev_pts})
                try_rp = True

        # For raster-point
        if rop == "r-p" or try_rp:
            try:
                self._fit_rst_pts(**kwargs)
            except NotImplementedCoregFit:
                warnings.warn(
                    f"No raster-point method found for coregistration {self.__class__.__name__}, "
                    f"trying point-point method by converting all elevation data to points.",
                    UserWarning,
                )
                ref_elev_pts = (
                    gu.Raster.from_array(data=kwargs["ref_elev"], transform=kwargs["transform"], crs=kwargs["crs"])
                    .to_pointcloud()
                    .ds
                )
                kwargs.update({"ref_elev": ref_elev_pts})
                try_pp = True

        # For point-point
        if rop == "p-p" or try_pp:
            try:
                self._fit_pts_pts(**kwargs)
            except NotImplementedCoregFit:
                if try_pp and try_rp:
                    raise NotImplementedCoregFit(
                        f"No raster-raster, raster-point or point-point method found for "
                        f"coregistration {self.__class__.__name__}."
                    )
                elif try_pp:
                    raise NotImplementedCoregFit(
                        f"No raster-point or point-point method found for coregistration {self.__class__.__name__}."
                    )
                else:
                    raise NotImplementedCoregFit(
                        f"No point-point method found for coregistration {self.__class__.__name__}."
                    )

    def _apply_func(self, **kwargs: Any) -> tuple[NDArrayf | gpd.GeoDataFrame, affine.Affine]:
        """Distribute to _apply_rst and _apply_pts based on input and method availability."""

        # If input is a raster
        if isinstance(kwargs["elev"], np.ndarray):

            # See if a _apply_rst exists
            try:
                # Run the associated apply function
                applied_elev, out_transform = self._apply_rst(**kwargs)  # pylint: disable=assignment-from-no-return

            # If it doesn't exist, use apply_matrix()
            except NotImplementedCoregApply:

                if self.is_affine:  # This only works for affine, however.

                    # In this case, resampling is necessary
                    if not kwargs["resample"]:
                        raise NotImplementedError(
                            f"Option `resample=False` not implemented for coreg method {self.__class__}"
                        )
                    kwargs.pop("resample")  # Need to removed before passing to apply_matrix

                    # Apply the matrix around the centroid (if defined, otherwise just from the center).
                    transform = kwargs.pop("transform")
                    applied_elev = _apply_matrix_rst(
                        dem=kwargs.pop("elev"),
                        transform=transform,
                        matrix=self.to_matrix(),
                        centroid=self._meta.get("centroid"),
                    )
                    out_transform = transform
                else:
                    raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_rst.")

        # If input is a point
        else:
            out_transform = None

            # See if an _apply_pts_func exists
            try:
                applied_elev = self._apply_pts(**kwargs)

            # If it doesn't exist, use opencv's perspectiveTransform
            except NotImplementedCoregApply:
                if self.is_affine:  # This only works on it's rigid, however.

                    applied_elev = _apply_matrix_pts(
                        epc=kwargs["elev"],
                        matrix=self.to_matrix(),
                        centroid=self._meta.get("centroid"),
                        z_name=kwargs.pop("z_name"),
                    )

                else:
                    raise ValueError("Cannot transform, Coreg method is non-affine and has no implemented _apply_pts.")

        return applied_elev, out_transform

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
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

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
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _fit_pts_pts(
        self,
        ref_elev: gpd.GeoDataFrame,
        tba_elev: gpd.GeoDataFrame,
        inlier_mask: NDArrayb,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        z_name: str,
        weights: NDArrayf | None = None,
        bias_vars: dict[str, NDArrayf] | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregFit("This step has to be implemented by subclassing.")

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")

    def _apply_pts(
        self,
        elev: gpd.GeoDataFrame,
        z_name: str = "z",
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:

        # FOR DEVELOPERS: This function needs to be implemented by subclassing.
        raise NotImplementedCoregApply("This should have been implemented by subclassing.")


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

    def _parse_bias_vars(self, step: int, bias_vars: dict[str, NDArrayf] | None) -> dict[str, NDArrayf]:
        """Parse bias variables for a pipeline step requiring them."""

        # Get number of non-affine coregistration requiring bias variables to be passed
        nb_needs_vars = sum(c._needs_vars for c in self.pipeline)

        # Get step object
        coreg = self.pipeline[step]

        # Check that all variable names of this were passed
        var_names = coreg._meta["bias_var_names"]

        # Raise error if bias_vars is None
        if bias_vars is None:
            msg = f"No `bias_vars` passed to .fit() for bias correction step {coreg.__class__} of the pipeline."
            if nb_needs_vars > 1:
                msg += (
                    " As you are using several bias correction steps requiring `bias_vars`, don't forget to "
                    "explicitly define their `bias_var_names` during "
                    "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
                )
            raise ValueError(msg)

        # Raise error if no variable were explicitly assigned and there is more than 1 step with bias_vars
        if var_names is None and nb_needs_vars > 1:
            raise ValueError(
                "When using several bias correction steps requiring `bias_vars` in a pipeline,"
                "the `bias_var_names` need to be explicitly defined at each step's "
                "instantiation, e.g. {}(bias_var_names=['slope']).".format(coreg.__class__.__name__)
            )

        # Raise error if the variables explicitly assigned don't match the ones passed in bias_vars
        if not all(n in bias_vars.keys() for n in var_names):
            raise ValueError(
                "Not all keys of `bias_vars` in .fit() match the `bias_var_names` defined during "
                "instantiation of the bias correction step {}: {}.".format(coreg.__class__, var_names)
            )

        # Add subset dict for this pipeline step to args of fit and apply
        return {n: bias_vars[n] for n in var_names}

    # Need to override base Coreg method to work on pipeline steps
    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType | gpd.GeoDataFrame,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        verbose: bool = False,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:

        # Check if subsample arguments are different from their default value for any of the coreg steps:
        # get default value in argument spec and "subsample" stored in meta, and compare both are consistent
        argspec = [inspect.getfullargspec(c.__class__) for c in self.pipeline]
        sub_meta = [c._meta["subsample"] for c in self.pipeline]
        sub_is_default = [
            argspec[i].defaults[argspec[i].args.index("subsample") - 1] == sub_meta[i]  # type: ignore
            for i in range(len(argspec))
        ]
        if subsample is not None and not all(sub_is_default):
            warnings.warn(
                "Subsample argument passed to fit() will override non-default subsample values defined for"
                " individual steps of the pipeline. To silence this warning: only define 'subsample' in "
                "either fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
            )
            # Filter warnings of individual pipelines now that the one above was raised
            warnings.filterwarnings("ignore", message="Subsample argument passed to*", category=UserWarning)

        # Pre-process the inputs, by reprojecting and subsampling, without any subsampling (done in each step)
        ref_dem, tba_dem, inlier_mask, transform, crs = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
        )

        tba_dem_mod = tba_dem.copy()
        out_transform = transform

        for i, coreg in enumerate(self.pipeline):
            if verbose:
                print(f"Running pipeline step: {i + 1} / {len(self.pipeline)}")

            main_args_fit = {
                "reference_elev": ref_dem,
                "to_be_aligned_elev": tba_dem_mod,
                "inlier_mask": inlier_mask,
                "transform": out_transform,
                "crs": crs,
                "z_name": z_name,
                "weights": weights,
                "verbose": verbose,
                "subsample": subsample,
                "random_state": random_state,
            }

            main_args_apply = {"elev": tba_dem_mod, "transform": out_transform, "crs": crs, "z_name": z_name}

            # If non-affine method that expects a bias_vars argument
            if coreg._needs_vars:
                step_bias_vars = self._parse_bias_vars(step=i, bias_vars=bias_vars)

                main_args_fit.update({"bias_vars": step_bias_vars})
                main_args_apply.update({"bias_vars": step_bias_vars})

            # Perform the step fit
            coreg.fit(**main_args_fit)

            # Step apply: one return for a geodataframe, two returns for array/transform
            if isinstance(tba_dem_mod, gpd.GeoDataFrame):
                tba_dem_mod = coreg.apply(**main_args_apply)
            else:
                tba_dem_mod, out_transform = coreg.apply(**main_args_apply)

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

    @overload
    def apply(
        self,
        elev: MArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[MArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        elev: NDArrayf,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:
        ...

    @overload
    def apply(
        self,
        elev: RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame:
        ...

    # Need to override base Coreg method to work on pipeline steps
    def apply(
        self,
        elev: MArrayf | NDArrayf | RasterType | gpd.GeoDataFrame,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        resample: bool = True,
        resampling: str | rio.warp.Resampling = "bilinear",
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        **kwargs: Any,
    ) -> RasterType | gpd.GeoDataFrame | tuple[NDArrayf, rio.transform.Affine] | tuple[MArrayf, rio.transform.Affine]:

        # First step and preprocessing
        if not self._fit_called and self._meta.get("matrix") is None:
            raise AssertionError(".fit() does not seem to have been called yet")

        elev_array, transform, crs = _preprocess_coreg_apply(elev=elev, transform=transform, crs=crs)

        elev_mod = elev_array.copy()
        out_transform = copy.copy(transform)

        # Apply each step of the coregistration
        for i, coreg in enumerate(self.pipeline):

            main_args_apply = {
                "elev": elev_mod,
                "transform": out_transform,
                "crs": crs,
                "z_name": z_name,
                "resample": resample,
                "resampling": resampling,
            }

            # If non-affine method that expects a bias_vars argument
            if coreg._needs_vars:
                step_bias_vars = self._parse_bias_vars(step=i, bias_vars=bias_vars)
                main_args_apply.update({"bias_vars": step_bias_vars})

            # Step apply: one return for a geodataframe, two returns for array/transform
            if isinstance(elev_mod, gpd.GeoDataFrame):
                elev_mod = coreg.apply(**main_args_apply, **kwargs)
            else:
                elev_mod, out_transform = coreg.apply(**main_args_apply, **kwargs)

        # Post-process output depending on input type
        applied_elev, out_transform = _postprocess_coreg_apply(
            elev=elev,
            applied_elev=elev_mod,
            transform=transform,
            out_transform=out_transform,
            crs=crs,
            resample=resample,
            resampling=resampling,
        )

        # Only return object if raster or geodataframe, also return transform if object was an array
        if isinstance(applied_elev, (gu.Raster, gpd.GeoDataFrame)):
            return applied_elev
        else:
            return applied_elev, out_transform

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

    def fit(
        self: CoregType,
        reference_elev: NDArrayf | MArrayf | RasterType,
        to_be_aligned_elev: NDArrayf | MArrayf | RasterType,
        inlier_mask: NDArrayb | Mask | None = None,
        bias_vars: dict[str, NDArrayf | MArrayf | RasterType] | None = None,
        weights: NDArrayf | None = None,
        subsample: float | int | None = None,
        transform: rio.transform.Affine | None = None,
        crs: rio.crs.CRS | None = None,
        z_name: str = "z",
        verbose: bool = False,
        random_state: int | np.random.Generator | None = None,
        **kwargs: Any,
    ) -> CoregType:

        if isinstance(reference_elev, gpd.GeoDataFrame) and isinstance(to_be_aligned_elev, gpd.GeoDataFrame):
            raise NotImplementedError("Blockwise coregistration does not yet support two elevation point cloud inputs.")

        # Check if subsample arguments are different from their default value for any of the coreg steps:
        # get default value in argument spec and "subsample" stored in meta, and compare both are consistent
        if not isinstance(self.procstep, CoregPipeline):
            steps = [self.procstep]
        else:
            steps = list(self.procstep.pipeline)
        argspec = [inspect.getfullargspec(s.__class__) for s in steps]
        sub_meta = [s._meta["subsample"] for s in steps]
        sub_is_default = [
            argspec[i].defaults[argspec[i].args.index("subsample") - 1] == sub_meta[i]  # type: ignore
            for i in range(len(argspec))
        ]
        if subsample is not None and not all(sub_is_default):
            warnings.warn(
                "Subsample argument passed to fit() will override non-default subsample values defined in the"
                " step within the blockwise method. To silence this warning: only define 'subsample' in "
                "either fit(subsample=...) or instantiation e.g., VerticalShift(subsample=...)."
            )

        # Pre-process the inputs, by reprojecting and subsampling, without any subsampling (done in each step)
        ref_dem, tba_dem, inlier_mask, transform, crs = _preprocess_coreg_fit(
            reference_elev=reference_elev,
            to_be_aligned_elev=to_be_aligned_elev,
            inlier_mask=inlier_mask,
            transform=transform,
            crs=crs,
        )

        groups = self.subdivide_array(tba_dem.shape if isinstance(tba_dem, np.ndarray) else ref_dem.shape)

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
            transform_subset = rio.transform.from_origin(west, top, transform.a, -transform.e)  # type: ignore
            procstep = self.procstep.copy()

            # Try to run the coregistration. If it fails for any reason, skip it and save the exception.
            try:
                procstep.fit(
                    reference_elev=ref_subset,
                    to_be_aligned_elev=tba_subset,
                    transform=transform_subset,
                    inlier_mask=mask_subset,
                    bias_vars=bias_vars,
                    weights=weights,
                    crs=crs,
                    z_name=z_name,
                    subsample=subsample,
                    random_state=random_state,
                    verbose=verbose,
                )
                nmad, median = procstep.error(
                    reference_elev=ref_subset,
                    to_be_aligned_elev=tba_subset,
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

        # Flag that the fitting function has been called.
        self._fit_called = True

        return self

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

            old_pos_arr = np.reshape([x_coord, y_coord, meta["representative_val"]], (1, 3))
            old_position = gpd.GeoDataFrame(
                geometry=gpd.points_from_xy(x=old_pos_arr[:, 0], y=old_pos_arr[:, 1], crs=None),
                data={"z": old_pos_arr[:, 2]},
            )

            new_position = self.procstep.apply(old_position)
            new_pos_arr = np.reshape(
                [new_position.geometry.x.values, new_position.geometry.y.values, new_position["z"].values], (1, 3)
            )

            points = np.append(points, np.dstack((old_pos_arr, new_pos_arr)), axis=0)

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

    def _apply_rst(
        self,
        elev: NDArrayf,
        transform: rio.transform.Affine,
        crs: rio.crs.CRS,
        bias_vars: dict[str, NDArrayf] | None = None,
        **kwargs: Any,
    ) -> tuple[NDArrayf, rio.transform.Affine]:

        if np.count_nonzero(np.isfinite(elev)) == 0:
            return elev, transform

        # Other option than resample=True is not implemented for this case
        if "resample" in kwargs and kwargs["resample"] is not True:
            raise NotImplementedError("Option `resample=False` not implemented for coreg method BlockwiseCoreg.")

        points = self.to_points()

        bounds, resolution = _transform_to_bounds_and_res(elev.shape, transform)

        representative_height = np.nanmean(elev)
        edges_source_arr = np.array(
            [
                [bounds.left + resolution / 2, bounds.top - resolution / 2, representative_height],
                [bounds.right - resolution / 2, bounds.top - resolution / 2, representative_height],
                [bounds.left + resolution / 2, bounds.bottom + resolution / 2, representative_height],
                [bounds.right - resolution / 2, bounds.bottom + resolution / 2, representative_height],
            ]
        )
        edges_source = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=edges_source_arr[:, 0], y=edges_source_arr[:, 1], crs=None),
            data={"z": edges_source_arr[:, 2]},
        )

        edges_dest = self.apply(edges_source)
        edges_dest_arr = np.array(
            [edges_dest.geometry.x.values, edges_dest.geometry.y.values, edges_dest["z"].values]
        ).T
        edges = np.dstack((edges_source_arr, edges_dest_arr))

        all_points = np.append(points, edges, axis=0)

        warped_dem = warp_dem(
            dem=elev,
            transform=transform,
            source_coords=all_points[:, :, 0],
            destination_coords=all_points[:, :, 1],
            resampling="linear",
        )

        return warped_dem, transform

    def _apply_pts(
        self, elev: gpd.GeoDataFrame, z_name: str = "z", bias_vars: dict[str, NDArrayf] | None = None, **kwargs: Any
    ) -> gpd.GeoDataFrame:
        """Apply the scaling model to a set of points."""
        points = self.to_points()

        new_coords = np.array([elev.geometry.x.values, elev.geometry.y.values, elev["z"].values]).T

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

            new_coords[:, dim] += model(elev.geometry.x.values, elev.geometry.y.values)

        gdf_new_coords = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x=new_coords[:, 0], y=new_coords[:, 1], crs=None), data={"z": new_coords[:, 2]}
        )

        return gdf_new_coords


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

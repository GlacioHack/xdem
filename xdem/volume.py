"""Functions to calculate changes in volume (aimed for glaciers)."""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.fill
import scipy.interpolate

import xdem


def hypsometric_binning(ddem: np.ndarray, ref_dem: np.ndarray, bins: Union[float, np.ndarray] = 50.0,
                        kind: str = "fixed", aggregation_function: Callable = np.median) -> pd.DataFrame:
    """
    Separate the dDEM in discrete elevation bins.
    The elevation bins will be calculated based on all ref_dem valid pixels.
    ddem may contain NaN/masked values over the same area, they will be excluded before the aggregation.

    It is assumed that the dDEM is calculated as 'ref_dem - dem' (not 'dem - ref_dem').

    :param ddem: The dDEM as a 2D or 1D array.
    :param ref_dem: The reference DEM as a 2D or 1D array.
    :param bins: The bin size, count, or array, depending on the binning method ('kind').
    :param kind: The kind of binning to do. Choices: ['fixed', 'count', 'quantile', 'custom'].
    :param aggregation_function: The function to aggregate the elevation values within a bin. Defaults to the median.

    :returns: A Pandas DataFrame with elevation bins and dDEM statistics.
    """
    assert ddem.shape == ref_dem.shape

    # Convert ddem mask into NaN
    ddem, _ = xdem.spatial_tools.get_array_and_mask(ddem)

    # Extract only the valid values, i.e. valid in ref_dem
    valid_mask = ~xdem.spatial_tools.get_mask(ref_dem)
    ddem = np.array(ddem[valid_mask])
    ref_dem = np.array(ref_dem[valid_mask])

    # If the bin size should be seen as a percentage.
    if kind == "fixed":
        zbins = np.arange(ref_dem.min(), ref_dem.max() + bins + 1e-6, step=bins)  # +1e-6 in case min=max (1 point)
    elif kind == "count":
        # Make bins between mean_dem.min() and a little bit above mean_dem.max().
        # The bin count has to be bins + 1 because zbins[0] will be a "below min value" bin, which will be irrelevant.
        zbins = np.linspace(ref_dem.min(), ref_dem.max() + 1e-6 / bins, num=int(bins + 1))
    elif kind == "quantile":
        # Make the percentile steps. The bins + 1 is explained above.
        steps = np.linspace(0, 100, num=int(bins) + 1)
        zbins = np.fromiter(
            (np.percentile(ref_dem, step) for step in steps),
            dtype=float
        )
        # The uppermost bin needs to be a tiny amount larger than the highest value to include it.
        zbins[-1] += 1e-6
    elif kind == "custom":
        zbins = bins
    else:
        raise ValueError(f"Invalid bin kind: {kind}. Choices: ['fixed', 'count', 'quantile', 'custom']")

    # Generate bins and get bin indices from the mean DEM
    indices = np.digitize(ref_dem, bins=zbins)

    # Calculate statistics for each bin.
    # If no values exist, all stats should be nans (except count with should be 0)
    # medians, means, stds, nmads = (np.zeros(shape=bins.shape[0] - 1, dtype=ddem.dtype) * np.nan, ) * 4
    values = np.zeros(shape=zbins.shape[0] - 1, dtype=ddem.dtype) * np.nan
    counts = np.zeros_like(values, dtype=int)
    for i in np.arange(indices.min(), indices.max() + 1):
        values_in_bin = ddem[indices == i]

        # Remove possible Nans
        values_in_bin = values_in_bin[np.isfinite(values_in_bin)]

        # Skip if no values are in the bin.
        if values_in_bin.shape[0] == 0:
            continue

        values[i - 1] = aggregation_function(values_in_bin)
        counts[i - 1] = values_in_bin.shape[0]

    # Collect the results in a dataframe
    output = pd.DataFrame(
        index=pd.IntervalIndex.from_breaks(zbins),
        data=np.vstack([
            values, counts
        ]).T,
        columns=["value", "count"]
    )

    return output


def interpolate_hypsometric_bins(hypsometric_bins: pd.DataFrame, value_column="value", method="polynomial", order=3,
                                 count_threshold: Optional[int] = None) -> pd.DataFrame:
    """
    Interpolate hypsometric bins using any valid Pandas interpolation technique.

    NOTE: It will not extrapolate!

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param value_column: The name of the column in 'hypsometric_bins' to use as values.
    :param method: Any valid Pandas interpolation technique (e.g. 'linear', 'polynomial', 'ffill', 'bfill').
    :param order: The polynomial order to use. Only used if method='polynomial'.
    :param count_threshold: Optional. A pixel count threshold to exclude during the curve fit (requires a 'count' column).
    :returns: Bins interpolated with the chosen interpolation method.
    """
    # Copy the bins that will be filled.
    bins = hypsometric_bins.copy()
    # Temporarily set the index to be the midpoint (for interpolation)
    bins.index = bins.index.mid

    # Set all bins that are under a (potentially) specified count to NaN (they should be excluded from interpolation)
    if count_threshold is not None:
        assert "count" in hypsometric_bins.columns, "'count' not a column in the dataframe"
        bins_under_threshold = bins["count"] < count_threshold
        bins.loc[bins_under_threshold, value_column] = np.nan

    # Interpolate all bins that are NaN.
    bins[value_column] = bins[value_column].interpolate(method=method, order=order, limit_direction="both")

    # If some points were temporarily set to NaN (to exclude from the interpolation), re-set them.
    if count_threshold is not None:
        bins.loc[bins_under_threshold, value_column] = hypsometric_bins.loc[bins_under_threshold.values, value_column]

    # Return the index to intervals instead of the midpoint.
    bins.index = hypsometric_bins.index

    return bins


def fit_hypsometric_bins_poly(hypsometric_bins: pd.DataFrame, value_column: str = "value", degree: int = 3,
                              iterations: int = 1, count_threshold: Optional[int] = None) -> pd.Series:
    """
    Fit a polynomial to the hypsometric bins.

    :param hypsometric_bins: Bins where nans will be interpolated.
    :param value_column: The name of the column in 'hypsometric_bins' to use as values.
    :param degree: The degree of the polynomial to use.
    :param iterations: The number of iterations to run. \
 At each iteration, values with residuals larger than 3 times the residuals' standard deviation are excluded.
    :param count_threshold: Optional. A pixel count threshold to exclude during the curve fit (requires a 'count' column).
    :returns: Bins replaced by the polynomial fit.
    """
    bins = hypsometric_bins.copy()
    bins.index = bins.index.mid

    if count_threshold is not None:
        assert "count" in hypsometric_bins.columns, f"'count' not a column in the dataframe"
        bins_under_threshold = bins["count"] < count_threshold
        bins.loc[bins_under_threshold, value_column] = np.nan

    # Remove invalid bins
    valids = np.isfinite(np.asarray(bins[value_column]))

    for k in range(iterations):

        # Fit polynomial
        x = bins.index[valids]
        y = bins[value_column][valids]
        pcoeff = np.polyfit(x, y, deg=degree)

        # Calculate residuals
        interpolated_values = np.polyval(pcoeff, bins.index)
        residuals = interpolated_values - bins[value_column]
        residuals_std = np.nanstd(residuals.values)

        # Filter outliers further than 3 std
        valids_old = np.copy(valids)
        valids[np.abs(residuals.values) > 3*residuals_std] = False
        if np.array_equal(valids, valids_old):
            break

    # Save as pandas' DataFrame
    output = pd.DataFrame(
        index=hypsometric_bins.index,
        data=np.vstack([
            interpolated_values, bins["count"]
        ]).T,
        columns=["value", "count"]
    )

    return output


def calculate_hypsometry_area(ddem_bins: Union[pd.Series, pd.DataFrame], ref_dem: np.ndarray,
                              pixel_size: Union[float, tuple[float, float]],
                              timeframe: str = "reference") -> pd.Series:
    """
    Calculate the associated representative area of the given dDEM bins.

    By default, the area bins will be representative of the mean timing between the reference and nonreference DEM:
    elevations = ref_dem - (h_vs_dh_funcion(ref_dem) / 2)
    This can be changed to either "reference":
    elevations = ref_dem
    Or "nonreference":
    elevations = ref_dem - h_vs_dh_function(ref_dem)

    :param ddem_bins: A Series or DataFrame of dDEM values. If a DataFrame is given, the column 'value' will be used.
    :param ref_dem: The reference DEM. This should not have any NaNs.
    :param pixel_size: The xy or (x, y) size of the reference DEM pixels in georeferenced coordinates.
    :param timeframe: The time at which the area bins are representative. Choices: "reference", "nonreference", "mean"

    :returns: The representative area within the given dDEM bins.
    """
    assert not np.any(np.isnan(ref_dem)), "The given reference DEM has NaNs. No NaNs are allowed to calculate area!"

    if timeframe not in ["reference", "nonreference", "mean"]:
        raise ValueError(f"Argument 'timeframe={timeframe}' is invalid. Choices: ['reference', 'nonreference', 'mean']")

    if isinstance(ddem_bins, pd.DataFrame):
        ddem_bins = ddem_bins["value"]
    assert not np.any(np.isnan(ddem_bins.values)), "The dDEM bins cannot contain NaNs. Remove or fill them first."
    # Generate a continuous elevation vs. dDEM function
    ddem_func = scipy.interpolate.interp1d(ddem_bins.index.mid, ddem_bins.values,
                                           kind="linear", fill_value="extrapolate")
    # Generate average elevations by subtracting half of the dDEM's values to the reference DEM
    if timeframe == "mean":
        elevations = ref_dem - (ddem_func(ref_dem.data) / 2)
    elif timeframe == "reference":
        elevations = ref_dem
    elif timeframe == "nonreference":
        elevations = ref_dem - ddem_func(ref_dem.data)

    # Extract the bins from the dDEM series and compute the frequency of points in the bins.
    bins = np.r_[[ddem_bins.index.left[0]], ddem_bins.index.right]
    bin_counts = np.histogram(elevations, bins=bins)[0]

    # Multiply the bin counts with the pixel area to get the full area.
    bin_area = bin_counts * (pixel_size ** 2 if not isinstance(pixel_size, tuple) else pixel_size[0] * pixel_size[1])

    # Put this in a series which will be returned.
    output = pd.Series(index=ddem_bins.index, data=bin_area)

    return output


def linear_interpolation(array: Union[np.ndarray, np.ma.masked_array], max_search_distance: int = 10,
                         extrapolate: bool = False, force_fill: bool = False) -> np.ndarray:
    """
    Interpolate a 2D array using rasterio's fillnodata.

    :param array: An array with NaNs or a masked array to interpolate.
    :param max_search_distance: The maximum number of pixels to search in all directions to find values \
to interpolate from. The default is 10.
    :param extrapolate: if False, will remove pixels that have been extrapolated by fillnodata. Default is False.
    :param force_fill: if True, will replace all remaining gaps by the median of all valid values. Default is False.

    :returns: A filled array with no NaNs
    """
    # Create a mask for where nans exist
    nan_mask = xdem.spatial_tools.get_mask(array)

    interpolated_array = rasterio.fill.fillnodata(array.copy(), mask=(~nan_mask).astype("uint8"),
                                                  max_search_distance=max_search_distance)

    # Remove extrapolated values: gaps up to the size of max_search_distance are kept,
    # but surfaces that artifically grow on the edges are removed
    if not extrapolate:
        interp_mask = cv2.morphologyEx((~nan_mask).squeeze().astype('uint8'), cv2.MORPH_CLOSE,
                                       kernel=np.ones((max_search_distance - 1, )*2)).astype('bool')
        if np.ndim(array) == 3:
            interpolated_array[:, ~interp_mask] = np.nan
        else:
            interpolated_array[~interp_mask] = np.nan

    # Fill the nans (values outside of the value boundaries) with the median value
    # This triggers a warning with np.masked_array's because it ignores the mask
    if force_fill:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpolated_array[np.isnan(interpolated_array)] = np.nanmedian(array)
    else:
        # If input is masked array, return a masked array
        extrap_mask = (interpolated_array != array.data)
        if isinstance(array, np.ma.masked_array):
            interpolated_array = np.ma.masked_array(interpolated_array, mask=(nan_mask & ~extrap_mask))

    return interpolated_array.reshape(array.shape)


def hypsometric_interpolation(voided_ddem: Union[np.ndarray, np.ma.masked_array],
                              ref_dem: Union[np.ndarray, np.ma.masked_array],
                              mask: np.ndarray) -> np.ma.masked_array:
    """
    Interpolate a dDEM using hypsometric interpolation within the given mask.

    The dDEM is assumed to have been created as "voided_ddem = reference_dem - other_dem".
    Areas outside the mask will be linearly interpolated, but are masked out.

    :param voided_ddem: A dDEM with voids (either an array with nans or a masked array).
    :param ref_dem: The reference DEM in the dDEM comparison.
    :param mask: A mask to delineate the area that will be interpolated (True means hypsometric will be used).
    """
    # Get ddem array with invalid pixels converted to NaN and mask of invalid pixels
    ddem, ddem_mask = xdem.spatial_tools.get_array_and_mask(voided_ddem)

    # Get ref_dem array with invalid pixels converted to NaN and mask of invalid pixels
    dem, dem_mask = xdem.spatial_tools.get_array_and_mask(ref_dem)

    # A mask of inlier values: The union of the mask and the inverted exclusion masks of both rasters.
    inlier_mask = mask & (~ddem_mask & ~dem_mask)
    if np.count_nonzero(inlier_mask) == 0:
        warnings.warn("No valid data found within mask, returning copy", UserWarning)
        return np.copy(ddem)

    # Estimate the elevation dependent gradient.
    gradient = xdem.volume.hypsometric_binning(
        ddem[inlier_mask],
        dem[inlier_mask]
    )

    #
    interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(gradient)

    gradient_model = scipy.interpolate.interp1d(
        interpolated_gradient.index.mid,
        interpolated_gradient["value"].values,
        fill_value="extrapolate"
    )

    # Create an idealized dDEM (only considering the dH gradient)
    idealized_ddem = np.zeros_like(dem)
    idealized_ddem[mask] = gradient_model(dem[mask])

    # Measure the difference between the original dDEM and the idealized dDEM
    assert ddem.shape == idealized_ddem.shape
    ddem_difference = ddem.astype("float64") - idealized_ddem.astype("float64")

    # Spatially interpolate the difference between these two products.
    #interpolated_ddem_diff = ddem_difference.copy()
    #interpolated_ddem_diff[ddem_mask] = np.nan
    # rasterio.fill.fillnodata(
    #    interpolated_ddem_diff, mask=~np.isnan(interpolated_ddem_diff))
    interpolated_ddem_diff = linear_interpolation(np.where(ddem_mask, np.nan, ddem_difference))

    # Correct the idealized dDEM with the difference to the original dDEM.
    corrected_ddem = idealized_ddem + interpolated_ddem_diff

    output = np.ma.masked_array(
        corrected_ddem,
        mask=(~mask & (ddem_mask | dem_mask))
    )

    assert output is not None

    return output


def local_hypsometric_interpolation(voided_ddem: Union[np.ndarray, np.ma.masked_array],
                                    ref_dem: Union[np.ndarray, np.ma.masked_array],
                                    mask: np.ndarray, min_coverage: float = 0.2,
                                    count_threshold: Optional[int] = 1, plot: bool = False) -> np.ma.masked_array:
    """
    Interpolate a dDEM using local hypsometric interpolation.
    The algorithm loops through each features in the vector file.

    The dDEM is assumed to have been created as "voided_ddem = reference_dem - other_dem".

    :param voided_ddem: A dDEM with voids (either an array with nans or a masked array).
    :param ref_dem: The reference DEM in the dDEM comparison.
    :param mask: A raster of same shape as voided_ddem and ref_dem, containing a diferent non-0 pixel value for \
each geometry on which to loop.
    :param min_coverage: Optional. The minimum coverage fraction to be considered for interpolation.
    :param count_threshold: Optional. A pixel count threshold to exclude during the hypsometric curve fit.
    :param plot: Set to True to display intermediate plots.

    :returns: A dDEM with gaps filled by applying a hypsometric interpolation for each geometry in mask, \
for areas filling the min_coverage criterion.
    """
    # Remove any unnecessary dimension
    orig_shape = voided_ddem.shape
    voided_ddem = voided_ddem.squeeze()
    ref_dem = ref_dem.squeeze()
    mask = mask.squeeze()

    # Check that all arrays have same dimensions
    assert voided_ddem.shape == ref_dem.shape == mask.shape

    # Get ddem array with invalid pixels converted to NaN and mask of invalid pixels
    ddem, ddem_mask = xdem.spatial_tools.get_array_and_mask(voided_ddem)

    # Get ref_dem array with invalid pixels converted to NaN and mask of invalid pixels
    dem, dem_mask = xdem.spatial_tools.get_array_and_mask(ref_dem)

    # A mask of inlier values: The union of the mask and the inverted exclusion masks of both rasters.
    inlier_mask = (mask != 0) & (~ddem_mask & ~dem_mask)
    if np.count_nonzero(inlier_mask) == 0:
        warnings.warn("No valid data found within mask, returning copy", UserWarning)
        return np.copy(ddem)

    if plot:
        plt.matshow(inlier_mask)
        plt.title("inlier mask")
        plt.show()

    # List of indexes to loop on
    geometry_index = np.unique(mask[mask != 0])
    print("Found {:d} geometries".format(len(geometry_index)))

    # Get fraction of valid pixels for each geometry
    coverage = np.zeros(len(geometry_index))
    for k, index in enumerate(geometry_index):
        local_inlier_mask = inlier_mask & (mask == index)
        total_pixels = np.count_nonzero((mask == index))
        valid_pixels = np.count_nonzero(local_inlier_mask)
        coverage[k] = valid_pixels/float(total_pixels)

    # Filter geometries with too little coverage
    valid_geometry_index = geometry_index[coverage >= min_coverage]
    print("Found {:d} geometries with sufficient coverage".format(len(valid_geometry_index)))

    idealized_ddem = -9999 * np.ones_like(dem)

    for k, index in enumerate(valid_geometry_index):

        # Mask of valid pixel within geometry
        local_mask = (mask == index)
        local_inlier_mask = inlier_mask & (local_mask)

        # Estimate the elevation dependent gradient
        gradient = xdem.volume.hypsometric_binning(
            ddem[local_mask],
            dem[local_mask]
        )

        # Remove bins with loo low count
        filt_gradient = gradient.copy()
        if count_threshold > 1:
            bins_under_threshold = filt_gradient["count"] < count_threshold
            filt_gradient.loc[bins_under_threshold, "value"] = np.nan

        # Interpolate missing elevation bins
        interpolated_gradient = xdem.volume.interpolate_hypsometric_bins(filt_gradient)

        # Create a model for 2D interpolation
        gradient_model = scipy.interpolate.interp1d(
            interpolated_gradient.index.mid,
            interpolated_gradient.values,
            fill_value="extrapolate"
        )

        if plot:
            local_ddem = np.where(local_inlier_mask, ddem, np.nan)
            vmax = max(np.abs(np.nanpercentile(local_ddem, [2, 98])))
            rowmin, rowmax, colmin, colmax = xdem.spatial_tools.get_valid_extent(mask == index)

            fig = plt.figure(figsize=(12, 8))
            plt.subplot(121)
            plt.imshow((mask == index)[rowmin:rowmax, colmin:colmax], cmap='Greys',
                       vmin=0, vmax=2, interpolation='none')

            plt.imshow(local_ddem[rowmin:rowmax, colmin:colmax], cmap='RdYlBu',
                       vmin=-vmax, vmax=vmax, interpolation='none')
            plt.colorbar()
            plt.title("ddem for geometry # {:d}".format(index))

            plt.subplot(122)
            plt.plot(gradient["value"], gradient.index.mid, label='raw')
            plt.plot(interpolated_gradient, gradient.index.mid, label='interpolated', ls='--')
            plt.xlabel('ddem')
            plt.ylabel('Elevation')
            plt.legend()
            plt.title("Average ddem per elevation bin")
            plt.tight_layout()
            plt.show()

        # Create an idealized dDEM (only considering the dH gradient)
        idealized_ddem[mask == index] = gradient_model(dem[mask == index])

    # Measure the difference between the original dDEM and the idealized dDEM
    assert ddem.shape == idealized_ddem.shape
    ddem_difference = ddem.astype("float32") - idealized_ddem.astype("float32")
    ddem_difference[idealized_ddem == -9999] = np.nan

    # Spatially interpolate the difference between these two products.
    interpolated_ddem_diff = linear_interpolation(np.where(ddem_mask, np.nan, ddem_difference))
    interpolated_ddem_diff[np.isnan(interpolated_ddem_diff)] = 0

    # Correct the idealized dDEM with the difference to the original dDEM.
    corrected_ddem = idealized_ddem + interpolated_ddem_diff

    output = np.ma.masked_array(
        corrected_ddem,
        mask=(corrected_ddem == -9999)  # mask=((mask != 0) & (ddem_mask | dem_mask))
    ).reshape(orig_shape)

    assert output is not None

    return output

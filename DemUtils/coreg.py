"""
DEM coregistration functions.


Author(s):
    Erik Schytt Holmlund (holmlund@vaw.baug.ethz.ch)
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import cv2
import numpy as np
import pdal
import rasterio as rio
import rasterio.warp
import rasterio.windows
import scipy
import scipy.interpolate
import scipy.optimize


def reproject_dem(dem: rio.DatasetReader, bounds: dict[str, float],
                  resolution: float, crs: Optional[rio.crs.CRS]) -> np.ndarray:
    """
    Reproject a DEM to the given bounds.

    param: dem: A DEM read through rasterio.
    param: bounds: The target west, east, north, and south bounding coordinates.
    param: resolution: The target resolution in metres.
    param: crs: Optional. The target CRS (defaults to the input DEM crs)

    return: destination: The elevation array in the destination bounds, resolution and CRS.
    """
    # Calculate new shape of the dataset
    dst_shape = (int((bounds["north"] - bounds["south"]) // resolution),
                 int((bounds["east"] - bounds["west"]) // resolution))

    # Make an Affine transform from the bounds and the new size
    dst_transform = rio.transform.from_bounds(**bounds, width=dst_shape[1], height=dst_shape[0])
    # Make an empty numpy array which will later be filled with elevation values
    destination = np.empty(dst_shape, dem.dtypes[0])
    # Set all values to nan right now
    destination[:, :] = np.nan

    # Reproject the DEM and put the output in the destination array
    rasterio.warp.reproject(
        source=dem.read(1),
        destination=destination,
        src_transform=dem.transform,
        dst_transform=dst_transform,
        resampling=rasterio.warp.Resampling.cubic_spline,
        src_crs=dem.crs,
        dst_crs=dem.crs if crs is None else crs
    )

    return destination


def write_geotiff(filepath: str, values: np.ndarray, crs: rio.crs.CRS, bounds: dict[str, float]) -> None:
    """
    Write a GeoTiff to the disk.

    param: filepath: The output filepath of the geotiff.
    param: values: The raster values to write.
    param: crs: The coordinate system of the raster.
    param: bounds: The bounding coordinates of the raster.
    """
    transform = rio.transform.from_bounds(**bounds, width=values.shape[1], height=values.shape[0])

    with rio.open(
            filepath,
            mode="w",
            driver="Gtiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            crs=crs,
            transform=transform,
            dtype=values.dtype) as outfile:
        outfile.write(values, 1)


def icp_coregistration(reference_filepath: str, aligned_filepath: str, output_filepath: str, pixel_buffer: int = 3) -> float:
    """
    Perform an ICP coregistration in areas where two DEMs overlap.

    param: reference_filepath: The input filepath to the DEM acting reference.
    param: aligned_filepath: The input filepath to the DEM acting aligned.
    param: output_filepath: The filepath of the aligned dataset after coregistration.
    param: pixel_buffer: The number of pixels to buffer the overlap mask with.

    return: fitness: The ICP fitness measure of the coregistration.
    """
    reference_dem = rio.open(reference_filepath)
    resolution = reference_dem.res[0]

    aligned_dem = rio.open(aligned_filepath)

    # TODO: Fix dangerous assumption here that aligned_dem has the same crs
    # Find new bounds that overlap with both datasets
    max_bounds = {
        "west": min(reference_dem.bounds.left, aligned_dem.bounds.left),
        "east": max(reference_dem.bounds.right, aligned_dem.bounds.right),
        "north": max(reference_dem.bounds.top, aligned_dem.bounds.top),
        "south": min(reference_dem.bounds.bottom, aligned_dem.bounds.bottom)
    }

    # Make the bounds correspond well to the resolution of the raster
    for corner in max_bounds:
        max_bounds[corner] -= max_bounds[corner] % resolution

    # Read and reproject the input data to the same shape
    reference = reproject_dem(reference_dem, max_bounds, resolution, crs=reference_dem.crs)
    aligned = reproject_dem(aligned_dem, max_bounds, resolution, crs=reference_dem.crs)

    # Make sure that the above step worked
    assert reference.shape == aligned.shape

    # Check where the datasets overlap (where both DEMs don't have nans)
    overlapping_nobuffer = np.logical_and(np.logical_not(np.isnan(reference)), np.logical_not(np.isnan(aligned)))
    # Buffer the mask to increase the likelyhood of including the correct values
    overlapping = scipy.ndimage.maximum_filter(overlapping_nobuffer, size=pixel_buffer, mode="constant")

    # Remove parts of the DEMs where no overlap existed
    reference[~overlapping] = np.nan
    aligned[~overlapping] = np.nan

    # Make a temporary directory to write the overlap-fixed DEMs to
    temporary_dir = tempfile.TemporaryDirectory()
    reference_temp_filepath = os.path.join(temporary_dir.name, "reference.tif")
    aligned_temp_filepath = os.path.join(temporary_dir.name, "aligned_pre_icp.tif")

    write_geotiff(reference_temp_filepath, reference, crs=reference_dem.crs, bounds=max_bounds)
    write_geotiff(aligned_temp_filepath, aligned, crs=reference_dem.crs, bounds=max_bounds)

    # Define values to fill the below pipeline with
    pdal_values = {
        "REFERENCE_FILEPATH": reference_temp_filepath,
        "ALIGNED_FILEPATH": aligned_temp_filepath,
        "OUTPUT_FILEPATH": output_filepath,
        "RESOLUTION": resolution
    }

    # Make the pipeline that will be provided to PDAL (read the two input DEMs, run ICP, save an output DEM)
    pdal_pipeline = '''
    [
        {
            "type": "readers.gdal",
            "filename": "REFERENCE_FILEPATH",
            "header": "Z"
        },
        {
            "type": "readers.gdal",
            "filename": "ALIGNED_FILEPATH",
            "header": "Z"
        },
        {
            "type": "filters.icp"
        },
        {
            "type": "writers.gdal",
            "filename": "OUTPUT_FILEPATH",
            "resolution": RESOLUTION,
            "output_type": "mean",
            "gdalopts": "COMPRESS=DEFLATE"
        }
    ]
    '''

    # Fill the pipeline "template" with appropriate values
    for key in pdal_values:
        pdal_pipeline = pdal_pipeline.replace(key, str(pdal_values[key]))

    # Make the pipeline, execute it, and extract the resultant metadata
    pipeline = pdal.Pipeline(pdal_pipeline)
    pipeline.execute()
    metadata = pipeline.metadata

    # Get the fitness value from the ICP coregistration
    fitness: float = json.loads(metadata)["metadata"]["filters.icp"]["fitness"]

    return fitness


def get_horizontal_shift(elevation_difference: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                         min_count: int = 30) -> tuple[float, float, float]:
    """
    Calculate the horizontal shift between two DEMs using the method presented in Nuth and Kääb (2011).

    param: elevation_difference: The elevation difference (reference_dem - aligned_dem).
    param: slope: A slope map with the same shape as elevation_difference (units = ??).
    param: apsect: An aspect map with the same shape as elevation_difference (units = ??).

    return: east_offset, north_offset, c_parameter: The offsets in easting, northing, and the c_parameter (altitude).
    """
    input_x_values = aspect

    with np.errstate(divide="ignore", invalid="ignore"):
        input_y_values = elevation_difference / slope

    # Remove non-finite values
    x_values = input_x_values[np.isfinite(input_x_values) & np.isfinite(input_y_values)]
    y_values = input_y_values[np.isfinite(input_x_values) & np.isfinite(input_y_values)]

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

        param: x_values: The x-values to feed the above function.
        param: parameters: The a, b, and c parameters to feed the above function

        return: estimated_ys: Estimated y-values with the same shape as the given x-values
        """
        return parameters[0] * np.cos(parameters[1] - x_values) + parameters[2]

    def residuals(parameters: tuple[float, float, float], y_values: np.ndarray, x_values: np.ndarray):
        """
        Get the residuals between the estimated and measured values using the given parameters.

        err(x, y) = est_y(x) - y

        param: parameters: The a, b, and c parameters to use for the estimation.
        param: y_values: The measured y-values.
        param: x_values: The measured x-values

        return: err: An array of residuals with the same shape as the input arrays.
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

    param: dem: A numpy array of elevation values.

    return: slope_px, aspect: The slope (in pixels??) and aspect (in radians) of the DEM.
    """

    # Calculate the gradient of the slope
    gradient_y, gradient_x = np.gradient(dem)

    slope_px = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    aspect = np.arctan(-gradient_x, gradient_y)
    aspect += np.pi

    return slope_px, aspect


def coregister_dem(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray, max_iterations: int = 200, error_threshold: float = 0.05) -> tuple[np.ndarray, float]:
    """
    Coregister a DEM using the Nuth and Kääb (2011) approach.

    param: reference_dem: The DEM acting reference.
    param: dem_to_be_aligned: The DEM to be aligned to the reference.
    param: max_iterations: The maximum of iterations to attempt the coregistration.
    param: error_threshold: The acceptable error threshold after which to stop the iterations.

    return: aligned_dem, nmad: The aligned DEM, and the NMAD (error) of the alignment.
    """
    # Make a new DEM which will be modified inplace
    aligned_dem = dem_to_be_aligned.copy()
    assert reference_dem.shape == aligned_dem.shape

    slope, aspect = calculate_slope_and_aspect(reference_dem)

    east_grid = np.arange(reference_dem.shape[1])
    north_grid = np.arange(reference_dem.shape[0])

    elevation_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=aligned_dem)
    nodata_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=np.isnan(aligned_dem))
    offset_east, offset_north = 0.0, 0.0

    for i in range(max_iterations):

        aligned_dem -= np.nanmedian(aligned_dem - reference_dem)

        elevation_difference = reference_dem - aligned_dem
        nmad = 1.4826 * np.nanmedian(np.abs(elevation_difference - np.nanmedian(elevation_difference)))

        if nmad < error_threshold:
            break

        east_offset, north_offset, _ = get_horizontal_shift(
            elevation_difference=elevation_difference,
            slope=slope,
            aspect=aspect
        )
        offset_east += east_offset
        offset_north += north_offset

        new_elevation = elevation_function(y=east_grid + offset_east, x=north_grid - offset_north)
        new_nans = nodata_function(y=east_grid + offset_east, x=north_grid - offset_north)
        new_elevation[new_nans] = np.nan

        aligned_dem = new_elevation

    print(f"Final easting offset: {offset_east} px, northing offset: {offset_north} px, NMAD: {nmad} m")


def test_icp():
    fitness = icp_coregistration(
        reference_filepath="examples/Longyearbyen/DEM_2009_ref.tif",
        aligned_filepath="examples/Longyearbyen/DEM_1995.tif",
        output_filepath="examples/Longyearbyen/DEM_1995_coreg.tif"
    )
    print(fitness)


def test_coregistration():
    reference_dem = cv2.imread("examples/Longyearbyen/DEM_2009_ref.tif", cv2.IMREAD_ANYDEPTH)

    aligned_dem = np.roll(reference_dem, shift=5, axis=0)

    coregister_dem(reference_dem, aligned_dem)


if __name__ == "__main__":

    test_coregistration()

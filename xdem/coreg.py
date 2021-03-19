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
import pyproj.crs
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import rasterio.windows  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from rasterio import Affine
from tqdm import trange

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


def write_geotiff(filepath: str, values: np.ndarray, crs: Optional[rio.crs.CRS],
                  bounds: rio.coords.BoundingBox) -> None:
    """
    Write a GeoTiff to the disk.

    :param filepath: The output filepath of the geotiff.
    :param values: The raster values to write.
    :param crs: The coordinate system of the raster.
    :param bounds: The bounding coordinates of the raster.
    """
    transform = rio.transform.from_bounds(*bounds, width=values.shape[1], height=values.shape[0])

    nodata_value = np.finfo(values.dtype).min

    # Replace np.nan with the minimum value for the dtype
    values[np.isnan(values)] = nodata_value

    with rio.open(
            filepath,
            mode="w",
            driver="Gtiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            crs=crs,
            transform=transform,
            dtype=values.dtype,
            nodata=nodata_value) as outfile:
        outfile.write(values, 1)


def run_pdal_pipeline(pipeline: str, output_metadata_file: Optional[str] = None,
                      parameters: Optional[dict[str, str]] = None, show_warnings: bool = False) -> dict[str, Any]:
    """
    Run a PDAL pipeline.

    :param pipeline: The pipeline to run.
    :param output_metadata_file: Optional. The filepath for the pipeline metadata.
    :param parameters: Optional. Parameters to fill the pipeline with, e.g. {"FILEPATH": "/path/to/file"}.
    :param show_warnings: Show the full stdout of the PDAL process.

    :raises ValueError: If the PDAL pipeline string is poorly formatted.

    :returns: output_meta: The metadata produced by the output.
    """
    # Create a temporary directory to save the output metadata in
    temp_dir = tempfile.TemporaryDirectory()
    # Fill the pipeline with parameters if given
    if parameters is not None:
        for key in parameters:
            # Warn if the key cannot be found in the pipeline
            if key not in pipeline and show_warnings:
                warnings.warn(
                    f"{key}:{parameters[key]} given to the PDAL pipeline but the key was not found", RuntimeWarning)
            # Replace every occurrence of the key inside the pipeline with its corresponding value
            pipeline = pipeline.replace(key, str(parameters[key]))

    try:
        json.loads(pipeline)  # Throws an error if the pipeline is poorly formatted
    except json.decoder.JSONDecodeError as exception:
        raise ValueError("Pipeline was poorly formatted: \n" + pipeline + "\n" + str(exception))

    # Run PDAL with the pipeline as the stdin
    commands = ["pdal", "pipeline", "--stdin", "--metadata", os.path.join(temp_dir.name, "meta.json")]
    stdout = subprocess.run(commands, input=pipeline, check=True, stdout=subprocess.PIPE,
                            encoding="utf-8", stderr=subprocess.PIPE).stdout

    if show_warnings and len(stdout.strip()) != 0:
        print(stdout)

    # Load the temporary metadata file
    with open(os.path.join(temp_dir.name, "meta.json")) as infile:
        output_meta = json.load(infile)

    # Save it with a different name if one was provided
    if output_metadata_file is not None:
        with open(output_metadata_file, "w") as outfile:
            json.dump(output_meta, outfile)

    return output_meta


def check_for_pdal(min_version="2.2.0") -> None:
    """
    Check that PDAL is installed and that it fills the minimum version requirement.

    :param min_version: The minimum allowed PDAL version.

    :raises AssertionError: If PDAL is not found or the version requirement is not met.
    """
    # Try to run pdal --version (will error out if PDAL is not installed.
    try:
        result = subprocess.run(["pdal", "--version"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding="utf-8", check=True)
    except FileNotFoundError:
        raise AssertionError("PDAL not found in path. Install it or check the $PATH variable")

    # Parse the pdal --version output
    for line in result.stdout.splitlines():
        if not "pdal" in line:
            continue
        version = line.split(" ")[1]

    # Fine if the minimum version is the installed version
    if version == min_version:
        return

    # If the version string is sorted before the min_version string, the version is too low.
    if sorted([version, min_version]).index(version) == 0:
        raise AssertionError(f"Installed PDAL has version: {version}, min required version is {min_version}")


def icp_coregistration_pdal(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray, bounds: rio.coords.BoundingBox,
                            mask: Optional[np.ndarray] = None, max_assumed_offset: Optional[float] = None,
                            verbose=False, metadata: Optional[dict[str, Any]] = None, **_) -> tuple[np.ndarray, float]:
    """
    Perform an ICP coregistration in areas where two DEMs overlap.

    :param reference_dem: The input array of the DEM acting reference.
    :param dem_to_be_aligned: The input array to the DEM acting aligned.
    :param bounds: The bounding coordinates of the reference_dem.
    :param mask: A boolean array of areas to exclude from the coregistration.
    :param max_assumed_offset: The maximum assumed offset between the DEMs in georeferenced horizontal units.
    :param verbose: Print progress messages.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "icp".

    :returns: The aligned DEM (array) and the NMAD error.

    # noqa: DAR101 **_
    """
    dem_to_be_aligned_unmasked = dem_to_be_aligned.copy()
    # Check that PDAL is installed.
    check_for_pdal()

    resolution = (bounds.right - bounds.left) / reference_dem.shape[1]
    # Make sure that the DEMs have the same shape
    assert reference_dem.shape == dem_to_be_aligned.shape, (reference_dem.shape, dem_to_be_aligned.shape)

    # Check where the datasets overlap (where both DEMs don't have nans)
    overlapping_nobuffer = np.logical_and(np.logical_not(np.isnan(reference_dem)),
                                          np.logical_not(np.isnan(dem_to_be_aligned)))
    # Buffer the mask to increase the likelyhood of including the correct values
    pixel_buffer = 3 if max_assumed_offset is None else int(max_assumed_offset / resolution)
    overlapping = scipy.ndimage.maximum_filter(overlapping_nobuffer, size=pixel_buffer, mode="constant")

    # Remove parts of the DEMs where no overlap existed
    reference_dem[~overlapping] = np.nan
    dem_to_be_aligned[~overlapping] = np.nan

    if mask is not None:
        reference_dem[mask] = np.nan
        dem_to_be_aligned[mask] = np.nan

    # Generate center point coordinates for each pixel
    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, num=reference_dem.shape[1]),
        np.linspace(bounds.bottom + resolution / 2, bounds.top -
                    resolution / 2, num=reference_dem.shape[0])[::-1]
    )

    # Make a temporary directory to write the overlap-fixed DEMs to
    temporary_dir = tempfile.TemporaryDirectory()
    reference_temp_filepath = os.path.join(temporary_dir.name, "reference.xyz")
    tba_temp_filepath = os.path.join(temporary_dir.name, "tba.xyz")
    tba_nonmasked_temp_filepath = os.path.join(temporary_dir.name, "tba_nonmasked.tif")
    output_dem_filepath = os.path.join(temporary_dir.name, "output_dem.tif")

    # Save the x, y and z coordinates into a temporary xyz point cloud which will be read by PDAL.
    for path, elev in zip([reference_temp_filepath, tba_temp_filepath], [reference_dem, dem_to_be_aligned]):
        data = np.dstack([
            x_coords[~np.isnan(elev)],
            y_coords[~np.isnan(elev)],
            elev[~np.isnan(elev)]
        ]).squeeze()
        assert data.shape[1] == 3, data.shape
        np.savetxt(path, data, delimiter=",", header="X,Y,Z")

    # Define values to fill the below pipeline with
    pdal_parameters = {
        "REFERENCE_FILEPATH": reference_temp_filepath,
        "ALIGNED_FILEPATH": tba_temp_filepath,
        "OUTPUT_FILEPATH": output_dem_filepath,
        "RESOLUTION": resolution,
        "WIDTH": reference_dem.shape[1],
        "HEIGHT": reference_dem.shape[0],
        "ORIGIN_X": bounds.left,
        "ORIGIN_Y": bounds.bottom
    }

    # Make the pipeline that will be provided to PDAL (read the two input DEMs, run ICP, save an output DEM)
    pdal_pipeline = '''
    [
        {
            "type": "readers.text",
            "filename": "REFERENCE_FILEPATH",
            "header": "X,Y,Z"
        },
        {
            "type": "readers.text",
            "filename": "ALIGNED_FILEPATH",
            "header": "X,Y,Z"
        },
        {
            "type": "filters.icp"
        },
        {
            "type": "writers.gdal",
            "filename": "OUTPUT_FILEPATH",
            "resolution": RESOLUTION,
            "output_type": "mean",
            "gdalopts": "COMPRESS=DEFLATE",
            "origin_x": ORIGIN_X,
            "origin_y": ORIGIN_Y,
            "width": WIDTH,
            "height": HEIGHT
        }
    ]
    '''

    if verbose:
        print("Running ICP coregistration...")
    pdal_metadata = run_pdal_pipeline(pdal_pipeline, parameters=pdal_parameters)

    if verbose:
        print("Done")
    aligned_dem_reader = rio.open(output_dem_filepath)
    aligned_dem = aligned_dem_reader.read(1)
    aligned_dem[aligned_dem == aligned_dem_reader.nodata] = np.nan

    # Calculate the NMAD from the elevation difference between the reference and aligned DEM
    elevation_difference = reference_dem - aligned_dem
    nmad = calculate_nmad(elevation_difference)

    # If a mask was given, correct the unmasked DEM using the estimated transform
    if mask is not None:
        write_geotiff(tba_nonmasked_temp_filepath, dem_to_be_aligned_unmasked, crs=None, bounds=bounds)
        pdal_parameters["INPUT_FILEPATH"] = tba_nonmasked_temp_filepath
        pdal_parameters["MATRIX"] = pdal_metadata["stages"]["filters.icp"]["composed"].replace("\n", " ")
        pdal_pipeline = """
        [
            {
                "type": "readers.gdal",
                "filename": "INPUT_FILEPATH",
                "header": "Z"

            },
            {
                "type": "filters.transformation",
                "matrix": "MATRIX"
            },
            {
                "type": "writers.gdal",
                "filename": "OUTPUT_FILEPATH",
                "resolution": RESOLUTION,
                "output_type": "mean",
                "gdalopts": "COMPRESS=DEFLATE",
                "origin_x": ORIGIN_X,
                "origin_y": ORIGIN_Y,
                "width": WIDTH,
                "height": HEIGHT
            }
        ]"""
        run_pdal_pipeline(pdal_pipeline, parameters=pdal_parameters)
        aligned_dem = rio.open(output_dem_filepath).read(1)

    if metadata is not None:
        metadata["icp_pdal"] = pdal_metadata["stages"]["filters.icp"]
        metadata["icp_pdal"]["nmad"] = nmad

    return aligned_dem, nmad


def icp_coregistration_opencv(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray, bounds: rio.coords.BoundingBox,
                              mask: Optional[np.ndarray] = None, max_iterations=100, tolerance=0.05,
                              rejection_scale=2.5, num_levels=6, metadata: Optional[dict[str, Any]] = None,
                              **_) -> tuple[np.ndarray, float]:
    """
    Coregister one DEM to a reference DEM.

    :param reference_dem: The input array of the DEM acting reference.
    :param dem_to_be_aligned: The input array to the DEM acting aligned.
    :param bounds: The bounding coordinates of the reference_dem.
    :param mask: A boolean array of areas to exclude from the coregistration.
    :param max_iterations: The maximum amount of iterations to run ICP.
    :param tolerance: The minimum difference between iterations after which to stop.
    :param rejection_scale: The threshold for outliers to be considered (scale * standard-deviation of residuals).
    :param num_levels: Number of octrees to consider. A higher number is faster but may be more inaccurate.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "icp".

    :returns: The aligned DEM (array) and the NMAD error.

    # noqa: DAR101 **_
    """
    if not _has_cv2:
        raise AssertionError("opencv (cv2) is not available and needs to be installed.")
    dem_to_be_aligned_unmasked = dem_to_be_aligned.copy()

    resolution = reference_dem.shape[1] / (bounds.right - bounds.left)
    points: dict[str, np.ndarray] = {}
    x_coords, y_coords = np.meshgrid(
        np.linspace(bounds.left + resolution / 2, bounds.right - resolution / 2, num=reference_dem.shape[1]),
        np.linspace(bounds.bottom + resolution / 2, bounds.top - resolution / 2, num=reference_dem.shape[0])[::-1]
    )
    x_coords -= bounds.left
    y_coords -= bounds.bottom

    if mask is not None:
        reference_dem[mask] = np.nan
        dem_to_be_aligned[mask] = np.nan

    for key, dem in zip(["ref", "tba", "tba_unmasked"], [reference_dem, dem_to_be_aligned, dem_to_be_aligned_unmasked]):

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

    icp = cv2.ppf_match_3d_ICP(max_iterations, tolerance, rejection_scale, num_levels)
    _, residual, transform = icp.registerModelToScene(points["tba"], points["ref"])

    assert residual < 1000, f"ICP coregistration failed: {residual=}, threshold: 1000"

    transformed_points = cv2.perspectiveTransform(points["tba_unmasked"][:, :3].reshape(1, -1, 3), transform).squeeze()

    aligned_dem = scipy.interpolate.griddata(
        points=transformed_points[:, :2],
        values=transformed_points[:, 2],
        xi=tuple(np.meshgrid(
            np.linspace(bounds.left, bounds.right, reference_dem.shape[1]) - bounds.left,
            np.linspace(bounds.bottom, bounds.top, reference_dem.shape[0])[::-1] - bounds.bottom
        )),
        method="nearest"
    )
    aligned_dem[np.isnan(dem_to_be_aligned_unmasked)] = np.nan

    nmad = calculate_nmad(aligned_dem - reference_dem)

    if metadata is not None:
        metadata["icp_opencv"] = {
            "transform": transform,
            "residual": residual,
            "nmad": nmad
        }

    return aligned_dem, nmad


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
              degree: int, verbose: bool = False, max_npts: int = 500_000,
              metadata: Optional[dict[str, Any]] = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Calculate a deramping function to account for rotational and non-rigid components of the elevation difference.

    :param elevation_difference: The elevation difference array to analyse.
    :param x_coordinates: x-coordinates of the above array (must have the same shape as elevation_difference)
    :param y_coordinates: y-coordinates of the above array (must have the same shape as elevation_difference)
    :param degree: The polynomial degree to estimate the ramp.
    :param max_npts: Maximum number of points to randomly extract.
    :param verbose: Print the least squares optimization progress.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "deramp".

    :returns: A callable function to estimate the ramp.
    """
    # Extract only the finite values of the elevation difference and corresponding coordinates.
    if isinstance(elevation_difference, np.ma.masked_array):
        valid_diffs = elevation_difference[~elevation_difference.mask]
        valid_x_coords = x_coordinates[~elevation_difference.mask]
        valid_y_coords = y_coordinates[~elevation_difference.mask]
    else:
        valid_diffs = elevation_difference[np.isfinite(elevation_difference)]
        valid_x_coords = x_coordinates[np.isfinite(elevation_difference)]
        valid_y_coords = y_coordinates[np.isfinite(elevation_difference)]

    # Randomly subsample the values if there are more than 500,000 of them.
    if len(valid_x_coords) > max_npts:
        random_indices = np.random.randint(0, len(valid_x_coords) - 1, max_npts)
        valid_diffs = valid_diffs[random_indices]
        valid_x_coords = valid_x_coords[random_indices]
        valid_y_coords = valid_y_coords[random_indices]

    # Create a function whose residuals will be attempted to minimise
    def estimate_values(x_coordinates: np.ndarray, y_coordinates: np.ndarray,
                        coefficients: np.ndarray, degree: int) -> np.ndarray:
        """
        Estimate values from a 2D-polynomial.

        :param x_coordinates: x-coordinates of the difference array (same shape as elevation_difference).
        :param y_coordinates: y-coordinates of the difference array (same shape as elevation_difference).
        :param coefficients: The coefficients (a, b, c, etc.) of the polynomial.
        :param degree: The degree of the polynomial.

        :raises ValueError: If the length of the coefficients list is not compatible with the degree.

        :returns: The values estimated by the polynomial.
        """
        # Check that the coefficient size is correct.
        coefficient_size = (degree + 1) * (degree + 2) / 2
        if len(coefficients) != coefficient_size:
            raise ValueError("Number of coefficients must be equal to",
                             (degree + 1) * (degree + 2) / 2)

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
            "nmad": calculate_nmad(residuals(coefficients, valid_diffs, valid_x_coords, valid_y_coords, degree))
        }

    # Return the function which can be used later.
    return ramp


def deramp_dem(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray, mask: Optional[np.ndarray] = None,
               deramping_degree: int = 1, verbose: bool = True,
               metadata: Optional[dict[str, Any]] = None, **_) -> tuple[np.ndarray, float]:
    """
    Deramp the given DEM using a reference DEM.

    :param reference_dem: The input array of the DEM acting reference.
    :param dem_to_be_aligned: The input array to the DEM acting aligned.
    :param mask: Optional. A boolean array to exclude areas from the analysis.
    :param deramping_degree: The polynomial degree to estimate the ramp with.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "deramp".

    :returns: The aligned DEM (array) and the NMAD error.

    # noqa: DAR101 **_
    """
    elevation_difference = (reference_dem - dem_to_be_aligned)

    # Apply the mask if it exists.
    if mask is not None:
        elevation_difference[mask] = np.nan

    # Generate arbitrary x- and y- coordinates to supply the deramping function with.
    x_coordinates, y_coordinates = np.meshgrid(
        np.linspace(0, 1, num=elevation_difference.shape[1]),
        np.linspace(0, 1, num=elevation_difference.shape[0])
    )

    # Estimate the ramp function.
    ramp = deramping(elevation_difference, x_coordinates, y_coordinates, deramping_degree,
                     verbose=verbose, metadata=metadata)

    # Correct the elevation difference with the ramp and measure the error.
    elevation_difference -= ramp(x_coordinates, y_coordinates)
    error = calculate_nmad(elevation_difference)

    # Correct the DEM to be aligned with the ramp
    aligned_dem = dem_to_be_aligned + ramp(x_coordinates, y_coordinates)

    return aligned_dem, error


def calculate_nmad(array: np.ndarray) -> float:
    """
    Calculate the normalized median absolute deviation of an array.

    :param array: A one- or multidimensional array.

    :returns: The NMAD of the array.
    """
    # TODO: Get a reference for why NMAD is used (and make sure the N stands for normalized)
    nmad = 1.4826 * np.nanmedian(np.abs(array - np.nanmedian(array)))

    return nmad


def amaury_coregister_dem(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray, mask: Optional[np.ndarray] = None,
                          max_iterations: int = 50, error_threshold: float = 0.05,
                          deramping_degree: Optional[int] = 1, verbose: bool = True,
                          metadata: Optional[dict[str, Any]] = None, **_) -> tuple[np.ndarray, float]:
    """
    Coregister a DEM using the Nuth and Kääb (2011) approach.

    :param reference_dem: The DEM acting reference.
    :param dem_to_be_aligned: The DEM to be aligned to the reference.
    :param mask: A boolean array of areas to exclude from the coregistration.
    :param max_iterations: The maximum of iterations to attempt the coregistration.
    :param error_threshold: The acceptable error threshold after which to stop the iterations.
    :param deramping_degree: Optional. The polynomial degree to estimate for deramping the offset field.
    :param verbose: Whether to print the progress or not.
    :param metadata: Optional. A metadata dictionary that will be updated with the key "nuth_kaab".

    :returns: The aligned DEM, and the NMAD (error) of the alignment.

    # noqa: DAR101 **_
    """
    # TODO: Add offset_east and offset_north as return variables?
    # Make a new DEM which will be modified inplace
    aligned_dem = dem_to_be_aligned.copy()
    reference_dem = reference_dem.copy()
    if mask is not None:
        aligned_dem[mask] = np.nan
        reference_dem[mask] = np.nan

    # Make sure that the DEMs have the same shape
    assert reference_dem.shape == aligned_dem.shape

    # Calculate slope and aspect maps from the reference DEM
    slope, aspect = calculate_slope_and_aspect(reference_dem)

    # Make index grids for the east and north dimensions
    east_grid = np.arange(reference_dem.shape[1])
    north_grid = np.arange(reference_dem.shape[0])

    # Make a function to estimate the aligned DEM (used to construct an offset DEM)
    elevation_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid,
                                                               z=np.where(np.isnan(aligned_dem), -9999, aligned_dem))
    # Make a function to estimate nodata gaps in the aligned DEM (used to fix the estimated offset DEM)
    nodata_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=np.isnan(aligned_dem))
    # Initialise east and north pixel offset variables (these will be incremented up and down)
    offset_east, offset_north = 0.0, 0.0

    # Iteratively run the analysis until the maximum iterations or until the error gets low enough
    for i in trange(max_iterations, disable=(not verbose), desc="Iteratively correcting dataset"):

        # Remove potential biases between the DEMs
        aligned_dem -= np.nanmedian(aligned_dem - reference_dem)

        # Calculate the elevation difference and the residual (NMAD) between them.
        elevation_difference = reference_dem - aligned_dem

        nmad = calculate_nmad(elevation_difference)

        assert ~np.isnan(nmad), (offset_east, offset_north)

        # Stop if the NMAD is low and a few iterations have been made
        if i > 5 and nmad < error_threshold:
            if verbose:
                print(f"NMAD went below the error threshold of {error_threshold}")
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

    if verbose:
        print(f"Final easting offset: {offset_east:.2f} px, northing offset: {offset_north:.2f} px, NMAD: {nmad:.3f} m")

    # Try to account for rotations between the datasets
    if deramping_degree is not None:

        # Calculate the elevation difference and the residual (NMAD) between them.
        elevation_difference = reference_dem - aligned_dem
        nmad = calculate_nmad(elevation_difference)

        # Remove outliers with an offset higher than three times the NMAD
        elevation_difference[np.abs(elevation_difference - np.nanmedian(elevation_difference)) > 3 * nmad] = np.nan

        # TODO: This makes the analysis georeferencing-invariant. Does this change the results?
        x_coordinates, y_coordinates = np.meshgrid(
            np.arange(elevation_difference.shape[1]),
            np.arange(elevation_difference.shape[0])
        )

        # Estimate the deramping function.
        ramp = deramping(
            elevation_difference=elevation_difference,
            x_coordinates=x_coordinates,
            y_coordinates=y_coordinates,
            degree=deramping_degree,
            verbose=verbose,
        )
        # Apply the deramping function to the dataset
        aligned_dem += ramp(x_coordinates, y_coordinates)

        # Calculate the final residual error of the analysis
        elevation_difference = reference_dem - aligned_dem
        nmad = calculate_nmad(elevation_difference)

        if verbose:
            print(f"NMAD after deramping (degree: {deramping_degree}): {nmad:.3f} m")

    if mask is not None:
        full_aligned_dem = dem_to_be_aligned.copy()
        # Make new functions using the full dataset instead of just the masked one.
        elevation_function = scipy.interpolate.RectBivariateSpline(
            x=north_grid,
            y=east_grid,
            z=np.where(np.isnan(full_aligned_dem), -9999, full_aligned_dem)
        )
        nodata_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=np.isnan(full_aligned_dem))

        aligned_dem = elevation_function(y=east_grid + offset_east, x=north_grid - offset_north)
        nans = nodata_function(y=east_grid + offset_east, x=north_grid - offset_north)
        aligned_dem[nans != 0] = np.nan

        if deramping_degree is not None:
            aligned_dem += ramp(x_coordinates, y_coordinates)

    if metadata is not None:
        metadata["nuth_kaab"] = {"offset_east_px": offset_east, "offset_north_px": offset_north, "nmad": nmad,
                                 "deramping_degree": deramping_degree}

    return aligned_dem, nmad


class CoregMethod(Enum):
    """A selection of a coregistration method."""

    ICP_PDAL = icp_coregistration_pdal
    ICP_OPENCV = icp_coregistration_opencv
    AMAURY = amaury_coregister_dem
    DERAMP = deramp_dem

    @staticmethod
    def from_str(string: str) -> CoregMethod:
        """
        Try to parse a coregistration method from a string.

        :param string: The string to attempt to parse.

        :raises ValueError: If the string could not be parsed.

        :returns: The parsed coregistration method.
        """
        valid_strings = {
            "icp_pdal": CoregMethod.ICP_PDAL,
            "icp": CoregMethod.ICP_OPENCV,
            "icp_opencv": CoregMethod.ICP_OPENCV,
            "amaury": CoregMethod.AMAURY,
            "nuth_kaab": CoregMethod.AMAURY,
            "deramp": CoregMethod.DERAMP
        }

        if string in valid_strings:
            return valid_strings[string]

        raise ValueError(f"'{string}' could not be parsed as a coregistration method."
                         f" Options: {list(valid_strings.keys())}")


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
        mask_array = mask.create_mask(reference_raster) == 255
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


def coregister(reference_raster: Union[str, gu.georaster.Raster], to_be_aligned_raster: Union[str, gu.georaster.Raster],
               method: Union[CoregMethod, str] = "nuth_kaab",
               mask: Optional[Union[str, gu.geovector.Vector, gu.georaster.Raster]] = None,
               verbose=True, **kwargs) -> tuple[gu.georaster.Raster, float]:
    """
    Coregister one DEM to another.

    The reference DEM must have the same X and Y resolution.

    :param reference_raster: The raster object or filepath to act as reference.
    :param to_be_aligned_raster: The raster object or filepath to be aligned.
    :param method: The coregistration method to use.
    :param mask: Optional. A Vector or Raster mask to exclude for the coregistration (e.g. glaciers).
    :param verbose: Whether to visually show the progress.
    :param **kwargs: Optional keyword arguments to feed the chosen coregistration method.

    :returns: A coregistered Raster and the NMAD of the (potentially) masked offsets.
    """
    # Load GeoUtils Rasters/Vectors if filepaths are given as arguments.
    if isinstance(reference_raster, str):
        reference_raster = gu.georaster.Raster(reference_raster)
    if isinstance(to_be_aligned_raster, str):
        to_be_aligned_raster = gu.georaster.Raster(to_be_aligned_raster)

    if isinstance(method, str):
        method = CoregMethod.from_str(method)
    # Make sure that the data is read into memory
    if reference_raster.data is None:
        reference_raster.load(1)

    # Make sure that the input data are in a float format.
    if reference_raster.data.dtype not in [np.float32, np.float64]:
        reference_raster.set_dtypes(np.float32, update_array=True)
    if to_be_aligned_raster.data.dtype not in [np.float32, np.float64]:
        to_be_aligned_raster.set_dtypes(np.float32, update_array=True)

    mask_array = mask_as_array(reference_raster, mask) if mask is not None else None

    assert np.diff(reference_raster.res)[0] == 0, "The X and Y resolution of the reference needs to be the same."

    to_be_aligned_dem = to_be_aligned_raster.reproject(reference_raster).data.squeeze()
    reference_dem = reference_raster.data.squeeze().copy()  # type: ignore

    # Set nodata values to nans
    to_be_aligned_dem[to_be_aligned_dem == to_be_aligned_raster.nodata] = np.nan
    reference_dem[reference_dem == reference_raster.nodata] = np.nan

    # Align the raster using the selected method. This returns a numpy array and the corresponding error
    aligned_dem, error = method(  # type: ignore
        reference_dem=reference_dem,
        dem_to_be_aligned=to_be_aligned_dem,
        mask=mask_array,
        bounds=reference_raster.bounds,
        verbose=verbose,
        **kwargs
    )

    # Construct a raster from the created numpy array
    aligned_raster = gu.georaster.Raster.from_array(
        data=aligned_dem,
        transform=reference_raster.transform,
        crs=reference_raster.crs
    )

    return aligned_raster, error


def superimpose(pc1: np.ndarray, pc2: np.ndarray,
                       weights: np.ndarray = None, allow_rescale: bool = False) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Takes two lists of xyz coordinates, (of the same length)
    and attempts to superimpose them using rotations, translations, and 
    (optionally) rescale operations in order to minimize the 
    root-mean-squared-distance (RMSD) between them.
    These operations should be applied to the "pc2" argument.

    This function implements a more general variant of the method from:
    R. Diamond, (1988)
    "A Note on the Rotational Superposition Problem",
    Acta Cryst. A44, pp. 211-216
    This version has been augmented slightly.  The version in the original 
    paper only considers rotation and translation and does not allow the 
    coordinates of either object to be rescaled (multiplication by a scalar).

    This code is largely inspired from https://github.com/jewettaij/superpose3d

    :param pc1: First point-cloud, array of dimension (N, 3)
    :param pc2: Second point-cloud, array of dimension (N, 3)
    :param weights: optional weights for the calculation of RMSD (same shape as PCs)
    :param allow_rescale: Attempt to rescale second point cloud

    :returns:
      (RMSD, optimal_translation, optimal_rotation, and optimal_scale_factor)
    """
    # Make sure input array are np arrays
    pc1 = np.asarray(pc1)
    pc2 = np.asarray(pc2)
    N = pc1.shape[0]

    # Checj arrays have same size
    if pc1.shape[0] != pc2.shape[0]:
        raise ValueError("Inputs should have the same size.")

    # Convert weights into array
    if (weights is None) or (len(weights) == 0):
        weights = np.full((N, 1), 1.0)
    else:
        # reshape weights so multiplications are done column-wise
        weights = np.array(weights).reshape(N, 1)

    # Find the center of mass of each object:
    center1 = np.sum(pc1 * weights, axis=0)
    center2 = np.sum(pc2 * weights, axis=0)

    sum_weights = np.sum(weights, axis=0)
    if sum_weights != 0:
        center1 /= sum_weights
        center2 /= sum_weights

    # Subtract the centers-of-mass from the original coordinates for each object
    pc1_centered = pc1 - center1
    pc2_centered = pc2 - center2

    # Calculate the "M" array from the Diamond paper (equation 16)
    M = np.matmul(pc2_centered.T, (pc1_centered * weights))

    # Calculate Q (equation 17)
    Q = M + M.T - 2*np.eye(3)*np.trace(M)

    # Calculate V (equation 18)
    V = np.empty(3)
    V[0] = M[1][2] - M[2][1]
    V[1] = M[2][0] - M[0][2]
    V[2] = M[0][1] - M[1][0]

    # Calculate "P" (equation 22)
    P = np.zeros((4, 4))
    P[:3, :3] = Q
    P[3, :3] = V
    P[:3, 3] = V

    # Calculate "p".
    # "p" contains the optimal rotation (in backwards-quaternion format)
    # (Note: A discussion of various quaternion conventions is included below.)
    # First, specify the default value for p:
    p = np.zeros(4)
    p[3] = 1.0           # p = [0,0,0,1]    default value
    pPp = 0.0            # = p^T * P * p    (zero by default)
    singular = (N < 2)   # (it doesn't make sense to rotate a single point)

    try:
        #http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
        aEigenvals, aaEigenvects = np.linalg.eigh(P)

    except np.linalg.LinAlgError:
        singular = True  # (I have never seen this happen.)

    if not singular:  # (don't crash if the caller supplies nonsensical input)
        i_eval_max = np.argmax(aEigenvals)
        pPp = np.max(aEigenvals)
        p[:] = aaEigenvects[:, i_eval_max]

    # normalize the vector
    # (It should be normalized already, but just in case it is not, do it again)
    p /= np.linalg.norm(p)

    # Finally, calculate the rotation matrix corresponding to "p"
    # (p is in backwards-quaternion format)

    aaRotate = np.empty((3, 3))
    aaRotate[0][0] = (p[0]*p[0])-(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aaRotate[1][1] = -(p[0]*p[0])+(p[1]*p[1])-(p[2]*p[2])+(p[3]*p[3])
    aaRotate[2][2] = -(p[0]*p[0])-(p[1]*p[1])+(p[2]*p[2])+(p[3]*p[3])
    aaRotate[0][1] = 2*(p[0]*p[1] - p[2]*p[3])
    aaRotate[1][0] = 2*(p[0]*p[1] + p[2]*p[3])
    aaRotate[1][2] = 2*(p[1]*p[2] - p[0]*p[3])
    aaRotate[2][1] = 2*(p[1]*p[2] + p[0]*p[3])
    aaRotate[0][2] = 2*(p[0]*p[2] + p[1]*p[3])
    aaRotate[2][0] = 2*(p[0]*p[2] - p[1]*p[3])

    # Alternatively, in modern python versions, this code also works:
    """
    from scipy.spatial.transform import Rotation as R
    the_rotation = R.from_quat(p)
    aaRotate = the_rotation.as_matrix()
    """

    # Optional: Decide the scale factor, c
    c = 1.0   # by default, don't rescale the coordinates
    if allow_rescale and (not singular):
        Waxaixai = np.sum(weights * pc2_centered ** 2)
        WaxaiXai = np.sum(weights * pc1_centered ** 2)

        c = (WaxaiXai + pPp) / Waxaixai

    # Finally compute the RMSD between the two coordinate sets:
    # First compute E0 from equation 24 of the paper
    E0 = np.sum((pc1_centered - c*pc2_centered)**2)
    sum_sqr_dist = max(0, E0 - c * 2.0 * pPp)

    rmsd = 0.0
    if sum_weights != 0.0:
        rmsd = np.sqrt(sum_sqr_dist/sum_weights)

    # Lastly, calculate the translational offset:
    # Recall that:
    #RMSD=sqrt((Σ_i  w_i * |X_i - (Σ_j c*R_ij*x_j + T_i))|^2) / (Σ_j w_j))
    #    =sqrt((Σ_i  w_i * |X_i - x_i'|^2) / (Σ_j w_j))
    #  where
    # x_i' = Σ_j c*R_ij*x_j + T_i
    #      = Xcm_i + c*R_ij*(x_j - xcm_j)
    #  and Xcm and xcm = center_of_mass for the frozen and mobile point clouds
    #                  = center1[]       and       center2[],  respectively
    # Hence:
    #  T_i = Xcm_i - Σ_j c*R_ij*xcm_j  =  aTranslate[i]
    aTranslate = center1 - np.matmul(c*aaRotate, center2).T.reshape(3,)

    return rmsd, aaRotate, aTranslate, c

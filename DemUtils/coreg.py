"""
DEM coregistration functions.

"Amaury's" functions are adapted from Amaury Dehecq.
Source: https://github.com/GeoUtils/geoutils/blob/master/geoutils/dem_coregistration.py

Author(s):
    Erik Schytt Holmlund (holmlund@vaw.baug.ethz.ch)

Date: 13 November 2020.
"""
from __future__ import annotations
import richdem as rd
from rasterio import Affine
import json
import os
import tempfile

from enum import Enum
from typing import Callable, Optional

import geoutils as gu
import numpy as np
import pdal
import rasterio as rio
import rasterio.warp  # pylint: disable=unused-import
import rasterio.windows  # pylint: disable=unused-import
import scipy
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
from tqdm import trange

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
   
    #Update geotransform
    ds_meta = ds.meta
    gt_orig = ds.transform
    gt_align = Affine(gt_orig.a, gt_orig.b, gt_orig.c+dx, \
                   gt_orig.d, gt_orig.e, gt_orig.f+dy)

    print("Original transform:", gt_orig)
    print("Updated transform:", gt_shift)

    #Update ds Geotransform
    ds_align = ds
    meta_update = ds.meta.copy()
    meta_update({"driver": "GTiff", "height": ds.shape[1],
                 "width": ds.shape[2], "transform": gt_align, "crs": ds.crs})
    #to split this part in two?
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

def rio_to_rda(ds:rio.DatasetReader)->rd.rdarray:
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

def get_terrainattr(ds:rio.DatasetReader,attrib='slope_degrees')->rd.rdarray:
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

def reproject_dem(dem: rio.DatasetReader, bounds: dict[str, float],
                  resolution: float, crs: Optional[rio.crs.CRS]) -> np.ndarray:
    """
    Reproject a DEM to the given bounds.

    :param dem: A DEM read through rasterio.
    :param bounds: The target west, east, north, and south bounding coordinates.
    :param resolution: The target resolution in metres.
    :param crs: Optional. The target CRS (defaults to the input DEM crs)

    :returns: The elevation array in the destination bounds, resolution and CRS.
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
    rio.warp.reproject(
        source=dem.read(1),
        destination=destination,
        src_transform=dem.transform,
        dst_transform=dst_transform,
        resampling=rio.warp.Resampling.cubic_spline,
        src_crs=dem.crs,
        dst_crs=dem.crs if crs is None else crs
    )

    return destination


def write_geotiff(filepath: str, values: np.ndarray, crs: rio.crs.CRS, bounds: dict[str, float]) -> None:
    """
    Write a GeoTiff to the disk.

    :param filepath: The output filepath of the geotiff.
    :param values: The raster values to write.
    :param crs: The coordinate system of the raster.
    :param bounds: The bounding coordinates of the raster.
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


def icp_coregistration(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray,
                       bounds: dict[str, float], crs=rio.crs.CRS, pixel_buffer: int = 3) -> tuple[np.ndarray, float]:
    """
    Perform an ICP coregistration in areas where two DEMs overlap.

    :param reference_filepath: The input filepath to the DEM acting reference.
    :param aligned_filepath: The input filepath to the DEM acting aligned.
    :param output_filepath: The filepath of the aligned dataset after coregistration.
    :param pixel_buffer: The number of pixels to buffer the overlap mask with.

    :returns: The ICP fitness measure of the coregistration.
    """
    # Make sure that the above step worked
    assert reference_dem.shape == dem_to_be_aligned.shape

    # Check where the datasets overlap (where both DEMs don't have nans)
    overlapping_nobuffer = np.logical_and(np.logical_not(np.isnan(reference_dem)),
                                          np.logical_not(np.isnan(dem_to_be_aligned)))
    # Buffer the mask to increase the likelyhood of including the correct values
    overlapping = scipy.ndimage.maximum_filter(overlapping_nobuffer, size=pixel_buffer, mode="constant")

    # Remove parts of the DEMs where no overlap existed
    reference_dem[~overlapping] = np.nan
    dem_to_be_aligned[~overlapping] = np.nan

    # Make a temporary directory to write the overlap-fixed DEMs to
    temporary_dir = tempfile.TemporaryDirectory()
    reference_temp_filepath = os.path.join(temporary_dir.name, "reference.tif")
    aligned_temp_filepath = os.path.join(temporary_dir.name, "aligned_pre_icp.tif")
    output_temp_filepath = os.path.join(temporary_dir.name, "output_dem.tif")

    write_geotiff(reference_temp_filepath, reference_dem, crs=crs, bounds=bounds)
    write_geotiff(aligned_temp_filepath, dem_to_be_aligned, crs=crs, bounds=bounds)

    resolution = int((bounds["east"] - bounds["west"]) // reference_dem.shape[1])

    # Define values to fill the below pipeline with
    pdal_parameters = {
        "REFERENCE_FILEPATH": reference_temp_filepath,
        "ALIGNED_FILEPATH": aligned_temp_filepath,
        "OUTPUT_FILEPATH": output_temp_filepath,
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
    for key in pdal_parameters:
        pdal_pipeline = pdal_pipeline.replace(key, str(pdal_parameters[key]))

    # Make the pipeline, execute it, and extract the resultant metadata
    pipeline = pdal.Pipeline(pdal_pipeline)
    pipeline.execute()
    metadata = pipeline.metadata

    output = rio.open(output_temp_filepath).read(1)

    # Get the fitness value from the ICP coregistration
    fitness: float = json.loads(metadata)["metadata"]["filters.icp"]["fitness"]

    return output, fitness


def get_horizontal_shift(elevation_difference: np.ndarray, slope: np.ndarray, aspect: np.ndarray,
                         min_count: int = 30) -> tuple[float, float, float]:
    """
    Calculate the horizontal shift between two DEMs using the method presented in Nuth and Kääb (2011).

    :param elevation_difference: The elevation difference (reference_dem - aligned_dem).
    :param slope: A slope map with the same shape as elevation_difference (units = pixels?)
    :param aspect: An aspect map with the same shape as elevation_difference (units = radians)

    :returns: The pixel offsets in easting, northing, and the c_parameter (altitude?).
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
    # TODO: Figure out why slope is called slope_px. What units is it in?
    # TODO: Change accordingly in the get_horizontal_shift docstring.

    # Calculate the gradient of the slope
    gradient_y, gradient_x = np.gradient(dem)

    slope_px = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    aspect = np.arctan(-gradient_x, gradient_y)
    aspect += np.pi

    return slope_px, aspect


def deramping(elevation_difference, x_coordinates: np.ndarray, y_coordinates: np.ndarray,
              degree: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Calculate a deramping function to account for rotational and non-rigid components of the elevation difference.

    :param elevation_difference: The elevation difference array to analyse.
    :param x_coordinates: x-coordinates of the above array (must have the same shape as elevation_difference)
    :param y_coordinates: y-coordinates of the above array (must have the same shape as elevation_difference)
    :param degree: The polynomial degree to estimate the ramp.

    :returns: A callable function to estimate the ramp.
    """
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

        :param x_coordinates: x-coordinates of the difference array (must have the same shape as elevation_difference)
        :param y_coordinates: y-coordinates of the difference array (must have the same shape as elevation_difference)
        :param coefficients: The coefficients (a, b, c, etc.) of the polynomial.
        :param degree: The degree of the polynomial.

        :returns: The values estimated by the polynomial.
        """
        # Check that the coefficient size is correct.
        coefficient_size = (degree + 1) * (degree + 2) / 2
        if len(coefficients) != coefficient_size:
            raise ValueError()

        # Do Amaury's black magic to estimate the values.
        estimated_values = np.sum([coefficients[k * (k + 1) // 2 + j] * x_coordinates ** (k - j) *
                                   y_coordinates ** j for k in range(degree + 1) for j in range(k + 1)], axis=0)
        return estimated_values

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
    coefficients, *_ = scipy.optimize.leastsq(
        func=residuals,
        x0=initial_guess,
        args=(valid_diffs, valid_x_coords, valid_y_coords, degree),
        full_output=True
    )

    # Generate the return-function which can correctly estimate the ramp
    def ramp(x_coordinates: np.ndarray, y_coordinates: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get the values of the ramp that corresponds to given coordinates.

        :param x_coordinates: x-coordinates of interest.
        :param y_coordinates: y-coordinates of interest.

        :returns: The estimated ramp offsets.
        """
        return estimate_values(x_coordinates, y_coordinates, coefficients, degree)

    # Return the function which can be used later.
    return ramp


def calculate_nmad(array: np.ndarray) -> float:
    """
    Calculate the normalized (?) median absolute deviation of an array.

    :param array: A one- or multidimensional array.

    :returns: The NMAD of the array.
    """
    # TODO: Get a reference for why NMAD is used (and make sure the N stands for normalized)
    nmad = 1.4826 * np.nanmedian(np.abs(array - np.nanmedian(array)))

    return nmad


def amaury_coregister_dem(reference_dem: np.ndarray, dem_to_be_aligned: np.ndarray,
                          max_iterations: int = 50, error_threshold: float = 0.05,
                          deramping_degree: Optional[int] = 1, verbose: bool = True, **_) -> tuple[np.ndarray, float]:
    """
    Coregister a DEM using the Nuth and Kääb (2011) approach.

    :param reference_dem: The DEM acting reference.
    :param dem_to_be_aligned: The DEM to be aligned to the reference.
    :param max_iterations: The maximum of iterations to attempt the coregistration.
    :param error_threshold: The acceptable error threshold after which to stop the iterations.
    :param deramping_degree: Optional. The polynomial degree to estimate for deramping the offset field.
    :param verbose: Whether to print the progress or not.

    :returns: The aligned DEM, and the NMAD (error) of the alignment.
    """
    # TODO: Add offset_east and offset_north as return variables?
    # Make a new DEM which will be modified inplace
    aligned_dem = dem_to_be_aligned.copy()

    # Make sure that the DEMs have the same shape
    assert reference_dem.shape == aligned_dem.shape

    # Calculate slope and aspect maps from the reference DEM
    slope, aspect = calculate_slope_and_aspect(reference_dem)

    # Make index grids for the east and north dimensions
    east_grid = np.arange(reference_dem.shape[1])
    north_grid = np.arange(reference_dem.shape[0])

    # Make a function to estimate the aligned DEM (used to construct an offset DEM)
    elevation_function = scipy.interpolate.RectBivariateSpline(x=north_grid, y=east_grid, z=aligned_dem)
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

        # Stop if the NMAD is low and a few iterations have been made
        if i > 5 and nmad < error_threshold:
            if verbose:
                print(f"NMAD went below the error threshold of {error_threshold}")
            break

        # Estimate the horizontal shift from the implementation by Nuth and Kääb (2011)
        east_diff, north_diff, _ = get_horizontal_shift(  # types: ignore
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
        #new_elevation[new_nans != 0] = np.nan

        # Assign the newly calculated elevations to the aligned_dem
        aligned_dem = new_elevation

    if verbose:
        print(f"Final easting offset: {offset_east:.2f} px, northing offset: {offset_north:.2f} px, NMAD: {nmad:.3f} m")

    # Try to account for rotations between the dataset
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
            degree=deramping_degree
        )
        # Apply the deramping function to the dataset
        aligned_dem -= ramp(x_coordinates, y_coordinates)

        # Calculate the final residual error of the analysis
        elevation_difference = reference_dem - aligned_dem
        nmad = calculate_nmad(elevation_difference)

        if verbose:
            print(f"NMAD after deramping (degree: {deramping_degree}): {nmad:.3f} m")

    return aligned_dem, nmad


class Method(Enum):
    """A selection of a coregistration method."""

    ICP = icp_coregistration
    AMAURY = amaury_coregister_dem


class Coregistration:
    """Metaclass for running different implementations of coregistration."""

    def __init__(self, method: Method):
        """
        Instantiate a Coregistration class.

        :param method: The selected coregistration method.
        """
        self.method = method

    def run(self, reference_raster: gu.georaster.Raster, to_be_aligned_raster: gu.georaster.Raster) -> tuple[gu.georaster.Raster, float]:
        """
        Run coregistration between two GeoUtils rasters, using the selected method.

        :param reference_raster: The raster acting reference.
        :param: to_be_aligned_raster: The raster to be aligned.

        :returns: The aligned raster and the error measure for the corresponding method.
        """
        to_be_aligned_raster_filename = to_be_aligned_raster.filename
        # Make sure that the data is read into memory
        if reference_raster.data is None:
            reference_raster.load(1)

        intersection = reference_raster.intersection(to_be_aligned_raster)
        bounds = dict(zip(["west", "south", "east", "north"], intersection))
        resolution = (bounds["east"] - bounds["west"]) / reference_raster.data.shape[2]

        # TODO: Read the to_be_aligned_dem from the actual to_be_aligned_raster and not its filename
        # TODO: The above leads to highly surprising results sometimes if the raster had been modified from its file.
        to_be_aligned_dem = reproject_dem(
            dem=rio.open(to_be_aligned_raster_filename),
            bounds=bounds,
            resolution=resolution,
            crs=reference_raster.crs
        )

        # Align the raster using the selected method. This returns a numpy array and the corresponding error
        aligned_dem, error = self.method(  # type: ignore
            reference_dem=reference_raster.data.squeeze(),
            dem_to_be_aligned=to_be_aligned_dem,
            bounds=bounds,
            crs=reference_raster.crs
        )

        # Construct a raster from the created numpy array
        aligned_raster = gu.georaster.Raster.from_array(
            data=aligned_dem,
            transform=reference_raster.transform,
            crs=reference_raster.crs
        )

        return aligned_raster, error


def test_icp_coregistration():
    """Test coregistration using ICP."""
    reference_raster = gu.georaster.Raster("examples/Longyearbyen/DEM_2009_ref.tif")
    to_be_aligned_raster = gu.georaster.Raster("examples/Longyearbyen/DEM_1995.tif")

    coregistration = Coregistration(method=Method.ICP)
    aligned_raster, error = coregistration.run(reference_raster, to_be_aligned_raster)

    print(error)
    print(aligned_raster)


def test_amaury_coregistration():
    """Test coregistration using Amaury's method."""
    reference_raster = gu.georaster.Raster("examples/Longyearbyen/DEM_2009_ref.tif")

    to_be_aligned_raster = gu.georaster.Raster("examples/Longyearbyen/DEM_1995.tif")

    coregistration = Coregistration(method=Method.AMAURY)
    aligned_raster, error = coregistration.run(reference_raster, to_be_aligned_raster)

    print(error)
    print(aligned_raster)


if __name__ == "__main__":
    test_icp_coregistration()
    test_amaury_coregistration()

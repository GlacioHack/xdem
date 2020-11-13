"""
DEM coregistration functions.


Author(s):
    Erik Schytt Holmlund (holmlund@vaw.baug.ethz.ch)
"""
from __future__ import annotations
import rasterio as rio
import richdem as rd
import numpy as np
from rasterio import Affine
import json
import os
import tempfile
from typing import Optional
import pdal
import rasterio.warp

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

#putting tests down here
if __name__ == '__main__':

    #TEST FOR RICHDEM SLOPE & OTHER TERRAIN ATTR WITHOUT USING GDAL
    fn_test = '/home/atom/ongoing/glaciohack_testdata/DEM_2001.64734089.tif'

    #1/ this works

    # to check it gives similar result with GDAL opening
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    ds = rio.open(fn_test)
    slp = get_terrainattr(ds,attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    #2/ this does not work (need to pass georeferencing to richDEM array, grid is not sufficient)
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))

    ds = rio.open(fn_test)
    slp = rd.TerrainAttribute(rd.rdarray(ds.read(1),no_data=ds.get_nodatavals()[0]), attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))


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
        resampling=rasterio.warp.Resampling.cubic,
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


def icp_coregistration(reference_filepath: str, aligned_filepath: str, output_filepath: str) -> float:
    """
    Perform an ICP coregistration in areas where two DEMs overlap.

    param: reference_filepath: The input filepath to the DEM acting reference.
    param: aligned_filepath: The input filepath to the DEM acting aligned.
    param: output_filepath: The filepath of the aligned dataset after coregistration.

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
    overlapping = np.logical_and(np.logical_not(np.isnan(reference)), np.logical_not(np.isnan(aligned)))

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
    fitness = json.loads(metadata)["metadata"]["filters.icp"]["fitness"]

    return fitness


if __name__ == "__main__":
    fitness = icp_coregistration(
        reference_filepath="examples/Longyearbyen/DEM_2009_ref.tif",
        aligned_filepath="examples/Longyearbyen/DEM_1995.tif",
        output_filepath="examples/Longyearbyen/DEM_1995_coreg.tif"
    )
    print(fitness)

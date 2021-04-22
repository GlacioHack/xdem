"""Functions to test the spatial tools.

Author(s):
    Erik S. Holmlund

"""
import os
import shutil
import subprocess
import tempfile
import warnings

import geoutils as gu
import numpy as np
import rasterio as rio

import xdem
from xdem import examples


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    diff = xdem.spatial_tools.subtract_rasters(
        examples.FILEPATHS["longyearbyen_ref_dem"],
        examples.FILEPATHS["longyearbyen_tba_dem"])

    assert np.nanmean(np.abs(diff.data)) < 100


def test_merge_rasters():
    """Split a DEM with some overlap, then merge it, and validate that it's still the same."""
    dem = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_ref_dem"])

    # Find the easting midpoint of the DEM
    x_midpoint = np.mean([dem.bounds.right, dem.bounds.left])
    x_midpoint -= x_midpoint % dem.res[0]

    # Cut the DEM into two DEMs that slightly overlap each other.
    dem1 = dem.reproject(dst_bounds=rio.coords.BoundingBox(
        right=x_midpoint + dem.res[0] * 3,
        left=dem.bounds.left,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom),
        dst_crs=dem.crs
    )
    dem2 = dem.reproject(dst_bounds=rio.coords.BoundingBox(
        left=x_midpoint - dem.res[0] * 3,
        right=dem.bounds.right,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom),
        dst_crs=dem.crs
    )

    # Merge the DEM and check that it closely resembles the initial DEM
    merged_dem = xdem.spatial_tools.merge_rasters([dem1, dem2])
    assert dem.data.shape == merged_dem.data.shape

    diff = dem.data - merged_dem.data

    assert np.abs(np.nanmean(diff)) < 0.05


def test_hillshade():
    """Test the hillshade algorithm, partly by comparing it to the GDAL hillshade function."""
    warnings.simplefilter("error")

    def make_gdal_hillshade(filepath) -> np.ndarray:
        temp_dir = tempfile.TemporaryDirectory()
        temp_hillshade_path = os.path.join(temp_dir.name, "hillshade.tif")
        gdal_commands = ["gdaldem", "hillshade",
                         filepath, temp_hillshade_path,
                         "-az", "315", "-alt", "45"]
        subprocess.run(gdal_commands, check=True, stdout=subprocess.PIPE)

        data = gu.Raster(temp_hillshade_path).data
        temp_dir.cleanup()
        return data

    filepath = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]
    dem = xdem.DEM(filepath)

    xdem_hillshade = xdem.spatial_tools.hillshade(dem.data, resolution=dem.res)
    if shutil.which("gdaldem") is not None:  # Compare with a GDAL version if GDAL commands are available.
        gdal_hillshade = make_gdal_hillshade(filepath)
        diff = gdal_hillshade - xdem_hillshade

        # Check that the xdem and gdal hillshades are relatively similar.
        assert np.mean(diff) < 5
        assert xdem.spatial_tools.nmad(diff.filled(np.nan)) < 5

    # Try giving the hillshade invalid arguments.
    try:
        xdem.spatial_tools.hillshade(dem.data, dem.res, azimuth=361)
    except ValueError as exception:
        if "Azimuth must be a value between 0 and 360" not in str(exception):
            raise exception
    try:
        xdem.spatial_tools.hillshade(dem.data, dem.res, altitude=91)
    except ValueError as exception:
        if "Altitude must be a value between 0 and 90" not in str(exception):
            raise exception

    try:
        xdem.spatial_tools.hillshade(dem.data, dem.res, z_factor=np.inf)
    except ValueError as exception:
        if "z_factor must be a non-negative finite value" not in str(exception):
            raise exception

    # Introduce some nans
    dem.data.mask = np.zeros_like(dem.data, dtype=bool)
    dem.data.mask.ravel()[np.random.choice(dem.data.size, 50000, replace=False)] = True

    # Make sure that this doesn't create weird division warnings.
    xdem.spatial_tools.hillshade(dem.data, dem.res)

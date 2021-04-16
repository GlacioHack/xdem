"""Functions to test the spatial tools.

Author(s):
    Erik S. Holmlund

"""
import geoutils as gu
import matplotlib.pyplot as plt
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
    dem1 = dem.copy()
    dem1.crop(rio.coords.BoundingBox(
        right=x_midpoint + dem.res[0] * 3,
        left=dem.bounds.left,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom)
    )
    dem2 = dem.copy()
    dem2.crop(rio.coords.BoundingBox(
        left=x_midpoint - dem.res[0] * 3,
        right=dem.bounds.right,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom)
    )

    # Merge the DEM and check that it closely resembles the initial DEM
    merged_dem = xdem.spatial_tools.merge_rasters([dem1, dem2])
    assert dem.data.shape == merged_dem.data.shape
    assert dem.bounds == merged_dem.bounds

    diff = dem.data - merged_dem.data

    assert np.abs(np.nanmean(diff)) < 0.0001

    # Check that include_ref works
    merged_dem2 = xdem.spatial_tools.merge_rasters([dem, dem1, dem2], include_ref=False)
    assert merged_dem2 == merged_dem

    # Check that use_ref_bounds work - use a DEM that do not cover the whole extent
    dem2 = dem.copy()
    dem2.crop(rio.coords.BoundingBox(
        left=x_midpoint - dem.res[0] * 3,
        right=dem.bounds.right - dem.res[0]*2,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom)
    )

    # This case should not preserve original extent
    merged_dem = xdem.spatial_tools.merge_rasters([dem1, dem2])
    assert merged_dem.bounds != dem.bounds

    # This case should preserve original extent
    merged_dem2 = xdem.spatial_tools.merge_rasters([dem, dem1, dem2], include_ref = False, use_ref_bounds=True)
    assert merged_dem2.bounds == dem.bounds

"""
Functions to test the spatial tools.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio

import xdem
from xdem import examples


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    diff = xdem.spatial_tools.subtract_rasters(
        examples.FILEPATHS["longyearbyen_ref_dem"], examples.FILEPATHS["longyearbyen_tba_dem"]
    )

    assert np.nanmean(np.abs(diff.data)) < 100


class TestMerging:
    """
    Test cases for stacking and merging DEMs
    Split a DEM with some overlap, then stack/merge it, and validate bounds and shape.
    """

    dem = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_ref_dem"])

    # Find the easting midpoint of the DEM
    x_midpoint = np.mean([dem.bounds.right, dem.bounds.left])
    x_midpoint -= x_midpoint % dem.res[0]

    # Cut the DEM into two DEMs that slightly overlap each other.
    dem1 = dem.copy()
    dem1.crop(
        rio.coords.BoundingBox(
            right=x_midpoint + dem.res[0] * 3, left=dem.bounds.left, top=dem.bounds.top, bottom=dem.bounds.bottom
        )
    )
    dem2 = dem.copy()
    dem2.crop(
        rio.coords.BoundingBox(
            left=x_midpoint - dem.res[0] * 3, right=dem.bounds.right, top=dem.bounds.top, bottom=dem.bounds.bottom
        )
    )

    # To check that use_ref_bounds work - create a DEM that do not cover the whole extent
    dem3 = dem.copy()
    dem3.crop(
        rio.coords.BoundingBox(
            left=x_midpoint - dem.res[0] * 3,
            right=dem.bounds.right - dem.res[0] * 2,
            top=dem.bounds.top,
            bottom=dem.bounds.bottom,
        )
    )

    def test_stack_rasters(self):
        """Test stack_rasters"""
        # Merge the two overlapping DEMs and check that output bounds and shape is correct
        stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem2])

        assert stacked_dem.count == 2
        assert self.dem.shape == stacked_dem.shape

        merged_bounds = xdem.spatial_tools.merge_bounding_boxes(
            [self.dem1.bounds, self.dem2.bounds], resolution=self.dem1.res[0]
        )
        assert merged_bounds == stacked_dem.bounds

        # Check that reference works with input Raster
        stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem2], reference=self.dem)
        assert self.dem.bounds == stacked_dem.bounds

        # Others than int or gu.Raster should raise a ValueError
        try:
            stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem2], reference="a string")
        except ValueError as exception:
            if "reference should be" not in str(exception):
                raise exception

        # Check that use_ref_bounds works - use a DEM that do not cover the whole extent

        # This case should not preserve original extent
        stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem3])
        assert stacked_dem.bounds != self.dem.bounds

        # This case should preserve original extent
        stacked_dem2 = xdem.spatial_tools.stack_rasters([self.dem1, self.dem3], reference=self.dem, use_ref_bounds=True)
        assert stacked_dem2.bounds == self.dem.bounds

    def test_merge_rasters(self):
        """Test merge_rasters"""
        # Merge the two overlapping DEMs and check that it closely resembles the initial DEM
        merged_dem = xdem.spatial_tools.merge_rasters([self.dem1, self.dem2])
        assert self.dem.data.shape == merged_dem.data.shape
        assert self.dem.bounds == merged_dem.bounds

        diff = self.dem.data - merged_dem.data

        assert np.abs(np.nanmean(diff)) < 0.0001

        # Check that reference works
        merged_dem2 = xdem.spatial_tools.merge_rasters([self.dem1, self.dem2], reference=self.dem)
        assert merged_dem2 == merged_dem


def test_hillshade():
    """Test the hillshade algorithm, partly by comparing it to the GDAL hillshade function."""
    warnings.simplefilter("error")

    def make_gdal_hillshade(filepath) -> np.ndarray:
        # rasterio strongly recommends against importing gdal along rio, so this is done here instead.
        from osgeo import gdal

        temp_dir = tempfile.TemporaryDirectory()
        temp_hillshade_path = os.path.join(temp_dir.name, "hillshade.tif")
        # gdal_commands = ["gdaldem", "hillshade",
        #                 filepath, temp_hillshade_path,
        #                 "-az", "315", "-alt", "45"]
        # subprocess.run(gdal_commands, check=True, stdout=subprocess.PIPE)
        gdal.DEMProcessing(
            destName=temp_hillshade_path,
            srcDS=filepath,
            processing="hillshade",
            options=gdal.DEMProcessingOptions(azimuth=315, altitude=45),
        )

        data = gu.Raster(temp_hillshade_path).data
        temp_dir.cleanup()
        return data

    filepath = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dem = xdem.DEM(filepath)

    xdem_hillshade = xdem.spatial_tools.hillshade(dem.data, resolution=dem.res)
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


def test_subdivide_array():

    test_shape = (6, 4)
    test_count = 4
    subdivision_grid = xdem.spatial_tools.subdivide_array(test_shape, test_count)

    assert subdivision_grid.shape == test_shape
    assert np.unique(subdivision_grid).size == test_count

    assert np.unique(xdem.spatial_tools.subdivide_array((3, 3), 3)).size == 3

    with pytest.raises(ValueError, match=r"Expected a 2D shape, got 1D shape.*"):
        xdem.spatial_tools.subdivide_array((5,), 2)

    with pytest.raises(ValueError, match=r"Shape.*smaller than.*"):
        xdem.spatial_tools.subdivide_array((5, 2), 15)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32", "float16"])
@pytest.mark.parametrize(
    "mask_and_viewable",
    [
        (None, True),  # An ndarray with no mask should support views
        (False, True),  # A masked array with an empty mask should support views
        ([True, False, False, False], False),  # A masked array with an occupied mask should not support views.
        ([False, False, False, False], True),  # A masked array with an empty occupied mask should support views.
    ],
)
@pytest.mark.parametrize(
    "shape_and_check_passes",
    [
        ((1, 2, 2), True),  # A 3D array with a shape[0] == 1 is okay.
        ((2, 1, 2), False),  # A 3D array with a shape[0] != 1 is not okay.
        ((2, 2), True),  # A 2D array is okay.
        ((4,), True),  # A 1D array is okay.
    ],
)
def test_get_array_and_mask(
    dtype: str,
    mask_and_viewable: tuple[None | bool | list[bool], bool],
    shape_and_check_passes: tuple[tuple[int, ...], bool],
):
    """Validate that the function returns views when expected, and copies otherwise."""
    warnings.simplefilter("error")

    masked_values, view_should_be_possible = mask_and_viewable
    shape, check_should_pass = shape_and_check_passes

    # Create an array of the specified dtype
    array = np.ones(shape, dtype=dtype)
    if masked_values is not None:
        if masked_values is False:
            array = np.ma.masked_array(array)
        else:
            array = np.ma.masked_array(array, mask=np.reshape(masked_values, array.shape))

    # Validate that incorrect shapes raise the correct error.
    if not check_should_pass:
        with pytest.raises(ValueError, match="Invalid array shape given"):
            xdem.spatial_tools.get_array_and_mask(array, check_shape=True)

        # Stop the test here as the failure is now validated.
        return

    # Get a copy of the array and check its shape (it should always pass at this point)
    arr, _ = xdem.spatial_tools.get_array_and_mask(array, copy=True, check_shape=True)

    # Validate that the array is a copy
    assert not np.shares_memory(arr, array)

    # If it was an integer dtype and it had a mask, validate that the array is now "float32"
    if np.issubdtype(dtype, np.integer) and np.any(masked_values or False):
        assert arr.dtype == "float32"

    # If there was no mask or the mask was empty, validate that arr and array are equivalent
    if not np.any(masked_values or False):
        assert np.sum(np.abs(array - arr)) == 0.0

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")

        # Try to create a view.
        arr_view, mask = xdem.spatial_tools.get_array_and_mask(array, copy=False)

        # If it should be possible, validate that there were no warnings.
        if view_should_be_possible:
            assert len(caught_warnings) == 0, (caught_warnings[0].message, array)
        # Otherwise, validate that one warning was raised with the correct text.
        else:
            assert len(caught_warnings) == 1
            assert "Copying is required" in str(caught_warnings[0].message)

    # Validate that the view shares memory if it was possible, or otherwise that it is a copy.
    if view_should_be_possible:
        assert np.shares_memory(array, arr_view)
    else:
        assert not np.shares_memory(array, arr_view)

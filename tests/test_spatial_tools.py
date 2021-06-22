"""Functions to test the spatial tools.

Author(s):
    Erik S. Holmlund
    Romain Hugonnet

"""
import os
import shutil
import subprocess
import tempfile
import warnings

import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
from sklearn.metrics import mean_squared_error, median_absolute_error

import xdem
from xdem import examples


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    diff = xdem.spatial_tools.subtract_rasters(
        examples.FILEPATHS["longyearbyen_ref_dem"],
        examples.FILEPATHS["longyearbyen_tba_dem"])

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

    # To check that use_ref_bounds work - create a DEM that do not cover the whole extent
    dem3 = dem.copy()
    dem3.crop(rio.coords.BoundingBox(
        left=x_midpoint - dem.res[0] * 3,
        right=dem.bounds.right - dem.res[0]*2,
        top=dem.bounds.top,
        bottom=dem.bounds.bottom)
    )

    def test_stack_rasters(self):
        """Test stack_rasters"""
        # Merge the two overlapping DEMs and check that output bounds and shape is correct
        stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem2])

        assert stacked_dem.count == 2
        assert self.dem.shape == stacked_dem.shape

        merged_bounds = xdem.spatial_tools.merge_bounding_boxes([self.dem1.bounds, self.dem2.bounds],
                                                                resolution=self.dem1.res[0])
        assert merged_bounds == stacked_dem.bounds

        # Check that reference works with input Raster
        stacked_dem = xdem.spatial_tools.stack_rasters(
            [self.dem1, self.dem2], reference=self.dem)
        assert self.dem.bounds == stacked_dem.bounds

        # Others than int or gu.Raster should raise a ValueError
        try:
            stacked_dem = xdem.spatial_tools.stack_rasters(
                [self.dem1, self.dem2], reference="a string")
        except ValueError as exception:
            if "reference should be" not in str(exception):
                raise exception

        # Check that use_ref_bounds works - use a DEM that do not cover the whole extent

        # This case should not preserve original extent
        stacked_dem = xdem.spatial_tools.stack_rasters([self.dem1, self.dem3])
        assert stacked_dem.bounds != self.dem.bounds

        # This case should preserve original extent
        stacked_dem2 = xdem.spatial_tools.stack_rasters(
            [self.dem1, self.dem3], reference=self.dem, use_ref_bounds=True)
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
        merged_dem2 = xdem.spatial_tools.merge_rasters(
            [self.dem1, self.dem2], reference=self.dem)
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
        #subprocess.run(gdal_commands, check=True, stdout=subprocess.PIPE)
        gdal.DEMProcessing(
            destName=temp_hillshade_path,
            srcDS=filepath,
            processing="hillshade",
            options=gdal.DEMProcessingOptions(azimuth=315, altitude=45)
        )

        data = gu.Raster(temp_hillshade_path).data
        temp_dir.cleanup()
        return data

    filepath = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]
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
    dem.data.mask.ravel()[np.random.choice(
        dem.data.size, 50000, replace=False)] = True

    # Make sure that this doesn't create weird division warnings.
    xdem.spatial_tools.hillshade(dem.data, dem.res)

class TestRobustFitting:

    def test_robust_polynomial_fit(self):

        np.random.seed(42)

        # x vector
        x = np.linspace(1,10,1000)
        # exact polynomial
        true_coefs = [-100, 5, 3, 2]
        y = true_coefs[0] + true_coefs[1] * x + true_coefs[2] * x**2 + true_coefs[3] * x**3
        # add some noise on top
        y += np.random.normal(loc=0,scale=3,size=1000)
        # and some outliers
        y[50:75] = 0
        y[900:925] = 1000

        # test linear estimators
        coefs, deg = xdem.spatial_tools.robust_polynomial_fit(x,y, estimator='Linear', linear_pkg='scipy', loss='soft_l1', f_scale=0.5)

        # scipy solution should be quite robust to outliers/noise (with the soft_l1 method and f_scale parameter)
        # however, it is subject to random processes inside the scipy function (couldn't find how to fix those...)
        assert deg == 3 or deg == 4 # can find degree 3, or 4 with coefficient close to 0
        assert np.abs(coefs[0] - true_coefs[0]) < 3
        assert np.abs(coefs[1] - true_coefs[1]) < 3
        assert np.abs(coefs[2] - true_coefs[2]) < 1
        assert np.abs(coefs[3] - true_coefs[3]) < 1

        # the sklearn Linear solution with MSE cost function will not be robust
        coefs2, deg2 = xdem.spatial_tools.robust_polynomial_fit(x,y, estimator='Linear', linear_pkg='sklearn', cost_func=mean_squared_error, margin_improvement=50)
        assert deg2 != 3
        # using the median absolute error should improve the fit, but the parameters will still be hard to constrain
        coefs3, deg3 = xdem.spatial_tools.robust_polynomial_fit(x,y, estimator='Linear', linear_pkg='sklearn', cost_func=median_absolute_error, margin_improvement=50)
        assert deg3 == 3
        assert np.abs(coefs3[0] - true_coefs[0]) > 50
        assert np.abs(coefs3[1] - true_coefs[1]) > 10
        assert np.abs(coefs3[2] - true_coefs[2]) > 5
        assert np.abs(coefs3[3] - true_coefs[3]) > 0.5

        # test robust estimator

        # Theil-Sen should have better coefficients
        coefs4, deg4 = xdem.spatial_tools.robust_polynomial_fit(x, y, estimator='Theil-Sen', random_state=42)
        assert deg4 == 3
        # high degree coefficients should be well constrained
        assert np.abs(coefs4[2] - true_coefs[2]) < 1
        assert np.abs(coefs4[3] - true_coefs[3]) < 1

        # RANSAC is not always optimal, here it does not work well
        coefs5, deg5 = xdem.spatial_tools.robust_polynomial_fit(x, y, estimator='RANSAC', random_state=42)
        assert deg5 != 3

        # Huber should perform well, close to the scipy robust solution
        coefs6, deg6 = xdem.spatial_tools.robust_polynomial_fit(x, y, estimator='Huber')
        assert deg6 == 3
        assert np.abs(coefs6[1] - true_coefs[1]) < 1
        assert np.abs(coefs6[2] - true_coefs[2]) < 1
        assert np.abs(coefs6[3] - true_coefs[3]) < 1

class TestSubsample:
    """
    Different examples of 1D to 3D arrays with masked values for testing.
    """

    # Case 1 - 1D array, 1 masked value
    array1D = np.ma.masked_array(np.arange(10), mask=np.zeros(10))
    array1D.mask[3] = True
    assert np.ndim(array1D) == 1
    assert np.count_nonzero(array1D.mask) > 0

    # Case 2 - 2D array, 1 masked value
    array2D = np.ma.masked_array(np.arange(9).reshape((3, 3)), mask=np.zeros((3, 3)))
    array2D.mask[0, 1] = True
    assert np.ndim(array2D) == 2
    assert np.count_nonzero(array2D.mask) > 0

    # Case 3 - 3D array, 1 masked value
    array3D = np.ma.masked_array(np.arange(9).reshape((1, 3, 3)), mask=np.zeros((1, 3, 3)))
    array3D = np.ma.vstack((array3D, array3D + 10))
    array3D.mask[0, 0, 1] = True
    assert np.ndim(array3D) == 3
    assert np.count_nonzero(array3D.mask) > 0

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])
    def test_subsample(self, array):
        """
        Test xdem.spatial_tools.subsample_raster.
        """
        # Test that subsample > 1 works as expected, i.e. output 1D array, with no masked values, or selected size
        for npts in np.arange(2, np.size(array)):
            random_values = xdem.spatial_tools.subsample_raster(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Test if subsample > number of valid values => return all
        random_values = xdem.spatial_tools.subsample_raster(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample = 1 => return all valid values
        random_values = xdem.spatial_tools.subsample_raster(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample < 1
        random_values = xdem.spatial_tools.subsample_raster(array, subsample=0.5)
        assert np.size(random_values) == int(np.size(array) * 0.5)

        # Test with optional argument return_indices
        indices = xdem.spatial_tools.subsample_raster(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.size(array) * 0.3)
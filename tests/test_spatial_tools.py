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
import pandas as pd
import pytest
import rasterio as rio
from sklearn.metrics import mean_squared_error, median_absolute_error

import xdem
from xdem import examples


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        diff = xdem.spatial_tools.subtract_rasters(
            examples.get_path("longyearbyen_ref_dem"),
            examples.get_path("longyearbyen_tba_dem"))

    assert np.nanmean(np.abs(diff.data)) < 100


def load_ref_and_diff() -> tuple[gu.georaster.Raster, gu.georaster.Raster, np.ndarray]:
    """Load example files to try coregistration methods with."""
    examples.download_longyearbyen_examples(overwrite=False)

    reference_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_ref_dem"])
    to_be_aligned_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_tba_dem"])
    glacier_mask = gu.geovector.Vector(examples.FILEPATHS["longyearbyen_glacier_outlines"])
    inlier_mask = ~glacier_mask.create_mask(reference_raster)

    metadata = {}
    # aligned_raster, _ = xdem.coreg.coregister(reference_raster, to_be_aligned_raster, method="amaury", mask=glacier_mask,
    #                                          metadata=metadata)
    nuth_kaab = xdem.coreg.NuthKaab()
    nuth_kaab.fit(reference_raster.data, to_be_aligned_raster.data,
                  inlier_mask=inlier_mask, transform=reference_raster.transform)
    aligned_raster = nuth_kaab.apply(to_be_aligned_raster.data, transform=reference_raster.transform)

    diff = gu.Raster.from_array((reference_raster.data - aligned_raster),
                                transform=reference_raster.transform, crs=reference_raster.crs)
    mask = glacier_mask.create_mask(diff)

    return reference_raster, diff, mask


class TestMerging:
    """
    Test cases for stacking and merging DEMs
    Split a DEM with some overlap, then stack/merge it, and validate bounds and shape.
    """
    dem = gu.georaster.Raster(examples.get_path("longyearbyen_ref_dem"))

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

class TestRobustFitting:

    @pytest.mark.parametrize("pkg_estimator", [('sklearn','Linear'), ('scipy','Linear'), ('sklearn','Theil-Sen'),
                                           ('sklearn','RANSAC'),('sklearn','Huber')])
    def test_robust_polynomial_fit(self, pkg_estimator: str) -> None:

        np.random.seed(42)

        # x vector
        x = np.linspace(1, 10, 1000)
        # exact polynomial
        true_coefs = [-100, 5, 3, 2]
        y = true_coefs[0] + true_coefs[1] * x + true_coefs[2] * x ** 2 + true_coefs[3] * x ** 3

        coefs, deg = xdem.spatial_tools.robust_polynomial_fit(x, y, linear_pkg=pkg_estimator[0], estimator=pkg_estimator[1], random_state=42)

        assert deg == 3 or deg == 4
        assert np.abs(coefs[0] - true_coefs[0]) <= 100
        assert np.abs(coefs[1] - true_coefs[1]) < 5
        assert np.abs(coefs[2] - true_coefs[2]) < 2
        assert np.abs(coefs[3] - true_coefs[3]) < 1

    def test_robust_polynomial_fit_noise_and_outliers(self):

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

    def test_robust_sumsin_fit(self) -> None:

        # x vector
        x = np.linspace(0, 10, 1000)
        # exact polynomial
        true_coefs = np.array([(5, 1, np.pi),(3, 0.3, 0)]).flatten()
        y = xdem.spatial_tools._fitfun_sumofsin(x,params=true_coefs)

        # check that the function runs
        coefs, deg = xdem.spatial_tools.robust_sumsin_fit(x,y, random_state=42)

        # check that the estimated sum of sinusoid correspond to the input
        for i in range(2):
            assert (coefs[3*i] - true_coefs[3*i]) < 0.01

        # test that using custom arguments does not create an error
        bounds = [(3,7),(0.1,3),(0,2*np.pi),(1,7),(0.1,1),(0,2*np.pi),(0,1),(0.1,1),(0,2*np.pi)]
        coefs, deg = xdem.spatial_tools.robust_sumsin_fit(x,y,bounds_amp_freq_phase=bounds, nb_frequency_max=2
                                                          , significant_res=0.01, random_state=42)

    def test_robust_simsin_fit_noise_and_outliers(self):

        # test the robustness to outliers

        np.random.seed(42)
        # x vector
        x = np.linspace(0, 10, 1000)
        # exact polynomial
        true_coefs = np.array([(5, 1, np.pi), (3, 0.3, 0)]).flatten()
        y = xdem.spatial_tools._fitfun_sumofsin(x, params=true_coefs)

        # adding some noise
        y += np.random.normal(loc=0, scale=0.25, size=1000)
        # and some outliers
        y[50:75] = -10
        y[900:925] = 10

        bounds = [(3, 7), (0.1, 3), (0, 2 * np.pi), (1, 7), (0.1, 1), (0, 2 * np.pi), (0, 1), (0.1, 1), (0, 2 * np.pi)]
        coefs, deg = xdem.spatial_tools.robust_sumsin_fit(x,y, random_state=42, bounds_amp_freq_phase=bounds)

        # should be less precise, but still on point
        # re-order output coefficient to match input
        if coefs[3] > coefs[0]:
            coefs = np.concatenate((coefs[3:],coefs[0:3]))

        # check values
        for i in range(2):
            assert (coefs[3*i] - true_coefs[3*i]) < 0.2
            assert (coefs[3 * i +1] - true_coefs[3 * i +1]) < 0.2
            error_phase = min(np.abs(coefs[3 * i + 2] - true_coefs[ 3* i + 2]), np.abs(2* np.pi - (coefs[3 * i + 2] - true_coefs[ 3* i + 2])))
            assert error_phase < 0.2

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
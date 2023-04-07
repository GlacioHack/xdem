"""Functions to test the filtering tools."""
from __future__ import annotations

import geoutils as gu
import numpy as np
import pytest

import xdem


class TestFilters:
    """Test cases for the filter functions."""

    # Load example data.
    dem_2009 = gu.Raster(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = gu.Raster(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(dem_2009, silent=True)

    def test_gauss(self) -> None:
        """Test applying the various Gaussian filters on DEMs with/without NaNs"""

        # Test applying scipy's Gaussian filter
        # smoothing should not yield values below.above original DEM
        dem_array = self.dem_1990.data
        dem_sm = xdem.filters.gaussian_filter_scipy(dem_array, sigma=5)
        assert np.min(dem_array) < np.min(dem_sm)
        assert np.max(dem_array) > np.max(dem_sm)
        assert dem_array.shape == dem_sm.shape

        # Test applying OpenCV's Gaussian filter
        dem_sm2 = xdem.filters.gaussian_filter_cv(dem_array, sigma=5)
        assert np.min(dem_array) < np.min(dem_sm2)
        assert np.max(dem_array) > np.max(dem_sm2)
        assert dem_array.shape == dem_sm2.shape

        # Assert that both implementations yield similar results
        assert np.nanmax(np.abs(dem_sm - dem_sm2)) < 1e-3

        # Test that it works with NaNs too
        nan_count = 1000
        cols = np.random.randint(0, high=self.dem_1990.width - 1, size=nan_count, dtype=int)
        rows = np.random.randint(0, high=self.dem_1990.height - 1, size=nan_count, dtype=int)
        dem_with_nans = np.copy(self.dem_1990.data).squeeze()
        dem_with_nans[rows, cols] = np.nan

        dem_sm = xdem.filters.gaussian_filter_scipy(dem_with_nans, sigma=10)
        assert np.nanmin(dem_with_nans) < np.min(dem_sm)
        assert np.nanmax(dem_with_nans) > np.max(dem_sm)

        dem_sm = xdem.filters.gaussian_filter_cv(dem_with_nans, sigma=10)
        assert np.nanmin(dem_with_nans) < np.min(dem_sm)
        assert np.nanmax(dem_with_nans) > np.max(dem_sm)

        # Test that it works with 3D arrays
        array_3d = np.vstack((dem_array[np.newaxis, :], dem_array[np.newaxis, :]))
        dem_sm = xdem.filters.gaussian_filter_scipy(array_3d, sigma=5)
        assert array_3d.shape == dem_sm.shape

        # 3D case not implemented with OpenCV
        pytest.raises(NotImplementedError, xdem.filters.gaussian_filter_cv, array_3d, sigma=5)

        # Tests that it fails with 1D arrays with appropriate error
        data = dem_array[:, 0]
        pytest.raises(ValueError, xdem.filters.gaussian_filter_scipy, data, sigma=5)
        pytest.raises(ValueError, xdem.filters.gaussian_filter_cv, data, sigma=5)

    def test_dist_filter(self) -> None:
        """Test that distance_filter works"""

        # Calculate dDEM
        ddem = self.dem_2009.data - self.dem_1990.data

        # Add random outliers
        count = 1000
        cols = np.random.randint(0, high=self.dem_1990.width - 1, size=count, dtype=int)
        rows = np.random.randint(0, high=self.dem_1990.height - 1, size=count, dtype=int)
        ddem.data[rows, cols] = 5000

        # Filter gross outliers
        filtered_ddem = xdem.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)

        # Check that all outliers were properly filtered
        assert np.all(np.isnan(filtered_ddem[rows, cols]))

        # Assert that non filtered pixels remain the same
        assert ddem.data.shape == filtered_ddem.shape
        assert np.all(ddem.data[np.isfinite(filtered_ddem)] == filtered_ddem[np.isfinite(filtered_ddem)])

        # Check that it works with NaNs too
        ddem.data[rows[:500], cols[:500]] = np.nan
        filtered_ddem = xdem.filters.distance_filter(ddem.data, radius=20, outlier_threshold=50)
        assert np.all(np.isnan(filtered_ddem[rows, cols]))

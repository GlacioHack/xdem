"""Functions to test the filtering tools."""

from __future__ import annotations

from collections.abc import Callable

import geoutils as gu
import numpy as np
import pytest

import xdem
from xdem._typing import NDArrayf


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

        # Test that it works with NaNs too
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.dem_1990.width - 1, size=nan_count, dtype=int)
        rows = rng.integers(0, high=self.dem_1990.height - 1, size=nan_count, dtype=int)
        dem_with_nans = np.copy(self.dem_1990.data).squeeze()
        dem_with_nans[rows, cols] = np.nan

        dem_sm = xdem.filters.gaussian_filter_scipy(dem_with_nans, sigma=10)
        assert np.nanmin(dem_with_nans) < np.min(dem_sm)
        assert np.nanmax(dem_with_nans) > np.max(dem_sm)

        # Test that it works with 3D arrays
        array_3d = np.vstack((dem_array[np.newaxis, :], dem_array[np.newaxis, :]))
        dem_sm = xdem.filters.gaussian_filter_scipy(array_3d, sigma=5)
        assert array_3d.shape == dem_sm.shape

        # Tests that it fails with 1D arrays with appropriate error
        data = dem_array[:, 0]
        pytest.raises(ValueError, xdem.filters.gaussian_filter_scipy, data, sigma=5)

    def test_dist_filter(self) -> None:
        """Test that distance_filter works"""

        # Calculate dDEM
        ddem = self.dem_2009.data - self.dem_1990.data

        # Add random outliers
        count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.dem_1990.width - 1, size=count, dtype=int)
        rows = rng.integers(0, high=self.dem_1990.height - 1, size=count, dtype=int)
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

    @pytest.mark.parametrize(
        "name, filter_func",
        [
            ("median", lambda arr: xdem.filters.median_filter_scipy(arr, **{"size": 5})),  # type:ignore
            ("mean", lambda arr: xdem.filters.mean_filter(arr, kernel_size=5)),  # type:ignore
            ("min", lambda arr: xdem.filters.min_filter_scipy(arr, **{"size": 5})),  # type:ignore
            ("max", lambda arr: xdem.filters.max_filter_scipy(arr, **{"size": 5})),  # type:ignore
        ],
    )
    def test_filters(self, name: str, filter_func: Callable[[NDArrayf], NDArrayf]) -> None:
        """Test that all the filters applied on DEMs with/without NaNs, work"""
        dem_array = self.dem_1990.data
        dem_filtered = filter_func(dem_array)

        if name in ("median", "mean"):
            assert np.min(dem_array) < np.min(dem_filtered)
            assert np.max(dem_array) > np.max(dem_filtered)
        elif name == "min":
            assert np.min(dem_array) == np.min(dem_filtered)
            assert np.max(dem_array) >= np.max(dem_filtered)
        elif name == "max":
            assert np.min(dem_array) <= np.min(dem_filtered)
            assert np.max(dem_array) == np.max(dem_filtered)

        assert dem_array.shape == dem_filtered.shape

        # Test that it works with NaNs too
        nan_count = 1000
        rng = np.random.default_rng(42)
        cols = rng.integers(0, high=self.dem_1990.width - 1, size=nan_count, dtype=int)
        rows = rng.integers(0, high=self.dem_1990.height - 1, size=nan_count, dtype=int)
        dem_with_nans = np.copy(self.dem_1990.data).squeeze()
        dem_with_nans[rows, cols] = np.nan

        dem_with_nans_filtered = filter_func(dem_with_nans)
        if name in ("median", "mean"):
            # smoothing should not yield values below.above original DEM
            assert np.nanmin(dem_with_nans) < np.nanmin(dem_with_nans_filtered)
            assert np.nanmax(dem_with_nans) > np.nanmax(dem_with_nans_filtered)
            assert np.min(dem_filtered) == np.nanmin(dem_with_nans_filtered)
            assert np.max(dem_filtered) == np.nanmax(dem_with_nans_filtered)
        elif name == "min":
            assert np.nanmin(dem_with_nans) == np.nanmin(dem_with_nans_filtered)
            assert np.min(dem_filtered) == np.nanmin(dem_with_nans_filtered)
            assert np.nanmax(dem_with_nans) > np.nanmax(dem_with_nans_filtered)
        elif name == "max":
            assert np.nanmin(dem_with_nans) < np.nanmin(dem_with_nans_filtered)
            assert np.nanmax(dem_with_nans) == np.nanmax(dem_with_nans_filtered)
            assert np.max(dem_filtered) == np.nanmax(dem_with_nans_filtered)

        # Test that it works with 3D arrays
        if name != "mean":
            array_3d = np.vstack((dem_array[np.newaxis, :], dem_array[np.newaxis, :]))
            dem_filtered = filter_func(array_3d)
            assert array_3d.shape == dem_filtered.shape

            # Tests that it fails with 1D arrays with appropriate error
            data = dem_array[:, 0]
            pytest.raises(ValueError, filter_func, data)

    def test_generic_filter(self) -> None:
        """Test that the generic filter applied on DEMs works"""

        dem_array = self.dem_1990.data
        dem_filtered = xdem.filters.generic_filter(dem_array, np.nanmin, **{"size": 5})  # type:ignore

        assert np.nansum(dem_array) != np.nansum(dem_filtered)

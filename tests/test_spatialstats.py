"""Functions to test the spatial statistics."""

from __future__ import annotations

import os
import warnings
from typing import Any

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
import skgstat
from geoutils import Raster, Vector

import xdem
from xdem import examples
from xdem._typing import NDArrayf
from xdem.spatialstats import EmpiricalVariogramKArgs, neff_hugonnet_approx

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from skgstat import models

PLOT = False


def load_ref_and_diff() -> tuple[Raster, Raster, NDArrayf, Vector]:
    """Load example files to try coregistration methods with."""

    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    outlines = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    ddem = Raster(examples.get_path("longyearbyen_ddem"))
    mask = outlines.create_mask(ddem)

    return reference_raster, ddem, mask, outlines


class TestBinning:

    # Load data for the entire test class
    ref, diff, mask, outlines = load_ref_and_diff()

    # Derive terrain attributes
    slope, aspect, maximum_curv = xdem.terrain.get_terrain_attribute(
        ref, attribute=["slope", "aspect", "maximum_curvature"]
    )

    def test_nd_binning(self) -> None:
        """Check that the nd_binning function works adequately and save dataframes to files for later tests"""

        # Subsampler
        indices = gu.raster.subsample_array(
            self.diff.data.flatten(), subsample=10000, return_indices=True, random_state=42
        )

        # 1D binning, by default will create 10 bins
        df = xdem.spatialstats.nd_binning(
            values=self.diff.data.flatten()[indices],
            list_var=[self.slope.data.flatten()[indices]],
            list_var_names=["slope"],
        )

        # Check length matches
        assert df.shape[0] == 10
        # Check bin edges match the minimum and maximum of binning variable
        assert np.nanmin(self.slope.data.flatten()[indices]) == np.min(pd.IntervalIndex(df.slope).left)
        assert np.nanmax(self.slope.data.flatten()[indices]) == np.max(pd.IntervalIndex(df.slope).right)

        # NMAD should go up quite a bit with slope, more than 8 m between the two extreme bins
        assert df.nmad.values[-1] - df.nmad.values[0] > 8

        # 1D binning with 20 bins
        df = xdem.spatialstats.nd_binning(
            values=self.diff.data.flatten()[indices],
            list_var=[self.slope.data.flatten()[indices]],
            list_var_names=["slope"],
            list_var_bins=20,
        )
        # Check length matches
        assert df.shape[0] == 20

        # Define function for custom stat
        def percentile_80(a: NDArrayf) -> np.floating[Any]:
            return np.nanpercentile(a, 80)

        # Check the function runs with custom functions
        df = xdem.spatialstats.nd_binning(
            values=self.diff.data.flatten()[indices],
            list_var=[self.slope.data.flatten()[indices]],
            list_var_names=["slope"],
            statistics=[percentile_80],
        )
        # Check that the count is added automatically by the function when not user-defined
        assert "count" in df.columns.values

        # 2D binning
        df = xdem.spatialstats.nd_binning(
            values=self.diff.data.flatten()[indices],
            list_var=[self.slope.data.flatten()[indices], self.ref.data.flatten()[indices]],
            list_var_names=["slope", "elevation"],
        )

        # Dataframe should contain two 1D binning of length 10 and one 2D binning of length 100
        assert df.shape[0] == (10 + 10 + 100)

        # 3D binning
        df = xdem.spatialstats.nd_binning(
            values=self.diff.data.flatten()[indices],
            list_var=[
                self.slope.data.flatten()[indices],
                self.ref.data.flatten()[indices],
                self.aspect.data.flatten()[indices],
            ],
            list_var_names=["slope", "elevation", "aspect"],
            list_var_bins=4,
        )

        # Dataframe should contain three 1D binning of length 10 and three 2D binning of length 100 and one 2D binning
        # of length 1000
        assert df.shape[0] == (4**3 + 3 * 4**2 + 3 * 4)

        # Save for later use
        df.to_csv(os.path.join(examples._EXAMPLES_DIRECTORY, "df_3d_binning_slope_elevation_aspect.csv"), index=False)

    def test_interp_nd_binning_artificial_data(self) -> None:
        """Check that the N-dimensional interpolation works correctly using artificial data"""

        # Check the function works with a classic input (see example)
        df = pd.DataFrame(
            {
                "var1": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                "var2": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3))
        fun = xdem.spatialstats.interp_nd_binning(
            df, list_var_names=["var1", "var2"], statistic="statistic", min_count=None
        )

        # Check that the dimensions are rightly ordered
        assert fun((1, 3)) == df[np.logical_and(df["var1"] == 1, df["var2"] == 3)]["statistic"].values[0]
        assert fun((3, 1)) == df[np.logical_and(df["var1"] == 3, df["var2"] == 1)]["statistic"].values[0]

        # Check interpolation falls right on values for points (1, 1), (1, 2) etc...
        for i in range(3):
            for j in range(3):
                x = df["var1"][3 * i + j]
                y = df["var2"][3 * i + j]
                stat = df["statistic"][3 * i + j]
                assert fun((x, y)) == stat

        # Check bilinear interpolation inside the grid
        points_in = [(1.5, 1.5), (1.5, 2.5), (2.5, 1.5), (2.5, 2.5)]
        for point in points_in:
            # The values are 1 off from Python indexes
            x = point[0] - 1
            y = point[1] - 1
            # Get four closest points on the grid
            xlow = int(x - 0.5)
            xupp = int(x + 0.5)
            ylow = int(y - 0.5)
            yupp = int(y + 0.5)
            # Check the bilinear interpolation matches the mean value of those 4 points (equivalent as its the middle)
            assert fun((y + 1, x + 1)) == np.mean([arr[xlow, ylow], arr[xupp, ylow], arr[xupp, yupp], arr[xlow, yupp]])

        # Check bilinear extrapolation for points at 1 spacing outside from the input grid
        points_out = (
            [(0, i) for i in np.arange(1, 4)]
            + [(i, 0) for i in np.arange(1, 4)]
            + [(4, i) for i in np.arange(1, 4)]
            + [(i, 4) for i in np.arange(4, 1)]
        )
        for point in points_out:
            x = point[0] - 1
            y = point[1] - 1
            val_extra = fun((y + 1, x + 1))
            # (OUTDATED: The difference between the points extrapolated outside should be linear with the grid edges,
            # # i.e. the same as the difference as the first points inside the grid along the same axis)
            # if point[0] == 0:
            #     diff_in = arr[x + 2, y] - arr[x + 1, y]
            #     diff_out = arr[x + 1, y] - val_extra
            # elif point[0] == 4:
            #     diff_in = arr[x - 2, y] - arr[x - 1, y]
            #     diff_out = arr[x - 1, y] - val_extra
            # elif point[1] == 0:
            #     diff_in = arr[x, y + 2] - arr[x, y + 1]
            #     diff_out = arr[x, y + 1] - val_extra
            # # has to be y == 4
            # else:
            #     diff_in = arr[x, y - 2] - arr[x, y - 1]
            #     diff_out = arr[x, y - 1] - val_extra
            # assert diff_in == diff_out

            # Update with nearest default: the value should be that of the nearest!
            if point[0] == 0:
                near = arr[x + 1, y]
            elif point[0] == 4:
                near = arr[x - 1, y]
            elif point[1] == 0:
                near = arr[x, y + 1]
            else:
                near = arr[x, y - 1]
            assert near == val_extra

        # Check that the output extrapolates as "nearest neighbour" far outside the grid
        points_far_out = (
            [(-10, i) for i in np.arange(1, 4)]
            + [(i, -10) for i in np.arange(1, 4)]
            + [(14, i) for i in np.arange(1, 4)]
            + [(i, 14) for i in np.arange(4, 1)]
        )
        for point in points_far_out:
            x = point[0] - 1
            y = point[1] - 1
            val_extra = fun((y + 1, x + 1))
            # Update with nearest default: the value should be that of the nearest!
            if point[0] == -10:
                near = arr[0, y]
            elif point[0] == 14:
                near = arr[-1, y]
            elif point[1] == -10:
                near = arr[x, 0]
            else:
                near = arr[x, -1]
            assert near == val_extra

        # Check that the output is rightly ordered in 3 dimensions, and works with varying dimension lengths
        vec1 = np.arange(1, 3)
        vec2 = np.arange(1, 4)
        vec3 = np.arange(1, 5)
        x, y, z = np.meshgrid(vec1, vec2, vec3)
        df = pd.DataFrame(
            {"var1": x.ravel(), "var2": y.ravel(), "var3": z.ravel(), "statistic": np.arange(len(x.ravel()))}
        )
        fun = xdem.spatialstats.interp_nd_binning(
            df, list_var_names=["var1", "var2", "var3"], statistic="statistic", min_count=None
        )
        for i in vec1:
            for j in vec2:
                for k in vec3:
                    assert (
                        fun((i, j, k))
                        == df[np.logical_and.reduce((df["var1"] == i, df["var2"] == j, df["var3"] == k))][
                            "statistic"
                        ].values[0]
                    )

        # Check that the linear extrapolation respects nearest neighbour and doesn't go negative

        # The following example used to give a negative value
        df = pd.DataFrame(
            {
                "var1": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
                "var2": [0, 0, 0, 0, 5, 5, 5, 5, 5.5, 5.5, 5.5, 5.5, 6, 6, 6, 6],
                "statistic": [0, 0, 0, 0, 1, 1, 1, 1, np.nan, 1, 1, np.nan, np.nan, 0, 0, np.nan],
            }
        )
        fun = xdem.spatialstats.interp_nd_binning(
            df, list_var_names=["var1", "var2"], statistic="statistic", min_count=None
        )

        # Check it is now positive or equal to zero
        assert fun((5, 100)) >= 0

    def test_interp_nd_binning_realdata(self) -> None:
        """Check that the function works well with outputs from the nd_binning function"""

        # Read nd_binning output
        df = pd.read_csv(
            os.path.join(examples._EXAMPLES_DIRECTORY, "df_3d_binning_slope_elevation_aspect.csv"), index_col=None
        )

        # First, in 1D
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names="slope")

        # Check a value is returned inside the grid
        assert np.isfinite(fun(([15],)))
        # Check the nmad increases with slope
        assert fun(([20],)) > fun(([0],))
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50],))))

        # Check when the first passed binning variable contains NaNs because of other binning variable
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names="elevation")

        # Then, in 2D
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=["slope", "elevation"])

        # Check a value is returned inside the grid
        assert np.isfinite(fun(([15], [1000])))
        # Check the nmad increases with slope
        assert fun(([40], [300])) > fun(([10], [300]))
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50], [-500, 3000]))))

        # Then in 3D
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=["slope", "elevation", "aspect"])

        # Check a value is returned inside the grid
        assert np.isfinite(fun(([15], [1000], [np.pi])))
        # Check the nmad increases with slope
        assert fun(([30], [300], [np.pi])) > fun(([10], [300], [np.pi]))
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50], [-500, 3000], [-2 * np.pi, 4 * np.pi]))))

    def test_get_perbin_nd_binning(self) -> None:
        """Test the get per-bin function."""

        # Read nd_binning output
        df = pd.read_csv(
            os.path.join(examples._EXAMPLES_DIRECTORY, "df_3d_binning_slope_elevation_aspect.csv"), index_col=None
        )

        # Get values for arrays from the above 3D binning
        perbin_values = xdem.spatialstats.get_perbin_nd_binning(
            df=df,
            list_var=[
                self.slope.data,
                self.ref.data,
                self.aspect.data,
            ],
            list_var_names=["slope", "elevation", "aspect"],
        )

        # Check that the function preserves the shape
        assert np.shape(self.slope.data) == np.shape(perbin_values)

        # Check that the bin are rightly recognized
        df = df[df.nd == 3]
        # Convert the intervals from string due to saving to file
        for var in ["slope", "elevation", "aspect"]:
            df[var] = [xdem.spatialstats._pandas_str_to_interval(x) for x in df[var]]

        # Take 1000 random points in the array
        rng = np.random.default_rng(42)
        xrand = rng.integers(low=0, high=perbin_values.shape[0], size=1000)
        yrand = rng.integers(low=0, high=perbin_values.shape[1], size=1000)

        for i in range(len(xrand)):

            # Get the value at the random point for elevation, slope, aspect
            x = xrand[i]
            y = yrand[i]
            h = self.ref.data.filled(np.nan)[x, y]
            slp = self.slope.data.filled(np.nan)[x, y]
            asp = self.aspect.data.filled(np.nan)[x, y]

            if np.logical_or.reduce((np.isnan(h), np.isnan(slp), np.isnan(asp))):
                continue

            # Isolate the bin in the dataframe
            index_bin = np.logical_and.reduce(
                (
                    [h in interv for interv in df["elevation"]],
                    [slp in interv for interv in df["slope"]],
                    [asp in interv for interv in df["aspect"]],
                )
            )
            # It might not exist in the binning intervals (if extreme values were not subsampled in test_nd_binning)
            if np.count_nonzero(index_bin) == 0:
                continue
            # Otherwise there should be only one
            assert np.count_nonzero(index_bin) == 1

            # Get the statistic value and verify that this was the one returned by the function
            statistic_value = df["nanmedian"][index_bin].values[0]
            # Nan equality does not work, so we compare finite values first
            if ~np.isnan(statistic_value):
                assert statistic_value == perbin_values[x, y]
            # And then check that a NaN is returned if it is the statistic
            else:
                assert np.isnan(perbin_values[x, y])

    def test_two_step_standardization(self) -> None:
        """Test two-step standardization function"""

        # Reproduce the first steps of binning
        df_binning = xdem.spatialstats.nd_binning(
            values=self.diff[~self.mask],
            list_var=[self.slope[~self.mask], self.maximum_curv[~self.mask]],
            list_var_names=["var1", "var2"],
            statistics=[gu.stats.nmad],
        )
        unscaled_fun = xdem.spatialstats.interp_nd_binning(
            df_binning, list_var_names=["var1", "var2"], statistic="nmad"
        )
        # The zscore spread should not be one right after binning
        zscores = self.diff[~self.mask] / unscaled_fun((self.slope[~self.mask], self.maximum_curv[~self.mask]))
        scale_fac = gu.stats.nmad(zscores)
        assert scale_fac != 1

        # Filter with a factor of 3 and the standard deviation (not default values) and check the function outputs
        # the exact same array
        zscores[np.abs(zscores) > 3 * np.nanstd(zscores)] = np.nan
        scale_fac_std = np.nanstd(zscores)
        zscores /= scale_fac_std
        zscores_2, final_func = xdem.spatialstats.two_step_standardization(
            dvalues=self.diff[~self.mask],
            list_var=[self.slope[~self.mask], self.maximum_curv[~self.mask]],
            unscaled_error_fun=unscaled_fun,
            spread_statistic=np.nanstd,
            fac_spread_outliers=3,
        )
        assert np.array_equal(zscores, zscores_2, equal_nan=True)

        # Check the output of the scaled function is simply the unscaled one times the spread statistic
        test_slopes = np.linspace(0, 50, 50)
        test_max_curvs = np.linspace(0, 10, 50)
        assert np.array_equal(
            unscaled_fun((test_slopes, test_max_curvs)) * scale_fac_std, final_func((test_slopes, test_max_curvs))
        )

    def test_estimate_model_heteroscedasticity_and_infer_from_stable(self) -> None:
        """Test consistency of outputs and errors in wrapper functions for estimation of heteroscedasticity"""

        # Test infer function
        errors_1, df_binning_1, err_fun_1 = xdem.spatialstats.infer_heteroscedasticity_from_stable(
            dvalues=self.diff, list_var=[self.slope, self.maximum_curv], unstable_mask=self.outlines
        )

        df_binning_2, err_fun_2 = xdem.spatialstats._estimate_model_heteroscedasticity(
            dvalues=self.diff[~self.mask],
            list_var=[self.slope[~self.mask], self.maximum_curv[~self.mask]],
            list_var_names=["var1", "var2"],
        )

        pd.testing.assert_frame_equal(df_binning_1, df_binning_2)
        test_slopes = np.linspace(0, 50, 50)
        test_max_curvs = np.linspace(0, 10, 50)
        assert np.array_equal(err_fun_1((test_slopes, test_max_curvs)), err_fun_2((test_slopes, test_max_curvs)))

        # Test the error map is consistent as well
        errors_2_arr = err_fun_2((self.slope.get_nanarray(), self.maximum_curv.get_nanarray()))
        errors_1_arr = gu.raster.get_array_and_mask(errors_1)[0]
        assert np.array_equal(errors_1_arr, errors_2_arr, equal_nan=True)

        # Save for use in TestVariogram
        errors_1.save(os.path.join(examples._EXAMPLES_DIRECTORY, "dh_error.tif"))

        # Check that errors are raised with wrong input
        with pytest.raises(ValueError, match="The values must be a Raster or NumPy array, or a list of those."):
            xdem.spatialstats.infer_heteroscedasticity_from_stable(
                dvalues="not_an_array", stable_mask=~self.mask, list_var=[self.slope.get_nanarray()]
            )
        with pytest.raises(ValueError, match="The stable mask must be a Vector, Mask, GeoDataFrame or NumPy array."):
            xdem.spatialstats.infer_heteroscedasticity_from_stable(
                dvalues=self.diff, stable_mask="not_a_vector_or_array", list_var=[self.slope.get_nanarray()]
            )
        with pytest.raises(ValueError, match="The unstable mask must be a Vector, Mask, GeoDataFrame or NumPy array."):
            xdem.spatialstats.infer_heteroscedasticity_from_stable(
                dvalues=self.diff, unstable_mask="not_a_vector_or_array", list_var=[self.slope.get_nanarray()]
            )

        with pytest.raises(
            ValueError,
            match="The stable mask can only passed as a Vector or GeoDataFrame if the input "
            "values contain a Raster.",
        ):
            xdem.spatialstats.infer_heteroscedasticity_from_stable(
                dvalues=self.diff.get_nanarray(), stable_mask=self.outlines, list_var=[self.slope.get_nanarray()]
            )

    def test_plot_binning(self) -> None:

        # Define placeholder data
        df = pd.DataFrame({"var1": [0, 1, 2], "var2": [2, 3, 4], "statistic": [0, 0, 0]})

        # Check that the 1D plotting fails with a warning if the variable or statistic is not well-defined
        with pytest.raises(ValueError, match='The variable "var3" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_1d_binning(df, var_name="var3", statistic_name="statistic")
        with pytest.raises(
            ValueError, match='The statistic "stat" is not part of the provided dataframe column names.'
        ):
            xdem.spatialstats.plot_1d_binning(df, var_name="var1", statistic_name="stat")

        # Same for the 2D plotting
        with pytest.raises(ValueError, match='The variable "var3" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_2d_binning(df, var_name_1="var3", var_name_2="var1", statistic_name="statistic")
        with pytest.raises(
            ValueError, match='The statistic "stat" is not part of the provided dataframe column names.'
        ):
            xdem.spatialstats.plot_2d_binning(df, var_name_1="var1", var_name_2="var1", statistic_name="stat")


class TestVariogram:

    ref, diff, mask, outlines = load_ref_and_diff()

    def test_sample_multirange_variogram_default(self) -> None:
        """Verify that the default function runs, and its basic output"""

        # Check the variogram output is consistent for a random state
        df = xdem.spatialstats.sample_empirical_variogram(values=self.diff, subsample=10, random_state=42)
        # assert df["exp"][15] == pytest.approx(5.11900520324707, abs=1e-3)
        assert df["lags"][15] == pytest.approx(5120)
        assert df["count"][15] == 2
        # With a single run, no error can be estimated
        assert all(np.isnan(df.err_exp.values))

        # Check that all type of coordinate inputs work
        # Only the array and the ground sampling distance
        xdem.spatialstats.sample_empirical_variogram(
            values=self.diff.data, gsd=self.diff.res[0], subsample=10, random_state=42
        )

        # Test multiple runs
        df2 = xdem.spatialstats.sample_empirical_variogram(
            values=self.diff, subsample=10, random_state=42, n_variograms=2
        )

        # Check that an error is estimated
        assert any(~np.isnan(df2.err_exp.values))

        # Test that running on several cores does not trigger any error
        xdem.spatialstats.sample_empirical_variogram(
            values=self.diff, subsample=10, random_state=42, n_variograms=2, n_jobs=2
        )

        # Test plotting of empirical variogram by itself
        if PLOT:
            xdem.spatialstats.plot_variogram(df2)

    def test_sample_empirical_variogram_speed(self) -> None:
        """Verify that no speed is lost outside of routines on variogram sampling by comparing manually to skgstat"""

        values = self.diff
        subsample = 10

        # First, run the xDEM wrapper function
        # t0 = time.time()
        df = xdem.spatialstats.sample_empirical_variogram(values=values, subsample=subsample, random_state=42)
        # t1 = time.time()

        # Second, do it manually with skgstat

        # Ground sampling distance
        gsd = values.res[0]
        # Coords
        nx, ny = values.shape
        x, y = np.meshgrid(np.arange(0, values.shape[0] * gsd, gsd), np.arange(0, values.shape[1] * gsd, gsd))
        coords = np.dstack((x.ravel(), y.ravel())).squeeze()

        # Redefine parameters fed to skgstat manually
        # Maxlag
        maxlag = np.sqrt(
            (np.max(coords[:, 0]) - np.min(coords[:, 0])) ** 2 + (np.max(coords[:, 1]) - np.min(coords[:, 1])) ** 2
        )

        # Binning function
        bin_func = []
        right_bin_edge = np.sqrt(2) * gsd
        while right_bin_edge < maxlag:
            bin_func.append(right_bin_edge)
            # We use the default exponential increasing factor of RasterEquidistantMetricSpace, adapted for grids
            right_bin_edge *= np.sqrt(2)
        bin_func.append(maxlag)
        # Extent
        extent = (np.min(coords[:, 0]), np.max(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 1]))
        # Shape
        shape = values.shape

        keyword_arguments = {"subsample": subsample, "extent": extent, "shape": shape}
        runs, samples, ratio_subsample = xdem.spatialstats._choose_cdist_equidistant_sampling_parameters(
            **keyword_arguments
        )

        # Index of valid values
        values_arr, mask_nodata = gu.raster.get_array_and_mask(values)

        # t3 = time.time()
        rems = skgstat.RasterEquidistantMetricSpace(
            coords=coords[~mask_nodata.ravel(), :],
            shape=shape,
            extent=extent,
            samples=samples,
            ratio_subsample=ratio_subsample,
            runs=runs,
            # Now even for a n_variograms=1 we sample other integers for the random number generator
            rnd=np.random.default_rng(42).choice(1, 1, replace=False),
        )
        V = skgstat.Variogram(
            rems,
            values=values_arr[~mask_nodata].ravel(),
            normalize=False,
            fit_method=None,
            bin_func=bin_func,
            maxlag=maxlag,
        )
        # t4 = time.time()

        # Get bins, empirical variogram values, and bin count
        bins, exp = V.get_empirical(bin_center=False)
        count = V.bin_count

        # Write to dataframe
        df2 = pd.DataFrame()
        df2 = df2.assign(exp=exp, bins=bins, count=count)
        df2 = df2.rename(columns={"bins": "lags"})
        df2["err_exp"] = np.nan
        df2.drop(df2.tail(1).index, inplace=True)
        df2 = df2.astype({"exp": "float64", "err_exp": "float64", "lags": "float64", "count": "int64"})

        # t2 = time.time()

        # Check if the two frames are equal
        pd.testing.assert_frame_equal(df, df2)

        # Check that the two ways are taking the same time with 50% margin
        # time_method_1 = t1 - t0
        # time_method_2 = t2 - t1
        # assert time_method_1 == pytest.approx(time_method_2, rel=0.5)

        # Check that all this time is based on variogram sampling at about 70%, even with the smallest number of
        # samples of 10
        # time_metricspace_variogram = t4 - t3
        # assert time_metricspace_variogram == pytest.approx(time_method_2, rel=0.3)

    @pytest.mark.parametrize(
        "subsample_method", ["pdist_point", "pdist_ring", "pdist_disk", "cdist_point"]
    )  # type: ignore
    def test_sample_multirange_variogram_methods(self, subsample_method) -> None:
        """Verify that all other methods run"""

        # Check the variogram estimation runs for several methods
        df = xdem.spatialstats.sample_empirical_variogram(
            values=self.diff, subsample=10, random_state=42, subsample_method=subsample_method
        )

        assert not df.empty

        # Check that the output is correct
        expected_columns = ["exp", "lags", "count"]
        expected_dtypes = [np.float64, np.float64, np.int64]
        for col in expected_columns:
            # Check that the column exists
            assert col in df.columns
            # Check that the column has the correct dtype
            assert df[col].dtype == expected_dtypes[expected_columns.index(col)]

    def test_sample_multirange_variogram_args(self) -> None:
        """Verify that optional parameters run only for their specific method, raise warning otherwise"""

        # Define parameters
        pdist_args: EmpiricalVariogramKArgs = {"pdist_multi_ranges": [0, self.diff.res[0] * 5, self.diff.res[0] * 10]}
        cdist_args: EmpiricalVariogramKArgs = {"ratio_subsample": 0.5, "runs": 10}
        nonsense_args = {"thisarg": "shouldnotexist"}

        # Check the function raises a warning for optional arguments incorrect to the method
        with pytest.warns(UserWarning):
            # An argument only use by cdist with a pdist method
            xdem.spatialstats.sample_empirical_variogram(
                values=self.diff, subsample=10, random_state=42, subsample_method="pdist_ring", **cdist_args
            )

        with pytest.warns(UserWarning):
            # Same here
            xdem.spatialstats.sample_empirical_variogram(
                values=self.diff,
                subsample=10,
                random_state=42,
                subsample_method="cdist_equidistant",
                **pdist_args,
            )

        with pytest.warns(UserWarning):
            # Should also raise a warning for a nonsense argument
            xdem.spatialstats.sample_empirical_variogram(
                values=self.diff,
                subsample=10,
                random_state=42,
                subsample_method="cdist_equidistant",
                **nonsense_args,  # type: ignore
            )

        # Check the function passes optional arguments specific to pdist methods without warning
        xdem.spatialstats.sample_empirical_variogram(
            values=self.diff, subsample=10, random_state=42, subsample_method="pdist_ring", **pdist_args
        )

        # Check the function passes optional arguments specific to cdist methods without warning
        xdem.spatialstats.sample_empirical_variogram(
            values=self.diff, random_state=42, subsample=10, subsample_method="cdist_equidistant", **cdist_args
        )

    # N is the number of samples in an ensemble
    @pytest.mark.parametrize("subsample", [10, 100, 1000, 10000])  # type: ignore
    @pytest.mark.parametrize("shape", [(50, 50), (100, 100), (500, 500)])  # type: ignore
    def test_choose_cdist_equidistant_sampling_parameters(self, subsample: int, shape: tuple[int, int]) -> None:
        """Verify that the automatically-derived parameters of equidistant sampling are sound"""

        # Assign an arbitrary extent
        extent = (0, 1, 0, 1)

        # The number of different pairwise combinations in a single ensemble (scipy.pdist function) is N*(N-1)/2
        # which is approximately N**2/2
        pdist_pairwise_combinations = subsample**2 / 2

        # Run the function
        keyword_arguments = {"subsample": subsample, "extent": extent, "shape": shape}
        runs, samples, ratio_subsample = xdem.spatialstats._choose_cdist_equidistant_sampling_parameters(
            **keyword_arguments
        )

        # There is at least 2 samples
        assert samples > 2
        # Can only be maximum 100 runs
        assert runs <= 100

        # Get maxdist
        maxdist = np.sqrt((extent[1] - extent[0]) ** 2 + (extent[3] - extent[2]) ** 2)
        res = np.mean([(extent[1] - extent[0]) / (shape[0] - 1), (extent[3] - extent[2]) / (shape[1] - 1)])
        # Then, we compute the radius from the center ensemble with the default value of subsample ratio in the function
        # skgstat.RasterEquidistantMetricSpace
        center_radius = np.sqrt(1.0 / ratio_subsample * samples / np.pi) * res
        nb_rings_final = int(2 * np.log(maxdist / center_radius) / np.log(2))
        cdist_pairwise_combinations = runs * samples**2 * nb_rings_final

        # Check the number of pairwise comparisons are the same (within 50%, due to rounding as integers)
        assert pdist_pairwise_combinations == pytest.approx(cdist_pairwise_combinations, rel=0.5, abs=10)

    def test_errors_subsample_parameter(self) -> None:
        """Tests that an error is raised when the subsample argument is too little"""

        keyword_arguments = {"subsample": 3, "extent": (0, 1, 0, 1), "shape": (10, 10)}

        with pytest.raises(ValueError, match="The number of subsamples needs to be at least 10."):
            xdem.spatialstats._choose_cdist_equidistant_sampling_parameters(**keyword_arguments)

    def test_multirange_fit_performance(self) -> None:
        """Verify that the fitting works with artificial dataset"""

        # First, generate a sum of modelled variograms: ranges and  partial sills for three models
        params_real = (100, 0.7, 1000, 0.2, 10000, 0.1)
        r1, ps1, r2, ps2, r3, ps3 = params_real

        x = np.linspace(10, 20000, 500)
        y = models.spherical(x, r=r1, c0=ps1) + models.spherical(x, r=r2, c0=ps2) + models.spherical(x, r=r3, c0=ps3)

        # Add some noise on top of it
        sig = 0.025
        rng = np.random.default_rng(42)
        y_noise = rng.normal(0, sig, size=len(x))

        y_simu = y + y_noise
        sigma = np.ones(len(x)) * sig

        # Put all in a dataframe
        df = pd.DataFrame()
        df = df.assign(lags=x, exp=y_simu, err_exp=sigma)

        # Run the fitting
        fun, params_est = xdem.spatialstats.fit_sum_model_variogram(["spherical", "spherical", "spherical"], df)

        for i in range(len(params_est)):
            # Assert all parameters were correctly estimated within a 30% relative margin
            assert params_real[2 * i] == pytest.approx(params_est["range"].values[i], rel=0.3)
            assert params_real[2 * i + 1] == pytest.approx(params_est["psill"].values[i], rel=0.3)

        if PLOT:
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun])

    def test_check_params_variogram_model(self) -> None:
        """Verify that the checking function for the modelled variogram parameters dataframe returns adequate errors"""

        # Check when missing a column
        with pytest.raises(
            ValueError,
            match='The dataframe with variogram parameters must contain the columns "model",' ' "range" and "psill".',
        ):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["spherical"], "range": [100]})
            )

        # Check with wrong model format
        list_supported_models = ["spherical", "gaussian", "exponential", "cubic", "stable", "matern"]
        with pytest.raises(
            ValueError,
            match="Variogram model name Supraluminal not recognized. Supported models are: "
            + ", ".join(list_supported_models)
            + ".",
        ):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["Supraluminal"], "range": [100], "psill": [1]})
            )

        # Check with wrong range format
        with pytest.raises(ValueError, match="The variogram ranges must be float or integer."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["spherical"], "range": ["a"], "psill": [1]})
            )

        # Check with negative range
        with pytest.raises(ValueError, match="The variogram ranges must have non-zero, positive values."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["spherical"], "range": [-1], "psill": [1]})
            )

        # Check with wrong partial sill format
        with pytest.raises(ValueError, match="The variogram partial sills must be float or integer."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["spherical"], "range": [100], "psill": ["a"]})
            )

        # Check with negative partial sill
        with pytest.raises(ValueError, match="The variogram partial sills must have non-zero, positive values."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["spherical"], "range": [100], "psill": [-1]})
            )

        # Check with a model that requires smoothness and without the smoothness column
        with pytest.raises(
            ValueError,
            match='The dataframe with variogram parameters must contain the column "smooth" '
            "for the smoothness factor when using Matern or Stable models.",
        ):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["stable"], "range": [100], "psill": [1]})
            )

        # Check with wrong smoothness format
        with pytest.raises(ValueError, match="The variogram smoothness parameter must be float or integer."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["stable"], "range": [100], "psill": [1], "smooth": ["a"]})
            )

        # Check with negative smoothness
        with pytest.raises(ValueError, match="The variogram smoothness parameter must have non-zero, positive values."):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={"model": ["stable"], "range": [100], "psill": [1], "smooth": [-1]})
            )

    def test_estimate_model_spatial_correlation_and_infer_from_stable(self) -> None:
        """Test consistency of outputs and errors in wrapper functions for estimation of spatial correlation"""

        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

        # Keep only data on stable
        diff_on_stable = self.diff.copy()
        diff_on_stable.set_mask(self.mask)

        # Load the error map from TestBinning
        errors = Raster(os.path.join(examples._EXAMPLES_DIRECTORY, "dh_error.tif"))

        # Standardize the differences
        zscores = diff_on_stable / errors

        # Run wrapper estimate and model function
        emp_vgm_1, params_model_vgm_1, _ = xdem.spatialstats._estimate_model_spatial_correlation(
            dvalues=zscores, list_models=["Gau", "Sph"], subsample=10, random_state=42
        )

        # Check that the output matches that of the original function under the same random state
        emp_vgm_2 = xdem.spatialstats.sample_empirical_variogram(
            values=zscores, estimator="dowd", subsample=10, random_state=42
        )
        pd.testing.assert_frame_equal(emp_vgm_1, emp_vgm_2)
        params_model_vgm_2 = xdem.spatialstats.fit_sum_model_variogram(
            list_models=["Gau", "Sph"], empirical_variogram=emp_vgm_2
        )[1]
        pd.testing.assert_frame_equal(params_model_vgm_1, params_model_vgm_2)

        # Run wrapper infer from stable function with a Raster and the mask, and check the consistency there as well
        emp_vgm_3, params_model_vgm_3, _ = xdem.spatialstats.infer_spatial_correlation_from_stable(
            dvalues=zscores, stable_mask=~self.mask, list_models=["Gau", "Sph"], subsample=10, random_state=42
        )
        pd.testing.assert_frame_equal(emp_vgm_1, emp_vgm_3)
        pd.testing.assert_frame_equal(params_model_vgm_1, params_model_vgm_3)

        # Run again with array instead of Raster as input
        zscores_arr = gu.raster.get_array_and_mask(zscores)[0]
        emp_vgm_4, params_model_vgm_4, _ = xdem.spatialstats.infer_spatial_correlation_from_stable(
            dvalues=zscores_arr,
            gsd=self.diff.res[0],
            stable_mask=~self.mask,
            list_models=["Gau", "Sph"],
            subsample=10,
            random_state=42,
        )
        pd.testing.assert_frame_equal(emp_vgm_1, emp_vgm_4)
        pd.testing.assert_frame_equal(params_model_vgm_1, params_model_vgm_4)

        # Run with a decent amount of samples to save to file
        _, params_model_vgm_5, _ = xdem.spatialstats.infer_spatial_correlation_from_stable(
            dvalues=zscores_arr,
            gsd=self.diff.res[0],
            stable_mask=~self.mask,
            list_models=["Gau", "Sph"],
            subsample=200,
            random_state=42,
        )
        # Save the modelled variogram for later used in TestNeffEstimation
        params_model_vgm_5.to_csv(
            os.path.join(examples._EXAMPLES_DIRECTORY, "df_variogram_model_params.csv"), index=False
        )

        # Check that errors are raised with wrong input
        with pytest.raises(ValueError, match="The values must be a Raster or NumPy array, or a list of those."):
            xdem.spatialstats.infer_spatial_correlation_from_stable(
                dvalues="not_an_array", stable_mask=~self.mask, list_models=["Gau", "Sph"], random_state=42
            )
        with pytest.raises(ValueError, match="The stable mask must be a Vector, Mask, GeoDataFrame or NumPy array."):
            xdem.spatialstats.infer_spatial_correlation_from_stable(
                dvalues=self.diff, stable_mask="not_a_vector_or_array", list_models=["Gau", "Sph"], random_state=42
            )
        with pytest.raises(ValueError, match="The unstable mask must be a Vector, Mask, GeoDataFrame or NumPy array."):
            xdem.spatialstats.infer_spatial_correlation_from_stable(
                dvalues=self.diff, unstable_mask="not_a_vector_or_array", list_models=["Gau", "Sph"], random_state=42
            )
        diff_on_stable_arr = gu.raster.get_array_and_mask(diff_on_stable)[0]
        with pytest.raises(
            ValueError,
            match="The stable mask can only passed as a Vector or GeoDataFrame if the input "
            "values contain a Raster.",
        ):
            xdem.spatialstats.infer_spatial_correlation_from_stable(
                dvalues=diff_on_stable_arr, stable_mask=self.outlines, list_models=["Gau", "Sph"], random_state=42
            )

    def test_empirical_fit_plotting(self) -> None:
        """Verify that the shape of the empirical variogram output works with the fit and plotting"""

        # Check the variogram estimation runs for a random state
        df = xdem.spatialstats.sample_empirical_variogram(
            values=self.diff.data, gsd=self.diff.res[0], subsample=50, random_state=42
        )

        # Single model fit
        fun, _ = xdem.spatialstats.fit_sum_model_variogram(["spherical"], empirical_variogram=df)

        # Triple model fit
        fun2, _ = xdem.spatialstats.fit_sum_model_variogram(
            ["spherical", "spherical", "spherical"], empirical_variogram=df
        )

        if PLOT:
            # Plot with a single model fit
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun])
            # Plot with a triple model fit
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun2])

        # Check that errors are raised with wrong inputs
        # If the experimental variogram values "exp" are not passed
        with pytest.raises(
            ValueError, match='The expected variable "exp" is not part of the provided dataframe column names.'
        ):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={"wrong_name": [1], "lags": [1], "count": [100]}))
        # If the spatial lags "lags" are not passed
        with pytest.raises(
            ValueError, match='The expected variable "lags" is not part of the provided dataframe column names.'
        ):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={"exp": [1], "wrong_name": [1], "count": [100]}))
        # If the pairwise sample count "count" is not passed
        with pytest.raises(
            ValueError, match='The expected variable "count" is not part of the provided dataframe column names.'
        ):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={"exp": [1], "lags": [1], "wrong_name": [100]}))


class TestNeffEstimation:

    ref, diff, _, outlines = load_ref_and_diff()

    @pytest.mark.parametrize("range1", [10**i for i in range(3)])  # type: ignore
    @pytest.mark.parametrize("psill1", [0.1, 1, 10])  # type: ignore
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])  # type: ignore
    @pytest.mark.parametrize("area", [10 ** (2 * i) for i in range(3)])  # type: ignore
    def test_neff_circular_single_range(self, range1: float, psill1: float, model1: float, area: float) -> None:
        """Test the accuracy of numerical integration for one to three models of spherical, gaussian or exponential
        forms to get the number of effective samples"""

        params_variogram_model = pd.DataFrame(data={"model": [model1], "range": [range1], "psill": [psill1]})

        # Exact integration
        neff_circ_exact = xdem.spatialstats.neff_circular_approx_theoretical(
            area=area, params_variogram_model=params_variogram_model
        )
        # Numerical integration
        neff_circ_numer = xdem.spatialstats.neff_circular_approx_numerical(
            area=area, params_variogram_model=params_variogram_model
        )

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.parametrize("range1", [10**i for i in range(2)])  # type: ignore
    @pytest.mark.parametrize("range2", [10**i for i in range(2)])  # type: ignore
    @pytest.mark.parametrize("range3", [10**i for i in range(2)])  # type: ignore
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])  # type: ignore
    @pytest.mark.parametrize("model2", ["spherical", "exponential", "gaussian", "cubic"])  # type: ignore
    def test_neff_circular_three_ranges(
        self, range1: float, range2: float, range3: float, model1: float, model2: float
    ) -> None:
        """Test the accuracy of numerical integration for one to three models of spherical, gaussian or
        exponential forms"""

        area = 1000
        psill1 = 1
        psill2 = 1
        psill3 = 1
        model3 = "spherical"

        params_variogram_model = pd.DataFrame(
            data={
                "model": [model1, model2, model3],
                "range": [range1, range2, range3],
                "psill": [psill1, psill2, psill3],
            }
        )

        # Exact integration
        neff_circ_exact = xdem.spatialstats.neff_circular_approx_theoretical(
            area=area, params_variogram_model=params_variogram_model
        )
        # Numerical integration
        neff_circ_numer = xdem.spatialstats.neff_circular_approx_numerical(
            area=area, params_variogram_model=params_variogram_model
        )

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    def test_neff_exact_and_approx_hugonnet(self) -> None:
        """Test the exact and approximated calculation of the number of effective sample by double covariance sum"""

        # Generate a gridded dataset with varying errors associated to each pixel
        shape = (15, 15)
        errors = np.ones(shape)

        # Coordinates
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        xx, yy = np.meshgrid(x, y)

        # Flatten everything
        coords = np.dstack((xx.ravel(), yy.ravel())).squeeze()
        errors = errors.ravel()

        # Create a list of variogram that, summed, represent the spatial correlation
        params_variogram_model = pd.DataFrame(
            data={"model": ["spherical", "gaussian"], "range": [5, 50], "psill": [0.5, 0.5]}
        )

        # Check that the function runs with default parameters
        neff_exact = xdem.spatialstats.neff_exact(
            coords=coords, errors=errors, params_variogram_model=params_variogram_model
        )

        # Check that the non-vectorized version gives the same result
        neff_exact_nv = xdem.spatialstats.neff_exact(
            coords=coords, errors=errors, params_variogram_model=params_variogram_model, vectorized=False
        )
        assert neff_exact == pytest.approx(neff_exact_nv, rel=0.001)

        # Check that the approximation function runs with default parameters, sampling 100 out of 250 samples
        neff_approx = neff_hugonnet_approx(
            coords=coords, errors=errors, params_variogram_model=params_variogram_model, subsample=100, random_state=42
        )

        # Check that the non-vectorized version gives the same result, sampling 100 out of 250 samples
        neff_approx_nv = neff_hugonnet_approx(
            coords=coords,
            errors=errors,
            params_variogram_model=params_variogram_model,
            subsample=100,
            vectorized=False,
            random_state=42,
        )

        assert neff_approx == pytest.approx(neff_approx_nv, rel=0.001)

        # Check that the approximation is about the same as the original estimate within 10%
        assert neff_approx == pytest.approx(neff_exact, rel=0.1)

        # Check that the approximation works even on large dataset without creating memory errors
        # 100,000 points squared (pairwise) should use more than 64GB of RAM without subsample
        rng = np.random.default_rng(42)
        coords = rng.normal(size=(100000, 2))
        errors = rng.normal(size=(100000))
        # This uses a subsample of 100, so should run just fine despite the large size
        neff_approx_nv = neff_hugonnet_approx(
            coords=coords,
            errors=errors,
            params_variogram_model=params_variogram_model,
            subsample=100,
            vectorized=True,
            random_state=42,
        )
        assert neff_approx_nv is not None

    def test_number_effective_samples(self) -> None:
        """Test that the wrapper function for neff functions behaves correctly and that output values are robust"""

        # The function should return the same result as neff_circular_approx_numerical when using a numerical area
        area = 10000
        params_variogram_model = pd.DataFrame(
            data={"model": ["spherical", "gaussian"], "range": [300, 3000], "psill": [0.5, 0.5]}
        )

        neff1 = xdem.spatialstats.neff_circular_approx_numerical(
            area=area, params_variogram_model=params_variogram_model
        )
        neff2 = xdem.spatialstats.number_effective_samples(area=area, params_variogram_model=params_variogram_model)

        assert neff1 == pytest.approx(neff2, rel=0.0001)

        # The function should return the same results as neff_hugonnet_approx when using a shape area
        # First, get the vector area and compute with the wrapper function
        res = 100.0
        outlines_brom = Vector(self.outlines.ds[self.outlines.ds["NAME"] == "Brombreen"])
        neff1 = xdem.spatialstats.number_effective_samples(
            area=outlines_brom,
            params_variogram_model=params_variogram_model,
            rasterize_resolution=res,
            random_state=42,
            subsample=10,
        )
        # Second, get coordinates manually and compute with the neff_approx_hugonnet function
        mask = outlines_brom.create_mask(xres=res, as_array=True)
        x = res * np.arange(0, mask.shape[0])
        y = res * np.arange(0, mask.shape[1])
        coords = np.array(np.meshgrid(y, x))
        coords_on_mask = coords[:, mask].T
        errors_on_mask = np.ones(len(coords_on_mask))
        neff2 = xdem.spatialstats.neff_hugonnet_approx(
            coords=coords_on_mask,
            errors=errors_on_mask,
            subsample=10,
            params_variogram_model=params_variogram_model,
            random_state=42,
        )
        # We can test the match between values accurately thanks to the random_state
        assert neff1 == pytest.approx(neff2, rel=0.00001)

        # Check that using a Raster as input for the resolution works
        neff3 = xdem.spatialstats.number_effective_samples(
            area=outlines_brom,
            subsample=10,
            params_variogram_model=params_variogram_model,
            rasterize_resolution=self.ref,
            random_state=42,
        )
        # The value should be nearly the same within 10% (the discretization grid is different so affects a tiny bit the
        # result)
        assert neff3 == pytest.approx(neff2, rel=0.1)

        # Check that the number of effective samples matches that of the circular approximation within 25%
        area_brom = np.sum(outlines_brom.ds.area.values)
        neff4 = xdem.spatialstats.number_effective_samples(
            area=area_brom, params_variogram_model=params_variogram_model
        )
        assert neff4 == pytest.approx(neff2, rel=0.25)
        # The circular approximation is always conservative, so should yield a smaller value
        assert neff4 < neff2

        # Check that errors are correctly raised
        with pytest.warns(
            UserWarning,
            match="Resolution for vector rasterization is not defined and thus set at 20% "
            "of the shortest correlation range, which might result in large memory usage.",
        ):
            xdem.spatialstats.number_effective_samples(
                area=outlines_brom, params_variogram_model=params_variogram_model
            )
        with pytest.raises(ValueError, match="Area must be a float, integer, Vector subclass or geopandas dataframe."):
            xdem.spatialstats.number_effective_samples(
                area="not supported", params_variogram_model=params_variogram_model
            )
        with pytest.raises(ValueError, match="The rasterize resolution must be a float, integer or Raster subclass."):
            xdem.spatialstats.number_effective_samples(
                area=outlines_brom, params_variogram_model=params_variogram_model, rasterize_resolution=(10, 10)
            )

    def test_spatial_error_propagation(self) -> None:
        """Test that the spatial error propagation wrapper function runs properly"""

        # Load the error map from TestBinning
        errors = Raster(os.path.join(examples._EXAMPLES_DIRECTORY, "dh_error.tif"))

        # Load the spatial correlation from TestVariogram
        params_variogram_model = pd.read_csv(
            os.path.join(examples._EXAMPLES_DIRECTORY, "df_variogram_model_params.csv"), index_col=None
        )

        # Run the function with vector areas
        areas_vector = [
            self.outlines.ds[self.outlines.ds["NAME"] == "Brombreen"],
            self.outlines.ds[self.outlines.ds["NAME"] == "Medalsbreen"],
        ]

        list_stderr_vec = xdem.spatialstats.spatial_error_propagation(
            areas=areas_vector,
            errors=errors,
            params_variogram_model=params_variogram_model,
            random_state=42,
            subsample=10,
        )

        # Run the function with numeric areas (sum needed for Medalsbreen that has two separate polygons)
        areas_numeric = [np.sum(area_vec.area.values) for area_vec in areas_vector]
        list_stderr = xdem.spatialstats.spatial_error_propagation(
            areas=areas_numeric, errors=errors, params_variogram_model=params_variogram_model
        )

        # Check that the outputs are consistent: the numeric method should always give a neff that is almost the same
        # (20% relative) for those two glaciers as their shape is not too different from a disk
        for i in range(2):
            assert list_stderr_vec[i] == pytest.approx(list_stderr[i], rel=0.2)


class TestSubSampling:
    def test_circular_masking(self) -> None:
        """Test that the circular masking works as intended"""

        # using default (center should be [2,2], radius 2)
        circ = xdem.spatialstats._create_circular_mask((5, 5))
        circ2 = xdem.spatialstats._create_circular_mask((5, 5), center=(2, 2), radius=2)

        # check default center and radius are derived properly
        assert np.array_equal(circ, circ2)

        # check mask
        # masking is not inclusive, i.e. exactly radius=2 won't include the 2nd pixel from the center, but radius>2 will
        eq_circ = np.zeros((5, 5), dtype=bool)
        eq_circ[1:4, 1:4] = True
        assert np.array_equal(circ, eq_circ)

        # check distance is not a multiple of pixels (more accurate subsampling)
        # will create a 1-pixel mask around the center
        circ3 = xdem.spatialstats._create_circular_mask((5, 5), center=(1, 1), radius=1)

        eq_circ3 = np.zeros((5, 5), dtype=bool)
        eq_circ3[1, 1] = True
        assert np.array_equal(circ3, eq_circ3)

        # will create a square mask (<1.5 pixel) around the center
        circ4 = xdem.spatialstats._create_circular_mask((5, 5), center=(1, 1), radius=1.5)
        # should not be the same as radius = 1
        assert not np.array_equal(circ3, circ4)

    def test_ring_masking(self) -> None:
        """Test that the ring masking works as intended"""

        # by default, the mask is only an outside circle (ring of size 0)
        ring1 = xdem.spatialstats._create_ring_mask((5, 5))
        circ1 = xdem.spatialstats._create_circular_mask((5, 5))

        assert np.array_equal(ring1, circ1)

        # test rings with different inner radius
        ring2 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=1, out_radius=2)
        ring3 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=0, out_radius=2)
        ring4 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=1.5, out_radius=2)

        assert np.logical_and(~np.array_equal(ring2, ring3), ~np.array_equal(ring3, ring4))

        # check default
        eq_ring2 = np.zeros((5, 5), dtype=bool)
        eq_ring2[1:4, 1:4] = True
        eq_ring2[2, 2] = False
        assert np.array_equal(ring2, eq_ring2)


class TestPatchesMethod:
    def test_patches_method_loop_quadrant(self) -> None:
        """Check that the patches method with quadrant loops (vectorized=False) functions correctly"""

        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 100000

        # Check the patches method runs
        df, df_full = xdem.spatialstats.patches_method(
            diff,
            unstable_mask=mask,
            gsd=gsd,
            areas=[area],
            random_state=42,
            n_patches=100,
            vectorized=False,
            return_in_patch_statistics=True,
        )

        # First, the summary dataframe
        assert df.shape == (1, 4)
        assert all(df.columns == ["nmad", "nb_indep_patches", "exact_areas", "areas"])

        # Check the sampling is fixed for a random state
        # assert df["nmad"][0] == pytest.approx(1.8401465163449207, abs=1e-3)
        assert df["nb_indep_patches"][0] == 100
        assert df["exact_areas"][0] == pytest.approx(df["areas"][0], rel=0.2)

        # Then, the full dataframe
        assert df_full.shape == (100, 5)

        # Check the sampling is always fixed for a random state
        assert df_full["tile"].values[0] == "47_17"

        # Check that all counts respect the default minimum percentage of 80% valid pixels
        assert all(df_full["count"].values > 0.8 * np.max(df_full["count"].values))

    def test_patches_method_convolution(self) -> None:
        """Check that the patches method with convolution (vectorized=True) functions correctly"""

        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 100000

        # First, the patches method runs with scipy
        df = xdem.spatialstats.patches_method(
            diff,
            unstable_mask=mask,
            gsd=gsd,
            areas=[area, area * 10],
            random_state=42,
            vectorized=True,
            convolution_method="scipy",
        )

        assert df.shape == (2, 4)
        assert all(df.columns == ["nmad", "nb_indep_patches", "exact_areas", "areas"])
        assert df["exact_areas"][0] == pytest.approx(df["areas"][0], rel=0.2)

        # Second, with numba
        # df, df_full = xdem.spatialstats.patches_method(
        #     diff,
        #     unstable_mask=mask.squeeze(),
        #     gsd=gsd,
        #     areas=[area],
        #     random_state=42,
        #     vectorized=True,
        #     convolution_method='numba',
        #     return_in_patch_statistics=True)
        #
        # assert df.shape == (1, 4)
        # assert all(df.columns == ['nmad', 'nb_indep_patches', 'exact_areas', 'areas'])
        # assert df['exact_areas'][0] == pytest.approx(df['areas'][0], rel=0.2)

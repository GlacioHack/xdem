"""Functions to test the spatial statistics."""
from __future__ import annotations
from typing import Tuple

import warnings

import geoutils as gu
import numpy as np
import pandas as pd
import pytest

import xdem
from xdem import examples

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from skgstat import models

PLOT = False

def load_ref_and_diff() -> tuple[gu.georaster.Raster, gu.georaster.Raster, np.ndarray]:
    """Load example files to try coregistration methods with."""

    reference_raster = gu.georaster.Raster(examples.get_path("longyearbyen_ref_dem"))
    glacier_mask = gu.geovector.Vector(examples.get_path("longyearbyen_glacier_outlines"))

    ddem = gu.georaster.Raster(examples.get_path('longyearbyen_ddem'))
    mask = glacier_mask.create_mask(ddem)

    return reference_raster, ddem, mask

class TestVariogram:

    # check that the scripts are running
    @pytest.mark.skip("This test fails randomly! It needs to be fixed.")
    def test_empirical_fit_variogram_running(self):

        # get some data
        diff, mask = load_ref_and_diff()[1:3]

        x, y = diff.coords(offset='center')
        coords = np.dstack((x.flatten(), y.flatten())).squeeze()

        # check the base script runs with right input shape
        df = xdem.spstats.get_empirical_variogram(
            dh=diff.data.flatten()[0:1000],
            coords=coords[0:1000, :],
            nsamp=1000)

        # check the wrapper script runs with various inputs
        # with gsd as input
        df_gsd = xdem.spstats.sample_multirange_empirical_variogram(
            dh=diff.data,
            gsd=diff.res[0],
            nsamp=1000)

        # with coords as input, and "uniform" bin_func
        df_coords = xdem.spstats.sample_multirange_empirical_variogram(
            dh=diff.data.flatten(),
            coords=coords,
            bin_func='uniform',
            nsamp=1000)

        # using more bins
        df_1000_bins = xdem.spstats.sample_multirange_empirical_variogram(
            dh=diff.data,
            gsd=diff.res[0],
            n_lags=1000,
            nsamp=1000)

        # using multiple runs with parallelized function
        df_sig = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data, gsd=diff.res[0], nsamp=1000,
                                                                    nrun=20, nproc=10, maxlag=10000)

        # test plotting
        if PLOT:
            xdem.spstats.plot_vgm(df_sig)

        # single model fit
        fun, _ = xdem.spstats.fit_model_sum_vgm(['Sph'], df_sig)
        if PLOT:
            xdem.spstats.plot_vgm(df_sig, fit_fun=fun)

        try:
            # triple model fit
            fun2, _ = xdem.spstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], emp_vgm_df=df_sig)
            if PLOT:
                xdem.spstats.plot_vgm(df_sig, fit_fun=fun2)
        except RuntimeError as exception:
            if "The maximum number of function evaluations is exceeded." not in str(exception):
                raise exception
            warnings.warn(str(exception))

    def test_multirange_fit_performance(self):

        # first, generate a true sum of variograms with some added noise
        r1, ps1, r2, ps2, r3, ps3 = (100, 0.7, 1000, 0.2, 10000, 0.1)

        x = np.linspace(10, 20000, 500)
        y = models.spherical(x, r=r1, c0=ps1) + models.spherical(x, r=r2, c0=ps2) \
            + models.spherical(x, r=r3, c0=ps3)

        sig = 0.025
        y_noise = np.random.normal(0, sig, size=len(x))

        y_simu = y + y_noise
        sigma = np.ones(len(x))*sig

        df = pd.DataFrame()
        df = df.assign(bins=x, exp=y_simu, exp_sigma=sig)

        # then, run the fitting
        fun, params = xdem.spstats.fit_model_sum_vgm(['Sph', 'Sph', 'Sph'], df)

        if PLOT:
            xdem.spstats.plot_vgm(df, fit_fun=fun)

    def test_neff_estimation(self):

        # test the precision of numerical integration for several spherical models

        # short range
        crange1 = [10**i for i in range(8)]
        # long range
        crange2 = [100*sr for sr in crange1]

        p1 = 0.8
        p2 = 0.2

        for r1 in crange1:
            r2 = crange2[crange1.index(r1)]

            # and for any glacier area
            for area in [10**i for i in range(10)]:

                neff_circ_exact = xdem.spstats.exact_neff_sphsum_circular(
                    area=area, crange1=r1, psill1=p1, crange2=r2, psill2=p2)
                neff_circ_numer = xdem.spstats.neff_circ(area, [(r1, 'Sph', p1), (r2, 'Sph', p2)])

                assert np.abs(neff_circ_exact-neff_circ_numer) < 0.001

class TestSubSampling:

    def test_circular_masking(self):
        """Test that the circular masking works as intended"""

        # using default (center should be [2,2], radius 2)
        circ = xdem.spstats.create_circular_mask((5,5))
        circ2 = xdem.spstats.create_circular_mask((5,5),center=[2,2],radius=2)

        # check default center and radius are derived properly
        assert np.array_equal(circ,circ2)

        # check mask
        # masking is not inclusive, i.e. exactly radius=2 won't include the 2nd pixel from the center, but radius>2 will
        eq_circ = np.zeros((5,5), dtype=bool)
        eq_circ[1:4,1:4]=True
        assert np.array_equal(circ,eq_circ)

        # check distance is not a multiple of pixels (more accurate subsampling)
        # will create a 1-pixel mask around the center
        circ3 = xdem.spstats.create_circular_mask((5,5),center=[1,1],radius=1)

        eq_circ3 = np.zeros((5,5), dtype=bool)
        eq_circ3[1,1] = True
        assert np.array_equal(circ3, eq_circ3)

        # will create a square mask (<1.5 pixel) around the center
        circ4 = xdem.spstats.create_circular_mask((5,5),center=[1,1],radius=1.5)
        # should not be the same as radius = 1
        assert not np.array_equal(circ3,circ4)


    def test_ring_masking(self):
        """Test that the ring masking works as intended"""

        # by default, the mask is only an outside circle (ring of size 0)
        ring1 = xdem.spstats.create_ring_mask((5,5))
        circ1 = xdem.spstats.create_circular_mask((5,5))

        assert np.array_equal(ring1,circ1)

        # test rings with different inner radius
        ring2 = xdem.spstats.create_ring_mask((5,5),in_radius=1,out_radius=2)
        ring3 = xdem.spstats.create_ring_mask((5,5),in_radius=0,out_radius=2)
        ring4 = xdem.spstats.create_ring_mask((5,5),in_radius=1.5,out_radius=2)

        assert np.logical_and(~np.array_equal(ring2,ring3),~np.array_equal(ring3,ring4))

        # check default
        eq_ring2 = np.zeros((5,5), dtype=bool)
        eq_ring2[1:4,1:4] = True
        eq_ring2[2,2] = False
        assert np.array_equal(ring2,eq_ring2)


class TestPatchesMethod:

    def test_patches_method(self):

        diff, mask = load_ref_and_diff()[1:3]

        warnings.filterwarnings("error")
        # check the patches method runs
        df_patches = xdem.spstats.patches_method(
            diff.data.squeeze(),
            mask=~mask.astype(bool).squeeze(),
            gsd=diff.res[0],
            area_size=10000
        )

class TestBinning:

    def test_nd_binning(self):

        ref, diff, mask = load_ref_and_diff()

        slope, aspect = xdem.coreg.calculate_slope_and_aspect(ref.data.squeeze())

        # 1d binning, by default will create 10 bins
        df = xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten()],list_var_names=['slope'])

        # check length matches
        assert df.shape[0] == 10
        # check bin edges match the minimum and maximum of binning variable
        assert np.nanmin(slope) == np.min(pd.IntervalIndex(df.slope).left)
        assert np.nanmax(slope) == np.max(pd.IntervalIndex(df.slope).right)

        # 1d binning with 20 bins
        df = xdem.spstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten()], list_var_names=['slope'],
                                           list_var_bins=[[20]])
        # check length matches
        assert df.shape[0] == 20

        # nmad goes up quite a bit with slope, we can expect a 10 m measurement error difference
        assert df.nmad.values[-1] - df.nmad.values[0] > 10

        # try custom stat
        def percentile_80(a):
            return np.nanpercentile(a, 80)

        # check the function runs with custom functions
        xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten()],list_var_names=['slope'], statistics=['count',percentile_80])

        # 2d binning
        df = xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten(),ref.data.flatten()],list_var_names=['slope','elevation'])

        # dataframe should contain two 1D binning of length 10 and one 2D binning of length 100
        assert df.shape[0] == (10 + 10 + 100)

        # nd binning
        df = xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten(),ref.data.flatten(),aspect.flatten()],list_var_names=['slope','elevation','aspect'])

        # dataframe should contain three 1D binning of length 10 and three 2D binning of length 100 and one 2D binning of length 1000
        assert df.shape[0] == (1000 + 3 * 100 + 3 * 10)

    def test_interp_nd_binning(self):

        # check the function works with a classic input (see example)
        df = pd.DataFrame({"var1": [1, 1, 1, 2, 2, 2, 3, 3, 3], "var2": [1, 2, 3, 1, 2, 3, 1, 2, 3],
                                "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3,3))
        fun = xdem.spstats.interp_nd_binning(df, list_var_names=["var1", "var2"], statistic="statistic", min_count=None)

        # check interpolation falls right on values for points (1, 1), (1, 2) etc...
        for i in range(3):
            for j in range(3):
                x = df['var1'][3 * i + j]
                y = df['var2'][3 * i + j]
                stat = df['statistic'][3 * i + j]
                assert fun((y, x)) == stat

        # check bilinear interpolation inside the grid
        points_in = [(1.5, 1.5), (1.5, 2.5), (2.5, 1.5), (2.5, 2.5)]
        for point in points_in:
            # the values are 1 off from Python indexes
            x = point[0] - 1
            y = point[1] - 1
            # get four closest points on the grid
            xlow = int(x - 0.5)
            xupp = int(x + 0.5)
            ylow = int(y - 0.5)
            yupp = int(y + 0.5)
            # check the bilinear interpolation matches the mean value of those 4 points (equivalent as its the middle)
            assert fun((y + 1, x + 1)) == np.mean([arr[xlow, ylow], arr[xupp, ylow], arr[xupp, yupp], arr[xlow, yupp]])

        # check bilinear extrapolation for points at 1 spacing outside from the input grid
        points_out = [(0, i) for i in np.arange(1, 4)] + [(i, 0) for i in np.arange(1, 4)] \
                     + [(4, i) for i in np.arange(1, 4)] + [(i, 4) for i in np.arange(4, 1)]
        for point in points_out:
            x = point[0] - 1
            y = point[1] - 1
            val_extra = fun((y + 1, x + 1))
            # the difference between the points extrapolated outside should be linear with the grid edges,
            # i.e. the same as the difference as the first points inside the grid along the same axis
            if point[0] == 0:
                diff_in = arr[x + 2, y] - arr[x + 1, y]
                diff_out = arr[x + 1, y] - val_extra
            elif point[0] == 4:
                diff_in = arr[x - 2, y] - arr[x - 1, y]
                diff_out = arr[x - 1, y] - val_extra
            elif point[1] == 0:
                diff_in = arr[x, y + 2] - arr[x, y + 1]
                diff_out = arr[x, y + 1] - val_extra
            # has to be y == 4
            else:
                diff_in = arr[x, y - 2] - arr[x, y - 1]
                diff_out = arr[x, y - 1] - val_extra
            assert diff_in == diff_out

        # check if it works with nd_binning output
        ref, diff, mask = load_ref_and_diff()
        slope, aspect = xdem.coreg.calculate_slope_and_aspect(ref.data.squeeze())

        df = xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten(),ref.data.flatten(),aspect.flatten()],list_var_names=['slope','elevation','aspect'])

        # in 1d
        fun = xdem.spstats.interp_nd_binning(df, list_var_names='slope')

        # check a value is returned inside the grid
        assert np.isfinite(fun([15]))
        # check the nmad increases with slope
        assert fun([20]) > fun([0])
        # check a value is returned outside the grid
        assert all(np.isfinite(fun([-5,50])))

        # in 2d
        fun = xdem.spstats.interp_nd_binning(df, list_var_names=['slope','elevation'])

        # check a value is returned inside the grid
        assert np.isfinite(fun([15, 1000]))
        # check the nmad increases with slope
        assert fun([20, 1000]) > fun([0, 1000])
        # check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50],[-500,3000]))))

        # in 3d, let's decrease the number of bins to get something with enough samples
        df = xdem.spstats.nd_binning(values=diff.data.flatten(),list_var=[slope.flatten(),ref.data.flatten(),aspect.flatten()],list_var_names=['slope','elevation','aspect'], list_var_bins=3)
        fun = xdem.spstats.interp_nd_binning(df, list_var_names=['slope','elevation','aspect'])

        # check a value is returned inside the grid
        assert np.isfinite(fun([15,1000, np.pi]))
        # check the nmad increases with slope
        assert fun([20, 1000, np.pi]) > fun([0, 1000, np.pi])
        # check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50],[-500,3000],[-2*np.pi,4*np.pi]))))



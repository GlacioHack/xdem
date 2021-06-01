"""
Routines to test the spatial statistics functions.

"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import xdem
from xdem import examples
import geoutils as gu
from skgstat import models
import pandas as pd

def load_diff() -> Tuple[gu.georaster.Raster, np.ndarray] :
    """Load example files to try coregistration methods with."""
    examples.download_longyearbyen_examples(overwrite=False)

    reference_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_ref_dem"])
    to_be_aligned_raster = gu.georaster.Raster(examples.FILEPATHS["longyearbyen_tba_dem"])
    glacier_mask = gu.geovector.Vector(examples.FILEPATHS["longyearbyen_glacier_outlines"])

    metadata = {}
    aligned_raster, _ = xdem.coreg.coregister(reference_raster, to_be_aligned_raster, method="amaury", mask=glacier_mask,
                                     metadata=metadata)

    diff = xdem.spatial_tools.subtract_rasters(reference_raster,aligned_raster)
    mask = glacier_mask.create_mask(diff)

    return diff, mask

class TestVariogram:

    # check that the scripts are running
    def test_empirical_fit_variogram_running(self):

        # get some data
        diff, mask = load_diff()
        x, y = diff.coords(offset='center')
        coords = np.dstack((x.flatten(), y.flatten())).squeeze()


        # check the base script runs with right input shape
        df = xdem.spstats.get_empirical_variogram(dh=diff.data.flatten()[0:1000],coords=coords[0:1000,:])

        # check the wrapper script runs with various inputs
        # with gsd as input
        df_gsd = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0])

        # with coords as input, and "uniform" bin_func
        df_coords = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data.flatten(),coords=coords,
                                                                       bin_func='uniform')

        # using more bins
        df_1000_bins = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0],n_lags=1000)

        # using multiple runs with parallelized function
        df_sig = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0], nsamp=1000,
                                                                    nrun=20, nproc=10, maxlag = 10000)

        # test plotting
        xdem.spstats.plot_vgm(df_sig)

        # single model fit
        fun, _ = xdem.spstats.fit_model_sum_vgm(['Sph'],df_sig)
        xdem.spstats.plot_vgm(df_sig,fit_fun=fun)

        # triple model fit
        fun2, _ = xdem.spstats.fit_model_sum_vgm(['Sph','Sph','Sph'],emp_vgm_df=df_sig)
        xdem.spstats.plot_vgm(df_sig,fit_fun=fun2)

    def test_multirange_fit_performance(self):

        # first, generate a true sum of variograms with some added noise
        r1, ps1, r2, ps2, r3, ps3 = (100,0.7,1000,0.2,10000,0.1)

        x = np.linspace(10,20000,500)
        y = models.spherical(x,r=r1,c0=ps1) + models.spherical(x,r=r2,c0=ps2) \
            + models.spherical(x,r=r3,c0=ps3)

        sig = 0.025
        y_noise = np.random.normal(0,sig,size=len(x))

        y_simu = y + y_noise
        sigma = np.ones(len(x))*sig

        df = pd.DataFrame()
        df = df.assign(bins=x,exp=y_simu,exp_sigma=sig)

        # then, run the fitting
        fun, params = xdem.spstats.fit_model_sum_vgm(['Sph','Sph','Sph'],df)

        xdem.spstats.plot_vgm(df,fit_fun=fun)


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

                neff_circ_exact = xdem.spstats.exact_neff_sphsum_circular(area=area,crange1=r1,psill1=p1,crange2=r2
                                                                          ,psill2=p2)
                neff_circ_numer = xdem.spstats.neff_circ(area,[(r1,'Sph',p1),(r2,'Sph',p2)])

                assert np.abs(neff_circ_exact-neff_circ_numer)<0.001

class TestSubSampling:

    def test_circular_masking(self):
        """Test that the circular masking works as intended"""

        # using default (center should be [2,2], radius 2)
        circ = xdem.spstats.create_circular_mask(5,5)
        circ2 = xdem.spstats.create_circular_mask(5,5,center=[2,2],radius=2)

        assert np.array_equal(circ,circ2)

        # check distance is not a multiple of pixels (more accurate subsampling)

        # will create a 1-pixel mask around the center
        circ3 = xdem.spstats.create_circular_mask(5,5,center=[1,1],radius=1)
        # will create a square mask (<1.5 pixel) around the center
        circ4 = xdem.spstats.create_circular_mask(5,5,center=[1,1],radius=1.5)

        assert not np.array_equal(circ3,circ4)


    def test_ring_masking(self):
        """Test that the ring masking works as intended"""

        # by default, the mask is only False (ring of size 0)
        ring1 = xdem.spstats.create_ring_mask(5,5)

        assert np.array_equal(ring1,np.zeros((5,5)))

        # test rings with different inner radius
        ring2 = xdem.spstats.create_ring_mask(5,5,in_radius=1,out_radius=2)
        ring3 = xdem.spstats.create_ring_mask(5,5,in_radius=0,out_radius=2)
        ring4 = xdem.spstats.create_ring_mask(5,5,in_radius=1.5,out_radius=2)

        assert np.logical_and(~np.array_equal(ring2,ring3),~np.array_equal(ring3,ring4))

        eq_ring3 = xdem.spstats.create_circular_mask(5,5)
        eq_ring3[2,2] = 0
        assert np.array_equal(ring3,eq_ring3)


class TestPatchesMethod:

    def test_patches_method(self):

        diff, mask = load_diff()

        # check the patches method runs
        df_patches = xdem.spstats.patches_method(diff.data,mask=~mask.astype(bool),gsd=diff.res[0],area_size=10000)




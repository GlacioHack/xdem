"""
Functions to test the spatial statistics.

"""
from __future__ import annotations
import numpy as np
import xdem
from xdem import examples
import geoutils as gu


def load_diff() -> tuple[gu.georaster.Raster, np.ndarray] :
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

class TestEmpiricalVariogram:

    def test_empirical_variogram(self):

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

                assert np.abs(neff_circ_exact-neff_circ_exact)<0.001




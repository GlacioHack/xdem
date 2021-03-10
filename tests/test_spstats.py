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
        df_gsd = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0],bin_func='even')
        df_coords = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data.flatten(),coords=coords,bin_func='uniform')
        df_1000_bins = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0],bin_func='even',n_lags=1000)
        df_sig = xdem.spstats.sample_multirange_empirical_variogram(dh=diff.data,gsd=diff.res[0],bin_func='even',
                                                                    nsamp=1000,nrun=30,nproc=10)




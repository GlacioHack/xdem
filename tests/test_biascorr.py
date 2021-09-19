import copy
import os
import tempfile
import time
import warnings
from typing import Any

import geoutils as gu
import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem import biascorr, coreg, examples, spatial_tools, spatialstats, misc
    import xdem


def load_examples() -> tuple[gu.georaster.Raster, gu.georaster.Raster, gu.geovector.Vector]:
    """Load example files to try coregistration methods with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_raster = gu.georaster.Raster(examples.get_path("longyearbyen_ref_dem"))
        to_be_aligned_raster = gu.georaster.Raster(examples.get_path("longyearbyen_tba_dem"))
        glacier_mask = gu.geovector.Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


class TestBiasCorrClass:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        verbose=False,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_biascorr(self):

        # Create a bias correction instance
        bcorr = biascorr.BiasCorr()

        # Check the _is_affine attribute is set correctly
        assert not bcorr._is_affine

        # Check that the fit function returns an error
        with pytest.raises(NotImplementedError):
            bcorr.fit(*self.fit_params)

        # Check the bias correction instantiation works with another bias function
        bcorr = biascorr.BiasCorr(bias_func=xdem.fit.robust_sumsin_fit)

    def test_biascorr1d(self):

        # Create a 1D bias correction
        bcorr1d = biascorr.BiasCorr1D()

        # Try to run the correction using the elevation as external variable
        elev_fit_params = self.fit_params.copy()
        elev_fit_params.update({'bias_var': self.ref.data})
        bcorr1d.fit(**elev_fit_params)
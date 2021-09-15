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

    def test_biascorr1d(self):



        # Fit the vertical shift model to the data
        vshiftcorr.fit(**self.fit_params)

        # Check that a vertical shift was found.
        assert vshiftcorr._meta.get("vshift") is not None
        assert vshiftcorr._meta["vshift"] != 0.0

        # Copy the vertical shift to see if it changes in the test (it shouldn't)
        vshift = copy.copy(vshiftcorr._meta["vshift"])

        # Check that the to_matrix function works as it should
        matrix = vshiftcorr.to_matrix()
        assert matrix[2, 3] == vshift, matrix

        # Check that the first z coordinate is now the vertical shift
        assert vshiftcorr.apply_pts(self.points)[0, 2] == vshiftcorr._meta["vshift"]

        # Apply the model to correct the DEM
        tba_unbiased = vshiftcorr.apply(self.tba.data, self.ref.transform)

        # Create a new vertical shift correction model
        vshiftcorr2 = coreg.VerticalShift()
        # Check that this is indeed a new object
        assert vshiftcorr is not vshiftcorr2
        # Fit the corrected DEM to see if the vertical shift will be close to or at zero
        vshiftcorr2.fit(reference_dem=self.ref.data, dem_to_be_aligned=tba_unbiased, transform=self.ref.transform,
                        inlier_mask=self.inlier_mask)
        # Test the vertical shift
        assert abs(vshiftcorr2._meta.get("vshift")) < 0.01

        # Check that the original model's vertical shift has not changed (that the _meta dicts are two different objects)
        assert vshiftcorr._meta["vshift"] == vshift


import warnings

import geoutils as gu
import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem.fit import robust_polynomial_fit, robust_sumsin_fit
    from xdem import biascorr, examples


def load_examples() -> tuple[gu.Raster, gu.Raster, gu.Vector]:
    """Load example files to try coregistration methods with."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference_raster = gu.Raster(examples.get_path("longyearbyen_ref_dem"))
        to_be_aligned_raster = gu.Raster(examples.get_path("longyearbyen_tba_dem"))
        glacier_mask = gu.Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_raster, to_be_aligned_raster, glacier_mask


class TestBiasCorr:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    fit_params = dict(
        reference_dem=ref.data,
        dem_to_be_aligned=tba.data,
        inlier_mask=inlier_mask,
        transform=ref.transform,
        crs=ref.crs,
        verbose=False,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_biascorr(self) -> None:
        """Test the parent class BiasCorr."""

        # Create a bias correction instance
        bcorr = biascorr.BiasCorr()

        # Check that the _is_affine attribute is set correctly
        assert not bcorr._is_affine

        # Check that the fit function returns an error
        with pytest.raises(NotImplementedError):
            bcorr.fit(*self.fit_params)

        # Check the bias correction instantiation works with another bias function
        bcorr = biascorr.BiasCorr(bias_func=robust_sumsin_fit)

    def test_biascorr1d(self):
        """Test the subclass BiasCorr1D."""

        # Create a 1D bias correction
        bcorr1d = biascorr.BiasCorr1D()

        # Try to run the correction using the elevation as external variable
        elev_fit_params = self.fit_params.copy()
        elev_fit_params.update({"bias_var": {"elevation": self.ref.data}})
        bcorr1d.fit(**elev_fit_params)

        # Apply the correction
        tba_corrected = bcorr1d.apply(dem=self.tba.data, transform=self.ref.transform, crs=self.ref.crs)

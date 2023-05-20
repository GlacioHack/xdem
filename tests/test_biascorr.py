"""Tests for the biascorr module (non-rigid coregistrations)."""
import warnings
import re

import scipy.optimize

import geoutils as gu
import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
        reference_dem=ref,
        dem_to_be_aligned=tba,
        inlier_mask=inlier_mask,
        verbose=False,
    )
    # Create some 3D coordinates with Z coordinates being 0 to try the apply_pts functions.
    points = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T

    def test_biascorr(self) -> None:
        """Test the parent class BiasCorr instantiation."""

        # Create a bias correction instance
        bcorr = biascorr.BiasCorr()

        # Check that the _is_affine attribute is set correctly
        assert not bcorr._is_affine
        assert bcorr._fit_or_bin == "fit"

        # Check the bias correction instantiation works with default arguments
        biascorr.BiasCorr()

        # Or with default bin arguments
        biascorr.BiasCorr(fit_or_bin="bin")


    def test_biascorr__errors(self) -> None:
        """Test the errors that should be raised by BiasCorr."""

        # And raises an error when "fit" or "bin" is wrongly passed
        with pytest.raises(ValueError, match="Argument `fit_or_bin` must be 'fit' or 'bin'."):
            biascorr.BiasCorr(fit_or_bin=True)  # type: ignore

        # For fit function
        with pytest.raises(TypeError, match=re.escape("Argument `fit_func` must be a function (callable) or the string '{}', "
                                            "got <class 'str'>.".format("', '".join(biascorr.fit_workflows.keys())))):
            biascorr.BiasCorr(fit_func="yay")  # type: ignore

        # For fit optimizer
        with pytest.raises(TypeError, match=re.escape("Argument `fit_optimizer` must be a function (callable), "
                                "got <class 'int'>.")):
            biascorr.BiasCorr(fit_optimizer=3)  # type: ignore

        # For bin sizes
        with pytest.raises(TypeError, match=re.escape("Argument `bin_sizes` must be an integer, or a dictionary of integers or tuples, "
                                "got <class 'dict'>.")):
            biascorr.BiasCorr(fit_or_bin="bin", bin_sizes={"a": 1.5})  # type: ignore

        # For bin statistic
        with pytest.raises(TypeError, match=re.escape("Argument `bin_statistic` must be a function (callable), "
                                "got <class 'str'>.")):
            biascorr.BiasCorr(fit_or_bin="bin", bin_statistic="count")  # type: ignore

        # For bin apply method
        with pytest.raises(TypeError, match=re.escape("Argument `bin_apply_method` must be the string 'linear' or 'per_bin', "
                                "got <class 'int'>.")):
            biascorr.BiasCorr(fit_or_bin="bin", bin_apply_method=1)  # type: ignore


    @pytest.mark.parametrize("fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, p: p[0]*np.exp(x) + p[1]))   # type: ignore
    @pytest.mark.parametrize("fit_optimizer", [scipy.optimize.curve_fit,])   # type: ignore
    def test_biascorr__fit(self, fit_func, fit_optimizer) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="fit", fit_func=fit_func, fit_optimizer=fit_optimizer)

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_dict = {"elevation": self.ref}
        elev_fit_params.update({"bias_vars": bias_dict})

        # To speed up the tests, pass niter to basinhopping through "nfreq_sumsin"
        if fit_func == "nfreq_sumsin":
            elev_fit_params.update({"niter": 1})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=100)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_dict)


    def test_biascorr1d(self):
        """Test the subclass BiasCorr1D."""

        pass

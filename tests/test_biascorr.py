"""Tests for the biascorr module (non-rigid coregistrations)."""
import warnings
import re

import scipy

import geoutils as gu
import numpy as np
import pytest

import xdem.terrain

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem import biascorr, examples
    from xdem.fit import polynomial_2d


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

        # Check default "fit" metadata was set properly
        assert bcorr._meta["fit_func"] == biascorr.fit_workflows["norder_polynomial"]["func"]
        assert bcorr._meta["fit_optimizer"] == biascorr.fit_workflows["norder_polynomial"]["optimizer"]

        # Check that the _is_affine attribute is set correctly
        assert not bcorr._is_affine
        assert bcorr._fit_or_bin == "fit"

        # Or with default bin arguments
        bcorr2 = biascorr.BiasCorr(fit_or_bin="bin")

        assert bcorr2._meta["bin_sizes"] == 10
        assert bcorr2._meta["bin_statistic"] == np.nanmedian
        assert bcorr2._meta["bin_apply_method"] == "linear"

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


    @pytest.mark.parametrize("fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: a*np.exp(x)+b))   # type: ignore
    @pytest.mark.parametrize("fit_optimizer", [scipy.optimize.curve_fit,])   # type: ignore
    def test_biascorr__fit_1d(self, fit_func, fit_optimizer) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="fit", fit_func=fit_func, fit_optimizer=fit_optimizer)

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # To speed up the tests, pass niter to basinhopping through "nfreq_sumsin"
        # Also fix random state for basinhopping
        if fit_func == "nfreq_sumsin":
            elev_fit_params.update({"niter": 1})
            elev_fit_params.update({"random_state": 42})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=100)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_func",
                             (polynomial_2d, lambda x, a, b, c, d: a * np.exp(x[0]) + x[1]*b + c/d))  # type: ignore
    @pytest.mark.parametrize("fit_optimizer", [scipy.optimize.curve_fit, ])  # type: ignore
    def test_biascorr__fit_2d(self, fit_func, fit_optimizer) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="fit", fit_func=fit_func, fit_optimizer=fit_optimizer)

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        # Passing p0 defines the number of parameters to solve for
        bcorr.fit(**elev_fit_params, subsample=100, p0=[0, 0, 0, 0])

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("bin_sizes",
                             (10,))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_1d(self, bin_sizes, bin_statistic) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=bin_sizes, bin_statistic=bin_statistic)

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=1000)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("bin_sizes",
                             (10,))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_2d(self, bin_sizes, bin_statistic) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=bin_sizes, bin_statistic=bin_statistic)

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=1000)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)


    def test_biascorr1d(self):
        """
        Test the subclass BiasCorr1D, which defines default parameters for 1D.
        The rest is already tested in test_biascorr.
        """

        # Try default "fit" parameters instantiation
        bcorr1d = biascorr.BiasCorr1D()

        assert bcorr1d._meta["fit_func"] == biascorr.fit_workflows["norder_polynomial"]["func"]
        assert bcorr1d._meta["fit_optimizer"] == biascorr.fit_workflows["norder_polynomial"]["optimizer"]

        # Try default "bin" parameter instantiation
        bcorr1d = biascorr.BiasCorr1D(fit_or_bin="bin")

        assert bcorr1d._meta["bin_sizes"] == 10
        assert bcorr1d._meta["bin_statistic"] == np.nanmedian
        assert bcorr1d._meta["bin_apply_method"] == "linear"

        elev_fit_params = self.fit_params.copy()
        # Raise error when wrong number of parameters are passed
        with pytest.raises(ValueError, match="A single variable has to be provided through the argument 'bias_vars', "
                                             "got 2."):
            bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
            bcorr1d.fit(**elev_fit_params, bias_vars=bias_vars_dict)


    def test_biascorr2d(self):
        """
        Test the subclass BiasCorr2D, which defines default parameters for 2D.
        The rest is already tested in test_biascorr.
        """

        # Try default "fit" parameters instantiation
        bcorr1d = biascorr.BiasCorr2D()

        assert bcorr1d._meta["fit_func"] == polynomial_2d
        assert bcorr1d._meta["fit_optimizer"] == scipy.optimize.curve_fit

        # Try default "bin" parameter instantiation
        bcorr1d = biascorr.BiasCorr2D(fit_or_bin="bin")

        assert bcorr1d._meta["bin_sizes"] == 10
        assert bcorr1d._meta["bin_statistic"] == np.nanmedian
        assert bcorr1d._meta["bin_apply_method"] == "linear"

        elev_fit_params = self.fit_params.copy()
        # Raise error when wrong number of parameters are passed
        with pytest.raises(ValueError, match="Exactly two variables have to be provided through the argument "
                                             "'bias_vars', got 1."):
            bias_vars_dict = {"elevation": self.ref}
            bcorr1d.fit(**elev_fit_params, bias_vars=bias_vars_dict)

    def test_directionalbias(self):
        """Test the subclass DirectionalBias."""

        # Try default "fit" parameters instantiation
        dirbias = biascorr.DirectionalBias(angle=45)

        assert dirbias._meta["fit_func"] == biascorr.fit_workflows["nfreq_sumsin"]["func"]
        assert dirbias._meta["fit_optimizer"] == biascorr.fit_workflows["nfreq_sumsin"]["optimizer"]
        assert dirbias._meta["angle"] == 45

    def test_directionalbias__synthetic(self):
        """Test the subclass DirectionalBias."""

        # Try default "fit" parameters instantiation
        dirbias = biascorr.DirectionalBias(angle=45)

        assert dirbias._meta["fit_func"] == biascorr.fit_workflows["nfreq_sumsin"]["func"]
        assert dirbias._meta["fit_optimizer"] == biascorr.fit_workflows["nfreq_sumsin"]["optimizer"]
        assert dirbias._meta["angle"] == 45

    def test_deramp(self):
        """Test the subclass Deramp."""

        # Try default "fit" parameters instantiation
        deramp = biascorr.Deramp()

        assert deramp._meta["fit_func"] == polynomial_2d
        assert deramp._meta["fit_optimizer"] == scipy.optimize.curve_fit
        assert deramp._meta["poly_order"] == 2

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5])  # type: ignore
    def test_deramp__synthetic(self, order: int):
        """Run the deramp for varying polynomial orders using a synthetic elevation difference."""

        # Get coordinates
        xx, yy = np.meshgrid(np.arange(0, self.ref.shape[1]), np.arange(0, self.ref.shape[0]))

        # Number of parameters for a 2D order N polynomial called through np.polyval2d
        nb_params = int((order + 1) * (order + 1))

        # Get a random number of parameters
        np.random.seed(42)
        params = np.random.normal(size=nb_params)

        # Create a synthetic bias and add to the DEM
        synthetic_bias = polynomial_2d((xx, yy), *params)
        bias_dem = self.ref - synthetic_bias

        # Fit
        deramp = biascorr.Deramp(poly_order=order)
        deramp.fit(reference_dem=self.ref, dem_to_be_aligned=bias_dem, subsample=10000, random_state=42)

        # Check high-order parameters are the same
        fit_params = deramp._meta["fit_params"]
        assert np.shape(fit_params) == np.shape(params)
        assert np.allclose(params.reshape(order + 1, order + 1)[-1:, -1:],
                           fit_params.reshape(order + 1, order + 1)[-1:, -1:], rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        corrected_dem = deramp.apply(bias_dem)
        assert np.nanvar(corrected_dem + bias_dem) < 0.01 * np.nanvar(synthetic_bias)


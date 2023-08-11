"""Tests for the biascorr module (non-rigid coregistrations)."""
from __future__ import annotations

import re
import warnings

import geoutils as gu
import numpy as np
import pytest
import scipy

import xdem.terrain

PLOT = False

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from xdem import examples
    from xdem.coreg import biascorr
    from xdem.fit import polynomial_2d, sumsin_1d


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

        assert bcorr2._fit_or_bin == "bin"

        # Or with default bin_and_fit arguments
        bcorr3 = biascorr.BiasCorr(fit_or_bin="bin_and_fit")

        assert bcorr3._meta["bin_sizes"] == 10
        assert bcorr3._meta["bin_statistic"] == np.nanmedian
        assert bcorr3._meta["fit_func"] == biascorr.fit_workflows["norder_polynomial"]["func"]
        assert bcorr3._meta["fit_optimizer"] == biascorr.fit_workflows["norder_polynomial"]["optimizer"]

        assert bcorr3._fit_or_bin == "bin_and_fit"

    def test_biascorr__errors(self) -> None:
        """Test the errors that should be raised by BiasCorr."""

        # And raises an error when "fit" or "bin" is wrongly passed
        with pytest.raises(ValueError, match="Argument `fit_or_bin` must be 'bin_and_fit', 'fit' or 'bin'."):
            biascorr.BiasCorr(fit_or_bin=True)  # type: ignore

        # For fit function
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument `fit_func` must be a function (callable) or the string '{}', "
                "got <class 'str'>.".format("', '".join(biascorr.fit_workflows.keys()))
            ),
        ):
            biascorr.BiasCorr(fit_func="yay")  # type: ignore

        # For fit optimizer
        with pytest.raises(
            TypeError, match=re.escape("Argument `fit_optimizer` must be a function (callable), " "got <class 'int'>.")
        ):
            biascorr.BiasCorr(fit_optimizer=3)  # type: ignore

        # For bin sizes
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument `bin_sizes` must be an integer, or a dictionary of integers or iterables, "
                "got <class 'dict'>."
            ),
        ):
            biascorr.BiasCorr(fit_or_bin="bin", bin_sizes={"a": 1.5})  # type: ignore

        # For bin statistic
        with pytest.raises(
            TypeError, match=re.escape("Argument `bin_statistic` must be a function (callable), " "got <class 'str'>.")
        ):
            biascorr.BiasCorr(fit_or_bin="bin", bin_statistic="count")  # type: ignore

        # For bin apply method
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Argument `bin_apply_method` must be the string 'linear' or 'per_bin', " "got <class 'int'>."
            ),
        ):
            biascorr.BiasCorr(fit_or_bin="bin", bin_apply_method=1)  # type: ignore

    @pytest.mark.parametrize(
        "fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
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

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=100, random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize(
        "fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c**d)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
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
        bcorr.fit(**elev_fit_params, subsample=100, p0=[0, 0, 0, 0], random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": 20}, {"elevation": (0, 500, 1000)}))  # type: ignore
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
        bcorr.fit(**elev_fit_params, subsample=1000, random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": (0, 500, 1000), "slope": (0, 20, 40)}))  # type: ignore
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
        bcorr.fit(**elev_fit_params, subsample=10000, random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize(
        "fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": np.arange(0, 1000, 100)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_and_fit_1d(self, fit_func, fit_optimizer, bin_sizes, bin_statistic) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the bin_and_fit case (called by all subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(
            fit_or_bin="bin_and_fit",
            fit_func=fit_func,
            fit_optimizer=fit_optimizer,
            bin_sizes=bin_sizes,
            bin_statistic=bin_statistic,
        )

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # To speed up the tests, pass niter to basinhopping through "nfreq_sumsin"
        # Also fix random state for basinhopping
        if fit_func == "nfreq_sumsin":
            elev_fit_params.update({"niter": 1})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_params, subsample=100, random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize(
        "fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c**d)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": (0, 500, 1000), "slope": (0, 20, 40)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_and_fit_2d(self, fit_func, fit_optimizer, bin_sizes, bin_statistic) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the bin_and_fit case (called by all subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(
            fit_or_bin="bin_and_fit",
            fit_func=fit_func,
            fit_optimizer=fit_optimizer,
            bin_sizes=bin_sizes,
            bin_statistic=bin_statistic,
        )

        # Run fit using elevation as input variable
        elev_fit_params = self.fit_params.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_params.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        # Passing p0 defines the number of parameters to solve for
        bcorr.fit(**elev_fit_params, subsample=100, p0=[0, 0, 0, 0], random_state=42)

        # Apply the correction
        bcorr.apply(dem=self.tba, bias_vars=bias_vars_dict)

    def test_biascorr1d(self) -> None:
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
        with pytest.raises(
            ValueError, match="A single variable has to be provided through the argument 'bias_vars', " "got 2."
        ):
            bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
            bcorr1d.fit(**elev_fit_params, bias_vars=bias_vars_dict)

    def test_biascorr2d(self) -> None:
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
        with pytest.raises(
            ValueError, match="Exactly two variables have to be provided through the argument " "'bias_vars', got 1."
        ):
            bias_vars_dict = {"elevation": self.ref}
            bcorr1d.fit(**elev_fit_params, bias_vars=bias_vars_dict)

    def test_directionalbias(self) -> None:
        """Test the subclass DirectionalBias."""

        # Try default "fit" parameters instantiation
        dirbias = biascorr.DirectionalBias(angle=45)

        assert dirbias._fit_or_bin == "bin_and_fit"
        assert dirbias._meta["fit_func"] == biascorr.fit_workflows["nfreq_sumsin"]["func"]
        assert dirbias._meta["fit_optimizer"] == biascorr.fit_workflows["nfreq_sumsin"]["optimizer"]
        assert dirbias._meta["angle"] == 45

    @pytest.mark.parametrize("angle", [20, 90, 210])  # type: ignore
    @pytest.mark.parametrize("nb_freq", [1, 2, 3])  # type: ignore
    def test_directionalbias__synthetic(self, angle, nb_freq) -> None:
        """Test the subclass DirectionalBias with synthetic data."""

        # Get along track
        xx = gu.raster.get_xy_rotated(self.ref, along_track_angle=angle)[0]

        # Get random parameters (3 parameters needed per frequency)
        np.random.seed(42)
        params = np.array([(5, 3000, np.pi), (1, 300, 0), (0.5, 100, np.pi / 2)]).flatten()
        nb_freq = 1
        params = params[0 : 3 * nb_freq]

        # Create a synthetic bias and add to the DEM
        synthetic_bias = sumsin_1d(xx.flatten(), *params)
        bias_dem = self.ref - synthetic_bias.reshape(np.shape(self.ref.data))

        # For debugging
        if PLOT:
            synth = self.ref.copy(new_array=synthetic_bias.reshape(np.shape(self.ref.data)))
            import matplotlib.pyplot as plt

            synth.show()
            plt.show()

            dirbias = biascorr.DirectionalBias(angle=angle, fit_or_bin="bin", bin_sizes=10000)
            dirbias.fit(reference_dem=self.ref, dem_to_be_aligned=bias_dem, subsample=10000, random_state=42)
            xdem.spatialstats.plot_1d_binning(
                df=dirbias._meta["bin_dataframe"], var_name="angle", statistic_name="nanmedian", min_count=0
            )
            plt.show()

        # Try default "fit" parameters instantiation
        dirbias = biascorr.DirectionalBias(angle=angle, bin_sizes=300)
        bounds = [
            (2, 10),
            (500, 5000),
            (0, 2 * np.pi),
            (0.5, 2),
            (100, 500),
            (0, 2 * np.pi),
            (0, 0.5),
            (10, 100),
            (0, 2 * np.pi),
        ]
        dirbias.fit(
            reference_dem=self.ref,
            dem_to_be_aligned=bias_dem,
            subsample=10000,
            random_state=42,
            bounds_amp_wave_phase=bounds,
            niter=10,
        )

        # Check all parameters are the same within 10%
        fit_params = dirbias._meta["fit_params"]
        assert np.shape(fit_params) == np.shape(params)
        assert np.allclose(params, fit_params, rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        corrected_dem = dirbias.apply(bias_dem)
        assert np.nanvar(corrected_dem - self.ref) < 0.01 * np.nanvar(synthetic_bias)

    def test_deramp(self) -> None:
        """Test the subclass Deramp."""

        # Try default "fit" parameters instantiation
        deramp = biascorr.Deramp()

        assert deramp._fit_or_bin == "fit"
        assert deramp._meta["fit_func"] == polynomial_2d
        assert deramp._meta["fit_optimizer"] == scipy.optimize.curve_fit
        assert deramp._meta["poly_order"] == 2

    @pytest.mark.parametrize("order", [1, 2, 3, 4])  # type: ignore
    def test_deramp__synthetic(self, order: int) -> None:
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

        # Check high-order parameters are the same within 10%
        fit_params = deramp._meta["fit_params"]
        assert np.shape(fit_params) == np.shape(params)
        assert np.allclose(
            params.reshape(order + 1, order + 1)[-1:, -1:], fit_params.reshape(order + 1, order + 1)[-1:, -1:], rtol=0.1
        )

        # Run apply and check that 99% of the variance was corrected
        corrected_dem = deramp.apply(bias_dem)
        assert np.nanvar(corrected_dem - self.ref) < 0.01 * np.nanvar(synthetic_bias)

    def test_terrainbias(self) -> None:
        """Test the subclass TerrainBias."""

        # Try default "fit" parameters instantiation
        tb = biascorr.TerrainBias()

        assert tb._fit_or_bin == "bin"
        assert tb._meta["bin_sizes"] == 100
        assert tb._meta["bin_statistic"] == np.nanmedian
        assert tb._meta["terrain_attribute"] == "maximum_curvature"

    def test_terrainbias__synthetic(self) -> None:
        """Test the subclass TerrainBias."""

        # Get maximum curvature
        maxc = xdem.terrain.get_terrain_attribute(self.ref, attribute="maximum_curvature")

        # Create a bias depending on bins
        synthetic_bias = np.zeros(np.shape(self.ref.data))

        # For each bin, a fake bias value is set in the synthetic bias array
        bin_edges = np.array((-1, 0, 0.1, 0.5, 2, 5))
        bias_per_bin = np.array((-5, 10, -2, 25, 5))
        for i in range(len(bin_edges) - 1):
            synthetic_bias[np.logical_and(maxc.data >= bin_edges[i], maxc.data < bin_edges[i + 1])] = bias_per_bin[i]

        # Add bias to the second DEM
        bias_dem = self.ref - synthetic_bias

        # Run the binning
        tb = biascorr.TerrainBias(
            terrain_attribute="maximum_curvature",
            bin_sizes={"maximum_curvature": bin_edges},
            bin_apply_method="per_bin",
        )
        # We don't want to subsample here, otherwise it might be very hard to derive maximum curvature...
        # TODO: Add the option to get terrain attribute before subsampling in the fit subclassing logic?
        tb.fit(reference_dem=self.ref, dem_to_be_aligned=bias_dem, random_state=42)

        # Check high-order parameters are the same within 10%
        bin_df = tb._meta["bin_dataframe"]
        assert [interval.left for interval in bin_df["maximum_curvature"].values] == list(bin_edges[:-1])
        assert [interval.right for interval in bin_df["maximum_curvature"].values] == list(bin_edges[1:])
        assert np.allclose(bin_df["nanmedian"], bias_per_bin, rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        # (we override the bias_var "max_curv" with that of the ref_dem to have a 1 on 1 match with the synthetic bias,
        # otherwise it is derived from the bias_dem which gives slightly different results than with ref_dem)
        corrected_dem = tb.apply(bias_dem, bias_vars={"maximum_curvature": maxc})
        assert np.nanvar(corrected_dem - self.ref) < 0.01 * np.nanvar(synthetic_bias)

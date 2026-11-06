"""Tests for the biascorr module (non-rigid coregistrations)."""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
import rasterio as rio
import scipy
from geoutils import Raster, Vector
from geoutils.multiproc import MultiprocConfig

import xdem.terrain
from xdem import DEM, examples, open_dem
from xdem.coreg import biascorr
from xdem.fit import polynomial_2d, sumsin_1d

PLOT = False


def load_examples() -> tuple[Raster, Raster, Vector]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path_test("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

    return ref_dem, tba_dem, glacier_mask


class TestBiasCorr:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    # Check all possibilities supported by biascorr:
    # Raster-Raster
    fit_args_rst_rst = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # Convert DEMs to points with a bit of subsampling for speed-up
    tba_pts = tba.to_pointcloud(data_column_name="z", subsample=3000, random_state=42)
    ref_pts = ref.to_pointcloud(data_column_name="z", subsample=3000, random_state=42)

    # Raster-Point
    fit_args_rst_pts = dict(reference_elev=ref, to_be_aligned_elev=tba_pts, inlier_mask=inlier_mask)

    # Point-Raster
    fit_args_pts_rst = dict(reference_elev=ref_pts, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    all_fit_args = [fit_args_rst_rst, fit_args_rst_pts, fit_args_pts_rst]

    def test_biascorr(self) -> None:
        """Test the parent class BiasCorr instantiation."""

        # Create a bias correction instance
        bcorr = biascorr.BiasCorr()

        # Check default "fit" .metadata was set properly
        assert bcorr.meta["inputs"]["fitorbin"]["fit_func"] == biascorr.fit_workflows["norder_polynomial"]["func"]
        assert (
            bcorr.meta["inputs"]["fitorbin"]["fit_optimizer"]
            == biascorr.fit_workflows["norder_polynomial"]["optimizer"]
        )
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] is None

        # Check that the _is_affine attribute is set correctly
        assert not bcorr._is_affine
        assert bcorr.meta["inputs"]["fitorbin"]["fit_or_bin"] == "fit"
        assert bcorr._needs_vars is True

        # Or with default bin arguments
        bcorr2 = biascorr.BiasCorr(fit_or_bin="bin")

        assert bcorr2.meta["inputs"]["fitorbin"]["bin_sizes"] == 10
        assert bcorr2.meta["inputs"]["fitorbin"]["bin_statistic"] == np.nanmedian
        assert bcorr2.meta["inputs"]["fitorbin"]["bin_apply_method"] == "linear"

        assert bcorr2.meta["inputs"]["fitorbin"]["fit_or_bin"] == "bin"

        # Or with default bin_and_fit arguments
        bcorr3 = biascorr.BiasCorr(fit_or_bin="bin_and_fit")

        assert bcorr3.meta["inputs"]["fitorbin"]["bin_sizes"] == 10
        assert bcorr3.meta["inputs"]["fitorbin"]["bin_statistic"] == np.nanmedian
        assert bcorr3.meta["inputs"]["fitorbin"]["fit_func"] == biascorr.fit_workflows["norder_polynomial"]["func"]
        assert (
            bcorr3.meta["inputs"]["fitorbin"]["fit_optimizer"]
            == biascorr.fit_workflows["norder_polynomial"]["optimizer"]
        )

        assert bcorr3.meta["inputs"]["fitorbin"]["fit_or_bin"] == "bin_and_fit"

        # Or defining bias variable names on instantiation as iterable
        bcorr4 = biascorr.BiasCorr(bias_var_names=("slope", "ncc"))
        assert bcorr4.meta["inputs"]["fitorbin"]["bias_var_names"] == ["slope", "ncc"]

        # Same using an array
        bcorr5 = biascorr.BiasCorr(bias_var_names=np.array(["slope", "ncc"]))
        assert bcorr5.meta["inputs"]["fitorbin"]["bias_var_names"] == ["slope", "ncc"]

        # Check that a custom non-linear function records the optimizer selected by Coreg
        bcorr6 = biascorr.BiasCorr(fit_func=lambda x, a: a * x)
        assert bcorr6.meta["inputs"]["fitorbin"]["fit_optimizer"] is scipy.optimize.curve_fit
        assert "curve_fit" in bcorr6.info(as_str=True)

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
            TypeError,
            match=re.escape(
                "Argument `fit_optimizer` must be a function (callable), 'ols' or None, got <class 'int'>."
            ),
        ):
            biascorr.BiasCorr(fit_optimizer=3)  # type: ignore

        # OLS requires a design matrix supplied by a linear method such as Deramp
        with pytest.raises(ValueError, match="The 'ols' fit optimizer requires a design matrix function."):
            bcorr_ols = biascorr.BiasCorr(fit_func=lambda x, a: a * x, fit_optimizer="ols")
            bcorr_ols.fit(
                **self.fit_args_rst_rst,
                bias_vars={"elevation": self.ref},
                subsample=100,
                random_state=42,
            )

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

        # When wrong number of parameters are passed

        # Copy fit parameters
        fit_args = self.fit_args_rst_rst.copy()
        with pytest.raises(
            ValueError,
            match=re.escape("A number of 1 variable(s) has to be provided through the argument 'bias_vars', " "got 2."),
        ):
            bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
            bcorr1d = biascorr.BiasCorr(bias_var_names=["elevation"])
            bcorr1d.fit(**fit_args, bias_vars=bias_vars_dict)

        with pytest.raises(
            ValueError,
            match=re.escape("A number of 2 variable(s) has to be provided through the argument " "'bias_vars', got 1."),
        ):
            bias_vars_dict = {"elevation": self.ref}
            bcorr2d = biascorr.BiasCorr(bias_var_names=["elevation", "slope"])
            bcorr2d.fit(**fit_args, bias_vars=bias_vars_dict)

        # When variables don't match
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The keys of `bias_vars` do not match the `bias_var_names` defined during " "instantiation: ['ncc']."
            ),
        ):
            bcorr1d2 = biascorr.BiasCorr(bias_var_names=["ncc"])
            bias_vars_dict = {"elevation": self.ref}
            bcorr1d2.fit(**fit_args, bias_vars=bias_vars_dict)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "The keys of `bias_vars` do not match the `bias_var_names` defined during "
                "instantiation: ['elevation', 'ncc']."
            ),
        ):
            bcorr2d2 = biascorr.BiasCorr(bias_var_names=["elevation", "ncc"])
            bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
            bcorr2d2.fit(**fit_args, bias_vars=bias_vars_dict)

    @pytest.mark.skipif(sys.platform != "linux", reason="Basinhopping from DirectionalBias fails on Mac")
    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b))
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )
    def test_biascorr__fit_1d(self, fit_args: Any, fit_func: Any, fit_optimizer: Any, capsys: Any) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="fit", fit_func=fit_func, fit_optimizer=fit_optimizer)

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # To speed up the tests, pass niter to basinhopping through "nfreq_sumsin"
        # Also fix random state for basinhopping
        if fit_func == "nfreq_sumsin":
            elev_fit_args.update({"niter": 1})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args, subsample=100, random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", [fit_args_rst_pts, fit_args_rst_rst])
    @pytest.mark.parametrize("fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c / x[0] + d))
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )
    def test_biascorr__fit_2d(self, fit_args: Any, fit_func: Any, fit_optimizer: Any) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="fit", fit_func=fit_func, fit_optimizer=fit_optimizer)

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        # Passing p0 defines the number of parameters to solve for
        bcorr.fit(**elev_fit_args, subsample=100, p0=[0, 0, 0, 0], random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation", "slope"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": 20}, {"elevation": (200, 500, 800)}))
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])
    def test_biascorr__bin_1d(self, fit_args: Any, bin_sizes: Any, bin_statistic: Any) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=bin_sizes, bin_statistic=bin_statistic)

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args, subsample=1000, random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("bin_sizes", (4, {"elevation": (200, 500, 800), "slope": (0, 20, 40)}))
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])
    def test_biascorr__bin_2d(self, fit_args: Any, bin_sizes: Any, bin_statistic: Any) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=bin_sizes, bin_statistic=bin_statistic)

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation", "slope"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.skipif(sys.platform != "linux", reason="Basinhopping from DirectionalBias fails on Mac")
    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b))
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": np.arange(200, 500, 10)}))
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])
    def test_biascorr__bin_and_fit_1d(
        self, fit_args: Any, fit_func: Any, fit_optimizer: Any, bin_sizes: Any, bin_statistic: Any
    ) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the bin_and_fit case (called by all subclasses)."""

        # Curve fit can be unhappy in certain circumstances for numerical estimation of covariance
        # We don't care for this test
        warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated*")
        # Apply the transform can create data exactly equal to the nodata
        warnings.filterwarnings("ignore", category=UserWarning, message="Unmasked values equal to the nodata value*")
        # Ignore SciKit-Learn warnings
        warnings.filterwarnings("ignore", message="Maximum number of iterations*")

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(
            fit_or_bin="bin_and_fit",
            fit_func=fit_func,
            fit_optimizer=fit_optimizer,
            bin_sizes=bin_sizes,
            bin_statistic=bin_statistic,
        )

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # To speed up the tests, pass niter to basinhopping through "nfreq_sumsin"
        # Keep one frequency here; model order selection is covered in test_fit
        if fit_func == "nfreq_sumsin":
            elev_fit_args.update({"niter": 1, "max_nb_frequency": 1})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args, subsample=1000, random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c / x[0] + d))
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": (200, 500, 800), "slope": (0, 20, 40)}))
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])
    def test_biascorr__bin_and_fit_2d(
        self, fit_args: Any, fit_func: Any, fit_optimizer: Any, bin_sizes: Any, bin_statistic: Any
    ) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the bin_and_fit case (called by all subclasses)."""

        # Curve fit can be unhappy in certain circumstances for numerical estimation of covariance
        # We don't care for this test
        warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated*")

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(
            fit_or_bin="bin_and_fit",
            fit_func=fit_func,
            fit_optimizer=fit_optimizer,
            bin_sizes=bin_sizes,
            bin_statistic=bin_statistic,
        )

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        # Passing p0 defines the number of parameters to solve for
        bcorr.fit(**elev_fit_args, subsample=1000, p0=[0, 0, 0, 0], random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation", "slope"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    def test_directionalbias(self) -> None:
        """Test the subclass DirectionalBias."""

        # Try default "fit" parameters instantiation
        dirbias = biascorr.DirectionalBias(angle=45)

        assert dirbias.meta["inputs"]["fitorbin"]["fit_or_bin"] == "bin_and_fit"
        assert dirbias.meta["inputs"]["fitorbin"]["fit_func"] == biascorr.fit_workflows["nfreq_sumsin"]["func"]
        assert (
            dirbias.meta["inputs"]["fitorbin"]["fit_optimizer"] == biascorr.fit_workflows["nfreq_sumsin"]["optimizer"]
        )
        assert dirbias.meta["inputs"]["specific"]["angle"] == 45
        assert dirbias._needs_vars is False

        # Check that variable names are defined during instantiation
        assert dirbias.meta["inputs"]["fitorbin"]["bias_var_names"] == ["angle"]

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("angle", [20])
    @pytest.mark.parametrize("nb_freq", [3])
    def test_directionalbias__synthetic(self, fit_args: Any, angle: float, nb_freq: int) -> None:
        """Test the subclass DirectionalBias with synthetic data."""

        # Get along track
        xx = gu.raster.get_xy_rotated(self.ref, along_track_angle=angle)[0]

        # Get random parameters (3 parameters needed per frequency)
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

            synth.plot()
            plt.show()

            dirbias = biascorr.DirectionalBias(angle=angle, fit_or_bin="bin", bin_sizes=10000)
            dirbias.fit(reference_elev=self.ref, to_be_aligned_elev=bias_dem, subsample=10000, random_state=42)
            gu.stats.plot_grouped_stats(
                table=dirbias.meta["outputs"]["fitorbin"]["bin_dataframe"],
                value="bias",
                statistic="nanmedian",
                min_count=0,
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
        elev_fit_args = fit_args.copy()
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            # Need a higher sample size to get the coefficients right here
            bias_elev = bias_dem.to_pointcloud(data_column_name="z").ds
        else:
            bias_elev = bias_dem
        dirbias.fit(
            elev_fit_args["reference_elev"],
            to_be_aligned_elev=bias_elev,
            bounds_amp_wave_phase=bounds,
            niter=2,
        )

        # Check all fit parameters are the same within 10%
        fit_params = dirbias.meta["outputs"]["fitorbin"]["fit_params"]
        assert np.shape(fit_params) == np.shape(params)
        assert np.allclose(params, fit_params, rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        corrected_dem = dirbias.apply(bias_dem)
        # Need to standardize by the synthetic bias spread to avoid huge/small values close to infinity
        assert np.nanvar((corrected_dem - self.ref) / np.nanstd(synthetic_bias)) < 0.01

    def test_deramp(self) -> None:
        """Test the subclass Deramp."""

        # Try default "fit" parameters instantiation
        deramp = biascorr.Deramp()

        assert deramp.meta["inputs"]["fitorbin"]["fit_or_bin"] == "fit"
        assert deramp.meta["inputs"]["fitorbin"]["fit_func"] == polynomial_2d
        assert deramp.meta["inputs"]["fitorbin"]["fit_optimizer"] == "ols"
        assert "ols" in deramp.info(as_str=True)
        assert deramp.meta["inputs"]["specific"]["poly_order"] == 2
        assert deramp._needs_vars is False

        # Check that variable names are defined during instantiation
        assert deramp.meta["inputs"]["fitorbin"]["bias_var_names"] == ["xx", "yy"]

        # A custom model keeps the non-linear default because the polynomial design matrix no longer applies
        deramp_custom = biascorr.Deramp(fit_func=lambda xy, *params: polynomial_2d(xy, *params))
        assert deramp_custom.meta["inputs"]["fitorbin"]["fit_optimizer"] is scipy.optimize.curve_fit

    @pytest.mark.parametrize("fit_args", all_fit_args)
    def test_deramp__fit_optimizer_override(self, fit_args: Any) -> None:
        """Checks that a user optimizer overrides OLS and produces a similar polynomial fit."""
        fit_args_rr = fit_args.copy()

        # 1/ Fit the polynomial with the default linear optimizer
        deramp_ols = biascorr.Deramp(poly_order=2)
        assert deramp_ols.meta["inputs"]["fitorbin"]["fit_optimizer"] == "ols"
        deramp_ols.fit(**fit_args_rr, subsample=2000, random_state=42)

        # 2/ Fit the same polynomial with an explicit curve_fit optimizer
        deramp_cf = biascorr.Deramp(poly_order=2, fit_optimizer=scipy.optimize.curve_fit)
        assert deramp_cf.meta["inputs"]["fitorbin"]["fit_optimizer"] is scipy.optimize.curve_fit
        deramp_cf.fit(**fit_args_rr, subsample=2000, random_state=42)

        # 3/ Check that both solvers return finite, numerically close coefficients
        params_ols = deramp_ols.meta["outputs"]["fitorbin"]["fit_params"]
        params_cf = deramp_cf.meta["outputs"]["fitorbin"]["fit_params"]
        assert np.all(np.isfinite(params_ols))
        assert np.all(np.isfinite(params_cf))
        assert np.allclose(params_ols, params_cf, atol=0.5)

    @pytest.mark.parametrize("fit_args", all_fit_args)
    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_deramp__synthetic(self, fit_args: Any, order: int) -> None:
        """Run the deramp for varying polynomial orders using a synthetic elevation difference."""

        # Get coordinates
        xx, yy = np.meshgrid(np.arange(0, self.ref.shape[1]), np.arange(0, self.ref.shape[0]))

        # Number of parameters for a 2D order N polynomial called through np.polyval2d
        nb_params = int((order + 1) * (order + 1))

        # Get a random number of parameters
        rng = np.random.default_rng(42)
        params = rng.normal(size=nb_params)

        # Create a synthetic bias and add to the DEM
        synthetic_bias = polynomial_2d((xx, yy), *params)
        bias_dem = self.ref - synthetic_bias

        # Fit
        deramp = biascorr.Deramp(poly_order=order)
        elev_fit_args = fit_args.copy()
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            bias_elev = bias_dem.to_pointcloud(data_column_name="z").ds
        else:
            bias_elev = bias_dem
        deramp.fit(elev_fit_args["reference_elev"], to_be_aligned_elev=bias_elev)

        # Check high-order fit parameters are the same within 10%
        fit_params = deramp.meta["outputs"]["fitorbin"]["fit_params"]
        assert np.shape(fit_params) == np.shape(params)
        assert np.allclose(
            params.reshape(order + 1, order + 1)[-1:, -1:], fit_params.reshape(order + 1, order + 1)[-1:, -1:], rtol=0.1
        )

        # Run apply and check that 99% of the variance was corrected
        corrected_dem = deramp.apply(bias_dem)
        # Need to standardize by the synthetic bias spread to avoid huge/small values close to infinity
        assert np.nanvar((corrected_dem - self.ref) / np.nanstd(synthetic_bias)) < 0.01

    def test_terrainbias(self) -> None:
        """Test the subclass TerrainBias."""

        # Try default "fit" parameters instantiation
        tb = biascorr.TerrainBias()

        assert tb.meta["inputs"]["fitorbin"]["fit_or_bin"] == "bin"
        assert tb.meta["inputs"]["fitorbin"]["bin_sizes"] == 100
        assert tb.meta["inputs"]["fitorbin"]["bin_statistic"] == np.nanmedian
        assert tb.meta["inputs"]["specific"]["terrain_attribute"] == "max_curvature"
        assert tb._needs_vars is False

        assert tb.meta["inputs"]["fitorbin"]["bias_var_names"] == ["max_curvature"]

    @pytest.mark.parametrize("fit_args", all_fit_args)
    def test_terrainbias__synthetic(self, fit_args: Any) -> None:
        """Test the subclass TerrainBias."""

        # Get maximum curvature
        maxc = xdem.terrain.get_terrain_attribute(self.ref, attribute="max_curvature", engine="scipy")

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
            terrain_attribute="max_curvature",
            bin_sizes={"max_curvature": bin_edges},
            bin_apply_method="per_bin",
        )
        elev_fit_args = fit_args.copy()
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            bias_elev = bias_dem.to_pointcloud(data_column_name="z").ds
        else:
            bias_elev = bias_dem
        tb.fit(
            elev_fit_args["reference_elev"],
            to_be_aligned_elev=bias_elev,
            bias_vars={"max_curvature": maxc},
        )

        # Check high-order parameters are the same within 10%
        bin_df = tb.meta["outputs"]["fitorbin"]["bin_dataframe"]
        assert [interval.left for interval in bin_df.index.get_level_values("max_curvature")] == pytest.approx(
            list(bin_edges[:-1])
        )
        assert [interval.right for interval in bin_df.index.get_level_values("max_curvature")] == pytest.approx(
            list(bin_edges[1:])
        )
        # assert np.allclose(bin_df["nanmedian"], bias_per_bin, rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        # (we override the bias_var "max_curv" with that of the ref_dem to have a 1 on 1 match with the synthetic bias,
        # otherwise it is derived from the bias_dem which gives slightly different results than with ref_dem)
        corrected_dem = tb.apply(bias_dem, bias_vars={"max_curvature": maxc})
        # Need to standardize by the synthetic bias spread to avoid huge/small values close to infinity
        assert np.nanvar((corrected_dem - self.ref) / np.nanstd(synthetic_bias)) < 0.01

    @pytest.mark.parametrize("rotation", [0, 25])
    @pytest.mark.parametrize("kind", ["deramp", "directional"])
    def test_fit__coordinate_predictors(self, kind: str, rotation: float) -> None:
        """Checks that sampled coordinate predictors match pixel indexes or GeoUtils along-track distances."""

        # Create a small raster with a grid, rotated or not
        values = np.arange(80.0, dtype=float).reshape(8, 10)
        transform = rio.Affine.translation(0, 100) * rio.Affine.rotation(rotation) * rio.Affine.scale(2, -3)
        reference = Raster.from_array(values, transform, 32632, area_or_point="Area")
        estimator: biascorr.BiasCorr
        if kind == "deramp":
            estimator = biascorr.Deramp()
            rows, columns = np.indices(reference.shape)
            expected = {"xx": columns, "yy": rows}
        else:
            estimator = biascorr.DirectionalBias(angle=-37)
            expected = {"angle": gu.raster.get_xy_rotated(reference, -37)[0]}

        # Inspect predictors passed to the existing binning and fitting method
        with patch.object(estimator, "_bin_or_and_fit_nd") as fitted:
            estimator.fit(reference, reference + 2, random_state=42)

        # Check equal coordinate calculation
        actual = fitted.call_args.kwargs["bias_vars"]
        assert set(actual) == set(expected)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name].ravel(), rtol=0, atol=1e-12)


@pytest.fixture
def bias_dem_paths(tmp_path: Path, rotation: float, estimator: biascorr.BiasCorr) -> tuple[Path, ...]:
    """Write curved DEMs and a masked predictor for bias correction comparisons."""

    # Create curved terrain with NaNs and one empty tile
    rows, columns = np.indices((20, 25))
    values = (100 + 5 * np.sin(columns / 4) + 3 * np.cos(rows / 5)).astype("float32")
    values[:7, :9] = np.nan
    values[12, 15] = np.nan
    # Slope requires square pixels
    pixel_height = 2 if isinstance(estimator, biascorr.TerrainBias) else 3
    raster = xdem.DEM.from_array(
        values,
        rio.Affine(2, 0, 500000, 0, -pixel_height, 7000000) * rio.Affine.rotation(rotation),
        32632,
        nodata=-9999,
        area_or_point="Point",
    )
    reference = (raster + 0.02 * columns + 0.03 * rows).translate(-0.3, 0.2)
    predictor = raster.copy(new_array=(columns + rows / 3).astype("float32"))
    predictor.data[11, 14] = np.ma.masked

    paths = tuple(tmp_path / f"input-{index}.tif" for index in range(3))
    for elevation, path in zip((reference, raster, predictor), paths):
        elevation.to_file(path)
    return paths


class TestBiasCorrChunked:
    """Test module for bias correction with Dask and multiprocessing."""

    @pytest.mark.parametrize(
        "estimator, backend, rotation",
        [
            pytest.param(biascorr.BiasCorr(fit_func=xdem.fit.polynomial_1d), "dask", 0, id="fit-dask"),
            pytest.param(
                biascorr.BiasCorr(fit_or_bin="bin_and_fit", fit_func=xdem.fit.polynomial_1d, bin_sizes=5),
                "mp",
                0,
                id="bin-fit-mp",
            ),
            pytest.param(biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=5), "mp", 0, id="linear-mp"),
            pytest.param(
                biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=5, bin_apply_method="per_bin"),
                "dask",
                0,
                id="per-bin-dask",
            ),
            pytest.param(biascorr.Deramp(poly_order=2), "dask", 25, id="deramp-dask-rotated"),
            pytest.param(biascorr.Deramp(poly_order=2), "mp", 0, id="deramp-mp"),
            pytest.param(
                biascorr.DirectionalBias(angle=30, fit_or_bin="bin", bin_sizes=5),
                "mp",
                25,
                id="directional-mp-rotated",
            ),
            pytest.param(
                biascorr.TerrainBias(terrain_attribute="elevation", fit_or_bin="bin", bin_sizes=5),
                "dask",
                0,
                id="elevation-dask",
            ),
            pytest.param(
                biascorr.TerrainBias(terrain_attribute="slope", fit_or_bin="bin", bin_sizes=5),
                "dask",
                0,
                id="slope-dask",
            ),
            pytest.param(
                biascorr.TerrainBias(terrain_attribute="slope", fit_or_bin="bin", bin_sizes=5),
                "mp",
                0,
                id="slope-mp",
            ),
        ],
    )
    def test_fit_and_apply__chunk_invariance(
        self,
        backend: str,
        estimator: biascorr.BiasCorr,
        rotation: float,
        bias_dem_paths: tuple[Path, ...],
        tmp_path: Path,
    ) -> None:
        """Checks that bias correction with Dask/MP match in-memory results without loading elevations or predictors."""

        from geoutils._dispatch import _get_raster_interface

        from tests.test_coreg.test_base import assert_coreg_meta_equal

        # 1/ Open inputs and prepare the bias predictor
        reference, raster, predictor = (DEM(path, load_data=True) for path in bias_dem_paths)
        bias_vars: dict[str, Any] | None = {"height": predictor} if estimator._needs_vars else None
        options: dict[str, Any] = dict(subsample=80, random_state=42)
        if bias_vars:
            options["p0"] = [0, 1]

        # 2/ Fit and apply the bias correction
        expected = estimator.copy().fit(reference, raster, bias_vars=bias_vars, **options)
        corrected = expected.apply(raster, bias_vars=bias_vars)

        if backend == "dask":
            # 2.1/ If using Dask
            pytest.importorskip("dask")
            chunks = ((6, 8), (7, 9), (6, 8))
            inputs = [
                open_dem(path, chunks={"y": rows, "x": columns}).dem
                for path, (rows, columns) in zip(bias_dem_paths, chunks)
            ]
            first, source, variable = inputs
            bias_vars = {"height": variable} if estimator._needs_vars else None
            assert all(not value.is_loaded for value in inputs)
            actual = estimator.copy().fit(first, source, bias_vars=bias_vars, **options)
            output = actual.apply(source, bias_vars=bias_vars)
            result = _get_raster_interface(output)
            assert isinstance(output, type(source._obj)) and not result.is_loaded
            values = output.compute().data
            assert not result.is_loaded and all(not value.is_loaded for value in inputs)
        else:
            # 2.2/ If using MP
            inputs = [DEM(path) for path in bias_dem_paths]
            first, source, variable = inputs
            config = MultiprocConfig(chunks=(6, 8), outfile=str(tmp_path / "output.tif"))
            bias_vars = {"height": variable} if estimator._needs_vars else None
            assert all(not value.is_loaded for value in inputs)
            actual = estimator.copy().fit(first, source, bias_vars=bias_vars, mp_config=config, **options)
            output = actual.apply(source, bias_vars=bias_vars, mp_config=config)
            result = _get_raster_interface(output)
            assert isinstance(output, DEM)
            assert not result.is_loaded
            values = result.to_nanarray()
            assert all(not value.is_loaded for value in inputs)

        # 3/ Check equality of fits, corrected values and metadata
        assert actual.meta["outputs"]["random"]["subsample_final"] == 80
        assert assert_coreg_meta_equal(actual.meta["outputs"], expected.meta["outputs"])
        np.testing.assert_array_equal(values, corrected.to_nanarray())
        assert (result.transform, result.crs, result.nodata, result.area_or_point) == (
            raster.transform,
            raster.crs,
            raster.nodata,
            raster.area_or_point,
        )

    @pytest.mark.parametrize(
        "kind, band_axis",
        [("array", False), ("array", True), ("predictor", False), ("predictor_array", False)],
    )
    def test_apply__chunk_invariance(self, kind: str, band_axis: bool) -> None:
        """Checks that array and predictor corrections match in-memory values and preserve Dask laziness."""

        pytest.importorskip("dask")

        import dask.array as da

        # 1/ Prepare a fitted correction and chunked inputs
        estimator: biascorr.BiasCorr
        source: Any
        rows, columns = np.indices((12, 15))
        values = (100 + np.sin(rows) + np.cos(columns)).astype("float32")
        raster = Raster.from_array(values, rio.Affine(2, 0, 500000, 0, -2, 7000000), 32632)
        if kind == "array":
            estimator = biascorr.Deramp(poly_order=1).fit(raster + rows + columns / 2, raster)
            # Use shorter final chunks, with or without a band axis
            if band_axis:
                values = values[np.newaxis]
            source = da.from_array(values, chunks=(5, 7) if not band_axis else (1, 5, 7))
            options = dict(transform=raster.transform, crs=raster.crs)
            expected, expected_transform = estimator.apply(values, **options)
            lazy_input = source
        else:
            estimator = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=4, bias_var_names=["height"])
            estimator.fit(raster + 2, raster, bias_vars={"height": raster})
            expected = estimator.apply(raster, bias_vars={"height": raster}).to_nanarray()
            source = raster.to_xarray()
            # Chunk a predictor with one missing value
            predictor = source.copy(deep=True)
            predictor.data[..., 1, 2] = np.nan
            predictor = predictor.chunk({"y": 3, "x": 4})
            expected[1, 2] = np.nan
            expected_transform = raster.transform
            lazy_input = predictor.data
        graph = lazy_input.__dask_graph__()

        # 2/ Apply to in-memory and Dask arrays
        if kind == "array":
            output, transform = estimator.apply(source, **options)
            assert isinstance(output, da.Array) and output.shape == source.shape
            computed = output.compute()
            expected = expected.reshape(source.shape)
        else:
            output = estimator.apply(
                source, bias_vars={"height": predictor.data if kind == "predictor_array" else predictor}
            )
            assert not output.rst.is_loaded
            computed = output.compute().data.squeeze()
            transform = output.rst.transform
            np.testing.assert_array_equal(source.data.squeeze(), values)
            assert not predictor.rst.is_loaded and not output.rst.is_loaded

        # 3/ Check equality of values, shapes and transforms
        np.testing.assert_array_equal(computed, expected)
        assert transform == expected_transform
        assert isinstance(lazy_input, da.Array) and lazy_input.__dask_graph__() is graph
        if kind == "array":
            assert isinstance(output, da.Array)

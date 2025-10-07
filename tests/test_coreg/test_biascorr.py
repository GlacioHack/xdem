"""Tests for the biascorr module (non-rigid coregistrations)."""

from __future__ import annotations

import re
import warnings

import geopandas as gpd
import geoutils as gu
import numpy as np
import pytest
import scipy

import xdem.terrain
from xdem import examples
from xdem.coreg import biascorr
from xdem.fit import polynomial_2d, sumsin_1d

PLOT = False


def load_examples() -> tuple[gu.Raster, gu.Raster, gu.Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = gu.Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = gu.Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = gu.Vector(examples.get_path("longyearbyen_glacier_outlines"))

    # Crop to smaller extents for test speed
    res = reference_dem.res
    crop_geom = (
        reference_dem.bounds.left,
        reference_dem.bounds.bottom,
        reference_dem.bounds.left + res[0] * 300,
        reference_dem.bounds.bottom + res[1] * 300,
    )
    reference_dem = reference_dem.crop(crop_geom)
    to_be_aligned_dem = to_be_aligned_dem.crop(crop_geom)

    return reference_dem, to_be_aligned_dem, glacier_mask


class TestBiasCorr:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    # Check all possibilities supported by biascorr:
    # Raster-Raster
    fit_args_rst_rst = dict(reference_elev=ref, to_be_aligned_elev=tba, inlier_mask=inlier_mask)

    # Convert DEMs to points with a bit of subsampling for speed-up
    tba_pts = tba.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
    ref_pts = ref.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds

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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize(
        "fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    def test_biascorr__fit_1d(self, fit_args, fit_func, fit_optimizer, capsys) -> None:
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

    @pytest.mark.parametrize("fit_args", [fit_args_rst_pts, fit_args_rst_rst])  # type: ignore
    @pytest.mark.parametrize(
        "fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c / x[0] + d)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    def test_biascorr__fit_2d(self, fit_args, fit_func, fit_optimizer) -> None:
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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": 20}, {"elevation": (0, 500, 1000)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_1d(self, fit_args, bin_sizes, bin_statistic) -> None:
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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": (0, 500, 1000), "slope": (0, 20, 40)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_2d(self, fit_args, bin_sizes, bin_statistic) -> None:
        """Test the _fit_func and apply_func methods of BiasCorr for the fit case (called by all its subclasses)."""

        # Create a bias correction object
        bcorr = biascorr.BiasCorr(fit_or_bin="bin", bin_sizes=bin_sizes, bin_statistic=bin_statistic)

        # Run fit using elevation as input variable
        elev_fit_args = fit_args.copy()
        bias_vars_dict = {"elevation": self.ref, "slope": xdem.terrain.slope(self.ref)}
        elev_fit_args.update({"bias_vars": bias_vars_dict})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args, subsample=10000, random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation", "slope"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize(
        "fit_func", ("norder_polynomial", "nfreq_sumsin", lambda x, a, b: x[0] * a + b)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (100, {"elevation": np.arange(0, 1000, 100)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_and_fit_1d(self, fit_args, fit_func, fit_optimizer, bin_sizes, bin_statistic) -> None:
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
        # Also fix random state for basinhopping
        if fit_func == "nfreq_sumsin":
            elev_fit_args.update({"niter": 1})

        # Run with input parameter, and using only 100 subsamples for speed
        bcorr.fit(**elev_fit_args, subsample=1000, random_state=42)

        # Check that variable names are defined during fit
        assert bcorr.meta["inputs"]["fitorbin"]["bias_var_names"] == ["elevation"]

        # Apply the correction
        bcorr.apply(elev=self.tba, bias_vars=bias_vars_dict)

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize(
        "fit_func", (polynomial_2d, lambda x, a, b, c, d: a * x[0] + b * x[1] + c / x[0] + d)
    )  # type: ignore
    @pytest.mark.parametrize(
        "fit_optimizer",
        [
            scipy.optimize.curve_fit,
        ],
    )  # type: ignore
    @pytest.mark.parametrize("bin_sizes", (10, {"elevation": (0, 500, 1000), "slope": (0, 20, 40)}))  # type: ignore
    @pytest.mark.parametrize("bin_statistic", [np.median, np.nanmean])  # type: ignore
    def test_biascorr__bin_and_fit_2d(self, fit_args, fit_func, fit_optimizer, bin_sizes, bin_statistic) -> None:
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

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("angle", [20, 90])  # type: ignore
    @pytest.mark.parametrize("nb_freq", [1, 2, 3])  # type: ignore
    def test_directionalbias__synthetic(self, fit_args, angle, nb_freq) -> None:
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
            xdem.spatialstats.plot_1d_binning(
                df=dirbias.meta["outputs"]["fitorbin"]["bin_dataframe"],
                var_name="angle",
                statistic_name="nanmedian",
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
            bias_elev = bias_dem.to_pointcloud(data_column_name="z", subsample=50000, random_state=42).ds
        else:
            bias_elev = bias_dem
        dirbias.fit(
            elev_fit_args["reference_elev"],
            to_be_aligned_elev=bias_elev,
            subsample=40000,
            random_state=42,
            bounds_amp_wave_phase=bounds,
            niter=20,
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
        assert deramp.meta["inputs"]["fitorbin"]["fit_optimizer"] == scipy.optimize.curve_fit
        assert deramp.meta["inputs"]["specific"]["poly_order"] == 2
        assert deramp._needs_vars is False

        # Check that variable names are defined during instantiation
        assert deramp.meta["inputs"]["fitorbin"]["bias_var_names"] == ["xx", "yy"]

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    @pytest.mark.parametrize("order", [1, 2, 3, 4])  # type: ignore
    def test_deramp__synthetic(self, fit_args, order: int) -> None:
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
            bias_elev = bias_dem.to_pointcloud(data_column_name="z", subsample=30000, random_state=42).ds
        else:
            bias_elev = bias_dem
        deramp.fit(elev_fit_args["reference_elev"], to_be_aligned_elev=bias_elev, subsample=20000, random_state=42)

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
        assert tb.meta["inputs"]["specific"]["terrain_attribute"] == "maximum_curvature"
        assert tb._needs_vars is False

        assert tb.meta["inputs"]["fitorbin"]["bias_var_names"] == ["maximum_curvature"]

    @pytest.mark.parametrize("fit_args", all_fit_args)  # type: ignore
    def test_terrainbias__synthetic(self, fit_args) -> None:
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
        elev_fit_args = fit_args.copy()
        if isinstance(elev_fit_args["to_be_aligned_elev"], gpd.GeoDataFrame):
            bias_elev = bias_dem.to_pointcloud(data_column_name="z", subsample=20000, random_state=42).ds
        else:
            bias_elev = bias_dem
        tb.fit(
            elev_fit_args["reference_elev"],
            to_be_aligned_elev=bias_elev,
            subsample=10000,
            random_state=42,
            bias_vars={"maximum_curvature": maxc},
        )

        # Check high-order parameters are the same within 10%
        bin_df = tb.meta["outputs"]["fitorbin"]["bin_dataframe"]
        assert [interval.left for interval in bin_df["maximum_curvature"].values] == pytest.approx(list(bin_edges[:-1]))
        assert [interval.right for interval in bin_df["maximum_curvature"].values] == pytest.approx(list(bin_edges[1:]))
        # assert np.allclose(bin_df["nanmedian"], bias_per_bin, rtol=0.1)

        # Run apply and check that 99% of the variance was corrected
        # (we override the bias_var "max_curv" with that of the ref_dem to have a 1 on 1 match with the synthetic bias,
        # otherwise it is derived from the bias_dem which gives slightly different results than with ref_dem)
        corrected_dem = tb.apply(bias_dem, bias_vars={"maximum_curvature": maxc})
        # Need to standardize by the synthetic bias spread to avoid huge/small values close to infinity
        assert np.nanvar((corrected_dem - self.ref) / np.nanstd(synthetic_bias)) < 0.01

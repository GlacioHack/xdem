"""Functions to test the spatial statistics."""
from __future__ import annotations
from typing import Tuple

import warnings
import time

import skgstat

import geoutils as gu
from geoutils import Raster, Vector
import numpy as np
import pandas as pd
import pytest

import xdem
from xdem import examples

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from skgstat import models

PLOT = False

def load_ref_and_diff() -> tuple[Raster, Raster, np.ndarray]:
    """Load example files to try coregistration methods with."""

    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    ddem = Raster(examples.get_path('longyearbyen_ddem'))
    mask = glacier_mask.create_mask(ddem)

    return reference_raster, ddem, mask

class TestVariogram:

    def test_sample_multirange_variogram_default(self):
        """Verify that the default function runs, and its basic output"""

        # Load data
        diff, mask = load_ref_and_diff()[1:3]

        # Check the variogram output is consistent for a random state
        df0 = xdem.spatialstats.sample_empirical_variogram(
            values=diff, subsample=50, random_state=42)
        assert df0.exp[0] == pytest.approx(31.72, 0.01)

        # Same check, using arguments "samples" and "runs" for historic reason which is to check if the output value
        # is the same since the beginning of the package
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff, samples=50, random_state=42, runs=2)

        # With random state, results should always be the same
        assert df.exp[0] == pytest.approx(2.38, 0.01)
        # With a single run, no error can be estimated
        assert all(np.isnan(df.err_exp.values))

        # Check that all type of coordinate inputs work
        # Only the array and the ground sampling distance
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff.data, gsd=diff.res[0], subsample=50,
            random_state=42)

        # Test multiple runs
        df2 = xdem.spatialstats.sample_empirical_variogram(
            values=diff, subsample=50, random_state=42, n_variograms=2)

        # Check that an error is estimated
        assert any(~np.isnan(df2.err_exp.values))

        # Test that running on several cores does not trigger any error
        df3 = xdem.spatialstats.sample_empirical_variogram(
            values=diff, subsample=50, random_state=42, n_variograms=2, n_jobs=2)

        # Test plotting of empirical variogram by itself
        if PLOT:
            xdem.spatialstats.plot_variogram(df2)

    @pytest.mark.parametrize('subsample_method',['pdist_point','pdist_ring','pdist_disk','cdist_point'])
    def test_sample_multirange_variogram_methods(self, subsample_method):
        """Verify that all other methods run"""

        # Load data
        diff, mask = load_ref_and_diff()[1:3]

        # Check the variogram estimation runs for several methods
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff, subsample=50, random_state=42,
            subsample_method=subsample_method)

        assert not df.empty

        # Check that the output is correct
        expected_columns = ['exp', 'lags', 'count']
        expected_dtypes = [np.float64, np.float64, np.int64]
        for col in expected_columns:
            # Check that the column exists
            assert col in df.columns
            # Check that the column has the correct dtype
            assert df[col].dtype == expected_dtypes[expected_columns.index(col)]


    def test_sample_multirange_variogram_args(self):
        """Verify that optional parameters run only for their specific method, raise warning otherwise"""

        # Load data
        diff, mask = load_ref_and_diff()[1:3]

        pdist_args = {'pdist_multi_ranges':[0, diff.res[0]*5, diff.res[0]*10]}
        cdist_args = {'ratio_subsample': 0.5, 'samples': 50, 'runs': 10}
        nonsense_args = {'thisarg': 'shouldnotexist'}

        # Check the function raises a warning for optional arguments incorrect to the method
        with pytest.warns(UserWarning):
            # An argument only use by cdist with a pdist method
            df = xdem.spatialstats.sample_empirical_variogram(
                values=diff, subsample=50, random_state=42,
                subsample_method='pdist_ring', **cdist_args)

        with pytest.warns(UserWarning):
            # Same here
            df = xdem.spatialstats.sample_empirical_variogram(
                values=diff, subsample=50, random_state=42,
                subsample_method='cdist_equidistant', runs=2, **pdist_args)

        with pytest.warns(UserWarning):
            # Should also raise a warning for a nonsense argument
            df = xdem.spatialstats.sample_empirical_variogram(
                values=diff, subsample=50, random_state=42,
                subsample_method='cdist_equidistant', runs=2, **nonsense_args)

        # Check the function passes optional arguments specific to pdist methods without warning
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff, subsample=50, random_state=42,
            subsample_method='pdist_ring', **pdist_args)

        # Check the function passes optional arguments specific to cdist methods without warning
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff, random_state=42, subsample_method='cdist_equidistant', **cdist_args)

    # N is the number of samples in an ensemble
    @pytest.mark.parametrize('subsample', [100, 1000, 10000])
    @pytest.mark.parametrize('shape', [(50, 50), (100, 100), (500, 500)])
    def test_choose_cdist_equidistant_sampling_parameters(self, subsample: int, shape: tuple[int]):
        """Verify that the automatically-derived parameters of equidistant sampling are sound"""

        # Assign an arbitrary extent
        extent = (0, 1, 0, 1)

        # Get maxdist
        maxdist = np.sqrt((extent[1] - extent[0]) ** 2 + (extent[3] - extent[2]) ** 2)
        res = np.mean([(extent[1] - extent[0]) / (shape[0] - 1), (extent[3] - extent[2]) / (shape[1] - 1)])
        # Then, we compute the radius from the center ensemble with the default value of subsample ratio in the function
        # skgstat.RasterEquidistantMetricSpace
        ratio_subsample = 0.2
        center_radius = np.sqrt(1. / ratio_subsample * subsample / np.pi) * res
        # Now, we can derive the number of successive disks that are going to be sampled in the grid
        equidistant_radii = [0.]
        increasing_rad = center_radius
        while increasing_rad < maxdist:
            equidistant_radii.append(increasing_rad)
            increasing_rad *= np.sqrt(2)
        nb_disk_samples = len(equidistant_radii)

        # ms = skgstat.RasterEquidistantMetricSpace(coords=np.ones(subsample), shape=shape, extent=extent, samples=)

        # The number of different pairwise combinations in a single ensemble (scipy.pdist function) is N*(N-1)/2
        # which is approximately N**2/2
        pdist_pairwise_combinations = subsample**2 / 2

        keyword_arguments = {'subsample':subsample , 'extent':extent, 'shape': shape, 'verbose': False}
        runs, samples = xdem.spatialstats._choose_cdist_equidistant_sampling_parameters(**keyword_arguments)
        cdist_pairwise_combinations = runs*samples**2*nb_disk_samples

        # Check the number of pairwise comparisons are the same (within 30%, due to rounding as integers)
        assert cdist_pairwise_combinations == pytest.approx(pdist_pairwise_combinations, rel=0.3)


    def test_multirange_fit_performance(self):
        """Verify that the fitting works with artificial dataset"""

        # First, generate a sum of modelled variograms: ranges and  partial sills for three models
        params_real = (100, 0.7, 1000, 0.2, 10000, 0.1)
        r1, ps1, r2, ps2, r3, ps3 = params_real

        x = np.linspace(10, 20000, 500)
        y = models.spherical(x, r=r1, c0=ps1) + models.spherical(x, r=r2, c0=ps2) \
            + models.spherical(x, r=r3, c0=ps3)

        # Add some noise on top of it
        sig = 0.025
        np.random.seed(42)
        y_noise = np.random.normal(0, sig, size=len(x))

        y_simu = y + y_noise
        sigma = np.ones(len(x))*sig

        # Put all in a dataframe
        df = pd.DataFrame()
        df = df.assign(lags=x, exp=y_simu, err_exp=sigma)

        # Run the fitting
        fun, params_est = xdem.spatialstats.fit_sum_model_variogram(['spherical', 'spherical', 'spherical'], df)

        for i in range(len(params_est)):
            # Assert all parameters were correctly estimated within a 30% relative margin
            assert params_real[2*i] == pytest.approx(params_est['range'].values[i],rel=0.3)
            assert params_real[2*i+1] == pytest.approx(params_est['psill'].values[i],rel=0.3)

        if PLOT:
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun])

    def test_check_params_variogram_model(self):
        """Verify that the checking function for the modelled variogram parameters dataframe returns adequate errors"""

        # Check when missing a column
        with pytest.raises(ValueError, match='The dataframe with variogram parameters must contain the columns "model",'
                                             ' "range" and "psill".'):
            xdem.spatialstats._check_validity_params_variogram(pd.DataFrame(data={'model':['spherical'], 'range':[100]}))

        # Check with wrong model format
        list_supported_models = ['spherical', 'gaussian', 'exponential', 'cubic', 'stable', 'matern']
        with pytest.raises(ValueError, match='Variogram model name Supraluminal not recognized. Supported models are: '+
                             ', '.join(list_supported_models)+'.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['Supraluminal'], 'range': [100], 'psill': [1]}))

        # Check with wrong range format
        with pytest.raises(ValueError, match='The variogram ranges must be float or integer.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['spherical'], 'range': ['a'], 'psill': [1]}))

        # Check with negative range
        with pytest.raises(ValueError, match='The variogram ranges must have non-zero, positive values.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['spherical'], 'range': [-1], 'psill': [1]}))

        # Check with wrong partial sill format
        with pytest.raises(ValueError, match='The variogram partial sills must be float or integer.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['spherical'], 'range': [100], 'psill': ['a']}))

        # Check with negative partial sill
        with pytest.raises(ValueError, match='The variogram partial sills must have non-zero, positive values.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['spherical'], 'range': [100], 'psill': [-1]}))

        # Check with a model that requires smoothness and without the smoothness column
        with pytest.raises(ValueError, match='The dataframe with variogram parameters must contain the column "smooth" '
                                             'for the smoothness factor when using Matern or Stable models.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['stable'], 'range': [100], 'psill': [1]}))

        # Check with wrong smoothness format
        with pytest.raises(ValueError, match='The variogram smoothness parameter must be float or integer.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['stable'], 'range': [100], 'psill': [1], 'smooth': ['a']}))

        # Check with negative smoothness
        with pytest.raises(ValueError, match='The variogram smoothness parameter must have non-zero, positive values.'):
            xdem.spatialstats._check_validity_params_variogram(
                pd.DataFrame(data={'model': ['stable'], 'range': [100], 'psill': [1], 'smooth': [-1]}))


    def test_empirical_fit_plotting(self):
        """Verify that the shape of the empirical variogram output works with the fit and plotting"""

        # Load data
        diff, mask = load_ref_and_diff()[1:3]

        # Check the variogram estimation runs for a random state
        df = xdem.spatialstats.sample_empirical_variogram(
            values=diff.data, gsd=diff.res[0], subsample=50, random_state=42)

        # Single model fit
        fun, _ = xdem.spatialstats.fit_sum_model_variogram(['spherical'], empirical_variogram=df)

        # Triple model fit
        fun2, _ = xdem.spatialstats.fit_sum_model_variogram(['spherical', 'spherical', 'spherical'], empirical_variogram=df)

        if PLOT:
            # Plot with a single model fit
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun])
            # Plot with a triple model fit
            xdem.spatialstats.plot_variogram(df, list_fit_fun=[fun2])

        # Check that errors are raised with wrong inputs
        # If the experimental variogram values "exp" are not passed
        with pytest.raises(ValueError, match='The expected variable "exp" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={'wrong_name':[1], 'lags':[1], 'count':[100]}))
        # If the spatial lags "lags" are not passed
        with pytest.raises(ValueError,
                           match='The expected variable "lags" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={'exp': [1], 'wrong_name': [1], 'count': [100]}))
        # If the pairwise sample count "count" is not passed
        with pytest.raises(ValueError,
                           match='The expected variable "count" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_variogram(pd.DataFrame(data={'exp': [1], 'lags': [1], 'wrong_name': [100]}))



class TestNeffEstimation:

    @pytest.mark.parametrize('range1', [10**i for i in range(3)])
    @pytest.mark.parametrize('psill1', [0.1, 1, 10])
    @pytest.mark.parametrize('model1', ['spherical', 'exponential', 'gaussian', 'cubic'])
    @pytest.mark.parametrize('area', [10**(2*i) for i in range(3)])
    def test_neff_circular_single_range(self, range1, psill1, model1, area):
        """ Test the accuracy of numerical integration for one to three models of spherical, gaussian or exponential
        forms to get the number of effective samples"""

        params_variogram_model = pd.DataFrame(data={'model':[model1], 'range':[range1], 'psill':[psill1]})

         # Exact integration
        neff_circ_exact = xdem.spatialstats.neff_circular_approx_theoretical(area=area, params_variogram_model=params_variogram_model)
        # Numerical integration
        neff_circ_numer = xdem.spatialstats.neff_circular_approx_numerical(area=area, params_variogram_model=params_variogram_model)

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.parametrize('range1', [10 ** i for i in range(2)])
    @pytest.mark.parametrize('range2', [10 ** i for i in range(2)])
    @pytest.mark.parametrize('range3', [10 ** i for i in range(2)])
    @pytest.mark.parametrize('model1', ['spherical', 'exponential', 'gaussian', 'cubic'])
    @pytest.mark.parametrize('model2', ['spherical', 'exponential', 'gaussian', 'cubic'])
    def test_neff_circular_three_ranges(self, range1, range2, range3, model1, model2):
        """ Test the accuracy of numerical integration for one to three models of spherical, gaussian or
        exponential forms"""

        area = 1000
        psill1 = 1
        psill2 = 1
        psill3 = 1
        model3 = 'spherical'

        params_variogram_model = pd.DataFrame(
            data={'model': [model1, model2, model3], 'range': [range1, range2, range3],
                  'psill': [psill1, psill2, psill3]})

        # Exact integration
        neff_circ_exact = xdem.spatialstats.neff_circular_approx_theoretical(area=area, params_variogram_model=params_variogram_model)
        # Numerical integration
        neff_circ_numer = xdem.spatialstats.neff_circular_approx_numerical(area=area, params_variogram_model=params_variogram_model)

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    def test_neff_exact_and_approx_hugonnet(self):
        """Test the exact and approximated calculation of the number of effective sample by double covariance sum"""

        # Generate a gridded dataset with varying errors associated to each pixel
        shape = (15, 15)
        errors = np.ones(shape)

        # Coordinates
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        xx, yy = np.meshgrid(x, y)

        # Flatten everything
        coords = np.dstack((xx.flatten(), yy.flatten())).squeeze()
        errors = errors.flatten()

        # Create a list of variogram that, summed, represent the spatial correlation
        params_variogram_model = pd.DataFrame(data={'model':['spherical', 'gaussian'], 'range':[5, 50], 'psill':[0.5, 0.5]})

        # Check that the function runs with default parameters
        t0 = time.time()
        neff_exact = xdem.spatialstats.neff_exact(coords=coords, errors=errors, params_variogram_model=params_variogram_model)
        t1 = time.time()

        # Check that the non-vectorized version gives the same result
        neff_exact_nv = xdem.spatialstats.neff_exact(coords=coords, errors=errors,
                                                     params_variogram_model=params_variogram_model, vectorized=False)
        t2 = time.time()
        assert neff_exact == pytest.approx(neff_exact_nv, rel=0.001)

        # Check that the vectorized version is faster (vectorized for about 250 points here)
        assert (t1 - t0) < (t2 - t1)

        # Check that the approximation function runs with default parameters, sampling 100 out of 250 samples
        t3 = time.time()
        neff_approx = xdem.spatialstats.neff_hugonnet_approx(coords=coords, errors=errors, params_variogram_model=params_variogram_model,
                                                             subsample=100, random_state=42)
        t4 = time.time()

        # Check that the non-vectorized version gives the same result, sampling 100 out of 250 samples
        neff_approx_nv = xdem.spatialstats.neff_hugonnet_approx(coords=coords, errors=errors, params_variogram_model=params_variogram_model,
                                                             subsample=100, vectorized=False, random_state=42)

        assert neff_approx == pytest.approx(neff_approx_nv, rel=0.001)

        # Check that the approximation version is faster
        assert (t4 - t3) < (t1 - t0)

        # Check that the approximation is about the same as the original estimate within 10%
        assert neff_approx == pytest.approx(neff_exact, rel=0.1)

    def test_number_effective_samples(self):
        """Test that the wrapper function for neff functions behaves correctly and that output values are robust"""

        # The function should return the same result as neff_circular_approx_numerical when using a numerical area
        area = 10000
        params_variogram_model = pd.DataFrame(data={'model':['spherical', 'gaussian'], 'range':[300, 3000], 'psill':[0.5, 0.5]})

        neff1 = xdem.spatialstats.neff_circular_approx_numerical(area=area, params_variogram_model=params_variogram_model)
        neff2 = xdem.spatialstats.number_effective_samples(area=area, params_variogram_model=params_variogram_model)

        assert neff1 == pytest.approx(neff2, rel=0.0001)

        # The function should return the same results as neff_hugonnet_approx when using a shape area
        # First, get the vector area and compute with the wrapper function
        res = 100.
        outlines = Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
        outlines_brom = Vector(outlines.ds[outlines.ds['NAME']=='Brombreen'])
        neff1 = xdem.spatialstats.number_effective_samples(area=outlines_brom, params_variogram_model=params_variogram_model,
                                                           rasterize_resolution=res, random_state=42)
        # Second, get coordinates manually and compute with the neff_approx_hugonnet function
        mask = outlines_brom.create_mask(xres=res)
        x = res * np.arange(0, mask.shape[0])
        y = res * np.arange(0, mask.shape[1])
        coords = np.array(np.meshgrid(y, x))
        coords_on_mask = coords[:, mask].T
        errors_on_mask = np.ones(len(coords_on_mask))
        neff2 = xdem.spatialstats.neff_hugonnet_approx(coords=coords_on_mask, errors=errors_on_mask,
                                                       params_variogram_model=params_variogram_model, random_state=42)
        # We can test the match between values accurately thanks to the random_state
        assert neff1 == pytest.approx(neff2, rel=0.00001)

        # Check that using a Raster as input for the resolution works
        raster = Raster(examples.get_path("longyearbyen_ref_dem"))
        neff3 = xdem.spatialstats.number_effective_samples(area=outlines_brom,
                                                           params_variogram_model=params_variogram_model,
                                                           rasterize_resolution=raster, random_state=42)
        # The value should be nearly the same within 2% (the discretization grid is different so affects a tiny bit the result)
        assert neff3 == pytest.approx(neff2, rel=0.02)

        # Check that the number of effective samples matches that of the circular approximation within 20%
        area_brom = np.sum(outlines_brom.ds.area.values)
        neff4 = xdem.spatialstats.number_effective_samples(area=area_brom, params_variogram_model=params_variogram_model)
        assert neff4 == pytest.approx(neff2, rel=0.2)
        # The circular approximation is always conservative, so should yield a smaller value
        assert neff4 < neff2

        # Check that errors are correctly raised
        with pytest.warns(UserWarning, match='Resolution for vector rasterization is not defined and thus set at 20% '
                                             'of the shortest correlation range, which might result in large memory usage.'):
            xdem.spatialstats.number_effective_samples(area=outlines_brom, params_variogram_model=params_variogram_model)
        with pytest.raises(ValueError, match='Area must be a float, integer, Vector subclass or geopandas dataframe.'):
            xdem.spatialstats.number_effective_samples(area='not supported', params_variogram_model=params_variogram_model)
        with pytest.raises(ValueError, match='The rasterize resolution must be a float, integer or Raster subclass.'):
            xdem.spatialstats.number_effective_samples(area=outlines_brom, params_variogram_model=params_variogram_model,
                                                       rasterize_resolution=(10, 10))


class TestSubSampling:

    def test_circular_masking(self):
        """Test that the circular masking works as intended"""

        # using default (center should be [2,2], radius 2)
        circ = xdem.spatialstats._create_circular_mask((5, 5))
        circ2 = xdem.spatialstats._create_circular_mask((5, 5), center=[2, 2], radius=2)

        # check default center and radius are derived properly
        assert np.array_equal(circ,circ2)

        # check mask
        # masking is not inclusive, i.e. exactly radius=2 won't include the 2nd pixel from the center, but radius>2 will
        eq_circ = np.zeros((5,5), dtype=bool)
        eq_circ[1:4,1:4]=True
        assert np.array_equal(circ,eq_circ)

        # check distance is not a multiple of pixels (more accurate subsampling)
        # will create a 1-pixel mask around the center
        circ3 = xdem.spatialstats._create_circular_mask((5, 5), center=[1, 1], radius=1)

        eq_circ3 = np.zeros((5,5), dtype=bool)
        eq_circ3[1,1] = True
        assert np.array_equal(circ3, eq_circ3)

        # will create a square mask (<1.5 pixel) around the center
        circ4 = xdem.spatialstats._create_circular_mask((5, 5), center=[1, 1], radius=1.5)
        # should not be the same as radius = 1
        assert not np.array_equal(circ3,circ4)


    def test_ring_masking(self):
        """Test that the ring masking works as intended"""
        warnings.simplefilter("error")

        # by default, the mask is only an outside circle (ring of size 0)
        ring1 = xdem.spatialstats._create_ring_mask((5, 5))
        circ1 = xdem.spatialstats._create_circular_mask((5, 5))

        assert np.array_equal(ring1,circ1)

        # test rings with different inner radius
        ring2 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=1, out_radius=2)
        ring3 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=0, out_radius=2)
        ring4 = xdem.spatialstats._create_ring_mask((5, 5), in_radius=1.5, out_radius=2)

        assert np.logical_and(~np.array_equal(ring2,ring3),~np.array_equal(ring3,ring4))

        # check default
        eq_ring2 = np.zeros((5,5), dtype=bool)
        eq_ring2[1:4,1:4] = True
        eq_ring2[2,2] = False
        assert np.array_equal(ring2,eq_ring2)


class TestPatchesMethod:

    def test_patches_method(self):

        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 10000

        # Check the patches method runs
        df = xdem.spatialstats.patches_method(
            diff.data,
            mask=~mask.astype(bool).squeeze(),
            gsd=gsd,
            area=area,
            random_state=42,
            n_patches=100
        )

        # Check we get the expected shape
        assert df.shape == (100, 4)

        # Check the sampling is always fixed for a random state
        assert df['tile'].values[0] == '31_184'
        assert df['nanmedian'].values[0] == pytest.approx(2.3, abs=0.01)

        # Check that all counts respect the default minimum percentage of 80% valid pixels
        assert all(df['count'].values > 0.8*np.max(df['count'].values))

class TestBinning:

    def test_nd_binning(self):

        ref, diff, mask = load_ref_and_diff()

        slope, aspect = xdem.coreg.calculate_slope_and_aspect(ref.data.squeeze())

        # 1D binning, by default will create 10 bins
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten()], list_var_names=['slope'])

        # Check length matches
        assert df.shape[0] == 10
        # Check bin edges match the minimum and maximum of binning variable
        assert np.nanmin(slope) == np.min(pd.IntervalIndex(df.slope).left)
        assert np.nanmax(slope) == np.max(pd.IntervalIndex(df.slope).right)

        # 1D binning with 20 bins
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten()], list_var_names=['slope'],
                                          list_var_bins=[[20]])
        # Check length matches
        assert df.shape[0] == 20

        # NMAD goes up quite a bit with slope, we can expect a 10 m measurement error difference
        assert df.nmad.values[-1] - df.nmad.values[0] > 10

        # Define function for custom stat
        def percentile_80(a):
            return np.nanpercentile(a, 80)
        # Check the function runs with custom functions
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten()], list_var_names=['slope'], statistics=[percentile_80])
        # Check that the count is added automatically by the function when not user-defined
        assert 'count' in df.columns.values

        # 2D binning
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten(), ref.data.flatten()], list_var_names=['slope', 'elevation'])

        # Dataframe should contain two 1D binning of length 10 and one 2D binning of length 100
        assert df.shape[0] == (10 + 10 + 100)

        # N-D binning
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten(), ref.data.flatten(), aspect.flatten()], list_var_names=['slope', 'elevation', 'aspect'])

        # Dataframe should contain three 1D binning of length 10 and three 2D binning of length 100 and one 2D binning of length 1000
        assert df.shape[0] == (1000 + 3 * 100 + 3 * 10)

    def test_interp_nd_binning(self):

        # Check the function works with a classic input (see example)
        df = pd.DataFrame({"var1": [1, 2, 3, 1, 2, 3, 1, 2, 3], "var2": [1, 1, 1, 2, 2, 2, 3, 3, 3],
                                "statistic": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3,3))
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=["var1", "var2"], statistic="statistic", min_count=None)

        # Check that the dimensions are rightly ordered
        assert fun((1, 3)) == df[np.logical_and(df['var1']==1, df['var2']==3)]['statistic'].values[0]
        assert fun((3, 1)) == df[np.logical_and(df['var1']==3, df['var2']==1)]['statistic'].values[0]

        # Check interpolation falls right on values for points (1, 1), (1, 2) etc...
        for i in range(3):
            for j in range(3):
                x = df['var1'][3 * i + j]
                y = df['var2'][3 * i + j]
                stat = df['statistic'][3 * i + j]
                assert fun((x, y)) == stat

        # Check bilinear interpolation inside the grid
        points_in = [(1.5, 1.5), (1.5, 2.5), (2.5, 1.5), (2.5, 2.5)]
        for point in points_in:
            # The values are 1 off from Python indexes
            x = point[0] - 1
            y = point[1] - 1
            # Get four closest points on the grid
            xlow = int(x - 0.5)
            xupp = int(x + 0.5)
            ylow = int(y - 0.5)
            yupp = int(y + 0.5)
            # Check the bilinear interpolation matches the mean value of those 4 points (equivalent as its the middle)
            assert fun((y + 1, x + 1)) == np.mean([arr[xlow, ylow], arr[xupp, ylow], arr[xupp, yupp], arr[xlow, yupp]])

        # Check bilinear extrapolation for points at 1 spacing outside from the input grid
        points_out = [(0, i) for i in np.arange(1, 4)] + [(i, 0) for i in np.arange(1, 4)] \
                     + [(4, i) for i in np.arange(1, 4)] + [(i, 4) for i in np.arange(4, 1)]
        for point in points_out:
            x = point[0] - 1
            y = point[1] - 1
            val_extra = fun((y + 1, x + 1))
            # The difference between the points extrapolated outside should be linear with the grid edges,
            # i.e. the same as the difference as the first points inside the grid along the same axis
            if point[0] == 0:
                diff_in = arr[x + 2, y] - arr[x + 1, y]
                diff_out = arr[x + 1, y] - val_extra
            elif point[0] == 4:
                diff_in = arr[x - 2, y] - arr[x - 1, y]
                diff_out = arr[x - 1, y] - val_extra
            elif point[1] == 0:
                diff_in = arr[x, y + 2] - arr[x, y + 1]
                diff_out = arr[x, y + 1] - val_extra
            # has to be y == 4
            else:
                diff_in = arr[x, y - 2] - arr[x, y - 1]
                diff_out = arr[x, y - 1] - val_extra
            assert diff_in == diff_out

        # Check that the output is rightly ordered in 3 dimensions, and works with varying dimension lengths
        vec1 = np.arange(1, 3)
        vec2 = np.arange(1, 4)
        vec3 = np.arange(1, 5)
        x, y, z = np.meshgrid(vec1, vec2, vec3)
        df = pd.DataFrame({"var1": x.flatten(), "var2": y.flatten(), "var3": z.flatten(),
                           "statistic": np.arange(len(x.flatten()))})
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=["var1", "var2", "var3"], statistic="statistic",
                                                  min_count=None)
        for i in vec1:
            for j in vec2:
                for k in vec3:
                    assert fun((i, j, k)) == \
                           df[np.logical_and.reduce((df['var1'] == i, df['var2'] == j, df['var3'] == k))]['statistic'].values[0]

        # Check if it works with nd_binning output
        ref, diff, mask = load_ref_and_diff()
        slope, aspect = xdem.coreg.calculate_slope_and_aspect(ref.data.squeeze())

        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten(), ref.data.flatten(), aspect.flatten()], list_var_names=['slope', 'elevation', 'aspect'])

        # First, in 1D
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names='slope')

        # Check a value is returned inside the grid
        assert np.isfinite(fun([15]))
        # Check the nmad increases with slope
        assert fun([20]) > fun([0])
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun([-5,50])))

        # Check when the first passed binning variable contains NaNs because of other binning variable
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names='elevation')

        # Then, in 2D
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=['slope', 'elevation'])

        # Check a value is returned inside the grid
        assert np.isfinite(fun([15, 1000]))
        # Check the nmad increases with slope
        assert fun([20, 800]) > fun([0, 800])
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50], [-500,3000]))))

        # The, in 3D, let's decrease the number of bins to get something with enough samples
        df = xdem.spatialstats.nd_binning(values=diff.data.flatten(), list_var=[slope.flatten(), ref.data.flatten(), aspect.flatten()], list_var_names=['slope', 'elevation', 'aspect'], list_var_bins=3)
        fun = xdem.spatialstats.interp_nd_binning(df, list_var_names=['slope', 'elevation', 'aspect'])

        # Check a value is returned inside the grid
        assert np.isfinite(fun([15,1000, np.pi]))
        # Check the nmad increases with slope
        assert fun([20, 500, np.pi]) > fun([0, 500, np.pi])
        # Check a value is returned outside the grid
        assert all(np.isfinite(fun(([-5, 50], [-500,3000], [-2*np.pi,4*np.pi]))))


    def test_plot_binning(self):

        # Define placeholder data
        df = pd.DataFrame({"var1": [0, 1, 2], "var2": [2, 3, 4], "statistic": [0, 0, 0]})

        # Check that the 1D plotting fails with a warning if the variable or statistic is not well-defined
        with pytest.raises(ValueError, match='The variable "var3" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_1d_binning(df, var_name='var3', statistic_name='statistic')
        with pytest.raises(ValueError, match='The statistic "stat" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_1d_binning(df, var_name='var1', statistic_name='stat')

        # Same for the 2D plotting
        with pytest.raises(ValueError, match='The variable "var3" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_2d_binning(df, var_name_1='var3', var_name_2='var1', statistic_name='statistic')
        with pytest.raises(ValueError, match='The statistic "stat" is not part of the provided dataframe column names.'):
            xdem.spatialstats.plot_2d_binning(df, var_name_1='var1', var_name_2='var1', statistic_name='stat')

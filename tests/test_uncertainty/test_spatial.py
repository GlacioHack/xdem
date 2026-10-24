"""Tests for spatial error propagation."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geopandas as gpd
import geoutils as gu
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from geoutils import Raster, Vector
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel
from shapely.geometry import box

import xdem
from xdem import examples
from xdem.uncertainty import spatial


def load_ref_and_diff() -> tuple[Raster, Raster, Any, Vector]:
    """Load example elevation differences and a glacier mask for empirical patch tests."""

    reference_raster = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    outlines = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))
    ddem = Raster(examples.get_path_test("longyearbyen_ddem"))
    mask = outlines.create_mask(ddem)

    return reference_raster, ddem, mask, outlines


class TestSpatialErrorPropagation:
    """Test module for propagating covariances over areas (boolean masks or vectors)."""

    def test_spatial_error_propagation__independent(self) -> None:
        """Checks that independent errors decrease with the square root of the observation count."""

        # Create observations with independent errors
        raster = gu.Raster.from_array(np.ones((3, 4)), Affine(10, 0, 0, 0, -10, 30), 32632)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])

        # Check sample count and uncertainty for rasters and points
        for support in [raster, raster.to_pointcloud()]:
            assert xdem.uncertainty.number_effective_samples(None, structure, support=support) == pytest.approx(12)
            assert xdem.uncertainty.spatial_error_propagation([None], structure, support=support) == pytest.approx(
                [2 / np.sqrt(12)]
            )

    def test_spatial_error_propagation__duplicate_labels(self) -> None:
        """Checks that duplicate point labels preserve weight order and area uncertainty."""

        # Create points with unequal weights and duplicate labels
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, np.ones(5), crs=32632)
        points.ds.index = ["b", "a", "b", "c", "a"]
        original_index = points.ds.index.copy()
        weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])

        # Normalize weights for the mean of independent observations
        structure = xdem.ErrorStructure([xdem.ErrorComponent("independent", 1.0)])
        normalized = weights[mask] / weights[mask].sum()
        expected = np.sqrt(np.sum(normalized**2))

        # Propagate uncertainty over the selected points
        result = xdem.uncertainty.spatial_error_propagation(
            [mask], structure, support=points, weights=weights, subsample=None
        )

        # Check uncertainty and unchanged labels
        assert result[0] == pytest.approx(expected)
        pd.testing.assert_index_equal(points.ds.index, original_index)

    def test_spatial_error_propagation__vector(self) -> None:
        """Checks that vector areas and masks give the same area uncertainty."""

        # Create a polygon covering half the grid
        raster = gu.Raster.from_array(np.ones((6, 8)), Affine(10, 0, 0, 0, -10, 60), 32632)
        vector = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 40, 60)], crs=raster.crs))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])

        # Calculate uncertainty from the selected pixel count
        mask = vector.create_mask(raster, as_array=True)
        expected = 2 / np.sqrt(np.count_nonzero(mask))

        # Compare vector inputs on rasters and points
        for support in [raster, raster.to_pointcloud()]:
            support_mask = vector.create_mask(support, as_array=True)
            expected_on_support = 2 / np.sqrt(np.count_nonzero(support_mask))
            for area in [vector, vector.ds]:
                result = xdem.uncertainty.spatial_error_propagation([area], structure, support=support, subsample=None)
                assert result == pytest.approx([expected_on_support])

        # Rasterize the polygon without a reference grid
        rasterized = xdem.uncertainty.spatial_error_propagation(
            [vector],
            structure,
            rasterize_resolution=10,
            subsample=None,
        )

        # Check uncertainty matches the original grid
        assert rasterized == pytest.approx([expected])

    @pytest.mark.parametrize("statistic", ["mean", "integral"])
    @pytest.mark.parametrize("point_input", [False, True])
    def test_propagate_uncertainty__weighted_areas(self, statistic: str, point_input: bool) -> None:
        """Checks that weighted area means and integrals have the expected estimate and uncertainty."""

        # Create an area with unequal weights
        values = np.arange(48, dtype=float).reshape(6, 8)
        raster = gu.Raster.from_array(values, Affine(2, 0, 0, 0, -2, 12), 32632)
        source = raster.to_pointcloud() if point_input else raster
        mask = np.zeros(source.data.shape, dtype=bool)
        mask.reshape(-1)[5:15] = True
        weights = np.linspace(1, 2, values.size).reshape(source.data.shape)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])

        # Propagate area uncertainty
        summary = xdem.uncertainty.propagate_uncertainty(
            source,
            "spatial",
            error_structure=structure,
            method="analytical",
            inputs={"spatial": {"areas": {"selected": mask}, "weights": weights}},
            operation_options={"statistic": statistic},
        )

        # Normalize mean weights or scale raster integrals by pixel area
        coefficients = weights[mask].copy()
        if statistic == "mean":
            coefficients /= coefficients.sum()
        elif not point_input:
            coefficients *= 4
        expected = coefficients @ source.data[mask]

        # Independent errors add in quadrature
        assert type(summary) is gu.PropagationSummary
        assert summary.estimate["selected"] == pytest.approx(expected)
        assert summary.std["selected"] == pytest.approx(0.2 * np.linalg.norm(coefficients))

    def test_propagate_uncertainty__reproject(self) -> None:
        """Checks that reprojection uncertainty matches GeoUtils."""

        # Create fine and coarse grids
        source = gu.Raster.from_array(np.arange(64, dtype=float).reshape(8, 8), Affine(1, 0, 0, 0, -1, 8), 32632)
        reference = gu.Raster.from_array(np.zeros((4, 4)), Affine(2, 0, 0, 0, -2, 8), 32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        options: dict[str, Any] = {
            "ref": reference,
            "resampling": Mean(),
            "nodata_propagation": "ignore",
        }

        # Run reprojection through xDEM and GeoUtils
        expected = gu.uncertainty.propagate(
            source.reproject,
            error_structure=errors,
            operation_kwargs=options,
            method="analytical",
        )
        actual = xdem.uncertainty.propagate_uncertainty(
            operation=source.reproject,
            error_structure=errors,
            operation_options=options,
            method="analytical",
        )

        # Check grids, estimates and uncertainties
        assert type(actual) is gu.PropagationSummary
        assert actual.std.georeferenced_grid_equal(expected.std)
        np.testing.assert_array_equal(actual.estimate.data, expected.estimate.data)
        np.testing.assert_array_equal(actual.std.data, expected.std.data)

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_spatial_error_propagation__covariance(self) -> None:
        """Checks that area uncertainty matches the weighted covariance sum across error components."""

        # 1/ Create error components
        # Create a grid with a varying predictor
        raster = gu.Raster.from_array(np.ones((3, 4)), Affine(10, 0, 0, 0, -10, 30), 32632)
        quality = np.arange(12).reshape(3, 4) / 12

        # Define increasing error magnitudes
        index = pd.Index([0.0, 1.0], name="quality")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        table = pd.DataFrame([[0.5, 100], [2, 100]], index=index, columns=columns)

        # Combine variable, constant and independent errors
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent(
                    "variable",
                    xdem.ErrorMagnitude.variable_from_grouped_stats(table),
                    VariogramModel("gaussian", 30, 1),
                ),
                xdem.ErrorComponent("fixed", 0.7, VariogramModel("spherical", 100, 1)),
                xdem.ErrorComponent("independent", 0.3),
            ]
        )

        # 2/ Calculate weighted covariance
        # Exclude one observation
        mask = np.ones(raster.shape, dtype=bool)
        mask[0, 0] = False
        coordinates = np.column_stack(raster.ij2xy(*np.where(mask)))
        covariance = structure.to_covariance_matrix(coordinates, predictors={"quality": quality[mask]})

        # Normalize area weights and sum covariance
        weights = np.arange(1, 13).reshape(raster.shape).astype(float)
        normalized = weights[mask] / weights[mask].sum()
        expected = np.sqrt(normalized @ covariance @ normalized)

        # 3/ Compare area uncertainty and effective sample size
        # Run area propagation with the same mask and weights
        result = xdem.uncertainty.spatial_error_propagation(
            [mask],
            structure,
            support=raster,
            predictors={"quality": quality},
            weights=weights,
            subsample=None,
        )

        # Check area uncertainty
        assert result == pytest.approx([expected])

        # Check effective sample size from mean magnitude and area uncertainty
        mean_sigma = np.sum(normalized * structure.predict_magnitude({"quality": quality[mask]}))
        assert xdem.uncertainty.number_effective_samples(
            mask,
            structure,
            support=raster,
            predictors={"quality": quality},
            weights=weights,
            subsample=None,
        ) == pytest.approx((mean_sigma / expected) ** 2)

    def test_propagate_uncertainty__area(self) -> None:
        """Checks that area simulations match analytical uncertainty and an equivalent callable."""

        # 1/ Run area simulations
        # Create a constant DEM and exclude two columns
        values = np.ones((6, 8))
        transform = Affine(10, 0, 0, 0, -10, 60)
        dem = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])
        mask = np.ones(dem.shape, dtype=bool)
        mask[:, :2] = False

        # Save simulated averages
        result = xdem.uncertainty.propagate_uncertainty(
            dem,
            "spatial",
            error_structure=structure,
            inputs={"spatial": {"areas": {"area": mask}}},
            n_samples=200,
            random_state=42,
            return_samples=True,
        )

        # Repeat with a callable and the same draws
        direct = xdem.uncertainty.propagate_uncertainty(
            dem,
            lambda value: np.ma.mean(value.data[mask]),
            error_structure=structure,
            n_samples=200,
            random_state=42,
        )

        # 2/ Compare results
        # Check output types
        assert isinstance(result.std, pd.Series)
        assert isinstance(result.mean, pd.Series)
        assert isinstance(result.estimate, pd.Series)
        assert isinstance(result.n_valid, pd.Series)
        assert isinstance(direct.std, float)
        assert isinstance(direct.mean, float)

        # Allow sampling variability across 200 simulations
        analytical_std = xdem.uncertainty.spatial_error_propagation([mask], structure, support=dem)[0]
        assert result.std["area"] == pytest.approx(analytical_std, rel=0.15)

        # Check the callable gives the same mean and uncertainty
        assert result.std["area"] == pytest.approx(direct.std)
        assert result.mean["area"] == pytest.approx(direct.mean)

        # Check simulation count and uncertainty from saved samples
        assert result.n_success == result.n_valid["area"] == 200
        assert result.samples is not None
        assert result.std["area"] == pytest.approx(result.samples["area"].std(ddof=1))

        # Check the estimate and unchanged elevations
        assert result.estimate["area"] == 1
        assert np.all(dem.data == 1)


class TestEffectiveSampleSize:
    """Test module for exact and approximate integration of spatial correlation models."""

    @pytest.mark.parametrize("range1", [10**i for i in range(3)])
    @pytest.mark.parametrize("psill1", [0.1, 1, 10])
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.parametrize("area", [10 ** (2 * i) for i in range(3)])
    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_circular_single_range(self, range1: float, psill1: float, model1: str, area: float) -> None:
        """Checks that radial quadrature matches the circular formula for each supported correlation model."""

        # Convert the model partial sill to an error magnitude while keeping unit correlation variance
        structure = xdem.ErrorStructure(
            [xdem.ErrorComponent("error", np.sqrt(psill1), VariogramModel(model1, range1, 1))]
        )

        # Exact integration
        neff_circ_exact = spatial.neff_circular_approx_theoretical(area=area, error_structure=structure)
        # Numerical integration
        neff_circ_numer = spatial.neff_circular_approx_numerical(area=area, error_structure=structure)

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.parametrize("range1", [10**i for i in range(2)])
    @pytest.mark.parametrize("range2", [10**i for i in range(2)])
    @pytest.mark.parametrize("range3", [10**i for i in range(2)])
    @pytest.mark.parametrize("model1", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.parametrize("model2", ["spherical", "exponential", "gaussian", "cubic"])
    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_circular_three_ranges(
        self, range1: float, range2: float, range3: float, model1: str, model2: str
    ) -> None:
        """Checks that radial quadrature and circular formulas agree for the sum of three correlation models."""

        # Use equal component variances to compare combinations of short and long correlation ranges
        area = 1000
        psill1 = 1
        psill2 = 1
        psill3 = 1
        model3 = "spherical"

        # Represent the three independent contributions as separate error components
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent(f"range{i}", np.sqrt(sill), VariogramModel(model, scale, 1))
                for i, (model, scale, sill) in enumerate(
                    zip([model1, model2, model3], [range1, range2, range3], [psill1, psill2, psill3])
                )
            ]
        )

        # Exact integration
        neff_circ_exact = spatial.neff_circular_approx_theoretical(area=area, error_structure=structure)
        # Numerical integration
        neff_circ_numer = spatial.neff_circular_approx_numerical(area=area, error_structure=structure)

        # Check results are the exact same
        assert neff_circ_exact == pytest.approx(neff_circ_numer, rel=0.001)

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_neff_exact_and_approx_hugonnet(self) -> None:
        """Checks that sampled covariance sums approximate the exact effective sample size within ten percent."""

        # Create a regular grid with the same error magnitude at every location
        shape = (15, 15)
        errors = np.ones(shape)

        # Coordinates
        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])
        xx, yy = np.meshgrid(x, y)

        # Flatten everything
        coords = np.dstack((xx.ravel(), yy.ravel())).squeeze()
        errors = errors.ravel()

        # Combine short and long range errors with equal variance contributions
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent("short", np.sqrt(0.5), VariogramModel("spherical", 5, 1)),
                xdem.ErrorComponent("long", np.sqrt(0.5), VariogramModel("gaussian", 50, 1)),
            ]
        )

        # Check that the function runs with default parameters
        neff_exact = spatial.neff_exact(coords=coords, error_structure=structure)

        # Check that the non-vectorized version gives the same result
        neff_exact_nv = spatial.neff_exact(coords=coords, error_structure=structure, vectorized=False)
        assert neff_exact == pytest.approx(neff_exact_nv, rel=0.001)

        # Check that the approximation function runs with default parameters, sampling 100 out of 225 samples
        neff_approx = spatial.neff_hugonnet_approx(
            coords=coords, error_structure=structure, subsample=100, random_state=42
        )

        # Check that the non-vectorized version gives the same result, sampling 100 out of 225 samples
        neff_approx_nv = spatial.neff_hugonnet_approx(
            coords=coords,
            error_structure=structure,
            subsample=100,
            vectorized=False,
            random_state=42,
        )

        assert neff_approx == pytest.approx(neff_approx_nv, rel=0.001)

        # Check that the approximation is about the same as the original estimate within 10%
        assert neff_approx == pytest.approx(neff_exact, rel=0.1)


class TestPatchesMethod:
    """Test module for empirical spread between stable terrain patches at specified averaging areas."""

    def test_patches_method_loop_quadrant(self) -> None:
        """Checks that quadrant sampling returns reproducible patch statistics and the requested integration area."""

        # Load elevation differences and exclude glaciers from the patch population
        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 10000

        # Check the patches method runs
        df, df_full = xdem.uncertainty.patches_method(
            diff,
            unstable_mask=mask,
            gsd=gsd,
            areas=[area],
            random_state=42,
            n_patches=7,
            vectorized=False,
            return_in_patch_statistics=True,
        )

        # First, the summary dataframe
        assert df.shape == (1, 4)
        assert all(df.columns == ["nmad", "nb_indep_patches", "exact_areas", "areas"])

        # Check the sampling is fixed for a random state
        assert df["nb_indep_patches"][0] == 7
        assert df["exact_areas"][0] == pytest.approx(df["areas"][0], rel=0.2)

        # Then, the full dataframe
        assert df_full.shape == (7, 5)

        # Check the sampling is always fixed for a random state
        assert df_full["tile"].values[0] == "7_9"

        # Check that all counts respect the default minimum percentage of 80% valid pixels
        assert all(df_full["count"].values > 0.8 * np.max(df_full["count"].values))

    def test_patches_method_convolution(self) -> None:
        """Checks that convolution produces one patch uncertainty summary per requested area."""

        # Load elevation differences and exclude glaciers from the patch population
        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 10000

        # First, the patches method runs with scipy
        df = xdem.uncertainty.patches_method(
            diff,
            unstable_mask=mask,
            gsd=gsd,
            areas=[area, area * 2],
            random_state=42,
            vectorized=True,
            convolution_method="scipy",
        )

        # Check the summary columns and allow for rasterization of the requested circular area
        assert df.shape == (2, 4)
        assert all(df.columns == ["nmad", "nb_indep_patches", "exact_areas", "areas"])
        assert df["exact_areas"][0] == pytest.approx(df["areas"][0], rel=0.2)

    def test_patches_method__array_masks(self) -> None:
        """Checks that array masks give the same patch statistics for rasters and arrays."""

        # Create stable and unstable regions
        values = np.random.default_rng(3).normal(size=(20, 20))
        stable = np.zeros(values.shape, dtype=bool)
        stable[2:-2, 2:-2] = True
        unstable = np.zeros(values.shape, dtype=bool)
        unstable[8:12, 8:12] = True
        raster = Raster.from_array(values, Affine(1, 0, 0, 0, -1, 20), 32632, nodata=-9999)

        # Run patches_method() on the raster and array
        options: dict[str, Any] = {
            "areas": [16, 36],
            "stable_mask": stable,
            "unstable_mask": unstable,
            "patch_shape": "square",
        }
        raster_result = xdem.uncertainty.patches_method(raster, **options)
        array_result = xdem.uncertainty.patches_method(values, gsd=1, **options)

        # Check equal statistics
        assert list(raster_result) == list(array_result)
        np.testing.assert_allclose(raster_result.to_numpy(), array_result.to_numpy())


class TestSpatialErrorPropagationErrors:
    """Test module for errors/warning during spatial error propagation."""

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_number_effective_samples__error_variable_magnitude(self) -> None:
        """Checks an error is raised for variable magnitudes without predictors passed."""

        # Define a variable error magnitude
        index = pd.Index([0.0, 1.0], name="quality")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        table = pd.DataFrame([[0.5, 100], [2, 100]], index=index, columns=columns)
        structure = xdem.ErrorStructure(
            [
                xdem.ErrorComponent(
                    "variable",
                    xdem.ErrorMagnitude.variable_from_grouped_stats(table),
                    VariogramModel("gaussian", 30, 1),
                )
            ]
        )

        # Check predictors are required
        with pytest.raises(ValueError, match="constant magnitudes"):
            xdem.uncertainty.number_effective_samples(100, structure)

    def test_number_effective_samples__error_independent(self) -> None:
        """Checks an error is raised for independent errors without an observation count."""

        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])
        with pytest.raises(ValueError, match="explicit observation support"):
            xdem.uncertainty.number_effective_samples(100, structure)

    def test_number_effective_samples__error_mask(self) -> None:
        """Checks an error is raised for a mask without observation locations."""

        # Create a mask and independent errors
        mask = np.ones((6, 8), dtype=bool)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])
        with pytest.raises(ValueError, match="support"):
            xdem.uncertainty.number_effective_samples(mask, structure)


class TestPatchesMethodErrors:
    """Test module for errors/warnings for patch method."""

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_patches_method__error_missing_numba(self) -> None:
        """Checks an error is raised when Numba convolution is requested without Numba."""

        # Load example differences
        diff, mask = load_ref_and_diff()[1:3]
        gsd = diff.res[0]
        area = 10000

        # Check the missing Numba error
        with pytest.raises(ImportError, match="Optional dependency 'numba' required.*"):
            xdem.uncertainty.patches_method(
                diff,
                unstable_mask=mask,
                gsd=gsd,
                areas=[area],
                random_state=42,
                n_patches=7,
                vectorized=True,
                return_in_patch_statistics=True,
                convolution_method="numba",
            )

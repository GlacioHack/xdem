"""Tests for random error fields, numerical propagation and empirical patch uncertainty."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geoutils as gu
import numpy as np
import pytest
from affine import Affine
from geoutils import Raster, Vector

import xdem
from xdem import examples


class TestRandomFields:
    """Generation of independent and spatially correlated error fields."""

    def test_random_fields_preserve_type_mask_and_random_state(self) -> None:
        """Checks that random fields preserve spatial types and masks and are reproducible for a fixed seed."""

        # Create a raster with one invalid cell and a constant independent error magnitude
        values = np.ma.masked_array(np.zeros((12, 10)), mask=False)
        values.mask[0, 0] = True
        like = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 120), 32632, nodata=-9999)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])

        # Repeat one draw and also request two independent realizations from the same generator
        first = structure.generate_random_field(like, random_state=42)
        second = structure.generate_random_field(like, random_state=42)
        multiple = structure.generate_random_field(like, n_fields=2, random_state=42)

        # Evaluate the corresponding point field and magnitude map
        point_field = structure.generate_random_field(like.to_pointcloud(), random_state=42)
        magnitude = structure.predict_magnitude(like=like)

        # Check that outputs retain the input spatial representation and invalid cell
        assert isinstance(first, gu.Raster)
        assert isinstance(point_field, gu.PointCloud)
        assert first.get_mask()[0, 0]
        assert magnitude.get_mask()[0, 0]

        # The same seed repeats a field, while successive draws produce different fields
        assert np.ma.allequal(first.data, second.data)
        assert len(multiple) == 2
        assert not np.ma.allequal(multiple[0].data, multiple[1].data)

    def test_correlated_random_field_uses_variogram_conversion(self) -> None:
        """Checks that a GeoUtils correlation model generates a finite random field on the requested grid."""

        # Require the optional simulator used for spatially correlated fields
        pytest.importorskip("gstools")

        # Define the output grid independently of the correlation range and magnitude
        like = gu.Raster.from_array(
            np.zeros((12, 10)),
            Affine(10, 0, 0, 0, -10, 120),
            32632,
            nodata=-9999,
        )

        # Use a unit variance Gaussian correlation and scale the resulting field by two
        correlation = gu.VariogramModel("gaussian", effective_range=50, partial_sill=1)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("spatial", 2, correlation)])

        # Generate the field through the public ErrorStructure method
        field = structure.generate_random_field(like, random_state=42)

        # Check the raster type, requested shape and finite simulated values
        assert isinstance(field, gu.Raster)
        assert field.shape == like.shape
        assert np.all(np.isfinite(field.data))


def load_ref_and_diff() -> tuple[Raster, Raster, Any, Vector]:
    """Load example elevation differences and a glacier mask for empirical patch tests."""

    # Load the reference grid and glacier outlines used to exclude unstable terrain
    reference_raster = Raster(examples.get_path_test("longyearbyen_ref_dem"))
    outlines = Vector(examples.get_path_test("longyearbyen_glacier_outlines"))

    # Read the elevation differences and construct the corresponding glacier mask
    ddem = Raster(examples.get_path_test("longyearbyen_ddem"))
    mask = outlines.create_mask(ddem)

    return reference_raster, ddem, mask, outlines


class TestPatchesMethod:
    """Empirical spread between stable terrain patches at specified averaging areas."""

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_patches_method_missing_numba(self) -> None:
        """Checks that requesting Numba patch convolution reports a clear error when Numba is unavailable."""

        # Use valid example data so the optional backend is the intended source of failure
        diff, mask = load_ref_and_diff()[1:3]
        gsd = diff.res[0]
        area = 10000

        # Request the Numba engine explicitly and check the dependency named by the error
        with pytest.raises(ImportError, match="Optional dependency 'numba' required.*"):

            df, df_full = xdem.uncertainty.patches_method(
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

    def test_patches_method_loop_quadrant(self) -> None:
        """Checks that quadrant sampling returns reproducible patch statistics and the requested integration area."""

        # Load elevation differences and exclude glaciers from the patch population
        diff, mask = load_ref_and_diff()[1:3]

        gsd = diff.res[0]
        area = 10000

        # Draw seven accepted patches and retain both their statistics and the area summary
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

        # Check that one requested area gives one row with uncertainty and patch counts
        assert df.shape == (1, 4)
        assert all(df.columns == ["nmad", "nb_indep_patches", "exact_areas", "areas"])

        # Allow for rounding the requested circular area to a whole number of raster cells
        assert df["nb_indep_patches"][0] == 7
        assert df["exact_areas"][0] == pytest.approx(df["areas"][0], rel=0.2)

        # Retain one row of descriptive statistics for each of the seven patches
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

        # Compute moving window means at two areas using the SciPy convolution engine
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


class TestNumericalPropagation:
    """Shared simulation summaries for spatial, terrain and callable operations."""

    def test_numerical_area_matches_analytical_and_callable(self) -> None:
        """Checks that numerical area uncertainty agrees with analytical propagation and an equivalent callable."""

        # Create a constant DEM with independent errors and exclude two columns from the average
        values = np.ones((6, 8))
        transform = Affine(10, 0, 0, 0, -10, 60)
        dem = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 2)])
        mask = np.ones(dem.shape, dtype=bool)
        mask[:, :2] = False

        # Keep the simulated averages so their sample standard deviation can be checked directly
        result = xdem.uncertainty.propagate_uncertainty(
            dem,
            "spatial",
            areas=[mask],
            error_structure=structure,
            nsim=200,
            random_state=42,
            return_samples=True,
        )

        # Repeat the same averaging operation as a callable using identical random draws
        direct = xdem.uncertainty.propagate_uncertainty(
            dem,
            lambda value: np.ma.mean(value.data[mask]),
            error_structure=structure,
            nsim=200,
            random_state=42,
        )

        # Allow sampling variability when comparing 200 realizations with the analytical standard error
        analytical_std = xdem.uncertainty.spatial_error_propagation([mask], structure, support=dem)[0]
        assert result.std[0] == pytest.approx(analytical_std, rel=0.15)

        # The named workflow and callable use the same draws and must have identical summaries
        assert result.std[0] == pytest.approx(direct.std)
        assert result.mean[0] == pytest.approx(direct.mean)

        # Check counts and verify the reported spread against the retained realization stack
        assert result.n_success == result.n_valid[0] == 200
        assert result.samples is not None
        assert result.std == pytest.approx(np.std(result.samples, axis=0, ddof=1))

        # Check the original estimate and ensure simulations leave the input elevations unchanged
        assert result.estimate[0] == 1
        assert np.all(dem.data == 1)

    def test_terrain_propagation_and_nonlinear_mean(self) -> None:
        """Checks that terrain propagation recomputes slope for each realization and retains the DEM grid."""

        # Start from flat terrain so its original slope is zero
        values = np.ones((10, 12))
        transform = Affine(10, 0, 0, 0, -10, 100)
        dem = xdem.DEM(gu.Raster.from_array(values, transform, 32632))
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 0.5)])

        # Add small elevation errors and retain the slope maps from four realizations
        result = xdem.uncertainty.propagate_uncertainty(
            dem,
            "terrain",
            attribute="slope",
            terrain_kwargs={"engine": "scipy"},
            error_structure=structure,
            nsim=4,
            random_state=3,
            return_samples=True,
        )

        # Perturbed slopes have a positive mean even though the unperturbed DEM is flat
        assert result.std.georeferenced_grid_equal(dem)
        assert np.nanmean(result.estimate.data) == pytest.approx(0)
        assert np.nanmean(result.mean.data) > 0

        # Compare the reported spread with a direct standard deviation of the retained maps
        assert result.samples is not None
        stack = np.ma.stack([sample.data for sample in result.samples])
        np.testing.assert_allclose(result.std.data, stack.std(axis=0, ddof=1), atol=1e-12)

        # Check that terrain propagation preserves the original elevations
        assert np.all(dem.data == 1)

    def test_circular_summary_and_partial_valid_counts(self) -> None:
        """Checks that circular summaries cross zero correctly and count missing outputs separately."""

        # Provide valid elevation inputs for a controlled sequence of operation outputs
        dem = gu.Raster.from_array(np.ones((2, 2)), Affine(1, 0, 0, 0, -1, 2), 32632)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 1)])

        # Return the original estimate first, then angles straddling zero and one missing value
        outputs = iter([np.array([0.0, 1.0]), np.array([359.0, np.nan]), np.array([1.0, 2.0])])

        # Summarize the two simulated outputs using a 360-degree period
        result = xdem.uncertainty.propagate_uncertainty(
            dem,
            lambda value: next(outputs),
            error_structure=structure,
            nsim=2,
            circular_period=360,
        )

        # Angles of 359 and 1 degrees have a mean of zero and a circular spread close to one degree
        assert min(result.mean[0], 360 - result.mean[0]) == pytest.approx(0, abs=1e-12)
        assert result.std[0] == pytest.approx(1, rel=0.001)

        # One valid realization cannot define the standard deviation of the second output
        assert result.n_valid.tolist() == [2, 1]
        assert np.isnan(result.std[1])

    def test_simulation_failure_diagnostics(self) -> None:
        """Checks that skipped simulations are recorded and an insufficient successful ensemble raises an error."""

        # Provide a valid small raster with independent measurement errors
        dem = gu.Raster.from_array(np.ones((2, 2)), Affine(1, 0, 0, 0, -1, 2), 32632)
        structure = xdem.ErrorStructure([xdem.ErrorComponent("measurement", 1)])

        # Allow the original calculation, then fail one of the three simulations
        calls = iter([1, 0, 1, 1])

        def operation(value: Any) -> float:
            """Return the spatial mean or raise the failure selected by the test sequence."""

            if next(calls) == 0:
                raise RuntimeError("failed fit")
            return float(np.mean(value.data))

        # Opt into skipping failed realizations and check the accompanying warning
        with pytest.warns(UserWarning, match="failed fit"):
            result = xdem.uncertainty.propagate_uncertainty(
                dem,
                operation,
                error_structure=structure,
                nsim=3,
                on_error="warn",
                random_state=3,
            )

        # Check that diagnostics retain the failed simulation number and message
        assert result.n_success == 2
        assert result.failures == {1: "failed fit"}

        # Fail both simulations to verify that no uncertainty is reported from an empty ensemble
        calls = iter([1, 0, 0])
        with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="Only 0 of 2"):
            xdem.uncertainty.propagate_uncertainty(dem, operation, error_structure=structure, nsim=2, on_error="warn")

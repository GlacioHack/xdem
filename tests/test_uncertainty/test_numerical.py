"""Tests for elevation propagation and empirical patch uncertainty."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geoutils as gu
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from geoutils import Raster, Vector

import xdem
from xdem import examples


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
    """Test module for empirical spread between stable terrain patches at specified averaging areas."""

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

    def test_patches_method_array_masks_match_raster(self) -> None:
        """Checks that array masks select the same patch population for raster and plain array values."""

        # Create a deterministic grid with an inner stable region and a smaller unstable exclusion
        values = np.random.default_rng(3).normal(size=(20, 20))
        stable = np.zeros(values.shape, dtype=bool)
        stable[2:-2, 2:-2] = True
        unstable = np.zeros(values.shape, dtype=bool)
        unstable[8:12, 8:12] = True
        raster = Raster.from_array(values, Affine(1, 0, 0, 0, -1, 20), 32632, nodata=-9999)

        # Estimate identical square patches through the spatial and plain array input paths
        options: dict[str, Any] = {
            "areas": [16, 36],
            "stable_mask": stable,
            "unstable_mask": unstable,
            "patch_shape": "square",
        }
        raster_result = xdem.uncertainty.patches_method(raster, **options)
        array_result = xdem.uncertainty.patches_method(values, gsd=1, **options)

        # Check that cosampling and direct Boolean selection produce the same complete summary table
        assert list(raster_result) == list(array_result)
        np.testing.assert_allclose(raster_result.to_numpy(), array_result.to_numpy())


class TestPatchesMethodErrors:
    """Test module for unavailable optional patch convolution engines."""

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_patches_method__error_missing_numba(self) -> None:
        """Checks that requesting Numba patch convolution reports a clear error when Numba is unavailable."""

        # Use valid example data so the optional backend is the intended source of failure
        diff, mask = load_ref_and_diff()[1:3]
        gsd = diff.res[0]
        area = 10000

        # Request the Numba engine explicitly and check the dependency named by the error
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


class TestNumericalPropagation:
    """Test module for simulation summaries of spatial, terrain and callable operations."""

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
            error_structure=structure,
            inputs={"spatial": {"areas": {"area": mask}}},
            n_samples=200,
            random_state=42,
            return_samples=True,
        )

        # Repeat the same averaging operation as a callable using identical random draws
        direct = xdem.uncertainty.propagate_uncertainty(
            dem,
            lambda value: np.ma.mean(value.data[mask]),
            error_structure=structure,
            n_samples=200,
            random_state=42,
        )
        assert isinstance(result.std, pd.Series)
        assert isinstance(result.mean, pd.Series)
        assert isinstance(result.estimate, pd.Series)
        assert isinstance(result.n_valid, pd.Series)
        assert isinstance(direct.std, float)
        assert isinstance(direct.mean, float)

        # Allow sampling variability when comparing 200 realizations with the analytical standard error
        analytical_std = xdem.uncertainty.spatial_error_propagation([mask], structure, support=dem)[0]
        assert result.std["area"] == pytest.approx(analytical_std, rel=0.15)

        # The named workflow and callable use the same draws and must have identical summaries
        assert result.std["area"] == pytest.approx(direct.std)
        assert result.mean["area"] == pytest.approx(direct.mean)

        # Check counts and verify the reported spread against the retained realization stack
        assert result.n_success == result.n_valid["area"] == 200
        assert result.samples is not None
        assert result.std["area"] == pytest.approx(result.samples["area"].std(ddof=1))

        # Check the original estimate and ensure simulations leave the input elevations unchanged
        assert result.estimate["area"] == 1
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
            error_structure=structure,
            operation_options={"attribute": "slope", "engine": "scipy"},
            n_samples=4,
            random_state=3,
            at="all",
            return_samples=True,
        )
        assert isinstance(result.estimate, Raster)
        assert isinstance(result.mean, Raster)
        assert isinstance(result.std, Raster)

        # Perturbed slopes have a positive mean even though the unperturbed DEM is flat
        assert result.std.georeferenced_grid_equal(dem)
        assert np.nanmean(result.estimate.data) == pytest.approx(0)
        assert np.nanmean(result.mean.data) > 0

        # Compare selected reported spreads with the retained value table
        assert result.samples is not None
        selected_std = np.asarray(result.std.data).reshape(-1)[result.selection["flat_index"]]
        np.testing.assert_allclose(selected_std, result.samples.std(axis=0, ddof=1), atol=1e-12)

        # Check that terrain propagation preserves the original elevations
        assert np.all(dem.data == 1)

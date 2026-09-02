"""
Tests for the elevation-uncertainty module fixes, on the Longyearbyen sample data:
  1. the sqrt(2) "same precision" correction (Hugonnet et al., 2022, Eq. 7-8),
  2. pre-coregistration before inferring the error structure,
  3. correct variogram plotting via the public variogram model function.

Cropped sample data (``get_path_test``) and a small ``nsim`` are used for CI speed.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import xdem
from xdem import coreg
from xdem.dem import DEM
from xdem.uncertainty.uncertainty import _infer_uncertainty, _propag_uncertainty_coreg

SEED = 42


def _finite_array(elev: object) -> np.ndarray:
    """Return a 1D float array of values from a Raster or PointCloud error field (masked -> NaN)."""
    data = elev.data  # type: ignore[attr-defined]
    if np.ma.isMaskedArray(data):
        arr = data.filled(np.nan)
    else:
        arr = np.asarray(data, dtype=float)
    return np.asarray(arr, dtype=float).ravel()


def _load_dems() -> tuple[DEM, DEM]:
    fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
    fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")
    return DEM(fn_ref), DEM(fn_tba)


class TestUncertaintyNotebook:

    def test_precision_of_other_sqrt2(self) -> None:
        """precision_of_other='same' divides the inferred error by sqrt(2) vs 'finer' (Eq. 7-8)."""
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)
        dem_ref, dem_tba = _load_dems()

        het_finer, _ = _infer_uncertainty(dem_ref, dem_tba, precision_of_other="finer", random_state=SEED)
        het_same, _ = _infer_uncertainty(dem_ref, dem_tba, precision_of_other="same", random_state=SEED)

        sig_finer = _finite_array(het_finer[0])
        sig_same = _finite_array(het_same[0])

        m = np.isfinite(sig_finer) & np.isfinite(sig_same)
        assert m.sum() > 0
        # Same random_state -> identical binning/fit, so the only difference is the sqrt(2) scaling.
        np.testing.assert_allclose(sig_same[m], sig_finer[m] / np.sqrt(2), rtol=1e-5)

    def test_precision_of_other_invalid_raises(self) -> None:
        """An unsupported precision (e.g. a coarser 'worse' other) raises instead of silently acting like 'finer'."""
        pytest.importorskip("skgstat")
        dem_ref, dem_tba = _load_dems()
        with pytest.raises(ValueError, match="precision_of_other"):
            _infer_uncertainty(dem_ref, dem_tba, precision_of_other="worse")  # type: ignore[arg-type]

    def test_precoreg_equivalence(self) -> None:
        """precoreg=True equals a manual fit->apply->propagate(precoreg=False) with a matched RNG stream."""
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)
        dem_ref, dem_tba = _load_dems()
        method = coreg.LZD()
        nsim = 5

        # Auto: precoreg performs the initial fit+apply internally, consuming the shared rng first.
        auto = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="ref",
            precoreg=True,
            random_state=SEED,
        )[0]

        # Manual mirror: advance an rng with the SAME initial fit + apply, then propagate with
        # precoreg=False passing the *advanced* generator so the random stream continues identically.
        rng = np.random.default_rng(SEED)
        c0 = method.copy()
        c0.fit(reference_elev=dem_ref, to_be_aligned_elev=dem_tba, inlier_mask=None, random_state=rng)
        dem_tba_align = c0.apply(dem_tba)
        manual = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba_align,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="ref",
            precoreg=False,
            random_state=rng,
        )[0]

        pd.testing.assert_frame_equal(auto, manual, check_exact=False, rtol=1e-6, atol=1e-6)

    def test_precoreg_deterministic(self) -> None:
        """Two precoreg=True runs with the same seed give identical reports."""
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)
        dem_ref, dem_tba = _load_dems()
        method = coreg.NuthKaab()
        nsim = 4

        kw = dict(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            precoreg=True,
            random_state=123,
        )
        r1 = _propag_uncertainty_coreg(**kw)[0]
        r2 = _propag_uncertainty_coreg(**kw)[0]
        pd.testing.assert_frame_equal(r1, r2, check_exact=False, rtol=0, atol=0)

    def test_diverged_simulation_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: Any,
        assert_and_allow_log: Callable[..., None],
    ) -> None:
        """A simulation whose coregistration fails to converge is skipped (with a warning) rather than
        aborting the whole Monte Carlo run; the remaining simulations still yield a finite report.

        A real divergence (e.g. NuthKaab exhausting its subsample on a small/pre-aligned extent) is
        numerics- and version-dependent, so the failure is forced deterministically here instead.
        """
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)
        dem_ref, dem_tba = _load_dems()

        # Make exactly one simulation's coregistration raise, independent of platform/library versions.
        original_fit = coreg.NuthKaab.fit
        state = {"n": 0}

        def flaky_fit(self: coreg.NuthKaab, *args: object, **kwargs: object) -> object:
            state["n"] += 1
            if state["n"] == 2:
                raise ValueError("forced non-convergence for test")
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(coreg.NuthKaab, "fit", flaky_fit)

        with caplog.at_level(logging.WARNING):
            summary = _propag_uncertainty_coreg(
                reference_elev=dem_ref,
                to_be_aligned_elev=dem_tba,
                coreg_method=coreg.NuthKaab(),
                nsim=4,
                error_applied_to="tba",
                precoreg=False,
                random_state=SEED,
            )[0]

        # The skip is expected: confirm it was logged (and allow it past the global log-warning collector).
        assert_and_allow_log(caplog, level=logging.WARNING, match="skipped")
        for k in ("tx", "ty", "tz"):
            assert np.isfinite(summary.loc[k, "std"])

    def test_variogram_plotting_invariant(self) -> None:
        """Fitted variogram (rises 0->sill) and correlation (falls 1->0) satisfy gamma = sill*(1 - rho).

        This is the hack-free basis for plotting the empirical + fitted variogram together:
        ``plot_variogram(corr_out[0], [xdem.spatialstats.get_variogram_model_func(corr_out[1])])``.
        """
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)
        dem_ref, dem_tba = _load_dems()

        _, corr_out = _infer_uncertainty(dem_ref, dem_tba, random_state=SEED)
        df_emp, params, corr_func = corr_out

        # Reconstruct the variogram model function from the already-public helper (the correct input for
        # plot_variogram, which expects a variogram rather than the returned correlation function).
        vario_func = xdem.spatialstats.get_variogram_model_func(params)
        total_sill = float(params["psill"].sum())
        assert total_sill > 0
        max_range = float(params["range"].max())

        h = np.array([0.0, max_range, 5.0 * max_range])
        rho = np.asarray(corr_func(h), dtype=float)
        gamma = np.asarray(vario_func(h), dtype=float)

        # Correlation starts at 1 and decays to ~0; variogram starts at 0 and rises to ~sill.
        np.testing.assert_allclose(rho[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(gamma[0], 0.0, atol=1e-9 + 1e-6 * total_sill)
        assert rho[-1] < 1e-2
        np.testing.assert_allclose(gamma[-1], total_sill, rtol=1e-2)

        # Exact invariant linking the two representations (what makes the plot render correctly).
        np.testing.assert_allclose(gamma, total_sill * (1.0 - rho), rtol=1e-9, atol=1e-12)

"""Functions to test the DEM tools."""

from __future__ import annotations

import warnings
from importlib.util import find_spec

import pandas as pd
import numpy as np
import pytest
import xdem
import xdem.vcrs
from xdem.dem import DEM
from xdem import coreg
from xdem.uncertainty.uncertainty import _infer_uncertainty, _propag_uncertainty_coreg

from geoutils import PointCloud, Raster


class TestUncertainty:

    def test_infer_uncertainty__inputs(self) -> None:

        # Import optional skgstat or skip test
        pytest.importorskip("skgstat")

        warnings.filterwarnings("ignore", category=UserWarning)

        fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        epc_ref = dem_ref.to_pointcloud()
        dem_tba = DEM(fn_tba)

        # Test with input "EPC, DEM"
        sig_h, corr_sig = _infer_uncertainty(epc_ref, dem_tba, random_state=42)
        assert isinstance(sig_h, PointCloud)
        assert callable(corr_sig)

        # Test with input "DEM, EPC"
        sig_h, corr_sig = _infer_uncertainty(dem_tba, epc_ref, random_state=42)
        assert isinstance(sig_h, Raster)
        assert callable(corr_sig)

        # Test with input "DEM, DEM"
        sig_h, corr_sig = _infer_uncertainty(dem_ref, dem_tba, random_state=42)
        assert isinstance(sig_h, Raster)
        assert callable(corr_sig)


    def test_propag_uncertainty_coreg__runs_and_reports(self) -> None:
        # Import optional skgstat or skip test
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)

        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_ref = dem_ref.to_pointcloud()

        # A lightweight coreg method for tests
        method = coreg.LZD()

        # Keep nsim small for CI speed
        nsim = 5
        seed = 42

        # --- DEM, DEM ---
        report = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            inlier_mask=None,
            random_state=seed,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )

        assert isinstance(report, pd.DataFrame)

        # Expect a per-parameter report for translations/rotations
        expected_index = ["tx", "ty", "tz", "rx", "ry", "rz"]
        for k in expected_index:
            assert k in report.index

        # Expect at least mean/std columns (as per your postproc)
        assert "mean" in report.columns
        assert "std" in report.columns

        # Values should be finite (std can be 0 in degenerate cases, but not NaN)
        assert np.isfinite(report.loc[expected_index, "mean"]).all()
        assert np.isfinite(report.loc[expected_index, "std"]).all()

        # --- EPC, DEM (mixed input) ---
        report2 = _propag_uncertainty_coreg(
            reference_elev=epc_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            inlier_mask=None,
            random_state=seed,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )
        assert isinstance(report2, pd.DataFrame)
        for k in expected_index:
            assert k in report2.index
        assert "mean" in report2.columns and "std" in report2.columns

        # --- DEM, DEM with error applied to ref (alternate branch) ---
        report3 = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="ref",
            inlier_mask=None,
            random_state=seed,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )
        assert isinstance(report3, pd.DataFrame)
        for k in expected_index:
            assert k in report3.index
        assert "mean" in report3.columns and "std" in report3.columns

    def test_propag_uncertainty_coreg__deterministic_seed(self) -> None:
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)

        fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)

        method = coreg.NuthKaab()
        nsim = 4
        seed = 123

        r1 = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            random_state=seed,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )
        r2 = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            random_state=seed,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )

        # Same seed should give identical report (within float tolerance)
        pd.testing.assert_frame_equal(r1, r2, check_exact=False, rtol=0, atol=0)
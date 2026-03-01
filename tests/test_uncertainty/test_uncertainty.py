"""Functions to test the DEM tools."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
import itertools

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
        hetesc_out, corr_out = _infer_uncertainty(epc_ref, dem_tba, random_state=42)
        assert isinstance(hetesc_out[0], PointCloud)
        assert callable(corr_out[2])

        # Test with input "DEM, EPC"
        hetesc_out, corr_out = _infer_uncertainty(dem_tba, epc_ref, random_state=42)
        assert isinstance(hetesc_out[0], Raster)
        assert callable(corr_out[2])

        # Test with input "DEM, DEM"
        hetesc_out, corr_out = _infer_uncertainty(dem_ref, dem_tba, random_state=42)
        assert isinstance(hetesc_out[0], Raster)
        assert callable(corr_out[2])

    def test_propag_uncertainty_coreg(self):
        """Check basic function run and outputs."""

        fn_ref = xdem.examples.get_path("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path("longyearbyen_tba_dem")
        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        method = coreg.LZD()

        report = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=5,
            error_applied_to="ref",
            inlier_mask=None,
            random_state=42,
        )[0]

        assert isinstance(report, pd.DataFrame)
        # Expect a per-parameter report for translations/rotations
        expected_index = ["tx", "ty", "tz", "rx", "ry", "rz"]
        for k in expected_index:
            assert k in report.index
        # Expect at least mean/std columns
        assert "mean" in report.columns
        assert "std" in report.columns
        # Values should be finite
        assert np.isfinite(report.loc[expected_index, "mean"]).all()
        assert np.isfinite(report.loc[expected_index, "std"]).all()


    def test_coreg__symmetry(self) -> None:

        fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_ref = dem_ref.to_pointcloud()
        epc_tba = dem_tba.to_pointcloud()

        # A lightweight symmetric coreg method for tests
        # method = coreg.ICP(method="point-to-point", sampling_strategy="independent", subsample=1)
        method = coreg.LZD(subsample=1)

        input_pairs = [
            ("DEM/DEM", dem_ref, dem_tba),
            ("DEM/EPC", dem_ref, epc_tba),
            ("EPC/DEM", epc_tba, dem_ref),
        ]
        for pair in input_pairs:
            m = method.copy()
            m.fit(pair[1], pair[2])
            tr = coreg.translations_rotations_from_matrix(m.to_matrix())

            print(pair[0])
            print(tr)

        assert False

    @pytest.mark.parametrize(
        "coreg_method",
        [
            pytest.param(coreg.ICP(method="point-to-plane"), id="ICP-p2plane"),
            pytest.param(coreg.ICP(method="point-to-point"), id="ICP-p2point"),
            pytest.param(coreg.LZD(), id="LZD"),
            pytest.param(coreg.NuthKaab(), id="NuthKaab"),
            pytest.param(coreg.CPD(lsg=False), id="CPD"),
            pytest.param(coreg.CPD(lsg=True), id="CPD-LSG"),
        ],
    )
    def test_propag_uncertainty_coreg__symmetry(self, coreg_method: coreg.Coreg) -> None:
        """
        Check that propagation of uncertainty is roughly symmetric:
        - Same result if DEM is exactly converted to EPC,
        - Same result when applying error to the other dataset (larger margin, as error can propagate differently
        for methods using gradient derived on a single source)
        """
        # Import optional skgstat or skip test
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)

        fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)
        epc_ref = dem_ref.to_pointcloud()
        epc_tba = dem_tba.to_pointcloud()

        # A lightweight symmetric coreg method for tests
        method = coreg_method.copy()
        # Keep nsim small for CI speed
        nsim = 10
        random_state = 42

        input_pairs = [
            ("DEM/DEM", dem_ref, dem_tba),
            ("DEM/EPC", dem_ref, epc_tba),
            ("EPC/DEM", epc_ref, dem_tba),
        ]
        error_sides = ["ref", "tba"]
        # We'll store the first output  (DEM, DEM, "ref"), and use it as a baseline to compare to others
        baseline = None
        baseline_label = None

        # We label the for loop for clearer error raising through pytest
        for (pair_label, ref_elev, tba_elev), side in itertools.product(input_pairs, error_sides):
            combo_label = f"{pair_label}, error_applied_to={side}"

            # Run propagation
            out = _propag_uncertainty_coreg(
                reference_elev=ref_elev,
                to_be_aligned_elev=tba_elev,
                coreg_method=method,
                nsim=nsim,
                error_applied_to=side,
                random_state=random_state,
            )[0]

            print(f"Label: {combo_label}")
            print("Out:", out)

            tn = ["tx", "ty", "tz"]
            rn = ["rx", "ry", "rz"]

            # Check relative equivalence to other runs
            if baseline is None:
                baseline = out
                baseline_label = combo_label

                # Translation magnitude rows (mean and std)
                t_mag_mean = float(np.sqrt((baseline.loc[tn, "mean"] ** 2).sum()))
                t_mag_std = float(np.sqrt((baseline.loc[tn, "std"] ** 2).sum()))
                baseline.loc["t_mag", ["mean", "std"]] = [t_mag_mean, t_mag_std]

                # Rotation magnitude rows (mean and std)
                r_mag_mean = float(np.sqrt((baseline.loc[rn, "mean"] ** 2).sum()))
                r_mag_std = float(np.sqrt((baseline.loc[rn, "std"] ** 2).sum()))
                baseline.loc["r_mag", ["mean", "std"]] = [r_mag_mean, r_mag_std]

            else:
                try:
                    # Dynamic rtol: 30% of the magnitude for both mean/std of translation/rotation
                    np.allclose(out.loc[tn, "mean"], baseline.loc[tn, "mean"], rtol=0.3 * t_mag_mean)
                    np.allclose(out.loc[tn, "std"], baseline.loc[tn, "std"], rtol=0.3 * t_mag_std)
                    np.allclose(out.loc[rn, "mean"], baseline.loc[rn, "mean"], rtol=0.3 * r_mag_mean)
                    np.allclose(out.loc[rn, "std"], baseline.loc[rn, "std"], rtol=0.3 * r_mag_std)

                except AssertionError as e:
                    raise AssertionError(
                        f"\ncoreg_method={getattr(method, '__class__', type(method)).__name__}\n"
                        f"Outputs differ across combinations but should be invariant.\n"
                        f"Baseline: {baseline_label}\n"
                        f"Current:  {combo_label}\n\n"
                        f"{e}"
                    )

    def test_propag_uncertainty_coreg__deterministic(self) -> None:
        pytest.importorskip("skgstat")
        warnings.filterwarnings("ignore", category=UserWarning)

        fn_ref = xdem.examples.get_path_test("longyearbyen_ref_dem")
        fn_tba = xdem.examples.get_path_test("longyearbyen_tba_dem")

        dem_ref = DEM(fn_ref)
        dem_tba = DEM(fn_tba)

        method = coreg.NuthKaab()
        nsim = 4
        random_state = 123

        r1 = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            random_state=random_state,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )[0]
        r2 = _propag_uncertainty_coreg(
            reference_elev=dem_ref,
            to_be_aligned_elev=dem_tba,
            coreg_method=method,
            nsim=nsim,
            error_applied_to="tba",
            random_state=random_state,
            kwargs_coreg_fit={},
            kwargs_infer_uncertainty={},
        )[0]

        # Same seed should give identical report (within float tolerance)
        pd.testing.assert_frame_equal(r1, r2, check_exact=False, rtol=0, atol=0)
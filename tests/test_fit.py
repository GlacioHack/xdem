"""
Functions to test the fitting tools.
"""

import platform
import warnings
from importlib.util import find_spec

import numpy as np
import pytest
from sklearn.metrics import mean_squared_error, median_absolute_error

import xdem
from xdem.fit import design_matrix_polynomial_2d, polynomial_2d


class TestRobustFitting:
    @pytest.mark.parametrize(
        "pkg_estimator",
        [
            ("sklearn", "Linear"),
            ("scipy", "Linear"),
            ("sklearn", "Theil-Sen"),
            ("sklearn", "RANSAC"),
            ("sklearn", "Huber"),
        ],
    )
    def test_robust_norder_polynomial_fit(self, pkg_estimator: str) -> None:

        # Import optional sklearn or skip test
        pytest.importorskip("sklearn")

        # Define x vector
        x = np.linspace(-50, 50, 1000)
        # Define exact polynomial
        true_coefs = [-100, 5, 3, 2]
        y = np.polyval(np.flip(true_coefs), x).astype(np.float32)

        # Run fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="lbfgs failed to converge")
            warnings.filterwarnings("ignore", message="Covariance of the parameters could not be*")
            coefs, deg = xdem.fit.robust_norder_polynomial_fit(
                x,
                y,
                linear_pkg=pkg_estimator[0],
                estimator_name=pkg_estimator[1],
                random_state=42,
                margin_improvement=50,
            )

        # Check coefficients are constrained
        assert deg == 3 or deg == 4
        error_margins = [100, 5, 2, 1]
        for i in range(4):
            assert coefs[i] == pytest.approx(true_coefs[i], abs=error_margins[i])

    @pytest.mark.skipif(find_spec("sklearn") is not None, reason="Only runs if scikit-learn is missing.")
    def test_robust_norder_polynomial_fit__missing_dep(self) -> None:
        """Check that proper import error is raised when sklearn is missing"""

        with pytest.raises(ImportError, match="Optional dependency 'scikit-learn' required.*"):
            xdem.fit.robust_norder_polynomial_fit(np.array([1]), np.array([1]), linear_pkg="sklearn")

    def test_robust_norder_polynomial_fit_noise_and_outliers(self) -> None:

        # Import optional sklearn or skip test
        pytest.importorskip("sklearn")

        # Ignore sklearn convergence warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="lbfgs failed to converge")

        rng = np.random.default_rng(42)

        # Define x vector
        x = np.linspace(1, 10, 1000)
        # Define an exact polynomial
        true_coefs = [-100, 5, 3, 2]
        y = np.polyval(np.flip(true_coefs), x).astype(np.float32)
        # Add some noise on top
        y += rng.normal(loc=0, scale=3, size=1000)
        # Add some outliers
        y[50:75] = 0.0
        y[900:925] = 1000.0

        # Run with the "Linear" estimator
        _, _ = xdem.fit.robust_norder_polynomial_fit(
            x, y, estimator_name="Linear", linear_pkg="scipy", method="trf"  # loss="soft_l1", f_scale=0.5
        )

        # TODO: understand why this is not robust since moving from least_squares() to curve_fit(), while the
        #  arguments passed are exactly the same...

        # Scipy solution should be quite robust to outliers/noise (with the soft_l1 method and f_scale parameter)
        # However, it is subject to random processes inside the scipy function (couldn't find how to fix those...)
        # It can find a degree 3, or 4 with coefficient close to 0
        # assert deg in [3, 4]
        # acceptable_scipy_linear_margins = [3, 3, 1, 1]
        # for i in range(4):
        #     assert coefs[i] == pytest.approx(true_coefs[i], abs=acceptable_scipy_linear_margins[i])

        # The sklearn Linear solution with MSE cost function will not be robust
        coefs2, deg2 = xdem.fit.robust_norder_polynomial_fit(
            x, y, estimator_name="Linear", linear_pkg="sklearn", cost_func=mean_squared_error, margin_improvement=50
        )
        # It won't find the right degree because of the outliers and noise
        assert deg2 != 3
        # Using the median absolute error should improve the fit
        coefs3, deg3 = xdem.fit.robust_norder_polynomial_fit(
            x, y, estimator_name="Linear", linear_pkg="sklearn", cost_func=median_absolute_error, margin_improvement=50
        )
        # Will find the right degree, but won't find the right coefficients because of the outliers and noise
        assert deg3 == 3
        sklearn_linear_error = [50, 10, 5, 0.5]
        for i in range(4):
            assert np.abs(coefs3[i] - true_coefs[i]) > sklearn_linear_error[i]

        # Now, the robust estimators
        # Theil-Sen should have better coefficients
        coefs4, deg4 = xdem.fit.robust_norder_polynomial_fit(x, y, estimator_name="Theil-Sen", random_state=42)
        assert deg4 == 3
        # High degree coefficients should be well constrained
        assert coefs4[2] == pytest.approx(true_coefs[2], abs=1.5)
        assert coefs4[3] == pytest.approx(true_coefs[3], abs=1.5)

        # RANSAC also works
        coefs5, deg5 = xdem.fit.robust_norder_polynomial_fit(x, y, estimator_name="RANSAC", random_state=42)
        assert deg5 == 3

        # Huber should perform well, close to the scipy robust solution
        coefs6, deg6 = xdem.fit.robust_norder_polynomial_fit(x, y, estimator_name="Huber")
        assert deg6 == 3
        for i in range(3):
            assert coefs6[i + 1] == pytest.approx(true_coefs[i + 1], abs=1)

    def test_robust_nfreq_sumsin_fit(self) -> None:

        # Define X vector
        x = np.linspace(0, 10, 1000)
        # Define exact sum of sinusoid signal
        true_coefs = np.array([(5, 3, np.pi), (2, 0.5, 0)]).flatten()
        y = xdem.fit.sumsin_1d(x, *true_coefs)

        # Check that the function runs (we passed a small niter to reduce the computing time of the test)
        coefs, deg = xdem.fit.robust_nfreq_sumsin_fit(x, y, random_state=42, niter=10)

        # Check that the estimated sum of sinusoid correspond to the input, with better tolerance on the highest
        # amplitude sinusoid
        # TODO: Work on making results not random between OS with basinhopping, this currently fails on Windows and Mac
        if platform.system() == "Linux":
            # Test all parameters
            for i in np.arange(6):
                # For the phase, check the circular variable with distance to modulo 2 pi
                if (i + 1) % 3 == 0:
                    coef_diff = coefs[i] - true_coefs[i] % (2 * np.pi)
                    assert np.minimum(coef_diff, np.abs(2 * np.pi - coef_diff)) < 0.1
                # Else check normally
                else:
                    assert coefs[i] == pytest.approx(true_coefs[i], abs=0.1)

        # Check that using custom arguments does not trigger an error
        bounds = [(1, 7), (1, 10), (0, 2 * np.pi), (1, 7), (0.1, 4), (0, 2 * np.pi)]
        coefs, deg = xdem.fit.robust_nfreq_sumsin_fit(
            x, y, bounds_amp_wave_phase=bounds, max_nb_frequency=2, hop_length=0.01, random_state=42, niter=1
        )

    def test_robust_nfreq_simsin_fit_noise_and_outliers(self) -> None:

        # Check robustness to outliers
        rng = np.random.default_rng(42)
        # Define X vector
        x = np.linspace(0, 10, 1000)
        # Define exact sum of sinusoid signal
        true_coefs = np.array([(5, 3, np.pi), (3, 0.5, 0)]).flatten()
        y = xdem.fit.sumsin_1d(x, *true_coefs)

        # Add some noise
        y += rng.normal(loc=0, scale=0.25, size=1000)
        # Add some outliers
        y[50:75] = -10
        y[900:925] = 10

        # Define first guess for bounds and run
        bounds = [(3, 7), (1, 5), (0, 2 * np.pi), (1, 7), (0.1, 1), (0, 2 * np.pi), (0.1, 1), (0.1, 1), (0, 2 * np.pi)]
        coefs, deg = xdem.fit.robust_nfreq_sumsin_fit(x, y, random_state=42, bounds_amp_wave_phase=bounds, niter=5)

        # Should be less precise, but still on point
        # We need to re-order output coefficient to match input
        if coefs[3] > coefs[0]:
            coefs = np.concatenate((coefs[3:], coefs[0:3]))

        # Check values
        for i in range(2):
            assert coefs[3 * i] == pytest.approx(true_coefs[3 * i], abs=0.2)
            assert coefs[3 * i + 1] == pytest.approx(true_coefs[3 * i + 1], abs=0.2)
            error_phase = min(
                np.abs(coefs[3 * i + 2] - true_coefs[3 * i + 2]),
                np.abs(2 * np.pi - (coefs[3 * i + 2] - true_coefs[3 * i + 2])),
            )
            assert error_phase < 0.2


class TestDesignMatrices:
    """Tests for OLS design matrix builders."""

    def test_design_matrix_polynomial_2d_shape(self) -> None:
        """Design matrix has the correct shape for each polynomial order."""
        rng = np.random.default_rng(42)
        N = 500
        x = rng.uniform(0, 100, N)
        y = rng.uniform(0, 100, N)
        xdata = np.vstack([x, y])

        for order in [1, 2, 3, 4]:
            dm = design_matrix_polynomial_2d(order)
            X = dm(xdata)
            assert X.shape == (N, (order + 1) ** 2)

    def test_design_matrix_polynomial_2d_reproduces_polynomial_2d(self) -> None:
        """The design matrix spans the same polynomial space as polynomial_2d.

        Fits a noise-free polynomial_2d surface with OLS and checks that the
        predictions via unnormalized coefficients match the original values.
        Uses a small coordinate range so the Vandermonde system is well-conditioned.
        """
        rng = np.random.default_rng(42)
        N = 500

        for order in [1, 2, 3, 4]:
            # Keep coordinates small so x_scale/y_scale don't explode the condition number
            x = rng.uniform(0, 10, N)
            y = rng.uniform(0, 10, N)
            xdata = np.vstack([x, y])
            true_params = rng.normal(size=(order + 1) ** 2)
            z = polynomial_2d((x, y), *true_params)

            dm = design_matrix_polynomial_2d(order)
            X = dm(xdata)
            coeffs_norm = np.linalg.lstsq(X, z, rcond=None)[0]
            coeffs_orig = dm.unnormalize_coeffs(coeffs_norm)

            pred = polynomial_2d((x, y), *coeffs_orig)
            assert np.allclose(pred, z, rtol=1e-6), f"Prediction mismatch at order={order}"

    def test_design_matrix_polynomial_2d_roundtrip(self) -> None:
        """OLS via the design matrix recovers known coefficients (small coordinate range)."""
        rng = np.random.default_rng(0)
        N = 1000
        # Small range keeps the Vandermonde system well-conditioned for all tested orders
        x = rng.uniform(0, 10, N)
        y = rng.uniform(0, 10, N)
        xdata = np.vstack([x, y])

        for order in [1, 2, 3, 4]:
            true_params = rng.normal(size=(order + 1) ** 2)
            z = polynomial_2d((x, y), *true_params)

            dm = design_matrix_polynomial_2d(order)
            X = dm(xdata)
            coeffs_norm = np.linalg.lstsq(X, z, rcond=None)[0]
            recovered = dm.unnormalize_coeffs(coeffs_norm)

            assert np.allclose(recovered, true_params, atol=1e-4), f"Failed roundtrip at order={order}"

    def test_design_matrix_polynomial_2d_unnormalize_identity_at_unit_scale(self) -> None:
        """unnormalize_coeffs is a no-op when coordinates are already in [-1, 1]."""
        rng = np.random.default_rng(7)
        N = 100
        x = rng.uniform(-1, 1, N)
        y = rng.uniform(-1, 1, N)
        xdata = np.vstack([x, y])
        order = 2

        dm = design_matrix_polynomial_2d(order)
        dm(xdata)  # sets x_scale = y_scale = 1 internally
        coeffs = rng.normal(size=(order + 1) ** 2)

        # With unit scale, unnormalize_coeffs should return the same values
        assert np.allclose(dm.unnormalize_coeffs(coeffs), coeffs)

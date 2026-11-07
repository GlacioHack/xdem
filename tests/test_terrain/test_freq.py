from __future__ import annotations

import numpy as np
import pytest

import xdem

PLOT = False


class TestFreqAttribute:
    filepath = xdem.examples.get_path_test("longyearbyen_ref_dem")
    dem = xdem.DEM(filepath)

    def test_texture_shading(self) -> None:
        """Test the texture_shading function."""

        # Test with a simple DEM
        dem_simple = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        # Test basic functionality
        result = xdem.terrain.texture_shading(dem_simple, alpha=0.8)

        # Check output properties
        assert result.shape == dem_simple.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))  # No NaN values for simple case

        # Test different alpha values
        result_low = xdem.terrain.texture_shading(dem_simple, alpha=0.5)
        result_mid = xdem.terrain.texture_shading(dem_simple, alpha=0.8)
        result_high = xdem.terrain.texture_shading(dem_simple, alpha=1.5)

        # Results should be different for different alpha values
        assert not np.array_equal(result_low, result_mid)
        assert not np.array_equal(result_mid, result_high)

        # Test with NaN values
        dem_with_nan = dem_simple.copy()
        dem_with_nan[0, 0] = np.nan

        result_nan = xdem.terrain.texture_shading(dem_with_nan, alpha=0.8)
        assert result_nan.shape == dem_with_nan.shape
        assert np.isnan(result_nan[0, 0])  # NaN should be preserved

        # Test error handling
        with pytest.raises(ValueError, match="Alpha must be between 0 and 2"):
            xdem.terrain.texture_shading(dem_simple, alpha=-0.1)

        with pytest.raises(ValueError, match="Alpha must be between 0 and 2"):
            xdem.terrain.texture_shading(dem_simple, alpha=2.1)

    def test_texture_shading_flat_surface(self) -> None:
        """Test all zero on flat DEM."""
        dem = np.ones((3, 3), dtype=np.float32) * 1000
        out = xdem.terrain.texture_shading(dem, alpha=0.8)
        assert np.allclose(out, 0.0, atol=1e-6)  # flat → 0 everywhere

    def test_texture_shading_planar_ramp(self) -> None:
        """Test expected variability on planar ramp."""
        dem_slope = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)

        alpha = 0.8
        out = xdem.terrain.texture_shading(dem_slope, alpha=alpha)

        # eps-scaled absolute tol for tiny float32+FFT differences
        eps = np.finfo(out.dtype).eps  # ~1.19e-7 for float32
        # Factor 1000 is a pragmatic buffer; empirically ~1e-4 absolute differences on 3x3 grids
        atol = float(1000.0 * eps * (np.max(np.abs(out)) + 1.0))

        # No variation between columns → diff across columns ~ 0
        col_diffs = np.diff(out, axis=1)
        assert np.allclose(col_diffs, 0.0, rtol=0.0, atol=atol)

        # 3) Each row is (near) constant
        row_stds = np.std(out, axis=1)
        assert np.all(row_stds <= atol)

        # 4) Monotonic by row mean (increasing because input slope increases with row)
        row_means = np.mean(out, axis=1)
        assert row_means[1] >= row_means[0] - atol
        assert row_means[2] >= row_means[1] - atol

    def test_texture_shading_offset_invariance_and_signed(self) -> None:
        """Test invariance to vertical offset and signed output on non-flat DEMs."""
        rng = np.random.RandomState(0)
        dem = rng.randn(3, 3).astype(np.float32)

        out = xdem.terrain.texture_shading(dem, alpha=0.8)
        out_offset = xdem.terrain.texture_shading(dem + 1234.5, alpha=0.8)

        # Compare after removing mean; allow eps-scaled atol for float32+FFT on tiny grids
        out_d = out - np.nanmean(out)
        off_d = out_offset - np.nanmean(out_offset)
        eps = np.finfo(out.dtype).eps  # ~1.19e-7 for float32
        # Factor 1000 is a pragmatic buffer; empirically ~1e-4 absolute differences on 3x3 grids
        atol = 1000.0 * eps * (np.max(np.abs(out_d)) + 1.0)
        np.testing.assert_allclose(out_d, off_d, atol=atol, rtol=0)

        # Signed response: expect both negative and positive values
        assert np.nanmin(out) < 0 and np.nanmax(out) > 0

    def test_texture_shading_spectral_shift_with_alpha(self) -> None:
        """
        Test spectral shift with increased alpha.
        Increasing alpha shifts spectral power toward higher frequencies.
        The fraction of total power above a median frequency cutoff should
        be larger for alpha=1.5 than for alpha=0.5.
        """
        rng = np.random.RandomState(1)
        dem = rng.randn(3, 3).astype(np.float32)

        out_lo = xdem.terrain.texture_shading(dem, alpha=0.5)
        out_hi = xdem.terrain.texture_shading(dem, alpha=1.5)

        # Power spectra
        F_lo = np.fft.fftshift(np.fft.fft2(out_lo))
        F_hi = np.fft.fftshift(np.fft.fft2(out_hi))
        P_lo = F_lo.real**2 + F_lo.imag**2
        P_hi = F_hi.real**2 + F_hi.imag**2

        # Radial frequency grid
        h, w = out_lo.shape
        ky = np.fft.fftshift(np.fft.fftfreq(h))
        kx = np.fft.fftshift(np.fft.fftfreq(w))
        KX, KY = np.meshgrid(kx, ky)
        R = np.sqrt(KX**2 + KY**2)

        # Use the median radius as a simple high/low frequency cutoff
        r_cut = np.median(R[R > 0])

        # Fraction of power above cutoff should increase with alpha
        frac_hi = P_hi[R > r_cut].sum() / P_hi.sum()
        frac_lo = P_lo[R > r_cut].sum() / P_lo.sum()

        # Higher alpha should put more power into higher frequencies
        assert frac_hi > frac_lo

    def test_texture_shading_linear_scaling(self) -> None:
        """
        Linearity: T(c * DEM) ≈ c * T(DEM).
        We set rtol/atol using machine epsilon (`eps`) of the dtype to account for
        normal float32+FFT rounding. `eps` is the smallest number where 1+eps != 1,
        so scaling tolerances by eps (and by output magnitude/scale_factor) makes
        the test robust but still tight.
        """
        rng = np.random.RandomState(0)
        dem = rng.randn(3, 3).astype(np.float32)

        alpha = 0.8
        scale_factor = 3000.0

        out1 = xdem.terrain.texture_shading(dem, alpha=alpha)
        out2 = xdem.terrain.texture_shading(scale_factor * dem, alpha=alpha)

        # Tolerances scaled to dtype precision and output magnitude
        eps = np.finfo(out1.dtype).eps  # ~1.19e-7 for float32
        # Factor 50 is a pragmatic buffer; empirically ~3e-5 relative differences on 3x3 grids
        rtol = float(50 * eps * scale_factor)
        atol = float(50 * eps * np.max(np.abs(scale_factor * out1)))

        np.testing.assert_allclose(out2, scale_factor * out1, rtol=rtol, atol=atol)

    def test_texture_shading_via_get_terrain_attribute(self) -> None:
        """Test texture_shading via the get_terrain_attribute interface."""

        # Test with a simple DEM
        dem_simple = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype="float32")

        # Test via get_terrain_attribute
        result = xdem.terrain.get_terrain_attribute(dem_simple, "texture_shading")

        # Check output properties
        assert result.shape == dem_simple.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))

        # Test with multiple attributes including texture_shading
        slope, texture = xdem.terrain.get_terrain_attribute(dem_simple, ["slope", "texture_shading"], resolution=1.0)

        assert slope.shape == dem_simple.shape
        assert texture.shape == dem_simple.shape
        assert not np.array_equal(slope, texture)  # Should be different attributes

    def test_texture_shading_real_dem(self) -> None:
        """Test texture_shading with a real DEM."""

        dem = self.dem.copy()

        # Test texture shading
        result = xdem.terrain.texture_shading(dem, alpha=0.8)

        # Check output properties
        assert result.shape == dem.shape
        assert np.issubdtype(result.dtype, np.floating)
        assert np.all(np.isfinite(result))

    def test_nextprod_fft(self) -> None:
        """Test the _nextprod_fft helper function."""

        # Test known values
        assert xdem.terrain.freq._nextprod_fft(1) == 1
        assert xdem.terrain.freq._nextprod_fft(10) == 16
        assert xdem.terrain.freq._nextprod_fft(20) == 32
        assert xdem.terrain.freq._nextprod_fft(32) == 32
        assert xdem.terrain.freq._nextprod_fft(100) == 128

        # Test that result is always >= input
        for size in [1, 5, 13, 25, 37, 63, 91]:
            result = xdem.terrain.freq._nextprod_fft(size)
            assert result >= size

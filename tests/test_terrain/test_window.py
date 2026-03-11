from __future__ import annotations

from importlib.util import find_spec
from typing import Literal

import numpy as np
import pytest
import scipy
from packaging.version import Version
from scipy.ndimage import binary_dilation

import xdem

PLOT = False


class TestTerrainAttribute:
    filepath = xdem.examples.get_path_test("longyearbyen_ref_dem")
    dem = xdem.DEM(filepath)

    def test_rugosity_jenness(self) -> None:
        """
        Test the rugosity with the same example as in Jenness (2004),
        https://doi.org/10.2193/0091-7648(2004)032[0829:CLSAFD]2.0.CO;2.
        """

        # Derive rugosity from the function
        dem = np.array([[190, 170, 155], [183, 165, 145], [175, 160, 122]], dtype="float32")

        # Derive rugosity
        rugosity = xdem.terrain.rugosity(dem, resolution=100.0)

        # Rugosity of Jenness (2004) example
        r = 10280.48 / 10000.0

        assert rugosity[1, 1] == pytest.approx(r, rel=10 ** (-4))

    # Loop for various elevation differences with the center
    @pytest.mark.parametrize("dh", np.linspace(0.01, 100, 3))
    # Loop for different resolutions
    @pytest.mark.parametrize("resolution", np.linspace(0.01, 100, 3))
    def test_rugosity_simple_cases(self, dh: float, resolution: float) -> None:
        """Test the rugosity calculation for simple cases."""

        # We here check the value for a fully symmetric case: the rugosity calculation can be simplified because all
        # eight triangles have the same surface area, see Jenness (2004).

        # Derive rugosity from the function
        dem = np.array([[1, 1, 1], [1, 1 + dh, 1], [1, 1, 1]], dtype="float64")

        rugosity = xdem.terrain.rugosity(dem, resolution=resolution)

        # Half surface length between the center and a corner cell (in 3D: accounting for elevation changes)
        side1 = np.sqrt(2 * resolution**2 + dh**2) / 2.0
        # Half surface length between the center and a side cell (in 3D: accounting for elevation changes)
        side2 = np.sqrt(resolution**2 + dh**2) / 2.0
        # Half surface length between the corner and side cell (no elevation changes on this side)
        side3 = resolution / 2.0

        # Formula for area A of one triangle
        s = (side1 + side2 + side3) / 2.0
        A = np.sqrt(s * (s - side1) * (s - side2) * (s - side3))

        # We sum the area of the eight triangles, and divide by the planimetric area (resolution squared)
        r = 8 * A / (resolution**2)

        # Check rugosity value is valid
        assert r == pytest.approx(rugosity[1, 1], rel=10 ** (-6))

    def test_fractal_roughness(self) -> None:
        """Test fractal roughness for synthetic cases for which we know the output."""

        # The fractal dimension of a line is 1 (a single pixel with non-zero value)
        dem = np.zeros((13, 13), dtype="float64")
        dem[1, 1] = 6.5
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6], 3) == np.float32(1.0)

        # The fractal dimension of plane is 2 (a plan of pixels with non-zero values)
        dem = np.zeros((13, 13), dtype="float64")
        dem[:, 1] = 13
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6], 3) == np.float32(2.0)

        # The fractal dimension of a cube is 3 (a block of pixels with non-zero values
        dem = np.zeros((13, 13), dtype="float64")
        dem[:, :6] = 13
        frac_rough = xdem.terrain.fractal_roughness(dem)
        assert np.round(frac_rough[6, 6], 3) == np.float32(3.0)

    @pytest.mark.parametrize(
        "attribute",
        [
            "topographic_position_index",
            "terrain_ruggedness_index_Riley",
            "terrain_ruggedness_index_Wilson",
            "roughness",
            "rugosity",
            "fractal_roughness",
        ],
    )
    def test_get_windowed_attribute__engine(self, attribute: str) -> None:
        """Check that all windowed attributes give the same results with SciPy or Numba."""

        pytest.importorskip("numba")

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(15, 15))
        dem[5, 5] = np.nan  # Add NaN to check propagation from an existing NaN is similar

        # Get TRI method if specified
        if "Wilson" in attribute or "Riley" in attribute:
            tri_method: Literal["Riley", "Wilson"]
            tri_method = attribute.split("_")[-1]  # type: ignore
            attribute = "terrain_ruggedness_index"
        # Otherwise use any one, doesn't matter
        else:
            tri_method = "Wilson"

        attrs_scipy = xdem.terrain.window._get_windowed_indexes(
            dem=dem, window_size=3, resolution=1, windowed_indexes=[attribute], tri_method=tri_method
        )
        attrs_numba = xdem.terrain.window._get_windowed_indexes(
            dem=dem, window_size=3, resolution=1, windowed_indexes=[attribute], tri_method=tri_method, engine="numba"
        )

        assert np.allclose(attrs_scipy, attrs_numba, equal_nan=True)

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_get_surface_attribute__missing_dep(self) -> None:
        """Check that proper import error is raised when numba is missing"""

        rnd = np.random.default_rng(42)
        # Leave just enough space to have a NaN in the middle and still have a ring of valid values
        # after NaN propagation from edges + center
        dem = rnd.normal(size=(11, 11))
        dem[5, 5] = np.nan

        with pytest.raises(ImportError, match="Optional dependency 'numba' required.*"):
            xdem.terrain.get_terrain_attribute(dem, resolution=1, attribute="roughness", engine="numba")

    @pytest.mark.skipif(
        Version(scipy.__version__) < Version("1.16.0"),
        reason="SciPy version is too old and does not yet support vectorized_filter.",
    )
    @pytest.mark.parametrize(
        "attribute",
        [
            "topographic_position_index",
            "terrain_ruggedness_index_Riley",
            "terrain_ruggedness_index_Wilson",
            "roughness",
            "rugosity",
            "fractal_roughness",
        ],
    )
    def test_get_windowed_attribute__scipy_backend(self, attribute: str) -> None:
        """Check that all windowed attributes give the same result with SciPy generic_filter or vectorized_filter."""

        rnd = np.random.default_rng(42)
        dem = rnd.normal(size=(15, 15))
        dem[5, 5] = np.nan  # Add NaN to check propagation from an existing NaN is similar

        # Get TRI method if specified
        if "Wilson" in attribute or "Riley" in attribute:
            tri_method: Literal["Riley", "Wilson"]
            tri_method = attribute.split("_")[-1]  # type: ignore
            attribute = "terrain_ruggedness_index"
        # Otherwise use any one, doesn't matter
        else:
            tri_method = "Wilson"

        attrs_vectorized = xdem.terrain.window._get_windowed_indexes(
            dem=dem,
            window_size=3,
            resolution=1,
            windowed_indexes=[attribute],
            tri_method=tri_method,
            engine="scipy",
            force_scipy_backend="vectorized",
        )
        attrs_generic = xdem.terrain.window._get_windowed_indexes(
            dem=dem,
            window_size=3,
            resolution=1,
            windowed_indexes=[attribute],
            tri_method=tri_method,
            engine="scipy",
            force_scipy_backend="generic",
        )

        assert np.allclose(attrs_vectorized, attrs_generic, equal_nan=True)

    @pytest.mark.parametrize(
        "attribute",
        [
            attr
            for attr in xdem.terrain.available_attributes
            if attr not in ["aspect", "slope", "hillshade"] and "curvature" not in attr
        ],
    )
    @pytest.mark.parametrize("window_size", [3, 5, 7])
    def test_windowed_attribute__nan_propag(self, attribute: str, window_size: int) -> None:
        """
        Check that NaN propagation behaves as intended for windowed attributes, in short: NaN are propagated
        from the edges and from NaNs based on window size.
        """

        # Rugosity is only defined for a window size of 3
        if attribute == "rugosity" and window_size != 3:
            return

        # TODO: Open issue on why fractal roughness/texture shading don't behave the same
        if attribute in ["fractal_roughness", "texture_shading"]:
            return

        # Generate DEM
        rng = np.random.default_rng(42)
        dem = rng.normal(size=(20, 20))
        # Introduce NaNs
        dem[4, 4:6] = np.nan
        dem[17, 16] = np.nan
        mask_nan_dem = ~np.isfinite(dem)

        # Generate attribute
        attr = xdem.terrain.get_terrain_attribute(dem, resolution=1, attribute=attribute, window_size=window_size)
        mask_nan_attr = ~np.isfinite(attr)

        # We dilate the initial mask by a structuring element matching the window size of the surface fit
        struct = np.ones((window_size, window_size), dtype=bool)
        hw = int(window_size / 2)
        eroded_mask_dem = binary_dilation(mask_nan_dem.astype(int), structure=struct, iterations=1)
        # On edges, NaN should be expanded by the half-width rounded down of the window
        eroded_mask_dem[:hw, :] = True
        eroded_mask_dem[-hw:, :] = True
        eroded_mask_dem[:, :hw] = True
        eroded_mask_dem[:, -hw:] = True
        # We check the two masks are indeed the same
        assert np.array_equal(eroded_mask_dem, mask_nan_attr)

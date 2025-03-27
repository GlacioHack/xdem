"""Functions to test the coregistration blockwise classes."""

from __future__ import annotations

import numpy as np
import pytest
from geoutils import Raster, Vector
from geoutils.raster import RasterType
import xdem
from xdem import examples
from xdem.coreg.base import Coreg


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    return reference_dem, to_be_aligned_dem, glacier_mask


def coreg_object(path):
    step = xdem.coreg.NuthKaab(vertical_shift=False)
    coreg_obj = xdem.coreg.BlockwiseCoreg(step=step, tile_size=500, apply_z_correction=True,
                                          output_path=str(path))

    return coreg_obj


class TestBlockwiseCoreg():
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    def test_init_with_valid_parameters(self, tmp_path):

        coreg_obj = coreg_object(tmp_path)

        assert coreg_obj.procstep == xdem.coreg.NuthKaab(vertical_shift=False)
        assert coreg_obj.tile_size == 500
        assert coreg_obj.apply_z_correction is True
        assert coreg_obj.output_path == str(tmp_path)
        assert coreg_obj.x_coords == []
        assert coreg_obj.y_coords == []
        assert coreg_obj.shifts_x == []
        assert coreg_obj.shifts_y == []

    def test_init_with_invalid_step(self, tmp_path):

        with pytest.raises(ValueError) as e:
            xdem.coreg.BlockwiseCoreg(step=Coreg, tile_size=300, apply_z_correction=True, output_path=str(tmp_path))

        assert str(e.value) == "The 'step' argument must be an instantiated Coreg subclass. Hint: write e.g. ICP() instead of ICP"

    def test_ransac_with_large_data(self, tmp_path):

        coreg_obj = coreg_object(tmp_path)

        np.random.seed(0)
        num_points = 1000
        x_coords = np.random.rand(num_points) * 100
        y_coords = np.random.rand(num_points) * 100
        # add noises
        shifts = 2 * x_coords + 3 * y_coords + 5 + np.random.randn(num_points) * 0.1
        a, b, c = coreg_obj.ransac(x_coords, y_coords, shifts)

        assert np.isclose(a, 2.0, atol=0.2)
        assert np.isclose(b, 3.0, atol=0.2)
        assert np.isclose(c, 5.0, atol=0.2)

    def test_ransac_with_insufficient_points(self, tmp_path):

        coreg_obj = coreg_object(tmp_path)

        x_coords = np.array([1, 2, 3, 4, 5])
        y_coords = np.array([1, 2, 3, 4, 5])
        shifts = np.array([2, 4, 6, 8, 10])

        # Appeler la fonction ransac et vérifier qu'elle lève une exception
        with pytest.raises(ValueError):
            coreg_obj.ransac(x_coords, y_coords, shifts)

    def test_blockwise_corep(self, tmp_path):
        blockwise = xdem.coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(vertical_shift=False),
                                              output_path=str(tmp_path))

        reference_dem, to_be_aligned_dem, inlier_mask = load_examples()

        blockwise.fit(reference_dem, to_be_aligned_dem)
        blockwise.apply(to_be_aligned_dem)

        # Ground truth is global coregistration
        nuth_kaab = xdem.coreg.NuthKaab()
        aligned_dem = nuth_kaab.fit_and_apply(reference_dem, to_be_aligned_dem, inlier_mask)

        assert np.allclose(aligned_dem.data, xdem.DEM(str(tmp_path) + "aligned_dem.tif").data, atol=1e-05)
"""Functions to test the example data."""

from __future__ import annotations

import glob
import os

import geoutils as gu
import numpy as np
import pytest
from geoutils import PointCloud, Raster, Vector
from rasterio.coords import BoundingBox

from xdem import examples
from xdem._typing import NDArrayf


def load_examples() -> tuple[Raster, Raster, Vector, Raster]:
    """Load example files to try coregistration methods with."""

    ref_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    tba_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))
    ddem = Raster(examples.get_path("longyearbyen_ddem"))

    return ref_dem, tba_dem, glacier_mask, ddem


class TestExamples:

    ref_dem, tba_dem, glacier_mask, ddem = load_examples()

    @pytest.mark.parametrize(
        "rst_truevals_abs",
        [
            (ref_dem, np.array([465.11816, 207.3236, 208.30563, 748.7337, 797.28644], dtype=np.float32), None),
            (tba_dem, np.array([464.6715, 213.7554, 207.8788, 760.8192, 797.3268], dtype=np.float32), None),
            (ddem, np.array([1.37, -1.67, 0.13, -10.10, 2.49], dtype=np.float32), 10e-3),
        ],
    )
    def test_array_content(self, rst_truevals_abs: tuple[Raster, NDArrayf, float]) -> None:
        """Let's ensure the data arrays in the examples are always the same by checking randomly some values"""

        rst = rst_truevals_abs[0]
        truevals = rst_truevals_abs[1]
        abs = rst_truevals_abs[2]

        rng = np.random.default_rng(42)
        values = rng.choice(rst.data.data.flatten(), size=5, replace=False)

        assert values == pytest.approx(truevals, abs=abs)

    # Note: Following PR #329, no gaps on DEM edges after coregistration
    @pytest.mark.parametrize("rst_and_truenodata", [(ref_dem, 0), (tba_dem, 0), (ddem, 0)])
    def test_array_nodata(self, rst_and_truenodata: tuple[Raster, int]) -> None:
        """Let's also check that the data arrays have always the same number of not finite values"""

        rst = rst_and_truenodata[0]
        truenodata = rst_and_truenodata[1]
        mask = gu.raster.get_array_and_mask(rst)[1]

        assert np.sum(mask) == truenodata

    def test_get_path_test_longyearbyen(self) -> None:
        """Let's ensure that the cropped data are successfully downloaded in case call from the test."""

        path = examples.get_path_test("longyearbyen_ref_dem")

        dest_shape = (54, 70)
        dest_bounds = BoundingBox(left=512310.0, bottom=8660950.0, right=513710.0, top=8662030.0)
        longyearbyen_dir = os.path.dirname(os.path.dirname(path))

        assert sum([len(files) for _, _, files in os.walk(os.path.join(longyearbyen_dir, "data"))]) == 26
        assert sum([len(files) for _, _, files in os.walk(os.path.join(longyearbyen_dir, "processed"))]) == 4

        # Verify Raster files: shape and bounds

        for test_file in glob.glob(os.path.join(longyearbyen_dir, "*", "*_test.tif")):
            assert Raster(test_file).shape == dest_shape
            assert Raster(test_file).bounds == dest_bounds

        # Verify EPC file: number of rpoints
        path = examples.get_path_test("longyearbyen_epc")
        assert len(PointCloud(path, data_column="h_li").ds) == 793

        # Verify Vectors files: geometries intersect the raster bound
        path = examples.get_path_test("longyearbyen_glacier_outlines")
        vec = gu.Vector(path)

        # Verify that geometries intersect with raster bound
        rst_poly = gu.projtools.bounds2poly(dest_bounds)
        intersects_new = []
        for poly in vec.ds.geometry:
            intersects_new.append(poly.intersects(rst_poly))
        assert np.all(intersects_new)

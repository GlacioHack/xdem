"""Functions to test the example data."""
from __future__ import annotations

import geoutils as gu
import numpy as np
import pytest
from geoutils import Raster, Vector

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
        "rst_and_truevals",
        [
            (ref_dem, np.array([868.6489, 623.42194, 180.57921, 267.30765, 601.67615], dtype=np.float32)),
            (tba_dem, np.array([875.2358, 625.0544, 182.9936, 272.6586, 606.2897], dtype=np.float32)),
            (
                ddem,
                np.array(
                    [
                        -4.669189453125000000e-02,
                        -7.413940429687500000e-01,
                        1.499481201171875000e-01,
                        1.095550537109375000e00,
                        -5.904846191406250000e00,
                    ],
                    dtype=np.float32,
                ),
            ),
        ],
    )  # type: ignore
    def test_array_content(self, rst_and_truevals: tuple[Raster, NDArrayf]) -> None:
        """Let's ensure the data arrays in the examples are always the same by checking randomly some values"""

        rst = rst_and_truevals[0]
        truevals = rst_and_truevals[1]
        np.random.seed(42)
        values = np.random.choice(rst.data.data.flatten(), size=5, replace=False)

        assert values == pytest.approx(truevals)

    # Note: Following PR #329, no gaps on DEM edges after coregistration
    @pytest.mark.parametrize("rst_and_truenodata", [(ref_dem, 0), (tba_dem, 0), (ddem, 0)])  # type: ignore
    def test_array_nodata(self, rst_and_truenodata: tuple[Raster, int]) -> None:
        """Let's also check that the data arrays have always the same number of not finite values"""

        rst = rst_and_truenodata[0]
        truenodata = rst_and_truenodata[1]
        mask = gu.raster.get_array_and_mask(rst)[1]

        assert np.sum(mask) == truenodata

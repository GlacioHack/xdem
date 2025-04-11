"""Functions to test the coregistration blockwise classes."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from geoutils import Raster, Vector
from geoutils.raster import RasterType

import xdem
from xdem import coreg, examples, misc, spatialstats
from xdem.coreg import BlockwiseCoreg
from xdem.coreg.base import Coreg


def load_examples() -> tuple[RasterType, RasterType, Vector]:
    """Load example files to try coregistration methods with."""

    reference_dem = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_dem = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    # Crop to smaller extents for test speed
    res = reference_dem.res
    crop_geom = (
        reference_dem.bounds.left,
        reference_dem.bounds.bottom,
        reference_dem.bounds.left + res[0] * 300,
        reference_dem.bounds.bottom + res[1] * 300,
    )
    reference_dem = reference_dem.crop(crop_geom)
    to_be_aligned_dem = to_be_aligned_dem.crop(crop_geom)

    return reference_dem, to_be_aligned_dem, glacier_mask


class TestBlockwiseCoreg:
    ref, tba, outlines = load_examples()  # Load example reference, to-be-aligned and mask.
    inlier_mask = ~outlines.create_mask(ref)

    def test_blockwise_coreg(self, tmp_path) -> None:

        blockwise = coreg.BlockwiseCoreg(xdem.coreg.NuthKaab(), output_path=str(tmp_path))
        blockwise.fit(self.ref, self.tba)
        blockwise.apply(self.tba)


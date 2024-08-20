# mypy: disable-error-code=misc
from typing import Dict

import geopandas as gpd
import numpy as np
import pytest
from geoutils import Raster, Vector
from geoutils.raster import RasterType

from xdem import coreg, examples


@pytest.fixture()
def load_examples() -> tuple[RasterType, RasterType, Vector, RasterType, Dict[RasterType, bool]]:
    """Load example files to try coregistration methods with."""

    # Load example reference, to-be-aligned and mask.
    reference_raster = Raster(examples.get_path("longyearbyen_ref_dem"))
    to_be_aligned_raster = Raster(examples.get_path("longyearbyen_tba_dem"))
    glacier_mask = Vector(examples.get_path("longyearbyen_glacier_outlines"))

    inlier_mask = ~glacier_mask.create_mask(reference_raster)

    fit_params = dict(
        reference_elev=reference_raster,
        to_be_aligned_elev=to_be_aligned_raster,
        inlier_mask=inlier_mask,
        verbose=False,
    )
    return reference_raster, to_be_aligned_raster, glacier_mask, inlier_mask, fit_params


@pytest.fixture()
def three_d_coordinates() -> gpd.GeoDataFrame:
    """
    Create some 3D coordinates with Z coordinates being 0 to try the apply functions.
    """
    ref = Raster(examples.get_path("longyearbyen_ref_dem"))
    points_arr = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [0, 0, 0, 0]], dtype="float64").T
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=points_arr[:, 0], y=points_arr[:, 1], crs=ref.crs), data={"z": points_arr[:, 2]}
    )
    return points


all_coregs = [
    coreg.VerticalShift,
    coreg.NuthKaab,
    coreg.ICP,
    coreg.Deramp,
    coreg.TerrainBias,
    coreg.DirectionalBias,
]

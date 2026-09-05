"""Deterministic elevation point clouds and optional LAS/LAZ fixtures."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from xdem import EPC


@pytest.fixture
def epc_frame() -> gpd.GeoDataFrame:
    """Create exactly representable point elevations and auxiliary columns on a small projected grid."""

    # Quarter-metre elevations and integer coordinates remain exact with millimetre LAS scales
    cols, rows = np.meshgrid(np.arange(7), np.arange(5))
    heights = 100 + np.arange(cols.size) * 0.25
    frame = gpd.GeoDataFrame(
        {"height": heights, "intensity": np.arange(cols.size, dtype=np.uint16)},
        geometry=gpd.points_from_xy(500000 + cols.ravel() * 20, 8600000 - rows.ravel() * 20),
        crs="EPSG:32633+5703",
    )
    frame.attrs["data_column"] = "height"
    frame.attrs["geometry_type"] = "Point"
    frame.attrs["survey"] = "synthetic"
    return frame


@pytest.fixture(params=[".las", ".laz"])
def epc_las_path(epc_frame: gpd.GeoDataFrame, tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """Write an elevation point cloud through xDEM in each available LAS compression format."""

    pytest.importorskip("laspy")
    if request.param == ".laz":
        pytest.importorskip("lazrs")

    # Use a format that can store a compound CRS and preserve the auxiliary intensity dimension
    path = tmp_path / ("elevation" + request.param)
    EPC(epc_frame, data_column="height").to_las(
        path, version="1.4", point_format=6, scales=(0.001, 0.001, 0.001), offsets=(500000, 8600000, 0)
    )
    return path

"""Fixtures shared by the native EPC and Pandas accessor tests in this directory.

``epc_frame`` gives ``test_base.py`` and ``test_pd_accessor.py`` the same in-memory point cloud. ``epc_las_path``
writes that cloud to LAS and LAZ so both modules can also check file-backed, Dask and multiprocessing behavior.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest

from xdem import EPC


@pytest.fixture
def epc_frame() -> gpd.GeoDataFrame:
    """Create the common in-memory point cloud used by EPC base and Pandas accessor tests."""

    # Use values that remain exact after LAS scaling so all backends can be compared without a tolerance
    cols, rows = np.meshgrid(np.arange(7), np.arange(5))
    heights = 100 + np.arange(cols.size) * 0.25

    # Include a compound CRS and an extra integer column to check elevation metadata and column preservation
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
    """Write the common point cloud as LAS and, when its optional backend is installed, LAZ."""

    # Skip only the formats whose optional reader or compression backend is unavailable
    pytest.importorskip("laspy")
    if request.param == ".laz":
        pytest.importorskip("lazrs")

    # Use a format that stores the compound CRS and the auxiliary intensity column used by the tests
    path = tmp_path / ("elevation" + request.param)
    EPC(epc_frame, data_column="height").to_las(
        path, version="1.4", point_format=6, scales=(0.001, 0.001, 0.001), offsets=(500000, 8600000, 0)
    )
    return path

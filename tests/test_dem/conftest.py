"""Fixtures shared by the native DEM and Xarray accessor tests in this directory.

The common file fixture lets ``test_base.py`` and ``test_xr_accessor.py`` compare both interfaces with the same
file-backed elevations, missing data and metadata. It is deliberately small so eager, Dask and multiprocessing cases
remain fast.
"""

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from xdem import DEM


@pytest.fixture
def accessor_dem_path(tmp_path: Path) -> Path:
    """Write the common GeoTIFF used by DEM base and Xarray accessor tests."""

    # Create relief that gives terrain methods non-constant results
    rows, cols = np.indices((35, 43), dtype=np.float32)
    values = 900 + rows**2 / 8 + 2 * cols + 3 * np.sin(cols / 3)

    # Place missing values across the chunk sizes used by the tests, including one at the raster edge
    values[12:15, 16:19] = np.nan
    values[0, 0] = np.nan
    transform = Affine(20, 0, 500000, 0, -20, 8600000)

    # Write a file so tests can check eager and lazy loading while preserving raster metadata
    path = tmp_path / "elevation.tif"
    dem = DEM.from_array(values, transform, 32633, nodata=-9999, area_or_point="Area", tags={"survey": "synthetic"})
    dem.to_file(path)
    return path

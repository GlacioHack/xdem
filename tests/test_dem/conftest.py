"""Small deterministic DEMs shared by native-class and accessor tests."""

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from xdem import DEM


@pytest.fixture
def accessor_dem_path(tmp_path: Path) -> Path:
    """Create a curved elevation surface with gaps crossing several prospective chunk boundaries."""

    # Combine smooth curvature and relief so derivatives, window metrics and FFTs are all nontrivial
    rows, cols = np.indices((35, 43), dtype=np.float32)
    values = 900 + rows**2 / 8 + 2 * cols + 3 * np.sin(cols / 3)
    values[12:15, 16:19] = np.nan
    values[0, 0] = np.nan
    transform = Affine(20, 0, 500000, 0, -20, 8600000)

    # Write explicit nodata and metadata for checking loading and output preservation
    path = tmp_path / "elevation.tif"
    dem = DEM.from_array(values, transform, 32633, nodata=-9999, area_or_point="Area", tags={"survey": "synthetic"})
    dem.to_file(path)
    return path

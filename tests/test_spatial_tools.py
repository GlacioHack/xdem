"""Functions to test the spatial tools.

Author(s):
    Erik S. Holmlund

"""
import numpy as np

import xdem as du
from tests.test_coreg import EXAMPLE_PATHS


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    diff = du.spatial_tools.subtract_rasters(EXAMPLE_PATHS["dem1"], EXAMPLE_PATHS["dem2"])

    assert np.nanmean(np.abs(diff.data)) < 100

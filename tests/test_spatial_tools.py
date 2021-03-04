"""Functions to test the spatial tools.

Author(s):
    Erik S. Holmlund

"""
import numpy as np

import xdem
from xdem import examples


def test_dem_subtraction():
    """Test that the DEM subtraction script gives reasonable numbers."""
    diff = xdem.spatial_tools.subtract_rasters(
        examples.FILEPATHS["longyearbyen_ref_dem"],
        examples.FILEPATHS["longyearbyen_tba_dem"])

    assert np.nanmean(np.abs(diff.data)) < 100

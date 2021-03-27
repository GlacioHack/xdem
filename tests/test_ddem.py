
import numpy as np

import xdem


class TestdDEM:
    dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])

    def test_init(self):
        ddem = xdem.dDEM(
            xdem.spatial_tools.subtract_rasters(self.dem_2009, self.dem_1990, resampling_method="nearest"),
            start_time=np.datetime64("1990-08-01"),
            end_time=np.datetime64("2009-08-01")
        )

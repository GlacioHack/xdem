import datetime

import geoutils as gu
import numpy as np

import xdem


class TesttDEM:
    dem_2009 = xdem.dem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = xdem.dem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
    outlines_1990 = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

    def test_create(self):

        timestamps = [datetime.datetime(1990, 8, 1), datetime.datetime(2009, 8, 1), datetime.datetime(2060, 8, 1)]

        scott_1990 = gu.geovector.Vector(
            self.outlines_1990.ds.loc[self.outlines_1990.ds["NAME"] == "Scott Turnerbreen"]
        )

        mask = (scott_1990.create_mask(self.dem_2009) == 255).reshape(self.dem_2009.data.shape)

        dem_2060 = self.dem_2009.copy()
        dem_2060.data[mask] -= 30

        tdem = xdem.volume.tDEM(
            [self.dem_1990, self.dem_2009, dem_2060],
            timestamps=timestamps,
            reference_dem=1
        )

        # Check that the first raster is the oldest one and
        assert tdem.dems[0].data.max() == self.dem_1990.data.max()
        assert tdem.reference_dem.data.max() == self.dem_2009.data.max()

        tdem.subtract_dems(resampling_method="nearest")

        assert np.mean(tdem.ddems[0].data) > 0

        cumulative_dh = tdem.get_cumulative_dh(mask=mask)

        assert cumulative_dh.iloc[0] > cumulative_dh.iloc[-1]

        # print(cumulative_dh)

        #raise NotImplementedError

    def test_dem_datetimes(self):
        """Try to create the tDEM without the timestamps argument (instead relying on datetime attributes)."""
        self.dem_1990.datetime = datetime.datetime(1990, 8, 1)
        self.dem_2009.datetime = datetime.datetime(2009, 8, 1)

        tdem = xdem.volume.tDEM(
            [self.dem_1990, self.dem_2009]
        )

        assert len(tdem.timestamps) > 0

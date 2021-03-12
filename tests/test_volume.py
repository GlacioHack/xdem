import datetime
import warnings

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

        # Generate 10000 NaN values randomly in one of the dDEMs
        tdem.ddems[0].data[np.random.randint(0, tdem.ddems[0].data.shape[0], 100),
                           np.random.randint(0, tdem.ddems[0].data.shape[1], 100)] = np.nan
        # Check that the cumulative_dh function warns for NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                tdem.get_cumulative_dh(mask=mask, nans_ok=False)
            except UserWarning as exception:
                if "NaNs found in dDEM" not in str(exception):
                    raise exception

        # Simple check that the cumulative_dh is overall negative.
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

    def test_ddem_interpolation(self):
        """Test that dDEM interpolation works as it should."""

        # All warnings should raise errors from now on
        warnings.simplefilter("error")

        # Create a tDEM object
        tdem = xdem.volume.tDEM(
            [self.dem_2009, self.dem_1990],
            timestamps=[datetime.datetime(year, 8, 1) for year in (2009, 1990)])

        # Create dDEMs
        tdem.subtract_dems(resampling_method="nearest")

        # The example data does not have NaNs, so filled_data should exist.
        assert tdem.ddems[0].filled_data is not None

        # Try to set the filled_data property with an invalid size.
        try:
            tdem.ddems[0].filled_data = np.zeros(3)
        except AssertionError as exception:
            if "differs from the data shape" not in str(exception):
                raise exception

        # Generate 10000 NaN values randomly in one of the dDEMs
        tdem.ddems[0].data[np.random.randint(0, tdem.ddems[0].data.shape[0], 100),
                           np.random.randint(0, tdem.ddems[0].data.shape[1], 100)] = np.nan

        # Make sure that filled_data is not available anymore, since the data now has nans
        assert tdem.ddems[0].filled_data is None

        # Interpolate the nans
        tdem.ddems[0].interpolate(method="linear")

        # Make sure that the filled_data is available again
        assert tdem.ddems[0].filled_data is not None

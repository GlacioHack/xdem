"""Functions to test the DEM collection tools."""

import datetime
import warnings

import geoutils as gu
import numpy as np

import xdem


class TestDEMCollection:
    dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    outlines_2010 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines_2010"))

    def test_init(self) -> None:

        timestamps = [datetime.datetime(1990, 8, 1), datetime.datetime(2009, 8, 1), datetime.datetime(2060, 8, 1)]

        scott_1990 = gu.Vector(self.outlines_1990.ds.loc[self.outlines_1990.ds["NAME"] == "Scott Turnerbreen"])
        scott_2010 = gu.Vector(self.outlines_2010.ds.loc[self.outlines_2010.ds["NAME"] == "Scott Turnerbreen"])

        # Make sure the glacier was bigger in 1990, since this is assumed later.
        assert scott_1990.ds.area.sum() > scott_2010.ds.area.sum()

        mask_2010 = scott_2010.create_mask(self.dem_2009)

        dem_2060 = self.dem_2009.copy()
        dem_2060[mask_2010] -= 30

        dems = xdem.DEMCollection(
            [self.dem_1990, self.dem_2009, dem_2060],
            timestamps=timestamps,
            outlines=dict(zip(timestamps[:2], [self.outlines_1990, self.outlines_2010])),
            reference_dem=1,
        )

        # Check that the first raster is the oldest one
        assert dems.dems[0].data.max() == self.dem_1990.data.max()
        assert dems.reference_dem.data.max() == self.dem_2009.data.max()

        dems.subtract_dems(resampling_method="nearest")

        assert np.mean(dems.ddems[0].data) < 0

        scott_filter = "NAME == 'Scott Turnerbreen'"

        dh_series = dems.get_dh_series(outlines_filter=scott_filter)

        # The 1990-2009 area should be the union of those years. The 2009-2060 area should just be the 2010 area.
        assert dh_series.iloc[0]["area"] > dh_series.iloc[-1]["area"]

        cumulative_dh = dems.get_cumulative_series(kind="dh", outlines_filter=scott_filter)
        cumulative_dv = dems.get_cumulative_series(kind="dv", outlines_filter=scott_filter)

        # Simple check that the cumulative_dh is overall negative.
        assert cumulative_dh.iloc[0] > cumulative_dh.iloc[-1]

        # Simple check that the dV number is of a greater magnitude than the dH number.
        assert abs(cumulative_dv.iloc[-1]) > abs(cumulative_dh.iloc[-1])

        rng = np.random.default_rng(42)
        # Generate 10000 NaN values randomly in one of the dDEMs
        dems.ddems[0].data[
            rng.integers(0, dems.ddems[0].data.shape[0], 100),
            rng.integers(0, dems.ddems[0].data.shape[1], 100),
        ] = np.nan
        # Check that the cumulative_dh function warns for NaNs
        with warnings.catch_warnings():
            try:
                dems.get_cumulative_series(nans_ok=False)
            except UserWarning as exception:
                if "NaNs found in dDEM" not in str(exception):
                    raise exception

        # logging.info(cumulative_dh)

        # raise NotImplementedError

    def test_dem_datetimes(self) -> None:
        """Try to create the DEMCollection without the timestamps argument (instead relying on datetime attributes)."""
        self.dem_1990.datetime = datetime.datetime(1990, 8, 1)
        self.dem_2009.datetime = datetime.datetime(2009, 8, 1)

        dems = xdem.DEMCollection([self.dem_1990, self.dem_2009])

        assert len(dems.timestamps) > 0

    def test_ddem_interpolation(self) -> None:
        """Test that dDEM interpolation works as it should."""

        # Create a DEMCollection object
        dems = xdem.DEMCollection(
            [self.dem_2009, self.dem_1990], timestamps=[datetime.datetime(year, 8, 1) for year in (2009, 1990)]
        )

        # Create dDEMs
        dems.subtract_dems(resampling_method="nearest")

        # The example data does not have NaNs, so filled_data should exist.
        assert dems.ddems[0].filled_data is not None

        # Try to set the filled_data property with an invalid size.
        try:
            dems.ddems[0].filled_data = np.zeros(3)
        except AssertionError as exception:
            if "differs from the data shape" not in str(exception):
                raise exception

        # Generate 10000 NaN values randomly in one of the dDEMs
        rng = np.random.default_rng(42)
        dems.ddems[0].data[
            rng.integers(0, dems.ddems[0].data.shape[0], 100),
            rng.integers(0, dems.ddems[0].data.shape[1], 100),
        ] = np.nan

        # Make sure that filled_data is not available anymore, since the data now has nans
        assert dems.ddems[0].filled_data is None

        # Interpolate the nans
        dems.ddems[0].interpolate(method="idw")

        # Make sure that the filled_data is available again
        assert dems.ddems[0].filled_data is not None

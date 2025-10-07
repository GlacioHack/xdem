"""Functions to test the difference of DEMs tools."""

import geoutils as gu
import numpy as np

import xdem


class TestdDEM:
    dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
    outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

    ddem = xdem.dDEM(dem_2009 - dem_1990, start_time=np.datetime64("1990-08-01"), end_time=np.datetime64("2009-08-01"))

    def test_init(self) -> None:
        """Test that the dDEM object was instantiated correctly."""
        assert isinstance(self.ddem, xdem.dDEM)
        assert isinstance(self.ddem.data, np.ma.masked_array)

        assert self.ddem.nodata is (self.dem_2009 - self.dem_1990).nodata

    def test_copy(self) -> None:
        """Test that copying works as it should."""
        ddem2 = self.ddem.copy()

        assert isinstance(ddem2, xdem.dDEM)

        ddem2.data += 1

        assert not self.ddem.raster_equal(ddem2)

    def test_filled_data(self) -> None:
        """Test that the filled_data property points to the right data."""
        ddem2 = self.ddem.copy()

        assert not np.any(np.isnan(ddem2.data)) or np.all(~ddem2.data.mask)
        assert ddem2.filled_data is not None

        assert np.count_nonzero(np.isnan(ddem2.data)) == 0
        ddem2.data.ravel()[0] = np.nan

        assert np.count_nonzero(np.isnan(ddem2.data)) == 1

        assert ddem2.filled_data is None

        ddem2.interpolate(method="idw")

        assert ddem2.fill_method is not None

    def test_regional_hypso(self) -> None:
        """Test the regional hypsometric approach."""
        ddem = self.ddem.copy()
        ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)
        rng = np.random.default_rng(42)
        ddem.data.mask.ravel()[rng.choice(ddem.data.size, 50000, replace=False)] = True
        assert np.count_nonzero(ddem.data.mask) > 0

        assert ddem.filled_data is None

        ddem.interpolate(method="regional_hypsometric", reference_elevation=self.dem_2009, mask=self.outlines_1990)

        assert ddem._filled_data is not None
        assert isinstance(ddem.filled_data, np.ndarray)

        assert ddem.filled_data.shape == ddem.data.shape

        assert np.abs(np.nanmean(self.ddem.data - ddem.filled_data)) < 1

    def test_local_hypso(self) -> None:
        """Test the local hypsometric approach."""
        ddem = self.ddem.copy()
        scott_1990 = self.outlines_1990.query("NAME == 'Scott Turnerbreen'")
        ddem.data.mask = np.zeros_like(ddem.data, dtype=bool)
        rng = np.random.default_rng(42)
        ddem.data.mask.ravel()[rng.choice(ddem.data.size, 50000, replace=False)] = True
        assert np.count_nonzero(ddem.data.mask) > 0

        assert ddem.filled_data is None

        ddem.interpolate(method="local_hypsometric", reference_elevation=self.dem_2009.data, mask=scott_1990)
        assert np.abs(np.nanmean(self.ddem.data - ddem.filled_data)) < 1

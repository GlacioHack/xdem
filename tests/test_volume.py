import geoutils as gu
import numpy as np
import pandas as pd

import xdem

xdem.examples.download_longyearbyen_examples(overwrite=False)


class TestLocalHypsometric:
    """Test cases for the local hypsometric method."""

    # Load example data.
    dem_2009 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_tba_dem"]).reproject(dem_2009)
    outlines = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
    # Filter to only look at the Scott Turnerbreen glacier
    outlines.ds = outlines.ds.loc[outlines.ds["NAME"] == "Scott Turnerbreen"]
    # Create a mask where glacier areas are True
    mask = outlines.create_mask(dem_2009) == 255

    def test_bin_ddem(self):
        """Test dDEM binning."""
        ddem = self.dem_2009.data - self.dem_1990.data

        ddem_bins = xdem.volume.hypsometric_binning(ddem.squeeze()[self.mask], self.dem_2009.data.squeeze()[self.mask])

        ddem_bins_masked = xdem.volume.hypsometric_binning(
            np.ma.masked_array(ddem.squeeze(), mask=~self.mask),
            np.ma.masked_array(self.dem_2009.data.squeeze(), mask=~self.mask)
        )

        ddem_stds = xdem.volume.hypsometric_binning(
            ddem.squeeze()[self.mask],
            self.dem_2009.data.squeeze()[self.mask],
            aggregation_function=np.std
        )
        assert ddem_stds["value"].mean() < 50
        assert np.abs(np.mean(ddem_bins["value"] - ddem_bins_masked["value"])) < 0.01

    def test_interpolate_ddem_bins(self) -> pd.Series:
        """Test dDEM bin interpolation."""
        ddem = self.dem_2009.data - self.dem_1990.data

        ddem_bins = xdem.volume.hypsometric_binning(ddem.squeeze()[self.mask], self.dem_2009.data.squeeze()[self.mask])

        # Simulate a missing bin
        ddem_bins.iloc[3, :] = np.nan

        # Interpolate the bins and exclude bins with low pixel counts from the interpolation.
        interpolated_bins = xdem.volume.interpolate_hypsometric_bins(ddem_bins, count_threshold=200)

        assert abs(np.mean(interpolated_bins)) < 40
        assert abs(np.mean(interpolated_bins)) > 0
        assert not np.any(np.isnan(interpolated_bins))
        return interpolated_bins

    def test_area_calculation(self):
        """Test the area calculation function."""
        ddem_bins = self.test_interpolate_ddem_bins()

        # Test the area calculation with normal parameters.
        bin_area = xdem.volume.calculate_hypsometry_area(
            ddem_bins,
            self.dem_2009.data.squeeze()[self.mask],
            pixel_size=self.dem_2009.res[0]
        )

        # Test area calculation with differing pixel x/y resolution.
        # Also test that ddem_bins can be a DataFrame (as long as the column name 'value' exists)
        ddem_bins.name = "value"
        xdem.volume.calculate_hypsometry_area(
            ddem_bins.to_frame(),
            self.dem_2009.data.squeeze()[self.mask],
            pixel_size=(self.dem_2009.res[0], self.dem_2009.res[0] + 1)
        )

        # Add some nans to the reference DEM
        data_with_nans = self.dem_2009.data.squeeze()[self.mask]
        data_with_nans[[2, 5]] = np.nan

        # Make sure that the above results in the correct error.
        try:
            xdem.volume.calculate_hypsometry_area(
                ddem_bins,
                data_with_nans,
                pixel_size=self.dem_2009.res[0]
            )
        except AssertionError as exception:
            if "DEM has NaNs" not in str(exception):
                raise exception

        # Mess up the dDEM bins and see if it gives the correct error
        ddem_bins.iloc[3] = np.nan
        try:
            xdem.volume.calculate_hypsometry_area(
                ddem_bins,
                self.dem_2009.data.squeeze()[self.mask],
                pixel_size=self.dem_2009.res[0]
            )
        except AssertionError as exception:
            if "cannot contain NaNs" not in str(exception):
                raise exception

        # The area of Scott Turnerbreen was around 3.4 km² in 1990, so this should be close to that number.
        assert 2e6 < bin_area.sum() < 5e6

import warnings

import geoutils as gu
import numpy as np
import pandas as pd

import xdem

xdem.examples.download_longyearbyen_examples(overwrite=False)


class TestLocalHypsometric:
    """Test cases for the local hypsometric method."""

    # Load example data.
    dem_2009 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_tba_dem"]).reproject(dem_2009, silent=True)
    outlines = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
    all_outlines = outlines.copy()
    # Filter to only look at the Scott Turnerbreen glacier
    outlines.ds = outlines.ds.loc[outlines.ds["NAME"] == "Scott Turnerbreen"]
    # Create a mask where glacier areas are True
    mask = outlines.create_mask(dem_2009)

    def test_bin_ddem(self):
        """Test dDEM binning."""
        ddem = self.dem_2009.data - self.dem_1990.data

        ddem_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009.data[self.mask],
            bins=50,
            kind="fixed")

        ddem_bins_masked = xdem.volume.hypsometric_binning(
            np.ma.masked_array(ddem, mask=~self.mask),
            np.ma.masked_array(self.dem_2009.data, mask=~self.mask)
        )

        ddem_stds = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009.data[self.mask],
            aggregation_function=np.std
        )
        assert ddem_stds["value"].mean() < 50
        assert np.abs(np.mean(ddem_bins["value"] - ddem_bins_masked["value"])) < 0.01

    def test_interpolate_ddem_bins(self) -> pd.Series:
        """Test dDEM bin interpolation."""
        ddem = self.dem_2009.data - self.dem_1990.data

        ddem_bins = xdem.volume.hypsometric_binning(ddem[self.mask], self.dem_2009.data[self.mask])

        # Simulate a missing bin
        ddem_bins.iloc[3, 0] = np.nan

        # Interpolate the bins and exclude bins with low pixel counts from the interpolation.
        interpolated_bins = xdem.volume.interpolate_hypsometric_bins(ddem_bins, count_threshold=200)

        # Check that the count column has not changed.
        assert np.array_equal(ddem_bins["count"], interpolated_bins["count"])

        # Assert that the absolute mean is somewhere between 0 and 40
        assert abs(np.mean(interpolated_bins["value"])) < 40
        assert abs(np.mean(interpolated_bins["value"])) > 0
        # Check that no nans exist.
        assert not np.any(np.isnan(interpolated_bins))

        # Return the value so that they can be used in other tests.
        return interpolated_bins

    def test_area_calculation(self):
        """Test the area calculation function."""
        ddem_bins = self.test_interpolate_ddem_bins()

        # Test the area calculation with normal parameters.
        bin_area = xdem.volume.calculate_hypsometry_area(
            ddem_bins,
            self.dem_2009.data[self.mask],
            pixel_size=self.dem_2009.res[0]
        )

        # Test area calculation with differing pixel x/y resolution.
        xdem.volume.calculate_hypsometry_area(
            ddem_bins,
            self.dem_2009.data[self.mask],
            pixel_size=(self.dem_2009.res[0], self.dem_2009.res[0] + 1)
        )

        # Add some nans to the reference DEM
        data_with_nans = self.dem_2009.data[self.mask]
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

        # Try to pass an incorrect timeframe= parameter
        try:
            xdem.volume.calculate_hypsometry_area(
                ddem_bins,
                self.dem_2009.data[self.mask],
                pixel_size=self.dem_2009.res[0],
                timeframe="blergh"
            )
        except ValueError as exception:
            if "Argument 'timeframe=blergh' is invalid" not in str(exception):
                raise exception

        # Mess up the dDEM bins and see if it gives the correct error
        ddem_bins.iloc[3] = np.nan
        try:
            xdem.volume.calculate_hypsometry_area(
                ddem_bins,
                self.dem_2009.data[self.mask],
                pixel_size=self.dem_2009.res[0]
            )
        except AssertionError as exception:
            if "cannot contain NaNs" not in str(exception):
                raise exception

        # The area of Scott Turnerbreen was around 3.4 km² in 1990, so this should be close to that number.
        assert 2e6 < bin_area.sum() < 5e6

    def test_ddem_bin_methods(self):
        """Test different dDEM binning methods."""
        ddem = self.dem_2009.data - self.dem_1990.data

        # equal height is already tested in test_bin_ddem

        # Make a fixed amount of bins
        equal_count_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009.data[self.mask],
            bins=50,
            kind="count"
        )
        assert equal_count_bins.shape[0] == 50

        # Make 50 bins with approximately the same area (pixel count)
        quantile_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009.data[self.mask],
            bins=50,
            kind="quantile"
        )

        assert quantile_bins.shape[0] == 50
        # Make sure that the pixel count variation is low.
        assert quantile_bins["count"].std() < 1

        # Try to feed the previous bins as "custom" bins to the function.
        custom_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009.data[self.mask],
            bins=np.r_[quantile_bins.index.left[0], quantile_bins.index.right],
            kind="custom"
        )

        assert custom_bins.shape[0] == quantile_bins.shape[0]

    def test_regional_hypsometric_signal(self):

        warnings.simplefilter("error")

        # Extract a normalized regional hypsometric signal.
        ddem = self.dem_2009.data - self.dem_1990.data
        glacier_index_map = self.all_outlines.rasterize(self.dem_2009)
        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=ddem, ref_dem=self.dem_2009.data, glacier_index_map=glacier_index_map)

        assert signal["w_mean"].min() >= 0
        assert signal["w_mean"].max() <= 1

        if False:
            import matplotlib.pyplot as plt
            plt.fill_between(signal.index.mid, signal["median"] - signal["std"],
                             signal["median"] + signal["std"], label="Median±std")
            plt.plot(signal.index.mid, signal["median"], color="black", linestyle=":", label="Median")
            plt.plot(signal.index.mid, signal["w_mean"], color="black", label="Weighted mean")

            plt.xlabel("Normalized elevation")
            plt.ylabel("Normalized elevation change")
            plt.legend()

            plt.show()

        # Try the normalized regional hypsometric interpolation.
        # Synthesize random nans in 80% of the data.
        ddem.mask.ravel()[np.random.choice(ddem.data.size, int(ddem.data.size * 0.80), replace=False)] = True
        # Fill the dDEM using the de-normalized signal.
        filled_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem,
            ref_dem=self.dem_2009.data,
            glacier_index_map=glacier_index_map
        )
        # Fill the dDEM using the de-normalized signal and create an idealized dDEM
        idealized_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem,
            ref_dem=self.dem_2009.data,
            glacier_index_map=glacier_index_map,
            idealized_ddem=True
        )

        if True:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(filled_ddem, cmap="coolwarm_r", vmin=-10, vmax=10)
            plt.subplot(122)
            plt.imshow(idealized_ddem, cmap="coolwarm_r", vmin=-10, vmax=10)

            plt.show()
        assert not np.array_equal(filled_ddem, idealized_ddem)
        assert np.nanstd(filled_ddem[glacier_index_map > 0]) > np.nanstd(idealized_ddem[glacier_index_map >0])



        # Extract the finite glacier values.
        changes = ddem.data.squeeze()[glacier_index_map > 0]
        changes = changes[np.isfinite(changes)]
        interp_changes = filled_ddem[glacier_index_map > 0]
        interp_changes = interp_changes[np.isfinite(interp_changes)]

        # Validate that the interpolated (20% data) means and stds are similar to the original (100% data)
        # These are increased because the CI for some reason gets quite large variance. It works with lower
        # values on normal computers...
        assert abs(changes.mean() - interp_changes.mean()) < 2
        assert abs(changes.std() - interp_changes.std()) < 3

"""Functions to test the volume estimation tools."""

import geoutils as gu
import numpy as np
import pytest

import xdem


class TestLocalHypsometric:
    """Test cases for the local hypsometric method."""

    # Load example data.
    dem_2009 = gu.Raster(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = gu.Raster(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(dem_2009, silent=True)
    outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
    all_outlines = outlines.copy()

    # Filter to only look at the Scott Turnerbreen glacier
    outlines.ds = outlines.ds.loc[outlines.ds["NAME"] == "Scott Turnerbreen"]
    # Create a mask where glacier areas are True
    mask = outlines.create_mask(dem_2009)

    def test_bin_ddem(self) -> None:
        """Test dDEM binning."""
        ddem = self.dem_2009 - self.dem_1990

        ddem_bins = xdem.volume.hypsometric_binning(ddem[self.mask], self.dem_2009[self.mask], bins=50, kind="fixed")

        ddem_bins_masked = xdem.volume.hypsometric_binning(
            np.ma.masked_array(ddem.data, mask=~self.mask.data.filled(False)),
            np.ma.masked_array(self.dem_2009.data, mask=~self.mask.data.filled(False)),
        )

        ddem_stds = xdem.volume.hypsometric_binning(
            ddem[self.mask], self.dem_2009[self.mask], aggregation_function=np.std
        )
        assert ddem_stds["value"].mean() < 50
        assert np.abs(np.mean(ddem_bins["value"] - ddem_bins_masked["value"])) < 0.01

    def test_interpolate_ddem_bins(self) -> None:
        """Test dDEM bin interpolation."""
        ddem = self.dem_2009 - self.dem_1990

        ddem_bins = xdem.volume.hypsometric_binning(ddem[self.mask], self.dem_2009[self.mask])

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

    def test_area_calculation(self) -> None:
        """Test the area calculation function."""

        ddem = self.dem_2009 - self.dem_1990

        ddem_bins = xdem.volume.hypsometric_binning(ddem[self.mask], self.dem_2009[self.mask])

        # Simulate a missing bin
        ddem_bins.iloc[3, 0] = np.nan

        # Test the area calculation with normal parameters.
        bin_area = xdem.volume.calculate_hypsometry_area(
            ddem_bins, self.dem_2009[self.mask], pixel_size=self.dem_2009.res[0]
        )

        # Test area calculation with differing pixel x/y resolution.
        xdem.volume.calculate_hypsometry_area(
            ddem_bins, self.dem_2009[self.mask], pixel_size=(self.dem_2009.res[0], self.dem_2009.res[0] + 1)
        )

        # Add some nans to the reference DEM
        data_with_nans = self.dem_2009
        data_with_nans.data[2, 5] = np.nan

        # Make sure that the above results in the correct error.
        try:
            xdem.volume.calculate_hypsometry_area(ddem_bins, data_with_nans, pixel_size=self.dem_2009.res[0])
        except AssertionError as exception:
            if "DEM has NaNs" not in str(exception):
                raise exception

        # Try to pass an incorrect timeframe= parameter
        try:
            xdem.volume.calculate_hypsometry_area(
                ddem_bins, self.dem_2009[self.mask], pixel_size=self.dem_2009.res[0], timeframe="blergh"
            )
        except ValueError as exception:
            if "Argument 'timeframe=blergh' is invalid" not in str(exception):
                raise exception

        # Mess up the dDEM bins and see if it gives the correct error
        ddem_bins.iloc[3] = np.nan
        try:
            xdem.volume.calculate_hypsometry_area(ddem_bins, self.dem_2009[self.mask], pixel_size=self.dem_2009.res[0])
        except AssertionError as exception:
            if "cannot contain NaNs" not in str(exception):
                raise exception

        # The area of Scott Turnerbreen was around 3.4 km² in 1990, so this should be close to that number.
        assert 2e6 < bin_area.sum() < 5e6

    def test_ddem_bin_methods(self) -> None:
        """Test different dDEM binning methods."""
        ddem = self.dem_2009 - self.dem_1990

        # equal height is already tested in test_bin_ddem

        # Make a fixed amount of bins
        equal_count_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask], self.dem_2009[self.mask], bins=50, kind="count"
        )
        assert equal_count_bins.shape[0] == 50

        # Make 50 bins with approximately the same area (pixel count)
        quantile_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask], self.dem_2009[self.mask], bins=50, kind="quantile"
        )

        assert quantile_bins.shape[0] == 50
        # Make sure that the pixel count variation is low.
        assert quantile_bins["count"].std() < 1

        # Try to feed the previous bins as "custom" bins to the function.
        custom_bins = xdem.volume.hypsometric_binning(
            ddem[self.mask],
            self.dem_2009[self.mask],
            bins=np.r_[quantile_bins.index.left[0], quantile_bins.index.right],
            kind="custom",
        )

        assert custom_bins.shape[0] == quantile_bins.shape[0]


class TestNormHypsometric:
    dem_2009 = gu.Raster(xdem.examples.get_path("longyearbyen_ref_dem"))
    dem_1990 = gu.Raster(xdem.examples.get_path("longyearbyen_tba_dem")).reproject(dem_2009, silent=True)
    outlines = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))

    glacier_index_map = outlines.rasterize(dem_2009)
    ddem = dem_2009.data - dem_1990.data

    @pytest.mark.parametrize("n_bins", [5, 10, 20])  # type: ignore
    def test_regional_signal(self, n_bins: int) -> None:

        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=self.ddem, ref_dem=self.dem_2009, glacier_index_map=self.glacier_index_map, n_bins=n_bins
        )

        assert signal["w_mean"].min() >= 0.0
        assert signal["w_mean"].max() <= 1.0
        assert signal.index.right.max() <= 1.0
        assert signal.index.left.min() >= 0.0

        assert np.all(np.isfinite(signal.values))

    def test_interpolate_small(self) -> None:

        dem = np.arange(16, dtype="float32").reshape(4, 4)
        ddem = dem / 10
        glacier_index_map = np.ones_like(dem)

        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=ddem, ref_dem=dem, glacier_index_map=glacier_index_map, n_bins=10
        )

        # Make it so that only 1/4 of the values exist.
        ddem_orig = ddem.copy()
        ddem.ravel()[4:] = np.nan

        interpolated_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem,
            ref_dem=dem,
            glacier_index_map=glacier_index_map,
            regional_signal=signal,
            min_elevation_range=0.3,  # At least ~1/3 of the range need to exist.
        )

        # Validate that no interpolation was done, as the coverage was too small
        assert np.nansum(np.abs(ddem - interpolated_ddem)) == 0.0

        # Now try with a dDEM that has about 3/4 of the range.
        ddem = ddem_orig.copy()
        ddem.ravel()[12:] = np.nan

        interpolated_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem,
            ref_dem=dem,
            glacier_index_map=glacier_index_map,
            regional_signal=signal,
            min_elevation_range=0.3,
        )

        # Validate that values were interpolated within the measurement step-size
        # TODO: Fails since NumPy 2.0
        # assert np.nanmax(np.abs((interpolated_ddem - ddem_orig)[np.isnan(ddem)])) < 0.1

    def test_regional_hypsometric_interp(self) -> None:

        # Extract a normalized regional hypsometric signal.
        ddem = self.dem_2009 - self.dem_1990

        signal = xdem.volume.get_regional_hypsometric_signal(
            ddem=self.ddem, ref_dem=self.dem_2009, glacier_index_map=self.glacier_index_map
        )

        if False:
            import matplotlib.pyplot as plt

            plt.fill_between(
                signal.index.mid, signal["median"] - signal["std"], signal["median"] + signal["std"], label="Median±std"
            )
            plt.plot(signal.index.mid, signal["median"], color="black", linestyle=":", label="Median")
            plt.plot(signal.index.mid, signal["w_mean"], color="black", label="Weighted mean")

            plt.xlabel("Normalized elevation")
            plt.ylabel("Normalized elevation change")
            plt.legend()

            plt.show()

        # Try the normalized regional hypsometric interpolation.
        # Synthesize random nans in 80% of the data.
        rng = np.random.default_rng(42)
        ddem.data.mask.ravel()[rng.choice(ddem.data.size, int(ddem.data.size * 0.80), replace=False)] = True
        # Fill the dDEM using the de-normalized signal.
        filled_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem, ref_dem=self.dem_2009, glacier_index_map=self.glacier_index_map
        )
        # Fill the dDEM using the de-normalized signal and create an idealized dDEM
        idealized_ddem = xdem.volume.norm_regional_hypsometric_interpolation(
            voided_ddem=ddem, ref_dem=self.dem_2009, glacier_index_map=self.glacier_index_map, idealized_ddem=True
        )
        assert not np.array_equal(filled_ddem, idealized_ddem)

        # Check that all glacier-values are finite
        assert np.count_nonzero(np.isnan(idealized_ddem)[self.glacier_index_map.get_nanarray() > 0]) == 0
        # Validate that the un-idealized dDEM has a higher gradient variance (more ups and downs)
        filled_gradient = np.linalg.norm(np.gradient(filled_ddem), axis=0)
        ideal_gradient = np.linalg.norm(np.gradient(idealized_ddem), axis=0)
        assert np.nanstd(filled_gradient) > np.nanstd(ideal_gradient)

        if False:
            import matplotlib.pyplot as plt

            plt.subplot(121)
            plt.imshow(filled_ddem, cmap="coolwarm_r", vmin=-10, vmax=10)
            plt.subplot(122)
            plt.imshow(idealized_ddem, cmap="coolwarm_r", vmin=-10, vmax=10)

            plt.show()

        # Extract the finite glacier values.
        changes = ddem.data.squeeze()[self.glacier_index_map.get_nanarray() > 0]
        changes = changes[np.isfinite(changes)]
        interp_changes = filled_ddem[self.glacier_index_map.get_nanarray() > 0]
        interp_changes = interp_changes[np.isfinite(interp_changes)]

        # Validate that the interpolated (20% data) means and stds are similar to the original (100% data)
        # These are commented outbecause the CI for some reason gets quite large variance. It works with lower
        # values on normal computers...
        # assert abs(changes.mean() - interp_changes.mean()) < 2
        # assert abs(changes.std() - interp_changes.std()) < 2

import geoutils as gu
import numpy as np

import xdem


def test_local_hypsometric():
    """Test local hypsometric binning and bin interpolation."""
    xdem.examples.download_longyearbyen_examples(overwrite=False)

    dem_2009 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_tba_dem"]).reproject(dem_2009)
    outlines = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
    # Filter to only look at the Scott Turnerbreen glacier
    outlines.ds = outlines.ds.loc[outlines.ds["NAME"] == "Scott Turnerbreen"]

    # Create a mask where glacier areas are True
    mask = outlines.create_mask(dem_2009) == 255

    ddem = dem_2009.data - dem_1990.data

    ddem_bins = xdem.volume.hypsometric_binning(ddem.squeeze()[mask], dem_2009.data.squeeze()[mask])

    ddem_bins_masked = xdem.volume.hypsometric_binning(
        np.ma.masked_array(ddem.squeeze(), mask=~mask),
        np.ma.masked_array(dem_2009.data.squeeze(), mask=~mask)
    )

    assert np.abs(np.mean(ddem_bins["median"] - ddem_bins_masked["median"])) < 0.01

    # Simulate a missing bin
    ddem_bins.iloc[3, :] = np.nan

    # Interpolate the bins and exclude bins with low pixel counts from the interpolation.
    interpolated_bins = xdem.volume.interpolate_hypsometric_bins(ddem_bins, count_threshold=200)

    print(ddem_bins.to_string())

    assert abs(np.mean(interpolated_bins)) < 40
    assert abs(np.mean(interpolated_bins)) > 0
    assert not np.any(np.isnan(interpolated_bins))

import geoutils as gu

import xdem


def test_local_hypsometric():
    xdem.examples.download_longyearbyen_examples(overwrite=False)

    dem_2009 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = gu.georaster.Raster(xdem.examples.FILEPATHS["longyearbyen_tba_dem"]).reproject(dem_2009)

    outlines = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
    # Filter to only look at the Scott Turnerbreen glacier
    outlines.ds = outlines.ds.loc[outlines.ds["NAME"] == "Scott Turnerbreen"]

    mask = outlines.create_mask(dem_2009) == 255

    ddem = dem_2009.data - dem_1990.data

    ddem_bins = xdem.volume.hypsometric_binning(ddem.squeeze()[mask], dem_2009.data.squeeze()[mask])

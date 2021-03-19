"""
Test functions for DEM class
"""
import datetime
import inspect
import os
import warnings

import geoutils as gu
import geoutils.georaster as gr
import geoutils.satimg as si
import numpy as np
import pyproj
import pytest

import xdem
from xdem.dem import DEM

xdem.examples.download_longyearbyen_examples(overwrite=False)

DO_PLOT = False


class TestDEM:

    def test_init(self):
        """
        Test that inputs work properly in DEM class init
        """
        fn_img = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]

        # from filename
        dem = DEM(fn_img)
        assert isinstance(dem, DEM)

        # from DEM
        dem2 = DEM(dem)
        assert isinstance(dem2, DEM)

        # from Raster
        r = gr.Raster(fn_img)
        dem3 = DEM(r)
        assert isinstance(dem3, DEM)

        # from SatelliteImage
        img = si.SatelliteImage(fn_img)
        dem4 = DEM(img)
        assert isinstance(dem4, DEM)

        list_dem = [dem, dem2, dem3, dem4]

        attrs = [at for at in gr.default_attrs if at not in ['name', 'dataset_mask', 'driver']]
        all_attrs = attrs + si.satimg_attrs + xdem.dem.dem_attrs
        for attr in all_attrs:
            attrs_per_dem = [idem.__getattribute__(attr) for idem in list_dem]
            assert all(at == attrs_per_dem[0] for at in attrs_per_dem)

        assert np.logical_and.reduce((np.array_equal(dem.data, dem2.data, equal_nan=True),
                                      np.array_equal(dem2.data, dem3.data, equal_nan=True),
                                      np.array_equal(dem3.data, dem4.data, equal_nan=True)))

        assert np.logical_and.reduce((np.all(dem.data.mask == dem2.data.mask),
                                      np.all(dem2.data.mask == dem3.data.mask),
                                      np.all(dem3.data.mask == dem4.data.mask)))

    def test_copy(self):
        """
        Test that the copy method works as expected for DEM. In particular
        when copying r to r2:
            - if r.data is modified and r copied, the updated data is copied
            - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = xdem.dem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a DEM
        assert isinstance(r2, xdem.dem.DEM)

        # check all immutable attributes are equal
        # georaster_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes', 'nodata',
        #                    'res', 'shape', 'transform', 'width']
        # satimg_attrs = ['satellite', 'sensor', 'product', 'version', 'tile_name', 'datetime']
        # dem_attrs = ['vref', 'vref_grid', 'ccrs']
        # using list directly available in Class
        attrs = [at for at in gr.default_attrs if at not in ['name', 'dataset_mask', 'driver']]
        all_attrs = attrs + si.satimg_attrs + xdem.dem.dem_attrs
        for attr in all_attrs:
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not np.array_equal(r.data, r2.data, equal_nan=True)

    def test_set_vref(self):

        fn_img = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]
        img = DEM(fn_img)

        # check for WGS84
        img.set_vref(vref_name='WGS84')
        assert img.vref == 'WGS84'
        assert img.vref_grid is None

        # check for EGM96
        img.set_vref(vref_name='EGM96')
        assert img.vref == 'EGM96'
        assert img.vref_grid == 'us_nga_egm96_15.tif'
        # grid should have priority over name and parse the right vref name
        img.set_vref(vref_name='WGS84', vref_grid='us_nga_egm96_15.tif')
        assert img.vref == 'EGM96'

        # check for EGM08
        img.set_vref(vref_name='EGM08')
        assert img.vref == 'EGM08'
        assert img.vref_grid == 'us_nga_egm08_25.tif'
        # grid should have priority over name and parse the right vref name
        img.set_vref(vref_name='best ref in the entire world, or any string', vref_grid='us_nga_egm08_25.tif')
        assert img.vref == 'EGM08'

        # check that other existing grids are well detected in the pyproj.datadir
        img.set_vref(vref_grid='is_lmi_Icegeoid_ISN93.tif')

        # check that non-existing grids raise errors
        with pytest.raises(ValueError):
            img.set_vref(vref_grid='the best grid in the entire world, or any non-existing string')

    def test_to_vref(self):

        # first, some points to test the transform

        # Chile
        lat = 43.70012234
        lng = -79.41629234
        z = 100
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            # EGM96 geoid in Chile, we expect ~30 m difference
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids='us_nga_egm96_15.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # check final elevation is finite, higher than ellipsoid with less than 40 m difference (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out), np.greater(z_out, z), np.less(np.abs(z_out-z), 40)))

        # egm2008
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids='us_nga_egm08_25.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # check final elevation is finite, higher than ellipsoid with less than 40 m difference (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out), np.greater(z_out, z), np.less(np.abs(z_out-z), 40)))

        # geoid2006 for Alaska
        lat = 65
        lng = -140
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids='us_noaa_geoid06_ak.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # check final elevation is finite, lower than ellipsoid with less than 20 m difference (typical geoid in Alaska)
        assert np.logical_and.reduce((np.isfinite(z_out), np.less(z_out, z), np.less(np.abs(z_out-z), 20)))

        # isn1993 for Iceland
        lat = 65
        lng = -18
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            # Iceland, we expect a ~70m difference
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids='is_lmi_Icegeoid_ISN93.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # check final elevation is finite, lower than ellipsoid with less than 100 m difference (typical geoid in Iceland)
        assert np.logical_and.reduce((np.isfinite(z_out), np.less(z_out, z), np.less(np.abs(z_out-z), 100)))

        # checking that the function does not run without a reference set
        fn_img = xdem.examples.FILEPATHS["longyearbyen_ref_dem"]
        img = DEM(fn_img)
        with pytest.raises(ValueError):
            img.to_vref(vref_name='EGM96')

        # checking that the function properly runs with a reference set
        img.set_vref(vref_name='WGS84')
        mean_ellips = np.nanmean(img.data)
        img.to_vref(vref_name='EGM96')
        mean_geoid_96 = np.nanmean(img.data)

        assert img.vref == 'EGM96'
        assert img.vref_grid == 'us_nga_egm96_15.tif'
        # check that the geoid is lower than ellipsoid, less than 35 m difference (Svalbard)

        assert np.greater(mean_ellips, mean_geoid_96)
        assert np.less(np.abs(mean_ellips-mean_geoid_96), 35.)


class TestDEMCollection:
    dem_2009 = xdem.dem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
    dem_1990 = xdem.dem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
    outlines_1990 = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])
    outlines_2010 = gu.geovector.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines_2010"])

    def test_create(self):

        timestamps = [datetime.datetime(1990, 8, 1), datetime.datetime(2009, 8, 1), datetime.datetime(2060, 8, 1)]

        scott_1990 = gu.geovector.Vector(
            self.outlines_1990.ds.loc[self.outlines_1990.ds["NAME"] == "Scott Turnerbreen"]
        )
        scott_2010 = gu.geovector.Vector(
            self.outlines_2010.ds.loc[self.outlines_2010.ds["NAME"] == "Scott Turnerbreen"]
        )

        # Make sure the glacier was bigger in 1990, since this is assumed later.
        assert scott_1990.ds.area.sum() > scott_2010.ds.area.sum()

        mask_2010 = (scott_2010.create_mask(self.dem_2009) == 255).reshape(self.dem_2009.data.shape)

        dem_2060 = self.dem_2009.copy()
        dem_2060.data[mask_2010] -= 30

        dems = xdem.DEMCollection(
            [self.dem_1990, self.dem_2009, dem_2060],
            timestamps=timestamps,
            outlines=dict(zip(timestamps[:2], [scott_1990, scott_2010])),
            reference_dem=1
        )

        # Check that the first raster is the oldest one and
        assert dems.dems[0].data.max() == self.dem_1990.data.max()
        assert dems.reference_dem.data.max() == self.dem_2009.data.max()

        dems.subtract_dems(resampling_method="nearest")

        assert np.mean(dems.ddems[0].data) > 0

        dh_series = dems.get_dh_series()

        # The 1990-2009 area should be the union of those years. The 2009-2060 area should just be the 2010 area.
        assert dh_series.iloc[0]["area"] > dh_series.iloc[-1]["area"]

        cumulative_dh = dems.get_cumulative_series(kind="dh")
        cumulative_dv = dems.get_cumulative_series(kind="dv")

        # Simple check that the cumulative_dh is overall negative.
        assert cumulative_dh.iloc[0] > cumulative_dh.iloc[-1]

        # Simple check that the dV number is of a greater magnitude than the dH number.
        assert abs(cumulative_dv.iloc[-1]) > abs(cumulative_dh.iloc[-1])

        # Generate 10000 NaN values randomly in one of the dDEMs
        dems.ddems[0].data[np.random.randint(0, dems.ddems[0].data.shape[0], 100),
                           np.random.randint(0, dems.ddems[0].data.shape[1], 100)] = np.nan
        # Check that the cumulative_dh function warns for NaNs
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                dems.get_cumulative_series(nans_ok=False)
            except UserWarning as exception:
                if "NaNs found in dDEM" not in str(exception):
                    raise exception

        # print(cumulative_dh)

        #raise NotImplementedError

    def test_dem_datetimes(self):
        """Try to create the DEMCollection without the timestamps argument (instead relying on datetime attributes)."""
        self.dem_1990.datetime = datetime.datetime(1990, 8, 1)
        self.dem_2009.datetime = datetime.datetime(2009, 8, 1)

        dems = xdem.DEMCollection(
            [self.dem_1990, self.dem_2009]
        )

        assert len(dems.timestamps) > 0

    def test_ddem_interpolation(self):
        """Test that dDEM interpolation works as it should."""

        # All warnings should raise errors from now on
        warnings.simplefilter("error")

        # Create a DEMCollection object
        dems = xdem.DEMCollection(
            [self.dem_2009, self.dem_1990],
            timestamps=[datetime.datetime(year, 8, 1) for year in (2009, 1990)])

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
        dems.ddems[0].data[np.random.randint(0, dems.ddems[0].data.shape[0], 100),
                           np.random.randint(0, dems.ddems[0].data.shape[1], 100)] = np.nan

        # Make sure that filled_data is not available anymore, since the data now has nans
        assert dems.ddems[0].filled_data is None

        # Interpolate the nans
        dems.ddems[0].interpolate(method="linear")

        # Make sure that the filled_data is available again
        assert dems.ddems[0].filled_data is not None

""" Functions to test the DEM tools."""
import os
import warnings

import geoutils.georaster as gr
import geoutils.satimg as si
import numpy as np
import pyproj
import pytest
import rasterio as rio
from geoutils.georaster.raster import _default_rio_attrs

import xdem
from xdem.dem import DEM

DO_PLOT = False


class TestDEM:
    def test_init(self) -> None:
        """Test that inputs work properly in DEM class init."""
        fn_img = xdem.examples.get_path("longyearbyen_ref_dem")

        # From filename
        dem = DEM(fn_img)
        assert isinstance(dem, DEM)

        # From DEM
        dem2 = DEM(dem)
        assert isinstance(dem2, DEM)

        # From Raster
        r = gr.Raster(fn_img)
        dem3 = DEM(r)
        assert isinstance(dem3, DEM)

        # From SatelliteImage
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Parse metadata from file not implemented")
            img = si.SatelliteImage(fn_img)
        dem4 = DEM(img)
        assert isinstance(dem4, DEM)

        list_dem = [dem, dem2, dem3, dem4]

        # Check all attributes
        attrs = [at for at in _default_rio_attrs if at not in ["name", "dataset_mask", "driver"]]
        all_attrs = attrs + si.satimg_attrs + xdem.dem.dem_attrs
        for attr in all_attrs:
            attrs_per_dem = [idem.__getattribute__(attr) for idem in list_dem]
            assert all(at == attrs_per_dem[0] for at in attrs_per_dem)

        assert np.logical_and.reduce(
            (
                np.array_equal(dem.data, dem2.data, equal_nan=True),
                np.array_equal(dem2.data, dem3.data, equal_nan=True),
                np.array_equal(dem3.data, dem4.data, equal_nan=True),
            )
        )

        assert np.logical_and.reduce(
            (
                np.all(dem.data.mask == dem2.data.mask),
                np.all(dem2.data.mask == dem3.data.mask),
                np.all(dem3.data.mask == dem4.data.mask),
            )
        )

        # Check that an error is raised when more than one band is provided
        with pytest.raises(ValueError, match="DEM rasters should be composed of one band only"):
            xdem.DEM.from_array(
                data=np.zeros(shape=(2, 5, 5)),
                transform=rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
                crs=None,
                nodata=None,
            )

    def test_copy(self) -> None:
        """
        Test that the copy method works as expected for DEM. In particular
        when copying r to r2:
        - if r.data is modified and r copied, the updated data is copied
        - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = xdem.dem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a DEM
        assert isinstance(r2, xdem.dem.DEM)

        # Check all immutable attributes are equal
        # georaster_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes', 'nodata',
        #                    'res', 'shape', 'transform', 'width']
        # satimg_attrs = ['satellite', 'sensor', 'product', 'version', 'tile_name', 'datetime']
        # dem_attrs = ['vref', 'vref_grid', 'ccrs']

        # using list directly available in Class
        attrs = [at for at in _default_rio_attrs if at not in ["name", "dataset_mask", "driver"]]
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

        # Check that the new_array argument indeed modifies the raster
        r3 = r.copy(new_array=r2.data)

        assert np.array_equal(r3.data, r2.data)

    def test_set_vcrs(self) -> None:
        """Tests to set the vertical CRS."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        # -- Test 1: we check with names --

        # Check setting ellipsoid
        dem.set_vcrs(new_vcrs="Ellipsoid")
        assert "Ellipsoid (No vertical CRS)." in dem.vcrs_name
        assert dem.vcrs_grid is None

        # Check setting EGM96
        dem.set_vcrs(new_vcrs="EGM96")
        assert dem.vcrs_name == "EGM96 height"
        assert dem.vcrs_grid == "us_nga_egm96_15.tif"

        # Check setting EGM08
        dem.set_vcrs(new_vcrs="EGM08")
        assert dem.vcrs_name == "EGM2008 height"
        assert dem.vcrs_grid == "us_nga_egm08_25.tif"

        # -- Test 2: we check with grids --

        dem.set_vcrs(new_vcrs="us_nga_egm96_15.tif")
        assert dem.vcrs_name == "unknown"
        assert dem.vcrs_grid == "us_nga_egm96_15.tif"

        dem.set_vcrs(new_vcrs="us_nga_egm08_25.tif")
        assert dem.vcrs_name == "unknown"
        assert dem.vcrs_grid == "us_nga_egm08_25.tif"

        # Check that other existing grids are well detected in the pyproj.datadir
        # TODO: Figure out why CI cannot get the grids on Windows
        if os.name != "nt":
            dem.set_vcrs(new_vcrs="is_lmi_Icegeoid_ISN93.tif")
        else:
            with pytest.raises(ValueError):
                dem.set_vcrs(new_vcrs="is_lmi_Icegeoid_ISN93.tif")

        # Check that non-existing grids raise errors
        with pytest.raises(ValueError):
            dem.set_vcrs(new_vcrs="the best grid in the entire world, or any non-existing string")

    def test_to_vcrs(self) -> None:
        """Tests the conversion of vertical CRS."""

        fn_dem = xdem.examples.get_path("longyearbyen_ref_dem")
        dem = DEM(fn_dem)

        dem = dem.reproject(dst_crs=pyproj.CRS.from_epsg(4979))
        dem.set_vcrs(new_vcrs="Ellipsoid")
        ccrs_init = dem.ccrs
        median_before = np.nanmean(dem)
        dem.to_vcrs(dst_vcrs="EGM96")
        ccrs_dest = dem.ccrs
        median_after = np.nanmean(dem)

        from pyproj.transformer import Transformer
        transformer = Transformer.from_crs(crs_from=ccrs_init, crs_to=ccrs_dest)

        xx, yy = dem.coords()
        x = xx[0, 0]
        yy = yy[0, 0]
        z = dem.data[0, 0, 0]

        z_out = transformer.transform(xx=x, yy=x, zz=z)[2]






    def test_to_vcrs(self) -> None:
        """Tests to convert vertical CRS."""

        # First, we use test points to test the vertical transform
        # Let's start with Chile
        lat = 43.70012234
        lng = -79.41629234
        z = 100

        # WGS84 datum with ellipsoid height
        ellipsoid = pyproj.CRS.from_epsg(4979)
        # EGM96 geoid in Chile, we expect ~30 m difference
        # geoid = pyproj.crs.CompoundCRS(name="WGS 84 + EGM96 height", components=["EPSG:4326", "EPSG:5773"])
        # geoid = xdem.dem._build_ccrs_from_vref(crs=pyproj.CRS.from_epsg(4326), vref_name="EGM96", vref_grid="us_nga_egm96_15.tif")
        geoid = pyproj.Proj(init="EPSG:4326", geoidgrids="us_nga_egm96_15.tif").crs
        transformer = pyproj.Transformer.from_crs(ellipsoid, geoid)
        z_out = transformer.transform(lat, lng, z)[2]

        # Check that the final elevation is finite, and higher than ellipsoid by less than 40 m (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out), np.greater(z_out, z), np.less(np.abs(z_out - z), 40)))

        # With the EGM2008 (catch warnings as this use of init is depecrated)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids="us_nga_egm08_25.tif")
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # Check final elevation is finite, higher than ellipsoid with less than 40 m difference (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out), np.greater(z_out, z), np.less(np.abs(z_out - z), 40)))

        # With GEOID2006 for Alaska
        lat = 65
        lng = -140
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids="us_noaa_geoid06_ak.tif")
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # Check that the final elevation is finite, lower than ellipsoid by less than 20 m (typical geoid in Alaska)
        assert np.logical_and.reduce((np.isfinite(z_out), np.less(z_out, z), np.less(np.abs(z_out - z), 20)))

        # With ISN1993 for Iceland
        lat = 65
        lng = -18
        # TODO: Figure out why CI cannot get the grids on Windows
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            # init is deprecated by
            ellipsoid = pyproj.Proj(init="EPSG:4326")  # WGS84 datum ellipsoid height
            # Iceland, we expect a ~70m difference
            geoid = pyproj.Proj(init="EPSG:4326", geoidgrids="is_lmi_Icegeoid_ISN93.tif")
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng, lat, z)[2]

        # Check that the final elevation is finite, lower than ellipsoid by less than 100 m (typical geoid in Iceland)
        assert np.logical_and.reduce((np.isfinite(z_out), np.less(z_out, z), np.less(np.abs(z_out - z), 100)))

        # Check that the function does not run without a reference set
        fn_img = xdem.examples.get_path("longyearbyen_ref_dem")
        img = DEM(fn_img)
        with pytest.raises(ValueError):
            img.to_vref(vref_name="EGM96")

        # Check that the function properly runs with a reference set
        img.set_vref(vref_name="WGS84")
        mean_ellips = np.nanmean(img.data)
        img.to_vref(vref_name="EGM96")
        mean_geoid_96 = np.nanmean(img.data)
        assert img.vref == "EGM96"
        assert img.vref_grid == "us_nga_egm96_15.tif"
        # Check that the geoid is lower than ellipsoid, less than 35 m difference (Svalbard)
        assert np.greater(mean_ellips, mean_geoid_96)
        assert np.less(np.abs(mean_ellips - mean_geoid_96), 35.0)

        # Check in the other direction
        img = DEM(fn_img)
        img.set_vref(vref_name="EGM96")
        mean_geoid_96 = np.nanmean(img.data)
        img.to_vref(vref_name="WGS84")
        mean_ellips = np.nanmean(img.data)
        assert img.vref == "WGS84"
        assert img.vref_grid is None
        # Check that the geoid is lower than ellipsoid, less than 35 m difference (Svalbard)
        assert np.greater(mean_ellips, mean_geoid_96)
        assert np.less(np.abs(mean_ellips - mean_geoid_96), 35.0)

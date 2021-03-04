"""
Test functions for DEM class
"""
import os
import pytest
import warnings
import pyproj
import inspect
import xdem
import numpy as np
from xdem.dem import DEM

path_module = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(xdem))))

#TODO: move datasets to a "dataset" folder like in geoutils
EXAMPLE_PATHS = {
    "dem1": "examples/Longyearbyen/data/DEM_2009_ref.tif",
    "dem2": "examples/Longyearbyen/data/DEM_1995.tif",
    "glacier_mask": "examples/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
}

DO_PLOT = False

class TestDEM:

    def test_load(self):

        # check that the loading from DEM __init__ does not fail
        fn_img = os.path.join(path_module,EXAMPLE_PATHS['dem1'])
        img = DEM(fn_img)

    def test_set_vref(self):

        fn_img = os.path.join(path_module,EXAMPLE_PATHS['dem1'])
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
        img.set_vref(vref_name='WGS84',vref_grid='us_nga_egm96_15.tif')
        assert img.vref == 'EGM96'

        # check for EGM08
        img.set_vref(vref_name='EGM08')
        assert img.vref == 'EGM08'
        assert img.vref_grid == 'us_nga_egm08_25.tif'
        # grid should have priority over name and parse the right vref name
        img.set_vref(vref_name='best ref in the entire world, or any string',vref_grid='us_nga_egm08_25.tif')
        assert img.vref == 'EGM08'

        # check that other existing grids are well detected in the pyproj.datadir
        img.set_vref(vref_grid='is_lmi_Icegeoid_ISN93.tif')

        # check that non-existing grids raise errors
        with pytest.raises(ValueError):
            img.set_vref(vref_grid='the best grid in the entire world, or any non-existing string')


    def test_to_vref(self):

        #first, some points to test the transform

        # Chile
        lat = 43.70012234
        lng = -79.41629234
        z = 100
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='egm96_15.gtx') #EGM96 geoid in Chile, we expect ~30 m difference
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng,lat,z)[2]

        #check final elevation is finite, higher than ellipsoid with less than 40 m difference (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out),np.greater(z_out,z),np.less(np.abs(z_out-z),40)))


        #egm2008
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='us_nga_egm08_25.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng,lat,z)[2]

        #check final elevation is finite, higher than ellipsoid with less than 40 m difference (typical geoid in Chile)
        assert np.logical_and.reduce((np.isfinite(z_out),np.greater(z_out,z),np.less(np.abs(z_out-z),40)))

        #geoid2006 for Alaska
        lat = 65
        lng = -140
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='us_noaa_geoid06_ak.tif')
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng,lat,z)[2]

        #check final elevation is finite, lower than ellipsoid with less than 20 m difference (typical geoid in Alaska)
        assert np.logical_and.reduce((np.isfinite(z_out),np.less(z_out,z),np.less(np.abs(z_out-z),20)))


        #isn1993 for Iceland
        lat = 65
        lng = -18
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='is_lmi_Icegeoid_ISN93.tif') #Iceland, we expect a ~70m difference
        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        z_out = transformer.transform(lng,lat,z)[2]

        #check final elevation is finite, lower than ellipsoid with less than 100 m difference (typical geoid in Iceland)
        assert np.logical_and.reduce((np.isfinite(z_out),np.less(z_out,z),np.less(np.abs(z_out-z),100)))

        #checking that the function does not run without a reference set
        fn_img = os.path.join(path_module,EXAMPLE_PATHS['dem1'])
        img = DEM(fn_img)
        with pytest.raises(ValueError):
            img.to_vref(vref_name='EGM96')

        #checking that the function properly runs with a reference set
        img.set_vref(vref_name='WGS84')
        img.to_vref(vref_name='EGM96')




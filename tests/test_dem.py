"""
Test functions for DEM class
"""
import os
import pytest
import warnings
import pyproj
from xdem.dem import DEM

#TODO: move datasets to a "dataset" folder like in geoutils
EXAMPLE_PATHS = {
    "dem1": "examples/Longyearbyen/data/DEM_2009_ref.tif",
    "dem2": "examples/Longyearbyen/data/DEM_1995.tif",
    "glacier_mask": "examples/Longyearbyen/data/glacier_mask/CryoClim_GAO_SJ_1990.shp"
}

DO_PLOT = False

class TestDEM:

    def test_load_DEM_subclass(self):

        fn_img = EXAMPLE_PATHS['dem1']

        img = DEM(fn_img)

    def test_vref(self):

        #somewhere in Chile
        lat = 43.70012234
        lng = -79.41629234
        z = 100

        #egm1996

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='egm96_15.gtx') #EGM96 geoid in Chile, we expect ~30 m difference

        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        print(transformer.transform(lng,lat,z))

        #egm2008
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='us_nga_egm08_25.tif')

        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        print(transformer.transform(lng,lat,z))

        #geoid2006 for Alaska
        lat = 65
        lng = -140
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='us_noaa_geoid06_ak.tif')


        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        print(transformer.transform(lng,lat,z))

        #isn1993 for Iceland
        lat = 65
        lng = -18
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="pyproj")
            #init is deprecated by
            ellipsoid=pyproj.Proj(init="EPSG:4326") #WGS84 datum ellipsoid height
            geoid=pyproj.Proj(init="EPSG:4326", geoidgrids='is_lmi_Icegeoid_ISN93.tif') #Iceland, we expect a ~70m difference

        transformer = pyproj.Transformer.from_proj(ellipsoid, geoid)
        print(transformer.transform(lng,lat,z))

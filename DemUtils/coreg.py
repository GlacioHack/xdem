import rasterio as rio
import richdem as rd
import numpy as np

def rio_to_rda(ds:rio.DatasetReader)->rd.rdarray:
    """
    Get georeferenced richDEM array from rasterio dataset
    :param ds: DEM
    :return: DEM
    """

    arr = ds.read(1)
    rda = rd.rdarray(arr, no_data=ds.get_nodatavals()[0])
    rda.geotransform = ds.get_transform()
    rda.projection = ds.get_gcps()

    return rda

def get_terrainattr(ds:rio.DatasetReader,attrib='slope_degrees')->rd.rdarray:
    """
    Derive terrain attribute for DEM opened with rasterio. One of "slope_degrees", "slope_percentage", "aspect",
    "profile_curvature", "planform_curvature", "curvature" and others (see richDEM documentation)
    :param ds: DEM
    :param attrib: terrain attribute
    :return:
    """

    rda = rio_to_rda(ds)
    terrattr = rd.TerrainAttribute(rda, attrib=attrib)

    return terrattr

#putting tests down here
if __name__ == '__main__':

    #TEST FOR RICHDEM SLOPE & OTHER TERRAIN ATTR WITHOUT USING GDAL
    fn_test = '/home/atom/ongoing/glaciohack_testdata/DEM_2001.64734089.tif'

    #1/ this works

    # to check it gives similar result with GDAL opening
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    ds = rio.open(fn_test)
    slp = get_terrainattr(ds,attrib='slope_degrees')
    rd.rdShow(slp,cmap='Spectral',figsize=(10,10))

    #2/ this does not work (need to pass georeferencing to richDEM array, grid is not sufficient)
    rda = rd.LoadGDAL(fn_test)
    slp = rd.TerrainAttribute(rda, attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))

    ds = rio.open(fn_test)
    slp = rd.TerrainAttribute(rd.rdarray(ds.read(1),no_data=ds.get_nodatavals()[0]), attrib='slope_degrees')
    rd.rdShow(slp, cmap='Spectral', figsize=(10, 10))

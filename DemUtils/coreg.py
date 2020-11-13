import rasterio as rio
import richdem as rd
import numpy as np
from rasterio import Affine

def apply_xy_shift(ds: rio.DatasetReader, dx: float, dy: float) -> np.ndarray:
    """
    Apply horizontal shift to rio dataset using Transform affine matrix
    :param ds: DEM
    :param dx: dx shift value
    :param dy: dy shift value
    
    Returns:
    Rio Dataset with updated transform
    """
    print("X shift: ", dx)
    print("Y shift: ", dy)
   
    #Update geotransform
    ds_meta = ds.meta
    gt_orig = ds.transform
    gt_align = Affine(gt_orig.a, gt_orig.b, gt_orig.c+dx, \
                   gt_orig.d, gt_orig.e, gt_orig.f+dy)

    print("Original transform:", gt_orig)
    print("Updated transform:", gt_shift)

    #Update ds Geotransform
    ds_align = ds
    meta_update = ds.meta.copy()
    meta_update({"driver": "GTiff", "height": ds.shape[1],
                 "width": ds.shape[2], "transform": gt_align, "crs": ds.crs})
    #to split this part in two?
    with rasterio.open(ds_align, "w", **meta_update) as dest:
        dest.write(ds_align)
        
    return ds_align

def apply_z_shift(ds: rio.DatasetReader, dz: float):
    """
    Apply vertical shift to rio dataset using Transform affine matrix
    :param ds: DEM
    :param dx: dz shift value
    """
    src_dem = rio.open(ds)
    a = src_dem.read(1)
    ds_shift = a + dz
    return ds_shift

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

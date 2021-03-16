"""
dem.py provides a class for working with digital elevation models (DEMs)
"""
import os
import pyproj
import warnings
from geoutils.georaster import Raster
from geoutils.satimg import SatelliteImage
from pyproj import Transformer
import json
import subprocess

def parse_vref_from_product(product):
    """

    :param product: Product name (typically from satimg.parse_metadata_from_fn)
    :type product: str

    :return: vref_name: Vertical reference name
    :rtype: vref_name: str
    """
    # sources for defining vertical references:
    # AW3D30: https://www.eorc.jaxa.jp/ALOS/en/aw3d30/aw3d30v11_format_e.pdf
    # SRTMGL1: https://lpdaac.usgs.gov/documents/179/SRTM_User_Guide_V3.pdf
    # SRTMv4.1: http://www.cgiar-csi.org/data/srtm-90m-digital-elevation-database-v4-1
    # ASTGTM2/ASTGTM3: https://lpdaac.usgs.gov/documents/434/ASTGTM_User_Guide_V3.pdf
    # NASADEM: https://lpdaac.usgs.gov/documents/592/NASADEM_User_Guide_V1.pdf !! HGTS is ellipsoid, HGT is EGM96 geoid !!
    # ArcticDEM (mosaic and strips): https://www.pgc.umn.edu/data/arcticdem/
    # REMA (mosaic and strips): https://www.pgc.umn.edu/data/rema/
    # TanDEM-X 90m global: https://geoservice.dlr.de/web/dataguide/tdm90/
    # COPERNICUS DEM: https://spacedata.copernicus.eu/web/cscda/dataset-details?articleId=394198

    if product in ['ArcticDEM/REMA','TDM1','NASADEM-HGTS']:
        vref_name = 'WGS84'
    elif product in ['AW3D30','SRTMv4.1','SRTMGL1','ASTGTM2','NASADEM-HGT']:
        vref_name = 'EGM96'
    elif product in ['COPDEM']:
        vref_name = 'EGM08'
    else:
        vref_name = None

    return vref_name


dem_attrs = ['vref','vref_grid','ccrs']

class DEM(SatelliteImage):

    def __init__(self, filename_or_dataset, vref_name=None, vref_grid=None, silent=False, **kwargs):
        """
        Load digital elevation model data through the Raster class, parse additional attributes from filename or metadata
        trougth the SatelliteImage class, and then parse vertical reference from DEM product name.
        For manual input, only one of "vref", "vref_grid" or "ccrs" is necessary to set the vertical reference.

        :param filename_or_dataset: The filename of the dataset.
        :type filename_or_dataset: str, DEM, SatelliteImage, Raster, rio.io.Dataset, rio.io.MemoryFile
        :param vref_name: Vertical reference name
        :type vref_name: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_grid: str
        :param silent: Whether to display vertical reference setting
        :param silent: boolean
        """

        # If DEM is passed, simply point back to DEM
        if isinstance(filename_or_dataset, DEM):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            super().__init__(filename_or_dataset, silent=silent, **kwargs)

        if self.nbands > 1:
            raise ValueError('DEM rasters should be composed of one band only')

        # user input
        self.vref = vref_name
        self.vref_grid = vref_grid
        self.ccrs = None

        # trying to get vref from product name (priority to user input)
        self.__parse_vref_from_fn(silent=silent)

    def copy(self, new_array=None):

        new_dem = super().copy()
        # those attributes are immutable, including pyproj.CRS
        # dem_attrs = ['vref','vref_grid','ccrs'] #taken outside of class
        for attrs in dem_attrs:
            setattr(new_dem, attrs, getattr(self, attrs))

        return new_dem

    def __parse_vref_from_fn(self,silent=False):

        """
        Attempts to pull vertical reference from product name identified by SatImg
        """

        if self.product is not None:
            vref = parse_vref_from_product(self.product)
            if vref is not None and self.vref is None:
                if not silent:
                    print('From product name "'+ str(self.product)+'": setting vertical reference as ' + str(vref))
                self.vref = vref
            elif vref is not None and self.vref is not None:
                if not silent:
                    print('Leaving user input of ' + str(self.vref) + ' for vertical reference despite reading ' + str(
                        vref) + ' from product name')
            else:
                if not silent:
                    print('Could not find a vertical reference based on product name: "'+str(self.product)+'"')


    def set_vref(self,vref_name=None,vref_grid=None,compute_ccrs=False):
        """
        Set vertical reference with a name or with a grid

        :param vref_name: Vertical reference name
        :type vref_name: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_grid: str
        :param compute_ccrs: Whether to compute the ccrs (read pyproj-data grid file)
        :type compute_ccrs: boolean

        :return:
        """

        # temporary fix for some CRS with proj < 7.2
        def get_crs(filepath: str) -> pyproj.CRS:
            """Get the CRS of a raster with the given filepath."""
            info = subprocess.run(
                ["gdalinfo", "-json", filepath],
                stdout=subprocess.PIPE,
                check=True,
                encoding="utf-8"
            ).stdout

            wkt_string = json.loads(info)["coordinateSystem"]["wkt"]

            return pyproj.CRS.from_wkt(wkt_string)

        #for names, we only look for WGS84 ellipsoid or the EGM96/EGM08 geoids: those are used 99% of the time
        if isinstance(vref_grid, str):

            if isinstance(vref_name, str):
                print('Both a vertical reference name and vertical grid are provided: defaulting to using grid only.')

            if vref_grid == 'us_nga_egm08_25.tif':
                self.vref = 'EGM08'
                self.vref_grid = vref_grid
            elif vref_grid == 'us_nga_egm96_15.tif':
                self.vref = 'EGM96'
                self.vref_grid = vref_grid
            else:
                if os.path.exists(os.path.join(pyproj.datadir.get_data_dir(),vref_grid)):
                    self.vref = 'Unknown vertical reference name from: '+vref_grid
                    self.vref_grid = vref_grid
                else:
                    raise ValueError('Grid not found in '+str(pyproj.datadir.get_data_dir())+': check if proj-data is '
                         'installed via conda-forge, the pyproj.datadir, and that you are using a grid available at '
                         'https://github.com/OSGeo/PROJ-data')
        elif isinstance(vref_name,str):
            if vref_name == 'WGS84':
                self.vref_grid = None
                self.vref = 'WGS84'  # WGS84 ellipsoid
            elif vref_name == 'EGM08':
                self.vref_grid = 'us_nga_egm08_25.tif'  # EGM2008 at 2.5 minute resolution
                self.vref = 'EGM08'
            elif vref_name == 'EGM96':
                self.vref_grid = 'us_nga_egm96_15.tif'  # EGM1996 at 15 minute resolution
                self.vref = 'EGM96'
            else:
                raise ValueError(
                    'Vertical reference name must be either "WGS84", "EGM96" or "EGM08". Otherwise, provide'
                    ' a geoid grid from PROJ DATA: https://github.com/OSGeo/PROJ-data')
        else:
            raise ValueError('Vertical reference name or vertical grid must be a string')

        #temporary fix to get all types of CRS
        if pyproj.proj_version_str >= "7.2.0":
            crs = self.crs
        else:
            crs = get_crs(self.filename)

        # no deriving the ccrs until those are used in a reprojection (requires pyproj-data grids = ~500Mo)
        if compute_ccrs:
            if self.vref == 'WGS84':
                # the WGS84 ellipsoid essentially corresponds to no vertical reference in pyproj
                self.ccrs = pyproj.CRS(crs)
            else:
                # for other vrefs, keep same horizontal projection and add geoid grid (the "dirty" way: because init is so
                # practical and still going to be used for a while)
                # see https://gis.stackexchange.com/questions/352277/including-geoidgrids-when-initializing-projection-via-epsg/352300#352300
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", module="pyproj")
                    self.ccrs = pyproj.Proj(init="EPSG:" + str(int(crs.to_epsg())), geoidgrids=self.vref_grid).crs

    def to_vref(self,vref_name='EGM96',vref_grid=None):

        """
        Convert between vertical references: ellipsoidal heights or geoid grids

        :param vref_name: Vertical reference name
        :type vref_grid: str
        :param vref_grid: Vertical reference grid (any grid file in https://github.com/OSGeo/PROJ-data)
        :type vref_name: str

        :return:
        """

        # all transformations grids file are described here: https://github.com/OSGeo/PROJ-data
        if self.vref is None and self.vref_grid is None:
            raise ValueError('The current DEM has not vertical reference: need to set one before attempting a conversion '
                             'towards another vertical reference.')
        elif isinstance(self.vref,str) and self.vref_grid is None:
            # to set the vref grid names automatically EGM96/08 for geoids + compute the ccrs
            self.set_vref(vref_name=self.vref,compute_ccrs=True)

        # inital ccrs
        ccrs_init = self.ccrs

        # destination crs

        # set the new reference (before calculation doesn't change anything, we need to update the data manually anyway)
        self.set_vref(vref_name=vref_name,vref_grid=vref_grid,compute_ccrs=True)
        ccrs_dest = self.ccrs

        # transform matrix
        transformer = Transformer.from_crs(ccrs_init, ccrs_dest)
        meta = self.ds.meta
        zz = self.data
        xx, yy = self.coords(offset='center')
        zz_trans = transformer.transform(xx,yy,zz[0,:])[2]
        zz[0,:] = zz_trans

        # update raster
        self._update(metadata=meta,imgdata=zz)
